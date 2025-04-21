import os
import sys
sys.path.append(os.path.join(os.getcwd(), "sparsh"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, get_linear_schedule_with_warmup
import argparse
import logging
from tqdm import tqdm
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
# import seaborn as sns  # Commented out seaborn dependency
import datetime  # Add datetime import for timestamp

from .data import load_classification_datasets
from sparsh.tactile_ssl.downstream_task.attentive_pooler import AttentivePooler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VitClassification:
    def __init__(self, args):
        """Initialize the ViT model for classification."""
        self.args = args
        
        # Check for GPU availability and configure for multi-GPU if available
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.device = torch.device('cuda')
            logger.info(f"Using {self.num_gpus} GPU{'s' if self.num_gpus > 1 else ''}")
            for i in range(self.num_gpus):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            self.num_gpus = 0
            self.device = torch.device('cpu')
            logger.info("No GPU available, using CPU")

        # Initialize wandb if requested
        if self.args.use_wandb:
            os.makedirs(self.args.output_dir, exist_ok=True)
            wandb.init(
                project=self.args.wandb_project, 
                name=self.args.wandb_run_name,
                dir=self.args.output_dir
            )
            wandb.config.update(vars(self.args))
        
        # Load datasets
        self.train_dataset, self.test_dataset = load_classification_datasets(
            img_size=args.img_size,
            train_data_budget=args.train_data_budget
        )
        
        # Get number of classes from the dataset
        if hasattr(self.train_dataset, 'classes'):
            self.num_classes = len(self.train_dataset.classes)
            self.class_names = self.train_dataset.classes
        else:
            # Handle case when using a subset
            dataset = self.train_dataset.dataset
            self.num_classes = len(dataset.classes)
            self.class_names = dataset.classes
            
        logger.info(f"Loaded {len(self.train_dataset)} training samples and {len(self.test_dataset)} test samples")
        logger.info(f"Number of classes: {self.num_classes}")
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Initialize ViT model
        self._init_model()
        
        # Set up optimizer and loss function
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + 
            list(self.attentive_pooler.parameters()) + 
            list(self.classifier.parameters()), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * args.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        # For tracking best model
        self.best_val_accuracy = 0.0
        self.patience_counter = 0

    def _init_model(self):
        """Initialize the ViT model."""
        logger.info(f"Loading pretrained model: {self.args.model_name}")
        try:
            # Load pretrained ViT model
            self.model = ViTForImageClassification.from_pretrained(self.args.model_name)

            # Freeze all parameters of the base model if requested
            if self.args.freeze_base_model:
                for param in self.model.vit.parameters():
                    param.requires_grad = False
            
            # Get the embedding dimension from the model
            num_features = self.model.classifier.in_features
            
            # Replace the classifier with an AttentivePooler-based classifier
            # First, remove the original classifier to avoid confusion
            self.model.classifier = nn.Identity()
            
            # Add AttentivePooler for better feature extraction
            self.attentive_pooler = AttentivePooler(
                num_queries=1,
                embed_dim=num_features,
            )
            
            # Add final classification layer
            self.classifier = nn.Sequential(
                nn.Linear(num_features, num_features // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(num_features // 2, self.num_classes)
            )
            
            logger.info("Model architecture modified with AttentivePooler for classification")
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.attentive_pooler = self.attentive_pooler.to(self.device)
            self.classifier = self.classifier.to(self.device)
            
            # Wrap with DataParallel if multiple GPUs are available
            if torch.cuda.device_count() > 1:
                logger.info(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
                self.model = nn.DataParallel(self.model)
                self.attentive_pooler = nn.DataParallel(self.attentive_pooler)
                self.classifier = nn.DataParallel(self.classifier)
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        self.attentive_pooler.train()
        self.classifier.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for i, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward
            features = self.model(images).logits  # Get features from ViT
            
            # Reshape features if needed
            if len(features.shape) == 2:
                features = features.unsqueeze(1)
                
            pooled_features = self.attentive_pooler(features).squeeze(1)  # Apply attentive pooling
            logits = self.classifier(pooled_features)  # Final prediction
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            # Backward + optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.attentive_pooler.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            running_loss += loss.item()
            
            # Track predictions for metrics
            _, preds = torch.max(logits, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            if i % self.args.log_interval == 0:
                logger.info(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
                if self.args.use_wandb:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/learning_rate": self.scheduler.get_last_lr()[0]
                    })
        
        epoch_loss = running_loss / len(self.train_loader)
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        logger.info(f"Epoch {epoch} training complete. Average loss: {epoch_loss:.4f}")
        logger.info(f"Training Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if self.args.use_wandb:
            wandb.log({
                "train/epoch_loss": epoch_loss,
                "train/accuracy": accuracy,
                "train/precision": precision,
                "train/recall": recall,
                "train/f1": f1,
                "epoch": epoch
            })
        
        return epoch_loss, accuracy

    def evaluate(self, epoch=None):
        """Evaluate the model on the test set."""
        self.model.eval()
        self.attentive_pooler.eval()
        self.classifier.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Forward
                features = self.model(images).logits  # Get features from ViT
                
                # Reshape features if needed
                if len(features.shape) == 2:
                    features = features.unsqueeze(1)
                    
                pooled_features = self.attentive_pooler(features).squeeze(1)  # Apply attentive pooling
                logits = self.classifier(pooled_features)  # Final prediction
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                val_loss += loss.item()
                
                # Track predictions
                _, preds = torch.max(logits, 1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Calculate per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=list(range(self.num_classes)), zero_division=0
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(self.num_classes)))
        
        avg_val_loss = val_loss / len(self.test_loader)
        
        logger.info(f"Validation loss: {avg_val_loss:.4f}")
        logger.info(f"Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if self.args.use_wandb and epoch is not None:
            # Log overall metrics
            wandb.log({
                "val_main/loss": avg_val_loss,
                "val_main/accuracy": accuracy,
                "val_main/precision": precision,
                "val_main/recall": recall,
                "val_main/f1": f1,
                "epoch": epoch
            })
            
            # Log per-class metrics
            for i, class_name in enumerate(self.class_names):
                wandb.log({
                    f"val/class/precision_{class_name}": class_precision[i],
                    f"val/class/recall_{class_name}": class_recall[i],
                    f"val/class/f1_{class_name}": class_f1[i],
                    "epoch": epoch
                })
            
            # Commented out confusion matrix logging
            # if epoch % 5 == 0:  # Log less frequently to reduce overhead
            #     self._log_confusion_matrix(all_labels, all_preds, epoch)
        
        return avg_val_loss, accuracy

    # Commented out confusion matrix logging method
    # def _log_confusion_matrix(self, true_labels, pred_labels, epoch):
    #     """Create and log a confusion matrix."""
    #     if not self.args.use_wandb:
    #         return
    #         
    #     cm = confusion_matrix(true_labels, pred_labels, labels=list(range(self.num_classes)))
    #     
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #                xticklabels=self.class_names,
    #                yticklabels=self.class_names)
    #     plt.xlabel('Predicted Labels')
    #     plt.ylabel('True Labels')
    #     plt.title(f'Confusion Matrix - Epoch {epoch}')
    #     plt.tight_layout()
    #     
    #     wandb.log({f"confusion_matrix_epoch_{epoch}": wandb.Image(plt)})
    #     plt.close()

    def save_model(self, path):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'attentive_pooler_state_dict': self.attentive_pooler.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        """Load a saved model."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.attentive_pooler.load_state_dict(checkpoint['attentive_pooler_state_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'best_val_accuracy' in checkpoint:
                self.best_val_accuracy = checkpoint['best_val_accuracy']
            logger.info(f"Model loaded from {path}")
            return True
        else:
            logger.warning(f"No model found at {path}")
            return False

    def train(self):
        """Run the full training process."""
        logger.info("Starting training process")
        
        # Resume from checkpoint if specified
        if self.args.resume_from:
            self.load_model(self.args.resume_from)

        for epoch in range(1, self.args.num_epochs + 1):
            train_loss, train_accuracy = self.train_epoch(epoch)
            val_loss, val_accuracy = self.evaluate(epoch)
            
            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.save_model(os.path.join(self.args.output_dir, "best_model.pt"))
                self.patience_counter = 0
                logger.info(f"New best model saved with validation accuracy: {val_accuracy:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement for {self.patience_counter} epochs")
            
            # Early stopping
            if self.args.patience > 0 and self.patience_counter >= self.args.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Save checkpoint
            if epoch % self.args.checkpoint_interval == 0:
                self.save_model(os.path.join(self.args.output_dir, f"checkpoint_epoch_{epoch}.pt"))
        
        logger.info("Training complete!")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        if self.args.use_wandb:
            wandb.finish()

def parse_args():
    """Parse command-line arguments."""
    # Create a timestamp for the default output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_dir = f"experiments/{timestamp}_vit_port_material_classification"
    
    parser = argparse.ArgumentParser(description="Train a Vision Transformer for port and material classification")
    
    # Model parameters
    parser.add_argument("--model-name", type=str, default="google/vit-base-patch16-224",
                       help="Pretrained model name or path")
    parser.add_argument("--img-size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--freeze-base-model", action="store_true", default=True,
                       help="Freeze base model weights")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=300,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.001,
                       help="Weight decay")
    parser.add_argument("--train-data-budget", type=float, default=1.0,
                       help="Fraction of training data to use (0-1)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loader workers")
    
    # Training control
    parser.add_argument("--log-interval", type=int, default=10,
                       help="Log interval for training batches")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                       help="Epoch interval for saving checkpoints")
    parser.add_argument("--patience", type=int, default=0,
                       help="Early stopping patience (0 to disable)")
    parser.add_argument("--resume-from", type=str, default="",
                       help="Resume training from checkpoint")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                       help="Directory to save model checkpoints")
    parser.add_argument("--use-wandb", action="store_true", default=True,
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="port-material-classification-vit",
                       help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default="vit-classification",
                       help="W&B run name")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run the training
    vit_clf = VitClassification(args)
    vit_clf.train()

if __name__ == "__main__":
    main()
