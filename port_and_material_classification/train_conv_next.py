import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import ConvNextForImageClassification, get_linear_schedule_with_warmup
import argparse
import logging
from tqdm import tqdm
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns

from port_and_material_classification.data import load_classification_datasets

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConvNextClassification:
    def __init__(self, args):
        """Initialize the ConvNext model for classification."""
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

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
        
        # Initialize ConvNext model
        self._init_model()
        
        # Set up optimizer and loss function
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
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
        """Initialize the ConvNext model."""
        logger.info(f"Loading pretrained model: {self.args.model_name}")
        try:
            # Load pretrained ConvNext model
            self.model = ConvNextForImageClassification.from_pretrained(self.args.model_name)

            # Freeze base model if requested
            if self.args.freeze_base_model:
                for name, param in self.model.named_parameters():
                    if 'classifier' not in name:
                        param.requires_grad = False
            
            # Modify the classifier head for our task
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, self.num_classes)
            )
            
            logger.info("ConvNext model architecture modified for classification")
            
            # Move model to device
            self.model = self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for i, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(images)
            logits = outputs.logits
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            # Backward + optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
                        "train_batch_loss": loss.item(), 
                        "learning_rate": self.scheduler.get_last_lr()[0]
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
                "train_epoch_loss": epoch_loss,
                "train_accuracy": accuracy,
                "train_precision": precision,
                "train_recall": recall,
                "train_f1": f1,
                "epoch": epoch
            })
        
        return epoch_loss, accuracy

    def evaluate(self, epoch=None):
        """Evaluate the model on the test set."""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Forward
                outputs = self.model(images)
                logits = outputs.logits
                
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
        
        avg_val_loss = val_loss / len(self.test_loader)
        
        logger.info(f"Validation loss: {avg_val_loss:.4f}")
        logger.info(f"Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if self.args.use_wandb and epoch is not None:
            # Log overall metrics
            wandb.log({
                "val_loss": avg_val_loss,
                "val_accuracy": accuracy,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
                "epoch": epoch
            })
            
            # Log per-class metrics
            for i, class_name in enumerate(self.class_names):
                wandb.log({
                    f"val_precision_{class_name}": class_precision[i],
                    f"val_recall_{class_name}": class_recall[i],
                    f"val_f1_{class_name}": class_f1[i],
                    "epoch": epoch
                })
            
            # Log confusion matrix
            if epoch % 5 == 0:  # Log less frequently to reduce overhead
                self._log_confusion_matrix(all_labels, all_preds, epoch)
        
        return avg_val_loss, accuracy

    def _log_confusion_matrix(self, true_labels, pred_labels, epoch):
        """Create and log a confusion matrix."""
        if not self.args.use_wandb:
            return
            
        cm = confusion_matrix(true_labels, pred_labels, labels=list(range(self.num_classes)))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.tight_layout()
        
        wandb.log({f"confusion_matrix_epoch_{epoch}": wandb.Image(plt)})
        plt.close()

    def save_model(self, path):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
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
    parser = argparse.ArgumentParser(description="Train a ConvNext model for port and material classification")
    
    # Model parameters
    parser.add_argument("--model-name", type=str, default="facebook/convnext-tiny-224",
                       help="Pretrained model name or path")
    parser.add_argument("--img-size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--freeze-base-model", action="store_true",
                       help="Freeze base model weights")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
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
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience (0 to disable)")
    parser.add_argument("--resume-from", type=str, default="",
                       help="Resume training from checkpoint")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="outputs/port_material_classification/convnext",
                       help="Output directory for models and logs")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="port-material-classification",
                       help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default="convnext-classification",
                       help="W&B run name")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run the training
    convnext_clf = ConvNextClassification(args)
    convnext_clf.train()

if __name__ == "__main__":
    main()
