import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import ConvNextForImageClassification, ConvNextConfig, get_linear_schedule_with_warmup
import argparse
import os
import logging
from tqdm import tqdm
import numpy as np
import wandb
import matplotlib.pyplot as plt
import datetime
from force_estimation.data import load_forceslip_datasets

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConvNextForceEstimation:
    def __init__(self, args):
        """Initialize the ConvNext model for force estimation."""
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load datasets
        self.train_dataset, self.test_dataset = load_forceslip_datasets()
        logger.info(f"Loaded {len(self.train_dataset)} training samples and {len(self.test_dataset)} test samples")
        
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
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # Using Smooth L1 Loss for training
        self.criterion = nn.SmoothL1Loss(beta=0.2)
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * args.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        # For tracking best model
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _init_model(self):
        """Initialize the ConvNext model."""
        logger.info(f"Loading pretrained model: {self.args.model_name}")
        try:
            # Load pretrained ConvNext model
            self.model = ConvNextForImageClassification.from_pretrained(self.args.model_name)

            # Freeze all parameters of the base model
            for name, param in self.model.named_parameters():
                if 'classifier' not in name:  # Freeze everything except the classifier
                    param.requires_grad = False
            
            # Modify the classifier head to output 3 values for force estimation
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 3)  # 3 for force XYZ
            )
            logger.info("ConvNext model architecture modified for force estimation")
            
            # Move model to device
            self.model = self.model.to(self.device)
            if torch.cuda.device_count() > 1:
                logger.info(f"Using {torch.cuda.device_count()} GPUs!")
                self.model = nn.DataParallel(self.model)
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_pred_forces = []
        all_true_forces = []
        
        for i, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            images = batch["image"].to(self.device)
            forces = batch["force"].to(self.device)  # Using absolute forces for training
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(images)
            logits = outputs.logits
            
            # Calculate loss
            loss = self.criterion(logits, forces)
            
            # Backward + optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            running_loss += loss.item()
            
            # Track predictions for metrics
            all_pred_forces.append(logits.detach().cpu().numpy())
            all_true_forces.append(forces.detach().cpu().numpy())
            
            if i % self.args.log_interval == 0:
                logger.info(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
                if self.args.use_wandb:
                    wandb.log({"train_batch_loss": loss.item(), 
                              "learning_rate": self.scheduler.get_last_lr()[0]})
        
        epoch_loss = running_loss / len(self.train_loader)
        
        # Calculate MSE for each axis
        all_pred_forces = np.concatenate(all_pred_forces, axis=0)
        all_true_forces = np.concatenate(all_true_forces, axis=0)
        axis_mse = np.mean((all_pred_forces - all_true_forces) ** 2, axis=0)
        
        logger.info(f"Epoch {epoch} training complete. Average loss: {epoch_loss:.4f}")
        logger.info(f"MSE by axis - X: {axis_mse[0]:.4f}, Y: {axis_mse[1]:.4f}, Z: {axis_mse[2]:.4f}")
        
        if self.args.use_wandb:
            wandb.log({
                "train_epoch_loss": epoch_loss,
                "train_mse_x": axis_mse[0],
                "train_mse_y": axis_mse[1],
                "train_mse_z": axis_mse[2],
                "epoch": epoch
            })
        
        return epoch_loss
    
    def evaluate(self, epoch=None):
        """Evaluate the model on the test set."""
        self.model.eval()
        val_loss = 0.0
        all_pred_forces = []
        all_true_forces = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                images = batch["image"].to(self.device)
                force_scale = batch["force_scale"].to(self.device)
                forces = batch["force"].to(self.device) * force_scale
                
                outputs = self.model(images)
                logits = outputs.logits * force_scale
                
                # Use MSE for validation loss calculation
                loss = nn.MSELoss()(logits, forces)
                val_loss += loss.item()
                
                all_pred_forces.append(logits.cpu().numpy())
                all_true_forces.append(forces.cpu().numpy())
        
        all_pred_forces = np.concatenate(all_pred_forces, axis=0)
        all_true_forces = np.concatenate(all_true_forces, axis=0)
        
        # Calculate overall MSE and per-axis MSE
        avg_val_loss = val_loss / len(self.test_loader)
        axis_mse = np.mean((all_pred_forces - all_true_forces) ** 2, axis=0)
        
        # Calculate RMSE for evaluation
        avg_val_rmse = np.sqrt(avg_val_loss)
        axis_rmse = np.sqrt(axis_mse)
        
        logger.info(f"Validation loss (MSE): {avg_val_loss:.4f}")
        logger.info(f"Validation RMSE: {avg_val_rmse:.4f}")
        logger.info(f"RMSE by axis - X: {axis_rmse[0]:.4f}, Y: {axis_rmse[1]:.4f}, Z: {axis_rmse[2]:.4f}")
        
        if self.args.use_wandb and epoch is not None:
            wandb.log({
                "val_mse": avg_val_loss,
                "val_rmse": avg_val_rmse,
                "val_rmse_x": axis_rmse[0],
                "val_rmse_y": axis_rmse[1],
                "val_rmse_z": axis_rmse[2],
                "val_mse_x": axis_mse[0],
                "val_mse_y": axis_mse[1],
                "val_mse_z": axis_mse[2],
                "epoch": epoch
            })
            
            # Create and log prediction scatter plots
            if epoch % 5 == 0:  # Log plots every 5 epochs to reduce overhead
                self._log_prediction_plots(all_true_forces, all_pred_forces, epoch)
        
        return avg_val_loss
    
    def _log_prediction_plots(self, true_forces, pred_forces, epoch):
        """Create and log scatter plots of true vs predicted forces."""
        if not self.args.use_wandb:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axis_labels = ["X", "Y", "Z"]
        
        for i in range(3):
            axes[i].scatter(true_forces[:, i], pred_forces[:, i], alpha=0.5)
            axes[i].set_xlabel(f"True Force {axis_labels[i]}")
            axes[i].set_ylabel(f"Predicted Force {axis_labels[i]}")
            axes[i].set_title(f"Force {axis_labels[i]}")
            
            # Add y=x line
            min_val = min(true_forces[:, i].min(), pred_forces[:, i].min())
            max_val = max(true_forces[:, i].max(), pred_forces[:, i].max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r-')
            
        plt.tight_layout()
        wandb.log({f"force_predictions_epoch_{epoch}": wandb.Image(fig)})
        plt.close(fig)
    
    def save_model(self, path):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
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
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
            logger.info(f"Model loaded from {path}")
            return True
        else:
            logger.warning(f"No model found at {path}")
            return False
    
    def finetune(self):
        """Run the full finetuning process."""
        logger.info("Starting finetuning process")
        
        # Initialize wandb if requested
        if self.args.use_wandb:
            # Create the output directory if it doesn't exist yet
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            # Initialize wandb with the output directory as the logging directory
            wandb.init(
                project=self.args.wandb_project, 
                name=self.args.wandb_run_name,
                dir=self.args.output_dir  # Set wandb logs to be saved in the output directory
            )
            wandb.config.update(vars(self.args))
        
        # Resume from checkpoint if specified
        if self.args.resume_from:
            self.load_model(self.args.resume_from)
        
        for epoch in range(1, self.args.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate(epoch)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(os.path.join(self.args.output_dir, "best_model.pt"))
                self.patience_counter = 0
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
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
        
        logger.info("Finetuning complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        if self.args.use_wandb:
            wandb.finish()


def parse_args():
    """Parse command-line arguments."""
    # Create a timestamp for the default output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_dir = f"experiments/{timestamp}_convnext_force_estimation"
    
    parser = argparse.ArgumentParser(description="Finetune ConvNext for force estimation")
    
    parser.add_argument("--model_name", type=str, default="facebook/convnext-base-224-22k",
                        help="Name of the pretrained ConvNext model to use")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                        help="Weight decay for the optimizer")
    parser.add_argument("--output_dir", type=str, default=default_output_dir,
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="How often to log training progress")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                        help="How often to save model checkpoints (in epochs)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for the data loaders")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--resume_from", type=str, default="",
                        help="Path to a checkpoint to resume training from")
    parser.add_argument("--use_wandb", action="store_true", default=True,
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="tactile_force_estimation_convnext",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="convnext-force-estimation",
                        help="W&B run name")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run the finetuning
    convnext_force = ConvNextForceEstimation(args)
    convnext_force.finetune()


if __name__ == "__main__":
    main()
