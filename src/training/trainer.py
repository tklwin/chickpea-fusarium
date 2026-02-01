"""
Training loop with W&B integration.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..data import get_dataloaders
from ..data.split import compute_class_weights, load_splits
from ..models import get_model
from .metrics import compute_metrics, get_classification_report, plot_confusion_matrix


class Trainer:
    """
    Training manager with W&B logging.
    
    Args:
        config: Configuration object (from configs/default.py)
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set seed for reproducibility
        self._set_seed(config.training.seed)
        
        # Initialize components
        self.model = self._build_model()
        self.dataloaders = self._build_dataloaders()
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        
        # History
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        # Initialize W&B
        self._init_wandb()
        
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        # For CUDA determinism (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _build_model(self) -> nn.Module:
        """Build and configure the model."""
        model = get_model(
            model_name=self.config.model.model_name,
            num_classes=self.config.data.num_classes,
            pretrained=self.config.model.pretrained,
            dropout=self.config.model.dropout,
            cbam_reduction=self.config.model.cbam_reduction,
            cbam_kernel_size=self.config.model.cbam_kernel_size,
        )
        
        model = model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel: {self.config.model.model_name}")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        return model
    
    def _build_dataloaders(self) -> Dict[str, DataLoader]:
        """Build data loaders."""
        # Get augmentation config as dict
        aug_config = {
            'horizontal_flip_p': self.config.augmentation.horizontal_flip_p,
            'vertical_flip_p': self.config.augmentation.vertical_flip_p,
            'rotate_limit': self.config.augmentation.rotate_limit,
            'rotate_p': self.config.augmentation.rotate_p,
            'brightness_limit': self.config.augmentation.brightness_limit,
            'contrast_limit': self.config.augmentation.contrast_limit,
            'color_jitter_p': self.config.augmentation.color_jitter_p,
            'random_resized_crop': self.config.augmentation.random_resized_crop,
            'crop_scale': self.config.augmentation.crop_scale,
            'coarse_dropout_p': self.config.augmentation.coarse_dropout_p,
            'max_holes': self.config.augmentation.max_holes,
            'max_height': self.config.augmentation.max_height,
            'max_width': self.config.augmentation.max_width,
        }
        
        return get_dataloaders(
            splits_dir=self.config.data.splits_dir,
            batch_size=self.config.data.batch_size,
            image_size=self.config.data.image_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            use_weighted_sampler=self.config.training.use_weighted_sampler,
            augmentation_config=aug_config if self.config.augmentation.enable_augmentation else None,
        )
    
    def _build_criterion(self) -> nn.Module:
        """Build loss function with optional class weights."""
        if self.config.training.use_class_weights:
            # Load training split to compute weights
            splits = load_splits(self.config.data.splits_dir)
            class_weights = compute_class_weights(
                splits['train'],
                num_classes=self.config.data.num_classes
            )
            weight = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight)
            print(f"\nUsing weighted CrossEntropyLoss")
        else:
            criterion = nn.CrossEntropyLoss()
            
        return criterion
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer."""
        if self.config.training.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
            
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        if self.config.training.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs - self.config.training.warmup_epochs,
                eta_min=self.config.training.min_lr
            )
        elif self.config.training.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        elif self.config.training.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=self.config.training.min_lr
            )
        else:
            scheduler = None
            
        return scheduler
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        if not self.config.wandb.enabled or not WANDB_AVAILABLE:
            self.use_wandb = False
            return
            
        self.use_wandb = True
        
        # Generate run name if not provided
        run_name = self.config.wandb.run_name
        if run_name is None:
            run_name = f"{self.config.model.model_name}_bs{self.config.data.batch_size}_lr{self.config.training.learning_rate}"
        
        wandb.init(
            entity=self.config.wandb.entity,
            project=self.config.wandb.project,
            name=run_name,
            tags=self.config.wandb.tags,
            notes=self.config.wandb.notes,
            config={
                # Flatten config for W&B
                "model": self.config.model.model_name,
                "pretrained": self.config.model.pretrained,
                "num_classes": self.config.data.num_classes,
                "batch_size": self.config.data.batch_size,
                "image_size": self.config.data.image_size,
                "epochs": self.config.training.epochs,
                "learning_rate": self.config.training.learning_rate,
                "optimizer": self.config.training.optimizer,
                "scheduler": self.config.training.scheduler,
                "weight_decay": self.config.training.weight_decay,
                "dropout": self.config.model.dropout,
                "use_class_weights": self.config.training.use_class_weights,
                "use_weighted_sampler": self.config.training.use_weighted_sampler,
                "augmentation_enabled": self.config.augmentation.enable_augmentation,
                "seed": self.config.training.seed,
            }
        )
        
        # Watch model gradients
        wandb.watch(self.model, log='gradients', log_freq=100)
        
    def train_one_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(
            self.dataloaders['train'],
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.epochs} [Train]"
        )
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader = None) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: DataLoader to use (default: val)
        
        Returns:
            Dict with loss, accuracy, and other metrics
        """
        if dataloader is None:
            dataloader = self.dataloaders['val']
            
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}/{self.config.training.epochs} [Val]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = compute_metrics(all_labels, all_preds)
        metrics['loss'] = running_loss / len(dataloader.dataset)
        
        return metrics
    
    def fit(self):
        """Run full training loop."""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss, train_acc = self.train_one_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if self.config.training.scheduler == "plateau":
                    self.scheduler.step(val_metrics['accuracy'])
                elif epoch >= self.config.training.warmup_epochs:
                    self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            # Log to W&B
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/precision': val_metrics['precision'],
                    'val/recall': val_metrics['recall'],
                    'val/f1': val_metrics['f1'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                }
                
                # Per-class metrics
                for i in range(self.config.data.num_classes):
                    log_dict[f'val/f1_class_{i}'] = val_metrics.get(f'f1_class_{i}', 0)
                
                wandb.log(log_dict)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.training.epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val F1: {val_metrics['f1']*100:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_f1 = val_metrics['f1']
                self.patience_counter = 0
                
                if self.config.training.save_best_only:
                    self._save_checkpoint('best.pth')
                    print(f"  ✓ New best model saved! (Acc: {self.best_val_acc*100:.2f}%)")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.config.training.early_stopping and self.patience_counter >= self.config.training.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Total time: {total_time / 60:.2f} minutes")
        print(f"Best Val Accuracy: {self.best_val_acc * 100:.2f}%")
        print(f"Best Val F1: {self.best_val_f1 * 100:.2f}%")
        print("=" * 60)
        
        # Final evaluation on test set
        self._final_evaluation()
        
        # Close W&B
        if self.use_wandb:
            wandb.finish()
    
    def _final_evaluation(self):
        """Evaluate on test set and generate reports."""
        print("\n--- Final Evaluation on Test Set ---")
        
        # Load best model
        self._load_checkpoint('best.pth')
        
        # Evaluate
        test_metrics = self.validate(self.dataloaders['test'])
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
        print(f"  Precision: {test_metrics['precision'] * 100:.2f}%")
        print(f"  Recall: {test_metrics['recall'] * 100:.2f}%")
        print(f"  F1 Score: {test_metrics['f1'] * 100:.2f}%")
        
        # Per-class metrics
        print(f"\nPer-class F1:")
        for i, name in enumerate(self.config.data.class_names):
            print(f"  {name}: {test_metrics.get(f'f1_class_{i}', 0) * 100:.2f}%")
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({
                'test/accuracy': test_metrics['accuracy'],
                'test/precision': test_metrics['precision'],
                'test/recall': test_metrics['recall'],
                'test/f1': test_metrics['f1'],
            })
            
            # Log summary metrics
            wandb.run.summary['best_val_accuracy'] = self.best_val_acc
            wandb.run.summary['test_accuracy'] = test_metrics['accuracy']
            wandb.run.summary['test_f1'] = test_metrics['f1']
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
    
    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = Path(self.config.training.checkpoint_dir) / filename
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
