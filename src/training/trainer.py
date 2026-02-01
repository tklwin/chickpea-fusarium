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
    def validate(self, dataloader: DataLoader = None, compute_auc: bool = True) -> Dict[str, float]:
        """
        Validate the model with comprehensive research-grade metrics.
        
        Args:
            dataloader: DataLoader to use (default: val)
            compute_auc: Whether to compute AUC-ROC (requires storing probabilities)
        
        Returns:
            Dict with loss, accuracy, and all research-grade metrics
        """
        if dataloader is None:
            dataloader = self.dataloaders['val']
            
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_probs = []  # For AUC-ROC computation
        all_labels = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}/{self.config.training.epochs} [Val]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if compute_auc:
                all_probs.extend(probs.cpu().numpy())
        
        # Compute comprehensive metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs) if compute_auc else None
        
        metrics = compute_metrics(
            all_labels, 
            all_preds, 
            y_prob=all_probs,
            num_classes=self.config.data.num_classes
        )
        metrics['loss'] = running_loss / len(dataloader.dataset)
        
        # Store predictions for confusion matrix logging
        self._last_val_preds = all_preds
        self._last_val_labels = all_labels
        self._last_val_probs = all_probs
        
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
            
            # Log to W&B (Research-Grade Metrics)
            if self.use_wandb:
                log_dict = {
                    # ===== Training Metrics =====
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    
                    # ===== Core Validation Metrics =====
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/balanced_accuracy': val_metrics.get('balanced_accuracy', 0),
                    
                    # ===== Weighted Averages (accounts for class imbalance) =====
                    'val/precision_weighted': val_metrics.get('precision_weighted', 0),
                    'val/recall_weighted': val_metrics.get('recall_weighted', 0),
                    'val/f1_weighted': val_metrics.get('f1_weighted', 0),
                    
                    # ===== Macro Averages (treats all classes equally) =====
                    'val/precision_macro': val_metrics.get('precision_macro', 0),
                    'val/recall_macro': val_metrics.get('recall_macro', 0),
                    'val/f1_macro': val_metrics.get('f1_macro', 0),
                    
                    # ===== Advanced Metrics (Research-Grade) =====
                    'val/cohen_kappa': val_metrics.get('cohen_kappa', 0),
                    'val/mcc': val_metrics.get('mcc', 0),  # Matthews Correlation Coefficient
                    
                    # ===== Learning Rate =====
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                }
                
                # ===== AUC-ROC Metrics (if available) =====
                if 'auc_ovr_macro' in val_metrics:
                    log_dict['val/auc_macro'] = val_metrics['auc_ovr_macro']
                    log_dict['val/auc_weighted'] = val_metrics.get('auc_ovr_weighted', 0)
                
                # ===== Per-Class Metrics =====
                class_names = self.config.data.class_names
                for i, name in enumerate(class_names):
                    log_dict[f'val/precision_{name}'] = val_metrics.get(f'precision_{name}', 0)
                    log_dict[f'val/recall_{name}'] = val_metrics.get(f'recall_{name}', 0)
                    log_dict[f'val/f1_{name}'] = val_metrics.get(f'f1_{name}', 0)
                    log_dict[f'val/accuracy_{name}'] = val_metrics.get(f'accuracy_{name}', 0)
                    if f'auc_{name}' in val_metrics:
                        log_dict[f'val/auc_{name}'] = val_metrics[f'auc_{name}']
                
                # ===== Log Confusion Matrix every 10 epochs =====
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    self._log_confusion_matrix_to_wandb(epoch + 1)
                
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
        
        # ===== Print Comprehensive Test Results =====
        print(f"\n{'='*60}")
        print("TEST SET RESULTS (Research-Grade Metrics)")
        print(f"{'='*60}")
        
        print(f"\n[Overall Metrics]")
        print(f"  Accuracy:          {test_metrics['accuracy'] * 100:.2f}%")
        print(f"  Balanced Accuracy: {test_metrics.get('balanced_accuracy', 0) * 100:.2f}%")
        
        print(f"\n[Weighted Averages]")
        print(f"  Precision: {test_metrics.get('precision_weighted', 0) * 100:.2f}%")
        print(f"  Recall:    {test_metrics.get('recall_weighted', 0) * 100:.2f}%")
        print(f"  F1 Score:  {test_metrics.get('f1_weighted', 0) * 100:.2f}%")
        
        print(f"\n[Macro Averages]")
        print(f"  Precision: {test_metrics.get('precision_macro', 0) * 100:.2f}%")
        print(f"  Recall:    {test_metrics.get('recall_macro', 0) * 100:.2f}%")
        print(f"  F1 Score:  {test_metrics.get('f1_macro', 0) * 100:.2f}%")
        
        print(f"\n[Advanced Metrics]")
        print(f"  Cohen's Kappa: {test_metrics.get('cohen_kappa', 0):.4f}")
        print(f"  MCC:           {test_metrics.get('mcc', 0):.4f}")
        if 'auc_ovr_macro' in test_metrics:
            print(f"  AUC-ROC Macro: {test_metrics.get('auc_ovr_macro', 0):.4f}")
        
        # Per-class metrics
        print(f"\n[Per-Class Metrics]")
        print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
        print(f"  {'-'*55}")
        for name in self.config.data.class_names:
            p = test_metrics.get(f'precision_{name}', 0) * 100
            r = test_metrics.get(f'recall_{name}', 0) * 100
            f1 = test_metrics.get(f'f1_{name}', 0) * 100
            acc = test_metrics.get(f'accuracy_{name}', 0) * 100
            print(f"  {name:<15} {p:>9.2f}% {r:>9.2f}% {f1:>9.2f}% {acc:>9.2f}%")
        
        print(f"{'='*60}")
        
        # ===== Log to W&B (Comprehensive) =====
        if self.use_wandb:
            test_log = {
                # Core metrics
                'test/accuracy': test_metrics['accuracy'],
                'test/balanced_accuracy': test_metrics.get('balanced_accuracy', 0),
                
                # Weighted averages
                'test/precision_weighted': test_metrics.get('precision_weighted', 0),
                'test/recall_weighted': test_metrics.get('recall_weighted', 0),
                'test/f1_weighted': test_metrics.get('f1_weighted', 0),
                
                # Macro averages
                'test/precision_macro': test_metrics.get('precision_macro', 0),
                'test/recall_macro': test_metrics.get('recall_macro', 0),
                'test/f1_macro': test_metrics.get('f1_macro', 0),
                
                # Advanced
                'test/cohen_kappa': test_metrics.get('cohen_kappa', 0),
                'test/mcc': test_metrics.get('mcc', 0),
            }
            
            # AUC if available
            if 'auc_ovr_macro' in test_metrics:
                test_log['test/auc_macro'] = test_metrics['auc_ovr_macro']
                test_log['test/auc_weighted'] = test_metrics.get('auc_ovr_weighted', 0)
            
            # Per-class
            for name in self.config.data.class_names:
                test_log[f'test/precision_{name}'] = test_metrics.get(f'precision_{name}', 0)
                test_log[f'test/recall_{name}'] = test_metrics.get(f'recall_{name}', 0)
                test_log[f'test/f1_{name}'] = test_metrics.get(f'f1_{name}', 0)
                test_log[f'test/accuracy_{name}'] = test_metrics.get(f'accuracy_{name}', 0)
                if f'auc_{name}' in test_metrics:
                    test_log[f'test/auc_{name}'] = test_metrics[f'auc_{name}']
            
            wandb.log(test_log)
            
            # ===== Log Final Confusion Matrix =====
            self._log_confusion_matrix_to_wandb(epoch="test")
            
            # ===== W&B Summary (for easy comparison across runs) =====
            wandb.run.summary['best_val_accuracy'] = self.best_val_acc
            wandb.run.summary['best_val_f1'] = self.best_val_f1
            wandb.run.summary['test_accuracy'] = test_metrics['accuracy']
            wandb.run.summary['test_balanced_accuracy'] = test_metrics.get('balanced_accuracy', 0)
            wandb.run.summary['test_f1_weighted'] = test_metrics.get('f1_weighted', 0)
            wandb.run.summary['test_f1_macro'] = test_metrics.get('f1_macro', 0)
            wandb.run.summary['test_cohen_kappa'] = test_metrics.get('cohen_kappa', 0)
            wandb.run.summary['test_mcc'] = test_metrics.get('mcc', 0)
            if 'auc_ovr_macro' in test_metrics:
                wandb.run.summary['test_auc_macro'] = test_metrics['auc_ovr_macro']
    
    def _log_confusion_matrix_to_wandb(self, epoch):
        """Log confusion matrix to W&B."""
        if not self.use_wandb or not hasattr(self, '_last_val_labels'):
            return
            
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            cm = confusion_matrix(self._last_val_labels, self._last_val_preds)
            
            # Normalized confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt='.2%',
                cmap='Blues',
                xticklabels=self.config.data.class_names,
                yticklabels=self.config.data.class_names,
                ax=ax
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix (Epoch {epoch})')
            plt.tight_layout()
            
            wandb.log({f"confusion_matrix/epoch_{epoch}": wandb.Image(fig)})
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not log confusion matrix: {e}")
    
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
            # weights_only=False needed for checkpoints with config objects (PyTorch 2.6+)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
