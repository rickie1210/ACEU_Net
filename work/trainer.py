"""
Trainer module - Encapsulates model training logic
"""
import os
import logging
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
from paddle.io import DataLoader, random_split
from tqdm import tqdm
from visualdl import LogWriter

from config import Config
from utils.dataset import BasicDataset
from losses import get_loss_function
from evaluator import ModelEvaluator
from metrics import calculate_iou


class ModelTrainer:
    """
    Model trainer class that encapsulates all training process logic
    """
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or Config()
        self.device = paddle.get_device()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._setup_logging()
        self._setup_data_loaders()
        self._setup_training_components()
        self._setup_evaluator()
        
    def _setup_logging(self):
        """Set up logging and visualization"""
        log_dir = f'./log_LR_{self.config.LEARNING_RATE}_BS_{self.config.BATCH_SIZE}_SCALE_{self.config.IMG_SCALE}'
        self.writer = LogWriter(logdir=log_dir)
        
        self.logger.info(f'''Starting training setup:
            Epochs:          {self.config.EPOCHS}
            Batch size:      {self.config.BATCH_SIZE}
            Learning rate:   {self.config.LEARNING_RATE}
            Device:          {self.device}
            Images scaling:  {self.config.IMG_SCALE}
        ''')
    
    def _setup_data_loaders(self):
        """Set up data loaders"""
        # Training and validation data
        dataset = BasicDataset(
            self.config.TRAIN_IMG_DIR, 
            self.config.TRAIN_MASK_DIR, 
            self.config.IMG_SCALE
        )
        
        n_val = int(len(dataset) * self.config.VALIDATION_PERCENT)
        n_train = len(dataset) - n_val
        train_data, val_data = random_split(dataset, [n_train, n_val])
        
        self.train_loader = DataLoader(
            train_data, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=4, 
            use_shared_memory=True
        )
        
        self.val_loader = DataLoader(
            val_data, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=4, 
            use_shared_memory=True, 
            drop_last=True
        )
        
        # Test data (if exists)
        self.test_loader = None
        if os.path.exists(self.config.TEST_IMG_DIR) and os.path.exists(self.config.TEST_MASK_DIR):
            test_dataset = BasicDataset(
                self.config.TEST_IMG_DIR, 
                self.config.TEST_MASK_DIR, 
                self.config.IMG_SCALE
            )
            self.test_loader = DataLoader(
                test_dataset, 
                batch_size=self.config.BATCH_SIZE, 
                shuffle=False, 
                num_workers=4, 
                use_shared_memory=True
            )
        
        self.n_train = n_train
        self.n_val = n_val
        
        self.logger.info(f'Training size: {self.n_train}, Validation size: {self.n_val}')
    
    def _setup_training_components(self):
        """Set up training components: optimizer, loss function, scheduler"""
        # Learning rate scheduler
        self.scheduler = optim.lr.ReduceOnPlateau(
            learning_rate=self.config.LEARNING_RATE,
            mode='min' if self.model.n_classes > 1 else 'max',
            patience=self.config.LR_PATIENCE
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            learning_rate=self.scheduler,
            parameters=self.model.parameters(),
            weight_decay=self.config.WEIGHT_DECAY,
            beta1=self.config.ADAM_BETA1,
            beta2=self.config.ADAM_BETA2
        )
        
        # Loss function
        if self.model.n_classes > 1:
            self.criterion = get_loss_function('cross_entropy')
        else:
            self.criterion = get_loss_function(
                'bce_dice',
                bce_weight=self.config.BCE_WEIGHT,
                dice_weight=self.config.DICE_WEIGHT
            )
        
        self.logger.info(f'Using loss function: {type(self.criterion).__name__}')
    
    def _setup_evaluator(self):
        """Set up evaluator"""
        self.evaluator = ModelEvaluator(self.model, self.device)
    
    def train_epoch(self, epoch):
        """Train single epoch"""
        self.model.train()
        epoch_loss = 0
        correct_train = 0
        total_train = 0
        global_step = epoch * len(self.train_loader)
        
        with tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.config.EPOCHS}', unit='img') as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                imgs = batch['image'].astype('float32')
                true_masks = batch['mask'].astype('float32' if self.model.n_classes == 1 else 'int64')
                
                masks_pred = self.model(imgs)
                loss = self.criterion(masks_pred, true_masks)
                
                epoch_loss += float(loss)
                self.writer.add_scalar('Loss/train_batch', float(loss), global_step + batch_idx)
                
                if self.model.n_classes == 1:
                    pred = (F.sigmoid(masks_pred) > 0.5).astype('int64')
                    true = (true_masks > 0.5).astype('int64')
                    correct = float((pred == true).sum())
                    total = int(pred.numel())
                    correct_train += correct
                    total_train += total
                
                self.optimizer.clear_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.GRAD_CLIP_NORM)
                self.optimizer.step()
                
                pbar.set_postfix(**{'loss (batch)': float(loss)})
                pbar.update(imgs.shape[0])
                
                if batch_idx % (len(self.train_loader) // self.config.LOG_INTERVAL) == 0:
                    self._log_training_progress(global_step + batch_idx, masks_pred, true_masks, imgs)
        
        train_loss = epoch_loss / len(self.train_loader)
        train_acc = correct_train / total_train if total_train > 0 else 0
        
        return {
            'train_loss': train_loss,
            'train_accuracy': train_acc
        }
    
    def validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with paddle.no_grad():
            for batch in self.val_loader:
                val_imgs = batch['image'].astype('float32')
                val_true_masks = batch['mask'].astype('float32' if self.model.n_classes == 1 else 'int64')
                
                val_masks_pred = self.model(val_imgs)
                val_batch_loss = self.criterion(val_masks_pred, val_true_masks)
                val_loss += float(val_batch_loss)
                
                if self.model.n_classes == 1:
                    val_pred = (F.sigmoid(val_masks_pred) > 0.5).astype('int64')
                    val_true = (val_true_masks > 0.5).astype('int64')
                    correct = float((val_pred == val_true).sum())
                    total = int(val_pred.numel())
                    correct_val += correct
                    total_val += total
        
        val_loss = val_loss / len(self.val_loader)
        val_acc = correct_val / total_val if total_val > 0 else 0
        dice_val = self.evaluator.evaluate_validation(self.val_loader)
        
        self.scheduler.step(dice_val)
        
        self.writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        self.writer.add_scalar('Accuracy/val_epoch', val_acc, epoch)
        self.writer.add_scalar('Dice/val_epoch', dice_val, epoch)
        self.writer.add_scalar('learning_rate', self.scheduler.last_lr, epoch)
        
        self.logger.info(
            f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Dice: {dice_val:.4f}'
        )
        
        return {
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_dice': dice_val
        }
    
    def _log_training_progress(self, global_step, masks_pred, true_masks, imgs):
        """Log images and metrics during training"""
        # Record parameter histograms
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param.numpy(), global_step)
            if param.grad is not None:
                self.writer.add_histogram(name + '/grad', param.grad.numpy(), global_step)
        
        # Quick validation
        val_score = self.evaluator.evaluate_validation(self.val_loader)
        iou_score = calculate_iou(masks_pred, true_masks)
        
        if self.model.n_classes > 1:
            self.logger.info('Validation cross entropy: {}'.format(val_score))
            self.writer.add_scalar('Loss/val_batch', val_score, global_step)
        else:
            self.logger.info('Validation Dice Coeff: {}'.format(val_score))
            self.writer.add_scalar('Dice/val_batch', val_score, global_step)
            self.logger.info(f'Validation IoU: {iou_score}')
            self.writer.add_scalar('IoU/val_batch', iou_score, global_step)
        
        self._log_images(global_step, imgs, true_masks, masks_pred)
    
    def _log_images(self, global_step, imgs, true_masks, masks_pred):
        imgs_transposed = paddle.transpose(imgs, perm=[0, 2, 3, 1])
        single_img = imgs_transposed[0].numpy()
        self.writer.add_image('images', single_img, global_step)
        
        if self.model.n_classes == 1:
            true_masks = paddle.transpose(true_masks, perm=[0, 2, 3, 1])
            true_masks = true_masks[0].numpy()
            self.writer.add_image('masks/true', true_masks, global_step)
            
            pred_mask = (F.sigmoid(masks_pred) > 0.5).astype('float32')
            pred_mask = paddle.transpose(pred_mask, perm=[0, 2, 3, 1])
            pred_mask = pred_mask[0].numpy()
            self.writer.add_image('masks/pred', pred_mask, global_step)
    
    def save_checkpoint(self, epoch, save_dir=None):
        if save_dir is None:
            save_dir = self.config.CHECKPOINT_DIR
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, f'CP_epoch{epoch + 1}.pdparams')
            paddle.save(self.model.state_dict(), checkpoint_path)
            self.logger.info(f'Checkpoint {epoch + 1} saved to {checkpoint_path}!')
            return checkpoint_path
        except Exception as e:
            self.logger.error(f'Failed to save checkpoint: {e}')
            return None
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        best_dice = 0
        best_checkpoint = None
        
        try:
            for epoch in range(self.config.EPOCHS):
                self.logger.info(f'Starting Epoch {epoch + 1}/{self.config.EPOCHS}')
                
                # Train one epoch
                train_results = self.train_epoch(epoch)
                
                # Validate one epoch
                val_results = self.validate_epoch(epoch)
                
                # Record epoch results
                self.writer.add_scalar('Loss/train_epoch', train_results['train_loss'], epoch)
                self.writer.add_scalar('Accuracy/train_epoch', train_results['train_accuracy'], epoch)
                
                # Save best model
                if val_results['val_dice'] > best_dice:
                    best_dice = val_results['val_dice']
                    best_checkpoint = self.save_checkpoint(epoch)
                
                # Save checkpoint periodically
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(epoch)
                
                self.logger.info(
                    f'Epoch {epoch + 1} completed - '
                    f'Train Loss: {train_results["train_loss"]:.4f}, '
                    f'Val Dice: {val_results["val_dice"]:.4f}'
                )

        except KeyboardInterrupt:
            self.logger.info('Training interrupted by user')
            interrupt_checkpoint = os.path.join(self.config.CHECKPOINT_DIR, 'INTERRUPTED.pdparams')
            paddle.save(self.model.state_dict(), interrupt_checkpoint)
            self.logger.info(f'Interrupt checkpoint saved to {interrupt_checkpoint}')
            raise

        except Exception as e:  
            self.logger.error(f'Training failed with error: {e}')
            self.logger.error(f'Error type: {type(e).__name__}')
            import traceback
            self.logger.error(f'Full traceback: {traceback.format_exc()}')
            raise

        finally:
            # Test set evaluation
            if self.test_loader is not None:
                self.logger.info("Starting test evaluation...")
                test_results = self.evaluator.evaluate_test(self.test_loader, detailed=True)
                
                # Record test results
                self.writer.add_scalar('Metrics/Test_F1_score', test_results.get('f1_score', 0), self.config.EPOCHS)
                self.writer.add_scalar('Metrics/Test_IoU', test_results.get('iou', 0), self.config.EPOCHS)
                self.writer.add_scalar('Metrics/Test_OA', test_results.get('overall_accuracy', 0), self.config.EPOCHS)
                self.writer.add_scalar('Metrics/Test_MCC', test_results.get('mcc', 0), self.config.EPOCHS)
            
            # Close writer
            self.writer.close()
            self.logger.info("Training completed!")
            
            return best_checkpoint
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        try:
            self.model.set_state_dict(paddle.load(checkpoint_path))
            self.logger.info(f'Model loaded from {checkpoint_path}')
            return True
        except Exception as e:
            self.logger.error(f'Failed to load checkpoint from {checkpoint_path}: {e}')
            return False