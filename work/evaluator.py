"""
Evaluator module - Encapsulates model evaluation logic
"""
import logging
import paddle
import paddle.nn.functional as F
from tqdm import tqdm

from losses import dice_coeff
from metrics import SegmentationMetrics, calculate_detailed_metrics


class ModelEvaluator:
    """
    Model evaluator class that encapsulates all evaluation-related functionality
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def evaluate_validation(self, loader):
        """
        Evaluate model on validation set, returns Dice coefficient
        """
        self.model.eval()
        mask_type = 'float32' if self.model.n_classes == 1 else 'int64'
        n_val = len(loader)
        tot = 0

        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for batch in loader:
                imgs, true_masks = batch['image'], batch['mask']
                imgs = imgs.astype('float32')
                true_masks = true_masks.astype(mask_type)

                with paddle.no_grad():
                    mask_pred = self.model(imgs)

                if self.model.n_classes > 1:
                    tot += F.cross_entropy(mask_pred, true_masks).item()
                else:
                    pred = F.sigmoid(mask_pred)
                    pred = (pred > 0.5).astype('float32')
                    tot += dice_coeff(pred, true_masks).item()
                pbar.update()

        self.model.train()
        return tot / n_val
    
    def evaluate_test(self, loader, detailed=True):
        """
        Evaluate model on test set, returns detailed metrics
        """
        self.model.eval()
        metrics = SegmentationMetrics(num_classes=self.model.n_classes)
        
        all_true_masks = []
        all_pred_masks = []

        with paddle.no_grad():
            with tqdm(total=len(loader), desc='Testing', unit='batch') as pbar:
                for batch in loader:
                    imgs = batch['image'].astype('float32')
                    true_masks = batch['mask']
                    
                    if self.model.n_classes == 1:
                        true_masks = true_masks.astype('float32')
                    else:
                        true_masks = true_masks.astype('int64')

                    # Model prediction
                    masks_pred = self.model(imgs)

                    # Update metrics
                    metrics.update(masks_pred, true_masks)
                    
                    # Collect predictions for detailed analysis
                    if detailed:
                        if self.model.n_classes == 1:
                            pred = (F.sigmoid(masks_pred) > 0.5).astype('int64')
                            true = (true_masks > 0.5).astype('int64')
                        else:
                            pred = masks_pred.argmax(axis=1)
                            true = true_masks.astype('int64')
                        
                        all_true_masks.extend(true.numpy().flatten())
                        all_pred_masks.extend(pred.numpy().flatten())
                    
                    pbar.update()

        # Calculate basic metrics
        results = metrics.compute()
        
        # If detailed metrics are needed, calculate additional metrics
        if detailed and all_true_masks:
            detailed_metrics = calculate_detailed_metrics(all_true_masks, all_pred_masks)
            results.update(detailed_metrics)
            
            # Log detailed results
            self.logger.info("=== Test Results ===")
            self.logger.info(f"IoU: {results.get('iou', 0):.4f}")
            self.logger.info(f"Dice Coefficient: {results.get('f1_score', 0):.4f}")
            self.logger.info(f"Precision: {results.get('precision', 0):.4f}")
            self.logger.info(f"Recall: {results.get('recall', 0):.4f}")
            self.logger.info(f"F1 Score: {results.get('f1_score', 0):.4f}")
            self.logger.info(f"Overall Accuracy: {results.get('overall_accuracy', 0):.4f}")
            self.logger.info(f"MCC: {results.get('mcc', 0):.4f}")
            
            if 'confusion_matrix' in results:
                self.logger.info(f"Confusion Matrix:\n{results['confusion_matrix']}")

        return results
    
    def evaluate_single_batch(self, imgs, true_masks):
        """
        Evaluate single batch, used for quick evaluation during training
        """
        self.model.eval()
        
        with paddle.no_grad():
            masks_pred = self.model(imgs)
            
            # Calculate loss
            if self.model.n_classes == 1:
                pred = F.sigmoid(masks_pred)
                pred_binary = (pred > 0.5).astype('float32')
                dice_score = dice_coeff(pred_binary, true_masks).item()
                return {'dice': dice_score, 'predictions': masks_pred}
            else:
                loss = F.cross_entropy(masks_pred, true_masks)
                return {'cross_entropy': loss.item(), 'predictions': masks_pred}
    
    def compute_class_metrics(self, loader):
        """
        Calculate detailed metrics for each class (for multi-class segmentation)
        """
        if self.model.n_classes <= 1:
            self.logger.warning("Class metrics only available for multi-class segmentation")
            return {}
        
        self.model.eval()
        class_metrics = {}
        
        # Initialize statistics for each class
        for class_id in range(self.model.n_classes):
            class_metrics[class_id] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        
        with paddle.no_grad():
            for batch in tqdm(loader, desc='Computing class metrics'):
                imgs = batch['image'].astype('float32')
                true_masks = batch['mask'].astype('int64')
                
                masks_pred = self.model(imgs)
                pred_classes = masks_pred.argmax(axis=1)
                
                # Calculate confusion matrix elements for each class
                for class_id in range(self.model.n_classes):
                    pred_class = (pred_classes == class_id).astype('int64')
                    true_class = (true_masks == class_id).astype('int64')
                    
                    class_metrics[class_id]['tp'] += int((pred_class * true_class).sum())
                    class_metrics[class_id]['fp'] += int((pred_class * (1 - true_class)).sum())
                    class_metrics[class_id]['tn'] += int(((1 - pred_class) * (1 - true_class)).sum())
                    class_metrics[class_id]['fn'] += int(((1 - pred_class) * true_class).sum())
        
        # Calculate metrics for each class
        results = {}
        for class_id, stats in class_metrics.items():
            tp, fp, tn, fn = stats['tp'], stats['fp'], stats['tn'], stats['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            
            results[f'class_{class_id}'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'iou': iou,
                'support': tp + fn
            }
        
        return results


def create_evaluator(model, device):
    """
    Factory function to create evaluator instance
    """
    return ModelEvaluator(model, device)