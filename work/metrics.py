"""
Evaluation metrics module - Define various evaluation metrics for semantic segmentation
"""
import numpy as np
import paddle
import paddle.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


def calculate_iou(pred, target, threshold=0.5):
    pred = (F.sigmoid(pred) > threshold).astype('float32')  # 直接使用float32
    target = target.astype('float32')  # 直接使用float32
    
    intersection = paddle.sum(pred * target, axis=[1, 2, 3])  # 使用乘法而不是按位与
    union = paddle.sum(pred, axis=[1, 2, 3]) + paddle.sum(target, axis=[1, 2, 3]) - intersection
    iou = intersection / (union + 1e-6)
    
    return iou.mean().item()


def calculate_dice_coefficient(pred, target, threshold=0.5):
    """
    Calculate Dice coefficient
    """
    pred = (F.sigmoid(pred) > threshold).astype('float32')
    target = target.astype('float32')
    
    intersection = paddle.sum(pred * target)
    dice = (2. * intersection) / (paddle.sum(pred) + paddle.sum(target) + 1e-6)
    
    return dice.item()


def calculate_pixel_accuracy(pred, true):
    """
    Calculate pixel accuracy
    """
    pred = (pred > 0.5).astype('int64') if pred.shape[1] == 1 else pred.argmax(axis=1)
    true = true.astype('int64')
    correct = (pred == true).sum().astype('float32')
    total = true.size
    return float(correct / total)


def calculate_confusion_matrix_metrics(true_labels, pred_labels):
    """
    Calculate various metrics based on confusion matrix
    """
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)
    return precision, recall, f1, accuracy


def calculate_detailed_metrics(true_labels, pred_labels):
    """
    Calculate detailed segmentation metrics including F1, IoU, OA, MCC
    """
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        # Handle single class case
        if len(np.unique(true_labels)) == 1 and len(np.unique(pred_labels)) == 1:
            if true_labels[0] == pred_labels[0]:
                TP = len(true_labels)
                TN = FP = FN = 0
            else:
                FN = len(true_labels) if true_labels[0] == 1 else 0
                FP = len(pred_labels) if pred_labels[0] == 1 else 0
                TP = TN = 0
        else:
            # Handle other cases
            TP = TN = FP = FN = 0
    
    # Calculate F1-score
    denominator_f1 = 2 * TP + FP + FN
    f1_score = (2 * TP) / denominator_f1 if denominator_f1 > 0 else 0
    
    # Calculate IoU
    denominator_iou = TP + FP + FN
    iou = TP / denominator_iou if denominator_iou > 0 else 0
    
    # Calculate Overall Accuracy (OA)
    denominator_oa = TP + TN + FP + FN
    oa = (TP + TN) / denominator_oa if denominator_oa > 0 else 0
    
    # Calculate Matthews Correlation Coefficient (MCC)
    try:
        numerator = np.float64(TP) * np.float64(TN) - np.float64(FP) * np.float64(FN)
        s = np.float64(TP + FN) * np.float64(TP + FP) * np.float64(TN + FN) * np.float64(TN + FP)
        denominator = np.sqrt(s) if s > 0 else 0
        mcc = numerator / denominator if denominator > 0 else 0
    except (RuntimeWarning, FloatingPointError):
        mcc = 0
    
    return {
        'f1_score': f1_score,
        'iou': iou,
        'overall_accuracy': oa,
        'mcc': mcc,
        'confusion_matrix': cm,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }


def calculate_mean_iou(pred, target, num_classes):
    """
    Calculate mean IoU for multi-class segmentation
    """
    ious = []
    pred = pred.argmax(axis=1) if pred.shape[1] > 1 else (F.sigmoid(pred) > 0.5).astype('int64')
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).astype('float32')
        target_cls = (target == cls).astype('float32')
        
        intersection = paddle.sum(pred_cls * target_cls)
        union = paddle.sum(pred_cls) + paddle.sum(target_cls) - intersection
        
        if union > 0:
            iou = intersection / union
            ious.append(iou.item())
        else:
            ious.append(float('nan'))  # Ignore classes that don't appear
    
    # Calculate mean IoU, ignoring nan values
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0


class SegmentationMetrics:
    """
    Semantic segmentation metrics calculation class
    """
    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.total_correct = 0
        self.total_pixels = 0
        self.all_predictions = []
        self.all_targets = []
    
    def update(self, pred, target):
        """Update statistics"""
        if self.num_classes == 1:
            pred_binary = (F.sigmoid(pred) > 0.5).astype('int64')
            target_binary = (target > 0.5).astype('int64')
        else:
            pred_binary = pred.argmax(axis=1)
            target_binary = target.astype('int64')
        
        # Update pixel accuracy statistics
        correct = (pred_binary == target_binary).sum().astype('float32')
        self.total_correct += float(correct)
        self.total_pixels += int(target_binary.numel())
        
        # Save predictions for other metrics calculation
        self.all_predictions.extend(pred_binary.numpy().flatten())
        self.all_targets.extend(target_binary.numpy().flatten())
    
    def compute(self):
        """Calculate all metrics"""
        if self.total_pixels == 0:
            return {}
        
        pixel_accuracy = self.total_correct / self.total_pixels
        
        # Calculate other metrics
        precision, recall, f1, accuracy = calculate_confusion_matrix_metrics(
            self.all_targets, self.all_predictions
        )
        
        detailed_metrics = calculate_detailed_metrics(
            self.all_targets, self.all_predictions
        )
        
        return {
            'pixel_accuracy': pixel_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            **detailed_metrics
        }