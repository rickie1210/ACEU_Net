"""
Loss function module - Define various loss functions for semantic segmentation
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer


class DiceLoss(nn.Layer):
    """
    DiceLoss
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, labels):
        probs = nn.functional.sigmoid(logits)
        intersection = paddle.sum(probs * labels)
        union = paddle.sum(probs) + paddle.sum(labels)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class BCE_DiceLoss(nn.Layer):
    """
    Combination of BCE and Dice loss, balances pixel-level classification and overall segmentation performance
    """
    def __init__(self, bce_weight=0.7, dice_weight=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, labels):
        loss_bce = self.bce(logits, labels)
        loss_dice = self.dice(logits, labels)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


class DiceCoeff(PyLayer):
    """
    Dice coefficient calculation for evaluating segmentation performance
    """
    @staticmethod
    def forward(ctx, input, target):
        eps = 0.0001
        inter = paddle.dot(input.reshape([-1]), target.reshape([-1]))
        union = paddle.sum(input) + paddle.sum(target) + eps
        
        t = (2 * inter + eps) / union
        ctx.save_for_backward(input, target, inter, union)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        input, target, inter, union = ctx.saved_tensor()
        grad_input = grad_target = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * union - inter) / (union * union)
        if ctx.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """
    Calculate Dice coefficient for batches
    """
    device = paddle.get_device()
    if device.startswith('gpu'):
        s = paddle.zeros([1], dtype='float32')
    else:
        s = paddle.zeros([1], dtype='float32')

    for i, (inp, tgt) in enumerate(zip(input, target)):
        s += DiceCoeff.apply(inp, tgt)

    return s / (i + 1)


class FocalLoss(nn.Layer):
    """
    Focal Loss for handling class imbalance problems
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = paddle.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return paddle.mean(focal_loss)
        elif self.reduction == 'sum':
            return paddle.sum(focal_loss)
        else:
            return focal_loss


class IoULoss(nn.Layer):
    """
    IoU loss function
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        
        intersection = paddle.sum(inputs * targets)
        union = paddle.sum(inputs) + paddle.sum(targets) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou


def get_loss_function(loss_type, **kwargs):
    """
    Factory function to return corresponding loss function based on type
    """
    loss_functions = {
        'dice': DiceLoss,
        'bce': nn.BCEWithLogitsLoss,
        'bce_dice': BCE_DiceLoss,
        'cross_entropy': nn.CrossEntropyLoss,
        'focal': FocalLoss,
        'iou': IoULoss
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_functions[loss_type](**kwargs)