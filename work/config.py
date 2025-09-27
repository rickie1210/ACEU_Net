"""
Configuration file - Centralized management of all hyperparameters and paths
"""
import os

class Config:
    # Data path configuration
    DATA_ROOT = '/home/aistudio/data/'
    
    # Training data paths
    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'Experiment1/train/images/')
    TRAIN_MASK_DIR = os.path.join(DATA_ROOT, 'Experiment1/train/labels/')
    
    # Test data paths
    TEST_IMG_DIR = os.path.join(DATA_ROOT, 'Experiment1/test/images/')
    TEST_MASK_DIR = os.path.join(DATA_ROOT, 'Experiment1/test/labels/')
    
    # Checkpoint save path
    CHECKPOINT_DIR = '/home/aistudio/work/points_check/'
    
    # Model configuration
    N_CHANNELS = 3  # RGB images
    N_CLASSES = 1   # Binary classification problem
    
    # Training hyperparameters
    EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    VALIDATION_PERCENT = 0.2
    IMG_SCALE = 1.0
    
    # Loss function weights
    BCE_WEIGHT = 0.6
    DICE_WEIGHT = 0.4
    
    # Optimizer configuration
    WEIGHT_DECAY = 1e-8
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    
    # Learning rate scheduler configuration
    LR_PATIENCE = 2
    GRAD_CLIP_NORM = 0.1
    
    # Logging configuration
    LOG_INTERVAL = 10  # Log every N batches
    
    # Environment variables
    @staticmethod
    def set_environment():
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"