"""
Main entry file - Program entry point
"""
import os
import sys
import logging
import argparse
import paddle

from config import Config
from trainer import ModelTrainer
from model import UNetEdge


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train semantic segmentation models (UNetEdge only)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument('-e', '--epochs', type=int, default=Config.EPOCHS,
                        help='Number of epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('-l', '--learning-rate', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('-s', '--scale', type=float, default=Config.IMG_SCALE,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', type=float, default=Config.VALIDATION_PERCENT * 100,
                        help='Percent of the data used as validation (0-100)')
    
    # Model parameters
    parser.add_argument('-m', '--model', type=str, default='UNetEdge',
                        choices=['UNetEdge'],
                        help='Model architecture to use')
    
    # File path parameters
    parser.add_argument('--train-img', type=str, default=Config.TRAIN_IMG_DIR,
                        help='Training images directory')
    parser.add_argument('--train-mask', type=str, default=Config.TRAIN_MASK_DIR,
                        help='Training masks directory')
    parser.add_argument('--test-img', type=str, default=Config.TEST_IMG_DIR,
                        help='Test images directory')
    parser.add_argument('--test-mask', type=str, default=Config.TEST_MASK_DIR,
                        help='Test masks directory')
    parser.add_argument('--checkpoint-dir', type=str, default=Config.CHECKPOINT_DIR,
                        help='Directory to save checkpoints')
    
    # Model loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load model from checkpoint file')
    
    # Loss function parameters
    parser.add_argument('--bce-weight', type=float, default=Config.BCE_WEIGHT,
                        help='BCE loss weight in combined loss')
    parser.add_argument('--dice-weight', type=float, default=Config.DICE_WEIGHT,
                        help='Dice loss weight in combined loss')
    
    # Running mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'test'],
                        help='Running mode: train, evaluate, or test')
    
    return parser.parse_args()


def get_model(model_name, n_channels=3, n_classes=1):
    """Get model instance based on name (UNetEdge only)"""
    if model_name != 'UNetEdge':
        raise ValueError(f"Only UNetEdge model is supported, got: {model_name}")
    
    # Create UNetEdge instance
    try:
        # Try to use bilinear parameter
        return UNetEdge(n_channels=n_channels, n_classes=n_classes, bilinear=True)
    except TypeError:
        # If model doesn't support bilinear parameter, don't use it
        return UNetEdge(n_channels=n_channels, n_classes=n_classes)


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


def main():
    """Main function"""
    # Set up environment
    Config.set_environment()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    args = get_args()
    
    # Set up device
    device = paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create config instance
    config = Config()
    
    # Update configuration (from command line arguments)
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.IMG_SCALE = args.scale
    config.VALIDATION_PERCENT = args.validation / 100
    config.TRAIN_IMG_DIR = args.train_img
    config.TRAIN_MASK_DIR = args.train_mask
    config.TEST_IMG_DIR = args.test_img
    config.TEST_MASK_DIR = args.test_mask
    config.CHECKPOINT_DIR = args.checkpoint_dir
    config.BCE_WEIGHT = args.bce_weight
    config.DICE_WEIGHT = args.dice_weight
    
    # Create model
    try:
        model = get_model(args.model, n_channels=config.N_CHANNELS, n_classes=config.N_CLASSES)
        logger.info(f'Created model: {args.model}')
        logger.info(f'Model parameters:')
        logger.info(f'  Input channels: {model.n_channels}')
        logger.info(f'  Output classes: {model.n_classes}')
        
        # Check if model has bilinear attribute
        if hasattr(model, 'bilinear'):
            logger.info(f'  Upscaling: {"Bilinear" if model.bilinear else "Transposed conv"}')
        
    except Exception as e:
        logger.error(f'Failed to create model {args.model}: {e}')
        return 1
    
    # Load pretrained model (if specified)
    if args.load:
        try:
            model.set_state_dict(paddle.load(args.load))
            logger.info(f'Loaded model from {args.load}')
        except Exception as e:
            logger.error(f'Failed to load model from {args.load}: {e}')
            return 1
    
    # Create trainer
    trainer = ModelTrainer(model, config)
    
    try:
        if args.mode == 'train':
            # Training mode
            logger.info("Starting training...")
            best_checkpoint = trainer.train()
            if best_checkpoint:
                logger.info(f"Training completed. Best model saved to: {best_checkpoint}")
            
        elif args.mode == 'evaluate':
            # Validation mode
            logger.info("Starting validation evaluation...")
            val_score = trainer.evaluator.evaluate_validation(trainer.val_loader)
            logger.info(f"Validation score: {val_score:.4f}")
            
        elif args.mode == 'test':
            # Test mode
            if trainer.test_loader is None:
                logger.error("Test data not available. Please check test data paths.")
                return 1
            
            logger.info("Starting test evaluation...")
            test_results = trainer.evaluator.evaluate_test(trainer.test_loader, detailed=True)
            
            # Print test results
            logger.info("=== Test Results Summary ===")
            for metric, value in test_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{metric}: {value:.4f}")
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return 1
    
    logger.info("Process completed successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main())