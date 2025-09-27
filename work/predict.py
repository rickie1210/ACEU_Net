"""
Model prediction script - Generate segmentation predictions from trained models
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image
from tqdm import tqdm

from config import Config
from model import UNetEdge
from utils.dataset import BasicDataset


class SegmentationPredictor:
    """
    Segmentation prediction class that handles model loading and inference
    """
    
    def __init__(self, model_path, model_name='UNetEdge', n_channels=3, n_classes=1, device='auto'):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.model_name = model_name
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.device = self._setup_device(device)
        self.model = self._load_model()
        
    def _setup_device(self, device):
        """Setup computing device"""
        if device == 'auto':
            device = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'
        self.logger.info(f'Using device: {device}')
        return paddle.set_device(device)
    
    def _get_model_class(self, model_name):
        """Get model class by name (UNetEdge only)"""
        if model_name != 'UNetEdge':
            raise ValueError(f"Only UNetEdge model is supported, got: {model_name}")
        
        return UNetEdge
    
    def _load_model(self):
        """Load model from checkpoint"""
        try:
            model_class = self._get_model_class(self.model_name)
            
            # Try to create model with bilinear parameter
            try:
                model = model_class(
                    n_channels=self.n_channels, 
                    n_classes=self.n_classes, 
                    bilinear=True
                )
            except TypeError:
                # If model doesn't support bilinear parameter
                model = model_class(
                    n_channels=self.n_channels, 
                    n_classes=self.n_classes
                )
            
            # Load weights
            state_dict = paddle.load(self.model_path)
            model.set_state_dict(state_dict)
            model.eval()
            
            self.logger.info(f"Model {self.model_name} loaded successfully from {self.model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image, scale_factor=1.0):
        """Preprocess input image"""
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Use BasicDataset preprocessing
        processed = BasicDataset.preprocess(Image.fromarray(image_np), scale_factor)
        
        # Convert to tensor and add batch dimension
        tensor = paddle.to_tensor(processed, dtype='float32')
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def postprocess_output(self, output, original_size, threshold=0.5, use_adaptive_threshold=False):
        """Postprocess model output to generate final mask"""
        # Get probability map
        if self.n_classes > 1:
            probs = F.softmax(output, axis=1)
            probs = probs.argmax(axis=1, keepdim=True).astype('float32')
        else:
            probs = F.sigmoid(output)
        
        # Convert to numpy
        probs_np = probs.squeeze().numpy()
        
        # Resize to original image size
        if probs_np.shape != original_size[::-1]:  # (H, W) vs (W, H)
            probs_np = cv2.resize(
                probs_np, 
                original_size, 
                interpolation=cv2.INTER_LINEAR
            )
        
        # Apply thresholding
        if use_adaptive_threshold and self.n_classes == 1:
            # Use Otsu's method for adaptive thresholding
            prob_img_8bit = (probs_np * 255).astype(np.uint8)
            otsu_thresh, binary_mask = cv2.threshold(
                prob_img_8bit, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            binary_mask = binary_mask / 255.0
            self.logger.info(f"Otsu threshold: {otsu_thresh/255:.3f}")
        else:
            # Use fixed threshold
            binary_mask = (probs_np > threshold).astype(np.float32)
        
        # Post-processing: remove small noise regions
        if np.sum(binary_mask) > 0:
            binary_mask_uint8 = binary_mask.astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(binary_mask_uint8)
            
            # Remove small components
            min_component_size = 50  # pixels
            for i in range(1, num_labels):
                if np.sum(labels == i) < min_component_size:
                    binary_mask[labels == i] = 0
        
        return binary_mask, probs_np
    
    def predict_single(self, image, scale_factor=1.0, threshold=0.5, use_adaptive_threshold=False):
        """Predict mask for a single image"""
        original_size = image.size if isinstance(image, Image.Image) else image.shape[:2][::-1]
        
        # Preprocess
        input_tensor = self.preprocess_image(image, scale_factor)
        
        # Inference
        with paddle.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        binary_mask, prob_map = self.postprocess_output(
            output, original_size, threshold, use_adaptive_threshold
        )
        
        return binary_mask, prob_map
    
    def predict_batch(self, images, scale_factor=1.0, threshold=0.5, use_adaptive_threshold=False):
        """Predict masks for a batch of images"""
        results = []
        
        for image in tqdm(images, desc="Processing images"):
            mask, prob = self.predict_single(image, scale_factor, threshold, use_adaptive_threshold)
            results.append((mask, prob))
        
        return results


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate segmentation predictions from trained models (UNetEdge only)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--model-path', '-m', required=True,
                        help='Path to model weights file (.pdparams)')
    parser.add_argument('--model-name', default='UNetEdge',
                        choices=['UNetEdge'],
                        help='Model architecture')
    parser.add_argument('--n-channels', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--n-classes', type=int, default=1,
                        help='Number of output classes')
    
    # Input/Output parameters
    parser.add_argument('--input', '-i', required=True,
                        help='Input image file or directory')
    parser.add_argument('--output', '-o',
                        help='Output directory (default: input_dir/predictions)')
    parser.add_argument('--input-format', default='*.png,*.jpg,*.jpeg,*.bmp,*.tif',
                        help='Input image formats (comma-separated)')
    
    # Processing parameters
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for input images')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Threshold for binary mask generation')
    parser.add_argument('--adaptive-threshold', '-a', action='store_true',
                        help='Use adaptive (Otsu) thresholding')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='Batch size for processing')
    
    # Output options
    parser.add_argument('--save-prob', action='store_true',
                        help='Save probability maps alongside binary masks')
    parser.add_argument('--save-overlay', action='store_true',
                        help='Save overlay of mask on original image')
    parser.add_argument('--no-binary', action='store_true',
                        help='Skip saving binary masks')
    
    # Visualization
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Show prediction results')
    
    # Device
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto',
                        help='Computing device')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_image_files(input_path, formats):
    """Get list of image files from input path"""
    formats = [fmt.strip().replace('*', '') for fmt in formats.split(',')]
    
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        image_files = []
        for fmt in formats:
            pattern = f"*{fmt}"
            image_files.extend(Path(input_path).glob(pattern))
        return [str(f) for f in sorted(image_files)]
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def save_results(image_files, results, output_dir, save_prob=False, save_overlay=False, no_binary=False):
    """Save prediction results"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (image_file, (mask, prob)) in enumerate(zip(image_files, results)):
        base_name = Path(image_file).stem
        
        # Save binary mask
        if not no_binary:
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(mask_path)
        
        # Save probability map
        if save_prob:
            prob_path = os.path.join(output_dir, f"{base_name}_prob.png")
            prob_img = Image.fromarray((prob * 255).astype(np.uint8))
            prob_img.save(prob_path)
        
        # Save overlay
        if save_overlay:
            original_img = Image.open(image_file).convert('RGB')
            original_np = np.array(original_img)
            
            # Create colored mask overlay
            mask_colored = np.zeros_like(original_np)
            mask_colored[mask > 0] = [255, 0, 0]  # Red overlay
            
            # Blend with original image
            overlay = cv2.addWeighted(original_np, 0.7, mask_colored, 0.3, 0)
            
            overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
            Image.fromarray(overlay).save(overlay_path)


def visualize_results(image_files, results, max_display=5):
    """Visualize prediction results"""
    try:
        import matplotlib.pyplot as plt
        
        n_display = min(len(results), max_display)
        fig, axes = plt.subplots(n_display, 3, figsize=(15, 5 * n_display))
        
        if n_display == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_display):
            image_file = image_files[i]
            mask, prob = results[i]
            
            # Original image
            original = Image.open(image_file)
            axes[i, 0].imshow(original)
            axes[i, 0].set_title(f'Original - {Path(image_file).name}')
            axes[i, 0].axis('off')
            
            # Probability map
            axes[i, 1].imshow(prob, cmap='hot')
            axes[i, 1].set_title('Probability Map')
            axes[i, 1].axis('off')
            
            # Binary mask
            axes[i, 2].imshow(mask, cmap='gray')
            axes[i, 2].set_title('Binary Mask')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logging.warning("Matplotlib not available, skipping visualization")


def main():
    """Main function"""
    args = get_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return 1
    
    # Get input files
    try:
        image_files = get_image_files(args.input, args.input_format)
        if not image_files:
            logger.error(f"No image files found in: {args.input}")
            return 1
        logger.info(f"Found {len(image_files)} images to process")
    except Exception as e:
        logger.error(f"Error reading input: {e}")
        return 1
    
    # Setup output directory
    if args.output:
        output_dir = args.output
    else:
        if os.path.isfile(args.input):
            output_dir = os.path.join(os.path.dirname(args.input), 'predictions')
        else:
            output_dir = os.path.join(args.input, 'predictions')
    
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize predictor
    try:
        predictor = SegmentationPredictor(
            model_path=args.model_path,
            model_name=args.model_name,
            n_channels=args.n_channels,
            n_classes=args.n_classes,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        return 1
    
    # Load images
    try:
        images = []
        for image_file in image_files:
            img = Image.open(image_file).convert('RGB')
            images.append(img)
        logger.info(f"Loaded {len(images)} images")
    except Exception as e:
        logger.error(f"Error loading images: {e}")
        return 1
    
    # Run predictions
    try:
        start_time = time.time()
        results = predictor.predict_batch(
            images,
            scale_factor=args.scale,
            threshold=args.threshold,
            use_adaptive_threshold=args.adaptive_threshold
        )
        end_time = time.time()
        
        logger.info(f"Prediction completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Average time per image: {(end_time - start_time) / len(images):.2f} seconds")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1
    
    # Save results
    try:
        save_results(
            image_files, results, output_dir,
            save_prob=args.save_prob,
            save_overlay=args.save_overlay,
            no_binary=args.no_binary
        )
        logger.info(f"Results saved to: {output_dir}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return 1
    
    # Visualize if requested
    if args.visualize:
        visualize_results(image_files, results)
    
    logger.info("Prediction completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())