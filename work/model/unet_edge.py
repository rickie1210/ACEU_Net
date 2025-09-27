import paddle
import paddle.nn.functional as F
from paddle import nn
import cv2
import numpy as np

class BaseEdgeDetector(nn.Layer):
    """Base edge detection module interface"""
    def __init__(self):
        super(BaseEdgeDetector, self).__init__()
        # Add edge enhancement convolution block - shared processing module for all methods
        self.edge_enhance = nn.Sequential(
            nn.Conv2D(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2D(16),
            nn.ReLU(),
            nn.Conv2D(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def detect_edges(self, img_gray):
        """Implement specific edge detection algorithm, implemented by subclasses"""
        raise NotImplementedError
        
    def forward(self, x):
        # Convert to numpy for processing
        x_np = x.numpy().transpose(0, 2, 3, 1)
        
        edges = []
        for img in x_np:
            # Convert to uint8
            img_uint8 = (img * 255).astype(np.uint8)
            # Convert to grayscale (if not already)
            if img_uint8.shape[-1] == 3:
                img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_uint8.squeeze()
                
            # Apply edge detection - implemented by subclasses
            edge = self.detect_edges(img_gray)
            
            # Expand dimensions
            edge = np.expand_dims(edge, axis=-1)
            edges.append(edge)
        
        # Convert back to paddle tensor
        edges = np.stack(edges, axis=0)
        edges = paddle.to_tensor(edges.transpose(0, 3, 1, 2), dtype='float32') / 255.0
        
        # Through learnable edge enhancement module
        enhanced_edges = self.edge_enhance(edges)
        
        return enhanced_edges


class AdaptiveCannyEdgeDetector(BaseEdgeDetector):
    """Improved adaptive Canny edge detection module"""
    def __init__(self, low_threshold=50, high_threshold=150, learning_rate=0.01):
        super(AdaptiveCannyEdgeDetector, self).__init__()
        # Learnable threshold parameters
        self.low_threshold = self.create_parameter(
            shape=[1], dtype='float32', 
            default_initializer=nn.initializer.Constant(low_threshold)
        )
        self.high_threshold = self.create_parameter(
            shape=[1], dtype='float32', 
            default_initializer=nn.initializer.Constant(high_threshold)
        )
        
        # Learning rate
        self.learning_rate = learning_rate
        
        # Add edge quality assessment network
        self.edge_quality_net = nn.Sequential(
            nn.Conv2D(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2D(1),
            nn.Flatten(),
            nn.Linear(8, 2),  # Output two values: one for low_threshold, one for high_threshold
            nn.Tanh()  # Limit output range to [-1, 1]
        )
        
        # Record historical thresholds for momentum updates
        self.low_th_ema = float(low_threshold)
        self.high_th_ema = float(high_threshold)
        self.ema_decay = 0.9
        
    def compute_edge_metric(self, edge_map):
        """Compute edge quality metrics"""
        # Edge density
        edge_density = paddle.mean(edge_map)
        
        # Edge continuity: compute adjacent pixel changes
        edge_continuity = paddle.mean(paddle.abs(
            edge_map[:, :, :-1, :-1] - edge_map[:, :, 1:, 1:]
        ))
        
        # Edge sharpness: compute gradient magnitude
        gradient_x = paddle.abs(edge_map[:, :, :, :-1] - edge_map[:, :, :, 1:])
        gradient_y = paddle.abs(edge_map[:, :, :-1, :] - edge_map[:, :, 1:, :])
        edge_sharpness = paddle.mean(gradient_x) + paddle.mean(gradient_y)
        
        return edge_density, edge_continuity, edge_sharpness
    
    def detect_edges(self, img_gray):
        # Get current thresholds
        low_th = paddle.clip(self.low_threshold, min=10, max=100)
        high_th = paddle.clip(self.high_threshold, min=low_th + 20, max=300)
        
        # Apply Canny
        edge = cv2.Canny(
            img_gray, 
            int(low_th.numpy()[0]), 
            int(high_th.numpy()[0])
        )
        return edge
    
    def forward(self, x):
        # First call parent class forward method to get edges
        enhanced_edges = super(AdaptiveCannyEdgeDetector, self).forward(x)
        
        # If in training mode, update thresholds
        if self.training:
            # Compute edge quality metrics
            edge_density, edge_continuity, edge_sharpness = self.compute_edge_metric(enhanced_edges)
            
            # Predict threshold adjustments through edge quality network
            threshold_adjustment = self.edge_quality_net(enhanced_edges)
            
            # Ensure adjustments are one-dimensional and get specific values
            if len(threshold_adjustment.shape) > 1:
                threshold_adjustment = threshold_adjustment.squeeze()
            
            low_arr = threshold_adjustment[0].numpy()
            high_arr = threshold_adjustment[1].numpy()
            low_adj = float(low_arr) if low_arr.ndim == 0 else float(low_arr[0])
            high_adj = float(high_arr) if high_arr.ndim == 0 else float(high_arr[0])
            
            # Compute new thresholds (use learning rate to control update speed)
            new_low = float(self.low_threshold.numpy()[0]) + self.learning_rate * low_adj * 10
            new_high = float(self.high_threshold.numpy()[0]) + self.learning_rate * high_adj * 20
            
            # Apply EMA update
            self.low_th_ema = self.ema_decay * self.low_th_ema + (1 - self.ema_decay) * new_low
            self.high_th_ema = self.ema_decay * self.high_th_ema + (1 - self.ema_decay) * new_high
            
            # Update parameters, ensure values are scalar
            clipped_low = np.clip(self.low_th_ema, 10, 100)
            clipped_high = np.clip(self.high_th_ema, self.low_th_ema + 20, 300)
            self.low_threshold.set_value(paddle.to_tensor([clipped_low], dtype='float32'))
            self.high_threshold.set_value(paddle.to_tensor([clipped_high], dtype='float32'))  
        return enhanced_edges
    
    def get_current_thresholds(self):
        """Get current thresholds (for monitoring and debugging)"""
        return {
            'low': float(self.low_threshold.numpy()[0]),
            'high': float(self.high_threshold.numpy()[0])
        }


# Ensure using the same component architecture, only changing the edge detection part

class ChannelAttention(nn.Layer):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2D(in_channels, in_channels // reduction_ratio, 1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(in_channels // reduction_ratio, in_channels, 1, bias_attr=False)
        )
        
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = F.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2D(2, 1, kernel_size, padding=kernel_size//2, bias_attr=False)
        
    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        out = paddle.concat([avg_out, max_out], axis=1)
        out = self.conv(out)
        out = F.sigmoid(out)
        return out


class EdgeWeightModule(nn.Layer):
    def __init__(self, in_channels, mid_channels=None):
        super(EdgeWeightModule, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2
        
        # Edge detection convolution layers
        self.edge_conv = nn.Sequential(
            nn.Conv2D(in_channels, mid_channels, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(mid_channels),
            nn.ReLU(),
            nn.Conv2D(mid_channels, 1, kernel_size=1, bias_attr=False)
        )
        
        # Edge weight computation
        self.weight_gate = nn.Sequential(
            nn.Conv2D(in_channels + 1, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(mid_channels),
            nn.ReLU(),
            nn.Conv2D(mid_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract edge features
        edge_maps = self.edge_conv(x)
        
        # Combine original features and edge features
        combined = paddle.concat([x, edge_maps], axis=1)
        
        # Compute edge weights
        weights = self.weight_gate(combined)
        
        # Apply weights
        weighted_features = x * weights
        
        return weighted_features


class EDAM(nn.Layer):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(EDAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.edge_weight = EdgeWeightModule(in_channels)
        
    def forward(self, x):
        # Original input saved for residual connection
        residual = x
        
        # Channel attention
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att
        
        # Edge weight enhancement
        edge_enhanced1 = self.edge_weight(residual)
        
        # First fusion
        edge_enhanced2 = edge_enhanced1 + x_channel 
        
        # Spatial attention
        spatial_att = self.spatial_attention(edge_enhanced2)
        out = edge_enhanced2 * spatial_att
        
        # Second residual connection
        out = out + edge_enhanced1
        
        return out


# InceptionBlock, Down, Up, etc. components (same as original code)
class InceptionBlock(nn.Layer):
    """Inception-style convolution block with optional EDAM attention mechanism"""
    def __init__(self, in_channels, out_channels, mid_channels=None, use_edam=False):
        super(InceptionBlock, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Ensure output channels divisible by 4
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        branch_channels = out_channels // 4
        
        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2D(branch_channels),
            nn.ReLU()
        )
        
        # 1x1 + 3x3 convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2D(branch_channels),
            nn.ReLU(),
            nn.Conv2D(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(branch_channels),
            nn.ReLU()
        )
        
        # 1x1 + 5x5 convolution branch (implemented as two 3x3)
        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2D(branch_channels),
            nn.ReLU(),
            nn.Conv2D(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(branch_channels),
            nn.ReLU(),
            nn.Conv2D(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(branch_channels),
            nn.ReLU()
        )
        
        # Pooling + 1x1 convolution branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2D(branch_channels),
            nn.ReLU()
        )
        
        # Add residual connection to enhance information flow
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=1, bias_attr=False),
                nn.BatchNorm2D(out_channels)
            )
            
        self.use_edam = use_edam
        if use_edam:
            self.edam = EDAM(out_channels)
    
    def forward(self, x):
        identity = x
        
        # Feature extraction from four branches
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate outputs from four branches
        outputs = paddle.concat([branch1, branch2, branch3, branch4], axis=1)
        
        # Residual connection
        if hasattr(self, 'shortcut'):
            outputs = outputs + self.shortcut(identity)
        
        # EDAM attention mechanism
        if self.use_edam:
            outputs = self.edam(outputs)
            
        return outputs


class Down(nn.Layer):
    """Downsampling module"""
    def __init__(self, in_channels, out_channels, use_edam=False):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2D(kernel_size=2, stride=2),
            InceptionBlock(in_channels, out_channels, use_edam=use_edam)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Layer):
    """Upsampling module"""
    def __init__(self, in_channels, out_channels, bilinear=True, use_edam=False):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = InceptionBlock(in_channels, out_channels, in_channels // 2, use_edam=use_edam)
        else:
            self.up = nn.Conv2DTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = InceptionBlock(in_channels, out_channels, use_edam=use_edam)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
                        
        # Feature fusion
        x = paddle.concat([x2, x1], axis=1)
        return self.conv(x)


class EdgeFusionModule(nn.Layer):
    """Edge feature fusion module"""
    def __init__(self, in_channels, edge_channels):
        super(EdgeFusionModule, self).__init__()
        
        # Edge feature transformation
        self.edge_conv = nn.Sequential(
            nn.Conv2D(edge_channels, in_channels//2, kernel_size=1),
            nn.BatchNorm2D(in_channels//2),
            nn.ReLU()
        )
        
        # Feature fusion post-processing, using InceptionBlock instead of original ConvBlock
        # Ensure input channel count is multiple of 4
        fusion_channels = ((in_channels + in_channels//2) // 4) * 4
        self.fusion_conv = InceptionBlock(in_channels + in_channels//2, fusion_channels, use_edam=True)
        
        # Output projection to match required channel count
        self.proj = None
        if fusion_channels != in_channels:
            self.proj = nn.Sequential(
                nn.Conv2D(fusion_channels, in_channels, kernel_size=1, bias_attr=False),
                nn.BatchNorm2D(in_channels)
            )
        
        # Feature selection gating
        self.gate = nn.Sequential(
            nn.Conv2D(in_channels + in_channels//2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge):
        # Transform edge features
        edge_feat = self.edge_conv(edge)
        
        # Concatenate features
        concat_feat = paddle.concat([x, edge_feat], axis=1)
        
        # Compute gate weights
        gate_weight = self.gate(concat_feat)
        
        # Apply fusion
        fused_feat = self.fusion_conv(concat_feat)
        
        # Adjust channel count if needed
        if self.proj is not None:
            fused_feat = self.proj(fused_feat)
        
        # Apply gating mechanism
        output = x + gate_weight * fused_feat
        
        return output


class PSPModule(nn.Layer):
    """Pyramid pooling module"""
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        
        # Reduce input channels to lower computation
        self.stages = nn.LayerList([
            nn.Sequential(
                nn.AdaptiveAvgPool2D(size),
                nn.Conv2D(in_channels, in_channels // len(sizes), 1, bias_attr=False),
                nn.BatchNorm2D(in_channels // len(sizes)),
                nn.ReLU()
            ) for size in sizes
        ])
        
        # Fuse multi-scale features
        self.bottleneck = nn.Sequential(
            nn.Conv2D(in_channels + in_channels, out_channels, 1),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        
        # Multi-scale feature extraction
        pyramids = []
        for stage in self.stages:
            pyramids.append(F.interpolate(
                stage(x), size=(h, w), mode='bilinear', align_corners=True
            ))
        
        # Concatenate multi-scale features with original features
        output = paddle.concat(pyramids + [x], axis=1)
        output = self.bottleneck(output)
        
        return output


class OutConv(nn.Layer):
    """Output convolution layer"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetEdge(nn.Layer):
    """General U-Net with Edge Detection model"""
    def __init__(self, n_channels, n_classes, bilinear=True, use_fusion_weights=False):
        super(UNetEdge, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Feature extraction base channel count
        # Ensure base channel count is multiple of 4 (requirement of Inception blocks)
        base_c = 64
        
        # Use AdaptiveCannyEdgeDetector only
        self.edge_detector = AdaptiveCannyEdgeDetector()
        
        # Encoder - use InceptionBlock instead of ConvBlock
        self.inc = InceptionBlock(n_channels, base_c)
        self.edge_inc = InceptionBlock(1, base_c)
        
        # Edge feature fusion
        self.edge_fusion1 = EdgeFusionModule(base_c, base_c)
        
        # Downsampling path
        self.down1 = Down(base_c, base_c*2, use_edam=True)
        self.down2 = Down(base_c*2, base_c*4, use_edam=True)
        self.down3 = Down(base_c*4, base_c*8, use_edam=True)
        
        # Bottleneck layer
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c*8, base_c*16//factor, use_edam=True)
        self.psp = PSPModule(base_c*16//factor, base_c*16//factor)
        
        # Upsampling path
        self.up1 = Up(base_c*16, base_c*8//factor, bilinear, use_edam=True)
        self.up2 = Up(base_c*8, base_c*4//factor, bilinear, use_edam=True)
        self.up3 = Up(base_c*4, base_c*2//factor, bilinear, use_edam=True)
        self.up4 = Up(base_c*2, base_c, bilinear, use_edam=True)
        
        # Output layer
        self.outc = OutConv(base_c, n_classes)
        
        # Deep supervision outputs
        self.deep_sup = nn.LayerList([
            nn.Sequential(
                nn.Conv2D(base_c*8//factor, n_classes, kernel_size=1),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            ),
            nn.Sequential(
                nn.Conv2D(base_c*4//factor, n_classes, kernel_size=1),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            ),
            nn.Sequential(
                nn.Conv2D(base_c*2//factor, n_classes, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
        ])
        
        # Fusion weights (optional as learnable parameters)
        self.use_fusion_weights = use_fusion_weights
        if use_fusion_weights:
            # Create learnable fusion weight parameters
            self.main_weight = self.create_parameter(
                shape=[1], dtype='float32',
                default_initializer=nn.initializer.Constant(0.6)
            )
            self.ds_weights = []
            for i in range(3):
                weight = self.create_parameter(
                    shape=[1], dtype='float32',
                    default_initializer=nn.initializer.Constant(0.2 if i == 0 else 0.1)
                )
                self.ds_weights.append(weight)
                # Add parameter to model's parameter list
                self.add_parameter(f"ds_weight_{i}", weight)

    def forward(self, x, return_deep_sup=False, fuse_predictions=True, return_edge=False):
        # Edge detection
        edge = self.edge_detector(x)
        edge_feat = self.edge_inc(edge)

        # Encoder path
        x1 = self.inc(x)
        x1 = self.edge_fusion1(x1, edge_feat)  # Fuse edge features
        
        # Downsampling
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck layer
        x5 = self.down4(x4)
        x5 = self.psp(x5)  # Apply PSP module for feature enhancement
        
        # Upsampling and collect deep supervision outputs
        x = self.up1(x5, x4)
        ds1 = self.deep_sup[0](x)
        
        x = self.up2(x, x3)
        ds2 = self.deep_sup[1](x)
        
        x = self.up3(x, x2)
        ds3 = self.deep_sup[2](x)
        
        x = self.up4(x, x1)
        
        # Main output
        logits = self.outc(x)
        
        # Handle return values
        results = []
        
        if fuse_predictions:
            # Fuse final prediction with deep supervision at all levels
            fused_logits = self.fuse_predictions(logits, [ds1, ds2, ds3])
            results.append(fused_logits)
        else:
            results.append(logits)
        
        if return_deep_sup:
            results.append([logits, ds1, ds2, ds3])
            
        if return_edge:
            results.append(edge)
            
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)
    
    # Fix: Properly define fuse_predictions as a class method
    def fuse_predictions(self, main_pred, deep_preds):
        """Fuse main prediction with deep supervision predictions
        
        Args:
            main_pred: Main output prediction
            deep_preds: List containing deep supervision predictions at all levels [ds1, ds2, ds3]
            
        Returns:
            Fused prediction result
        """
        if self.use_fusion_weights:
            # Use learnable weight parameters
            # Ensure weights sum to 1, use softmax for normalization
            all_weights = [self.main_weight] + self.ds_weights
            weights = F.softmax(paddle.stack(all_weights))
            
            # Apply weights to each prediction
            main_weight = weights[0]
            ds_weights = weights[1:]
            
            # Initialize fused result
            fused = main_weight * main_pred
            
            # Fuse deep supervision predictions at all levels
            for i, pred in enumerate(deep_preds):
                fused = fused + ds_weights[i] * pred
        else:
            # Use fixed weights
            main_weight = 0.6
            ds_weights = [0.2, 0.1, 0.1]  # Weights for ds1, ds2, ds3
            
            # Initialize fused result
            fused = main_weight * main_pred
            
            # Fuse deep supervision predictions at all levels
            for i, pred in enumerate(deep_preds):
                fused = fused + ds_weights[i] * pred
            
        return fused
