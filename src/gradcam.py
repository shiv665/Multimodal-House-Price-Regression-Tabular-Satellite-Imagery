import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from src.config import cfg


class GradCAM:
    """
    Enhanced Grad-CAM++ implementation for better satellite imagery visualization.
    
    Improvements over basic Grad-CAM:
    1. Grad-CAM++ weighting for better localization
    2. Multi-layer fusion for richer feature maps
    3. Contrast enhancement for clearer heatmaps
    4. Gaussian smoothing to reduce noise
    5. Better normalization using percentile clipping
    """
    
    def __init__(self, model, target_layer_idx=-2, use_gradcam_pp=True):
        """
        Initialize Enhanced GradCAM for the HybridMultimodalModel.
        
        Args:
            model: HybridMultimodalModel instance
            target_layer_idx: Index of the layer in image_encoder to use for CAM
                             Default -2 targets the last conv block (layer4)
            use_gradcam_pp: Use Grad-CAM++ weighting (better localization)
        """
        self.model = model
        self.use_gradcam_pp = use_gradcam_pp
        
        # Get the target layer from image_encoder (ResNet without final fc)
        encoder_children = list(self.model.image_encoder.children())
        self.target_layer = encoder_children[target_layer_idx]  # layer4 by default
        
        # Also hook earlier layer for multi-scale fusion
        self.target_layer_early = encoder_children[-3] if len(encoder_children) > 3 else None  # layer3
        
        self.gradients = None
        self.activations = None
        self.gradients_early = None
        self.activations_early = None
        
        # Register hooks for main layer
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)
        
        # Register hooks for early layer (multi-scale)
        if self.target_layer_early is not None:
            self.target_layer_early.register_forward_hook(self._forward_hook_early)
            self.target_layer_early.register_full_backward_hook(self._backward_hook_early)

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def _backward_hook_early(self, module, grad_input, grad_output):
        self.gradients_early = grad_output[0]

    def _forward_hook(self, module, input, output):
        self.activations = output
    
    def _forward_hook_early(self, module, input, output):
        self.activations_early = output

    def _compute_gradcam_pp_weights(self, grads, acts):
        """
        Compute Grad-CAM++ weights for better pixel-level localization.
        Grad-CAM++ uses second and third derivatives for better weighting.
        """
        # Grad-CAM++ alpha computation
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        
        # Sum of activations across spatial dimensions
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        
        # Alpha weights (Grad-CAM++ formula)
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # Apply ReLU to gradients and weight by alpha
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        
        return weights

    def _compute_cam(self, grads, acts, use_pp=True):
        """Compute the class activation map."""
        if use_pp and self.use_gradcam_pp:
            weights = self._compute_gradcam_pp_weights(grads, acts)
        else:
            # Standard Grad-CAM weights
            weights = grads.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)
        
        return cam

    def _enhance_contrast(self, cam, percentile_low=2, percentile_high=98):
        """
        Enhance contrast using percentile-based normalization.
        This makes subtle activations more visible.
        """
        # Percentile clipping for better contrast
        low = np.percentile(cam, percentile_low)
        high = np.percentile(cam, percentile_high)
        
        cam = np.clip(cam, low, high)
        cam = (cam - low) / (high - low + 1e-8)
        
        # Apply power transform to boost mid-range values
        cam = np.power(cam, 0.7)  # Gamma correction
        
        return cam

    def __call__(self, img_tensor, tab_tensor, smooth=True, multi_scale=True, 
                 enhance_contrast=True):
        """
        Generate enhanced Grad-CAM visualization.
        
        Args:
            img_tensor: Image tensor
            tab_tensor: Tabular feature tensor
            smooth: Apply Gaussian smoothing to reduce noise
            multi_scale: Fuse multiple layers for richer visualization
            enhance_contrast: Apply contrast enhancement
            
        Returns:
            cam_resized: Resized CAM heatmap (0-1 normalized)
        """
        self.model.eval()
        
        # Prepare inputs
        img_tensor = img_tensor.unsqueeze(0).to(cfg.device)
        tab_tensor = tab_tensor.unsqueeze(0).to(cfg.device)
        img_tensor.requires_grad_(True)
        
        # Forward pass
        out = self.model(img_tensor, tab_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        out.backward(retain_graph=True)
        
        # Compute main CAM
        grads = self.gradients
        acts = self.activations
        cam = self._compute_cam(grads, acts)
        cam = cam.squeeze().detach().cpu().numpy()
        
        # Multi-scale fusion with earlier layer
        if multi_scale and self.gradients_early is not None:
            grads_early = self.gradients_early
            acts_early = self.activations_early
            cam_early = self._compute_cam(grads_early, acts_early, use_pp=False)
            cam_early = cam_early.squeeze().detach().cpu().numpy()
            
            # Resize early CAM to match main CAM
            cam_early_resized = cv2.resize(cam_early, (cam.shape[1], cam.shape[0]), 
                                           interpolation=cv2.INTER_LINEAR)
            
            # Normalize both before fusion
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam_early_resized = (cam_early_resized - cam_early_resized.min()) / \
                               (cam_early_resized.max() - cam_early_resized.min() + 1e-8)
            
            # Weighted fusion (main layer gets more weight)
            cam = 0.7 * cam + 0.3 * cam_early_resized
        
        # Apply Gaussian smoothing to reduce noise
        if smooth:
            cam = gaussian_filter(cam, sigma=1.0)
        
        # Enhance contrast
        if enhance_contrast:
            cam = self._enhance_contrast(cam)
        else:
            # Basic normalization
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to image size
        cam_resized = cv2.resize(cam, (cfg.img_size, cfg.img_size), 
                                  interpolation=cv2.INTER_LINEAR)
        
        return cam_resized


def create_enhanced_overlay(original_img, cam, alpha=0.5, colormap=cv2.COLORMAP_JET,
                           threshold=0.3):
    """
    Create an enhanced overlay visualization.
    
    Args:
        original_img: Original image (H, W, 3) uint8
        cam: CAM heatmap (H, W) float 0-1
        alpha: Blend weight for heatmap
        colormap: OpenCV colormap to use
        threshold: Minimum activation to show (increases contrast)
        
    Returns:
        overlay: Blended visualization
    """
    # Apply threshold to reduce noise
    cam_thresholded = np.where(cam > threshold, cam, 0)
    
    # Re-normalize after thresholding
    if cam_thresholded.max() > 0:
        cam_thresholded = (cam_thresholded - cam_thresholded.min()) / \
                          (cam_thresholded.max() - cam_thresholded.min() + 1e-8)
    
    # Create heatmap
    heatmap = np.uint8(cam_thresholded * 255)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Variable alpha based on activation intensity
    # Higher activations get more visibility
    alpha_map = cam_thresholded * alpha + (1 - alpha) * 0.2
    alpha_map = np.expand_dims(alpha_map, axis=2)
    
    # Blend with variable alpha
    overlay = original_img * (1 - alpha_map) + heatmap_color * alpha_map
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay


