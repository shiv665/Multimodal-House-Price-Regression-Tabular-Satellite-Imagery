import torch
import numpy as np
import cv2
from src.config import cfg

class GradCAM:
    def __init__(self, model, target_layer="layer4"):
        self.model = model
        self.target_layer = dict(self.model.cnn.named_children())[target_layer]
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _forward_hook(self, module, input, output):
        self.activations = output

    def __call__(self, img_tensor, tab_tensor):
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0).to(cfg.device)
        tab_tensor = tab_tensor.unsqueeze(0).to(cfg.device)
        out = self.model(img_tensor, tab_tensor)
        out.backward()
        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        # Resize using OpenCV instead of PIL
        cam_resized = cv2.resize(cam, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        return cam_resized