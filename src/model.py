import torch
import math
from gsplat import rasterization
from torch import nn

class MotionPolicyNetwork(nn.Module):
    """
    The 'Brain': Converts Audio Features -> Gaussian Deformations.
    Simple MLP for demonstration (in prod: Transformer).
    """
    def __init__(self, input_dim=1024, hidden_dim=256, num_gaussians=100000):
        super().__init__()
        # Predicting Delta Position (XYZ) and Delta Rotation (Quaternion)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_gaussians * 7) # 3 for pos, 4 for rot
        )
        self.num_gaussians = num_gaussians

    def forward(self, audio_features):
        # audio_features: [B, 1024]
        deltas = self.net(audio_features)
        deltas = deltas.view(-1, self.num_gaussians, 7)
        return deltas[..., :3], deltas[..., 3:]

class AvatarEngine:
    def __init__(self, model_path="checkpoints/avatar_canonical.ply", device="cuda"):
        self.device = device
        # Load Canonical Gaussians (Means, Scales, Quats, Opacities, Colors)
        # Simplified loader for demo purposes
        self.means = torch.rand((100000, 3), device=device, dtype=torch.float32)
        self.scales = torch.rand((100000, 3), device=device, dtype=torch.float32)
        self.quats = torch.rand((100000, 4), device=device, dtype=torch.float32)
        self.opacities = torch.ones((100000, 1), device=device, dtype=torch.float32)
        self.colors = torch.rand((100000, 3), device=device, dtype=torch.float32)
        
        # Load Motion Policy
        self.policy = MotionPolicyNetwork().to(device)
        self.policy.eval()

        # Camera Intrinsics (Static for webcam view)
        self.view_matrix = torch.eye(4, device=device).unsqueeze(0) # [1, 4, 4]
        self.K = torch.tensor([[800, 0, 256], [0, 800, 256], [0, 0, 1]], device=device).unsqueeze(0)
    
    def render_frame(self, audio_features):
        """
        Takes audio features, deforms gaussians, and rasterizes.
        """
        with torch.no_grad():
            # 1. Predict Deformations
            d_pos, d_rot = self.policy(audio_features)
            
            # 2. Apply Deltas (Deformation Field)
            # In production: Use quaternion multiplication for rotation
            cur_means = self.means + d_pos.squeeze(0)
            cur_quats = self.quats + d_rot.squeeze(0)

            # 3. Rasterize using gsplat (CUDA accelerated)
            # Returns: [1, H, W, 3]
            renders, _, _ = rasterization(
                means=cur_means,
                quats=cur_quats,
                scales=self.scales,
                opacities=self.opacities,
                colors=self.colors,
                viewmats=self.view_matrix, # View Matrix
                Ks=self.K,                 # Intrinsics
                width=512,
                height=512
            )
            
            return renders[0] # Return the first batch item