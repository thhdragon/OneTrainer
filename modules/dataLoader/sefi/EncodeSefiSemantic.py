from contextlib import nullcontext
import einops
import torch
import torch.nn.functional as F

from mgds.MGDS import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class EncodeSefiSemantic(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            in_name: str,
            out_name: str,
            dino_encoder,
            semvae,
            autocast_contexts: list = None,
            dtype: torch.dtype = None,
    ):
        super().__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.dino_encoder = dino_encoder
        self.semvae = semvae
        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.in_name, index) # Shape [3, H, W]

        # The image is in float [0, 1] range.
        # Ensure it has a batch dimension: [1, 3, H, W]
        img_batched = image.unsqueeze(0)
        B, C, H, W = img_batched.shape

        # Patch size is 256x256
        p1, p2 = max(1, H // 256), max(1, W // 256)

        # Pad or interpolate to multiple of 256 if not already
        target_h, target_w = p1 * 256, p2 * 256
        if H != target_h or W != target_w:
            img_batched = F.interpolate(img_batched, size=(target_h, target_w), mode='bicubic')

        # Rearrange to patches of 256x256: [B, C, H, W] -> [B*p1*p2, C, 256, 256]
        x_patches = einops.rearrange(img_batched, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', p1=p1, p2=p2)
        
        # Resize to 224x224 as required by DINOv2
        x_patches_224 = F.interpolate(x_patches, size=(224, 224), mode='bicubic')

        # DINOv2 normalization mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        device = next(self.dino_encoder.parameters()).device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        x_normalized = (x_patches_224.to(device=device, dtype=mean.dtype) - mean) / std

        # Extract features and encode via SemVAE
        with self._all_contexts(self.autocast_contexts):
            with torch.no_grad():
                features = self.dino_encoder.forward_features(x_normalized)
                patch_tokens = features['x_norm_patchtokens'] # [B_patches, 256, 1024]
                
                # SemVAE encoding
                z_sem_patches, _ = self.semvae.encode(patch_tokens.to(dtype=next(self.semvae.parameters()).dtype))
                # z_sem_patches is [B_patches, 256, 16]
                
                # Rearrange patches: [B_patches, 256, 16] -> [B_patches, 16, 16, 16]
                z_sem_patches = einops.rearrange(z_sem_patches, 'b (h w) d -> b d h w', h=16, w=16)
                
                # Rearrange back to full grid: [B_patches, 16, 16, 16] -> [B, 16, H//16, W//16]
                z_sem = einops.rearrange(z_sem_patches, '(b p1 p2) d h w -> b d (p1 h) (p2 w)', b=B, p1=p1, p2=p2)
                
                # Squeeze to remove batch dimension -> [16, H//16, W//16]
                z_sem = z_sem.squeeze(0).to(device='cpu')

        return {
            self.out_name: z_sem
        }
