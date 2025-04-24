import os
import torch
from torch import nn
from typing import Optional
from modeling_dinov2 import Dinov2WithRegistersConfig, Dinov2WithRegistersModel, TransformerOutput
from safetensors.torch import load_file
from torchvision.transforms import v2 as T
from PIL import Image
from utils import AlphaComposite


class Dinov2VisionEncoder(nn.Module):
    def __init__(self, model_dir, attn_implementation="sdpa"):
        super().__init__()
        config = Dinov2WithRegistersConfig()
        config._attn_implementation = attn_implementation
        self.model = Dinov2WithRegistersModel(config)
        ckpt_file = os.path.join(model_dir, "dinov2.safetensors")
        state_dict = load_file(ckpt_file, device='cpu')
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
    
    @torch.no_grad()
    def forward(
        self, 
        pixel_values: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        attn_bias: Optional[torch.Tensor] = None
    ) -> TransformerOutput:

        return self.model(
            pixel_values=pixel_values,
            masks=masks,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            attn_bias=attn_bias
        )


class DINOv2Preprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = T.Compose([
            T.PILToTensor(),
            AlphaComposite(),
            T.Resize(
                    256, 
                    interpolation=T.InterpolationMode.BICUBIC, 
                    antialias=True
                ),
            T.CenterCrop(224),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def forward(self, image_or_path):
        if isinstance(image_or_path, str):
            image_or_path = Image.open(image_or_path)

        if image_or_path.mode not in ["RGB", "RGBA"]:
            image_or_path = image_or_path.convert("RGB")

        t = self.transform(image_or_path)

        return t.unsqueeze(0)
