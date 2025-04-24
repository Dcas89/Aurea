import os
import torch
from torch import nn
from typing import Optional
from modeling_siglip2 import Siglip2VisionConfig, Siglip2VisionModel, BaseModelOutput
from safetensors.torch import load_file
from torchvision.transforms import v2 as T
from PIL import Image
from utils import AlphaComposite


class Siglip2VisionEncoder(nn.Module):
    def __init__(self, model_dir, attn_implementation="sdpa"):
        super().__init__()
        config = Siglip2VisionConfig()
        config._attn_implementation = attn_implementation
        self.model = Siglip2VisionModel(config)
        ckpt_file = os.path.join(model_dir, "siglip2.safetensors")
        state_dict = load_file(ckpt_file, device='cpu')
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.no_grad()
    def forward(
        self, 
        pixel_values: torch.Tensor,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        interpolate_pos_encoding: bool = False
    ) -> BaseModelOutput:
        
        return self.model(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding
        )


class SigLip2Preprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = T.Compose([
            T.PILToTensor(),
            AlphaComposite(),
            T.Resize(
                    (224, 224), 
                    interpolation=T.InterpolationMode.BILINEAR, 
                    antialias=True
                ),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def forward(self, image_or_path):
        if isinstance(image_or_path, str):
            image_or_path = Image.open(image_or_path)

        if image_or_path.mode not in ["RGB", "RGBA"]:
            image_or_path = image_or_path.convert("RGB")

        t = self.transform(image_or_path)

        return t.unsqueeze(0)