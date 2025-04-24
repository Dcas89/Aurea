import os
import torch
from torch import nn
import transformers
from transformers import GPT2Tokenizer
from modeling_aurea import AureaPhi4Config, AureaPhi4ForCausalLM, VProjector, MultimodalBlock
from sr_block import SpatialRangeBlock
from d_encoder import Dinov2VisionEncoder, DINOv2Preprocessor
from s_encoder import Siglip2VisionEncoder, SigLip2Preprocessor
from generate import text_generation
import json
from safetensors.torch import load_file
import itertools
from dl_utils import download_ckpts
import gc

transformers.logging.set_verbosity_error()

class AureaPhi4(nn.Module):
    def __init__(
        self, 
        base_model_path, 
        device, 
        dtype, 
        use_cuda_kernel: bool = True,
        use_cache: bool = True,
        use_checkpointing: bool = False
        ):
        super().__init__()
        self.base_model_path = base_model_path
        self.device = device
        self.dtype = dtype
        self.use_cuda_kernel = use_cuda_kernel
        self.use_cache = use_cache
        self.use_checkpointing = use_checkpointing
        self.lm = self.load_model()

        self.lm = self.lm.to(device=self.device, dtype=self.dtype)
        self.text_emb = self.lm.get_input_embeddings()

    def load_model(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.base_model_path, 
            add_bos_token=False, 
            add_eos_token=False,
            add_prefix_space=False
        )
        self.config = AureaPhi4Config(
            use_cache=self.use_cache,
            use_checkpointing=self.use_checkpointing
        )
        aurea_phi = AureaPhi4ForCausalLM(self.config)

        lm = self.load_base_ckpt(
            model=aurea_phi
        )
        return lm
    
    def load_base_ckpt(self, model):
        index_file_name = f"{self.base_model_path}/model.safetensors.index.json"
        
        with open(index_file_name, "r") as f:
            index = json.load(f)
        
        if "weight_map" in index:
            index = index["weight_map"]
            checkpoint_files = sorted(list(set(index.values())))
            checkpoint_files = [os.path.join(self.base_model_path, f) for f in checkpoint_files]

        state_dict = {}

        for ckpt in checkpoint_files:
            print(f"Loading checkpoint file: {ckpt}")
            checkpoint = self.load_st(ckpt_file=ckpt)
            if checkpoint is not None:
                state_dict.update(checkpoint)

        if state_dict:
            model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("No state dictionary was loaded from checkpoint files.")
        
        return model
    
    def load_st(self, ckpt_file):
        return load_file(
            ckpt_file, 
            device=self.device.type
        )


class Aurea(nn.Module):
    def __init__(
        self,
        root_dir: str,
        dtype: torch.dtype = torch.bfloat16,
        use_cuda_kernel: bool = True,
        artifacts_folder: str = "model_artifacts"
    ):
        super().__init__()
        if root_dir is None:
            raise ValueError("Please provide a valid root directory.")
        download_ckpts(
            project_root=root_dir,
            artifacts_folder=artifacts_folder
        )
        self.model_dir = os.path.join(root_dir, artifacts_folder)
        self.dtype = dtype
        self.use_cuda_kernel = use_cuda_kernel

        self.aurea_phi = AureaPhi4(
            base_model_path=self.model_dir,
            device=torch.device('cpu'),
            dtype=dtype,
            use_cuda_kernel=use_cuda_kernel,
            use_cache=True,
            use_checkpointing=False,
        )

        self.user_token = "<|user|>"
        self.system_token = "<|system|>"
        self.assistant_token = "<|assistant|>"
        self.eos_token = "<|end|>"
        self.img_token = "<|image|>"
        self.system_prompt = (
            "You are AureaPhi4, a helpful vision-language assistant skilled in analyzing images and text "
            "for tasks such as detailed captioning, visual Q&A, spatial reasoning, object detection, "
            "classification, OCR, and advanced multimodal reasoning."
        )

        self.d_preprocessor = DINOv2Preprocessor()
        self.s_preprocessor = SigLip2Preprocessor()

        self.cross_sync()
        self.load_module_weights(self.model_dir)
        self.aurea_phi.lm.eval()

        self.device = torch.device('cuda' if use_cuda_kernel and torch.cuda.is_available() else 'cpu')
        self.to(dtype=self.dtype, device=self.device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        prompt: str,
        image_path: str = None,
        max_new_tokens: int = 50,
        temperature: float = 0.1,
        repetition_penalty: float = 1.1,
        filter_kwargs: dict = {'thres': 0.9, 'top_k': 50},
        use_dynamic_top_k: bool = False,
        min_top_k: int = 50,
        max_top_k: int = 90,
        filter_fn = None,
        exclude_prompt: bool = True,
    ) -> str:
        """
        Generate text from a prompt, optionally using an image.

        Parameters
        ----------
        prompt : str
            The user-provided text prompt.
        image_path : str, optional
            Local path to an image file for visual grounding. If omitted, runs
            text-only generation.
        max_new_tokens : int, default=50
            Maximum number of new tokens to generate.
        temperature : float, default=0.1
            Sampling temperature (lower = more deterministic).
        repetition_penalty : float, default=1.1
            Penalty >1.0 to discourage token repetition.
        filter_kwargs : dict, default={'thres': 0.9, 'top_k': 50}
            Arguments for the filtering function (threshold, top-k).
        use_dynamic_top_k : bool, default=False
            If True, interpolate top_k between `max_top_k` → `min_top_k` over
            the token steps.
        min_top_k, max_top_k : int, defaults=50, 90
            Bounds for dynamic top-k sampling.
        filter_fn : callable, optional
            Custom logit-filtering function. If None, uses default top-p/top-k.
        exclude_prompt : bool, default=True
            If True, strips the original prompt from the returned text.

        Returns
        -------
        str
            The decoded generated text.
        """
        if image_path:
            message = (
                f"{self.system_token}{self.system_prompt}{self.eos_token}"
                f"{self.user_token}{self.img_token}{prompt}{self.eos_token}{self.assistant_token}"
            )
        else:
            message = f"{self.user_token}{prompt}{self.eos_token}{self.assistant_token}"

        inputs = self.aurea_phi.tokenizer(message, return_tensors='pt')['input_ids'].to(self.device)

        d_feat = s_feat = None
        if image_path:
            img_full = os.path.expanduser(image_path)
            d_feat = self.d_preprocessor(img_full).to(self.device)
            s_feat = self.s_preprocessor(img_full).to(self.device)

        with torch.no_grad():
            output_ids = text_generation(
                model=self.aurea_phi.lm,
                input_ids=inputs,
                d_features=d_feat,
                s_features=s_feat,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                filter_kwargs=filter_kwargs,
                use_dynamic_top_k=use_dynamic_top_k,
                min_top_k=min_top_k,
                max_top_k=max_top_k,
                filter_fn=filter_fn,
                eos_token=200020,
                exclude_prompt=exclude_prompt,
            )

        result = self.aurea_phi.tokenizer.decode(output_ids[0])
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    
    def _make_prefix_map(self, root: nn.Module) -> dict[nn.Module, str]:
        return {mod: f"{name}." for name, mod in root.named_modules()}
    
    def _load_slice(self, module: nn.Module,
                          prefix: str,
                          state_dict: dict):
        sub_sd = {k[len(prefix):]: v
                  for k, v in state_dict.items()
                  if k.startswith(prefix)}

        assert sub_sd, f"{prefix}: no tensors found in the checkpoint"
        module.load_state_dict(sub_sd, strict=True)
    
    def load_module_weights(self, model_dir: str):
        ckpt = os.path.join(model_dir, "aureamodel.safetensors")
        full_sd = load_file(ckpt, device="cpu")

        srb = self.aurea_phi.lm.sr_block

        modules = [
            srb.s_proj, srb.s_norm, srb.d_norm, srb.proj_out,
            *itertools.chain.from_iterable(srb.sr_attn1_layers),
            *itertools.chain.from_iterable(srb.sr_attn2_layers),
            *(layer for layer in self.aurea_phi.lm.model.layers
              if isinstance(layer, MultimodalBlock)),
            self.aurea_phi.lm.v_proj,
        ]

        prefix_map = self._make_prefix_map(self)
        for m in modules:
            self._load_slice(m, prefix_map[m], full_sd)

    def to(self, *args, **kwargs):
        desired_dtype = kwargs.pop('dtype', None)
        with torch.no_grad():
            super().to(*args, **kwargs)
            if desired_dtype is not None:
                p_list = []
                if hasattr(self.aurea_phi.lm, 'sr_block'):
                    if hasattr(self.aurea_phi.lm.sr_block, 'd_encoder'):
                        p_list.extend(list(self.aurea_phi.lm.sr_block.d_encoder.parameters()))
                    if hasattr(self.aurea_phi.lm.sr_block, 's_encoder'):
                        p_list.extend(list(self.aurea_phi.lm.sr_block.s_encoder.parameters()))
                    if hasattr(self.aurea_phi.lm.sr_block, 'sr_attn1_layers'):
                        for sr_attn in self.aurea_phi.lm.sr_block.sr_attn1_layers:
                            for module in sr_attn:
                                if hasattr(module, 'sigma_spatial'):
                                    p_list.append(module.sigma_spatial)
                    if hasattr(self.aurea_phi.lm.sr_block, 'sr_attn2_layers'):
                        for sr_attn in self.aurea_phi.lm.sr_block.sr_attn2_layers:
                            for module in sr_attn:
                                if hasattr(module, 'sigma_spatial'):
                                    p_list.append(module.sigma_spatial)
                for param in p_list:
                    param.data = param.data.to(torch.float32)
                for param in self.parameters():
                    if not any(p.data_ptr() == param.data_ptr() for p in p_list):
                        param.data = param.data.to(desired_dtype)
        
            dtype_mappings = {
                'rotary_emb': {
                    'inv_freq': torch.float32,
                    'scaling_factor': torch.float32
                }
            }

            rotary_emb = getattr(self.aurea_phi.lm.model, 'rotary_emb', None)
            if rotary_emb is not None:
                self._update_buffer_dtypes(rotary_emb, dtype_mappings['rotary_emb'])
        
        return self

    def _update_buffer_dtypes(self, module, dtype_map):
        for buffer_name, dtype in dtype_map.items():
            buffer = getattr(module, buffer_name, None)
            if buffer is not None:
                buffer.data = buffer.data.to(dtype)
            else:
                print(f"Buffer '{buffer_name}' not found in module '{module}'.")

    def create_set_sr_block(self):
        dinov2 = Dinov2VisionEncoder(
            model_dir = self.model_dir
        )

        siglip2 = Siglip2VisionEncoder(
            model_dir = self.model_dir
        )

        sr_block = SpatialRangeBlock(
            s_dim = 1152,
            d_dim = 1536,
            patch_size = 16,
            radius = 3,
            use_cuda_kernel = self.use_cuda_kernel,
            d_encoder = dinov2,
            s_encoder = siglip2,
            train_block = False,
            depth = 4,
            fcl_module = None,
            init_weights = False
        )

        self.aurea_phi.lm.set_sr_block(sr_block)
        return
    
    def create_set_mm_blocks(self):
        mm_blocks = nn.ModuleList()
        blocks = list(self.aurea_phi.lm.model.layers)

        mm_blocks.append(
            MultimodalBlock(config=self.aurea_phi.config, layer_idx=0)
        )
        
        i = 0
        position = 1
        
        while i < len(blocks):
            original_block = blocks[i]
            original_block.layer_idx = position
            mm_blocks.append(original_block)
            position += 1
            i += 1
            
            if i < len(blocks):
                original_block = blocks[i]
                original_block.layer_idx = position
                mm_blocks.append(original_block)
                position += 1
                i += 1
                
            if i < len(blocks):
                mm_blocks.append(
                    MultimodalBlock(config=self.aurea_phi.config, layer_idx=position)
                )
                position += 1
        
        mm_blocks.append(
            MultimodalBlock(config=self.aurea_phi.config, layer_idx=position)
        )
        
        self.aurea_phi.lm.model.layers = mm_blocks
        return
    
    def create_set_v_proj(self):
        v_proj = VProjector(
            input_dim = 3072,
            output_dim = 3072,
            mult = 4,
            activation = nn.SiLU,
            init_weights = False
        )

        self.aurea_phi.lm.set_v_projector(v_proj)
        return

    def cross_sync(self):
        self.create_set_sr_block()
        self.create_set_mm_blocks()
        self.create_set_v_proj()
        return
