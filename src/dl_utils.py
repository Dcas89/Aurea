import os
from huggingface_hub import snapshot_download

def download_ckpts(
        project_root: str,
        repo_id: str = "Dcas89/Aurea",
        revision: str = "main",
        artifacts_folder: str = "model_artifacts"
    ):

    file_list = [
        "added_tokens.json",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "aureamodel.safetensors",
        "dinov2.safetensors",
        "siglip2.safetensors",
    ]

    artifacts_dir = os.path.join(project_root, artifacts_folder)
    os.makedirs(artifacts_dir, exist_ok=True)

    missing = [fn for fn in file_list if not os.path.isfile(os.path.join(artifacts_dir, fn))]
    if missing:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=artifacts_dir,
            repo_type="model",
            local_dir_use_symlinks=False,
            allow_patterns=missing,
        )
        
        still_missing = [fn for fn in file_list if not os.path.isfile(os.path.join(artifacts_dir, fn))]
        if still_missing:
            raise FileNotFoundError(f"Failed to download: {still_missing}")