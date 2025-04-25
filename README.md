# Aurea: Adaptive Multimodal Fusion for Vision-Language Models

Aurea is an open-source research project aimed at advancing vision-language model (VLM) pretraining by leveraging cutting-edge vision encoders—DINOv2 and SigLIP2. The core of Aurea is a novel adaptive **spatial-range attention mechanism** that intelligently fuses spatial and semantic information from encoder-derived visual features, enabling richer and more context-aware representations for various downstream tasks.

## Key Features

- **Multiple Vision Encoders:** Input images are encoded separately by DINOv2 and SigLIP2.

- **Multi-stage Fusion:** The `SpatialRangeBlock` fuses these inputs through multiple layers of `SpatialRangeAttention`, which selectively aggregates features by jointly considering spatial proximity and semantic similarity. This is performed with a highly optimized fused CUDA kernel.

- **Flexible Language Model Integration:** While Phi-4 is the default language model, Aurea is designed for easy adaptation to other pretrained language models with minimal engineering effort.

- **Model Weights:** Two model checkpoints are provided: (1) base pretrained weights (trained on a ~558k image subset of LAION) and (2) instruction-tuned weights (further fine-tuned on ~625k samples from LLaVA 1.5 datasets). All checkpoints can be downloaded directly from the Hugging Face repository: [HF REPO](https://huggingface.co/Dcas89/Aurea)

- **Extensible and Modular:** The code supports straightforward extension, experimentation, and integration with novel encoders or downstream tasks.

### How It Works

<p align="center">
  <img src="./assets/sr_attn_demo.gif" alt="Spatial-Range Attention Demo" width="50%">
</p>

This visualization demonstrates how the spatial-range attention module operates over a feature map sampled every 16 × 16 pixels. For each query location, the module attends to a 7 × 7 neighborhood (radius = 3).

- The pale-grey square indicates the receptive field window centered on the query position.
- The orange patch marks the query (center pixel).
- Neighboring patches within the receptive field are tinted red, with intensity proportional to their learned attention weights, computed as:

$$\text{combined\\_kernel} = \text{softmax}(\text{range similarity}) \cdot \text{gaussian}(\text{spatial distance}) + \text{residual fix-up}$$

In essence, this mechanism dynamically integrates spatial proximity and feature similarity to determine which pixels influence each query position, producing context-aware and semantically meaningful feature aggregation.

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/Dcas89/Aurea.git
cd Aurea
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

## Usage

First, initialize the Aurea model:

```python
from entry import Aurea

aurea = Aurea(root_dir='/path/to/Aurea')
```

> **Note:** When initializing the model, all required model checkpoints will be downloaded automatically.

### Image + Text Generation (Basic)

Generate text based on an image and prompt:

```python
# Basic image + text generation
response = aurea.generate(
    prompt="How many remote control devices are in this image?", 
    image_path='./assets/cats.png'  # Example image included in the repo
)
print(response)
```

### Generation with Custom Parameters

Tune generation parameters for more control:

```python
# Advanced generation with custom parameters
response = aurea.generate(
    prompt="Only one cat is wearing a collar in the image. Which cat is it? Answer Briefly: Left, Right, or Both", 
    image_path='./assets/cats.png',  # Example image included in the repo
    max_new_tokens=50,          # Maximum number of tokens to generate
    temperature=0.1,            # Lower values make output more deterministic
    repetition_penalty=1.1,     # Penalizes token repetition (>1.0)
    filter_kwargs={'thres': 0.90, 'top_k': 50},  # Parameters for filtering function
    use_dynamic_top_k=False,    # Whether to use dynamic top-k sampling
    min_top_k=50,               # Minimum top-k value if using dynamic top-k
    max_top_k=90,               # Maximum top-k value if using dynamic top-k
    filter_fn=None,             # Custom filtering function
    exclude_prompt=True         # Whether to exclude prompt from returned text
)
print(response)
```

### Logit Filtering

Using a specific filtering function (e.g., top_p):

```python
from generate import top_p

response = aurea.generate(
    prompt="Only one cat is wearing a collar in the image. What is the color of the collar? Answer Briefly: Blue, Light Green, Yellow", 
    image_path='./assets/cats.png',  # Example image included in the repo
    max_new_tokens=50,
    temperature=0.1,
    repetition_penalty=1.1,
    filter_kwargs={'thres': 0.99, 'top_k': 50},
    filter_fn=top_p,            # Using top-p sampling
    exclude_prompt=True
)
print(response)
```

### Dynamic Top-K Sampling

Example using dynamic top-k sampling (interpolating from max_top_k to min_top_k over generation):

```python
response = aurea.generate(
    prompt="What does the logo say and what does it represent?", 
    image_path='./assets/mazure.png',
    max_new_tokens=100,
    temperature=0.1,
    repetition_penalty=1.1,
    filter_kwargs={'thres': 0.99, 'top_k': 50},
    use_dynamic_top_k=True,     # Enable dynamic top-k sampling
    min_top_k=50,               # Lower bound for top-k
    max_top_k=90,               # Upper bound for top-k
    filter_fn=None,
    exclude_prompt=True
)

print(response)
```

### Text-Only Generation

Aurea can also be used for text-only tasks:

```python
# Text-only generation (no image)
response = aurea.generate(
    prompt="What is CUDA programming?",
    max_new_tokens=200,
    temperature=0.1,
    repetition_penalty=1.1,
    filter_kwargs={'thres': 0.9, 'top_k': 50},
    exclude_prompt=True
)
print(response)
```

# Aurea: Technical Details

## 1. Spatial-Range Attention

### Preliminaries and Definitions

Let the input feature tensor be:

$$
X \in \mathbb{R}^{B \times C \times H \times W}
$$

where $B$ is batch size, $C$ the feature dimension, and $H, W$ spatial dimensions.

Define the neighborhood radius $r$ and diameter $d = 2r + 1$. The spatial neighborhood around $(h,w)$ is:

$$
\mathcal{N}(h, w) = \{(h + \Delta h, w + \Delta w) \mid \Delta h, \Delta w \in [-r, r]\}
$$

### Spatial Kernel

Define normalized spatial offsets for each neighborhood index $n \in \{1, \dots, d^2\}$:

$$
(\Delta x_n, \Delta y_n) \in [-1, 1] \times [-1, 1]
$$

The spatial Gaussian kernel is:

$$
K_{\text{spatial}}(n) = \exp\left(-\frac{\Delta x_n^2 + \Delta y_n^2}{2 \sigma_{\text{spatial}}^2}\right)
$$

where $\sigma_{\text{spatial}}$ is a learnable parameter controlling spatial focus.

### Range Kernel

Let $\text{proj}_x \in \mathbb{R}^{B \times C \times H \times W}$ be the projected semantic features.

Extract neighborhood patches $Q \in \mathbb{R}^{B \times C \times d^2 \times H \times W}$ from $\text{proj}_x$ via unfold.

Define similarity scores at each spatial location as scaled dot product between center vectors and their neighborhoods:

$$\mathrm{sim}\_{b,n,h,w} = \frac{\sum_{c=1}^{C} Q_{b,c,n,h,w} \cdot \text{proj}\_{x,b,c,h,w}}{\sqrt{C}}$$

Apply softmax over neighborhood dimension $n$:

$$K\_{\text{range},b,n,h,w} = \frac{\exp(\mathrm{sim}\_{b,n,h,w})}{\sum_{n'=1}^{d^2} \exp(\mathrm{sim}\_{b,n',h,w})}$$

### Combined Kernel

Combine spatial and range kernels element-wise:

$$
K_{b,n,h,w} = K_{\text{range}, b,n,h,w} \cdot K_{\text{spatial}}(n)
$$

### Aggregation and Output

Extract corresponding spatial feature neighborhoods $N \in \mathbb{R}^{B \times C \times d^2 \times H \times W}$ from input features.

Aggregate weighted neighborhood features:

$$
X_{\text{aggregated}, b,c,h,w} = \sum_{n=1}^{d^2} K_{b,n,h,w} \cdot N_{b,c,n,h,w}
$$

Apply an output projection and add a residual connection:

$$
X_{\text{final}} = X_{\text{residual}} + \text{OutputProj}(X_{\text{aggregated}})
$$

### 1.1 Auxiliary Training Losses

#### Feature Consistency Loss (FCL)

Feature Consistency Loss (FCL) is used to ensure that the fused output features preserve relevant structural and semantic information from both input modalities. As guidance, features used were extracted from pretrained vision encoders—DINOV2 and SigLIP2—denoted as $d_{\text{in}}$ and $s_{\text{in}}$.

Given the fused output features after the spatial-range attention layers, $F_{\text{fused}}$, the consistency loss is computed as:

$$\mathrm{FCL}\_{\text{total}} = \mathrm{FCL}(d\_{\text{in}}, F\_{\text{fused}}) + \mathrm{FCL}(s\_{\text{in}}, F\_{\text{fused}})$$

This dual guidance encourages the fused representation to maintain alignment with both semantic views, promoting cross-modal coherence.

#### Feature Diversity Loss (FDL)

Meanwhile, the Feature Diversity Loss (FDL) is applied solely on the fused features $F_{\text{fused}}$, encouraging the feature channels to capture diverse and complementary information by minimizing inter-channel correlation.

Together, these losses balance fidelity to input modalities (via FCL) and richness of learned representations (via FDL), helping the model learn robust, multi-source aware features.

## 2. Multimodal Integration

### 2.1 MultimodalBlock

The `MultimodalBlock` integrates visual features into the language model through:

1. **Cross-Attention**
2. **Feed-Forward Network**
3. **Normalization & Residual Connections**

## 3. Implementation Details

### 3.1 Pure PyTorch Implementation (`SpatialRangeAttention`)

- Written entirely with standard PyTorch operations (unfold, einsum, padding)
- Maximizes compatibility and ease of inspection or debugging
- Suitable for CPU-only or CUDA-disabled environments
- Has higher memory usage and lower speed due to intermediate tensors and multiple kernel launches

### 3.2 CUDA Kernel (`SpatialRangeAttentionCUDA`)

This CUDA kernel implements an efficient Spatial-Range Attention mechanism that significantly accelerates the naive PyTorch reference implementation by fusing multiple operations and carefully optimizing memory and compute patterns. 

The key optimizations are:

- **Operation Fusion**: Extracts local patches, computes spatial Gaussian weights, similarity-based range kernel (dot products and softmax), and forms combined attention weights within a single CUDA kernel. This eliminates multiple expensive kernel launches and large intermediate tensors common in PyTorch-based unfold/einsum implementations.

- **Shared Memory Tiling**: Input semantic and spatial feature maps are loaded into on-chip shared memory tiles, including halo regions for neighborhoods with reflection padding handled in kernel indexing. This drastically reduces global memory loads and improves data reuse during neighborhood similarity computations.

- **Spatial Gaussian Computation**: The spatial Gaussian kernel is computed dynamically per-thread for each neighborhood location using a lightweight formula.

- **Reflection Padding via Indexing**: Instead of creating padded input tensors, the kernel handles symmetric (reflect) padding implicitly by adjusting indices during data loads.

- **Chunked Feature Dimension Processing**: Feature dimension is processed in chunks to fit within shared memory limits. Dot product computations between the center pixel and its neighborhood in the semantic feature space are tiled and accumulated to maximize occupancy and arithmetic intensity.

- **Softmax and Attention Weight Computation**: The kernel performs numerically stable softmax over the range kernel similarity scores, then multiplies by spatial Gaussian weights to produce combined attention weights directly on device without intermediate tensor allocations.

- **Direct Neighborhood Feature Caching**: Alongside attention weights, the kernel caches corresponding spatial neighborhoods for subsequent weighted aggregation steps, avoiding costly unfold operations in PyTorch.

- **Backward Pass Support**: A matching and efficient CUDA backward kernel provides gradients w.r.t. input features and the spatial standard deviation parameter (sigma_spatial), leveraging atomic operations and shared memory reductions.

## 4. Performance Considerations

### 4.1 Mixed Precision and Memory Efficiency

- Core spatial-range attention in FP32 for numerical stability
- Other components use BF16 for efficiency
- The LM uses Group Query Attention and KV caching

## Future Development

While the current release includes both pretrained and instruction-tuned weights, there's significant potential in extending training further. However, as compute-intensive training is challenging for individual researchers, I welcome collaboration on:

- Extended pretraining, instruction tuning, or reinforcement learning with additional visual-language datasets
- Encoder and architectural variants/optimizations
- Evaluation on diverse benchmarks

If you're interested in collaborating or contributing compute resources to advance this research, please open an issue or reach out directly.

## References

- [SigLIP 2: Multilingual Vision-Language Encoders](https://doi.org/10.48550/arXiv.2502.14786)
- [Phi-4 Technical Report](https://doi.org/10.48550/arXiv.2412.08905)
- [DINOv2: Learning Robust Visual Features without Supervision](https://doi.org/10.48550/arXiv.2304.07193)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD)

## License

This project is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- The CUDA spatial-range attention is inspired by and adapted from LLaVA-UHD.
- Some components were adapted from [lucidrains](https://github.com/lucidrains) repositories, which provide excellent implementations of various transformer and attention mechanisms.
- Thanks to the open-source community for DINOv2, SigLIP2, LLaVA, LlaVA-UHD, and Phi-4.
- Thanks to Hugging Face for their [Transformers](https://github.com/huggingface/transformers) and [Accelerate](https://github.com/huggingface/accelerate) libraries.

This project incorporates code and models from:

- Phi-4 Mini: Copyright (c) 2025 Microsoft Corporation
- DINOv2: Copyright (c) 2024 Meta Platforms, Inc.
- SigLIP2: Copyright (c) 2025 Google LLC
