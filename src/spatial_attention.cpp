#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> sp_attn_fwd_cuda(
    const torch::Tensor& proj_feats,
    const torch::Tensor& spatial_feats,
    float sigma_spatial,
    int radius
);

std::vector<torch::Tensor> sp_attn_bwd_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& grad_neighborhoods,
    const torch::Tensor& attention_weights,
    const torch::Tensor& spatial_weights_in,
    const torch::Tensor& proj_feats,
    const torch::Tensor& spatial_feats,
    float sigma_spatial,
    int radius
);

std::vector<torch::Tensor> sp_attn_fwd(
    const torch::Tensor& proj_feats,
    const torch::Tensor& spatial_feats,
    float sigma_spatial,
    int radius
) {

    TORCH_CHECK(proj_feats.is_cuda(), "proj_feats must be a CUDA tensor");
    TORCH_CHECK(spatial_feats.is_cuda(), "spatial_feats must be a CUDA tensor");
    TORCH_CHECK(proj_feats.is_contiguous(), "proj_feats must be contiguous");
    TORCH_CHECK(spatial_feats.is_contiguous(), "spatial_feats must be contiguous");
    TORCH_CHECK(proj_feats.dim() == 4, "proj_feats must be a 4D tensor");
    TORCH_CHECK(spatial_feats.dim() == 4, "spatial_feats must be a 4D tensor");
    TORCH_CHECK(radius > 0, "radius must be positive");
    TORCH_CHECK((2 * radius + 1) * (2 * radius + 1) <= 169, "neighborhood size too large");
    TORCH_CHECK(sigma_spatial > 0.0f, "sigma_spatial must be positive");
    
    return sp_attn_fwd_cuda(proj_feats, spatial_feats, sigma_spatial, radius);
}

std::vector<torch::Tensor> sp_attn_bwd(
    const torch::Tensor& grad_output,
    const torch::Tensor& grad_neighborhoods,
    const torch::Tensor& attention_weights,
    const torch::Tensor& spatial_weights_in,
    const torch::Tensor& proj_feats,
    const torch::Tensor& spatial_feats,
    float sigma_spatial,
    int radius
) {

    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(grad_neighborhoods.is_cuda(), "grad_neighborhoods must be a CUDA tensor");
    TORCH_CHECK(attention_weights.is_cuda(), "attention_weights must be a CUDA tensor");
    TORCH_CHECK(spatial_weights_in.is_cuda(), "spatial_weights_in must be a CUDA tensor");
    TORCH_CHECK(proj_feats.is_cuda(), "proj_feats must be a CUDA tensor");
    TORCH_CHECK(spatial_feats.is_cuda(), "spatial_feats must be a CUDA tensor");
    
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_neighborhoods.is_contiguous(), "grad_neighborhoods must be contiguous");
    TORCH_CHECK(attention_weights.is_contiguous(), "attention_weights must be contiguous");
    TORCH_CHECK(spatial_weights_in.is_contiguous(), "spatial_weights_in must be contiguous");
    TORCH_CHECK(proj_feats.is_contiguous(), "proj_feats must be contiguous");
    TORCH_CHECK(spatial_feats.is_contiguous(), "spatial_feats must be contiguous");
    
    return sp_attn_bwd_cuda(
        grad_output,
        grad_neighborhoods,
        attention_weights,
        spatial_weights_in,
        proj_feats,
        spatial_feats,
        sigma_spatial,
        radius
    );
}