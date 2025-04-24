#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__device__ inline int get_4d_idx(int b, int c, int h, int w, int C, int H, int W) {
    return ((b * C + c) * H + h) * W + w;
}

__device__ inline int get_cache_idx(int i, int b, int h, int w, int diameter_sq, int H, int W) {
    const int batch_offset = b * diameter_sq;
    return ((batch_offset + i) * H + h) * W + w;
}

__device__ inline float compute_spatial_weight(
    int idx,
    int diameter,
    float sigma_spatial
) {
    int dy = idx / diameter;
    int dx = idx % diameter;
    
    float fy = -1.f + 2.f * (float)dy / (float)(diameter - 1);
    float fx = -1.f + 2.f * (float)dx / (float)(diameter - 1);
    
    float r2 = fx * fx + fy * fy;
    return expf(-r2 / (2.f * sigma_spatial * sigma_spatial));
}

__device__ inline int reflect_coord(int x, int limit) {
    while (x < 0 || x >= limit) {
        if (x < 0) {
            x = -x;
        } else {
            x = 2 * limit - 2 - x;
        }
    }
    return x;
}

__global__ void sp_attn_fwd_kernel(
    const float* __restrict__ proj_feats,
    const float* __restrict__ spatial_feats,
    float* __restrict__ attention_out,
    float* __restrict__ spatial_weights_out,
    float* __restrict__ neighborhoods_out,
    const int B, const int C, const int H, const int W,
    const float sigma_spatial,
    const int radius,
    const int BLOCK_W,
    const int BLOCK_H,
    const int CHUNK
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int b = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int out_x = bx * BLOCK_W + tx;
    const int out_y = by * BLOCK_H + ty;

    if (b >= B || out_y >= H || out_x >= W) return;

    const int diameter = 2 * radius + 1;
    const int diameter_sq = diameter * diameter;
    const float scale = sqrtf((float)C);

    const int tile_w = BLOCK_W + 2 * radius;
    const int tile_h = BLOCK_H + 2 * radius;

    extern __shared__ float shared_mem[];
    float* sProj = shared_mem;
    float* sSpatial = shared_mem + CHUNK * tile_h * tile_w;

    auto smem_idx = [&](int c, int y, int x) {
        return y * tile_w + x + c * (tile_w * tile_h);
    };

    float local_dots[169];
    for (int p = 0; p < diameter_sq; p++) {
        local_dots[p] = 0.0f;

        float w_spatial = compute_spatial_weight(p, diameter, sigma_spatial);
        spatial_weights_out[get_cache_idx(p, b, out_y, out_x, diameter_sq, H, W)] = w_spatial;
    }

    for (int cStart = 0; cStart < C; cStart += CHUNK) {
        const int cEnd = min(cStart + CHUNK, C);
        const int chunkSize = cEnd - cStart;

        for (int load_y = ty; load_y < tile_h; load_y += blockDim.y) {
            for (int load_x = tx; load_x < tile_w; load_x += blockDim.x) {

                const int gx = reflect_coord(bx * BLOCK_W + (load_x - radius), W);
                const int gy = reflect_coord(by * BLOCK_H + (load_y - radius), H);

                for (int cc = 0; cc < chunkSize; cc++) {
                    const int c = cStart + cc;
                    const int gidx = get_4d_idx(b, c, gy, gx, C, H, W);
                    const int sidx = smem_idx(cc, load_y, load_x);
                    
                    sProj[sidx] = proj_feats[gidx];
                    sSpatial[sidx] = spatial_feats[gidx];
                }
            }
        }

        __syncthreads();

        const int center_y = ty + radius;
        const int center_x = tx + radius;

        for (int cc = 0; cc < chunkSize; cc++) {
            const float center_val = sProj[smem_idx(cc, center_y, center_x)];
            const int c = cStart + cc;

            for (int p = 0; p < diameter_sq; p++) {
                const int dy = p / diameter - radius;
                const int dx = p % diameter - radius;
                
                const int ny = center_y + dy;
                const int nx = center_x + dx;

                const float neigh_val = sProj[smem_idx(cc, ny, nx)];
                local_dots[p] += center_val * neigh_val;

                neighborhoods_out[
                    (((b * C + c) * diameter_sq + p) * H + out_y) * W + out_x
                ] = sSpatial[smem_idx(cc, ny, nx)];
            }
        }

        __syncthreads();
    }

    for (int p = 0; p < diameter_sq; p++) {
        local_dots[p] /= scale;
    }

    float max_val = local_dots[0];
    for (int p = 1; p < diameter_sq; p++) {
        max_val = fmaxf(max_val, local_dots[p]);
    }

    float sum_exp = 0.0f;
    for (int p = 0; p < diameter_sq; p++) {
        local_dots[p] = expf(local_dots[p] - max_val);
        sum_exp += local_dots[p];
    }
    sum_exp = fmaxf(sum_exp, 1e-12f);

    for (int p = 0; p < diameter_sq; p++) {
        const float range_weight = local_dots[p] / sum_exp;
        const float spatial_weight = spatial_weights_out[
            get_cache_idx(p, b, out_y, out_x, diameter_sq, H, W)
        ];
        const float combined_weight = range_weight * spatial_weight;
        attention_out[get_cache_idx(p, b, out_y, out_x, diameter_sq, H, W)] = combined_weight;
    }
}

__device__ inline float spatial_derivative(
    int idx,
    int diameter,
    float sigma_spatial
) {
    int dy = idx / diameter;
    int dx = idx % diameter;

    float scale = 2.f / (float)(diameter - 1);
    float fy = -1.f + (float)dy * scale;
    float fx = -1.f + (float)dx * scale;

    float r2 = fmaxf(fx * fx + fy * fy, 1e-12f);
    float sigma_sq = sigma_spatial * sigma_spatial;

    float w = expf(-r2 / (2.f * sigma_sq));

    float deriv = (r2 / (sigma_spatial * sigma_sq)) * w;

    deriv = fminf(deriv, 1e3f);
    return fmaxf(deriv, 1e-12f);
}

__global__ void sp_attn_bwd_kernel(
    float* __restrict__ grad_proj_feats,
    float* __restrict__ grad_spatial_feats,
    float* __restrict__ grad_sigma,

    const float* __restrict__ grad_attn,
    const float* __restrict__ grad_neighborhoods,
    const float* __restrict__ attention_weights,
    const float* __restrict__ spatial_weights_in,
    const float* __restrict__ proj_feats,
    const float* __restrict__ spatial_feats,

    const int B, const int C,
    const int H, const int W,
    const float sigma_spatial,
    const int radius
) {
    extern __shared__ float shared_mem[];
    float* shared_grad_sigma = shared_mem;

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int threads_per_block = blockDim.x * blockDim.y;

    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < threads_per_block) {
        shared_grad_sigma[tid] = 0.0f;
    }
    __syncthreads();

    if (b >= B || h >= H || w >= W) {
        return;
    }

    const int diameter = 2 * radius + 1;
    const int diameter_sq = diameter * diameter;
    const float scale = sqrtf((float)C);

    float thread_grad_sigma = 0.0f;

    float spatial_weights[169];
    float range_kernel[169];

    float weighted_sum_total = 0.0f;
    for (int p = 0; p < diameter_sq; p++) {
        const size_t idx = get_cache_idx(p, b, h, w, diameter_sq, H, W);
        float sw = spatial_weights_in[idx];
        float aw = attention_weights[idx];
        spatial_weights[p] = sw;
        range_kernel[p] = aw / (sw + 1e-12f);

        float g = grad_attn[idx];
        weighted_sum_total += (g * aw);
    }

    for (int p = 0; p < diameter_sq; p++) {
        int dy = (p / diameter) - radius;
        int dx = (p % diameter) - radius;
    
        int nh = reflect_coord(h + dy, H);
        int nw = reflect_coord(w + dx, W);
    
        const float grad_attn_p = grad_attn[
            get_cache_idx(p, b, h, w, diameter_sq, H, W)
        ];
        const float range_val = range_kernel[p];
        const float spatial_weight = spatial_weights[p];
    
        float grad_scale = range_val * (grad_attn_p * spatial_weight
                                        - weighted_sum_total) / scale;
    
        for (int c = 0; c < C; c++) {
            float center_val  = proj_feats[get_4d_idx(b, c, h,  w,  C, H, W)];
            float neighbor_val = proj_feats[get_4d_idx(b, c, nh, nw, C, H, W)];
    
            float grad_center  = grad_scale * neighbor_val;
            float grad_neighbor = grad_scale * center_val;
    
            atomicAdd(&grad_proj_feats[get_4d_idx(b, c, h,  w,  C, H, W)], grad_center);
            atomicAdd(&grad_proj_feats[get_4d_idx(b, c, nh, nw, C, H, W)], grad_neighbor);
        }
    
        float d_spatial = spatial_derivative(p, diameter, sigma_spatial);
        thread_grad_sigma += grad_attn_p * range_val * d_spatial;
    
        for (int c = 0; c < C; c++) {
            float grad_n = grad_neighborhoods[
                ((((b * C) + c) * diameter_sq + p) * H + h) * W + w
            ];
            atomicAdd(&grad_spatial_feats[get_4d_idx(b, c, nh, nw, C, H, W)], grad_n);
        }
    }

    shared_grad_sigma[tid] = thread_grad_sigma;
    __syncthreads();

    for (unsigned int s = threads_per_block / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_grad_sigma[tid] += shared_grad_sigma[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(grad_sigma, shared_grad_sigma[0]);
    }
}

std::vector<torch::Tensor> sp_attn_fwd_cuda(
    const torch::Tensor& proj_feats,
    const torch::Tensor& spatial_feats,
    float sigma_spatial,
    int radius
) {
    const auto B = proj_feats.size(0);
    const auto C = proj_feats.size(1);
    const auto H = proj_feats.size(2);
    const auto W = proj_feats.size(3);

    const int BLOCK_W = (W <= 16) ? 16 : 32;
    const int BLOCK_H = (W <= 16) ? 8 : 4;
    
    const int diameter = 2 * radius + 1;
    const int diameter_sq = diameter * diameter;

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    const int maxSharedMem = props.sharedMemPerBlock;

    const int tile_w = BLOCK_W + 2 * radius;
    const int tile_h = BLOCK_H + 2 * radius;
    const int elements_per_map = tile_w * tile_h;
    const int MIN_CHUNK = (W <= 16) ? 4 : 8;
    const int bytes_per_element = 2 * elements_per_map * sizeof(float);
    const int max_chunk = maxSharedMem / bytes_per_element;
    const int aligned_chunk = (max_chunk / 32) * 32;
    const int CHUNK = std::max(MIN_CHUNK, (aligned_chunk > 0 ? aligned_chunk : MIN_CHUNK));
    
    auto attention_out = torch::empty(
        {B, diameter_sq, H, W},
        proj_feats.options()
    );
    
    auto spatial_weights_out = torch::empty(
        {B, diameter_sq, H, W},
        proj_feats.options()
    );
    
    auto neighborhoods = torch::empty(
        {B, C, diameter_sq, H, W},
        spatial_feats.options()
    );

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid(
        (W + BLOCK_W - 1) / BLOCK_W,
        (H + BLOCK_H - 1) / BLOCK_H,
        B
    );

    const size_t smem_size = 2 * CHUNK * elements_per_map * sizeof(float);
    
    sp_attn_fwd_kernel<<<grid, block, smem_size>>>(
        proj_feats.data_ptr<float>(),
        spatial_feats.data_ptr<float>(),
        attention_out.data_ptr<float>(),
        spatial_weights_out.data_ptr<float>(),
        neighborhoods.data_ptr<float>(),
        B, C, H, W,
        sigma_spatial,
        radius,
        BLOCK_W,
        BLOCK_H,
        CHUNK
    );

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    
    return {attention_out, spatial_weights_out, neighborhoods};
}

std::vector<torch::Tensor> sp_attn_bwd_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& grad_neighborhoods,
    const torch::Tensor& attention_weights,
    const torch::Tensor& spatial_weights,
    const torch::Tensor& proj_feats,
    const torch::Tensor& spatial_feats,
    float sigma_spatial,
    int radius
) {
    const auto B = proj_feats.size(0);
    const auto C = proj_feats.size(1);
    const auto H = proj_feats.size(2);
    const auto W = proj_feats.size(3);

    auto grad_proj = torch::zeros_like(proj_feats);
    auto grad_spatial = torch::zeros_like(spatial_feats);
    auto grad_sigma = torch::zeros({1}, proj_feats.options());

    const dim3 threads(16, 16);
    const dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y,
        B
    );
    const int threads_per_block = threads.x * threads.y;
    const int shared_mem_size = threads_per_block * sizeof(float);

    sp_attn_bwd_kernel<<<blocks, threads, shared_mem_size>>>(
        grad_proj.data_ptr<float>(),
        grad_spatial.data_ptr<float>(),
        grad_sigma.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_neighborhoods.data_ptr<float>(),
        attention_weights.data_ptr<float>(),
        spatial_weights.data_ptr<float>(),
        proj_feats.data_ptr<float>(),
        spatial_feats.data_ptr<float>(),
        B, C, H, W,
        sigma_spatial,
        radius
    );

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    return {grad_proj, grad_spatial, grad_sigma};
}