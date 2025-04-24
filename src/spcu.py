import os
import torch
from torch.utils.cpp_extension import load_inline


def compile_extension(ptx_fallback=False):

    current_dir = os.path.dirname(os.path.abspath(__file__))

    if torch.cuda.is_available():
        arches = set()
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            arches.add(f"{major}.{minor}")
        
        arch_list = ';'.join(sorted(arches))
        os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list
        print(f"Detected GPU architectures: {arch_list}")
    else:
        raise RuntimeError("No CUDA GPUs available")

    cpp_path = os.path.join(current_dir, 'spatial_attention.cpp')
    cuda_path = os.path.join(current_dir, 'spatial_attn.cu')
    
    with open(cpp_path, 'r') as f:
        cpp_source = f.read()
    with open(cuda_path, 'r') as f:
        cuda_source = f.read()

    arch_flags = [
        f'-gencode=arch=compute_{arch.replace(".", "")},code=sm_{arch.replace(".", "")}'
        for arch in arches
    ]

    if ptx_fallback:
        max_arch = max(arches).replace(".", "")
        arch_flags.append(f'-gencode=arch=compute_{max_arch},code=compute_{max_arch}')
        print("Using PTX fallback")

    extra_cuda_cflags = [
        '-O3',
        '-std=c++17',
    ] + arch_flags + [
        '--fmad=false',
        '--prec-div=true',
        '--prec-sqrt=true',
        '--ftz=false',
        '--ptxas-options=-v',
        '-lineinfo',
        '--extended-lambda',
        '-Xcompiler',
        '-fPIC'
    ]
    
    extra_cflags = ['-O2', '-std=c++17']

    try:
        extension = load_inline(
            name='spatial_range_attention_cuda',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['sp_attn_fwd', 'sp_attn_bwd'],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=True,
            with_cuda=True
        )
        print("CUDA extension compiled successfully!")
        return extension
    except Exception as e:
        print(f"Detailed error during compilation: {str(e)}")
        raise