import torch

def clear_gpu_cache():
    """Clear the GPU memory cache."""
    torch.cuda.empty_cache()
    print("GPU cache cleared.")