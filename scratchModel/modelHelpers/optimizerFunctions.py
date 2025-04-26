# Add memory tracking and optimization functions
import gc
import torch
from functools import lru_cache
from PIL import Image


def optimize_memory():
    """Free unused memory and report usage"""
    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")

@lru_cache(maxsize=512)
def cached_image_load(path):
    """Cache image loading to reduce disk I/O"""
    return Image.open(path).convert('RGB')


def custom_collate(batch):
    """Efficient collate function that avoids unnecessary copies"""
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    meta = [item[2] for item in batch]

    # Process by patch size
    combined_tensors = []
    for i in range(len(images[0])):
        combined_tensors.append(torch.stack([img[i] for img in images]))

    return combined_tensors, torch.tensor(labels), meta