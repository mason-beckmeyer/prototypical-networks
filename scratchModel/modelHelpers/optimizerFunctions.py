# Add memory tracking and optimization functions
import gc
import torch
from functools import lru_cache
from PIL import Image
from pathlib import Path  # Add this import if not already present


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


def find_best_model(directory=None):
    """Find the best model file in the directory based on accuracy in filename"""
    if directory is None:
        directory = "/Users/mbbec/PycharmProjects/prototypical-networks"

    # Convert string to Path object
    if isinstance(directory, str):
        directory = Path(directory)

    model_files = list(directory.glob("protonet_best_val_*.pth"))
    if not model_files:
        return None

    # Extract accuracies from filenames using regex
    import re
    best_acc = 0
    best_file = None
    for file in model_files:
        match = re.search(r'acc(\d+\.\d+)\.pth', str(file))
        if match:
            acc = float(match.group(1))
            if acc > best_acc:
                best_acc = acc
                best_file = file

    if best_file:
        print(f"Found best model: {best_file} (accuracy: {best_acc:.4f})")
    else:
        print("No model files found")

    return best_file

