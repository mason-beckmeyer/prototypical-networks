import torch

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def extract_patches(image, patch_size,stride):
    patches = image.unfold(2, patch_size,stride).unfold(3, patch_size, stride)
    patches = patches.contiguous().view(-1, image.size(1), patch_size, patch_size)
    return patches
