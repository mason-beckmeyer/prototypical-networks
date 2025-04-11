import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from tqdm import tqdm
from rasterio.features import rasterize
from shapely.geometry import shape
import geopandas as gpd
import rasterio
import json

DAMAGE_LABELS = {
    'no-damage': 0,
    'minor-damage': 1,
    'major-damage': 2,
    'destroyed': 3
}

class XBDPatchDataset(Dataset):
    def __init__(self, img_pre_dir, img_post_dir, mask_post_dir, patch_size=128, transform=None, max_per_class=100):
        self.img_pre_dir = img_pre_dir
        self.img_post_dir = img_post_dir
        self.mask_post_dir = mask_post_dir
        self.patch_size = patch_size
        self.transform = transform or Compose([Resize((patch_size, patch_size)), ToTensor()])
        self.max_per_class = max_per_class
        self.data = self._index_patches_by_class()

    def _index_patches_by_class(self):
        data_by_class = {i: [] for i in DAMAGE_LABELS.values()}
        print("Reading from:", self.mask_post_dir)

        for fname in os.listdir(self.mask_post_dir):
            if not fname.endswith('.png'):
                continue

            basename = fname.replace('_post_disaster_target.png', '')
            post_path = os.path.join(self.img_post_dir, f"{basename}_post_disaster.png")
            pre_path = os.path.join(self.img_pre_dir, f"{basename}_pre_disaster.png")
            mask_path = os.path.join(self.mask_post_dir, fname)

            if not (os.path.exists(pre_path) and os.path.exists(post_path)):
                print(f"⚠️ Skipping {fname}: missing pre/post image")
                continue

            try:
                # Open mask as grayscale
                mask = Image.open(mask_path).convert("L").resize((1024, 1024))
                mask_np = np.array(mask)

                unique_vals, counts = np.unique(mask_np, return_counts=True)
                class_dist = dict(zip(unique_vals, counts))
                print(f"📊 {fname} → {class_dist}")

                if set(unique_vals) == {0}:
                    print(f"🚫 Skipping {fname}: only background")
                    continue

                for label_str, label_id in DAMAGE_LABELS.items():
                    class_mask = (mask_np == label_id)
                    indices = np.argwhere(class_mask)
                    np.random.shuffle(indices)

                    for y, x in indices[:self.max_per_class]:
                        top = max(0, y - self.patch_size // 2)
                        left = max(0, x - self.patch_size // 2)
                        bottom = top + self.patch_size
                        right = left + self.patch_size

                        data_by_class[label_id].append({
                            'pre_path': pre_path,
                            'post_path': post_path,
                            'mask_path': mask_path,
                            'coords': (left, top, right, bottom),
                            'label': label_id
                        })
            except Exception as e:
                print(f"❌ Failed to process mask: {fname} → {e}")
                continue

        return data_by_class

    def get_episode(self, n_way=2, k_shot=5, q_query=5):
        selected_classes = random.sample(list(self.data.keys()), n_way)

        xs, xq = [], []
        for cls_idx, cls in enumerate(selected_classes):
            examples = random.sample(self.data[cls], k_shot + q_query)
            support, query = examples[:k_shot], examples[k_shot:]

            def load_patch(example):
                # Load pre-disaster and post-disaster images as RGB
                pre = Image.open(example['pre_path']).convert('RGB').crop(example['coords'])
                post = Image.open(example['post_path']).convert('RGB').crop(example['coords'])

                # Transform both images and stack them along the channel dimension
                pre_tensor = self.transform(pre)  # [3, H, W]
                post_tensor = self.transform(post)  # [3, H, W]
                patch = torch.cat([pre_tensor, post_tensor], dim=0)  # [6, H, W]

                return patch

            xs.append(torch.stack([load_patch(e) for e in support]))
            xq.append(torch.stack([load_patch(e) for e in query]))

        return {
            'xs': torch.stack(xs),  # [n_way, k_shot, 6, H, W]
            'xq': torch.stack(xq)   # [n_way, q_query, 6, H, W]
        }

    def __len__(self):
        return 100000  # arbitrarily large since we're sampling episodes

    def __getitem__(self, idx):
        return self.get_episode()


def load(opt, splits):
    loaders = {}
    for split in splits:
        root = os.path.join(opt['data.root'], split)
        dataset = XBDPatchDataset(
            img_pre_dir=os.path.join(root, 'img_pre/'),
            img_post_dir=os.path.join(root, 'img_post/'),
            mask_post_dir=os.path.join(root, 'gt_post/'),
            patch_size=opt.get('data.patch_size', 128),
        )

        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        loaders[split] = loader

    return loaders
