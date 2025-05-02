import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
import matplotlib.pyplot as plt
import matplotlib.cm
from tqdm import tqdm
import copy
import torch.serialization
from modelHelpers.optimizerFunctions import optimize_memory,cached_image_load,custom_collate,find_best_model


DAMAGE_LABELS = {
    'no-damage': 1,
    'minor-damage': 2,
    'major-damage': 3,
    'destroyed': 4
}


# --- CBAM Module for Attention ---
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        #Streamlined spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #Channel attention with inplace operations where possible
        y = self.avg_pool(x)
        y = self.sigmoid(self.fc(y))
        x = x * y

        #Spatial attention with combined pooling operations
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_out, avg_out], dim=1)
        spatial_weight = self.sigmoid(self.spatial_conv(spatial_input))

        return x * spatial_weight


# --- Residual Block for Deeper Encoder ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# --- ProtoNet with Multi-Scale and Attention ---
class ProtoNetEnhanced(nn.Module):
    def __init__(self, input_channels=6, hidden_dim=64, num_scales=2, dropout_rate=0.3):
        super(ProtoNetEnhanced, self).__init__()
        self.num_scales = num_scales
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Deeper residual blocks
                ResidualBlock(hidden_dim, hidden_dim * 2),
                CBAM(hidden_dim * 2),
                nn.MaxPool2d(2),

                ResidualBlock(hidden_dim * 2, hidden_dim * 4),
                CBAM(hidden_dim * 4),
                nn.MaxPool2d(2),

                ResidualBlock(hidden_dim * 4, hidden_dim * 8),
                CBAM(hidden_dim * 8),
                nn.AdaptiveAvgPool2d(2)  # Final pool to fixed size  # Ensures fixed spatial size
            ) for _ in range(num_scales)
        ])
        # Compute the flattened feature size: hidden_dim * 2 * 4 * 4
        self.feature_dim = (hidden_dim * 8) * 2 * 2
        self.fusion = nn.Conv1d(self.feature_dim, hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    # Add these methods to the ProtoNetEnhanced class
    def calculate_prototypes(self, support_features, support_labels):
        """Optimized prototype calculation"""
        n_way = len(torch.unique(support_labels))
        prototypes = torch.zeros(n_way, support_features.shape[1], device=support_features.device)

        for i in range(n_way):
            mask = (support_labels == i)
            if mask.any():
                # One vectorized operation instead of mean reduction
                prototypes[i] = support_features[mask].mean(dim=0)
        return prototypes

    def compute_distances(self, query_features, prototypes):
        # Normalize feature vectors (important for cosine-like distance)
        query_features = F.normalize(query_features, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)


        n_query = query_features.size(0)
        n_prototypes = prototypes.size(0)
        query_features = query_features.unsqueeze(1)
        prototypes = prototypes.unsqueeze(0)
        return torch.sum((query_features - prototypes) ** 2, dim=2)

    def forward(self, x_list):
        features = []

        # Use gradient checkpointing for memory efficiency
        features = []
        for x, encoder in zip(x_list, self.encoders):
            feat = encoder(x)  # Direct call without checkpointing
            features.append(feat.view(feat.size(0), -1))

        fused = torch.stack(features, dim=2)  # [batch, feature_dim, num_scales]
        fused = self.dropout(fused)  # Apply dropout before fusion
        fused = self.fusion(fused)  # [batch, hidden_dim, num_scales]
        batch_size = fused.size(0)
        fused = fused.view(batch_size, -1)
        return fused
    def _run_encoder(self, x, encoder):
        """Helper function for gradient checkpointing"""
        return encoder(x)

# --- Enhanced XBDPatchDataset with Multi-Scale---
class XBDPatchDatasetEnhanced(Dataset):
    def __init__(self, root_dir: Path, split: str = 'train', patch_sizes: List[int] = [128, 256],
                 transform: Optional[Dict[int, transforms.Compose]] = None, max_patches_per_class: int = 100,
                 skip_extraction: bool = False, device: str = 'cpu',
                 initial_patches: Optional[List[Dict[str, Union[str, int, Tuple[int, int, int, int]]]]] = None,
                 initial_patches_by_class: Optional[
                     Dict[int, List[Dict[str, Union[str, int, Tuple[int, int, int, int]]]]]] = None):
        self.root_dir_path = root_dir / split
        self.patch_sizes = patch_sizes
        self.transform = transform or {ps: self._default_transform(ps) for ps in patch_sizes}
        self.max_patches_per_class = max_patches_per_class
        self.split_name = split
        self.device = device

        if skip_extraction and initial_patches is not None and initial_patches_by_class is not None:
            self.patches = initial_patches
            self.patches_by_class = initial_patches_by_class
        else:
            print(f"Indexing patches from {split} split at {self.root_dir_path}...")
            self.pre_image_dir = self.root_dir_path / 'img_pre'
            self.post_image_dir = self.root_dir_path / 'img_post'
            self.post_mask_dir = self.root_dir_path / 'gt_post'

            if not self.root_dir_path.exists():
                raise FileNotFoundError(f"Root directory for split '{split}' not found: {self.root_dir_path}")
            if not self.pre_image_dir.exists() or not self.post_image_dir.exists() or not self.post_mask_dir.exists():
                raise FileNotFoundError(
                    f"Required subdirectories (img_pre, img_post, gt_post) not found in {self.root_dir_path}")

            self.patches_by_class = self._extract_patches_by_class()
            self.patches = []
            for label, patches in self.patches_by_class.items():
                self.patches.extend(patches)

            print(f"Total patches in '{split}': {len(self.patches)}")
            for label, patches in self.patches_by_class.items():
                print(f"  Class {label} ({self._get_damage_label_name_static(label)}): {len(patches)} patches")
            if not self.patches:
                print(f"Warning: No patches were extracted for the '{split}' split. Check paths and data.")

    def _default_transform(self, patch_size):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(size=(patch_size, patch_size), scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    @staticmethod
    def _get_damage_label_name_static(label_id: int) -> str:
        for name, id_val in DAMAGE_LABELS.items():
            if id_val == label_id:
                return name
        return f"Unknown ({label_id})"

    def _get_damage_label_name(self, label_id: int) -> str:
        return self._get_damage_label_name_static(label_id)

    def _extract_patches_by_class(self) -> Dict[int, List[Dict[str, Union[str, int, Tuple[int, int, int, int]]]]]:
        patches_by_class = {label: [] for label in DAMAGE_LABELS.values()}
        post_images = list(self.post_image_dir.glob('*.png'))
        if not post_images:
            print(f"Warning: No PNG images found in {self.post_image_dir}. Check directory/files.")
            return patches_by_class

        print(f"Found {len(post_images)} post-disaster images in {self.post_image_dir}")
        for img_path in tqdm(post_images, desc=f"Processing images in {self.split_name}"):
            base_name = img_path.stem.replace('_post_disaster', '')
            post_mask_path = self.post_mask_dir / f"{base_name}_post_disaster_target{img_path.suffix}"
            pre_img_path = self.pre_image_dir / f"{base_name}_pre_disaster{img_path.suffix}"

            if not post_mask_path.exists() or not pre_img_path.exists():
                continue

            try:
                mask = Image.open(post_mask_path).convert('L')
                mask_np = np.array(mask)
                img_height, img_width = mask_np.shape

                unique_labels_in_mask = np.unique(mask_np)
                print(f"Processing {img_path.name}: Found labels {unique_labels_in_mask}")
                relevant_labels = set(unique_labels_in_mask) & set(patches_by_class.keys())

                if not relevant_labels:
                    continue

                for label in relevant_labels:
                    indices = np.argwhere(mask_np == label)
                    if len(indices) == 0:
                        continue

                    np.random.shuffle(indices)
                    num_to_sample = min(len(indices), self.max_patches_per_class)
                    sampled_indices = indices[:num_to_sample]

                    for y, x in sampled_indices:
                        patch_info = {
                            'pre_path': str(pre_img_path),
                            'post_path': str(img_path),
                            'coords': {},  # Store coords for each patch size
                            'label': label
                        }
                        valid_patch = True
                        for ps in self.patch_sizes:
                            half_patch = ps // 2
                            top = max(0, y - half_patch)
                            left = max(0, x - half_patch)
                            bottom = top + ps
                            right = left + ps

                            if bottom > img_height:
                                bottom = img_height
                                top = max(0, bottom - ps)
                            if right > img_width:
                                right = img_width
                                left = max(0, right - ps)

                            if bottom - top != ps or right - left != ps:
                                valid_patch = False
                                break

                            patch_info['coords'][ps] = (left, top, right, bottom)

                        if valid_patch:
                            patches_by_class[label].append(patch_info)
            except Exception as e:
                print(f"Error processing image {img_path.name}: {e}")
                continue
        return patches_by_class

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        pre_image = cached_image_load(patch_info['pre_path'])
        post_image = cached_image_load(patch_info['post_path'])

        # Process all scales at once
        combined_images = []
        for ps in self.patch_sizes:
            transform = self.transform[ps]
            coords = patch_info['coords'][ps]
            pre_crop = pre_image.crop(coords)
            post_crop = post_image.crop(coords)

            pre_transformed = transform(pre_crop)
            post_transformed = transform(post_crop)
            combined_images.append(torch.cat((pre_transformed, post_transformed), dim=0))

        return combined_images, patch_info['label'], patch_info

    def sample_episode(self, n_way: int, k_shot: int, q_query: int,selected_classes=None) -> Dict:
        """Efficiently sample episode with optimized memory usage"""
        # Pre-filter viable classes (classes with enough samples)
        if selected_classes is None:

            viable_classes = [cls for cls, examples in self.patches_by_class.items()
                              if len(examples) >= k_shot + q_query]

            if len(viable_classes) < n_way:
                print(f"⚠️ Not enough examples for {n_way}-way. Reducing to {len(viable_classes)}-way.")
                n_way = len(viable_classes)
                if n_way == 0:
                    raise ValueError("No classes have sufficient examples for episode sampling")

            selected_original_classes = random.sample(viable_classes, k=n_way)
        else:
            selected_original_classes = selected_classes

        #Prepare containers
        support_images = []
        query_images = []
        support_labels = []
        query_labels = []
        original_to_episode_label = {orig_label: i for i, orig_label in enumerate(selected_original_classes)}

        #Process selected classes efficiently
        for original_label in selected_original_classes:
            episode_label = original_to_episode_label[original_label]
            class_patches = self.patches_by_class[original_label]

            # Select random indices without replacement
            indices = torch.randperm(len(class_patches))[:k_shot + q_query].tolist()
            selected_patches_info = [class_patches[i] for i in indices]

            # Process support set (first k_shot samples)
            for patch_info in selected_patches_info[:k_shot]:
                images, _, _ = self.__getitem__(self.patches.index(patch_info))
                support_images.append(images)
                support_labels.append(episode_label)

            # Process query set (remaining samples)
            for patch_info in selected_patches_info[k_shot:]:
                images, _, _ = self.__getitem__(self.patches.index(patch_info))
                query_images.append(images)
                query_labels.append(episode_label)

        # Create and return the episode data
        xs = [torch.stack([img[i] for img in support_images]) for i in range(len(self.patch_sizes))]
        xq = [torch.stack([img[i] for img in query_images]) for i in range(len(self.patch_sizes))]

        return {
            'xs': xs,
            'xq': xq,
            'ys': torch.tensor(support_labels, dtype=torch.long),
            'yq': torch.tensor(query_labels, dtype=torch.long),
            'original_classes': selected_original_classes
        }


# --- Visualization Functions ---
def get_damage_name(label_id: int) -> str:
    for name, id_val in DAMAGE_LABELS.items():
        if id_val == label_id:
            return name.replace('-', ' ').title()
    return f"Unknown ({label_id})"


def load_trained_model(model_class: type, model_path: str, input_channels: int, hidden_dim: int, device: str,
                       num_scales: int = 2) -> Optional[ProtoNetEnhanced]:
    model = model_class(input_channels=input_channels, hidden_dim=hidden_dim, num_scales=num_scales)
    try:
        print(f"Attempting to load checkpoint from {model_path}...")
        # Check if file exists and is accessible
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {model_path}")
        if not os.path.isfile(model_path):
            raise ValueError(f"Path is not a file: {model_path}")

        # Try loading with weights_only=False to inspect the content
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"Checkpoint type: {type(checkpoint)}")
        print(f"Checkpoint content: {checkpoint if isinstance(checkpoint, dict) else str(checkpoint)[:100]}...")

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Found 'model_state_dict' in checkpoint.")
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            print("Treating checkpoint as state dict directly.")
        else:
            raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}. Expected a dict or state dict.")

        if not isinstance(state_dict, dict):
            raise ValueError(f"State dict is not a dictionary: {type(state_dict)}")
        print(f"State dict keys: {list(state_dict.keys())[:5]}...")

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError as e:
        print(f"Error: Model file not found at {model_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        return None


def get_prediction_support_set(train_dataset: XBDPatchDatasetEnhanced, k_shot: int, device: str) -> Tuple[
    Optional[List[torch.Tensor]], Optional[torch.Tensor], Optional[List[int]]]:
    support_images = []
    support_episode_labels = []

    # Only use classes that have sufficient examples in the dataset
    available_classes = [label for label, patches in train_dataset.patches_by_class.items()
                         if len(patches) >= k_shot]

    if not available_classes:
        print(f"FATAL ERROR: No classes have sufficient examples for {k_shot}-shot support set")
        return None, None, None

    print(f"Creating support set with {len(available_classes)} available classes: {available_classes}")

    # Map original labels to sequential episode labels (0, 1, 2, ...)
    original_class_order = sorted(available_classes)
    episode_label_map = {original_label: i for i, original_label in enumerate(original_class_order)}

    for original_label in original_class_order:
        episode_label = episode_label_map[original_label]
        class_patches_info = train_dataset.patches_by_class.get(original_label, [])
        num_available = len(class_patches_info)

        if num_available < k_shot:
            print(
                f"Warning: Class {original_label} ({get_damage_name(original_label)}) only has {num_available} samples (need {k_shot}). Sampling with replacement.")
            selected_patches_info = random.choices(class_patches_info, k=k_shot)
        else:
            selected_patches_info = random.sample(class_patches_info, k=k_shot)

        loaded_count = 0
        for patch_info in selected_patches_info:
            try:
                idx = train_dataset.patches.index(patch_info)
                combined, _, _ = train_dataset[idx]
                support_images.append(combined)
                support_episode_labels.append(episode_label)
                loaded_count += 1
            except Exception as e:
                print(f"Error loading support patch: {e}. Skipping.")

        print(
            f"Loaded {loaded_count}/{k_shot} support samples for class {original_label} ({get_damage_name(original_label)})")

    if not support_images:
        print("FATAL ERROR: Failed to load any support samples")
        return None, None, None

    support_images_tensors = [torch.stack([img[i] for img in support_images]).to(device) for i in
                              range(len(train_dataset.patch_sizes))]
    support_labels_tensor = torch.tensor(support_episode_labels, dtype=torch.long).to(device)

    print(f"Successfully created support set with shapes: {[t.shape for t in support_images_tensors]}")
    return support_images_tensors, support_labels_tensor, original_class_order


def visualize_predictions(model: ProtoNetEnhanced, vis_dataset: XBDPatchDatasetEnhanced,
                          train_dataset: XBDPatchDatasetEnhanced, k_shot: int,
                          num_samples: int = 10, device: str = 'cpu') -> None:
    if not model:
        print("Model not provided or loaded. Cannot visualize predictions.")
        return
    if not vis_dataset or len(vis_dataset) == 0:
        print("Visualization dataset is empty or not provided. Cannot visualize.")
        return
    if not train_dataset or len(train_dataset) == 0:
        print("Training dataset for support set is empty or not provided. Cannot visualize.")
        return

    model.eval()
    model.to(device)

    print("\n--- Visualizing Predictions ---")
    print(f"Attempting to generate {num_samples} visualizations...")

    num_to_vis = min(num_samples, len(vis_dataset))
    if num_to_vis == 0:
        print("No samples available in the visualization dataset.")
        return

    print("Generating support set for predictions...")
    support_images, support_labels, class_order = get_prediction_support_set(train_dataset, k_shot, device)

    if support_images is None:
        print("Failed to create support set. Aborting visualization.")
        return
    n_way = len(class_order)
    print(f"Working with {n_way}-way classification using available classes")
    if n_way < len(DAMAGE_LABELS):
        print(f"Note: Only {n_way}/{len(DAMAGE_LABELS)} damage classes available in support set")
        print(f"Available classes: {[get_damage_name(cls) for cls in class_order]}")

    try:
        with torch.no_grad():
            support_features = model(support_images)
            if support_features.shape[0] != support_labels.shape[0]:
                raise ValueError(
                    f"Support features ({support_features.shape[0]}) and labels ({support_labels.shape[0]}) count mismatch.")
            prototypes = model.calculate_prototypes(support_features, support_labels)
            print(f"Prototypes calculated for {prototypes.shape[0]} classes.")
    except Exception as e:
        print(f"Error calculating prototypes: {e}. Aborting visualization.")
        return

    vis_indices = random.sample(range(len(vis_dataset)), num_to_vis)
    print(f"Selected {len(vis_indices)} indices to visualize.")

    successful_visualizations = 0
    for i, idx in enumerate(vis_indices):
        print(f"\nProcessing sample {i + 1}/{len(vis_indices)} (Index: {idx})")
        try:
            query_tensors, true_label, metadata = vis_dataset[idx]
            query_tensors = [t.to(device).unsqueeze(0) for t in query_tensors]

            pre_img_orig = Image.open(metadata['pre_path']).convert('RGB')
            post_img_orig = Image.open(metadata['post_path']).convert('RGB')
            coords_raw = metadata['coords'][128]  # Use 128x128 for visualization
            coords = tuple(int(c) for c in coords_raw)
            # if not (isinstance(coords, tuple) and len(coords) == 4 and all(isinstance(c, int) for c in coords)):
            #     print(f"Error: Invalid coordinates format in metadata for index {idx}: {coords}. Skipping.")
            #     continue
            pre_patch_orig = pre_img_orig.crop(coords)
            post_patch_orig = post_img_orig.crop(coords)

            with torch.no_grad():
                query_features = model(query_tensors)
                distances = model.compute_distances(query_features, prototypes)
                if distances.numel() == 0 or distances.shape[1] != prototypes.shape[0]:
                    print(
                        f"Error: Distance calculation failed or resulted in unexpected shape {distances.shape} (expected [1, {prototypes.shape[0]}]). Skipping.")
                    continue
                pred_index = torch.argmin(distances, dim=1).item()

            predicted_label = class_order[pred_index] if pred_index < len(class_order) else -1
            true_name = get_damage_name(true_label)
            pred_name = get_damage_name(predicted_label) if predicted_label != -1 else "Prediction Error"

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.patch.set_facecolor('white')

            axes[0].imshow(pre_patch_orig)
            axes[0].set_title("Pre-Disaster Patch")
            axes[0].axis('off')

            axes[1].imshow(post_patch_orig)
            axes[1].set_title("Post-Disaster Patch")
            axes[1].axis('off')

            correct = (true_label == predicted_label)
            title_color = 'green' if correct else 'red'
            if predicted_label == -1:
                title_color = 'black'

            fig.suptitle(f'Sample {i + 1} - Index: {idx}\nPrediction: {pred_name} (True: {true_name})',
                         fontsize=14, color=title_color, y=0.98)

            plt.tight_layout(rect=[0, 0.03, 1, 0.92])
            save_path = f'prediction_visualization_sample{i + 1}_idx{idx}.png'
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")
            plt.close(fig)
            successful_visualizations += 1

        except Exception as e:
            print(f"Error visualizing index {idx}: {e}. Skipping.")
            plt.close('all')

    print(
        f"\nFinished visualization attempts. Successfully generated {successful_visualizations}/{len(vis_indices)} images.")


# --- Training Function ---
def train_protonet_with_patches(model, train_dataset, val_dataset,
                                n_way, k_shot, q_query,
                                num_episodes=100, learning_rate=0.001,
                                val_interval=20, early_stopping_patience=None,
                                device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)

    if device.startswith('cuda'):
        torch.backends.cudnn.benchmark = True

    # aggressive caching
    print("Preloading ALL images into memory")
    image_cache = {}
    unique_paths = set()
    for patch in train_dataset.patches:
        unique_paths.add(patch['pre_path'])
        unique_paths.add(patch['post_path'])
    if val_dataset:
        for patch in val_dataset.patches:
            unique_paths.add(patch['pre_path'])
            unique_paths.add(patch['post_path'])

    # Load all unique images
    for path in tqdm(unique_paths, desc="Preloading images"):
        try:
            image_cache[path] = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")

    # Redefine image loading function to use preloaded cache
    def optimized_cached_load(path):
        return image_cache[path]

    # Replace functions
    train_dataset.cached_image_load = optimized_cached_load
    if val_dataset:
        val_dataset.cached_image_load = optimized_cached_load

    # Create optimizer with higher learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=0.0001
    )
    scaler = torch.cuda.amp.GradScaler() if device.startswith('cuda') else None
    for epoch in range(5):  # First 5 episodes
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * (epoch + 1) / 5
    best_val_acc = -1.0
    best_model_state = None
    epochs_without_improvement = 0
    train_losses = []
    train_accuracies = []

    # Precompute class probabilities for weighted sampling
    class_weights = {label: (1.0 / len(patches)) ** 1.5 for label, patches in train_dataset.patches_by_class.items()}
    class_weights = {k: v / min(class_weights.values()) for k, v in class_weights.items()}
    print("Class weights ",class_weights)
    viable_classes = []
    class_probabilities = []
    for cls, examples in train_dataset.patches_by_class.items():
        if len(examples) >= k_shot + q_query:
            viable_classes.append(cls)
            class_probabilities.append(class_weights[cls])

    # Normalize probabilities
    class_probabilities = np.array(class_probabilities) / sum(class_probabilities)

    print(f"Starting training on {device} for {num_episodes} episodes...")
    print(f"Using {len(viable_classes)} viable classes for {n_way}-way classification")

    # PERFORMANCE OPTIMIZED TRAINING LOOP
    for episode in range(num_episodes):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Silent episode sampling
        try:
            # Weighted sample classes without printing warnings
            if len(viable_classes) < n_way:
                current_n_way = len(viable_classes)
                episode_classes = viable_classes
            else:
                current_n_way = n_way
                episode_classes = np.random.choice(
                    viable_classes,
                    size=current_n_way,
                    replace=False,
                    p=class_probabilities
                )

            # Use dataset's sample_episode with our selected classes
            episode_data = train_dataset.sample_episode(current_n_way, k_shot, q_query,
                                                        selected_classes=episode_classes)

            xs, xq = episode_data['xs'], episode_data['xq']
            ys, yq = episode_data['ys'].to(device), episode_data['yq'].to(device)
            xs = [x.to(device) for x in xs]
            xq = [x.to(device) for x in xq]

            if any(x.shape[0] == 0 for x in xs) or any(x.shape[0] == 0 for x in xq):
                if episode % 10 == 0:  # Only log occasionally
                    print(f"Episode {episode + 1}: Empty support/query set, skipping")
                scheduler.step()
                continue
        except Exception as e:
            if episode % 10 == 0:  # Only log occasionally
                print(f"Episode {episode + 1}: Sampling error: {str(e)[:80]}...")
            scheduler.step()
            continue

        try:
            with torch.amp.autocast(device_type=device.split(':')[0], enabled=scaler is not None):
                support_features = model(xs)
                query_features = model(xq)
                prototypes = model.calculate_prototypes(support_features, ys)
                distances = model.compute_distances(query_features, prototypes)
                log_p_y = F.log_softmax(-distances, dim=1)
                loss = F.nll_loss(log_p_y, yq)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                _, predicted_labels = torch.min(distances, 1)
                comparison = (predicted_labels == yq)
                accuracy = comparison.float().mean().item()
            train_losses.append(loss.item())
            train_accuracies.append(accuracy)
        except Exception as e:
            if episode % 10 == 0:  # Only log occasionally
                print(f"Episode {episode + 1}: Forward/backward error: {str(e)[:80]}...")
            optimizer.zero_grad(set_to_none=True)
            continue

        scheduler.step()

        # Only print every 5 episodes to reduce overhead
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: Loss = {loss.item():.4f}, Acc = {accuracy:.4f}")

        # Validate less frequently
        if val_dataset is not None and (episode + 1) % val_interval == 0:
            print(f"\n--- Running Validation @ Episode {episode + 1} ---")
            try:
                # Less validation episodes
                val_results = evaluate_model(model, val_dataset, num_episodes=10, n_way=n_way,
                                             k_shot=k_shot, q_query=q_query, device=device,
                                             eval_mode='validation')
                val_acc = val_results['overall_accuracy']
                print(f"--- Validation Accuracy: {val_acc:.4f} ---")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0

                    # Save checkpoint very infrequently (only every 10 validations)
                    if (episode + 1) % (5) == 0:
                        checkpoint_path = f'protonet_best_val_ep{episode + 1}_acc{val_acc:.4f}.pth'
                        torch.save({'model_state_dict': best_model_state}, checkpoint_path)
                        print(f"Checkpoint saved to {checkpoint_path}")
                else:
                    epochs_without_improvement += 1
                if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping after {epochs_without_improvement} checks.")
                    break
            except Exception as e:
                print(f"Error during validation: {str(e)[:80]}...")
            optimize_memory()

    # Clean up image cache and restore original function
    image_cache.clear()

    print("\nTraining finished.")
    if best_model_state is not None:
        print(f"Loading best model (Acc: {best_val_acc:.4f})")
        model.load_state_dict(best_model_state)

    return model


# --- Evaluation Function ---
def evaluate_model(model: ProtoNetEnhanced, dataset: XBDPatchDatasetEnhanced, num_episodes: int = 100,
                   n_way: int = 4, k_shot: int = 5, q_query: int = 10,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                   eval_mode: str = 'test') -> Dict:
    model.eval()
    all_accuracies = []
    class_correct_counts = {label: 0 for label in DAMAGE_LABELS.values()}
    class_total_counts = {label: 0 for label in DAMAGE_LABELS.values()}
    class_true_positives = {label: 0 for label in DAMAGE_LABELS.values()}
    class_predicted_counts = {label: 0 for label in DAMAGE_LABELS.values()}
    confusion_matrix = np.zeros((n_way, n_way))
    print(f"Starting {eval_mode} evaluation ({dataset.split_name}) with {num_episodes} episodes...")

    for episode in tqdm(range(num_episodes), desc=f"Evaluating ({eval_mode} - {dataset.split_name})"):
        with torch.no_grad():
            try:
                episode_data = dataset.sample_episode(n_way, k_shot, q_query)
                xs, xq = episode_data['xs'], episode_data['xq']
                ys, yq = episode_data['ys'].to(device), episode_data['yq'].to(device)
                xs = [x.to(device) for x in xs]
                xq = [x.to(device) for x in xq]
                original_classes = episode_data['original_classes']
                if any(x.shape[0] == 0 for x in xs) or any(x.shape[0] == 0 for x in xq):
                    raise ValueError("Empty support or query set")
            except ValueError as e:
                print(f"Error sampling eval episode {episode + 1} ({eval_mode}): {e}. Skipping.")
                continue
            except Exception as e:
                print(f"Unexpected error sampling eval episode {episode + 1} ({eval_mode}): {e}. Skipping.")
                continue

            try:
                support_features = model(xs)
                query_features = model(xq)
                prototypes = model.calculate_prototypes(support_features, ys)
                distances = model.compute_distances(query_features, prototypes)
                _, predicted_episode_labels = torch.min(distances, 1)

                comparison = (predicted_episode_labels == yq)
                accuracy = comparison.float().mean().item()
                all_accuracies.append(accuracy)

                episode_to_original_label = {i: orig_label for i, orig_label in enumerate(original_classes)}
                for i in range(len(yq)):
                    true_episode_label = yq[i].item()
                    pred_episode_label = predicted_episode_labels[i].item()
                    if true_episode_label not in episode_to_original_label:
                        continue
                    true_original_label = episode_to_original_label[true_episode_label]
                    pred_original_label = episode_to_original_label.get(pred_episode_label, -1)
                    if pred_original_label == -1:
                        continue
                    class_total_counts[true_original_label] += 1
                    class_predicted_counts[pred_original_label] += 1
                    if pred_episode_label == true_episode_label:
                        class_correct_counts[true_original_label] += 1
                        class_true_positives[true_original_label] += 1
                    if 0 <= pred_episode_label < n_way and 0 <= true_episode_label < n_way:
                        confusion_matrix[true_episode_label, pred_episode_label] += 1
                    else:
                        print(
                            f"Warning: Invalid episode label detected in evaluation. True: {true_episode_label}, Pred: {pred_episode_label}. Max expected: {n_way - 1}")
            except Exception as e:
                print(f"Error during evaluation forward pass episode {episode + 1} ({eval_mode}): {e}. Skipping episode.")
                continue

    overall_accuracy_mean = np.mean(all_accuracies) if all_accuracies else 0.0
    overall_accuracy_std = np.std(all_accuracies) if all_accuracies else 0.0
    class_accuracies = {}
    class_f1_scores = {}
    class_precision = {}
    class_recall = {}
    valid_original_classes = []
    for label in DAMAGE_LABELS.values():
        if class_total_counts[label] > 0:
            acc = class_correct_counts[label] / class_total_counts[label]
            precision = class_true_positives[label] / class_predicted_counts[label] if class_predicted_counts[label] > 0 else 0.0
            recall = class_true_positives[label] / class_total_counts[label] if class_total_counts[label] > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            class_accuracies[label] = acc
            class_f1_scores[label] = f1
            class_precision[label] = precision
            class_recall[label] = recall
            valid_original_classes.append(label)

    macro_f1 = np.mean(list(class_f1_scores.values())) if class_f1_scores else 0.0
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_cm = np.divide(confusion_matrix, row_sums, out=np.zeros_like(confusion_matrix), where=row_sums != 0)

    results = {
        'overall_accuracy': overall_accuracy_mean,
        'overall_accuracy_std': overall_accuracy_std,
        'class_accuracies': class_accuracies,
        'class_f1_scores': class_f1_scores,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'macro_f1': macro_f1,
        'confusion_matrix': normalized_cm,
        'evaluated_original_classes': sorted(valid_original_classes),
        'n_way_evaluated': n_way,
        'all_accuracies': all_accuracies  # For episode-wise distribution
    }
    print(f"Evaluation finished ({eval_mode}). Overall Accuracy: {overall_accuracy_mean:.4f} +/- {overall_accuracy_std:.4f}")
    print(f"Macro-Averaged F1-Score: {macro_f1:.4f}")
    return results


# --- Results Visualization Function ---
def visualize_results(results: Dict, prefix: str = "") -> None:
    damage_names = {1: "No Damage", 2: "Minor", 3: "Major", 4: "Destroyed"}
    print(f"\n--- {prefix} Evaluation Results ---")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f} (std: {results['overall_accuracy_std']:.4f})")
    print(f"Macro-Averaged F1-Score: {results['macro_f1']:.4f}")
    class_accuracies = results['class_accuracies']
    class_f1_scores = results['class_f1_scores']
    class_precision = results['class_precision']
    class_recall = results['class_recall']
    evaluated_original_classes = results['evaluated_original_classes']
    if not evaluated_original_classes:
        print("No classes evaluated, skipping plots.")
        return

    plot_labels = [damage_names.get(label, f"Class {label}") for label in evaluated_original_classes]
    plot_accuracies = [class_accuracies[label] for label in evaluated_original_classes]
    plot_f1_scores = [class_f1_scores[label] for label in evaluated_original_classes]
    plot_precision = [class_precision[label] for label in evaluated_original_classes]
    plot_recall = [class_recall[label] for label in evaluated_original_classes]

    # Print per-class metrics
    print("Per-Class Metrics:")
    for label, acc, f1, prec, rec in zip(plot_labels, plot_accuracies, plot_f1_scores, plot_precision, plot_recall):
        print(f"  {label}: Accuracy={acc:.4f}, F1-Score={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

    # Plot per-class accuracies
    plt.figure(figsize=(8, 5))
    bars = plt.bar(plot_labels, plot_accuracies, color='skyblue')
    plt.ylabel('Accuracy')
    plt.title(f'{prefix} Per-Class Accuracy')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        y_value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, y_value + 0.01, f'{y_value:.2f}', va='bottom', ha='center')
    plt.tight_layout()
    plt.savefig(f'{prefix}_class_accuracies.png')
    print(f"Saved per-class accuracy plot to {prefix}_class_accuracies.png")
    plt.close()

    # Plot per-class F1-scores
    plt.figure(figsize=(8, 5))
    bars = plt.bar(plot_labels, plot_f1_scores, color='lightcoral')
    plt.ylabel('F1-Score')
    plt.title(f'{prefix} Per-Class F1-Score')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        y_value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, y_value + 0.01, f'{y_value:.2f}', va='bottom', ha='center')
    plt.tight_layout()
    plt.savefig(f'{prefix}_class_f1_scores.png')
    print(f"Saved per-class F1-score plot to {prefix}_class_f1_scores.png")
    plt.close()

    # Plot episode-wise accuracy distribution
    all_accuracies = results['all_accuracies']
    if all_accuracies:
        plt.figure(figsize=(8, 5))
        plt.hist(all_accuracies, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Episode Accuracy')
        plt.ylabel('Frequency')
        plt.title(f'{prefix} Episode-Wise Accuracy Distribution')
        plt.tight_layout()
        plt.savefig(f'{prefix}_accuracy_distribution.png')
        print(f"Saved accuracy distribution plot to {prefix}_accuracy_distribution.png")
        plt.close()

    # Plot confusion matrix
    confusion_matrix = results['confusion_matrix']
    n_way = results['n_way_evaluated']
    cm_labels = [f"Cls {i}" for i in range(n_way)]
    if confusion_matrix.size == 0 or n_way == 0:
        return
    plt.figure(figsize=(7, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title(f'{prefix} Avg. Norm. Confusion Matrix ({n_way}-way)')
    plt.colorbar(label='Normalized Frequency')
    tick_marks = np.arange(len(cm_labels))
    plt.xticks(tick_marks, cm_labels, rotation=45, ha='right')
    plt.yticks(tick_marks, cm_labels)
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, f'{confusion_matrix[i, j]:.2f}', ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    plt.ylabel('True Episode Label (0..N-1)')
    plt.xlabel('Predicted Episode Label (0..N-1)')
    plt.tight_layout()
    plt.savefig(f'{prefix}_confusion_matrix.png')
    print(f"Saved confusion matrix plot to {prefix}_confusion_matrix.png")
    plt.close()


RUN_VISUALIZATION_ONLY = False
MODEL_LOAD_PATH = find_best_model()
NUM_VISUALIZATIONS = 30

def main() -> None:
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
   # xbd_root = Path(r"C:\Users\joshp_ya\PycharmProjects\earthquake")
    xbd_root = Path(r"/Users/mbbec/PycharmProjects/prototypical-networks/scratchModel/data")

    patch_sizes = [128, 256]
    hidden_dim = 64
    n_way = 4
    k_shot = 10
    q_query = 15
    validation_split_ratio = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_patches_per_class_extraction = 150

    train_transforms = {
        ps: transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((ps, ps)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]) for ps in patch_sizes
    }
    eval_transforms = {
        ps: transforms.Compose([
            transforms.Resize((ps, ps)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]) for ps in patch_sizes
    }

    train_dataset = None
    val_dataset = None
    test_dataset = None

    try:
        print("\n--- Loading Test Dataset ---")
        test_dataset = XBDPatchDatasetEnhanced(root_dir=xbd_root, split='test', patch_sizes=patch_sizes,
                                               transform=eval_transforms, max_patches_per_class=max_patches_per_class_extraction,
                                               device=device)
        if test_dataset and len(test_dataset) > 0:
            print(f"Test dataset loaded with {len(test_dataset)} patches")
            for label, patches in test_dataset.patches_by_class.items():
                print(f"  Class {label} ({get_damage_name(label)}): {len(patches)} patches")
    except Exception as e:
        print(f"Could not load test set: {e}")

    try:
        print("\n--- Loading Full Training Dataset ---")
        full_train_dataset = XBDPatchDatasetEnhanced(root_dir=xbd_root, split='train', patch_sizes=patch_sizes,
                                                     transform=train_transforms, max_patches_per_class=max_patches_per_class_extraction,
                                                     device=device)
        if full_train_dataset and len(full_train_dataset) > 0:
            print(f"Full train dataset loaded with {len(full_train_dataset)} patches")
            for label, patches in full_train_dataset.patches_by_class.items():
                print(f"  Class {label} ({get_damage_name(label)}): {len(patches)} patches")

        val_dir = xbd_root / 'val'
        if val_dir.exists():
            print("\n--- Loading Pre-defined Validation Dataset ---")
            val_dataset = XBDPatchDatasetEnhanced(root_dir=xbd_root, split='val', patch_sizes=patch_sizes,
                                                  transform=eval_transforms, max_patches_per_class=max_patches_per_class_extraction,
                                                  device=device)
            train_dataset = full_train_dataset
        else:
            print("\n--- Creating Validation Split from Training Data ---")
            train_subset_patches_by_class = {label: [] for label in DAMAGE_LABELS.values()}
            val_subset_patches_by_class = {label: [] for label in DAMAGE_LABELS.values()}
            for label, patches in full_train_dataset.patches_by_class.items():
                if not patches:
                    continue
                random.shuffle(patches)
                split_idx = int(len(patches) * (1.0 - validation_split_ratio))
                train_subset_patches_by_class[label] = patches[:split_idx]
                val_subset_patches_by_class[label] = patches[split_idx:]
            train_subset_patches = [p for label in DAMAGE_LABELS.values() for p in train_subset_patches_by_class[label]]
            val_subset_patches = [p for label in DAMAGE_LABELS.values() for p in val_subset_patches_by_class[label]]

            val_dataset = XBDPatchDatasetEnhanced(root_dir=xbd_root, split='val_from_train', patch_sizes=patch_sizes,
                                                  transform=eval_transforms, skip_extraction=True,
                                                  initial_patches=val_subset_patches,
                                                  initial_patches_by_class=val_subset_patches_by_class, device=device)
            train_dataset = XBDPatchDatasetEnhanced(root_dir=xbd_root, split='train_subset', patch_sizes=patch_sizes,
                                                    transform=train_transforms, skip_extraction=True,
                                                    initial_patches=train_subset_patches,
                                                    initial_patches_by_class=train_subset_patches_by_class,
                                                    device=device)
            print(f"Resulting Train Patches: {len(train_dataset)}")
            print(f"Resulting Validation Patches: {len(val_dataset)}")

    except FileNotFoundError as e:
        print(f"Fatal Error: Could not load necessary dataset files: {e}")
        return
    except Exception as e:
        print(f"Fatal Error during dataset loading: {e}")
        return

    if RUN_VISUALIZATION_ONLY and (not test_dataset or len(test_dataset) == 0):
        print("Warning: No test dataset loaded. Visualization might fail if 'vis_dataset' cannot be set.")
    if RUN_VISUALIZATION_ONLY and (not full_train_dataset or len(full_train_dataset) == 0):
        print("Fatal: Full training dataset needed for support set in visualization mode, but failed to load. Exiting.")
        return

    print("\n--- Initializing Model ---")

    model = ProtoNetEnhanced(
        input_channels=6,
        hidden_dim=96,
        num_scales=2,
        dropout_rate=0.35
    )

    # Use larger learning rate with warm restarts

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Created ProtoNetEnhanced model with {model_params:,} trainable parameters")

    if not RUN_VISUALIZATION_ONLY and train_dataset and val_dataset:
        print("\n--- Training Model ---")
        model = train_protonet_with_patches(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
            num_episodes=200,
            device=device,
            val_interval=20,
            early_stopping_patience=5
        )
        print("\n--- Training Completed ---")
    print("\n--- Final Evaluation on Test Set ---")
    if test_dataset and len(test_dataset) > 0:
        test_results = evaluate_model(model, test_dataset, num_episodes=20, n_way=n_way, k_shot=k_shot,
                                      q_query=q_query, device=device, eval_mode='test')
        visualize_results(test_results, prefix="test")
    else:
        print("Skipping test set evaluation: Test dataset not available.")

    print(f"\n--- Loading model for visualization from: {MODEL_LOAD_PATH} ---")
    if 'full_train_dataset' not in locals() or not full_train_dataset:
        print("Error: Full training dataset required for support set but not loaded. Cannot visualize.")
        return

    model_to_visualize = load_trained_model(ProtoNetEnhanced, MODEL_LOAD_PATH, input_channels=6, hidden_dim=64, device=device, num_scales=len(patch_sizes))
    train_data_for_vis_support = full_train_dataset

    print("\n--- Preparing for Visualization ---")
    if model_to_visualize is not None:
        vis_target_dataset = None
        if test_dataset and len(test_dataset) > 0:
            print("Using Test dataset for visualization.")
            vis_target_dataset = test_dataset
        elif val_dataset and len(val_dataset) > 0:
            print("Using Validation dataset for visualization (Test dataset not available/loaded).")
            vis_target_dataset = val_dataset
        else:
            print("Warning: No Test or Validation dataset available to visualize predictions from.")

        if train_data_for_vis_support is not None and vis_target_dataset:
            visualize_predictions(
                model=model_to_visualize,
                vis_dataset=vis_target_dataset,
                train_dataset=train_data_for_vis_support,
                k_shot=k_shot,
                num_samples=NUM_VISUALIZATIONS,
                device=device
            )
        else:
            print("Skipping visualization as training data or target dataset is not available.")
    else:
        print("Skipping visualization as no model was loaded successfully.")

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()