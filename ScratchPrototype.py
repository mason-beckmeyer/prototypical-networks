
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import copy
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.utils import resample
import csv
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import time
from torch.amp import autocast, GradScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DAMAGE_LABELS = {
    'no-damage': 1,
    'minor-damage': 2,
    'major-damage': 3,
    'destroyed': 4
}
skipped_episodes = 0
same_image_episodes = Counter()

# --- Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- ProtoNet with SE Blocks and Grad-CAM ---
class ProtoNet(nn.Module):
    def __init__(self, input_channels=6, feature_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),
            nn.Dropout(0.5)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(64 * 8 * 8, feature_dim)
        self.feature_dim = feature_dim
        self.last_conv = self.encoder[15]
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x, return_features=False):
        feature_maps = self.encoder(x)
        pooled_features = self.adaptive_pool(feature_maps)
        features = pooled_features.view(pooled_features.size(0), -1)
        features = self.fc(features)
        if return_features:
            return features, feature_maps
        return features

    def calculate_prototypes(self, features, labels, robust=False):
        unique_labels = torch.unique(labels)
        feature_dim = features.shape[1]
        device = features.device
        prototype_list = []

        for label_val_tensor in unique_labels:
            label_val = label_val_tensor.item()
            mask = (labels == label_val_tensor)
            class_features = features[mask]
            if class_features.shape[0] == 0:
                logging.warning(f"Empty class for label {label_val} in prototype calculation")
                prototype_list.append(torch.zeros(feature_dim, device=device, requires_grad=False))
                continue
            if robust:
                prototype = torch.median(class_features, dim=0)[0]
            else:
                prototype = class_features.mean(dim=0)
            prototype_list.append(prototype)

        if not prototype_list:
            raise ValueError("No prototypes computed; all classes in support set were empty.")
        return torch.stack(prototype_list, dim=0)

    def compute_distances(self, query_features, prototypes):
        diff = query_features.unsqueeze(1) - prototypes.unsqueeze(0)
        distances = torch.sum(diff ** 2, dim=2)
        return distances

    def get_se_l2_loss(self, l2_lambda=1e-4):
        l2_loss = 0.0
        for module in self.modules():
            if isinstance(module, SEBlock):
                for param in module.parameters():
                    if param.requires_grad:
                        l2_loss += torch.sum(param ** 2)
        return l2_lambda * l2_loss

    def compute_gradcam(self, x, target_class, device='cpu'):
        was_training = self.training
        self.eval()
        x = x.to(device)
        x.requires_grad_(True)
        self.gradients = None
        self.activations = None

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        forward_handle = self.last_conv.register_forward_hook(forward_hook)
        backward_handle = self.last_conv.register_backward_hook(backward_hook)

        try:
            features, _ = self.forward(x, return_features=True)
            self.zero_grad()
            score = features[0, target_class]
            score.backward()
            if self.gradients is None or self.activations is None:
                logging.warning("Gradients or activations not captured")
                return np.zeros((x.size(2), x.size(3)))
            pooled_gradients = torch.mean(self.gradients, dim=[2, 3])
            for i in range(self.activations.size(1)):
                self.activations[0, i] *= pooled_gradients[0, i]
            heatmap = torch.mean(self.activations[0], dim=0)
            heatmap = F.relu(heatmap)
            heatmap /= torch.max(heatmap) + 1e-8
            heatmap = heatmap.detach().cpu().numpy()
        except Exception as e:
            logging.error(f"Grad-CAM failed: {str(e)}")
            heatmap = np.zeros((x.size(2), x.size(3)))
        finally:
            forward_handle.remove()
            backward_handle.remove()
        if was_training:
            self.train()
        return heatmap

    def get_prototype_patches(self, support_features, support_labels, support_patches, num_top=3):
        prototypes = self.calculate_prototypes(support_features, support_labels)
        prototype_patches = {label.item(): [] for label in torch.unique(support_labels)}
        for i, label in enumerate(torch.unique(support_labels)):
            label_val = label.item()
            mask = (support_labels == label)
            class_features = support_features[mask]
            class_patches = [support_patches[j] for j in range(len(support_patches)) if mask[j]]
            if len(class_features) == 0:
                continue
            prototype = prototypes[i]
            distances = torch.sum((class_features - prototype) ** 2, dim=1)
            _, top_indices = torch.topk(distances, k=min(num_top, len(distances)), largest=False)
            prototype_patches[label_val] = [class_patches[j] for j in top_indices]
        return prototype_patches

class XBDPatchDataset(Dataset):
    def __init__(self, root_dir: Path, split: str = 'train', patch_size: int = 64,
                 transform: Optional[transforms.Compose] = None, max_patches_per_class: int = 100,
                 skip_extraction: bool = False, device: str = 'cpu',
                 initial_patches: Optional[List[Tuple]] = None,
                 initial_patches_by_class: Optional[Dict[int, List[Tuple]]] = None):
        self.root_dir_path = root_dir / split
        self.patch_size = patch_size
        self.transform = transform or self._default_transform()
        self.max_patches_per_class = max_patches_per_class
        self.split_name = split
        self.device = device
        self.patch_metadata = []
        self.metadata_by_class = {label: [] for label in DAMAGE_LABELS.values()}

        if skip_extraction and initial_patches is not None and initial_patches_by_class is not None:
            self._load_initial_metadata(initial_patches, initial_patches_by_class)
        else:
            logging.info(f"Indexing patches from {split} split at {self.root_dir_path}")
            self.pre_image_dir = self.root_dir_path / 'img_pre'
            self.post_image_dir = self.root_dir_path / 'img_post'
            self.post_mask_dir = self.root_dir_path / 'gt_post'

            if not all([self.root_dir_path.exists(), self.pre_image_dir.exists(),
                        self.post_image_dir.exists(), self.post_mask_dir.exists()]):
                raise FileNotFoundError(f"Required directories not found in {self.root_dir_path}")

            all_post_images = sorted(list(self.post_image_dir.glob('*.png')))
            if not all_post_images:
                raise FileNotFoundError(f"No post-disaster images found in {self.post_image_dir}")

            class_4_images = []
            other_images = []
            for img_path in all_post_images:
                base_name = img_path.stem.replace('_post_disaster', '')
                post_mask_path = self.post_mask_dir / f"{base_name}_post_disaster_target{img_path.suffix}"
                if not post_mask_path.exists():
                    continue
                with Image.open(post_mask_path).convert('L') as mask:
                    mask_np = np.array(mask)
                    if 4 in mask_np:
                        class_4_images.append(img_path)
                    else:
                        other_images.append(img_path)

            logging.info(f"Total images: {len(all_post_images)}, Class 4 images: {len(class_4_images)}, Other images: {len(other_images)}")

            class_4_images.sort()
            other_images.sort()

            total_images = len(all_post_images)
            test_size = int(0.2 * total_images)
            min_class_4_test = max(1, int(0.2 * len(class_4_images))) if class_4_images else 0

            test_class_4_images = class_4_images[:min_class_4_test]
            remaining_test_slots = test_size - len(test_class_4_images)
            test_other_images = other_images[:remaining_test_slots] if remaining_test_slots > 0 else []

            if split == 'test':
                post_images = test_class_4_images + test_other_images
            else:
                train_class_4_images = class_4_images[min_class_4_test:]
                train_other_images = other_images[len(test_other_images):]
                post_images = train_class_4_images + train_other_images

            logging.info(f"{split} split: {len(post_images)} images, Class 4 images: {sum(1 for img in post_images if img in class_4_images)}")

            self._extract_patches_by_class(post_images)

    def _default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def _load_initial_metadata(self, initial_patches, initial_patches_by_class):
        for patch in initial_patches:
            if not isinstance(patch, tuple) or len(patch) != 4:
                continue
            self.patch_metadata.append(patch)
        for label, patches in initial_patches_by_class.items():
            for patch in patches:
                if not isinstance(patch, tuple) or len(patch) != 4 or patch[3] != label:
                    continue
                self.metadata_by_class[label].append(patch)

    def _extract_patches_by_class(self, post_images: List[Path]):
        for img_path in tqdm(post_images, desc=f"Processing images in {self.split_name}"):
            base_name = img_path.stem.replace('_post_disaster', '')
            post_mask_path = self.post_mask_dir / f"{base_name}_post_disaster_target{img_path.suffix}"
            pre_img_path = self.pre_image_dir / f"{base_name}_pre_disaster{img_path.suffix}"

            if not (post_mask_path.exists() and pre_img_path.exists()):
                continue

            with Image.open(post_mask_path).convert('L') as mask:
                mask_np = np.array(mask)
                img_height, img_width = mask_np.shape
                unique_labels = np.unique(mask_np)
                invalid_labels = [lbl for lbl in unique_labels if lbl not in DAMAGE_LABELS.values()]
                if invalid_labels:
                    logging.warning(f"Invalid labels {invalid_labels} found in mask {post_mask_path}, skipping invalid labels")

                for label in DAMAGE_LABELS.values():
                    indices = np.argwhere(mask_np == label)
                    if len(indices) == 0:
                        continue
                    indices = indices[np.lexsort((indices[:, 1], indices[:, 0]))]
                    num_to_sample = min(len(indices), self.max_patches_per_class) if label != 4 else len(indices)
                    sampled_indices = indices[:num_to_sample]

                    for y, x in sampled_indices:
                        half_patch = self.patch_size // 2
                        top, left = max(0, y - half_patch), max(0, x - half_patch)
                        bottom, right = top + self.patch_size, left + self.patch_size

                        if bottom > img_height:
                            bottom, top = img_height, max(0, img_height - self.patch_size)
                        if right > img_width:
                            right, left = img_width, max(0, img_width - self.patch_size)

                        if bottom - top != self.patch_size or right - left != self.patch_size:
                            continue

                        coords = {'coords': {self.patch_size: (left, top, right, bottom)}}
                        metadata = (str(pre_img_path), str(img_path), coords['coords'], label)
                        self.patch_metadata.append(metadata)
                        self.metadata_by_class[label].append(metadata)

                # Log unique images per class
                for label in DAMAGE_LABELS.values():
                    unique_images = set(meta[1] for meta in self.metadata_by_class[label])
                    logging.info(f"Class {label} ({get_damage_name(label)}): {len(unique_images)} unique images, {len(self.metadata_by_class[label])} patches")

    def __len__(self):
        return len(self.patch_metadata)

    def __getitem__(self, idx):
        pre_path, post_path, coords, label = self.patch_metadata[idx]
        with Image.open(pre_path).convert('RGB') as pre_image, Image.open(post_path).convert('RGB') as post_image:
            left, top, right, bottom = [int(coord) for coord in coords[self.patch_size]]
            pre_patch = pre_image.crop((left, top, right, bottom))
            post_patch = post_image.crop((left, top, right, bottom))
            pre_transformed = self.transform(pre_patch)
            post_transformed = self.transform(post_patch)
            combined = torch.cat((pre_transformed, post_transformed), dim=0)
            return combined, label, pre_patch, post_patch

    def sample_episode(self, n_way: int, k_shot: int, q_query: int, class_counts: Dict[int, int], episode_num: int,
                       is_training: bool = False, class_counter: Counter = None):
        global skipped_episodes, same_image_episodes

        available_classes = [label for label, patches in self.metadata_by_class.items() if
                            len(patches) >= (k_shot + q_query)]
        logging.info(f"Episode {episode_num}: Available classes: {[get_damage_name(label) for label in available_classes]}, "
                     f"Class 4 patches: {len(self.metadata_by_class[4])}")

        if len(available_classes) < n_way:
            raise ValueError(f"Not enough classes ({len(available_classes)} available, need {n_way})")

        # Probability-based class selection
        valid_class_counts = {label: max(1, count) for label, count in class_counts.items()
                              if count > 0 and label in available_classes}
        prob_array = [1.0 / np.sqrt(valid_class_counts.get(label, 1)) if label in valid_class_counts else 1.0
                      for label in available_classes]
        total_prob = sum(prob_array)
        prob_array = [p / total_prob if total_prob > 0 else 1.0 / len(available_classes) for p in prob_array]
        logging.info(f"Episode {episode_num} Class probabilities: "
                     f"{[f'Class {label} ({get_damage_name(label)}): {p:.4f}' for label, p in zip(available_classes, prob_array)]}")

        # Force class 4 inclusion
        force_class = None
        if is_training and class_counter is not None and 4 in available_classes:
            recent_class_4 = class_counter[4] / max(1, sum(class_counter.values()))
            if recent_class_4 < 0.1 and episode_num % 25 == 0:
                force_class = 4
                logging.info(f"Episode {episode_num}: Forcing Class 4, recent proportion: {recent_class_4:.4f}")
        elif not is_training and 4 in available_classes:
            force_class = 4
            logging.info(f"Episode {episode_num}: Forcing Class 4 in test set")

        # Select classes
        max_attempts = 20
        for attempt in range(max_attempts):
            if force_class and force_class in available_classes:
                temp_classes = [force_class]
                remaining_classes = [c for c in available_classes if c != force_class]
                remaining_probs = [p for c, p in zip(available_classes, prob_array) if c != force_class]
                if remaining_probs and len(remaining_classes) >= (n_way - 1):
                    remaining_probs = np.array(remaining_probs) / sum(remaining_probs)
                    additional = np.random.choice(remaining_classes, size=n_way - 1, replace=False, p=remaining_probs)
                    temp_classes.extend(additional.tolist())
                    selected_classes = temp_classes
                    break
            else:
                try:
                    selected_classes = np.random.choice(available_classes, size=n_way, replace=False, p=prob_array).tolist()
                    break
                except ValueError:
                    selected_classes = random.sample(available_classes, n_way)
                    break
        else:
            selected_classes = random.sample(available_classes, min(n_way, len(available_classes)))

        episode_label_map = {i: orig_label for i, orig_label in enumerate(set(selected_classes))}
        original_to_episode_label = {v: k for k, v in episode_label_map.items()}
        logging.info(f"Episode {episode_num} selected classes: {[get_damage_name(label) for label in set(selected_classes)]}")

        support_images, query_images = [], []
        support_labels, query_labels = [], []
        support_patches, query_patches = [], []
        support_metadata, query_metadata = [], []

        for original_label in set(selected_classes):
            episode_label = original_to_episode_label[original_label]
            class_metadata = self.metadata_by_class[original_label]
            total_needed = k_shot + q_query

            if len(class_metadata) < total_needed:
                selected_metadata = class_metadata.copy()
                selected_metadata.extend(random.choices(class_metadata, k=total_needed - len(class_metadata)))
            else:
                selected_metadata = random.sample(class_metadata, total_needed)

            image_sources = defaultdict(list)
            for i, meta in enumerate(selected_metadata):
                _, post_path, _, _ = meta
                img_source = Path(post_path).stem
                image_sources[img_source].append((i, meta))

            image_keys = list(image_sources.keys())
            random.shuffle(image_keys)
            unique_images = len(image_keys)

            # Adjust k_shot and q_query based on unique images
            k_shot_adjusted = k_shot
            q_query_adjusted = q_query
            if unique_images < k_shot + q_query:
                if unique_images == 1:
                    k_shot_adjusted = 1
                    q_query_adjusted = 1
                else:
                    k_shot_adjusted = max(1, unique_images // 2)
                    q_query_adjusted = max(1, unique_images - k_shot_adjusted)
                logging.info(f"Class {original_label} ({get_damage_name(original_label)}): Adjusted k_shot={k_shot_adjusted}, q_query={q_query_adjusted} due to {unique_images} unique images")

            # Sample patches, ensuring no patch overlap between support and query
            support_indices = []
            query_indices = []
            if unique_images == 1 and k_shot_adjusted + q_query_adjusted <= len(class_metadata):
                # For single-image classes (e.g., class 4), sample distinct patches
                patch_indices = random.sample(range(len(selected_metadata)), k_shot_adjusted + q_query_adjusted)
                support_indices = patch_indices[:k_shot_adjusted]
                query_indices = patch_indices[k_shot_adjusted:k_shot_adjusted + q_query_adjusted]
                logging.info(f"Class {original_label} ({get_damage_name(original_label)}): Using {len(support_indices)} support and {len(query_indices)} query patches from single image")
            elif unique_images >= k_shot_adjusted + q_query_adjusted:
                # Sufficient unique images: assign disjoint images
                support_img_keys = image_keys[:k_shot_adjusted]
                query_img_keys = image_keys[k_shot_adjusted:k_shot_adjusted + q_query_adjusted]
                for img_key in support_img_keys:
                    for idx, _ in image_sources[img_key]:
                        if len(support_indices) < k_shot_adjusted:
                            support_indices.append(idx)
                for img_key in query_img_keys:
                    for idx, _ in image_sources[img_key]:
                        if len(query_indices) < q_query_adjusted:
                            query_indices.append(idx)
            else:
                # Allow same-image sampling with relaxed threshold
                same_image_ratio = same_image_episodes[original_label] / (episode_num + 1) if episode_num > 0 else 0.0
                if same_image_ratio < 0.5:
                    logging.warning(f"Class {original_label} ({get_damage_name(original_label)}): Allowing same-image sampling "
                                    f"(ratio={same_image_ratio:.4f}, unique_images={unique_images}, needed={k_shot_adjusted + q_query_adjusted})")
                    same_image_episodes[original_label] += 1
                    support_indices = random.sample(range(len(selected_metadata)), k_shot_adjusted)
                    remaining = [i for i in range(len(selected_metadata)) if i not in support_indices]
                    query_indices = random.sample(remaining, min(q_query_adjusted, len(remaining)))
                else:
                    logging.warning(f"Episode {episode_num}: Skipping due to insufficient unique images for class {original_label} "
                                    f"(unique_images={unique_images}, needed={k_shot_adjusted + q_query_adjusted}, same_image_ratio={same_image_ratio:.4f})")
                    skipped_episodes += 1
                    return None

            if not support_indices or not query_indices:
                logging.warning(f"Episode {episode_num}: No valid support or query indices for class {original_label}")
                skipped_episodes += 1
                return None

            metadata_to_idx = {id(metadata): idx for idx, metadata in enumerate(self.patch_metadata)}
            support_metadata.extend([selected_metadata[i] for i in support_indices])
            query_metadata.extend([selected_metadata[i] for i in query_indices])

            for meta in support_metadata[-k_shot_adjusted:]:
                idx = metadata_to_idx.get(id(meta))
                if idx is not None:
                    combined, _, pre_patch, post_patch = self[idx]
                    support_images.append(combined)
                    support_labels.append(episode_label)
                    support_patches.append((pre_patch, post_patch))
            for meta in query_metadata[-q_query_adjusted:]:
                idx = metadata_to_idx.get(id(meta))
                if idx is not None:
                    combined, _, pre_patch, post_patch = self[idx]
                    query_images.append(combined)
                    query_labels.append(episode_label)
                    query_patches.append((pre_patch, post_patch))

        if not support_images or not query_images:
            logging.warning(f"Empty episode {episode_num}, total skipped: {skipped_episodes + 1}")
            skipped_episodes += 1
            return None

        support_image_sources = set(meta[1] for meta in support_metadata)
        query_image_sources = set(meta[1] for meta in query_metadata)
        if support_image_sources & query_image_sources:
            logging.warning(f"Episode {episode_num}: Overlap in support and query images: {support_image_sources & query_image_sources}")

        xs = torch.stack(support_images)
        xq = torch.stack(query_images)
        ys = torch.tensor(support_labels, dtype=torch.long)
        yq = torch.tensor(query_labels, dtype=torch.long)

        episode_support = Counter(yq.tolist())
        valid_labels = all(k in episode_label_map for k in episode_support)
        if not valid_labels:
            logging.warning(f"Invalid labels in episode {episode_num}: {episode_support}")
            skipped_episodes += 1
            return None

        logging.info(f"Episode {episode_num} query distribution: "
                     f"{[f'Class {get_damage_name(episode_label_map.get(k, -1))} (Episode {k}): {v}' for k, v in episode_support.items()]}")

        episode_idx_to_original_label_map = {v: k for k, v in original_to_episode_label.items()}
        return {
            'xs': xs, 'xq': xq, 'ys': ys, 'yq': yq,
            'original_classes': list(set(selected_classes)),
            'episode_original_to_episode_label_map': original_to_episode_label,
            'episode_idx_to_original_label_map': episode_idx_to_original_label_map,
            'support_patches': support_patches,
            'query_patches': query_patches,
            'support_metadata': support_metadata,
            'query_metadata': query_metadata
        }

# --- Training Function ---
def train_protonet_with_patches(model, train_dataset, val_dataset, n_way, k_shot, q_query, num_episodes=400,
                                device='cpu', class_counts=None, gamma_base=2.0, label_smoothing=0.05, patience=2, config=None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    scaler = GradScaler() if device == 'cuda' else None
    best_val_acc = -1.0
    best_model_state = None
    no_improve_count = 0

    log_file = 'training_logs.csv'
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc',
                         'Val_Macro_Precision', 'Val_Macro_Recall', 'Val_Macro_F1',
                         'Val_Class1_Acc', 'Val_Class2_Acc', 'Val_Class3_Acc', 'Val_Class4_Acc',
                         'Val_Class1_Support', 'Val_Class2_Support', 'Val_Class3_Support', 'Val_Class4_Support'])

    episode_class_counts = Counter()
    logging.info("Generating fixed validation episodes for consistent evaluation...")
    fixed_val_episodes = []
    val_class_counter = Counter()

    for i in range(30):
        val_episode = val_dataset.sample_episode(n_way, k_shot, q_query, class_counts, i,
                                                 is_training=False, class_counter=val_class_counter)
        if val_episode is not None:
            fixed_val_episodes.append(val_episode)
            for label in val_episode['original_classes']:
                if label in DAMAGE_LABELS.values():
                    val_class_counter[label] += 1

    logging.info(f"Generated {len(fixed_val_episodes)} fixed validation episodes")
    logging.info(f"Validation class distribution: {dict(val_class_counter)}")

    def focal_loss(logits, targets, num_classes, class_counts=None, gamma=gamma_base, ls=label_smoothing):
        log_probs = F.log_softmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - target_probs) ** gamma
        if class_counts is not None:
            total_samples = sum(class_counts.values())
            class_weights = torch.tensor([total_samples / (num_classes * class_counts.get(t.item(), 1))
                                          for t in targets], device=logits.device)
            focal_weight = focal_weight * class_weights
        loss = -focal_weight * target_log_probs
        if ls > 0:
            smooth_loss = -log_probs.mean(dim=1)
            loss = (1 - ls) * loss + ls * smooth_loss
        return loss.mean()

    def mixup_query_only(x, y, alpha=0.2):
        if alpha <= 0 or x.size(0) < 2:
            return x, y, y, 1.0
        batch_size = x.size(0)
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def evaluate_fixed_validation_episodes(model, episodes, device):
        model.eval()
        total_loss = 0.0
        all_predicted = []
        all_true = []
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for episode_data in episodes:
                if episode_data is None:
                    continue
                xs, xq = episode_data['xs'].to(device), episode_data['xq'].to(device)
                ys, yq = episode_data['ys'].to(device), episode_data['yq'].to(device)
                support_features = model(xs)
                query_features = model(xq)
                prototypes = model.calculate_prototypes(support_features, ys)
                distances = model.compute_distances(query_features, prototypes)
                num_classes = len(set(episode_data['original_classes']))
                loss = focal_loss(-distances, yq, num_classes, class_counts)
                total_loss += loss.item()
                predicted = torch.argmin(distances, dim=1)
                all_predicted.extend(predicted.cpu().numpy())
                all_true.extend(yq.cpu().numpy())
                for pred, true in zip(predicted.cpu().numpy(), yq.cpu().numpy()):
                    class_total[true] += 1
                    if pred == true:
                        class_correct[true] += 1

        avg_loss = total_loss / len(episodes) if episodes else 0.0
        overall_acc = accuracy_score(all_true, all_predicted) if all_true else 0.0
        class_accuracies = {class_idx: class_correct[class_idx] / class_total[class_idx] if class_total[class_idx] > 0 else 0.0
                            for class_idx in class_total}
        return {
            'loss': avg_loss,
            'accuracy': overall_acc,
            'class_accuracies': class_accuracies,
            'class_support': dict(class_total)
        }

    for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
        model.train()
        episode_data = train_dataset.sample_episode(n_way, k_shot, q_query, class_counts, episode,
                                                    is_training=True, class_counter=episode_class_counts)
        if episode_data is None:
            continue
        xs, xq = episode_data['xs'].to(device), episode_data['xq'].to(device)
        ys, yq = episode_data['ys'].to(device), episode_data['yq'].to(device)
        for label in episode_data['original_classes']:
            if label in DAMAGE_LABELS.values():
                episode_class_counts[label] += 1

        xs_clean = xs
        xq_mixed, yq_a, yq_b, lam_q = mixup_query_only(xq, yq, alpha=0.2)
        optimizer.zero_grad()

        if device == 'cuda' and scaler is not None:
            with autocast('cuda'):
                support_features = model(xs_clean)
                query_features = model(xq_mixed)
                prototypes = model.calculate_prototypes(support_features, ys)
                distances = model.compute_distances(query_features, prototypes)
                num_classes = len(set(episode_data['original_classes']))
                loss = lam_q * focal_loss(-distances, yq_a, num_classes, class_counts) + \
                       (1 - lam_q) * focal_loss(-distances, yq_b, num_classes, class_counts)
                se_l2_loss = model.get_se_l2_loss()
                total_loss = loss + se_l2_loss
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            support_features = model(xs_clean)
            query_features = model(xq_mixed)
            prototypes = model.calculate_prototypes(support_features, ys)
            distances = model.compute_distances(query_features, prototypes)
            num_classes = len(set(episode_data['original_classes']))
            loss = lam_q * focal_loss(-distances, yq_a, num_classes, class_counts) + \
                   (1 - lam_q) * focal_loss(-distances, yq_b, num_classes, class_counts)
            se_l2_loss = model.get_se_l2_loss()
            total_loss = loss + se_l2_loss
            total_loss.backward()
            optimizer.step()

        scheduler.step()

        if episode % 25 == 0 and val_dataset and fixed_val_episodes:
            logging.info(f"Evaluating on {len(fixed_val_episodes)} fixed validation episodes...")
            val_results = evaluate_fixed_validation_episodes(model, fixed_val_episodes, device)
            val_acc = val_results['accuracy']
            val_loss = val_results['loss']
            logging.info(f"Episode {episode}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
            logging.info(f"  Class accuracies: {val_results['class_accuracies']}")
            logging.info(f"  Class support: {val_results['class_support']}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                logging.info(f"Updated best model with validation accuracy: {best_val_acc:.4f}")
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                logging.info(f"Early stopping at episode {episode}")
                break

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                class_accs = val_results['class_accuracies']
                class_support = val_results['class_support']
                writer.writerow([
                    episode, total_loss.item(), 0.0,
                    val_loss, val_acc, 0.0, 0.0, 0.0,
                    class_accs.get(0, 0.0), class_accs.get(1, 0.0),
                    class_accs.get(2, 0.0), class_accs.get(3, 0.0),
                    class_support.get(0, 0), class_support.get(1, 0),
                    class_support.get(2, 0), class_support.get(3, 0)
                ])

    torch.save({
        'model_state_dict': model.state_dict(),
        'best_val_acc': best_val_acc,
        'episode': episode,
        'final_class_counts': dict(episode_class_counts),
        'val_class_counts': dict(val_class_counter),
        'config': config if 'config' in globals() else {}
    }, 'trained_model.pth')

    if best_model_state:
        model.load_state_dict(best_model_state)
        logging.info(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
    return model

# --- Evaluation Function ---
def evaluate_model(model, dataset, num_episodes=100, n_way=3, k_shot=10, q_query=5, device='cpu', class_counts=None,
                   start_episode=1, vis_dir='visualizations'):
    model.eval()
    all_accuracies = []
    inference_times = []
    class_correct_counts = {label: 0 for label in DAMAGE_LABELS.values()}
    class_total_counts = {label: 0 for label in DAMAGE_LABELS.values()}
    all_true_labels = []
    all_pred_labels = []
    episode_class_counts = Counter()
    episode_confusion_counts = {label: Counter() for label in DAMAGE_LABELS.values()}
    all_heatmaps = []
    all_prototype_patches = []
    heatmap_variances = []
    prototype_distances = {label: [] for label in DAMAGE_LABELS.values()}
    global_image_count = 0

    os.makedirs(vis_dir, exist_ok=True)

    def save_visualization(heatmap, patch, filename, label, predicted, is_pre=False):
        if label not in DAMAGE_LABELS.values() or predicted not in DAMAGE_LABELS.values():
            logging.warning(f"Skipping visualization for invalid label {label} or predicted {predicted}")
            return
        plt.figure(figsize=(6, 6))
        patch_np = np.array(patch.convert('RGB')) / 255.0
        heatmap_resized = np.array(Image.fromarray(heatmap).resize((patch_np.shape[1], patch_np.shape[0])))
        plt.imshow(patch_np)
        plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)
        plt.title(f"{'Pre' if is_pre else 'Post'}-Disaster Patch\nTrue: {get_damage_name(label)}, Pred: {get_damage_name(predicted)}")
        plt.axis('off')
        plt.savefig(os.path.join(vis_dir, filename), bbox_inches='tight')
        plt.close()

    for episode in tqdm(range(start_episode, start_episode + num_episodes), desc="Evaluating"):
        episode_data = dataset.sample_episode(n_way, k_shot, q_query, class_counts, episode, is_training=False,
                                              class_counter=episode_class_counts)
        if episode_data is None:
            logging.info(f"Skipping episode {episode} due to invalid data")
            continue
        xs, xq = episode_data['xs'].to(device), episode_data['xq'].to(device)
        ys, yq = episode_data['ys'].to(device), episode_data['yq'].to(device)
        support_patches = episode_data['support_patches']
        query_patches = episode_data['query_patches']
        original_classes = episode_data['original_classes']
        episode_idx_to_original_label = episode_data['episode_idx_to_original_label_map']

        for label in original_classes:
            if label in DAMAGE_LABELS.values():
                episode_class_counts[label] += 1
            else:
                logging.warning(f"Invalid class label {label} in episode {episode}")

        start_time = time.time()
        with torch.no_grad():
            support_features = model(xs)
            query_features = model(xq)
            prototypes = model.calculate_prototypes(support_features, ys)
            distances = model.compute_distances(query_features, prototypes)
            _, predicted = torch.min(distances, 1)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        accuracy = (predicted == yq).float().mean().item()
        all_accuracies.append(accuracy)

        for i in range(xq.size(0)):
            global_image_count += 1
            query_input = xq[i:i + 1].requires_grad_(True)
            heatmap = model.compute_gradcam(query_input, predicted[i].item(), device=device)
            all_heatmaps.append(heatmap)
            heatmap_var = np.var(heatmap)
            heatmap_variances.append(heatmap_var)
            pre_patch, post_patch = query_patches[i]
            true_original_label_viz = episode_idx_to_original_label.get(yq[i].item(), -1)
            pred_original_label_viz = episode_idx_to_original_label.get(predicted[i].item(), -1)
            if global_image_count % 50 == 0 and true_original_label_viz in DAMAGE_LABELS.values() and pred_original_label_viz in DAMAGE_LABELS.values():
                save_visualization(heatmap, pre_patch, f'ep{episode}_query{i}_pre.png', true_original_label_viz,
                                   pred_original_label_viz, is_pre=True)
                save_visualization(heatmap, post_patch, f'ep{episode}_query{i}_post.png', true_original_label_viz,
                                   pred_original_label_viz, is_pre=False)

        with torch.no_grad():
            for label in torch.unique(ys):
                mask = (ys == label)
                class_features = support_features[mask]
                if class_features.size(0) == 0:
                    continue
                proto = prototypes[label]
                dists = torch.sqrt(torch.sum((class_features - proto) ** 2, dim=1)).cpu().numpy()
                original_label = episode_idx_to_original_label.get(label.item(), -1)
                if original_label not in DAMAGE_LABELS.values():
                    logging.warning(f"Invalid original label {original_label} for episode label {label} in episode {episode}")
                    continue
                prototype_distances[original_label].extend(dists.tolist())

            proto_patches = model.get_prototype_patches(support_features, ys, support_patches, num_top=3)
            for label, patches in proto_patches.items():
                for j, (pre_patch, post_patch) in enumerate(patches):
                    plt.figure(figsize=(6, 6))
                    plt.imshow(np.array(post_patch.convert('RGB')) / 255.0)
                    plt.title(f"Prototype Patch for {get_damage_name(label)}")
                    plt.axis('off')
                    plt.savefig(os.path.join(vis_dir, f'ep{episode}_proto_label{label}_patch{j}.png'), bbox_inches='tight')
                    plt.close()
                all_prototype_patches.append((label, patches))

            unique_classes = list(dict.fromkeys(original_classes))
            episode_to_original_label = {i: orig_label for i, orig_label in enumerate(unique_classes)}
            for i in range(len(yq)):
                true_episode_label = yq[i].item()
                pred_episode_label = predicted[i].item()
                true_original_label = episode_to_original_label.get(true_episode_label, -1)
                pred_original_label = episode_to_original_label.get(pred_episode_label, -1)
                if pred_original_label == -1 or true_original_label == -1:
                    logging.warning(f"Invalid label mapping in episode {episode}: true={true_episode_label}, pred={pred_episode_label}")
                    continue
                class_total_counts[true_original_label] += 1
                if pred_episode_label == true_episode_label:
                    class_correct_counts[true_original_label] += 1
                all_true_labels.append(true_original_label)
                all_pred_labels.append(pred_original_label)
                episode_confusion_counts[true_original_label][pred_original_label] += 1

        logging.info(f"Episode {episode} Confusion: "
                     f"{[f'True {get_damage_name(t)} -> {dict(c)}' for t, c in episode_confusion_counts.items() if c]}")

    logging.info(f"Evaluation Class Frequencies: {dict(episode_class_counts)}")
    overall_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
    avg_inference_time = np.mean(inference_times) if inference_times else 0.0
    class_accuracies = {label: class_correct_counts[label] / class_total_counts[label] if class_total_counts[label] > 0 else 0.0
                        for label in DAMAGE_LABELS.values()}

    precision, recall, f1, support = precision_recall_fscore_support(
        all_true_labels, all_pred_labels, labels=list(DAMAGE_LABELS.values()), average=None, zero_division=0
    )
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=list(DAMAGE_LABELS.values()))

    if all_true_labels:
        accuracies = []
        for _ in range(1000):
            boot_true, boot_pred = resample(all_true_labels, all_pred_labels)
            acc = accuracy_score(boot_true, boot_pred)
            accuracies.append(acc)
        acc_ci = np.percentile(accuracies, [2.5, 97.5])
        logging.info(f"95% CI for accuracy: {acc_ci}")
    else:
        acc_ci = [0.0, 0.0]

    class_metrics = {
        label: {
            'accuracy': class_accuracies[label],
            'precision': precision[idx],
            'recall': recall[idx],
            'f1': f1[idx],
            'support': support[idx]
        }
        for idx, label in enumerate(DAMAGE_LABELS.values())
    }

    avg_heatmap_variance = np.mean(heatmap_variances) if heatmap_variances else 0.0
    prototype_distance_stats = {
        label: {
            'mean': np.mean(dists) if dists else 0.0,
            'std': np.std(dists) if dists else 0.0
        }
        for label, dists in prototype_distances.items()
    }

    return {
        'overall_accuracy': overall_accuracy,
        'avg_inference_time': avg_inference_time,
        'class_accuracies': class_accuracies,
        'class_metrics': class_metrics,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'confusion_matrix': cm.tolist(),
        'heatmaps': all_heatmaps,
        'prototype_patches': all_prototype_patches,
        'heatmap_variance': avg_heatmap_variance,
        'prototype_distances': prototype_distance_stats,
        'class_frequencies': dict(episode_class_counts),
        'accuracy_ci': acc_ci
    }

# --- Ablation Study ---
def run_ablation_study(model, dataset, num_episodes=100, n_way=3, k_shot=10, q_query=5, device='cpu', class_counts=None,
                       vis_dir='visualizations'):
    logging.info("Evaluating SE-enhanced ProtoNet...")
    se_results = evaluate_model(model, dataset, num_episodes, n_way, k_shot, q_query, device, class_counts,
                                start_episode=1, vis_dir=vis_dir)

    class BaselineProtoNet(nn.Module):
        def __init__(self, input_channels=6, feature_dim=64):
            super(BaselineProtoNet, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
            self.fc = nn.Linear(64 * 8 * 8, feature_dim)
            self.feature_dim = feature_dim
            self.last_conv = self.encoder[12]
            self.gradients = None
            self.activations = None

        def save_gradient(self, grad):
            self.gradients = grad

        def forward(self, x, return_features=False):
            feature_maps = self.encoder(x)
            pooled_features = self.adaptive_pool(feature_maps)
            features = pooled_features.view(pooled_features.size(0), -1)
            features = self.fc(features)
            if return_features:
                return features, feature_maps
            return features

        def calculate_prototypes(self, features, labels):
            unique_labels = torch.unique(labels)
            feature_dim = features.shape[1]
            device = features.device
            prototype_list = []
            for label_val_tensor in unique_labels:
                label_val = label_val_tensor.item()
                mask = (labels == label_val_tensor)
                class_features = features[mask]
                if class_features.shape[0] == 0:
                    logging.warning(f"Empty class for label {label_val} in prototype calculation")
                    prototype_list.append(torch.zeros(feature_dim, device=device, requires_grad=False))
                    continue
                prototype = class_features.mean(dim=0)
                prototype_list.append(prototype)
            if not prototype_list:
                raise ValueError("No prototypes computed; all classes in support set were empty.")
            return torch.stack(prototype_list, dim=0)

        def compute_distances(self, query_features, prototypes):
            diff = query_features.unsqueeze(1) - prototypes.unsqueeze(0)
            distances = torch.sum(diff ** 2, dim=2)
            return distances

        def compute_gradcam(self, x, target_class, device='cpu'):
            was_training = self.training
            self.eval()
            x = x.to(device)
            x.requires_grad_(True)
            self.gradients = None
            self.activations = None

            def forward_hook(module, input, output):
                self.activations = output

            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0]

            forward_handle = self.last_conv.register_forward_hook(forward_hook)
            backward_handle = self.last_conv.register_backward_hook(backward_hook)

            try:
                features, _ = self.forward(x, return_features=True)
                self.zero_grad()
                score = features[0, target_class]
                score.backward()
                if self.gradients is None or self.activations is None:
                    logging.warning("Gradients or activations not captured")
                    return np.zeros((x.size(2), x.size(3)))
                pooled_gradients = torch.mean(self.gradients, dim=[2, 3])
                for i in range(self.activations.size(1)):
                    self.activations[0, i] *= pooled_gradients[0, i]
                heatmap = torch.mean(self.activations[0], dim=0)
                heatmap = F.relu(heatmap)
                heatmap /= torch.max(heatmap) + 1e-8
                heatmap = heatmap.detach().cpu().numpy()
            except Exception as e:
                logging.error(f"Grad-CAM failed: {str(e)}")
                heatmap = np.zeros((x.size(2), x.size(3)))
            finally:
                forward_handle.remove()
                backward_handle.remove()
            if was_training:
                self.train()
            return heatmap

        def get_prototype_patches(self, support_features, support_labels, support_patches, num_top=3):
            prototypes = self.calculate_prototypes(support_features, support_labels)
            prototype_patches = {label.item(): [] for label in torch.unique(support_labels)}
            for i, label in enumerate(torch.unique(support_labels)):
                label_val = label.item()
                mask = (support_labels == label)
                class_features = support_features[mask]
                class_patches = [support_patches[j] for j in range(len(support_patches)) if mask[j]]
                if len(class_features) == 0:
                    continue
                prototype = prototypes[i]
                distances = torch.sum((class_features - prototype) ** 2, dim=1)
                _, top_indices = torch.topk(distances, k=min(num_top, len(distances)), largest=False)
                prototype_patches[label_val] = [class_patches[j] for j in top_indices]
            return prototype_patches

    baseline_model = BaselineProtoNet(input_channels=6, feature_dim=64).to(device)
    model_dict = model.state_dict()
    baseline_dict = baseline_model.state_dict()
    se_to_baseline_mapping = {
        'encoder.0.weight': 'encoder.0.weight',
        'encoder.0.bias': 'encoder.0.bias',
        'encoder.5.weight': 'encoder.4.weight',
        'encoder.5.bias': 'encoder.4.bias',
        'encoder.10.weight': 'encoder.8.weight',
        'encoder.10.bias': 'encoder.8.bias',
        'encoder.15.weight': 'encoder.12.weight',
        'encoder.15.bias': 'encoder.12.bias',
        'encoder.1.weight': 'encoder.1.weight',
        'encoder.1.bias': 'encoder.1.bias',
        'encoder.1.running_mean': 'encoder.1.running_mean',
        'encoder.1.running_var': 'encoder.1.running_var',
        'encoder.6.weight': 'encoder.5.weight',
        'encoder.6.bias': 'encoder.5.bias',
        'encoder.6.running_mean': 'encoder.5.running_mean',
        'encoder.6.running_var': 'encoder.5.running_var',
        'encoder.11.weight': 'encoder.9.weight',
        'encoder.11.bias': 'encoder.9.bias',
        'encoder.11.running_mean': 'encoder.9.running_mean',
        'encoder.11.running_var': 'encoder.9.running_var',
        'encoder.16.weight': 'encoder.13.weight',
        'encoder.16.bias': 'encoder.13.bias',
        'encoder.16.running_mean': 'encoder.13.running_mean',
        'encoder.16.running_var': 'encoder.13.running_var',
        'fc.weight': 'fc.weight',
        'fc.bias': 'fc.bias'
    }
    for se_key, baseline_key in se_to_baseline_mapping.items():
        if se_key in model_dict and baseline_key in baseline_dict:
            baseline_dict[baseline_key].copy_(model_dict[se_key])
    baseline_model.load_state_dict(baseline_dict)

    logging.info("Evaluating baseline ProtoNet (no SE blocks)...")
    baseline_results = evaluate_model(baseline_model, dataset, num_episodes, n_way, k_shot, q_query, device,
                                      class_counts, start_episode=num_episodes + 1, vis_dir=vis_dir)
    return {
        'se_results': se_results,
        'baseline_results': baseline_results
    }

def get_damage_name(label_id: int) -> str:
    for name, id_val in DAMAGE_LABELS.items():
        if id_val == label_id:
            return name.replace('-', ' ').title()
    return f"Unknown ({label_id})"

def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    config = {
        'dataset': {
            'root_dir': r"C:\Users\joshp_ya\PycharmProjects\earthquake",
            'patch_size': 64,
            'max_patches_per_class': 100
        },
        'model': {
            'input_channels': 6,
            'feature_dim': 64
        },
        'training': {
            'n_way': 3,
            'k_shot': 5,
            'q_query': 5,
            'num_episodes': 100,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'gamma_base': 2.0,
            'label_smoothing': 0.05,
            'patience': 2
        },
        'evaluation': {
            'num_episodes': 100,
            'vis_dir': 'visualizations'
        }
    }

    xbd_root = Path(config['dataset']['root_dir'])
    patch_size = config['dataset']['patch_size']
    feature_dim = config['model']['feature_dim']
    n_way = config['training']['n_way']
    k_shot = config['training']['k_shot']
    q_query = config['training']['q_query']
    validation_split_ratio = 0.2
    device = config['training']['device']
    max_patches_per_class_extraction = config['dataset']['max_patches_per_class']
    num_episodes = config['training']['num_episodes']
    gamma_base = config['training']['gamma_base']
    label_smoothing = config['training']['label_smoothing']
    patience = config['training']['patience']

    class_counts = {1: 3600, 2: 750, 3: 270, 4: 60}

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    logging.info("\n--- Loading Datasets ---")
    full_train_dataset = XBDPatchDataset(root_dir=xbd_root, split='train', patch_size=patch_size,
                                         transform=train_transforms, max_patches_per_class=max_patches_per_class_extraction,
                                         device=device)
    test_dataset = XBDPatchDataset(root_dir=xbd_root, split='test', patch_size=patch_size,
                                   transform=eval_transforms, max_patches_per_class=max_patches_per_class_extraction,
                                   device=device)

    logging.info("\n--- Dataset Class Distribution ---")
    for label in DAMAGE_LABELS.values():
        logging.info(f"Class {label} ({get_damage_name(label)}): {len(full_train_dataset.metadata_by_class[label])} patches")

    train_subset_patches_by_class = {label: [] for label in DAMAGE_LABELS.values()}
    val_subset_patches_by_class = {label: [] for label in DAMAGE_LABELS.values()}
    for label, metadata_list in full_train_dataset.metadata_by_class.items():
        if not metadata_list:
            continue
        random.shuffle(metadata_list)
        split_idx = int(len(metadata_list) * (1.0 - validation_split_ratio))
        train_subset_patches_by_class[label].extend(metadata_list[:split_idx])
        val_subset_patches_by_class[label].extend(metadata_list[split_idx:])

    train_subset_patches = [patch for label in DAMAGE_LABELS.values() for patch in train_subset_patches_by_class[label]]
    val_subset_patches = [patch for label in DAMAGE_LABELS.values() for patch in val_subset_patches_by_class[label]]

    train_dataset = XBDPatchDataset(root_dir=xbd_root, split='train_subset', patch_size=patch_size,
                                    transform=train_transforms, skip_extraction=True,
                                    initial_patches=train_subset_patches,
                                    initial_patches_by_class=train_subset_patches_by_class, device=device)
    val_dataset = XBDPatchDataset(root_dir=xbd_root, split='val_from_train', patch_size=patch_size,
                                  transform=eval_transforms, skip_extraction=True,
                                  initial_patches=val_subset_patches,
                                  initial_patches_by_class=val_subset_patches_by_class, device=device)

    logging.info("\n--- Initializing Model ---")
    model = ProtoNet(input_channels=config['model']['input_channels'], feature_dim=feature_dim).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Created ProtoNet with SE-enhanced CNN backbone, {model_params:,} trainable parameters")

    checkpoint_path = 'trained_model.pth'
    if os.path.exists(checkpoint_path):
        logging.info(f"\n--- Found Checkpoint at {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            best_val_acc = checkpoint.get('best_val_acc', -1.0)
            last_episode = checkpoint.get('episode', 0)
            logging.info(f"Checkpoint Details: Best Validation Accuracy = {best_val_acc:.4f}, Last Episode = {last_episode}")
        else:
            model.load_state_dict(checkpoint)
            logging.info("Loaded legacy checkpoint (no metadata available).")
            best_val_acc = -1.0
            last_episode = 0
        model = model.to(device)
        retrain = input("Checkpoint exists. Retrain from scratch? (yes/no, default=no): ").lower() == 'yes'
        if retrain:
            logging.info("\n--- Retraining Model from Scratch ---")
            model = train_protonet_with_patches(model, train_dataset, val_dataset, n_way, k_shot, q_query,
                                                num_episodes=num_episodes, device=device, class_counts=class_counts,
                                                gamma_base=gamma_base, label_smoothing=label_smoothing, patience=patience, config=config)
            logging.info(f"\n--- Saved Retrained Model to {checkpoint_path} ---")
        else:
            logging.info("\n--- Skipping Training, Using Existing Checkpoint ---")
    else:
        logging.info("\n--- No Checkpoint Found, Training Model ---")
        model = train_protonet_with_patches(model, train_dataset, val_dataset, n_way, k_shot, q_query,
                                            num_episodes=num_episodes, device=device, class_counts=class_counts,
                                            gamma_base=gamma_base, label_smoothing=label_smoothing, patience=patience, config=config)
        logging.info(f"\n--- Saved Trained Model to {checkpoint_path} ---")

    logging.info("\n--- Final Evaluation on Test Set ---")
    test_class_counts = {label: len(test_dataset.metadata_by_class[label]) for label in DAMAGE_LABELS.values()}
    logging.info(f"Natural test class distribution: {test_class_counts}")
    ablation_results = run_ablation_study(model, test_dataset, num_episodes=config['evaluation']['num_episodes'],
                                          n_way=n_way, k_shot=k_shot, q_query=q_query, device=device,
                                          class_counts=test_class_counts, vis_dir=config['evaluation']['vis_dir'])
    test_results = ablation_results['se_results']
    baseline_results = ablation_results['baseline_results']

    logging.info("\n--- SE-Enhanced ProtoNet Results ---")
    logging.info(f"Test Overall Accuracy: {test_results['overall_accuracy']:.4f}")
    logging.info(f"95% CI for Accuracy: {test_results['accuracy_ci']}")
    logging.info(f"Average Inference Time per Episode: {test_results['avg_inference_time']:.4f} seconds")
    logging.info(f"Macro Precision: {test_results['macro_precision']:.4f}")
    logging.info(f"Macro Recall: {test_results['macro_recall']:.4f}")
    logging.info(f"Macro F1 Score: {test_results['macro_f1']:.4f}")
    logging.info(f"Average Heatmap Variance: {test_results['heatmap_variance']:.4f}")
    logging.info("\nPer-Class Metrics:")
    for label, metrics in test_results['class_metrics'].items():
        logging.info(f"Class {label} ({get_damage_name(label)}):")
        logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"  Precision: {metrics['precision']:.4f}")
        logging.info(f"  Recall: {metrics['recall']:.4f}")
        logging.info(f"  F1 Score: {metrics['f1']:.4f}")
        logging.info(f"  Support: {metrics['support']}")
        logging.info(f"  Prototype Distance (Mean ± Std): {test_results['prototype_distances'][label]['mean']:.4f} ± "
                     f"{test_results['prototype_distances'][label]['std']:.4f}")
    logging.info("\nConfusion Matrix:")
    logging.info("(Rows: True Labels, Columns: Predicted Labels)")
    logging.info("Classes: [No Damage, Minor Damage, Major Damage, Destroyed]")
    logging.info(np.array(test_results['confusion_matrix']))
    logging.info("\nClass Frequencies: " + str(test_results['class_frequencies']))

    logging.info("\n--- Baseline ProtoNet Results (No SE Blocks) ---")
    logging.info(f"Test Overall Accuracy: {baseline_results['overall_accuracy']:.4f}")
    logging.info(f"Average Inference Time per Episode: {baseline_results['avg_inference_time']:.4f} seconds")
    logging.info(f"Macro F1 Score: {baseline_results['macro_f1']:.4f}")

if __name__ == "__main__":
    main()