import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import matplotlib.pyplot as plt


class ProtoNet(nn.Module):
    def __init__(self, input_channels: int = 3, hidden_dim: int = 64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            self._conv_block(input_channels, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
        )

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def calculate_prototypes(self, support_features: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        classes = torch.unique(support_labels)
        n_classes = len(classes)
        prototypes = torch.zeros(n_classes, support_features.shape[1], device=support_features.device)
        for i, c in enumerate(classes):
            mask = support_labels == c
            prototypes[i] = support_features[mask].mean(0)
        return prototypes

    def compute_distances(self, query_features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        n_query = query_features.shape[0]
        n_prototypes = prototypes.shape[0]
        query_features = query_features.unsqueeze(1).expand(n_query, n_prototypes, -1)
        prototypes = prototypes.unsqueeze(0).expand(n_query, n_prototypes, -1)
        return torch.sum((query_features - prototypes) ** 2, dim=2)


class XBDMaskDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', use_both_images: bool = True,
                 transform: Optional[transforms.Compose] = None, extension: str = '*.png'):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.use_both_images = use_both_images
        self.extension = extension

        self.pre_image_dir = self.root_dir / 'img_pre'
        self.post_image_dir = self.root_dir / 'img_post'
        self.pre_mask_dir = self.root_dir / 'gt_pre'
        self.post_mask_dir = self.root_dir / 'gt_post'

        if not self.post_image_dir.exists() or not self.post_mask_dir.exists():
            raise FileNotFoundError(f"Required directories not found in {self.root_dir}")

        self.samples = []

        post_images = list(self.post_image_dir.glob(self.extension))
        post_masks = list(self.post_mask_dir.glob(self.extension))
        pre_images = list(self.pre_image_dir.glob(self.extension)) if self.use_both_images else []
        pre_masks = list(self.pre_image_dir.glob(self.extension)) if self.use_both_images else []

        print(f"[{split}] Post images: {len(post_images)} (e.g., {post_images[:2] if post_images else 'None'})")
        print(f"[{split}] Post masks: {len(post_masks)} (e.g., {post_masks[:2] if post_masks else 'None'})")
        if self.use_both_images:
            print(f"[{split}] Pre images: {len(pre_images)} (e.g., {pre_images[:2] if pre_images else 'None'})")
            print(f"[{split}] Pre masks: {len(pre_masks)} (e.g., {pre_masks[:2] if pre_masks else 'None'})")

        for img_path in post_images:
            base_name = img_path.stem.replace('_post_disaster', '')
            post_mask_path = self.post_mask_dir / f"{base_name}_post_disaster_target{img_path.suffix}"

            if post_mask_path.exists():
                if self.use_both_images:
                    pre_img_path = self.pre_image_dir / f"{base_name}_pre_disaster{img_path.suffix}"
                    pre_mask_path = self.pre_mask_dir / f"{base_name}_pre_disaster_target{img_path.suffix}"

                    if pre_img_path.exists() and pre_mask_path.exists():
                        self.samples.append({
                            'post_img': str(img_path),
                            'post_mask': str(post_mask_path),
                            'pre_img': str(pre_img_path),
                            'pre_mask': str(pre_mask_path)
                        })
                    else:
                        print(f"[{split}] Missing pre image or mask for {base_name}")
                else:
                    self.samples.append({
                        'post_img': str(img_path),
                        'post_mask': str(post_mask_path)
                    })

        print(f"[{split}] Valid samples: {len(self.samples)}")

        damage_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for sample in self.samples:
            post_mask = Image.open(sample['post_mask']).convert('L')  # Load as grayscale
            post_mask_np = np.array(post_mask)
            label = self.get_damage_label_from_mask(post_mask_np)
            damage_counts[label] += 1
        print(f"[{split}] Damage distribution: {damage_counts}")

        if not self.samples:
            raise ValueError(f"No valid samples found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def get_damage_label_from_mask(self, mask: np.ndarray) -> int:
        # Debug: Check mask shape and unique values
        if not hasattr(self, 'debug_printed'):
            print(f"Mask shape: {mask.shape}")
            unique_values = np.unique(mask)
            print(f"Unique values in first mask: {unique_values}")
            self.debug_printed = True

        # Since mask is grayscale, pixel values are the class labels
        damage_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for cls in range(5):
            damage_counts[cls] = np.sum(mask == cls)

        if not hasattr(self, 'debug_printed_counts'):
            print(f"Pixel counts in first mask: {damage_counts}")
            self.debug_printed_counts = True

        # Remove background and prioritize higher damage levels
        del damage_counts[0]
        if sum(damage_counts.values()) == 0:
            return 1  # Default to no damage if no damage pixels
        # Prioritize highest damage level present
        for cls in [4, 3, 2, 1]:
            if damage_counts[cls] > 0:
                return cls
        return 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, dict]:
        sample_data = self.samples[idx]
        post_img = Image.open(sample_data['post_img']).convert('RGB')
        post_mask = Image.open(sample_data['post_mask']).convert('L')  # Load as grayscale
        post_mask_np = np.array(post_mask)
        damage_label = self.get_damage_label_from_mask(post_mask_np)

        if self.use_both_images:
            pre_img = Image.open(sample_data['pre_img']).convert('RGB')
            if self.transform:
                pre_img = self.transform(pre_img)
                post_img = self.transform(post_img)
            image = torch.cat([pre_img, post_img], dim=0)
        else:
            if self.transform:
                post_img = self.transform(post_img)
            image = post_img

        metadata = {
            'post_img_path': sample_data['post_img'],
            'damage_label': damage_label
        }
        return image, damage_label, metadata

class XBDEpisodeSampler:
    def __init__(self, dataset: XBDMaskDataset, n_way: int, n_support: int, n_query: int):
        self.dataset = dataset
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query

        self.label_to_indices = {}
        for idx, (_, label, _) in enumerate(dataset):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        self.labels = [label for label in self.label_to_indices
                       if len(self.label_to_indices[label]) >= (n_support + n_query)]

        print(f"Available classes for episodes: {self.labels}")
        for label in self.labels:
            print(f"Class {label}: {len(self.label_to_indices[label])} samples")

    def sample_episode(self) -> Tuple[List[int], List[int], List[int]]:
        n_way = min(self.n_way, len(self.labels))
        episode_labels = random.sample(self.labels, n_way)

        support_indices = []
        query_indices = []

        for label in episode_labels:
            label_indices = self.label_to_indices[label]
            if len(label_indices) < (self.n_support + self.n_query):
                samples = random.choices(label_indices, k=self.n_support + self.n_query)
            else:
                samples = random.sample(label_indices, self.n_support + self.n_query)
            support_indices.extend(samples[:self.n_support])
            query_indices.extend(samples[self.n_support:])
        return support_indices, query_indices, episode_labels


def train_protonet(model: ProtoNet, train_sampler: XBDEpisodeSampler, val_sampler: Optional[XBDEpisodeSampler] = None,
                   num_episodes: int = 10000, learning_rate: float = 0.001,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> ProtoNet:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_acc = 0.0
    best_model = None

    for episode in range(num_episodes):
        model.train()
        optimizer.zero_grad()
        support_indices, query_indices, episode_labels = train_sampler.sample_episode()
        support_labels = torch.tensor([episode_labels.index(train_sampler.dataset[idx][1]) for idx in support_indices], device=device)
        query_labels = torch.tensor([episode_labels.index(train_sampler.dataset[idx][1]) for idx in query_indices], device=device)
        support_images = torch.stack([train_sampler.dataset[idx][0] for idx in support_indices]).to(device)
        query_images = torch.stack([train_sampler.dataset[idx][0] for idx in query_indices]).to(device)

        support_features = model(support_images)
        query_features = model(query_images)
        prototypes = model.calculate_prototypes(support_features, support_labels)
        distances = model.compute_distances(query_features, prototypes)
        log_p_y = F.log_softmax(-distances, dim=1)
        loss = F.nll_loss(log_p_y, query_labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(log_p_y, 1)
        accuracy = (predicted == query_labels).float().mean().item()

        if episode % 100 == 0:
            print(f"Episode {episode}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.4f}")

        if val_sampler is not None and episode % 500 == 0:
            model.eval()
            val_accuracies = []
            for _ in range(100):
                with torch.no_grad():
                    support_indices, query_indices, episode_labels = val_sampler.sample_episode()
                    support_labels = torch.tensor([episode_labels.index(val_sampler.dataset[idx][1]) for idx in support_indices], device=device)
                    query_labels = torch.tensor([episode_labels.index(val_sampler.dataset[idx][1]) for idx in query_indices], device=device)
                    support_images = torch.stack([val_sampler.dataset[idx][0] for idx in support_indices]).to(device)
                    query_images = torch.stack([val_sampler.dataset[idx][0] for idx in query_indices]).to(device)
                    support_features = model(support_images)
                    query_features = model(query_images)
                    prototypes = model.calculate_prototypes(support_features, support_labels)
                    distances = model.compute_distances(query_features, prototypes)
                    _, predicted = torch.min(distances, 1)
                    val_accuracies.append((predicted == query_labels).float().mean().item())
            val_acc = np.mean(val_accuracies)
            print(f"Validation accuracy: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model.state_dict().copy()

    if best_model:
        model.load_state_dict(best_model)
    return model


def evaluate_model(model: ProtoNet, test_sampler: XBDEpisodeSampler, num_episodes: int = 1000,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
    model.eval()
    accuracies = []
    class_accuracies = {label: [] for label in test_sampler.labels}

    for _ in range(num_episodes):
        with torch.no_grad():
            support_indices, query_indices, episode_labels = test_sampler.sample_episode()
            support_labels = torch.tensor([episode_labels.index(test_sampler.dataset[idx][1]) for idx in support_indices], device=device)
            query_labels = torch.tensor([episode_labels.index(test_sampler.dataset[idx][1]) for idx in query_indices], device=device)
            idx_to_label = {i: label for i, label in enumerate(episode_labels)}
            support_images = torch.stack([test_sampler.dataset[idx][0] for idx in support_indices]).to(device)
            query_images = torch.stack([test_sampler.dataset[idx][0] for idx in query_indices]).to(device)
            support_features = model(support_images)
            query_features = model(query_images)
            prototypes = model.calculate_prototypes(support_features, support_labels)
            distances = model.compute_distances(query_features, prototypes)
            _, predicted = torch.min(distances, 1)
            accuracy = (predicted == query_labels).float().mean().item()
            accuracies.append(accuracy)
            for i, label_idx in enumerate(query_labels):
                actual_label = idx_to_label[label_idx.item()]
                is_correct = (predicted[i] == label_idx).item()
                class_accuracies[actual_label].append(is_correct)

    avg_accuracy = np.mean(accuracies)
    avg_class_accuracies = {label: np.mean(accs) if accs else 0 for label, accs in class_accuracies.items()}
    return {'overall_accuracy': avg_accuracy, 'class_accuracies': avg_class_accuracies}


def visualize_results(results: Dict[str, float]):
    damage_names = {1: "No Damage", 2: "Minor Damage", 3: "Major Damage", 4: "Destroyed"}
    print(f"Overall accuracy: {results['overall_accuracy']:.4f}")
    class_accs = results['class_accuracies']
    labels = [damage_names.get(label, f"Class {label}") for label in class_accs.keys()]
    accs = list(class_accs.values())
    plt.figure(figsize=(10, 6))
    plt.bar(labels, accs)
    plt.ylim(0, 1)
    plt.title('Per-class Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Damage Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_accuracies.png')
    plt.close()


def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    xbd_root = r"C:\Users\joshp_ya\PycharmProjects\earthquake"

    train_dataset = XBDMaskDataset(
        root_dir=xbd_root,
        split='train',
        use_both_images=True,
        transform=transform,
        extension='*.png'
    )
    test_dataset = XBDMaskDataset(
        root_dir=xbd_root,
        split='test',
        use_both_images=True,
        transform=transform,
        extension='*.png'
    )

    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    image, label, metadata = train_dataset[0]
    print(f"Image shape: {image.shape}, Label: {label}, Metadata: {metadata}")

    n_way, n_support, n_query = 4, 5, 10
    train_sampler = XBDEpisodeSampler(train_dataset, n_way, n_support, n_query)
    val_sampler = XBDEpisodeSampler(val_dataset, n_way, n_support, n_query)
    test_sampler = XBDEpisodeSampler(test_dataset, n_way, n_support, n_query)

    model = ProtoNet(input_channels=6, hidden_dim=64)
    trained_model = train_protonet(model, train_sampler, val_sampler, num_episodes=1000)
    torch.save(trained_model.state_dict(), 'protonet_xbd.pth')
    results = evaluate_model(trained_model, test_sampler)
    visualize_results(results)
    print("Training and evaluation completed!")


if __name__ == "__main__":
    main()