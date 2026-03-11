import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms


class OmniglotWithSplit(Dataset):
    """Wrap background/evaluation Omniglot splits into one dataset with split flags."""

    def __init__(self, root: str, download: bool = True) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 1.0 - x),
            ]
        )

        self.background = torchvision.datasets.Omniglot(
            root=root,
            background=True,
            download=download,
            transform=transform,
        )
        self.evaluation = torchvision.datasets.Omniglot(
            root=root,
            background=False,
            download=download,
            transform=transform,
        )

        self.bg_len = len(self.background)
        self.total_len = self.bg_len + len(self.evaluation)

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int, int]:
        if index < self.bg_len:
            image, label = self.background[index]
            is_background = 1
        else:
            image, label = self.evaluation[index - self.bg_len]
            is_background = 0
        return image, int(label), is_background, index


class ConvAutoEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_fc = nn.Linear(64 * 7 * 7, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder_conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.encoder_fc(feat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        feat = self.decoder_fc(z).view(-1, 64, 7, 7)
        return self.decoder_conv(feat)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


@dataclass
class TrainingConfig:
    latent_dim: int
    epochs: int
    batch_size: int
    learning_rate: float
    normalize_embedding: bool


@dataclass
class RunSummary:
    config: TrainingConfig
    final_train_loss: float
    cosine_10way_1shot_accuracy: float
    cosine_test_history: List[Dict[str, float]]
    train_size: int
    test_size: int
    neighbors_k: int
    output_hdf5: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_autoencoder(
    model: ConvAutoEncoder,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    eval_every: int,
    eval_fn: Callable[[], float],
) -> Tuple[List[float], List[Dict[str, float]]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    epoch_losses: List[float] = []
    eval_history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        running_loss = 0.0
        total_samples = 0

        for images, _, _, _ in loader:
            images = images.to(device)
            _, recon = model(images)
            loss = criterion(recon, images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        mean_loss = running_loss / max(total_samples, 1)
        epoch_losses.append(mean_loss)
        print(f"[latent={model.encoder_fc.out_features}] epoch {epoch + 1}/{epochs} loss={mean_loss:.6f}")

        if (epoch + 1) % eval_every == 0:
            acc = float(eval_fn())
            eval_history.append({"epoch": float(epoch + 1), "accuracy": acc})
            print(f"[latent={model.encoder_fc.out_features}] test@epoch {epoch + 1}: {acc:.2f}%")

    if epochs % eval_every != 0:
        acc = float(eval_fn())
        eval_history.append({"epoch": float(epochs), "accuracy": acc})
        print(f"[latent={model.encoder_fc.out_features}] test@epoch {epochs}: {acc:.2f}%")

    return epoch_losses, eval_history


@torch.no_grad()
def encode_dataset(
    model: ConvAutoEncoder,
    loader: DataLoader,
    device: torch.device,
    normalize_embedding: bool,
) -> Dict[str, np.ndarray]:
    model.eval()

    all_embeddings: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_split_flags: List[np.ndarray] = []

    for images, labels, is_background, _ in loader:
        images = images.to(device)
        z = model.encode(images)

        if normalize_embedding:
            z = F.normalize(z, p=2, dim=1)

        all_embeddings.append(z.cpu().numpy().astype(np.float32))
        all_labels.append(labels.numpy().astype(np.int64))
        all_split_flags.append(is_background.numpy().astype(np.uint8))

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    split_flags = np.concatenate(all_split_flags, axis=0)

    return {
        "embeddings": embeddings,
        "labels": labels,
        "is_background": split_flags,
    }


@torch.no_grad()
def angular_neighbors_ground_truth(
    test_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    neighbors_k: int,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if train_embeddings.shape[0] == 0:
        raise ValueError("train embeddings are empty")
    if test_embeddings.shape[0] == 0:
        raise ValueError("test embeddings are empty")

    k = min(neighbors_k, train_embeddings.shape[0])

    train_t = torch.from_numpy(train_embeddings).to(device)
    train_t = F.normalize(train_t, p=2, dim=1)

    all_neighbors: List[np.ndarray] = []
    all_distances: List[np.ndarray] = []

    for start in range(0, test_embeddings.shape[0], batch_size):
        end = min(start + batch_size, test_embeddings.shape[0])
        test_t = torch.from_numpy(test_embeddings[start:end]).to(device)
        test_t = F.normalize(test_t, p=2, dim=1)

        cosine_scores = test_t @ train_t.t()
        top_scores, top_indices = torch.topk(cosine_scores, k=k, dim=1, largest=True, sorted=True)

        # In ANN benchmark angular setting, distance is commonly 1 - cosine.
        top_distances = 1.0 - top_scores

        all_neighbors.append(top_indices.cpu().numpy().astype(np.int64))
        all_distances.append(top_distances.cpu().numpy().astype(np.float64))

    neighbors = np.concatenate(all_neighbors, axis=0)
    distances = np.concatenate(all_distances, axis=0)
    return neighbors, distances


def build_ann_benchmark_arrays(
    encoded: Dict[str, np.ndarray],
    neighbors_k: int,
    device: torch.device,
    gt_batch_size: int,
) -> Dict[str, np.ndarray]:
    is_background = encoded["is_background"].astype(bool)
    train_mask = is_background
    test_mask = ~is_background

    train = encoded["embeddings"][train_mask]
    test = encoded["embeddings"][test_mask]

    neighbors, distances = angular_neighbors_ground_truth(
        test_embeddings=test,
        train_embeddings=train,
        neighbors_k=neighbors_k,
        device=device,
        batch_size=gt_batch_size,
    )

    return {
        "train": train.astype(np.float16),
        "test": test.astype(np.float16),
        "neighbors": neighbors.astype(np.int64),
        "distances": distances.astype(np.float64),
    }


def save_hdf5_ann_benchmark(output_path: Path, data: Dict[str, np.ndarray], dimension: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        # Match reference schema exactly.
        f.attrs["dimension"] = int(dimension)
        f.attrs["distance"] = "angular"
        f.attrs["point_type"] = "float"
        f.attrs["type"] = "dense"

        f.create_dataset("train", data=data["train"], compression="gzip")
        f.create_dataset("test", data=data["test"], compression="gzip")
        f.create_dataset("neighbors", data=data["neighbors"], compression="gzip")
        f.create_dataset("distances", data=data["distances"], compression="gzip")


def build_class_to_indices(dataset: Dataset) -> Dict[int, List[int]]:
    class_to_indices: Dict[int, List[int]] = {}
    for index in range(len(dataset)):
        _, label = dataset[index]
        class_to_indices.setdefault(int(label), []).append(index)
    return class_to_indices


@torch.no_grad()
def evaluate_10way_1shot_cosine(
    model: ConvAutoEncoder,
    dataset: Dataset,
    device: torch.device,
    episodes: int,
    n_way: int,
    k_shot: int,
    n_query: int,
) -> float:
    model.eval()
    class_to_indices = build_class_to_indices(dataset)
    valid_classes = [
        class_id for class_id, indices in class_to_indices.items() if len(indices) >= k_shot + n_query
    ]

    if len(valid_classes) < n_way:
        raise ValueError(f"Not enough classes for {n_way}-way evaluation. Available={len(valid_classes)}")

    total_correct = 0
    total_count = 0

    for _ in range(episodes):
        episode_classes = random.sample(valid_classes, n_way)
        support_images: List[torch.Tensor] = []
        query_images: List[torch.Tensor] = []
        query_targets: List[int] = []

        for local_id, class_id in enumerate(episode_classes):
            selected = random.sample(class_to_indices[class_id], k_shot + n_query)
            support_idx = selected[:k_shot]
            query_idx = selected[k_shot:]

            for idx in support_idx:
                image, _ = dataset[idx]
                support_images.append(image)

            for idx in query_idx:
                image, _ = dataset[idx]
                query_images.append(image)
                query_targets.append(local_id)

        support_batch = torch.stack(support_images, dim=0).to(device)
        query_batch = torch.stack(query_images, dim=0).to(device)
        query_targets_t = torch.tensor(query_targets, dtype=torch.long, device=device)

        support_emb = F.normalize(model.encode(support_batch), p=2, dim=1)
        query_emb = F.normalize(model.encode(query_batch), p=2, dim=1)
        prototypes = support_emb.view(n_way, k_shot, -1).mean(dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)

        logits = query_emb @ prototypes.t()
        preds = torch.argmax(logits, dim=1)

        total_correct += (preds == query_targets_t).sum().item()
        total_count += query_targets_t.numel()

    return 100.0 * total_correct / max(total_count, 1)


def run_for_dimension(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    output_dir: Path,
    normalize_embedding: bool,
    test_episodes: int,
    n_way: int,
    k_shot: int,
    n_query: int,
    eval_every: int,
    neighbors_k: int,
    gt_batch_size: int,
) -> RunSummary:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    encode_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = ConvAutoEncoder(latent_dim=latent_dim).to(device)
    eval_fn = lambda: evaluate_10way_1shot_cosine(
        model=model,
        dataset=eval_dataset,
        device=device,
        episodes=test_episodes,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
    )

    losses, eval_history = train_autoencoder(
        model=model,
        loader=train_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        eval_every=eval_every,
        eval_fn=eval_fn,
    )

    encoded = encode_dataset(
        model=model,
        loader=encode_loader,
        device=device,
        normalize_embedding=normalize_embedding,
    )

    ann_data = build_ann_benchmark_arrays(
        encoded=encoded,
        neighbors_k=neighbors_k,
        device=device,
        gt_batch_size=gt_batch_size,
    )

    cosine_acc = eval_history[-1]["accuracy"] if eval_history else float("nan")
    print(
        f"[latent={latent_dim}] Cosine {n_way}-way-{k_shot}-shot "
        f"(n_query={n_query}, episodes={test_episodes}) final = {cosine_acc:.2f}%"
    )

    output_path = output_dir / f"omniglot-{latent_dim}-angular.hdf5"
    save_hdf5_ann_benchmark(output_path, ann_data, dimension=latent_dim)

    summary = RunSummary(
        config=TrainingConfig(
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            normalize_embedding=normalize_embedding,
        ),
        final_train_loss=losses[-1] if losses else float("nan"),
        cosine_10way_1shot_accuracy=cosine_acc,
        cosine_test_history=eval_history,
        train_size=int(ann_data["train"].shape[0]),
        test_size=int(ann_data["test"].shape[0]),
        neighbors_k=int(ann_data["neighbors"].shape[1]),
        output_hdf5=str(output_path),
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Omniglot benchmark vectors with 128D and 256D encoders."
    )
    parser.add_argument("--data-root", type=str, default=".data")
    parser.add_argument("--output-dir", type=str, default="Benchmark_generation/outputs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-episodes", type=int, default=500)
    parser.add_argument("--n-way", type=int, default=10)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--n-query", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--neighbors-k", type=int, default=100)
    parser.add_argument("--gt-batch-size", type=int, default=256)
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[128, 256],
        help="Latent dimensions to generate.",
    )
    parser.add_argument(
        "--disable-normalize-embedding",
        action="store_true",
        help="Disable L2 normalization on output embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize_embedding = not args.disable_normalize_embedding

    print(f"Using device: {device}")
    print("Loading Omniglot (background + evaluation)...")

    dataset = OmniglotWithSplit(root=args.data_root, download=True)
    print(f"Total samples: {len(dataset)}")

    summaries: List[RunSummary] = []
    for dim in args.dims:
        print(f"\n=== Training encoder for latent_dim={dim} ===")
        summary = run_for_dimension(
            train_dataset=dataset,
            eval_dataset=dataset.evaluation,
            latent_dim=dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            output_dir=output_dir,
            normalize_embedding=normalize_embedding,
            test_episodes=args.test_episodes,
            n_way=args.n_way,
            k_shot=args.k_shot,
            n_query=args.n_query,
            eval_every=args.eval_every,
            neighbors_k=args.neighbors_k,
            gt_batch_size=args.gt_batch_size,
        )
        summaries.append(summary)
        print(f"Saved: {summary.output_hdf5}")

    summary_path = output_dir / "generation_summary.json"
    serializable = [
        {
            "config": asdict(s.config),
            "final_train_loss": s.final_train_loss,
            "cosine_10way_1shot_accuracy": s.cosine_10way_1shot_accuracy,
            "cosine_test_history": s.cosine_test_history,
            "train_size": s.train_size,
            "test_size": s.test_size,
            "neighbors_k": s.neighbors_k,
            "output_hdf5": s.output_hdf5,
        }
        for s in summaries
    ]
    summary_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
