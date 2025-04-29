import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from bayesian_torch.layers import Conv2dFlipout, LinearFlipout
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
from torchvision import datasets

# -------------------------------
# Parameters
# -------------------------------
# Dataset selection: 'omniglot', 'cifar-10', 'mnist'
DATASET = 'mnist'
N_WAY = 5
K_SHOT = 1
N_QUERY = 5
NUM_EPOCHS = 100
EPISODES_PER_EPOCH = 50
TEST_EPISODES = 500
NOISE_STD_MIN = 1
NOISE_LEVELS = np.concatenate([np.linspace(0, 9, 10), np.logspace(1, 3, 21)])
NUM_NOISE = len(NOISE_LEVELS)
NUM_ITER_AVG = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Model Definitions
# -------------------------------
class BayesianMANN(nn.Module):
    def __init__(self, dataset, out_dim=128 ,quantize=False):
        super().__init__()
        planes = [128, 128, 128, 128]
        in_channels = 1 if dataset in ["omniglot", "mnist"] else 3
        self.conv1 = Conv2dFlipout(in_channels, planes[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = Conv2dFlipout(planes[0], planes[1], kernel_size=5, stride=1, padding=2)
        self.conv3 = Conv2dFlipout(planes[1], planes[2], kernel_size=3, stride=2, padding=1)
        self.conv4 = Conv2dFlipout(planes[2], planes[3], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = LinearFlipout(planes[3] * 7 * 7, out_dim)  # Adjusted for input size
        self.quantize = quantize
        self.scale = None

    def forward(self, x, sample=True, apply_quantize=None):
        out, _ = self.conv1(x, sample)
        out = self.relu(out)
        out, _ = self.conv2(out, sample)
        out = self.relu(out)
        out, _ = self.conv3(out, sample)
        out = self.relu(out)
        out, _ = self.conv4(out, sample)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out, _ = self.fc(out, sample)
        
        if apply_quantize:
            # 量化 embedding
            quantized_emb, self.scale = quantize_to_int8(emb, self.scale)
            # 立即反量化回浮點，以便後續處理
            emb = dequantize_from_int8(quantized_emb, self.scale)
        return out

class MANN(nn.Module):
    def __init__(self, dataset, out_dim=128, quantize=False):
        super().__init__()
        in_channels = 1 if dataset in ["omniglot", "mnist"] else 3
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 7 * 7, out_dim)  # Adjusted for input size
        self.quantize = quantize
        self.scale = None  # 用於存儲縮放因子

    def forward(self, x, apply_quantize=None):
        if apply_quantize is None:
            apply_quantize = self.quantize
            
        emb = self.fc(self.model(x).view(x.size(0), -1))
        
        if apply_quantize:
            # 量化 embedding
            quantized_emb, self.scale = quantize_to_int8(emb, self.scale)
            # 立即反量化回浮點，以便後續處理
            emb = dequantize_from_int8(quantized_emb, self.scale)
        return emb

def bayesian_loss(model, criterion, outputs, targets, kl_weight=1.0):
    return criterion(outputs, targets) + kl_weight * get_kl_loss(model)

def cosine_similarity(a, b):
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.t())

def hamming_distance(a, b):
    # 將embedding二值化（例如以0為閾值）
    a_bin = (a > 0).float()
    b_bin = (b > 0).float()
    # Hamming距離越小越相似，這裡回傳負的距離作為分數
    dist = (a_bin.unsqueeze(1) != b_bin.unsqueeze(0)).float().sum(dim=2)
    return -dist

def linf_distance(a, b):
    # L-infinity距離越小越相似，這裡回傳負的距離作為分數
    dist = torch.max(torch.abs(a.unsqueeze(1) - b.unsqueeze(0)), dim=2)[0]
    return -dist

# -------------------------------
# Data Utilities
# -------------------------------
# 添加量化相關函數
def quantize_to_int8(tensor, scale=None):
    """
    將浮點張量量化為int8
    
    Args:
        tensor: 輸入張量
        scale: 縮放因子，如果為None則動態計算
        
    Returns:
        量化後的int8張量和縮放因子
    """
    if scale is None:
        abs_max = torch.max(torch.abs(tensor)).item()
        scale = 127.0 / (abs_max + 1e-8)
    
    # 使用 STE (Straight-Through Estimator) 技巧
    # 前向傳播時進行量化，反向傳播時梯度直接通過
    scaled = tensor * scale
    # detach-重連接技巧，確保梯度能夠流動
    quantized = scaled.detach().round().clamp(-128, 127) - scaled.detach() + scaled
    return quantized, scale

def dequantize_from_int8(quantized, scale):
    """
    將int8張量反量化為浮點
    
    Args:
        quantized: 量化後的int8張量
        scale: 縮放因子
        
    Returns:
        反量化後的浮點張量
    """
    return quantized.float() / scale

def organize_by_class(dataset):
    data_by_class = {}
    for img, label in dataset:
        if label not in data_by_class:
            data_by_class[label] = []
        data_by_class[label].append(img)
    return data_by_class

def sample_episode(data_by_class, n_way, k_shot, n_query):
    selected_classes = random.sample(list(data_by_class.keys()), n_way)
    support_images, support_labels, query_images, query_labels = [], [], [], []
    for idx, cls in enumerate(selected_classes):
        images = data_by_class[cls]
        if len(images) < (k_shot + n_query):
            images = images * ((k_shot + n_query) // len(images) + 1)
        selected_imgs = random.sample(images, k_shot + n_query)
        support_images += selected_imgs[:k_shot]
        support_labels += [idx] * k_shot
        query_images += selected_imgs[k_shot:]
        query_labels += [idx] * n_query
    return (
        torch.stack(support_images),
        torch.tensor(support_labels),
        torch.stack(query_images),
        torch.tensor(query_labels),
    )

# -------------------------------
# Embedding Histogram
# -------------------------------
def plot_embedding_histogram(model, data_by_class, n_samples=100, title="Embedding Value Distribution"):
    model.eval()
    images = []
    for _ in range(n_samples):
        cls = random.choice(list(data_by_class.keys()))
        img = random.choice(data_by_class[cls])
        images.append(img)
    images = torch.stack(images).to(DEVICE)
    with torch.no_grad():
        embeddings = model(images)
    values = embeddings.cpu().numpy().flatten()
    mean_val, std_val, min_val, max_val = np.mean(values), np.std(values), np.min(values), np.max(values)
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, alpha=0.7)
    plt.axvline(x=mean_val, color='r', linestyle='--', label=f"Mean: {mean_val:.4f}")
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()
    print(f"Embedding stats: mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}, dim={embeddings.shape[1]}")
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()
    
def visualize_quantization_effect(model, data_by_class, n_samples=10):
    """Visualize the difference between original and quantized embeddings"""
    model.eval()
    images = []
    for _ in range(n_samples):
        cls = random.choice(list(data_by_class.keys()))
        img = random.choice(data_by_class[cls])
        images.append(img)
    images = torch.stack(images).to(DEVICE)
    
    with torch.no_grad():
        # Get non-quantized embedding
        model.quantize = False
        orig_emb = model(images)
        
        # Get quantized embedding
        model.quantize = True
        quant_emb = model(images)
        
        # Get int8 values directly (for display only)
        quantized_emb, scale = quantize_to_int8(orig_emb, None)
    
    # Calculate statistics
    plt.figure(figsize=(15, 12))
    
    # Original float distribution
    plt.subplot(3, 2, 5)
    plt.hist(orig_emb.cpu().numpy().flatten(), bins=50, alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(f"Original Float Distribution (Mean={orig_emb.mean().item():.4f})")
    plt.grid(True)
    
    # Int8 quantized value distribution
    plt.subplot(3, 2, 6)
    plt.hist(quantized_emb.cpu().numpy().flatten(), bins=range(-128, 129), alpha=0.7)
    plt.xlabel("Int8 Value")
    plt.ylabel("Frequency")
    plt.title(f"Int8 Quantized Value Distribution (Scale={scale:.4f})")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("quantization_effect_detailed.png", dpi=300)
    plt.show()
    
    # Print quantization statistics
    print(f"Original float range: [{orig_emb.min().item():.4f}, {orig_emb.max().item():.4f}]")
    print(f"Original float mean: {orig_emb.mean().item():.4f}, std: {orig_emb.std().item():.4f}")
    print(f"Int8 value range: [{quantized_emb.min().item()}, {quantized_emb.max().item()}]")
    print(f"Scale factor: {scale:.4f}")
    
    # Calculate Int8 value distribution
    int8_values = quantized_emb.cpu().numpy().flatten().astype(np.int8)
    unique_vals, counts = np.unique(int8_values, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    
    print("\nInt8 value distribution (top 10 most common values):")
    for i in range(min(10, len(unique_vals))):
        idx = sorted_indices[i]
        print(f"  Value {unique_vals[idx]}: {counts[idx]} times ({counts[idx]/len(int8_values)*100:.2f}%)")

# -------------------------------
# Training & Testing Functions
# -------------------------------
def train_model(model, data_by_class, n_way, k_shot, n_query, num_epochs, episodes_per_epoch, optimizer, criterion, use_noise=False, noise_std=1.0, is_bayesian=False, distance_fn=cosine_similarity):
    model.train()
    for epoch in range(num_epochs):
        for _ in range(episodes_per_epoch):
            support_imgs, support_labels, query_imgs, query_labels = sample_episode(data_by_class, n_way, k_shot, n_query)
            support_imgs, support_labels = support_imgs.to(DEVICE), support_labels.to(DEVICE)
            query_imgs, query_labels = query_imgs.to(DEVICE), query_labels.to(DEVICE)
            
            support_emb = model(support_imgs)
            query_emb = model(query_imgs)
            
            if use_noise:
                support_emb = support_emb + torch.randn_like(support_emb) * noise_std
                query_emb = query_emb + torch.randn_like(query_emb) * noise_std

            # 使用指定的距離函數
            scores = distance_fn(query_emb, support_emb)
            
            if is_bayesian:
                beta = min(1, (epoch + 1) / num_epochs)
                loss = bayesian_loss(model, criterion, scores, query_labels, beta)
            else:
                loss = criterion(scores, query_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def test_model(model, data_by_class, n_way, k_shot, n_query, test_episodes, noise_levels, use_noise=True, distance_fn=cosine_similarity):
    model.eval()
    acc_list = []
    for noise_std in noise_levels:
        total_correct, total_num = 0, 0
        with torch.no_grad():
            for _ in range(test_episodes):
                support_imgs, support_labels, query_imgs, query_labels = sample_episode(data_by_class, n_way, k_shot, n_query)
                support_imgs, support_labels = support_imgs.to(DEVICE), support_labels.to(DEVICE)
                query_imgs, query_labels = query_imgs.to(DEVICE), query_labels.to(DEVICE)
                
                support_emb = model(support_imgs)
                query_emb = model(query_imgs)
                
                if use_noise:
                    support_emb = support_emb + torch.randn_like(support_emb) * noise_std
                    query_emb = query_emb + torch.randn_like(query_emb) * noise_std

                # 使用指定的距離函數
                scores = distance_fn(query_emb, support_emb)
                _, max_indices = torch.max(scores, dim=1)
                pred = support_labels[max_indices]
                total_correct += (pred == query_labels).sum().item()
                total_num += query_labels.size(0)
        
        acc = total_correct / total_num * 100.0
        acc_list.append(acc)
    return np.array(acc_list)

# -------------------------------
# Dataset Preparation
# -------------------------------
if DATASET == 'omniglot':
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x)
    ])
    train_dataset = torchvision.datasets.Omniglot(root='./data', background=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.Omniglot(root='./data', background=False, download=True, transform=transform)
elif DATASET == 'cifar-10':
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
elif DATASET == 'mnist':
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='./mnist', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./mnist', train=False, transform=transform, download=True)

train_data = organize_by_class(train_dataset)
test_data = organize_by_class(test_dataset)

# -------------------------------
# Main Experiment Loop
# -------------------------------
acc_array = np.zeros((3, NUM_NOISE))

acc_array = np.zeros((3, NUM_NOISE))
acc_array_quantized = np.zeros((3, NUM_NOISE))

distance_fns = [
    ("cosine", cosine_similarity),
    ("hamming", hamming_distance),
    ("linf", linf_distance)
]

results = np.zeros((3, 3, NUM_NOISE))  # [train][test][noise]

for it in range(NUM_ITER_AVG):
    for i, (train_name, train_fn) in enumerate(distance_fns):
        for j, (test_name, test_fn) in enumerate(distance_fns):
            print(f"Train: {train_name}, Test: {test_name}")
            # 只允許 cosine_similarity 訓練
            if train_name != "cosine":
                continue
            model = MANN(dataset=DATASET, quantize=False).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, EPISODES_PER_EPOCH, optimizer, nn.CrossEntropyLoss(), use_noise=False, distance_fn=train_fn)
            model.quantize = True
            results[i, j] = test_model(model, train_data, N_WAY, K_SHOT, N_QUERY, TEST_EPISODES, NOISE_LEVELS, use_noise=True, distance_fn=test_fn)

np.save("all_distance_results.npy", results)

plt.xscale('log')
plt.xlabel("Noise Std")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Noise Level for All Train/Test Distance Combinations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("distance_combo_results.png", dpi=300)
plt.show()