import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from bayesian_torch.layers import Conv2dFlipout
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
    def __init__(self, dataset="omniglot"):
        super().__init__()
        planes = [128, 128, 128, 128]
        in_channels = 1 if dataset in ["omniglot", "mnist"] else 3
        self.conv1 = Conv2dFlipout(in_channels, planes[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = Conv2dFlipout(planes[0], planes[1], kernel_size=5, stride=1, padding=2)
        self.conv3 = Conv2dFlipout(planes[1], planes[2], kernel_size=3, stride=2, padding=1)
        self.conv4 = Conv2dFlipout(planes[2], planes[3], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, sample=True):
        out, _ = self.conv1(x, sample)
        out = self.relu(out)
        out, _ = self.conv2(out, sample)
        out = self.relu(out)
        out, _ = self.conv3(out, sample)
        out = self.relu(out)
        out, _ = self.conv4(out, sample)
        out = self.relu(out)
        return out.view(out.size(0), -1)

class MANN(nn.Module):
    def __init__(self, dataset):
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

    def forward(self, x):
        return self.model(x).view(x.size(0), -1)

def bayesian_loss(model, criterion, outputs, targets, kl_weight=1.0):
    return criterion(outputs, targets) + kl_weight * get_kl_loss(model)

# -------------------------------
# Data Utilities
# -------------------------------
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

# -------------------------------
# Training & Testing Functions
# -------------------------------
def train_model(model, data_by_class, n_way, k_shot, n_query, num_epochs, episodes_per_epoch, optimizer, criterion, use_noise=False, noise_std=1.0, is_bayesian=False):
    model.train()
    for epoch in range(num_epochs):
        for _ in range(episodes_per_epoch):
            support_imgs, support_labels, query_imgs, query_labels = sample_episode(data_by_class, n_way, k_shot, n_query)
            support_imgs, support_labels = support_imgs.to(DEVICE), support_labels.to(DEVICE)
            query_imgs, query_labels = query_imgs.to(DEVICE), query_labels.to(DEVICE)
            support_emb = model(support_imgs)
            query_emb = model(query_imgs)
            if use_noise:
                support_emb += torch.randn_like(support_emb) * noise_std
                query_emb += torch.randn_like(query_emb) * noise_std
            support_norm = F.normalize(support_emb, p=2, dim=1)
            query_norm = F.normalize(query_emb, p=2, dim=1)
            scores = torch.mm(query_norm, support_norm.t())
            if is_bayesian:
                beta = min(1, (epoch + 1) / num_epochs)
                loss = bayesian_loss(model, criterion, scores, query_labels, beta)
            else:
                loss = criterion(scores, query_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def test_model(model, data_by_class, n_way, k_shot, n_query, test_episodes, noise_levels, use_noise=True):
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
                    support_emb += torch.randn_like(support_emb) * noise_std
                    query_emb += torch.randn_like(query_emb) * noise_std
                support_norm = F.normalize(support_emb, p=2, dim=1)
                query_norm = F.normalize(query_emb, p=2, dim=1)
                scores = torch.mm(query_norm, support_norm.t())
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

for it in range(NUM_ITER_AVG):
    # 1. No noise training
    model = MANN(dataset=DATASET).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, EPISODES_PER_EPOCH, optimizer, criterion, use_noise=False)
    acc_array[0] += test_model(model, train_data, N_WAY, K_SHOT, N_QUERY, TEST_EPISODES, NOISE_LEVELS, use_noise=True)

    # 2. Noise injection training
    model = MANN(dataset=DATASET).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, EPISODES_PER_EPOCH, optimizer, criterion, use_noise=True, noise_std=NOISE_STD_MIN)
    acc_array[1] += test_model(model, train_data, N_WAY, K_SHOT, N_QUERY, TEST_EPISODES, NOISE_LEVELS, use_noise=True)

    # 3. Bayesian model
    model = BayesianMANN(dataset=DATASET).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, EPISODES_PER_EPOCH, optimizer, criterion, use_noise=False, is_bayesian=True)
    acc_array[2] += test_model(model, train_data, N_WAY, K_SHOT, N_QUERY, TEST_EPISODES, NOISE_LEVELS, use_noise=True)

acc_array /= NUM_ITER_AVG
np.savetxt("accuracy_results_with_Bayesian.csv", acc_array, delimiter=",", fmt="%.2f")

# -------------------------------
# Plot Results
# -------------------------------
plt.figure(figsize=(8, 6))
plt.plot(NOISE_LEVELS, acc_array[0], label="No noise", linestyle='-', marker='o')
plt.plot(NOISE_LEVELS, acc_array[1], label="Noise injection", linestyle='--', marker='s')
plt.plot(NOISE_LEVELS, acc_array[2], label="Bayesian", linestyle='-.', marker='^')
plt.xlabel("Noise STDEV")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Noise STDEV")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Optional: Plot embedding histogram for analysis
# -------------------------------
# plot_embedding_histogram(model, train_data, n_samples=200, title="Embedding Value Distribution")