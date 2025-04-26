import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import time
from bayesian_torch.layers import Conv2dFlipout, LinearFlipout
from bayesian_torch.models.dnn_to_bnn import get_kl_loss, dnn_to_bnn
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 1. 定義權重初始化與CNN模型
# -------------------------------

class BayesianMANN(nn.Module):
    def __init__(self, dataset="omniglot"):
        nn.Module.__init__(self)
        self.planes = [128, 128, 128, 128]
        
        # 根據資料集選擇輸入通道
        in_channels = 1 if dataset in ["omniglot", "mnist"] else 3

        self.conv1 = Conv2dFlipout(in_channels, self.planes[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = Conv2dFlipout(self.planes[0], self.planes[1], kernel_size=5, stride=1, padding=2)
        self.conv3 = Conv2dFlipout(self.planes[1], self.planes[2], kernel_size=3, stride=2, padding=1)
        self.conv4 = Conv2dFlipout(self.planes[2], self.planes[3], kernel_size=3, stride=1, padding=1)

        # self.fc = LinearFlipout(self.planes[3] * 7 * 7, 128)  # 假設輸入 28x28，經過 conv 會降到 7x7
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x, sample=True):
        out, _ = self.conv1(x, sample)
        out = self.relu(out)
        out, _ = self.conv2(out, sample)
        out = self.relu(out)
        # out = self.maxpool(out)
        out, _ = self.conv3(out, sample)
        out = self.relu(out)
        out, _ = self.conv4(out, sample)
        out = self.relu(out)
        
        out = out.view(out.size(0), -1)
        
        return out

def bayesian_loss(model, criterion, outputs, targets, kl_weight=1.0):
    loss = criterion(outputs, targets) + kl_weight * get_kl_loss(model)
    return loss

class MANN(nn.Module):
    def __init__(self, dataset):
        nn.Module.__init__(self)
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

def apply_lsh(embeddings, hash_vectors, slope = 30):
    """
    將嵌入向量轉成 LSH binary hash 向量
    """
    hashed_embeddings = torch.tanh(slope * embeddings @ hash_vectors.t())  # (num_samples, n_bits)
    return hashed_embeddings

# -------------------------------
# 2. 資料結構整理
# -------------------------------

# 將 dataset 依照類別重新整理為 dict: class_idx -> list of (image, label)
def organize_by_class(dataset):
    data_by_class = {}
    for img, label in dataset:
        if label not in data_by_class:
            data_by_class[label] = []
        data_by_class[label].append(img)
    return data_by_class

from matplotlib import colormaps
def visualize_episode_embeddings(filename, support_embeddings, support_labels, 
                                 query_embeddings, query_labels, title="Embedding Visualization (t-SNE)"):

    # 將資料合併
    all_embeddings = torch.cat([support_embeddings, query_embeddings], dim=0).detach().cpu().numpy()
    all_labels = torch.cat([support_labels, query_labels], dim=0).cpu().numpy()

    # 建立標記：支援為 0，查詢為 1
    domain_flag = torch.cat([
        torch.zeros(len(support_embeddings)),
        torch.ones(len(query_embeddings))
    ]).numpy()

    # t-SNE 降維
    perplexity = min(30, (support_embeddings.size(0) + query_embeddings.size(0)) // 3)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # 建立顏色對應（新版寫法）
    unique_labels = np.unique(all_labels)
    cmap = colormaps['tab10']

    # 畫圖
    plt.figure(figsize=(10, 8))
    for i, class_id in enumerate(unique_labels):
        color = cmap(i % 10)  # 限制在前 10 色
        for domain in [0, 1]:
            idx = (all_labels == class_id) & (domain_flag == domain)
            marker = 'o' if domain == 0 else 'x'
            label_name = f"Class {class_id} - {'Support' if domain == 0 else 'Query'}"
            plt.scatter(
                embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
                label=label_name, marker=marker, color=color, alpha=0.7, s=60
            )

    plt.legend()
    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------------
# 3. 定義 n-way k-shot Episode 產生器
# -------------------------------
def sample_episode(data_by_class, n_way, k_shot, n_query):
    """
    從資料中隨機選取 n 個類別，每個類別抽取 k_shot 個 support 與 n_query 個 query images
    回傳：
        support_images: tensor (n_way * k_shot, C, H, W)
        support_labels: tensor (n_way * k_shot)
        query_images: tensor (n_way * n_query, C, H, W)
        query_labels: tensor (n_way * n_query)
    """
    selected_classes = random.sample(list(data_by_class.keys()), n_way)
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []
    for idx, cls in enumerate(selected_classes):
        images = data_by_class[cls]
        # 若數量不足，隨機複製抽取
        if len(images) < (k_shot + n_query):
            images = images * ((k_shot+n_query) // len(images) + 1)
        selected_imgs = random.sample(images, k_shot + n_query)
        support_imgs = selected_imgs[:k_shot]
        query_imgs = selected_imgs[k_shot:]
        support_images += support_imgs
        support_labels += [idx] * k_shot
        query_images += query_imgs
        query_labels += [idx] * n_query
    # 將 list 轉為 tensor
    support_images = torch.stack(support_images)
    support_labels = torch.tensor(support_labels)
    query_images = torch.stack(query_images)
    query_labels = torch.tensor(query_labels)
    return support_images, support_labels, query_images, query_labels

def plot_and_save_tsne(models, filename, title="t-SNE: Support vs Query Embeddings (No-Noise)", 
                       hash_vectors=None):
    cpumodel = models.to('cpu')

    support_imgs, support_labels, query_imgs, query_labels = sample_episode(train_data, 5, 1, 10)
    
    support_embeddings = cpumodel(support_imgs)
    query_embeddings = cpumodel(query_imgs)
    
    noise_s = torch.randn_like(support_embeddings) * noise_std_min * 10  # Support Embeddings 加 noise
    noise_q = torch.randn_like(query_embeddings) * noise_std_min * 10    # Query Embeddings 加 noise
    noisy_support_embeddings = support_embeddings + noise_s
    noisy_query_embeddings = query_embeddings + noise_q
    
    visualize_episode_embeddings(
        filename=filename,
        support_embeddings=support_embeddings,
        support_labels=support_labels,
        query_embeddings=query_embeddings,
        query_labels=query_labels,
        title=title,
    )
    visualize_episode_embeddings(
        filename="noisy "+filename,
        support_embeddings=noisy_support_embeddings,
        support_labels=support_labels,
        query_embeddings=noisy_query_embeddings,
        query_labels=query_labels,
        title=title,
    )
    if hash_vectors != None:
        hash_vectors = hash_vectors.to("cpu")
        hashed_support = apply_lsh(support_embeddings, hash_vectors)
        hashed_query = apply_lsh(query_embeddings, hash_vectors)
        noisy_hashed_support = apply_lsh(noisy_support_embeddings, hash_vectors)
        noisy_hashed_query = apply_lsh(noisy_query_embeddings, hash_vectors)
        visualize_episode_embeddings(
            filename="LSH "+filename,
            support_embeddings=hashed_support,
            support_labels=support_labels,
            query_embeddings=hashed_query,
            query_labels=query_labels,
            title=title+" LSH",
        )
        visualize_episode_embeddings(
            filename="LSH noisy "+filename,
            support_embeddings=noisy_hashed_support,
            support_labels=support_labels,
            query_embeddings=noisy_hashed_query,
            query_labels=query_labels,
            title=title+" LSH",
        )

    models = models.to(device)
    
# Import dataset
dataset_collection = {'omniglot', 'cifar-10', 'mnist'}
dataset_selected = 'mnist'

if dataset_selected == 'omniglot':
    # 由 torchvision 下載 Omniglot，這裡使用灰階影像
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        # Omniglot 圖片原本是白底黑字，轉換時反轉顏色
        transforms.Lambda(lambda x: 1.0 - x)
    ])
    # 下載背景集作為訓練集 & evaluation 集作為測試集
    train_dataset = torchvision.datasets.Omniglot(root='./data', background=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.Omniglot(root='./data', background=False, download=True, transform=transform)
  
elif dataset_selected == 'cifar-10': 
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # CIFAR-10 原本就是 32x32，不改變大小
        transforms.ToTensor(),
    ])
    # 讀取 CIFAR-100 數據集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

elif dataset_selected == 'mnist': 
    transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
    train_dataset = datasets.MNIST(root='./mnist', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./mnist', train=False, transform=transform, download=True)
    

# 轉換成 dict 以便 Few-Shot Learning 使用
train_data = organize_by_class(train_dataset)
test_data = organize_by_class(test_dataset)

# -------------------------------
# 4. 訓練參數與模型、optimizer設定
# -------------------------------
noise = np.concatenate([np.linspace(0, 9, 10), np.logspace(1, 3, 21)])
num_noise = len(noise)
num_models = 1
num_iteration_of_average = 5
Acc_array = np.zeros((num_models * 3, num_noise))
print(Acc_array)

# 設定 n_way, k_shot, n_query
n_way = 5
k_shot = 1
n_query = 5
num_epochs = 100
episodes_per_epoch = 50
# 是否加入 Zero-Mean Gaussian Noise
noise_std_min = 1  # 控制 noise 的強度（標準差）

# LSH hash vectors
LSH_bits = 1024
output_dim = 128 * 7 * 7
hash_vectors = F.normalize(torch.randn(LSH_bits, output_dim), dim=1).to(device)  # 固定 hash 向量


# -------------------------------
# 4. 訓練參數與模型、optimizer設定
# -------------------------------
for it in range(num_iteration_of_average):
    
    # 是否加入 Zero-Mean Gaussian Noise
    use_noise_in_training = False  # 控制是否加入 noise
    use_noise_in_testing = True  # 控制是否加入 noise
    
    # 建立模型，指定 dataset='omniglot'
    models = MANN(dataset=dataset_selected).to(device)
    optimizers = optim.Adam(models.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # -------------------------------
    # 5. 訓練迴圈
    # -------------------------------
    
    models.train()
    noise_std = noise_std_min
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for episode in range(episodes_per_epoch):
            # 抽取一個 episode
            support_imgs, support_labels, query_imgs, query_labels = sample_episode(train_data, n_way, k_shot, n_query)
            support_imgs, support_labels = support_imgs.to(device), support_labels.to(device)
            query_imgs, query_labels = query_imgs.to(device), query_labels.to(device)
    
            # 使用 CNN 提取特徵
            support_embeddings = models(support_imgs)  # (n_way*k_shot, embedding_dim)
            query_embeddings = models(query_imgs)      # (n_way*n_query, embedding_dim)
    
            if use_noise_in_training:
                noise_s = torch.randn_like(support_embeddings) * noise_std  # Support Embeddings 加 noise
                noise_q = torch.randn_like(query_embeddings) * noise_std    # Query Embeddings 加 noise
                support_embeddings = support_embeddings + noise_s
                query_embeddings = query_embeddings + noise_q
    
            # 計算 Cosine Similarity
            query_norm = F.normalize(query_embeddings, p=2, dim=1)  # (n_way*n_query, embedding_dim)
            support_norm = F.normalize(support_embeddings, p=2, dim=1)  # (n_way*k_shot, embedding_dim)
            scores = torch.mm(query_norm, support_norm.t())  # Cosine 相似度矩陣 (n_way*n_query, n_way*k_shot)
    
            # 找到最相似的 Support Data
            _, max_indices = torch.max(scores, dim=1)  # 找到相似度最高的 Support Data 索引
            predicted_labels = support_labels[max_indices]  # 用最相似的 Support Data 標籤作為 Query 的分類結果
    
            # 計算 Loss 並更新權重
            loss = criterion(scores, query_labels)
            optimizers.zero_grad()
            loss.backward()
            optimizers.step()
            epoch_loss += loss.item()
    
        # print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # -------------------------------
    # 6. 模型測試：以多個 episode 評估準確率
    # -------------------------------
    
    models.eval()
    test_episodes = 500
    total_correct = 0
    total_num = 0
    for j in range(num_noise):
        noise_std = noise[j]
        with torch.no_grad():
            for episode in range(test_episodes):
                # 抽取一個 episode
                support_imgs, support_labels, query_imgs, query_labels = sample_episode(train_data, n_way, k_shot, n_query)
                support_imgs, support_labels = support_imgs.to(device), support_labels.to(device)
                query_imgs, query_labels = query_imgs.to(device), query_labels.to(device)
    
                # 使用 CNN 提取特徵
                support_embeddings = models(support_imgs)  # (n_way*k_shot, embedding_dim)
                query_embeddings = models(query_imgs)      # (n_way*n_query, embedding_dim)
    
                if use_noise_in_testing:
                    noise_s = torch.randn_like(support_embeddings) * noise_std  # Support Embeddings 加 noise
                    noise_q = torch.randn_like(query_embeddings) * noise_std    # Query Embeddings 加 noise
                    support_embeddings = support_embeddings + noise_s
                    query_embeddings = query_embeddings + noise_q
    
                # 計算 Cosine Similarity
                query_norm = F.normalize(query_embeddings, p=2, dim=1)  # (n_way*n_query, embedding_dim)
                support_norm = F.normalize(support_embeddings, p=2, dim=1)  # (n_way*k_shot, embedding_dim)
                scores = torch.mm(query_norm, support_norm.t())  # Cosine 相似度矩陣 (n_way*n_query, n_way*k_shot)
    
                # 找到最相似的 Support Data
                _, max_indices = torch.max(scores, dim=1)  # 找到相似度最高的 Support Data 索引
                pred = support_labels[max_indices]  # 用最相似的 Support Data 標籤作為 Query 的分類結果
    
                total_correct += (pred == query_labels).sum().item()
                total_num += query_labels.size(0)

        test_acc = total_correct / total_num * 100.0
        print(f"Model No-Noise, Noise {j}: Final Test Accuracy over {test_episodes} episodes: {test_acc:.2f}%")
        
    
        Acc_array[0][j] += test_acc
    
    # plot_and_save_tsne(models, filename=f"No-Noise, it={it}", 
    #                    title="t-SNE: Support vs Query Embeddings (No-Noise)",
    #                    hash_vectors=hash_vectors)
    # # Save to CSV file
    # np.savetxt("accuracy_results.csv", Acc_array, delimiter=",", fmt="%.2f")
# -------------------------------
# 4. 訓練參數與模型、optimizer設定
# -------------------------------
for it in range(num_iteration_of_average):
    use_noise_in_training = True  # 控制是否加入 noise
    use_noise_in_testing = True  # 控制是否加入 noise
    
    # 建立模型，指定 dataset='omniglot'
    models = MANN(dataset=dataset_selected).to(device)
    optimizers = optim.Adam(models.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # -------------------------------
    # 5. 訓練迴圈
    # -------------------------------
    
    models.train()
    noise_std = noise_std_min
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for episode in range(episodes_per_epoch):
            # 抽取一個 episode
            support_imgs, support_labels, query_imgs, query_labels = sample_episode(train_data, n_way, k_shot, n_query)
            support_imgs, support_labels = support_imgs.to(device), support_labels.to(device)
            query_imgs, query_labels = query_imgs.to(device), query_labels.to(device)
    
            # 使用 CNN 提取特徵
            support_embeddings = models(support_imgs)  # (n_way*k_shot, embedding_dim)
            query_embeddings = models(query_imgs)      # (n_way*n_query, embedding_dim)
    
            if use_noise_in_training:
                noise_s = torch.randn_like(support_embeddings) * noise_std  # Support Embeddings 加 noise
                noise_q = torch.randn_like(query_embeddings) * noise_std    # Query Embeddings 加 noise
                support_embeddings = support_embeddings + noise_s
                query_embeddings = query_embeddings + noise_q
    
            # 計算 Cosine Similarity
            query_norm = F.normalize(query_embeddings, p=2, dim=1)  # (n_way*n_query, embedding_dim)
            support_norm = F.normalize(support_embeddings, p=2, dim=1)  # (n_way*k_shot, embedding_dim)
            scores = torch.mm(query_norm, support_norm.t())  # Cosine 相似度矩陣 (n_way*n_query, n_way*k_shot)
    
            # 找到最相似的 Support Data
            _, max_indices = torch.max(scores, dim=1)  # 找到相似度最高的 Support Data 索引
            predicted_labels = support_labels[max_indices]  # 用最相似的 Support Data 標籤作為 Query 的分類結果
    
            # 計算 Loss 並更新權重
            loss = criterion(scores, query_labels)
            optimizers.zero_grad()
            loss.backward()
            optimizers.step()
            epoch_loss += loss.item()
    
        # avg_loss = epoch_loss / episodes_per_epoch
        # print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # -------------------------------
    # 6. 模型測試：以多個 episode 評估準確率
    # -------------------------------
    
    models.eval()
    test_episodes = 500
    total_correct = 0
    total_num = 0
    for j in range(num_noise):
        noise_std = noise[j]
        with torch.no_grad():
            for episode in range(test_episodes):
                # 抽取一個 episode
                support_imgs, support_labels, query_imgs, query_labels = sample_episode(train_data, n_way, k_shot, n_query)
                support_imgs, support_labels = support_imgs.to(device), support_labels.to(device)
                query_imgs, query_labels = query_imgs.to(device), query_labels.to(device)
    
                # 使用 CNN 提取特徵
                support_embeddings = models(support_imgs)  # (n_way*k_shot, embedding_dim)
                query_embeddings = models(query_imgs)      # (n_way*n_query, embedding_dim)
    
                if use_noise_in_testing:
                    noise_s = torch.randn_like(support_embeddings) * noise_std  # Support Embeddings 加 noise
                    noise_q = torch.randn_like(query_embeddings) * noise_std    # Query Embeddings 加 noise
                    support_embeddings = support_embeddings + noise_s
                    query_embeddings = query_embeddings + noise_q
    
                # 計算 Cosine Similarity
                query_norm = F.normalize(query_embeddings, p=2, dim=1)  # (n_way*n_query, embedding_dim)
                support_norm = F.normalize(support_embeddings, p=2, dim=1)  # (n_way*k_shot, embedding_dim)
                scores = torch.mm(query_norm, support_norm.t())  # Cosine 相似度矩陣 (n_way*n_query, n_way*k_shot)
    
                # 找到最相似的 Support Data
                _, max_indices = torch.max(scores, dim=1)  # 找到相似度最高的 Support Data 索引
                pred = support_labels[max_indices]  # 用最相似的 Support Data 標籤作為 Query 的分類結果
    
                total_correct += (pred == query_labels).sum().item()
                total_num += query_labels.size(0)
        
        test_acc = total_correct / total_num * 100.0
        print(f"Model Yes-Noise, Noise {j}: Final Test Accuracy over {test_episodes} episodes: {test_acc:.2f}%")
            
        Acc_array[1][j] += test_acc
    
    # plot_and_save_tsne(models, filename=f"With-Noise, it={it}", 
    #                    title="t-SNE: Support vs Query Embeddings (Noisy Emb.)",
    #                    hash_vectors=hash_vectors)
    # # Save to CSV file
    # np.savetxt("accuracy_results.csv", Acc_array, delimiter=",", fmt="%.2f")

# -------------------------------
# 4. 訓練參數與模型、optimizer設定
# -------------------------------
for it in range(num_iteration_of_average):
    # 是否加入 Zero-Mean Gaussian Noise
    use_noise_in_training = False  # 控制是否加入 noise
    use_noise_in_testing = True  # 控制是否加入 noise
    
    # 建立模型，指定 dataset='omniglot'
    models = BayesianMANN(dataset=dataset_selected).to(device)
    optimizers = optim.Adam(models.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # -------------------------------
    # 5. 訓練迴圈
    # -------------------------------
    
    models.train()
    noise_std = noise_std_min * num_noise
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for episode in range(episodes_per_epoch):
            # 抽取一個 episode
            support_imgs, support_labels, query_imgs, query_labels = sample_episode(train_data, n_way, k_shot, n_query)
            support_imgs, support_labels = support_imgs.to(device), support_labels.to(device)
            query_imgs, query_labels = query_imgs.to(device), query_labels.to(device)
    
            # 使用 CNN 提取特徵
            support_embeddings = models(support_imgs)  # (n_way*k_shot, embedding_dim)
            query_embeddings = models(query_imgs)      # (n_way*n_query, embedding_dim)
    
            if use_noise_in_training:
                noise_s = torch.randn_like(support_embeddings) * noise_std  # Support Embeddings 加 noise
                noise_q = torch.randn_like(query_embeddings) * noise_std    # Query Embeddings 加 noise
                support_embeddings = support_embeddings + noise_s
                query_embeddings = query_embeddings + noise_q
    
            # 計算 Cosine Similarity
            query_norm = F.normalize(query_embeddings, p=2, dim=1)  # (n_way*n_query, embedding_dim)
            support_norm = F.normalize(support_embeddings, p=2, dim=1)  # (n_way*k_shot, embedding_dim)
            scores = torch.mm(query_norm, support_norm.t())  # Cosine 相似度矩陣 (n_way*n_query, n_way*k_shot)
    
            # 找到最相似的 Support Data
            _, max_indices = torch.max(scores, dim=1)  # 找到相似度最高的 Support Data 索引
            predicted_labels = support_labels[max_indices]  # 用最相似的 Support Data 標籤作為 Query 的分類結果
    
            # 計算 Loss 並更新權重
            beta = min(1, epoch / num_epochs)
            loss = bayesian_loss(models, criterion, scores, query_labels, beta)
            
            optimizers.zero_grad()
            loss.backward()
            optimizers.step()
            epoch_loss += loss.item()
    
        # avg_loss = epoch_loss / episodes_per_epoch
        # print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # -------------------------------
    # 6. 模型測試：以多個 episode 評估準確率
    # -------------------------------
    
    models.eval()
    test_episodes = 500
    total_correct = 0
    total_num = 0
    for j in range(num_noise):
        noise_std = noise[j]
        with torch.no_grad():
            for episode in range(test_episodes):
                # 抽取一個 episode
                support_imgs, support_labels, query_imgs, query_labels = sample_episode(train_data, n_way, k_shot, n_query)
                support_imgs, support_labels = support_imgs.to(device), support_labels.to(device)
                query_imgs, query_labels = query_imgs.to(device), query_labels.to(device)
    
                # 使用 CNN 提取特徵
                support_embeddings = models(support_imgs)  # (n_way*k_shot, embedding_dim)
                query_embeddings = models(query_imgs)      # (n_way*n_query, embedding_dim)
    
                if use_noise_in_testing:
                    noise_s = torch.randn_like(support_embeddings) * noise_std  # Support Embeddings 加 noise
                    noise_q = torch.randn_like(query_embeddings) * noise_std    # Query Embeddings 加 noise
                    support_embeddings = support_embeddings + noise_s
                    query_embeddings = query_embeddings + noise_q
    
                # 計算 Cosine Similarity
                query_norm = F.normalize(query_embeddings, p=2, dim=1)  # (n_way*n_query, embedding_dim)
                support_norm = F.normalize(support_embeddings, p=2, dim=1)  # (n_way*k_shot, embedding_dim)
                scores = torch.mm(query_norm, support_norm.t())  # Cosine 相似度矩陣 (n_way*n_query, n_way*k_shot)
    
                # 找到最相似的 Support Data
                _, max_indices = torch.max(scores, dim=1)  # 找到相似度最高的 Support Data 索引
                pred = support_labels[max_indices]  # 用最相似的 Support Data 標籤作為 Query 的分類結果
    
                total_correct += (pred == query_labels).sum().item()
                total_num += query_labels.size(0)
        
        test_acc = total_correct / total_num * 100.0
        print(f"Model Bayesian, Noise {j}: Final Test Accuracy over {test_episodes} episodes: {test_acc:.2f}%")
            
        Acc_array[2][j] += test_acc

    # plot_and_save_tsne(models, filename=f"Bayesian, it={it}", 
    #                    title="t-SNE: Support vs Query Embeddings (Bayesian NN)",
    #                    hash_vectors=hash_vectors)
    
for it in range(num_iteration_of_average):
    # 是否加入 Zero-Mean Gaussian Noise
    use_noise_in_training = False  # 控制是否加入 noise
    use_noise_in_testing = True  # 控制是否加入 noise
    
    # 建立模型，指定 dataset='omniglot'
    models = BayesianMANN(dataset=dataset_selected).to(device)
    optimizers = optim.Adam(models.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # -------------------------------
    # 5. 訓練迴圈
    # -------------------------------
    
    models.train()
    noise_std = noise_std_min * num_noise
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for episode in range(episodes_per_epoch):
            # 抽取一個 episode
            support_imgs, support_labels, query_imgs, query_labels = sample_episode(train_data, n_way, k_shot, n_query)
            support_imgs, support_labels = support_imgs.to(device), support_labels.to(device)
            query_imgs, query_labels = query_imgs.to(device), query_labels.to(device)
    
            # 使用 CNN 提取特徵
            support_embeddings = models(support_imgs)  # (n_way*k_shot, embedding_dim)
            query_embeddings = models(query_imgs)      # (n_way*n_query, embedding_dim)
    
            if use_noise_in_training:
                noise_s = torch.randn_like(support_embeddings) * noise_std  # Support Embeddings 加 noise
                noise_q = torch.randn_like(query_embeddings) * noise_std    # Query Embeddings 加 noise
                support_embeddings = support_embeddings + noise_s
                query_embeddings = query_embeddings + noise_q
    
            # 計算 Cosine Similarity
            query_norm = F.normalize(query_embeddings, p=2, dim=1)  # (n_way*n_query, embedding_dim)
            support_norm = F.normalize(support_embeddings, p=2, dim=1)  # (n_way*k_shot, embedding_dim)
            scores = torch.mm(query_norm, support_norm.t())  # Cosine 相似度矩陣 (n_way*n_query, n_way*k_shot)
    
            # 找到最相似的 Support Data
            _, max_indices = torch.max(scores, dim=1)  # 找到相似度最高的 Support Data 索引
            predicted_labels = support_labels[max_indices]  # 用最相似的 Support Data 標籤作為 Query 的分類結果
    
            # 計算 Loss 並更新權重
            beta = min(1, 0.5 * epoch / num_epochs)
            loss = bayesian_loss(models, criterion, scores, query_labels, beta)
            
            optimizers.zero_grad()
            loss.backward()
            optimizers.step()
            epoch_loss += loss.item()
    
        # avg_loss = epoch_loss / episodes_per_epoch
        # print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # -------------------------------
    # 6. 模型測試：以多個 episode 評估準確率
    # -------------------------------
    
    models.eval()
    test_episodes = 500
    total_correct = 0
    total_num = 0
    for j in range(num_noise):
        noise_std = noise[j]
        with torch.no_grad():
            for episode in range(test_episodes):
                # 抽取一個 episode
                support_imgs, support_labels, query_imgs, query_labels = sample_episode(train_data, n_way, k_shot, n_query)
                support_imgs, support_labels = support_imgs.to(device), support_labels.to(device)
                query_imgs, query_labels = query_imgs.to(device), query_labels.to(device)
    
                # 使用 CNN 提取特徵
                support_embeddings = models(support_imgs)  # (n_way*k_shot, embedding_dim)
                query_embeddings = models(query_imgs)      # (n_way*n_query, embedding_dim)
    
                if use_noise_in_testing:
                    noise_s = torch.randn_like(support_embeddings) * noise_std  # Support Embeddings 加 noise
                    noise_q = torch.randn_like(query_embeddings) * noise_std    # Query Embeddings 加 noise
                    support_embeddings = support_embeddings + noise_s
                    query_embeddings = query_embeddings + noise_q
    
                # 計算 Cosine Similarity
                query_norm = F.normalize(query_embeddings, p=2, dim=1)  # (n_way*n_query, embedding_dim)
                support_norm = F.normalize(support_embeddings, p=2, dim=1)  # (n_way*k_shot, embedding_dim)
                scores = torch.mm(query_norm, support_norm.t())  # Cosine 相似度矩陣 (n_way*n_query, n_way*k_shot)
    
                # 找到最相似的 Support Data
                _, max_indices = torch.max(scores, dim=1)  # 找到相似度最高的 Support Data 索引
                pred = support_labels[max_indices]  # 用最相似的 Support Data 標籤作為 Query 的分類結果
    
                total_correct += (pred == query_labels).sum().item()
                total_num += query_labels.size(0)
        
        test_acc = total_correct / total_num * 100.0
        print(f"Model Bayesian, Noise {j}: Final Test Accuracy over {test_episodes} episodes: {test_acc:.2f}%")
            
        Acc_array[2][j] += test_acc
    
Acc_array = Acc_array / num_iteration_of_average
# # Save to CSV file
np.savetxt("accuracy_results_with_Bayesian.csv", Acc_array, delimiter=",", fmt="%.2f")

import matplotlib.pyplot as plt

# x 軸數據
x_values = noise

# 繪製三條曲線
plt.figure(figsize=(8, 6))
plt.plot(x_values, Acc_array[0], label="No noise", linestyle='-', marker='o')
plt.plot(x_values, Acc_array[1], label="Noise injection", linestyle='--', marker='s')
plt.plot(x_values, Acc_array[2], label="Bayesian", linestyle='-.', marker='^')

# 圖片標籤與標題
plt.xlabel("Noise STDEV")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Noise STDEV")
# plt.xscale("log")  # 這行是設定 x 軸為對數刻度喵！
plt.legend()
plt.grid(True)
# 顯示圖表
plt.show()
