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
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import pandas as pd

# -------------------------------
# Parameters
# -------------------------------
DATASET = 'omniglot'
N_WAY = 5
K_SHOT = 1
N_QUERY = 5
NUM_EPOCHS = 100
EPISODES_PER_EPOCH = 11
TEST_EPISODES = 200
NOISE_STD_MIN = 0.2
NOISE_LEVELS = np.concatenate([np.linspace(0, 9, 10), np.logspace(1, 3, 21)])
NUM_NOISE = len(NOISE_LEVELS)
NUM_ITER_AVG = 5
OUTPUT_DIM = 32 if DATASET in ['omniglot', 'mnist'] else 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Model Definitions
# -------------------------------
class BayesianMANN(nn.Module):
    def __init__(self, dataset, out_dim=1024, quantize=False):
        super().__init__()
        in_channels = 1 if dataset in ['omniglot', 'mnist'] else 3
        planes = [128, 128, 128, 128] if dataset in ['omniglot', 'mnist'] else [128, 256, 512, 1024]
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
    def __init__(self, dataset, out_dim=1024, quantize=False):
        super().__init__()
        in_channels = 1 if dataset in ['omniglot', 'mnist'] else 3
        planes = [128, 128, 128, 128] if dataset in ['omniglot', 'mnist'] else [128, 256, 512, 1024]
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, planes[0], kernel_size=5, stride=2, padding=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(planes[0], planes[1], kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(planes[1], planes[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(planes[2], planes[3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.fc = nn.Linear(planes[3] * 7 * 7, out_dim)  # Adjusted for input size
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
    
class EnhancedMANN(nn.Module):
    def __init__(self, dataset, out_dim=1024, quantize=False):
        super().__init__()
        in_channels = 1 if dataset in ['omniglot', 'mnist'] else 3
        
        # 針對不同數據集調整通道數
        if dataset in ['omniglot', 'mnist']:
            base_filters = 64
            planes = [base_filters, base_filters*2, base_filters*2, base_filters*4]
        else:  # CIFAR-10 使用更多特徵圖
            base_filters = 64
            planes = [base_filters, base_filters*2, base_filters*4, base_filters*8]
        
        # 更強大的特徵提取器
        self.model = nn.Sequential(
            # 第一個卷積區塊
            nn.Conv2d(in_channels, planes[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes[0]),
            nn.ReLU(),
            nn.Conv2d(planes[0], planes[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes[0]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14
            
            # 第二個卷積區塊
            nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes[1]),
            nn.ReLU(),
            nn.Conv2d(planes[1], planes[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes[1]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7
            
            # 第三個卷積區塊
            nn.Conv2d(planes[1], planes[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes[2]),
            nn.ReLU(),
            nn.Conv2d(planes[2], planes[3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes[3]),
            nn.ReLU(),
        )
        
        # 計算最終特徵圖大小
        feature_size = planes[3] * 7 * 7
        
        # 多層全連接網路進行嵌入
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_size // 2, out_dim)
        )
        
        self.quantize = quantize
        self.scale = None

    def forward(self, x, apply_quantize=None):
        if apply_quantize is None:
            apply_quantize = self.quantize
            
        features = self.model(x)
        emb = self.fc(features.view(features.size(0), -1))
        
        if apply_quantize:
            # 量化 embedding
            quantized_emb, self.scale = quantize_to_int8(emb, self.scale)
            # 立即反量化回浮點，以便後續處理
            emb = dequantize_from_int8(quantized_emb, self.scale)
        return emb

class EnhancedBayesianMANN(nn.Module):
    def __init__(self, dataset, out_dim=1024, quantize=False):
        super().__init__()
        in_channels = 1 if dataset in ['omniglot', 'mnist'] else 3
        
        # 針對不同數據集調整通道數
        if dataset in ['omniglot', 'mnist']:
            base_filters = 64
            planes = [base_filters, base_filters*2, base_filters*2, base_filters*4]
        else:  # CIFAR-10 使用更多特徵圖
            base_filters = 64
            planes = [base_filters, base_filters*2, base_filters*4, base_filters*8]
        
        # 第一個卷積區塊
        self.conv1a = Conv2dFlipout(in_channels, planes[0], kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(planes[0])
        self.conv1b = Conv2dFlipout(planes[0], planes[0], kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(planes[0])
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14
        
        # 第二個卷積區塊
        self.conv2a = Conv2dFlipout(planes[0], planes[1], kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(planes[1])
        self.conv2b = Conv2dFlipout(planes[1], planes[1], kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(planes[1])
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7
        
        # 第三個卷積區塊
        self.conv3a = Conv2dFlipout(planes[1], planes[2], kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(planes[2])
        self.conv3b = Conv2dFlipout(planes[2], planes[3], kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(planes[3])
        
        self.relu = nn.ReLU()
        
        # 計算最終特徵圖大小
        feature_size = planes[3] * 7 * 7
        
        # 貝葉斯全連接層
        self.fc1 = LinearFlipout(feature_size, feature_size // 2)
        self.fc2 = LinearFlipout(feature_size // 2, out_dim)
        
        self.quantize = quantize
        self.scale = None

    def forward(self, x, sample=True, apply_quantize=None):
        # 第一個區塊
        out, _ = self.conv1a(x, sample)
        out = self.bn1a(out)
        out = self.relu(out)
        out, _ = self.conv1b(out, sample)
        out = self.bn1b(out)
        out = self.relu(out)
        out = self.pool1(out)
        
        # 第二個區塊
        out, _ = self.conv2a(out, sample)
        out = self.bn2a(out)
        out = self.relu(out)
        out, _ = self.conv2b(out, sample)
        out = self.bn2b(out)
        out = self.relu(out)
        out = self.pool2(out)
        
        # 第三個區塊
        out, _ = self.conv3a(out, sample)
        out = self.bn3a(out)
        out = self.relu(out)
        out, _ = self.conv3b(out, sample)
        out = self.bn3b(out)
        out = self.relu(out)
        
        # 全連接層
        out = out.view(out.size(0), -1)
        out, _ = self.fc1(out, sample)
        out = self.relu(out)
        out, _ = self.fc2(out, sample)
        
        if apply_quantize:
            # 量化 embedding
            quantized_emb, self.scale = quantize_to_int8(out, self.scale)
            # 立即反量化回浮點，以便後續處理
            out = dequantize_from_int8(quantized_emb, self.scale)
            
        return out

def bayesian_loss(model, criterion, outputs, targets, kl_weight=1.0):
    return criterion(outputs, targets) + kl_weight * get_kl_loss(model)

def apply_consistent_quantization(support_emb, query_emb):
    """
    Apply consistent quantization to both support and query embeddings
    """
    # Combine embeddings to compute common scale
    all_embs = torch.cat([support_emb, query_emb], dim=0)
    abs_max = torch.max(torch.abs(all_embs)).item()
    scale = 127.0 / (abs_max + 1e-8)
    
    # Apply quantization with common scale
    quantized_support, _ = quantize_to_int8(support_emb, scale)
    quantized_query, _ = quantize_to_int8(query_emb, scale)
    
    # Convert back to float
    support_emb_q = dequantize_from_int8(quantized_support, scale)
    query_emb_q = dequantize_from_int8(quantized_query, scale)
    
    return support_emb_q, query_emb_q



# -------------------------------
# Distance Functions
# -------------------------------
def cosine_similarity(a, b):
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.t())

def apply_consistent_quantization(support_emb, query_emb):
    all_embs = torch.cat([support_emb, query_emb], dim=0)
    abs_max = torch.max(torch.abs(all_embs)).item()
    scale = 127.0 / (abs_max + 1e-8)
    quantized_support, _ = quantize_to_int8(support_emb, scale)
    quantized_query, _ = quantize_to_int8(query_emb, scale)
    support_emb_q = dequantize_from_int8(quantized_support, scale)
    query_emb_q = dequantize_from_int8(quantized_query, scale)
    return support_emb_q, query_emb_q

def hamming_distance(a, b):
    """
    Hamming distance based on sign: 先將 embedding 轉成 sign（>0 為 1，<=0 為 0），
    再計算 sign 不同的比例，最後回傳「相似度」(1-距離)，shape [N, M]。
    """
    a_q, b_q = apply_consistent_quantization(a, b)
    a_sign = (a_q > 0).to(torch.uint8)
    b_sign = (b_q > 0).to(torch.uint8)
    diff = (a_sign.unsqueeze(1) != b_sign.unsqueeze(0)).sum(dim=2)
    sim = 1.0 - diff.float() / a_sign.size(1)
    return sim

def linf_distance(a, b):
    dist = torch.max(torch.abs(a.unsqueeze(1) - b.unsqueeze(0)), dim=2)[0]
    return -dist

def l1_distance(a, b):
    dist = torch.sum(torch.abs(a.unsqueeze(1) - b.unsqueeze(0)), dim=2)
    return -dist
def l2_distance(a, b):
    """L2距離函數"""
    dist = torch.norm(a.unsqueeze(1) - b.unsqueeze(0), p=2, dim=2)
    return -dist  # 轉換為相似度

def l1_similarity_for_training(a, b):
    distances = torch.sum(torch.abs(a.unsqueeze(1) - b.unsqueeze(0)), dim=2)
    max_dist = torch.max(distances) + 1e-8
    similarities = 1.0 - distances / max_dist
    return similarities

def linf_similarity_for_training(a, b):
    abs_diff = torch.abs(a.unsqueeze(1) - b.unsqueeze(0))
    weights = torch.softmax(100.0 * abs_diff, dim=2)
    weighted_diff = (abs_diff * weights).sum(dim=2)
    max_value = torch.max(weighted_diff) + 1e-8
    similarities = 1.0 - weighted_diff / max_value
    return similarities

def lsh_similarity(a, b, num_bits=32*8, random_seed=42):
    torch.manual_seed(random_seed)
    device = a.device
    D = a.size(1)
    hyperplanes = torch.randn(D, num_bits, device=device)
    a_proj = (a @ hyperplanes) > 0
    b_proj = (b @ hyperplanes) > 0
    a_proj = a_proj.int()
    b_proj = b_proj.int()
    matches = (a_proj.unsqueeze(1) == b_proj.unsqueeze(0)).sum(dim=2)
    sim = matches.float() / num_bits
    return sim

# -------------------------------
# Data Utilities
# -------------------------------

def visualize_tsne_embeddings(model, data_by_class, n_way, k_shot, n_query, distance_fns):
    """
    使用t-SNE視覺化不同距離度量下的嵌入向量關係
    """
    model.eval()
    
    # 採樣一個episode用於視覺化
    support_imgs, support_labels, query_imgs, query_labels = sample_episode(data_by_class, n_way, k_shot, n_query)
    support_imgs, support_labels = support_imgs.to(DEVICE), support_labels.to(DEVICE)
    query_imgs, query_labels = query_imgs.to(DEVICE), query_labels.to(DEVICE)
    
    with torch.no_grad():
        # 獲取原始嵌入向量
        support_emb = model(support_imgs)
        query_emb = model(query_imgs)
        
        # 量化嵌入向量到 INT8
        all_embs_original = torch.cat([support_emb, query_emb], dim=0)
        abs_max = torch.max(torch.abs(all_embs_original)).item()
        scale = 127.0 / (abs_max + 1e-8)
        
        # 量化然後反量化所有嵌入向量
        quantized_embs, _ = quantize_to_int8(all_embs_original, scale)
        all_embs = dequantize_from_int8(quantized_embs, scale)
        
        # 準備標籤
        all_labels = torch.cat([support_labels, query_labels]).cpu().numpy()
        is_query = np.concatenate([np.zeros(len(support_labels)), np.ones(len(query_labels))])
        
        # 轉換到CPU
        all_embs_np = all_embs.cpu().numpy()
        
        # 計算距離矩陣 (為每種距離度量)
        distance_matrices = []
        for dist_name, dist_fn in distance_fns:
            print(f"計算 {dist_name} 距離矩陣...")
            
            if dist_name == "Cosine (No Quant)":
                # 使用原始嵌入向量
                original_embs_np = all_embs_original.cpu().numpy()
                scores = dist_fn(torch.from_numpy(original_embs_np).to(DEVICE), 
                                torch.from_numpy(original_embs_np).to(DEVICE)).cpu().numpy()
                distances = 1.0 - scores
                distances = np.maximum(distances, 0.0)
            elif dist_name == "Cosine":
                # 使用量化後的嵌入向量
                scores = dist_fn(torch.from_numpy(all_embs_np).to(DEVICE), 
                                torch.from_numpy(all_embs_np).to(DEVICE)).cpu().numpy()
                distances = 1.0 - scores
                distances = np.maximum(distances, 0.0)
            else:
                # 其他距離度量（已量化）
                scores = dist_fn(torch.from_numpy(all_embs_np).to(DEVICE), 
                                torch.from_numpy(all_embs_np).to(DEVICE)).cpu().numpy()
                distances = -scores
                distances = distances - np.min(distances)
            
            # 最後的安全檢查，確保沒有負值
            if np.any(distances < 0):
                print(f"警告：{dist_name}距離矩陣中存在負值，將其設為0")
                distances = np.maximum(distances, 0.0)
                
            distance_matrices.append((dist_name, distances))
            
        # 為每種距離度量創建一個圖
        plt.figure(figsize=(len(distance_fns)*6, 12))
        
        for idx, (dist_name, distances) in enumerate(distance_matrices):
            # 使用t-SNE進行降維
            print(f"執行 {dist_name} 的t-SNE降維...")
            tsne = TSNE(n_components=2, 
                metric="precomputed", 
                init="random",
                perplexity=15,  # 較小的 perplexity 值
                random_state=42,
                n_iter=2000,  # 增加迭代次數提高穩定性
                learning_rate="auto")  # 自動學習率
            tsne_result = tsne.fit_transform(distances)
            
            # 繪製t-SNE結果
            ax = plt.subplot(1, len(distance_fns), idx+1)
            
            # 添加網格
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 確保有外框
            ax.set_frame_on(True)
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            
            # 創建顏色映射
            cmap = plt.cm.get_cmap('tab10', n_way)
            
            # 分別繪製support和query點
            for class_idx in range(n_way):
                # Support points (圓圈)
                mask_support = (all_labels == class_idx) & (is_query == 0)
                plt.scatter(tsne_result[mask_support, 0], tsne_result[mask_support, 1], 
                           c=[cmap(class_idx)], marker='o', s=100, label=f"Support {class_idx}" if idx == 0 else None)
                
                # Query points (X形)
                mask_query = (all_labels == class_idx) & (is_query == 1)
                plt.scatter(tsne_result[mask_query, 0], tsne_result[mask_query, 1], 
                           c=[cmap(class_idx)], marker='x', s=100, label=f"Query {class_idx}" if idx == 0 else None)
                
                # 移除連接線部分，不再將query點與support點連接
            
            plt.title(f"t-SNE with {dist_name} distance", fontsize=14)
            # 保留坐標軸以顯示網格
            # plt.axis('on')
            
        # 只在第一個子圖顯示圖例
        if len(distance_matrices) > 0:
            plt.figlegend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=n_way*2)
        
        plt.tight_layout()
        plt.savefig(f"tsne_{DATASET}_N{n_way}_K{k_shot}_Q{n_query}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
def visualize_tsne_for_different_training_metrics(trained_models, model_names, data_by_class, n_way, k_shot, n_query, distance_fns):
    """
    使用t-SNE比較不同訓練度量的嵌入空間結構
    
    Args:
        trained_models: 使用不同度量訓練的模型列表
        model_names: 模型名稱列表 (訓練度量名稱)
        data_by_class: 按類別組織的數據
        n_way, k_shot, n_query: 任務設定
        distance_fns: 要測試的距離函數
    """
    # 採樣一個固定的episode，用於所有模型的比較
    support_imgs, support_labels, query_imgs, query_labels = sample_episode(data_by_class, n_way, k_shot, n_query)
    support_imgs, support_labels = support_imgs.to(DEVICE), support_labels.to(DEVICE)
    query_imgs, query_labels = query_imgs.to(DEVICE), query_labels.to(DEVICE)
    
    # 準備標籤
    all_labels = torch.cat([support_labels, query_labels]).cpu().numpy()
    is_query = np.concatenate([np.zeros(len(support_labels)), np.ones(len(query_labels))])
    
    # 為每個模型創建一個大圖
    for model_idx, (model, model_name) in enumerate(zip(trained_models, model_names)):
        model.eval()
        print(f"\n正在為 {model_name} 訓練的模型生成t-SNE可視化...")
        
        with torch.no_grad():
            # 獲取嵌入向量
            support_emb = model(support_imgs)
            query_emb = model(query_imgs)
            
            # 量化嵌入向量
            all_embs_original = torch.cat([support_emb, query_emb], dim=0)
            abs_max = torch.max(torch.abs(all_embs_original)).item()
            scale = 127.0 / (abs_max + 1e-8)
            
            quantized_embs, _ = quantize_to_int8(all_embs_original, scale)
            all_embs = dequantize_from_int8(quantized_embs, scale)
            all_embs_np = all_embs.cpu().numpy()
            original_embs_np = all_embs_original.cpu().numpy()
            
            # 計算距離矩陣
            distance_matrices = []
            for dist_name, dist_fn in distance_fns:
                if dist_name == "Cosine (No Quant)":
                    # 使用原始嵌入向量
                    scores = dist_fn(torch.from_numpy(original_embs_np).to(DEVICE), 
                                    torch.from_numpy(original_embs_np).to(DEVICE)).cpu().numpy()
                    distances = 1.0 - scores
                    distances = np.maximum(distances, 0.0)
                elif dist_name == "Cosine":
                    # 使用量化後的嵌入向量
                    scores = dist_fn(torch.from_numpy(all_embs_np).to(DEVICE), 
                                    torch.from_numpy(all_embs_np).to(DEVICE)).cpu().numpy()
                    distances = 1.0 - scores
                    distances = np.maximum(distances, 0.0)
                else:
                    # 其他距離度量
                    scores = dist_fn(torch.from_numpy(all_embs_np).to(DEVICE), 
                                    torch.from_numpy(all_embs_np).to(DEVICE)).cpu().numpy()
                    distances = -scores
                    distances = distances - np.min(distances)
                
                # 確保沒有負值
                if np.any(distances < 0):
                    distances = np.maximum(distances, 0.0)
                    
                distance_matrices.append((dist_name, distances))
            
            # 為每種距離度量創建t-SNE可視化
            plt.figure(figsize=(len(distance_fns)*6, 12))
            
            for idx, (dist_name, distances) in enumerate(distance_matrices):
                # t-SNE降維
                tsne = TSNE(n_components=2, 
                    metric="precomputed", 
                    init="random",
                    perplexity=15,
                    random_state=42,
                    n_iter=2000,
                    learning_rate="auto")
                tsne_result = tsne.fit_transform(distances)
                
                # 繪製結果
                ax = plt.subplot(1, len(distance_fns), idx+1)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_frame_on(True)
                
                # 創建顏色映射
                cmap = plt.cm.get_cmap('tab10', n_way)
                
                # 分別繪製support和query點
                for class_idx in range(n_way):
                    # Support points
                    mask_support = (all_labels == class_idx) & (is_query == 0)
                    plt.scatter(tsne_result[mask_support, 0], tsne_result[mask_support, 1], 
                              c=[cmap(class_idx)], marker='o', s=100, 
                              label=f"Support {class_idx}" if idx == 0 else None)
                    
                    # Query points
                    mask_query = (all_labels == class_idx) & (is_query == 1)
                    plt.scatter(tsne_result[mask_query, 0], tsne_result[mask_query, 1], 
                              c=[cmap(class_idx)], marker='x', s=100, 
                              label=f"Query {class_idx}" if idx == 0 else None)
                
                plt.title(f"{model_name} - {dist_name}", fontsize=14)
            
            # 添加圖例
            if len(distance_matrices) > 0:
                plt.figlegend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=n_way*2)
            
            plt.tight_layout()
            plt.savefig(f"tsne_{model_name.lower().replace(' ', '_').replace('-', '_')}_{DATASET}.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    # 為每種距離度量創建不同訓練方法的比較圖
    for dist_idx, (dist_name, _) in enumerate(distance_fns):
        plt.figure(figsize=(len(trained_models)*5, 10))
        print(f"\n為 {dist_name} 距離度量創建不同訓練方法的比較...")
        
        for model_idx, (model, model_name) in enumerate(zip(trained_models, model_names)):
            model.eval()
            
            with torch.no_grad():
                # 獲取嵌入向量
                support_emb = model(support_imgs)
                query_emb = model(query_imgs)
                
                # 量化嵌入向量
                all_embs_original = torch.cat([support_emb, query_emb], dim=0)
                abs_max = torch.max(torch.abs(all_embs_original)).item()
                scale = 127.0 / (abs_max + 1e-8)
                
                quantized_embs, _ = quantize_to_int8(all_embs_original, scale)
                all_embs = dequantize_from_int8(quantized_embs, scale)
                all_embs_np = all_embs.cpu().numpy()
                original_embs_np = all_embs_original.cpu().numpy()
                
                # 計算特定距離度量的距離矩陣
                dist_fn = distance_fns[dist_idx][1]
                if dist_name == "Cosine (No Quant)":
                    scores = dist_fn(torch.from_numpy(original_embs_np).to(DEVICE), 
                                   torch.from_numpy(original_embs_np).to(DEVICE)).cpu().numpy()
                    distances = 1.0 - scores
                else:
                    scores = dist_fn(torch.from_numpy(all_embs_np).to(DEVICE), 
                                   torch.from_numpy(all_embs_np).to(DEVICE)).cpu().numpy()
                    distances = 1.0 - scores if dist_name == "Cosine" else -scores
                
                # 確保沒有負值
                if np.any(distances < 0):
                    distances = distances - np.min(distances)
                
                # t-SNE降維
                tsne = TSNE(n_components=2, 
                    metric="precomputed", 
                    init="random",
                    perplexity=15,
                    random_state=42,
                    n_iter=2000,
                    learning_rate="auto")
                tsne_result = tsne.fit_transform(distances)
                
                # 繪製結果
                ax = plt.subplot(1, len(trained_models), model_idx+1)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 分別繪製support和query點
                for class_idx in range(n_way):
                    # Support points
                    mask_support = (all_labels == class_idx) & (is_query == 0)
                    plt.scatter(tsne_result[mask_support, 0], tsne_result[mask_support, 1], 
                              c=[cmap(class_idx)], marker='o', s=100, 
                              label=f"Support {class_idx}" if model_idx == 0 else None)
                    
                    # Query points
                    mask_query = (all_labels == class_idx) & (is_query == 1)
                    plt.scatter(tsne_result[mask_query, 0], tsne_result[mask_query, 1], 
                              c=[cmap(class_idx)], marker='x', s=100, 
                              label=f"Query {class_idx}" if model_idx == 0 else None)
                
                plt.title(f"{dist_name} - {model_name}", fontsize=14)
        
        # 添加圖例
        plt.figlegend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=n_way*2)
        
        plt.tight_layout()
        plt.savefig(f"tsne_comparison_{dist_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{DATASET}.png", dpi=300, bbox_inches='tight')
        plt.show()

# 添加量化相關函數
def quantize_to_int8(tensor, scale=None):
    '''
    將浮點張量量化為int8
    
    Args:
        tensor: 輸入張量
        scale: 縮放因子，如果為None則動態計算
        
    Returns:
        量化後的int8張量和縮放因子
    '''
    
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
    '''
    將int8張量反量化為浮點
    
    Args:
        quantized: 量化後的int8張量
        scale: 縮放因子
        
    Returns:
        反量化後的浮點張量
    '''
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
# Training & Testing Functions
# -------------------------------
def train_model(model, data_by_class, n_way, k_shot, n_query, num_epochs, episodes_per_epoch, 
               optimizer, criterion, use_noise=False, noise_std=0.1, is_bayesian=False, distance_fn=cosine_similarity):
    return train_model_with_progress(model, data_by_class, n_way, k_shot, n_query, num_epochs, 
                                   episodes_per_epoch, optimizer, criterion, use_noise, noise_std, is_bayesian, distance_fn)

# 添加進度監控的訓練函數
def train_model_with_progress(model, data_by_class, n_way, k_shot, n_query, num_epochs, 
                             episodes_per_epoch, optimizer, criterion, use_noise=False, 
                             noise_std=0.1, is_bayesian=False, distance_fn=cosine_similarity,
                             noise_multiplier_mean=2.0, noise_multiplier_std=0.5):
    """
    帶進度監控的訓練函數
    
    Args:
        noise_multiplier_mean: 高斯分佈倍率的均值
        noise_multiplier_std: 高斯分佈倍率的標準差
    """
    model.train()
    for epoch in range(num_epochs):
        if epoch % 10 == 0:  # 每10個epoch顯示一次進度
            print(f"    Epoch {epoch}/{num_epochs}")
            
        for episode in range(episodes_per_epoch):
            try:
                support_imgs, support_labels, query_imgs, query_labels = sample_episode(
                    data_by_class, n_way, k_shot, n_query)
                support_imgs, support_labels = support_imgs.to(DEVICE), support_labels.to(DEVICE)
                query_imgs, query_labels = query_imgs.to(DEVICE), query_labels.to(DEVICE)
                
                support_emb = model(support_imgs)
                query_emb = model(query_imgs)
                
                if use_noise:
                    # 為每個嵌入向量的每個維度生成高斯倍率
                    # 使用乘法噪聲而非加法噪聲
                    gaussian_multiplier_support = torch.normal(
                        mean=noise_multiplier_mean, 
                        std=noise_multiplier_std, 
                        size=support_emb.shape, 
                        device=DEVICE
                    )
                    
                    gaussian_multiplier_query = torch.normal(
                        mean=noise_multiplier_mean, 
                        std=noise_multiplier_std, 
                        size=query_emb.shape, 
                        device=DEVICE
                    )
                    
                    # 確保倍率為正數（可選：也可以允許負值）
                    gaussian_multiplier_support = torch.clamp(gaussian_multiplier_support, min=0.1)
                    gaussian_multiplier_query = torch.clamp(gaussian_multiplier_query, min=0.1)
                    
                    # 使用乘法噪聲：embedding = embedding * gaussian_multiplier
                    support_emb = support_emb * gaussian_multiplier_support
                    query_emb = query_emb * gaussian_multiplier_query

                # 使用指定的距離函數
                scores = distance_fn(query_emb, support_emb)
                
                if is_bayesian:
                    beta = min(1, (epoch + 1) / num_epochs) / 100000
                    loss = bayesian_loss(model, criterion, scores, query_labels, beta)
                else:
                    loss = criterion(scores, query_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            except Exception as e:
                print(f"    Episode {episode} 錯誤: {str(e)}")
                continue

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
                
                # 獲取原始嵌入向量
                support_emb = model(support_imgs)
                query_emb = model(query_imgs)
                
                if use_noise:
                    support_emb = support_emb + torch.randn_like(support_emb) * noise_std
                    query_emb = query_emb + torch.randn_like(query_emb) * noise_std

                # 量化嵌入向量到 INT8
                # 使用相同的縮放因子以確保一致性
                all_embs = torch.cat([support_emb, query_emb], dim=0)
                abs_max = torch.max(torch.abs(all_embs)).item()
                scale = 127.0 / (abs_max + 1e-8)
                
                # 量化然後反量化 support embeddings
                quantized_support, _ = quantize_to_int8(support_emb, scale)
                support_emb = dequantize_from_int8(quantized_support, scale)
                
                # 量化然後反量化 query embeddings
                quantized_query, _ = quantize_to_int8(query_emb, scale)
                query_emb = dequantize_from_int8(quantized_query, scale)

                # 使用指定的距離函數計算分數
                scores = distance_fn(query_emb, support_emb)
                _, max_indices = torch.max(scores, dim=1)
                pred = support_labels[max_indices]
                total_correct += (pred == query_labels).sum().item()
                total_num += query_labels.size(0)
        
        acc = total_correct / total_num * 100.0
        acc_list.append(acc)
    return np.array(acc_list)

def test_model_with_enforced_quantization(model, data_by_class, n_way, k_shot, n_query, 
                                          test_episodes, distance_fn):
    """
    Test function that enforces quantization for all distance metrics
    """
    model.eval()
    total_correct, total_num = 0, 0
    
    with torch.no_grad():
        for _ in range(test_episodes):
            # Sample episode
            support_imgs, support_labels, query_imgs, query_labels = sample_episode(
                data_by_class, n_way, k_shot, n_query)
            support_imgs, support_labels = support_imgs.to(DEVICE), support_labels.to(DEVICE)
            query_imgs, query_labels = query_imgs.to(DEVICE), query_labels.to(DEVICE)
            
            # Get embeddings
            support_emb = model(support_imgs)
            query_emb = model(query_imgs)
            
            # Apply consistent quantization
            support_emb_q, query_emb_q = apply_consistent_quantization(support_emb, query_emb)
            
            # Calculate scores using the specified distance function
            scores = distance_fn(query_emb_q, support_emb_q)
            
            # Get predictions
            _, max_indices = torch.max(scores, dim=1)
            pred = support_labels[max_indices]
            total_correct += (pred == query_labels).sum().item()
            total_num += query_labels.size(0)
    
    acc = total_correct / total_num * 100.0
    return acc

def test_model_without_quantization(model, data_by_class, n_way, k_shot, n_query, 
                                  test_episodes, distance_fn):
    """
    測試函數，直接使用原始嵌入向量而不進行量化
    """
    model.eval()
    total_correct, total_num = 0, 0
    
    with torch.no_grad():
        for _ in range(test_episodes):
            # 採樣 episode
            support_imgs, support_labels, query_imgs, query_labels = sample_episode(
                data_by_class, n_way, k_shot, n_query)
            support_imgs, support_labels = support_imgs.to(DEVICE), support_labels.to(DEVICE)
            query_imgs, query_labels = query_imgs.to(DEVICE), query_labels.to(DEVICE)
            
            # 獲取嵌入向量（不做量化）
            support_emb = model(support_imgs)
            query_emb = model(query_imgs)
            
            # 使用指定的距離函數計算分數
            scores = distance_fn(query_emb, support_emb)
            
            # 獲取預測結果
            _, max_indices = torch.max(scores, dim=1)
            pred = support_labels[max_indices]
            total_correct += (pred == query_labels).sum().item()
            total_num += query_labels.size(0)
    
    acc = total_correct / total_num * 100.0
    return acc

## -------------------------------
# Dataset Preparation
# -------------------------------
if DATASET == 'omniglot':
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x)
    ])
    train_dataset = torchvision.datasets.Omniglot(root='.data', background=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.Omniglot(root='.data', background=False, download=True, transform=transform)
elif DATASET == 'cifar-10':
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CIFAR10(root='.data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='.data', train=False, download=True, transform=transform)
elif DATASET == 'mnist':
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='.mnist', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='.mnist', train=False, transform=transform, download=True)

train_data = organize_by_class(train_dataset)
test_data = organize_by_class(test_dataset)

# -------------------------------
# 實驗 function 區塊
# -------------------------------
    
def experiment_training_methods_comparison():
    """
    比較三種訓練方法：
    1. 普通MANN (訓練時使用cosine similarity)
    2. Noise-aware training (訓練時在embedding使用高斯乘法噪聲後再用cosine similarity)
    3. Bayesian NN (使用貝葉斯層和KL loss)
    測試時都不加噪聲，但使用不同的距離度量
    """
    print("\n=== 訓練方法比較實驗 ===")
    print(f"實驗設定: NUM_EPOCHS={NUM_EPOCHS}, EPISODES_PER_EPOCH={EPISODES_PER_EPOCH}")
    print(f"TEST_EPISODES={TEST_EPISODES}, NUM_ITER_AVG={NUM_ITER_AVG}")
    
    # 定義訓練方法
    training_methods = [
        ("Standard MANN", "standard"),
        ("Noise-Aware MANN", "noise_aware"), 
        ("Bayesian MANN", "bayesian")
    ]
    
    # 測試距離度量 - 增加LSH
    test_distance_fns = [
        ("Cosine", cosine_similarity),
        ("L-2", l2_distance),
        ("L-1", l1_distance),
        ("L-inf", linf_distance),
        ("LSH", lambda a, b: lsh_similarity(a, b, num_bits=OUTPUT_DIM*8)),
    ]
    
    # 結果存儲 [training_method, test_metric, iteration]
    all_results = np.zeros((len(training_methods), len(test_distance_fns), NUM_ITER_AVG))
    
    # 乘法噪聲設定
    noise_multiplier_mean = 1.0    # 高斯分佈倍率的均值（1.0表示不變）
    noise_multiplier_std = 0.2     # 高斯分佈倍率的標準差
    
    print(f"Noise-Aware training 設定:")
    print(f"  倍率分佈: N({noise_multiplier_mean}, {noise_multiplier_std}²)")
    print(f"  每個維度都會乘以從此分佈採樣的倍率")
    print(f"  理論倍率範圍約: [{noise_multiplier_mean - 2*noise_multiplier_std:.2f}, {noise_multiplier_mean + 2*noise_multiplier_std:.2f}] (95%)")
    
    for train_idx, (train_name, train_type) in enumerate(training_methods):
        print(f"\n==== 訓練方法: {train_name} ====")
        
        for it in range(NUM_ITER_AVG):
            print(f"迭代 {it+1}/{NUM_ITER_AVG}")
            
            # 確保每次迭代後清理GPU記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            try:
                # 根據訓練方法選擇模型
                if train_type == "bayesian":
                    # 使用貝葉斯模型
                    if DATASET in ['omniglot', 'mnist']:
                        model = BayesianMANN(dataset=DATASET, out_dim=OUTPUT_DIM).to(DEVICE)
                    else:
                        model = EnhancedBayesianMANN(dataset=DATASET, out_dim=OUTPUT_DIM).to(DEVICE)
                else:
                    # 使用普通模型
                    if DATASET in ['omniglot', 'mnist']:
                        model = MANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
                    else:
                        model = EnhancedMANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
                
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                criterion = nn.CrossEntropyLoss()
                
                # 訓練模型
                print(f"  開始訓練...")
                if train_type == "standard":
                    # 標準訓練
                    train_model_with_progress(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, 
                                            EPISODES_PER_EPOCH, optimizer, criterion, 
                                            use_noise=False, is_bayesian=False, distance_fn=cosine_similarity)
                                           
                elif train_type == "noise_aware":
                    # 乘法噪聲訓練
                    train_model_with_progress(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, 
                                            EPISODES_PER_EPOCH, optimizer, criterion, 
                                            use_noise=True, noise_std=None,  # 不再使用noise_std
                                            is_bayesian=False, distance_fn=cosine_similarity,
                                            noise_multiplier_mean=noise_multiplier_mean,
                                            noise_multiplier_std=noise_multiplier_std)
                                           
                elif train_type == "bayesian":
                    # 貝葉斯訓練
                    train_model_with_progress(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, 
                                            EPISODES_PER_EPOCH, optimizer, criterion, 
                                            use_noise=False, is_bayesian=True, distance_fn=cosine_similarity)
                
                print(f"  訓練完成，開始測試...")
                
                # 測試不同距離度量
                iteration_results = []
                for test_idx, (test_name, test_fn) in enumerate(test_distance_fns):
                    print(f"    測試 {test_name}")
                    
                    # 量化的測試
                    acc = test_model_with_enforced_quantization(
                        model, test_data, N_WAY, K_SHOT, N_QUERY, TEST_EPISODES, test_fn)
                    
                    all_results[train_idx, test_idx, it] = acc
                    iteration_results.append(acc)
                    print(f"      準確率: {acc:.2f}%")
                
                # 清理當前模型以釋放記憶體
                del model, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"迭代 {it+1} 發生錯誤: {str(e)}")
                # 填入NaN值以保持結果矩陣完整性
                for test_idx in range(len(test_distance_fns)):
                    all_results[train_idx, test_idx, it] = np.nan
                continue
        
        # 打印當前訓練方法的結果（忽略NaN值）
        valid_results = all_results[train_idx, :, ~np.isnan(all_results[train_idx, 0, :])]
        if valid_results.size > 0:
            means = np.nanmean(all_results[train_idx], axis=1)
            stds = np.nanstd(all_results[train_idx], axis=1)
            print(f"\n{train_name} 結果:")
            for i, (test_name, _) in enumerate(test_distance_fns):
                if not np.isnan(means[i]):
                    print(f"  {test_name}: {means[i]:.2f}% ± {stds[i]:.2f}%")
                else:
                    print(f"  {test_name}: 無有效結果")
    
    # 保存結果
    print("\n保存實驗結果...")
    np.savez(f"{DATASET}_Noise-Aware_noise_training_comparison.npz",
             training_methods=[name for name, _ in training_methods],
             test_metrics=[name for name, _ in test_distance_fns],
             results=all_results,
             noise_multiplier_mean=noise_multiplier_mean,
             noise_multiplier_std=noise_multiplier_std)
    
    # 計算平均結果和標準差（處理NaN值）
    mean_results = np.nanmean(all_results, axis=2)
    std_results = np.nanstd(all_results, axis=2)
    
    # 創建詳細結果DataFrame
    detailed_results = []
    for i, (train_name, _) in enumerate(training_methods):
        for j, (test_name, _) in enumerate(test_distance_fns):
            detailed_results.append({
                'Training_Method': train_name,
                'Test_Metric': test_name,
                'Mean_Accuracy': mean_results[i, j] if not np.isnan(mean_results[i, j]) else None,
                'Std_Dev': std_results[i, j] if not np.isnan(std_results[i, j]) else None
            })
    
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{DATASET}_Noise-Aware_training_detailed_results.csv", index=False)
    
    # 改善的可視化（增強字體樣式）
    print("\n生成結果可視化...")
    valid_mask = ~np.isnan(mean_results)
    
    if np.any(valid_mask):
        plt.figure(figsize=(16, 12))
        
        # 設定全局字體大小
        plt.rcParams.update({'font.size': 14})
        
        # 子圖2: 柱狀圖（改善樣式）
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bar_width = 0.25
        x = np.arange(len(test_distance_fns))
        
        for i, (train_name, _) in enumerate(training_methods):
            valid_indices = ~np.isnan(mean_results[i, :])
            if np.any(valid_indices):
                y_vals = np.nan_to_num(mean_results[i, :], nan=0)
                y_errs = np.nan_to_num(std_results[i, :], nan=0)
                plt.bar(x + i*bar_width - bar_width, y_vals,
                        yerr=y_errs, capsize=5, width=bar_width,
                        color=colors[i], label=train_name, alpha=0.8)
        
        plt.xlabel('Test Distance Metric', fontweight='bold', fontsize=14)
        plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=14)
        plt.ylim(70, 80)
        plt.title('Performance Comparison', fontweight='bold', fontsize=16)
        plt.xticks(x, [name for name, _ in test_distance_fns], 
                  rotation=45, fontweight='bold', fontsize=12)
        plt.yticks(fontweight='bold', fontsize=12)
        plt.legend(prop={'weight': 'bold', 'size': 12})
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 添加噪聲設定信息到圖上
        info_text = f'Noise-Aware : Multiplier~N({noise_multiplier_mean},{noise_multiplier_std}²)'
        plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic')
        
        plt.savefig(f"{DATASET}_Noise-Aware_training_results.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # 打印總結
    print("\n=== 實驗總結 ===")
    print(f"Noise-Aware training 使用設定:")
    print(f"  倍率分佈: N({noise_multiplier_mean}, {noise_multiplier_std}²)")
    print(f"  噪聲類型: 乘法噪聲 (embedding * gaussian_multiplier)")
    
    for i, (train_name, _) in enumerate(training_methods):
        valid_results = mean_results[i, ~np.isnan(mean_results[i, :])]
        if len(valid_results) > 0:
            best_idx = np.nanargmax(mean_results[i, :])
            if not np.isnan(mean_results[i, best_idx]):
                best_metric = test_distance_fns[best_idx][0]
                best_acc = mean_results[i, best_idx]
                best_std = std_results[i, best_idx]
                print(f"\n{train_name}:")
                print(f"  最佳測試度量: {best_metric} ({best_acc:.2f}% ± {best_std:.2f}%)")
        else:
            print(f"\n{train_name}: 無有效結果")
    
    # 重置字體設定
    plt.rcParams.update(plt.rcParamsDefault)
    
    return all_results, mean_results, std_results

def experiment_snr_analysis():
    """
    SNR分析實驗：
    1. 用cosine similarity訓練一個模型
    2. 計算多種距離度量：cosine, L1, L2, L-inf, LSH
    3. 將所有距離映射到[0,1]範圍
    4. 以cosine distance作為信號，其他距離作為噪聲，計算SNR = |noisy signal - signal|/1
    """
    print("\n=== 信噪比(SNR)分析實驗 ===")
    print(f"實驗設定: NUM_EPOCHS={NUM_EPOCHS}, EPISODES_PER_EPOCH={EPISODES_PER_EPOCH}")
    
    distance_functions = [
        ("Cosine", cosine_similarity),
        ("L1", l1_distance), 
        ("L2", l2_distance),
        ("L-inf", linf_distance),
        ("LSH", lambda a, b: lsh_similarity(a, b, num_bits=OUTPUT_DIM*8))
    ]
    
    # 存儲每個迭代的詳細SNR結果
    all_iteration_snr_values = []  # 每個迭代的SNR值分布
    snr_means_per_iteration = np.zeros((NUM_ITER_AVG, len(distance_functions)-1))
    all_distances_normalized = []
    
    for iteration in range(NUM_ITER_AVG):
        print(f"\n迭代 {iteration+1}/{NUM_ITER_AVG}")
        
        # 清理GPU記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            # 創建並訓練模型
            if DATASET in ['omniglot', 'mnist']:
                model = MANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
            else:
                model = EnhancedMANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            print("  訓練模型...")
            train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, 
                       EPISODES_PER_EPOCH, optimizer, criterion, 
                       use_noise=False, is_bayesian=False, distance_fn=cosine_similarity)
            
            print("  收集距離數據...")
            model.eval()
            
            # 收集大量測試episodes的距離數據
            all_distance_matrices = {name: [] for name, _ in distance_functions}
            
            with torch.no_grad():
                for episode in range(TEST_EPISODES):
                    if episode % 100 == 0:
                        print(f"    處理episode {episode}/{TEST_EPISODES}")
                    
                    # 採樣episode
                    support_imgs, support_labels, query_imgs, query_labels = sample_episode(
                        test_data, N_WAY, K_SHOT, N_QUERY)
                    support_imgs = support_imgs.to(DEVICE)
                    query_imgs = query_imgs.to(DEVICE)
                    
                    # 獲取嵌入向量
                    support_emb = model(support_imgs)
                    query_emb = model(query_imgs)
                    
                    # 量化嵌入向量（保持一致性）
                    support_emb_q, query_emb_q = apply_consistent_quantization(support_emb, query_emb)
                    
                    # 計算所有距離度量
                    for dist_name, dist_fn in distance_functions:
                        if dist_name == "Cosine":
                            # 對於cosine，我們計算距離（1-相似度）
                            similarity_matrix = dist_fn(query_emb_q, support_emb_q)
                            distance_matrix = 1.0 - similarity_matrix
                        else:
                            # 對於其他距離，轉換負的相似度為正的距離
                            similarity_matrix = dist_fn(query_emb_q, support_emb_q)
                            distance_matrix = -similarity_matrix
                        
                        # 收集距離值（展平為一維）
                        distances_flat = distance_matrix.cpu().numpy().flatten()
                        all_distance_matrices[dist_name].extend(distances_flat)
            
            print("  計算SNR...")
            
            # 將所有距離轉換為numpy數組
            for dist_name in all_distance_matrices:
                all_distance_matrices[dist_name] = np.array(all_distance_matrices[dist_name])
            
            # 歸一化所有距離到[0,1]範圍
            normalized_distances = {}
            for dist_name, distances in all_distance_matrices.items():
                # 使用min-max歸一化
                min_val = np.min(distances)
                max_val = np.max(distances)
                if max_val > min_val:
                    normalized = (distances - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros_like(distances)
                normalized_distances[dist_name] = normalized
                
                print(f"    {dist_name}: Original range [{min_val:.4f}, {max_val:.4f}] -> Normalized range [{np.min(normalized):.4f}, {np.max(normalized):.4f}]")
            
            # 以Cosine distance作為信號
            signal = normalized_distances["Cosine"]
            
            # 存儲這個迭代的SNR值
            iteration_snr_values = {}
            
            # 計算每個其他距離度量與cosine的SNR
            iteration_snr_means = []
            for i, (dist_name, _) in enumerate(distance_functions[1:]):  # 跳過cosine
                noisy_signal = normalized_distances[dist_name]
                
                # 計算SNR = |noisy signal - signal| / 1
                snr_values = np.abs(noisy_signal - signal) / 1.0
                
                # 計算這個距離度量的平均SNR
                mean_snr = np.mean(snr_values)
                iteration_snr_means.append(mean_snr)
                
                # 存儲所有SNR值用於分布分析
                iteration_snr_values[dist_name] = snr_values
                
                print(f"    SNR (|{dist_name} - Cosine|/1): 平均 = {mean_snr:.4f}, 標準差 = {np.std(snr_values):.4f}")
            
            snr_means_per_iteration[iteration] = iteration_snr_means
            all_iteration_snr_values.append(iteration_snr_values)
            
            # 保存第一次迭代的歸一化距離（用於可視化）
            if iteration == 0:
                all_distances_normalized = normalized_distances.copy()
            
            # 清理記憶體
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"迭代 {iteration+1} 發生錯誤: {str(e)}")
            snr_means_per_iteration[iteration] = np.full(len(distance_functions)-1, np.nan)
            all_iteration_snr_values.append({})
            continue
    
    # 計算總體統計
    mean_snr = np.nanmean(snr_means_per_iteration, axis=0)
    std_snr = np.nanstd(snr_means_per_iteration, axis=0)
    
    print("\n=== SNR分析結果 (|noisy signal - signal|/1) ===")
    for i, (dist_name, _) in enumerate(distance_functions[1:]):
        print(f"{dist_name}: 平均SNR = {mean_snr[i]:.4f} ± {std_snr[i]:.4f}")
    
    # 保存結果
    np.savez(f"{DATASET}_snr_analysis.npz",
             distance_names=[name for name, _ in distance_functions[1:]],
             snr_means_per_iteration=snr_means_per_iteration,
             mean_snr=mean_snr,
             std_snr=std_snr,
             all_iteration_snr_values=all_iteration_snr_values)
    
    # 可視化結果 - 分開繪製
    print("\n生成SNR可視化...")
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 圖1: Mean SNR Comparison
    plt.figure(figsize=(12, 8))
    x_pos = np.arange(len(distance_functions)-1)
    
    bars = plt.bar(x_pos, mean_snr, yerr=std_snr, capsize=5, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.xlabel('Distance Metric', fontweight='bold', fontsize=14)
    plt.ylabel('SNR (|noisy-signal|/1)', fontweight='bold', fontsize=14)
    plt.title('Mean SNR Comparison\n(|noisy signal - signal|/1)', fontweight='bold', fontsize=16)
    plt.xticks(x_pos, [name for name, _ in distance_functions[1:]], rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # 在柱子上添加數值標籤
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, mean_snr, std_snr)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.002,
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{DATASET}_snr_mean_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 圖2-5: 每個距離度量的SNR值分布直方圖
    if all_iteration_snr_values and len(all_iteration_snr_values) > 0:
        first_iteration_snr = all_iteration_snr_values[0]
        
        for idx, (dist_name, _) in enumerate(distance_functions[1:]):
            if dist_name in first_iteration_snr and len(first_iteration_snr[dist_name]) > 0:
                plt.figure(figsize=(10, 6))
                
                snr_values = first_iteration_snr[dist_name]
                plt.hist(snr_values, bins=50, alpha=0.7, color=colors[idx], 
                        density=True, edgecolor='black', linewidth=0.5)
                
                # 添加統計信息
                mean_val = np.mean(snr_values)
                std_val = np.std(snr_values)
                median_val = np.median(snr_values)
                
                plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {mean_val:.3f}')
                plt.axvline(median_val, color='blue', linestyle='-.', linewidth=2, 
                           label=f'Median: {median_val:.3f}')
                plt.axvline(mean_val + std_val, color='red', linestyle=':', linewidth=1, alpha=0.7)
                plt.axvline(mean_val - std_val, color='red', linestyle=':', linewidth=1, alpha=0.7)
                
                plt.xlabel('SNR Value', fontweight='bold', fontsize=14)
                plt.ylabel('Density', fontweight='bold', fontsize=14)
                plt.title(f'{dist_name} SNR Distribution\n(First Iteration)', fontweight='bold', fontsize=16)
                plt.legend(fontsize=12)
                plt.grid(alpha=0.3)
                
                # 添加統計文本框
                stats_text = f'Statistics:\nMean: {mean_val:.4f}\nStd: {std_val:.4f}\nMedian: {median_val:.4f}\nMin: {np.min(snr_values):.4f}\nMax: {np.max(snr_values):.4f}'
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(f"{DATASET}_snr_distribution_{dist_name.lower().replace('-', '_')}.png", dpi=300, bbox_inches='tight')
                plt.show()
    
    # 圖6: 所有距離度量的SNR分布比較（箱線圖）
    plt.figure(figsize=(12, 8))
    snr_data_for_boxplot = []
    labels_for_boxplot = []
    
    for i, (dist_name, _) in enumerate(distance_functions[1:]):
        valid_iterations = snr_means_per_iteration[:, i][~np.isnan(snr_means_per_iteration[:, i])]
        if len(valid_iterations) > 0:
            snr_data_for_boxplot.append(valid_iterations)
            labels_for_boxplot.append(dist_name)
    
    if snr_data_for_boxplot:
        bp = plt.boxplot(snr_data_for_boxplot, labels=labels_for_boxplot, patch_artist=True, 
                        showmeans=True, meanline=True)
        
        # 自定義箱線圖顏色
        for patch, color in zip(bp['boxes'], colors[:len(snr_data_for_boxplot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 設置均值線顏色
        for meanline in bp['means']:
            meanline.set_color('red')
            meanline.set_linewidth(2)
    
    plt.xlabel('Distance Metric', fontweight='bold', fontsize=14)
    plt.ylabel('SNR Distribution Across Iterations', fontweight='bold', fontsize=14)
    plt.title('SNR Distribution Comparison\n(Across All Iterations)', fontweight='bold', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{DATASET}_snr_boxplot_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 圖7: SNR隨迭代變化的線圖
    plt.figure(figsize=(12, 8))
    for i, (dist_name, _) in enumerate(distance_functions[1:]):
        valid_snr = snr_means_per_iteration[:, i][~np.isnan(snr_means_per_iteration[:, i])]
        if len(valid_snr) > 0:
            plt.plot(range(1, len(valid_snr)+1), valid_snr, 
                    marker='o', label=dist_name, color=colors[i], 
                    linewidth=2, markersize=8)
    
    plt.xlabel('Iteration', fontweight='bold', fontsize=14)
    plt.ylabel('Mean SNR', fontweight='bold', fontsize=14)
    plt.title('SNR Consistency Across Iterations', fontweight='bold', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{DATASET}_snr_iteration_trends.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 創建詳細結果表格
    detailed_results = []
    for i, (dist_name, _) in enumerate(distance_functions[1:]):
        valid_iterations = snr_means_per_iteration[:, i][~np.isnan(snr_means_per_iteration[:, i])]
        detailed_results.append({
            'Distance_Metric': dist_name,
            'Mean_SNR': mean_snr[i],
            'Std_SNR': std_snr[i],
            'Min_SNR': np.min(valid_iterations) if len(valid_iterations) > 0 else np.nan,
            'Max_SNR': np.max(valid_iterations) if len(valid_iterations) > 0 else np.nan,
            'Median_SNR': np.median(valid_iterations) if len(valid_iterations) > 0 else np.nan,
            'Valid_Iterations': len(valid_iterations)
        })
    
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{DATASET}_snr_analysis_detailed_results.csv", index=False)
    
    # 保存每個迭代的詳細SNR分布數據
    iteration_details = []
    for iteration in range(NUM_ITER_AVG):
        if iteration < len(all_iteration_snr_values):
            for dist_name in all_iteration_snr_values[iteration]:
                snr_values = all_iteration_snr_values[iteration][dist_name]
                iteration_details.append({
                    'Iteration': iteration + 1,
                    'Distance_Metric': dist_name,
                    'Mean_SNR': np.mean(snr_values),
                    'Std_SNR': np.std(snr_values),
                    'Min_SNR': np.min(snr_values),
                    'Max_SNR': np.max(snr_values),
                    'Median_SNR': np.median(snr_values),
                    'Q25_SNR': np.percentile(snr_values, 25),
                    'Q75_SNR': np.percentile(snr_values, 75)
                })
    
    iteration_df = pd.DataFrame(iteration_details)
    iteration_df.to_csv(f"{DATASET}_snr_iteration_details.csv", index=False)
    
    print("\n詳細結果已保存到CSV文件")
    print("\n=== 實驗總結 ===")
    
    # 找出最佳和最差的距離度量（SNR越小越好）
    best_idx = np.argmin(mean_snr)
    worst_idx = np.argmax(mean_snr)
    
    print(f"最低SNR（最相似）: {distance_functions[best_idx + 1][0]} ({mean_snr[best_idx]:.4f} ± {std_snr[best_idx]:.4f})")
    print(f"最高SNR（最不同）: {distance_functions[worst_idx + 1][0]} ({mean_snr[worst_idx]:.4f} ± {std_snr[worst_idx]:.4f})")
    print(f"SNR範圍: {np.max(mean_snr) - np.min(mean_snr):.4f}")
    
    # 打印每個度量的詳細統計
    print("\n詳細統計:")
    for i, (dist_name, _) in enumerate(distance_functions[1:]):
        valid_iterations = snr_means_per_iteration[:, i][~np.isnan(snr_means_per_iteration[:, i])]
        if len(valid_iterations) > 0:
            print(f"{dist_name}:")
            print(f"  平均: {mean_snr[i]:.4f}")
            print(f"  標準差: {std_snr[i]:.4f}")
            print(f"  範圍: [{np.min(valid_iterations):.4f}, {np.max(valid_iterations):.4f}]")
            print(f"  中位數: {np.median(valid_iterations):.4f}")
    
    return snr_means_per_iteration, mean_snr, std_snr, all_iteration_snr_values, all_distances_normalized

def experiment_noise_multiplier_sweep():
    """
    Noise multiplier mean sweep experiment:
    Use only Noise-aware training method, sweep noise multiplier mean from 0 to 3 (step 0.1)
    Test performance of all distance metrics
    """
    print("\n=== Noise Multiplier Mean Sweep Experiment ===")
    print(f"Experiment settings: NUM_EPOCHS={NUM_EPOCHS}, EPISODES_PER_EPOCH={EPISODES_PER_EPOCH}")
    print(f"TEST_EPISODES={TEST_EPISODES}, NUM_ITER_AVG={NUM_ITER_AVG}")
    
    # Define noise multiplier mean range
    noise_means = np.arange(0.0, 3.1, 0.1)  # 0.0 to 3.0, step 0.1
    noise_multiplier_std = 0.2  # Fixed standard deviation
    
    print(f"Noise multiplier mean range: {noise_means[0]:.1f} to {noise_means[-1]:.1f} ({len(noise_means)} values)")
    print(f"Fixed standard deviation: {noise_multiplier_std}")
    
    # Test distance metrics
    test_distance_fns = [
        ("Cosine", cosine_similarity),
        ("L-2", l2_distance), 
        ("L-1", l1_distance),
        ("L-inf", linf_distance),
        ("LSH", lambda a, b: lsh_similarity(a, b, num_bits=OUTPUT_DIM*8)),
    ]
    
    # Result storage [noise_mean_idx, test_metric, iteration]
    all_results = np.zeros((len(noise_means), len(test_distance_fns), NUM_ITER_AVG))
    
    for noise_idx, noise_mean in enumerate(noise_means):
        print(f"\n==== Noise Multiplier Mean: {noise_mean:.1f} ({noise_idx+1}/{len(noise_means)}) ====")
        print(f"Current setting: Multiplier distribution N({noise_mean:.1f}, {noise_multiplier_std}²)")
        
        for it in range(NUM_ITER_AVG):
            print(f"  Iteration {it+1}/{NUM_ITER_AVG}")
            
            # Ensure GPU memory cleanup after each iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            try:
                # Create model
                if DATASET in ['omniglot', 'mnist']:
                    model = MANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
                else:
                    model = EnhancedMANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
                
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                criterion = nn.CrossEntropyLoss()
                
                # Use Noise-aware training
                print(f"    Starting training (noise_mean={noise_mean:.1f})...")
                train_model_with_progress(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, 
                                        EPISODES_PER_EPOCH, optimizer, criterion, 
                                        use_noise=True, noise_std=None,
                                        is_bayesian=False, distance_fn=cosine_similarity,
                                        noise_multiplier_mean=noise_mean,
                                        noise_multiplier_std=noise_multiplier_std)
                
                print(f"    Training completed, starting testing...")
                
                # Test different distance metrics
                iteration_results = []
                for test_idx, (test_name, test_fn) in enumerate(test_distance_fns):
                    print(f"      Testing {test_name}")
                    
                    # Quantized testing
                    acc = test_model_with_enforced_quantization(
                        model, test_data, N_WAY, K_SHOT, N_QUERY, TEST_EPISODES, test_fn)
                    
                    all_results[noise_idx, test_idx, it] = acc
                    iteration_results.append(acc)
                    print(f"        Accuracy: {acc:.2f}%")
                
                # Clean up current model to free memory
                del model, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"    Iteration {it+1} error: {str(e)}")
                # Fill NaN values to maintain result matrix integrity
                for test_idx in range(len(test_distance_fns)):
                    all_results[noise_idx, test_idx, it] = np.nan
                continue
        
        # Print current noise mean results (ignore NaN values)
        valid_results = all_results[noise_idx, :, ~np.isnan(all_results[noise_idx, 0, :])]
        if valid_results.size > 0:
            means = np.nanmean(all_results[noise_idx], axis=1)
            stds = np.nanstd(all_results[noise_idx], axis=1)
            print(f"\n  Noise Mean {noise_mean:.1f} Results:")
            for i, (test_name, _) in enumerate(test_distance_fns):
                if not np.isnan(means[i]):
                    print(f"    {test_name}: {means[i]:.2f}% ± {stds[i]:.2f}%")
                else:
                    print(f"    {test_name}: No valid results")
    
    # Save results
    print("\nSaving experiment results...")
    np.savez(f"{DATASET}_noise_multiplier_sweep.npz",
             noise_means=noise_means,
             noise_multiplier_std=noise_multiplier_std,
             test_metrics=[name for name, _ in test_distance_fns],
             results=all_results)
    
    # Calculate average results and standard deviation (handle NaN values)
    mean_results = np.nanmean(all_results, axis=2)
    std_results = np.nanstd(all_results, axis=2)
    
    # Create detailed results DataFrame
    detailed_results = []
    for i, noise_mean in enumerate(noise_means):
        for j, (test_name, _) in enumerate(test_distance_fns):
            detailed_results.append({
                'Noise_Multiplier_Mean': noise_mean,
                'Test_Metric': test_name,
                'Mean_Accuracy': mean_results[i, j] if not np.isnan(mean_results[i, j]) else None,
                'Std_Dev': std_results[i, j] if not np.isnan(std_results[i, j]) else None
            })
    
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{DATASET}_noise_multiplier_sweep_detailed_results.csv", index=False)
    
    # Visualization - Bar chart with noise means as grouped bars
    print("\nGenerating result visualization...")
    
    # Select specific noise means for cleaner visualization (every 0.5)
    selected_noise_indices = [i for i, nm in enumerate(noise_means) if nm % 0.5 == 0]
    selected_noise_means = noise_means[selected_noise_indices]
    selected_mean_results = mean_results[selected_noise_indices, :]
    selected_std_results = std_results[selected_noise_indices, :]
    
    plt.figure(figsize=(20, 12))
    
    # Color map for different noise means
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_noise_means)))
    
    # Bar width and positions
    bar_width = 0.12
    x_pos = np.arange(len(test_distance_fns))
    
    # Plot bars for each noise mean
    for i, (noise_mean, color) in enumerate(zip(selected_noise_means, colors)):
        # Get results for this noise mean across all metrics
        accuracies = selected_mean_results[i, :]
        errors = selected_std_results[i, :]
        
        # Filter out NaN values
        valid_mask = ~np.isnan(accuracies)
        if np.any(valid_mask):
            positions = x_pos + i * bar_width - (len(selected_noise_means) - 1) * bar_width / 2
            
            bars = plt.bar(positions[valid_mask], accuracies[valid_mask], 
                          width=bar_width, color=color, alpha=0.8,
                          yerr=errors[valid_mask], capsize=3,
                          label=f'Noise Mean = {noise_mean:.1f}',
                          edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Distance Metric', fontweight='bold', fontsize=16)
    plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
    plt.title(f'Performance vs Noise Multiplier Mean\n(std={noise_multiplier_std})', 
              fontweight='bold', fontsize=18)
    
    # Set x-axis labels
    plt.xticks(x_pos, [name for name, _ in test_distance_fns], 
               fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Add grid
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{DATASET}_noise_multiplier_sweep_bar_chart.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Alternative visualization with all noise means (heatmap style bar chart)
    plt.figure(figsize=(25, 10))
    
    # Create a more detailed bar chart with more noise means (every 0.2)
    detailed_noise_indices = [i for i, nm in enumerate(noise_means) if nm % 0.2 == 0]
    detailed_noise_means = noise_means[detailed_noise_indices]
    detailed_mean_results = mean_results[detailed_noise_indices, :]
    detailed_std_results = std_results[detailed_noise_indices, :]
    
    colors_detailed = plt.cm.coolwarm(np.linspace(0, 1, len(detailed_noise_means)))
    
    bar_width_detailed = 0.05
    
    for i, (noise_mean, color) in enumerate(zip(detailed_noise_means, colors_detailed)):
        accuracies = detailed_mean_results[i, :]
        errors = detailed_std_results[i, :]
        
        valid_mask = ~np.isnan(accuracies)
        if np.any(valid_mask):
            positions = x_pos + i * bar_width_detailed - (len(detailed_noise_means) - 1) * bar_width_detailed / 2
            
            plt.bar(positions[valid_mask], accuracies[valid_mask], 
                   width=bar_width_detailed, color=color, alpha=0.8,
                   edgecolor='none')
    
    plt.xlabel('Distance Metric', fontweight='bold', fontsize=16)
    plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
    plt.title(f'Detailed Performance vs Noise Multiplier Mean\n(std={noise_multiplier_std})', 
              fontweight='bold', fontsize=18)
    
    plt.xticks(x_pos, [name for name, _ in test_distance_fns], 
               fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14)
    
    # Add colorbar to show noise mean values
    sm = plt.cm.ScalarMappable(cmap='coolwarm', 
                              norm=plt.Normalize(vmin=detailed_noise_means.min(), 
                                               vmax=detailed_noise_means.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Noise Multiplier Mean', fontweight='bold', fontsize=14)
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{DATASET}_noise_multiplier_sweep_detailed_bar_chart.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical analysis and summary
    print("\n=== Experiment Summary ===")
    
    # Find best noise mean for each distance metric
    best_settings = {}
    for j, (test_name, _) in enumerate(test_distance_fns):
        valid_mask = ~np.isnan(mean_results[:, j])
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            valid_means = mean_results[valid_mask, j]
            best_idx_in_valid = np.argmax(valid_means)
            best_idx_overall = valid_indices[best_idx_in_valid]
            
            best_noise_mean = noise_means[best_idx_overall]
            best_acc = mean_results[best_idx_overall, j]
            best_std = std_results[best_idx_overall, j]
            
            best_settings[test_name] = {
                'noise_mean': best_noise_mean,
                'accuracy': best_acc,
                'std': best_std
            }
            
            print(f"\n{test_name}:")
            print(f"  Best noise mean: {best_noise_mean:.1f}")
            print(f"  Best accuracy: {best_acc:.2f}% ± {best_std:.2f}%")
    
    # Save best settings
    best_settings_df = pd.DataFrame.from_dict(best_settings, orient='index')
    best_settings_df.to_csv(f"{DATASET}_noise_multiplier_best_settings.csv")
    
    # Find global best setting
    all_valid_accs = mean_results[~np.isnan(mean_results)]
    if len(all_valid_accs) > 0:
        global_best_idx = np.unravel_index(np.nanargmax(mean_results), mean_results.shape)
        global_best_noise_mean = noise_means[global_best_idx[0]]
        global_best_metric = test_distance_fns[global_best_idx[1]][0]
        global_best_acc = mean_results[global_best_idx]
        global_best_std = std_results[global_best_idx]
        
        print(f"\nGlobal best setting:")
        print(f"  Noise mean: {global_best_noise_mean:.1f}")
        print(f"  Test metric: {global_best_metric}")
        print(f"  Accuracy: {global_best_acc:.2f}% ± {global_best_std:.2f}%")
    
    print(f"\nDetailed results saved to:")
    print(f"  - {DATASET}_noise_multiplier_sweep.npz")
    print(f"  - {DATASET}_noise_multiplier_sweep_detailed_results.csv")
    print(f"  - {DATASET}_noise_multiplier_best_settings.csv")
    
    return all_results, mean_results, std_results, noise_means, best_settings
# -------------------------------
# 主程式
# -------------------------------
if __name__ == "__main__":
    experiment_training_methods_comparison()
    # experiment_snr_analysis()
    # experiment_noise_multiplier_sweep()
    print("所有實驗、比較和可視化已完成！")