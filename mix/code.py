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
EPISODES_PER_EPOCH = 10
TEST_EPISODES = 500
NOISE_STD_MIN = 1
NOISE_LEVELS = np.concatenate([np.linspace(0, 9, 10), np.logspace(1, 3, 21)])
NUM_NOISE = len(NOISE_LEVELS)
NUM_ITER_AVG = 10
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
               optimizer, criterion, use_noise=False, noise_std=1.0, is_bayesian=False, distance_fn=cosine_similarity):
    return train_model_with_progress(model, data_by_class, n_way, k_shot, n_query, num_epochs, 
                                   episodes_per_epoch, optimizer, criterion, use_noise, noise_std, is_bayesian, distance_fn)

# 添加進度監控的訓練函數
def train_model_with_progress(model, data_by_class, n_way, k_shot, n_query, num_epochs, 
                             episodes_per_epoch, optimizer, criterion, use_noise=False, 
                             noise_std=1.0, is_bayesian=False, distance_fn=cosine_similarity):
    """帶進度監控的訓練函數"""
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
                    support_emb = support_emb + torch.randn_like(support_emb) * noise_std
                    query_emb = query_emb + torch.randn_like(query_emb) * noise_std

                # 使用指定的距離函數
                scores = distance_fn(query_emb, support_emb)
                
                if is_bayesian:
                    beta = min(1, (epoch + 1) / num_epochs) * 0.2
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
    2. Noise-aware training (訓練時在embedding加入高斯噪聲後再用cosine similarity)
    3. Bayesian NN (使用貝葉斯層和KL loss)
    測試時都不加噪聲，但使用不同的距離度量
    """
    print("\n=== 訓練方法比較實驗 ===")
    print(f"實驗設定: NUM_EPOCHS={NUM_EPOCHS}, EPISODES_PER_EPOCH={EPISODES_PER_EPOCH}")
    print(f"TEST_EPISODES={TEST_EPISODES}, NUM_ITER_AVG={NUM_ITER_AVG}")
    
    # 定義訓練方法
    training_methods = [
        ("Standard MANN", "standard"),
        ("Noise-aware MANN", "noise_aware"), 
        ("Bayesian MANN", "bayesian")
    ]
    
    # 測試距離度量 - 增加LSH
    test_distance_fns = [
        ("Cosine", cosine_similarity),
        ("L-1", l1_distance),
        ("L-inf", linf_distance),
        
        ("LSH", lambda a, b: lsh_similarity(a, b, num_bits=OUTPUT_DIM*8)),
        # ("Hamming", hamming_distance),
    ]
    
    # 結果存儲 [training_method, test_metric, iteration]
    all_results = np.zeros((len(training_methods), len(test_distance_fns), NUM_ITER_AVG))
    
    # 噪聲標準差（用於noise-aware training）
    training_noise_std = 1
    
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
                    train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, 
                               EPISODES_PER_EPOCH, optimizer, criterion, 
                               use_noise=False, is_bayesian=False, distance_fn=cosine_similarity)
                               
                elif train_type == "noise_aware":
                    # 噪聲感知訓練
                    train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, 
                               EPISODES_PER_EPOCH, optimizer, criterion, 
                               use_noise=True, noise_std=training_noise_std, 
                               is_bayesian=False, distance_fn=cosine_similarity)
                               
                elif train_type == "bayesian":
                    # 貝葉斯訓練
                    train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, 
                               EPISODES_PER_EPOCH, optimizer, criterion, 
                               use_noise=False, is_bayesian=True, distance_fn=cosine_similarity)
                
                print(f"  訓練完成，開始測試...")
                
                # 測試不同距離度量
                iteration_results = []
                for test_idx, (test_name, test_fn) in enumerate(test_distance_fns):
                    print(f"    測試 {test_name}")
                    
                    if test_name == "Cosine (No Quant)":
                        # 不量化的測試
                        acc = test_model_without_quantization(
                            model, test_data, N_WAY, K_SHOT, N_QUERY, TEST_EPISODES, test_fn)
                    else:
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
    np.savez(f"{DATASET}_training_methods_comparison.npz",
             training_methods=[name for name, _ in training_methods],
             test_metrics=[name for name, _ in test_distance_fns],
             results=all_results,
             training_noise_std=training_noise_std)
    
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
    results_df.to_csv(f"{DATASET}_training_methods_detailed_results.csv", index=False)
    
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
        plt.ylim(70, 85)
        plt.title('Performance Comparison', fontweight='bold', fontsize=16)
        plt.xticks(x, [name for name, _ in test_distance_fns], 
                  rotation=45, fontweight='bold', fontsize=12)
        plt.yticks(fontweight='bold', fontsize=12)
        plt.legend(prop={'weight': 'bold', 'size': 12})
        plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 打印總結
    print("\n=== 實驗總結 ===")
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
                
                # 顯示量化影響（如果有Cosine相關結果）
                if not np.isnan(mean_results[i, 0]) and not np.isnan(mean_results[i, 1]):
                    cosine_quant = mean_results[i, 0]
                    cosine_no_quant = mean_results[i, 1]
                    quantization_impact = cosine_no_quant - cosine_quant
                    print(f"  量化影響 (Cosine): {quantization_impact:+.2f}% (未量化: {cosine_no_quant:.2f}%, 量化: {cosine_quant:.2f}%)")
        else:
            print(f"\n{train_name}: 無有效結果")
    
    # 重置字體設定
    plt.rcParams.update(plt.rcParamsDefault)
    
    return all_results, mean_results, std_results

# -------------------------------
# 主程式
# -------------------------------
if __name__ == "__main__":
    experiment_training_methods_comparison()
    print("所有實驗、比較和可視化已完成！")