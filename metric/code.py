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
EPISODES_PER_EPOCH = 20
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

distance_fns = [
    ("Cosine", cosine_similarity),
    ("L-1", l1_distance),
    ("L-inf", linf_distance),
]

# -------------------------------
# 實驗 function 區塊
# -------------------------------
def experiment_main_distance_metrics():
    results = np.zeros((len(distance_fns), NUM_ITER_AVG))
    for it in range(NUM_ITER_AVG):
        if DATASET in ['omniglot', 'mnist']:
            model = MANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
        else:
            model = EnhancedMANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, EPISODES_PER_EPOCH, 
                    optimizer, nn.CrossEntropyLoss(), use_noise=False, distance_fn=cosine_similarity)
        for i, (test_name, test_fn) in enumerate(distance_fns):
            print(f"Iteration {it+1}/{NUM_ITER_AVG}, Testing with {test_name}")
            acc = test_model_with_enforced_quantization(model, test_data, N_WAY, K_SHOT, N_QUERY, TEST_EPISODES, test_fn)
            results[i, it] = acc

    mean_results = np.mean(results, axis=1)
    std_results = np.std(results, axis=1)
    print("\nFinal Results (Trained with Cosine Similarity):")
    for i, (metric_name, _) in enumerate(distance_fns):
        print(f"{metric_name}: {mean_results[i]:.2f}% ± {std_results[i]:.2f}%")
    result_dict = {
        'metric': [name for name, _ in distance_fns],
        'mean_accuracy': mean_results,
        'std_dev': std_results
    }
    for i in range(NUM_ITER_AVG):
        result_dict[f'iter_{i+1}'] = results[:, i]
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv("distance_metric_comparison.csv", index=False)
    np.savez("distance_metric_comparison.npz", mean=mean_results, std=std_results, raw=results, metrics=[name for name, _ in distance_fns])
    plt.figure(figsize=(10, 6))
    x = np.arange(len(distance_fns))
    plt.bar(x, mean_results, yerr=std_results, capsize=8, width=0.6)
    plt.xticks(x, [name for name, _ in distance_fns], fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=18, fontweight='bold')
    plt.title("Model Performance with Different Distance Metrics\n(Trained with Cosine Similarity)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(85, 100)
    plt.tight_layout()
    plt.savefig("distance_metric_comparison.png", dpi=300)
    plt.show()
    print("執行t-SNE分析以可視化不同距離度量下的嵌入向量關係...")
    visualize_tsne_embeddings(model, test_data, N_WAY, K_SHOT, N_QUERY, distance_fns)

def experiment_training_with_different_metrics():
    training_distances = [
        ("Cosine", cosine_similarity),
        ("L-1", l1_similarity_for_training),
        ("L-inf", linf_similarity_for_training)
    ]
    all_training_results = np.zeros((len(training_distances), len(distance_fns), NUM_ITER_AVG))
    trained_models = []
    model_names = []
    for train_idx, (train_name, train_fn) in enumerate(training_distances):
        print(f"\n==== 使用 {train_name} 距離訓練模型 ====\n")
        for it in range(NUM_ITER_AVG):
            print(f"實驗 {it+1}/{NUM_ITER_AVG}")
            if DATASET in ['omniglot', 'mnist']:
                model = MANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
            else:
                model = EnhancedMANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, EPISODES_PER_EPOCH,
                        optimizer, nn.CrossEntropyLoss(), use_noise=False, distance_fn=train_fn)
            if it == NUM_ITER_AVG - 1:
                trained_models.append(model)
                model_names.append(f"Trained with {train_name}")
            for test_idx, (test_name, test_fn) in enumerate(distance_fns):
                print(f"  測試 {test_name}")
                acc = test_model_with_enforced_quantization(model, test_data, N_WAY, K_SHOT, N_QUERY, TEST_EPISODES, test_fn)
                all_training_results[train_idx, test_idx, it] = acc
        means = np.mean(all_training_results[train_idx], axis=1)
        stds = np.std(all_training_results[train_idx], axis=1)
        print(f"\n{train_name} 訓練結果:")
        for i, (metric_name, _) in enumerate(distance_fns):
            print(f"{metric_name}: {means[i]:.2f}% ± {stds[i]:.2f}%")
    np.savez(f"{DATASET}_different_training_metrics_results.npz",
             training_metrics=[name for name, _ in training_distances],
             test_metrics=[name for name, _ in distance_fns],
             results=all_training_results)
    mean_results = np.mean(all_training_results, axis=2)
    std_results = np.std(all_training_results, axis=2)
    plt.figure(figsize=(12, 8))
    bar_width = 0.25
    x = np.arange(len(distance_fns))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (train_name, _) in enumerate(training_distances):
        plt.bar(x + i*bar_width - bar_width, mean_results[i, :],
                yerr=std_results[i, :], capsize=5, width=bar_width,
                color=colors[i], label=f"Trained with {train_name}")
    plt.xlabel('Testing Distance Metric', fontsize=18, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
    plt.title('Model Performance with Different Training and Testing Distance Metrics', fontsize=14)
    plt.xticks(x, [name for name, _ in distance_fns], fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(85, 100)
    plt.tight_layout()
    plt.savefig(f"{DATASET}_testing_metrics_comparison.png", dpi=300)
    plt.show()
    print("\n正在生成t-SNE嵌入空間比較...")
    visualize_tsne_for_different_training_metrics(
        trained_models, 
        model_names, 
        test_data, 
        N_WAY, 
        K_SHOT, 
        N_QUERY, 
        distance_fns
    )

def experiment_lsh_bits_analysis():
    print("\n重新訓練一個模型（cosine similarity）供 LSH bit 數量分析...")
    lsh_bits_list = [16, 24, 32, 48, 64, 128, 256, 400, 512, 768, 1024, 1532, 2048, 3000, 4096, 8192]
    lsh_accs_all = np.zeros((len(lsh_bits_list), NUM_ITER_AVG))
    for idx, num_bits in enumerate(lsh_bits_list):
        print(f"\nLSH bits = {num_bits}，進行 {NUM_ITER_AVG} 次訓練與測試...")
        for it in range(NUM_ITER_AVG):
            if DATASET in ['omniglot', 'mnist']:
                lsh_model = MANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
            else:
                lsh_model = EnhancedMANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
            lsh_optimizer = optim.Adam(lsh_model.parameters(), lr=1e-3)
            train_model(
                lsh_model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, EPISODES_PER_EPOCH,
                lsh_optimizer, nn.CrossEntropyLoss(), use_noise=False, distance_fn=cosine_similarity
            )
            def lsh_fn(a, b, num_bits=num_bits):
                return lsh_similarity(a, b, num_bits=num_bits)
            acc = test_model_with_enforced_quantization(
                lsh_model, test_data, N_WAY, K_SHOT, N_QUERY, TEST_EPISODES, lsh_fn
            )
            lsh_accs_all[idx, it] = acc
            print(f"  第 {it+1} 次：{acc:.2f}%")
    lsh_accs_mean = np.mean(lsh_accs_all, axis=1)
    lsh_accs_std = np.std(lsh_accs_all, axis=1)
    plt.figure(figsize=(8, 5))
    plt.errorbar(lsh_bits_list, lsh_accs_mean, yerr=lsh_accs_std, marker='o', capsize=5)
    plt.xscale("log")
    plt.xlabel("Number of LSH bits")
    plt.ylabel("Accuracy (%)")
    plt.title("Effect of LSH Bit Number on Model Performance")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{DATASET}_lsh_bits_analysis.png", dpi=300)
    plt.show()
    pd.DataFrame({
        "lsh_bits": lsh_bits_list,
        "mean_accuracy": lsh_accs_mean,
        "std": lsh_accs_std
    }).to_csv(f"{DATASET}_lsh_bits_analysis.csv", index=False)
    print("LSH bit 數量分析已完成，結果已儲存。")

def experiment_cosine_hamming_lsh_avg():
    print("\n=== 額外測試：Cosine similarity 訓練模型在 Cosine, Hamming, LSH 距離下的表現 (多次平均) ===")
    extra_test_metrics = [
        ("Cosine", cosine_similarity),
        ("LSH (Same Output Bit)", lambda a, b: lsh_similarity(a, b, num_bits=OUTPUT_DIM*8)),
        ("LSH (Same Output Dim)", lambda a, b: lsh_similarity(a, b, num_bits=OUTPUT_DIM*8)),
        ("Hamming", hamming_distance),
    ]
    extra_results = np.zeros((len(extra_test_metrics), NUM_ITER_AVG))
    for it in range(NUM_ITER_AVG):
        print(f"\n[Iteration {it+1}/{NUM_ITER_AVG}]")
        if DATASET in ['omniglot', 'mnist']:
            extra_model = MANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
        else:
            extra_model = EnhancedMANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
        extra_optimizer = optim.Adam(extra_model.parameters(), lr=1e-3)
        train_model(
            extra_model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, EPISODES_PER_EPOCH,
            extra_optimizer, nn.CrossEntropyLoss(), use_noise=False, distance_fn=cosine_similarity
        )
        for i, (name, fn) in enumerate(extra_test_metrics):
            acc = test_model_with_enforced_quantization(
                extra_model, test_data, N_WAY, K_SHOT, N_QUERY, TEST_EPISODES, fn
            )
            extra_results[i, it] = acc
            print(f"  {name} distance accuracy: {acc:.2f}%")
    extra_mean = np.mean(extra_results, axis=1)
    extra_std = np.std(extra_results, axis=1)
    print("\n=== 額外測試平均結果 ===")
    for i, (name, _) in enumerate(extra_test_metrics):
        print(f"{name}: {extra_mean[i]:.2f}% ± {extra_std[i]:.2f}%")
    plt.figure(figsize=(7, 5))
    x = np.arange(len(extra_test_metrics))
    plt.bar(x, extra_mean, yerr=extra_std, capsize=8, width=0.6, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xticks(x, [name for name, _ in extra_test_metrics], fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=16, fontweight='bold')
    plt.title("Cosine-trained Model on Different Distance Metrics", fontsize=14)
    plt.ylim(75, 85)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{DATASET}_extra_distance_metrics.png", dpi=300)
    plt.show()

# -------------------------------
# 主程式
# -------------------------------
if __name__ == "__main__":
    # experiment_main_distance_metrics()
    # experiment_training_with_different_metrics()
    # experiment_lsh_bits_analysis()
    experiment_cosine_hamming_lsh_avg()
    print("所有實驗、比較和可視化已完成！")