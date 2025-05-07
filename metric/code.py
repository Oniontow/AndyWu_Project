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


# -------------------------------
# Parameters
# -------------------------------
# Dataset selection 'omniglot', 'cifar-10', 'mnist'
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

def cosine_similarity(a, b):
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.t())

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

def hybrid_distance(a, b, alpha=0.7, beta=0.3):
    """
    Hybrid distance metric combining cosine similarity with other metrics
    
    Args:
        a, b: Embedding vectors
        alpha: Weight for cosine similarity
        beta: Weight for L1 distance (remaining weight is for Hamming)
    """
    cosine_scores = cosine_similarity(a, b)
    l1_scores = l1_distance(a, b)
    
    # Normalize L1 scores
    l1_min = torch.min(l1_scores)
    l1_max = torch.max(l1_scores)
    l1_norm = (l1_scores - l1_min) / (l1_max - l1_min + 1e-8)
    
    # Get enhanced hamming distance
    hamming_scores = enhanced_hamming_distance(a, b)
    hamming_min = torch.min(hamming_scores)
    hamming_max = torch.max(hamming_scores)
    hamming_norm = (hamming_scores - hamming_min) / (hamming_max - hamming_min + 1e-8)
    
    # Combine all scores
    return alpha * cosine_scores + beta * l1_norm + (1-alpha-beta) * hamming_norm

def hamming_distance(a, b):
    """
    計算兩組嵌入向量間的位元層級 Hamming 距離
    對於嵌入向量中的每個元素，將其視為 8 位元的整數，並計算對應位置上位元不同的數量
    """
    # 確保張量已被量化為整數範圍
    a_int = a.round().long() & 0xFF  # 取最低8位
    b_int = b.round().long() & 0xFF
    
    # 擴展維度以進行廣播
    a_expanded = a_int.unsqueeze(1)  # shape: [query_size, 1, dim]
    b_expanded = b_int.unsqueeze(0)  # shape: [1, support_size, dim]
    
    # 計算位元層級的 Hamming 距離
    bit_diff = torch.zeros(a_expanded.size(0), b_expanded.size(1), a_expanded.size(2), dtype=torch.long).to(a.device)
    
    # 對每一位進行 XOR 操作，數 1 的個數
    for i in range(8):
        bit_mask = 1 << i
        bit_diff += ((a_expanded & bit_mask) != (b_expanded & bit_mask)).long()
    
    # 對所有維度加總距離
    dist = bit_diff.sum(dim=2)
    
    # 返回負距離，使得值越大表示越相似
    return -dist

def linf_distance(a, b):
    # L-infinity距離越小越相似，這裡回傳負的距離作為分數
    dist = torch.max(torch.abs(a.unsqueeze(1) - b.unsqueeze(0)), dim=2)[0]
    return -dist

def l1_distance(a, b):
    """
    L1 distance (Manhattan distance) - sum of absolute differences
    Return negative distance so that higher values mean more similar (consistent with cosine)
    """
    dist = torch.sum(torch.abs(a.unsqueeze(1) - b.unsqueeze(0)), dim=2)
    return -dist

def l1_similarity_for_training(a, b):
    """
    可訓練的L1相似度函數 - 將L1距離轉換為相似度分數
    這個函數會返回相似度而非距離，以便用於訓練
    """
    # 計算L1距離
    distances = torch.sum(torch.abs(a.unsqueeze(1) - b.unsqueeze(0)), dim=2)
    
    # 將距離歸一化為相似度分數 (0-1範圍)，方便訓練
    # 使用批次內的最大距離進行歸一化
    max_dist = torch.max(distances) + 1e-8
    similarities = 1.0 - distances / max_dist
    
    return similarities

def linf_similarity_for_training(a, b, iterations=300, temperature=10.0):
    """
    改進的L-infinity相似度函數 - 使用多層softmax加強對最大差異的敏感度
    
    Args:
        a, b: 嵌入向量
        iterations: softmax疊加的次數
        temperature: softmax溫度參數，控制軟化程度
    """
    # 計算絕對差異
    abs_diff = torch.abs(a.unsqueeze(1) - b.unsqueeze(0))  # [query_size, support_size, dim]
    
    # 對每個維度進行多次softmax處理，加強對最大差異的敏感度
    # 先將差異矩陣轉置，使維度在前面，便於對每對向量的所有維度應用softmax
    # [query_size, support_size, dim] -> [query_size, support_size, dim]
    weights = abs_diff.clone()
    
    # 多次應用softmax，每次都會進一步強化最大值
    for _ in range(iterations):
        # 在維度方向應用softmax
        weights = F.softmax(weights * temperature, dim=2)
    
    # 加權平均，注重最大差異的維度
    # 使用weights作為注意力，強調最大差異的維度
    weighted_diff = (abs_diff * weights).sum(dim=2)
    
    # 歸一化為相似度分數
    max_value = torch.max(weighted_diff) + 1e-8
    similarities = 1.0 - weighted_diff / max_value
    
    return similarities

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

# -------------------------------
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
# Main Experiment Loop
# -------------------------------
distance_fns = [
    ("Cosine (No Quant)", cosine_similarity),  
    ("Cosine", cosine_similarity),
    ("L-1", l1_distance),
    ("L-inf", linf_distance)
]

# results = np.zeros((len(distance_fns), NUM_ITER_AVG))

# for it in range(NUM_ITER_AVG):
#     # Train a model using cosine similarity
#     if DATASET in ['omniglot', 'mnist']:
#         model = MANN(dataset=DATASET, quantize=False, out_dim = OUTPUT_DIM).to(DEVICE)
#     else:
#         model = EnhancedMANN(dataset=DATASET, quantize=False, out_dim = OUTPUT_DIM).to(DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, EPISODES_PER_EPOCH, 
#                 optimizer, nn.CrossEntropyLoss(), use_noise=False, distance_fn=cosine_similarity)
    

#     # Test with each distance metric
#     for i, (test_name, test_fn) in enumerate(distance_fns):
#         print(f"Iteration {it+1}/{NUM_ITER_AVG}, Testing with {test_name}")
        
#         # 根據測試名稱決定是否量化
#         if test_name == "Cosine (No Quant)":
#             # 不進行量化的測試
#             acc = test_model_without_quantization(model, test_data, N_WAY, K_SHOT, N_QUERY, 
#                                                 TEST_EPISODES, test_fn)
#         else:
#             # 進行量化的測試
#             acc = test_model_with_enforced_quantization(model, test_data, N_WAY, K_SHOT, N_QUERY, 
#                                                       TEST_EPISODES, test_fn)
        
#         results[i, it] = acc

# # Print results
# # Calculate mean and standard deviation before printing results
# mean_results = np.mean(results, axis=1)
# std_results = np.std(results, axis=1)

# # Print results
# print("\nFinal Results (Trained with Cosine Similarity):")
# for i, (metric_name, _) in enumerate(distance_fns):
#     print(f"{metric_name}: {mean_results[i]:.2f}% ± {std_results[i]:.2f}%")

# # 將結果存成CSV格式
# import pandas as pd
# result_dict = {
#     'metric': [name for name, _ in distance_fns],
#     'mean_accuracy': mean_results,
#     'std_dev': std_results
# }
# # 為每次迭代創建單獨的列
# for i in range(NUM_ITER_AVG):
#     result_dict[f'iter_{i+1}'] = results[:, i]

# result_df = pd.DataFrame(result_dict)
# result_df.to_csv("distance_metric_comparison.csv", index=False)
# print("Results saved to distance_metric_comparison.csv")

# # Save results in NPZ format as before
# # Save results in NPZ format as before
# np.savez("distance_metric_comparison.npz", 
#          mean=mean_results, 
#          std=std_results,
#          raw=results, 
#          metrics=[name for name, _ in distance_fns])

# # Create bar plot
# plt.figure(figsize=(10, 6))
# x = np.arange(len(distance_fns))
# plt.bar(x, mean_results, yerr=std_results, capsize=8, width=0.6)
# plt.xticks(x, [name for name, _ in distance_fns])
# plt.ylabel("Accuracy (%)")
# plt.title("Model Performance with Different Distance Metrics\n(Trained with Cosine Similarity)")
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # 設置y軸範圍，只顯示80%以上的部分
# plt.ylim(60, 90)

# plt.tight_layout()
# plt.savefig("distance_metric_comparison.png", dpi=300)
# plt.show()

# print("執行t-SNE分析以可視化不同距離度量下的嵌入向量關係...")
# visualize_tsne_embeddings(model, test_data, N_WAY, K_SHOT, N_QUERY, distance_fns)

# -------------------------------
# 額外實驗：使用不同的距離度量訓練模型
# -------------------------------

print("\n正在進行額外實驗：使用不同的距離度量訓練模型...\n")

# 定義訓練距離度量
training_distances = [
    ("Cosine", cosine_similarity),
    ("L-1", l1_similarity_for_training),
    ("L-inf", linf_similarity_for_training)
]

# 用於存儲所有結果的數組
all_training_results = np.zeros((len(training_distances), len(distance_fns), NUM_ITER_AVG))

# 存儲每種訓練距離最後一次迭代的模型，用於t-SNE比較
trained_models = []
model_names = []

# 對每種訓練距離測試
for train_idx, (train_name, train_fn) in enumerate(training_distances):
    print(f"\n==== 使用 {train_name} 距離訓練模型 ====\n")
    
    # 針對每種訓練距離重複實驗多次
    for it in range(NUM_ITER_AVG):
        print(f"實驗 {it+1}/{NUM_ITER_AVG}")
        
        # 初始化模型
        if DATASET in ['omniglot', 'mnist']:
            model = MANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
        else:
            model = EnhancedMANN(dataset=DATASET, quantize=False, out_dim=OUTPUT_DIM).to(DEVICE)
            
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # 使用指定的距離函數訓練模型
        train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, EPISODES_PER_EPOCH,
                  optimizer, nn.CrossEntropyLoss(), use_noise=False, distance_fn=train_fn)
        
        # 儲存最後一次迭代的模型
        if it == NUM_ITER_AVG - 1:
            trained_models.append(model)
            model_names.append(f"Trained with {train_name}")
        
        # 使用不同的度量測試模型
        for test_idx, (test_name, test_fn) in enumerate(distance_fns):
            print(f"  測試 {test_name}")
            
            # 根據測試名稱決定是否量化
            if test_name == "Cosine (No Quant)":
                # 不進行量化的測試
                acc = test_model_without_quantization(model, test_data, N_WAY, K_SHOT, N_QUERY,
                                                    TEST_EPISODES, test_fn)
            else:
                # 進行量化的測試
                acc = test_model_with_enforced_quantization(model, test_data, N_WAY, K_SHOT, N_QUERY,
                                                          TEST_EPISODES, test_fn)
            
            all_training_results[train_idx, test_idx, it] = acc
    
    # 計算平均值和標準差
    means = np.mean(all_training_results[train_idx], axis=1)
    stds = np.std(all_training_results[train_idx], axis=1)
    
    # 輸出當前訓練距離度量的結果
    print(f"\n{train_name} 訓練結果:")
    for i, (metric_name, _) in enumerate(distance_fns):
        print(f"{metric_name}: {means[i]:.2f}% ± {stds[i]:.2f}%")

# 保存實驗結果
np.savez("different_training_metrics_results.npz",
         training_metrics=[name for name, _ in training_distances],
         test_metrics=[name for name, _ in distance_fns],
         results=all_training_results)

# 創建比較圖表
# ... [保留原始的條形圖代碼]

# 生成t-SNE可視化比較
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

print("所有實驗、比較和可視化已完成！")