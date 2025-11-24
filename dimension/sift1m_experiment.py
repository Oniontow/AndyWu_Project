import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import umap
import h5py
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------------
# Parameters
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIFT_DIM = 128
REDUCED_DIM = 64

# Experiment configurations
EXPERIMENTS = [
    {"query_size": 1000, "database_size": 10000},
    {"query_size": 1000, "database_size": 50000},
    {"query_size": 1000, "database_size": 100000}
]

# -------------------------------
# INT3 Quantization Functions
# -------------------------------
def quantize_to_int3(data):
    """
    將數據量化到INT3 (-4 到 3，共8個離散值)
    
    Args:
        data: numpy array or torch tensor
        
    Returns:
        quantized data (same type as input)
    """
    is_torch = isinstance(data, torch.Tensor)
    
    if is_torch:
        arr = data.cpu().numpy()
    else:
        arr = data
    
    # 計算縮放因子
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # 將數據縮放到 [-4, 3] 範圍
    scaled = (arr - min_val) / (max_val - min_val + 1e-8) * 7 - 4
    
    # 四捨五入並裁剪到 INT3 範圍
    quantized = np.clip(np.round(scaled), -4, 3).astype(np.int8)
    
    if is_torch:
        return torch.from_numpy(quantized).to(data.device)
    else:
        return quantized

def dequantize_from_int3(quantized, original_min, original_max):
    """
    將INT3數據反量化回浮點數（用於某些計算）
    
    Args:
        quantized: INT3 quantized data
        original_min: 原始數據的最小值
        original_max: 原始數據的最大值
        
    Returns:
        dequantized data
    """
    is_torch = isinstance(quantized, torch.Tensor)
    
    if is_torch:
        arr = quantized.cpu().numpy()
    else:
        arr = quantized
    
    # 從 [-4, 3] 範圍反向縮放
    dequantized = ((arr + 4) / 7.0) * (original_max - original_min + 1e-8) + original_min
    
    if is_torch:
        return torch.from_numpy(dequantized.astype(np.float32)).to(quantized.device)
    else:
        return dequantized.astype(np.float32)

# -------------------------------
# AutoEncoder Model
# -------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=64):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 80),
            nn.ReLU(),
            nn.Linear(80, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 80),
            nn.ReLU(),
            nn.Linear(80, 96),
            nn.ReLU(),
            nn.Linear(96, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x):
        return self.encoder(x)

def train_autoencoder(data, input_dim=128, latent_dim=64, epochs=50, batch_size=256):
    """
    訓練自編碼器
    
    Args:
        data: 訓練數據 (numpy array)
        input_dim: 輸入維度
        latent_dim: 潛在空間維度
        epochs: 訓練輪數
        batch_size: 批次大小
        
    Returns:
        trained autoencoder model
    """
    model = AutoEncoder(input_dim, latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 轉換為torch tensor
    data_tensor = torch.FloatTensor(data).to(DEVICE)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # 隨機打亂數據
        indices = torch.randperm(len(data_tensor))
        
        for i in range(0, len(data_tensor), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = data_tensor[batch_indices]
            
            optimizer.zero_grad()
            _, reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    return model

# -------------------------------
# Data Processing Methods
# -------------------------------
def method1_direct_int3(data):
    """
    方法1: 直接將128維向量量化為INT3
    
    Args:
        data: (N, 128) numpy array
        
    Returns:
        (N, 128) INT3 quantized array
    """
    return quantize_to_int3(data)

def method2_average_pooling_int3(data):
    """
    方法2: 將相鄰兩維做平均，從128維降到64維，然後量化為INT3
    
    Args:
        data: (N, 128) numpy array
        
    Returns:
        (N, 64) INT3 quantized array
    """
    # 將數據reshape成 (N, 64, 2)，然後沿最後一維取平均
    reshaped = data.reshape(data.shape[0], 64, 2)
    averaged = np.mean(reshaped, axis=2)
    return quantize_to_int3(averaged)

def method3_pca_int3(data, query_data=None):
    """
    方法3: 使用PCA降維到64維，然後量化為INT3
    
    Args:
        data: (N, 128) numpy array (database/training data)
        query_data: (M, 128) numpy array (query data), optional
        
    Returns:
        if query_data is None:
            pca_model, (N, 64) INT3 quantized array
        else:
            pca_model, (N, 64) INT3 quantized database, (M, 64) INT3 quantized queries
    """
    pca = PCA(n_components=REDUCED_DIM)
    pca.fit(data)
    
    data_reduced = pca.transform(data)
    data_quantized = quantize_to_int3(data_reduced)
    
    if query_data is not None:
        query_reduced = pca.transform(query_data)
        query_quantized = quantize_to_int3(query_reduced)
        return pca, data_quantized, query_quantized
    
    return pca, data_quantized

def method4_umap_int3(data, query_data=None, n_neighbors=15, min_dist=0.1):
    """
    方法4: 使用UMAP降維到64維，然後量化為INT3
    
    Args:
        data: (N, 128) numpy array (database/training data)
        query_data: (M, 128) numpy array (query data), optional
        n_neighbors: UMAP參數
        min_dist: UMAP參數
        
    Returns:
        if query_data is None:
            umap_model, (N, 64) INT3 quantized array
        else:
            umap_model, (N, 64) INT3 quantized database, (M, 64) INT3 quantized queries
    """
    reducer = umap.UMAP(
        n_components=REDUCED_DIM,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    
    data_reduced = reducer.fit_transform(data)
    data_quantized = quantize_to_int3(data_reduced)
    
    if query_data is not None:
        query_reduced = reducer.transform(query_data)
        query_quantized = quantize_to_int3(query_reduced)
        return reducer, data_quantized, query_quantized
    
    return reducer, data_quantized

def method5_autoencoder_int3(data, query_data=None, epochs=50):
    """
    方法5: 使用AutoEncoder降維到64維，然後量化為INT3
    
    Args:
        data: (N, 128) numpy array (database/training data)
        query_data: (M, 128) numpy array (query data), optional
        epochs: 訓練輪數
        
    Returns:
        if query_data is None:
            ae_model, (N, 64) INT3 quantized array
        else:
            ae_model, (N, 64) INT3 quantized database, (M, 64) INT3 quantized queries
    """
    ae_model = train_autoencoder(data, SIFT_DIM, REDUCED_DIM, epochs=epochs)
    ae_model.eval()
    
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data).to(DEVICE)
        data_reduced = ae_model.encode(data_tensor).cpu().numpy()
        data_quantized = quantize_to_int3(data_reduced)
        
        if query_data is not None:
            query_tensor = torch.FloatTensor(query_data).to(DEVICE)
            query_reduced = ae_model.encode(query_tensor).cpu().numpy()
            query_quantized = quantize_to_int3(query_reduced)
            return ae_model, data_quantized, query_quantized
    
    return ae_model, data_quantized

# -------------------------------
# Distance Calculation & Evaluation
# -------------------------------
def calculate_l2_distances(queries, database):
    """
    計算L2距離（歐式距離）
    
    Args:
        queries: (M, D) array
        database: (N, D) array
        
    Returns:
        (M, N) distance matrix
    """
    # 使用sklearn的pairwise_distances，支持INT3數據
    return pairwise_distances(queries, database, metric='euclidean')

def evaluate_recall_at_k(distances, ground_truth, k_values=[1, 10, 100]):
    """
    計算Recall@K指標
    
    Args:
        distances: (M, N) distance matrix
        ground_truth: (M, K_true) ground truth nearest neighbors
        k_values: list of k values to evaluate
        
    Returns:
        dict of recall@k values
    """
    # 獲取距離矩陣中最近的鄰居索引
    sorted_indices = np.argsort(distances, axis=1)
    
    recalls = {}
    for k in k_values:
        # 獲取前k個最近鄰
        top_k_predictions = sorted_indices[:, :k]
        
        # 計算每個查詢的recall
        query_recalls = []
        for i in range(len(ground_truth)):
            # 真實最近鄰（取前k個作為參考）
            true_neighbors = set(ground_truth[i, :min(k, ground_truth.shape[1])])
            # 預測最近鄰
            pred_neighbors = set(top_k_predictions[i])
            # 計算交集
            intersection = true_neighbors & pred_neighbors
            # Recall = |交集| / |真實最近鄰|
            recall = len(intersection) / len(true_neighbors) if len(true_neighbors) > 0 else 0
            query_recalls.append(recall)
        
        recalls[f'recall@{k}'] = np.mean(query_recalls)
    
    return recalls

# -------------------------------
# SIFT1M Dataset Loading
# -------------------------------
def load_sift1m_data(base_path='.data/sift1m'):
    """
    載入SIFT1M數據集
    
    數據集下載: http://corpus-texmex.irisa.fr/
    
    Returns:
        base_vectors: (1000000, 128) base vectors
        query_vectors: (10000, 128) query vectors
        ground_truth: (10000, 100) ground truth nearest neighbors
    """
    def read_fvecs(filename):
        """讀取.fvecs格式文件"""
        with open(filename, 'rb') as f:
            # 讀取維度
            d = np.fromfile(f, dtype=np.int32, count=1)[0]
            f.seek(0)
            # 讀取所有數據
            data = np.fromfile(f, dtype=np.uint8)
            # Reshape: 每個向量前面有4字節的維度信息
            data = data.reshape(-1, d + 4)
            # 去掉維度信息，只保留向量數據
            return data[:, 4:].copy().astype(np.float32)
    
    def read_ivecs(filename):
        """讀取.ivecs格式文件（用於ground truth）"""
        with open(filename, 'rb') as f:
            d = np.fromfile(f, dtype=np.int32, count=1)[0]
            f.seek(0)
            data = np.fromfile(f, dtype=np.int32)
            data = data.reshape(-1, d + 1)
            return data[:, 1:].copy()
    
    print("載入SIFT1M數據集...")
    
    # 載入base vectors (database)
    base_file = os.path.join(base_path, 'sift_base.fvecs')
    base_vectors = read_fvecs(base_file)
    print(f"Base vectors shape: {base_vectors.shape}")
    
    # 載入query vectors
    query_file = os.path.join(base_path, 'sift_query.fvecs')
    query_vectors = read_fvecs(query_file)
    print(f"Query vectors shape: {query_vectors.shape}")
    
    # 載入ground truth
    gt_file = os.path.join(base_path, 'sift_groundtruth.ivecs')
    ground_truth = read_ivecs(gt_file)
    print(f"Ground truth shape: {ground_truth.shape}")
    
    return base_vectors, query_vectors, ground_truth

# -------------------------------
# Main Experiment Function
# -------------------------------
def run_experiment(base_vectors, query_vectors, ground_truth, 
                   query_size, database_size):
    """
    執行單個實驗配置
    
    Args:
        base_vectors: 完整的database vectors
        query_vectors: 完整的query vectors
        ground_truth: 完整的ground truth
        query_size: 本次實驗使用的query數量
        database_size: 本次實驗使用的database大小
        
    Returns:
        results_df: pandas DataFrame with results
    """
    print(f"\n{'='*60}")
    print(f"實驗配置: {query_size} queries @ {database_size} database")
    print(f"{'='*60}\n")
    
    # 抽取數據子集
    db_subset = base_vectors[:database_size]
    query_subset = query_vectors[:query_size]
    gt_subset = ground_truth[:query_size]
    
    # 過濾ground truth，只保留在database範圍內的索引
    gt_filtered = []
    for gt_row in gt_subset:
        filtered_row = gt_row[gt_row < database_size]
        gt_filtered.append(filtered_row)
    
    results = []
    
    # ========== 方法1: 直接INT3量化128維 ==========
    print("\n[方法1] 直接INT3量化 (128維)...")
    start_time = time.time()
    db_method1 = method1_direct_int3(db_subset)
    query_method1 = method1_direct_int3(query_subset)
    distances_method1 = calculate_l2_distances(query_method1, db_method1)
    recalls_method1 = evaluate_recall_at_k(distances_method1, np.array([gt[:10] for gt in gt_filtered]), k_values=[1, 10])
    time_method1 = time.time() - start_time
    
    results.append({
        'method': 'Method 1: Direct INT3 (128D)',
        'dimension': 128,
        'recall@1': recalls_method1['recall@1'],
        'recall@10': recalls_method1['recall@10'],
        'time_seconds': time_method1
    })
    print(f"  Recall@1: {recalls_method1['recall@1']:.4f}")
    print(f"  Recall@10: {recalls_method1['recall@10']:.4f}")
    print(f"  Time: {time_method1:.2f}s")
    
    # ========== 方法2: 平均池化降維 + INT3 ==========
    print("\n[方法2] 平均池化降維 + INT3 (64維)...")
    start_time = time.time()
    db_method2 = method2_average_pooling_int3(db_subset)
    query_method2 = method2_average_pooling_int3(query_subset)
    distances_method2 = calculate_l2_distances(query_method2, db_method2)
    recalls_method2 = evaluate_recall_at_k(distances_method2, np.array([gt[:10] for gt in gt_filtered]), k_values=[1, 10])
    time_method2 = time.time() - start_time
    
    results.append({
        'method': 'Method 2: Avg Pooling + INT3 (64D)',
        'dimension': 64,
        'recall@1': recalls_method2['recall@1'],
        'recall@10': recalls_method2['recall@10'],
        'time_seconds': time_method2
    })
    print(f"  Recall@1: {recalls_method2['recall@1']:.4f}")
    print(f"  Recall@10: {recalls_method2['recall@10']:.4f}")
    print(f"  Time: {time_method2:.2f}s")
    
    # ========== 方法3: PCA降維 + INT3 ==========
    print("\n[方法3] PCA降維 + INT3 (64維)...")
    start_time = time.time()
    pca_model, db_method3, query_method3 = method3_pca_int3(db_subset, query_subset)
    distances_method3 = calculate_l2_distances(query_method3, db_method3)
    recalls_method3 = evaluate_recall_at_k(distances_method3, np.array([gt[:10] for gt in gt_filtered]), k_values=[1, 10])
    time_method3 = time.time() - start_time
    
    results.append({
        'method': 'Method 3: PCA + INT3 (64D)',
        'dimension': 64,
        'recall@1': recalls_method3['recall@1'],
        'recall@10': recalls_method3['recall@10'],
        'time_seconds': time_method3
    })
    print(f"  Recall@1: {recalls_method3['recall@1']:.4f}")
    print(f"  Recall@10: {recalls_method3['recall@10']:.4f}")
    print(f"  Time: {time_method3:.2f}s")
    
    # ========== 方法4: UMAP降維 + INT3 ==========
    print("\n[方法4] UMAP降維 + INT3 (64維)...")
    start_time = time.time()
    umap_model, db_method4, query_method4 = method4_umap_int3(db_subset, query_subset)
    distances_method4 = calculate_l2_distances(query_method4, db_method4)
    recalls_method4 = evaluate_recall_at_k(distances_method4, np.array([gt[:10] for gt in gt_filtered]), k_values=[1, 10])
    time_method4 = time.time() - start_time
    
    results.append({
        'method': 'Method 4: UMAP + INT3 (64D)',
        'dimension': 64,
        'recall@1': recalls_method4['recall@1'],
        'recall@10': recalls_method4['recall@10'],
        'time_seconds': time_method4
    })
    print(f"  Recall@1: {recalls_method4['recall@1']:.4f}")
    print(f"  Recall@10: {recalls_method4['recall@10']:.4f}")
    print(f"  Time: {time_method4:.2f}s")
    
    # ========== 方法5: AutoEncoder降維 + INT3 ==========
    print("\n[方法5] AutoEncoder降維 + INT3 (64維)...")
    start_time = time.time()
    ae_model, db_method5, query_method5 = method5_autoencoder_int3(db_subset, query_subset, epochs=50)
    distances_method5 = calculate_l2_distances(query_method5, db_method5)
    recalls_method5 = evaluate_recall_at_k(distances_method5, np.array([gt[:10] for gt in gt_filtered]), k_values=[1, 10])
    time_method5 = time.time() - start_time
    
    results.append({
        'method': 'Method 5: AutoEncoder + INT3 (64D)',
        'dimension': 64,
        'recall@1': recalls_method5['recall@1'],
        'recall@10': recalls_method5['recall@10'],
        'time_seconds': time_method5
    })
    print(f"  Recall@1: {recalls_method5['recall@1']:.4f}")
    print(f"  Recall@10: {recalls_method5['recall@10']:.4f}")
    print(f"  Time: {time_method5:.2f}s")
    
    return pd.DataFrame(results)

# -------------------------------
# Visualization
# -------------------------------
def plot_results(all_results, experiment_configs):
    """
    繪製實驗結果圖表
    
    Args:
        all_results: list of DataFrames
        experiment_configs: list of experiment configurations
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('SIFT1M Dataset: Comparison of Dimensionality Reduction + INT3 Quantization', 
                 fontsize=16, fontweight='bold')
    
    for idx, (results_df, config) in enumerate(zip(all_results, experiment_configs)):
        # Recall@1 對比
        ax1 = axes[0, idx]
        methods = results_df['method'].str.replace('Method ', 'M').str.replace(': ', '\n', 1)
        ax1.bar(range(len(results_df)), results_df['recall@1'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax1.set_xticks(range(len(results_df)))
        ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Recall@1', fontsize=12, fontweight='bold')
        ax1.set_title(f"{config['query_size']}@{config['database_size']}", fontsize=12, fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_ylim([0, 1])
        
        # Recall@10 對比
        ax2 = axes[1, idx]
        ax2.bar(range(len(results_df)), results_df['recall@10'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax2.set_xticks(range(len(results_df)))
        ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Recall@10', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('sift1m_experiment_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------------
# Main Execution
# -------------------------------
def main():
    """
    主函數：執行所有實驗
    """
    # 載入SIFT1M數據集
    base_vectors, query_vectors, ground_truth = load_sift1m_data()
    
    all_results = []
    
    # 執行所有實驗配置
    for config in EXPERIMENTS:
        results_df = run_experiment(
            base_vectors, 
            query_vectors, 
            ground_truth,
            config['query_size'],
            config['database_size']
        )
        
        # 保存結果
        filename = f"sift1m_results_{config['query_size']}@{config['database_size']}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\n結果已保存至: {filename}")
        
        all_results.append(results_df)
    
    # 繪製比較圖
    plot_results(all_results, EXPERIMENTS)
    
    # 生成總結報告
    print("\n" + "="*80)
    print("實驗總結")
    print("="*80)
    for config, results_df in zip(EXPERIMENTS, all_results):
        print(f"\n配置: {config['query_size']}@{config['database_size']}")
        print(results_df.to_string(index=False))
    
    print("\n所有實驗完成！")

if __name__ == "__main__":
    main()
