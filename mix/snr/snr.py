import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import datasets
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
NUM_ITER_AVG = 10
OUTPUT_DIM = 32 if DATASET in ['omniglot', 'mnist'] else 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Model Definitions
# -------------------------------
class MANN(nn.Module):
    def __init__(self, dataset, out_dim=1024):
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
        self.fc = nn.Linear(planes[3] * 7 * 7, out_dim)

    def forward(self, x):
        return self.fc(self.model(x).view(x.size(0), -1))

class EnhancedMANN(nn.Module):
    def __init__(self, dataset, out_dim=1024):
        super().__init__()
        in_channels = 1 if dataset in ['omniglot', 'mnist'] else 3
        
        if dataset in ['omniglot', 'mnist']:
            base_filters = 64
            planes = [base_filters, base_filters*2, base_filters*2, base_filters*4]
        else:
            base_filters = 64
            planes = [base_filters, base_filters*2, base_filters*4, base_filters*8]
        
        self.model = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, planes[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes[0]),
            nn.ReLU(),
            nn.Conv2d(planes[0], planes[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes[0]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes[1]),
            nn.ReLU(),
            nn.Conv2d(planes[1], planes[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes[1]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(planes[1], planes[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes[2]),
            nn.ReLU(),
            nn.Conv2d(planes[2], planes[3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes[3]),
            nn.ReLU(),
        )
        
        feature_size = planes[3] * 7 * 7
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_size // 2, out_dim)
        )

    def forward(self, x):
        features = self.model(x)
        return self.fc(features.view(features.size(0), -1))

# -------------------------------
# Distance Functions
# -------------------------------
def cosine_similarity(a, b):
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.t())

def l1_distance(a, b):
    dist = torch.sum(torch.abs(a.unsqueeze(1) - b.unsqueeze(0)), dim=2)
    return -dist

def l2_distance(a, b):
    dist = torch.norm(a.unsqueeze(1) - b.unsqueeze(0), p=2, dim=2)
    return -dist

def linf_distance(a, b):
    dist = torch.max(torch.abs(a.unsqueeze(1) - b.unsqueeze(0)), dim=2)[0]
    return -dist

def lsh_similarity(a, b, num_bits=256, random_seed=42):
    torch.manual_seed(random_seed)
    device = a.device
    D = a.size(1)
    hyperplanes = torch.randn(D, num_bits, device=device)
    a_proj = (a @ hyperplanes) > 0
    b_proj = (b @ hyperplanes) > 0
    matches = (a_proj.unsqueeze(1) == b_proj.unsqueeze(0)).sum(dim=2)
    return matches.float() / num_bits

# -------------------------------
# Quantization Functions
# -------------------------------
def quantize_to_int8(tensor, scale=None):
    if scale is None:
        abs_max = torch.max(torch.abs(tensor)).item()
        scale = 127.0 / (abs_max + 1e-8)
    
    scaled = tensor * scale
    quantized = scaled.detach().round().clamp(-128, 127) - scaled.detach() + scaled
    return quantized, scale

def dequantize_from_int8(quantized, scale):
    return quantized.float() / scale

def apply_consistent_quantization(support_emb, query_emb):
    all_embs = torch.cat([support_emb, query_emb], dim=0)
    abs_max = torch.max(torch.abs(all_embs)).item()
    scale = 127.0 / (abs_max + 1e-8)
    
    quantized_support, _ = quantize_to_int8(support_emb, scale)
    quantized_query, _ = quantize_to_int8(query_emb, scale)
    
    support_emb_q = dequantize_from_int8(quantized_support, scale)
    query_emb_q = dequantize_from_int8(quantized_query, scale)
    
    return support_emb_q, query_emb_q

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
# Training & Testing Functions
# -------------------------------
def train_model(model, data_by_class, n_way, k_shot, n_query, num_epochs, episodes_per_epoch, 
               optimizer, criterion, use_noise=False, noise_std=0.1):
    model.train()
    
    for epoch in range(num_epochs):
        if epoch % 10 == 0:
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

                scores = cosine_similarity(query_emb, support_emb)
                loss = criterion(scores, query_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            except Exception as e:
                print(f"    Episode {episode} error: {str(e)}")
                continue

def test_model_with_quantization(model, data_by_class, n_way, k_shot, n_query, 
                                test_episodes, distance_fn):
    model.eval()
    total_correct, total_num = 0, 0
    
    with torch.no_grad():
        for _ in range(test_episodes):
            support_imgs, support_labels, query_imgs, query_labels = sample_episode(
                data_by_class, n_way, k_shot, n_query)
            support_imgs, support_labels = support_imgs.to(DEVICE), support_labels.to(DEVICE)
            query_imgs, query_labels = query_imgs.to(DEVICE), query_labels.to(DEVICE)
            
            support_emb = model(support_imgs)
            query_emb = model(query_imgs)
            support_emb_q, query_emb_q = apply_consistent_quantization(support_emb, query_emb)
            
            scores = distance_fn(query_emb_q, support_emb_q)
            _, max_indices = torch.max(scores, dim=1)
            pred = support_labels[max_indices]
            total_correct += (pred == query_labels).sum().item()
            total_num += query_labels.size(0)
    
    return total_correct / total_num * 100.0

# -------------------------------
# Dataset Preparation
# -------------------------------
def load_dataset(dataset_name):
    if dataset_name == 'omniglot':
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x)
        ])
        train_dataset = torchvision.datasets.Omniglot(root='.data', background=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.Omniglot(root='.data', background=False, download=True, transform=transform)
    
    elif dataset_name == 'cifar-10':
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.CIFAR10(root='.data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='.data', train=False, download=True, transform=transform)
    
    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(root='.mnist', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='.mnist', train=False, transform=transform, download=True)
    
    return train_dataset, test_dataset

# Load dataset
train_dataset, test_dataset = load_dataset(DATASET)
train_data = organize_by_class(train_dataset)
test_data = organize_by_class(test_dataset)

# -------------------------------
# Main Experiment
# -------------------------------

def save_and_visualize_noise_results(distance_functions, noise_means_per_iteration, 
                                    mean_noise, std_noise, all_iteration_noise_values):
    """Save results and create visualizations"""
    
    print("\n=== Relative Noise Strength Analysis Results ===")
    for i, (dist_name, _) in enumerate(distance_functions[1:]):
        print(f"{dist_name}: Mean Noise Strength = {mean_noise[i]:.4f} ± {std_noise[i]:.4f}")
    
    # Save results
    np.savez(f"{DATASET}_noise_analysis.npz",
             distance_names=[name for name, _ in distance_functions[1:]],
             noise_means_per_iteration=noise_means_per_iteration,
             mean_noise=mean_noise,
             std_noise=std_noise,
             all_iteration_noise_values=all_iteration_noise_values)
    
    # Create visualizations
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Mean Noise Strength Comparison (without error bars)
    plt.figure(figsize=(12, 8))
    x_pos = np.arange(len(distance_functions)-1)
    
    bars = plt.bar(x_pos, mean_noise, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.xlabel('Distance Metric', fontweight='bold', fontsize=14)
    plt.ylabel('Relative Noise Strength', fontweight='bold', fontsize=14)
    plt.title('Mean Relative Noise Strength Comparison', fontweight='bold', fontsize=16)
    plt.xticks(x_pos, [name for name, _ in distance_functions[1:]], rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, mean_noise)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{DATASET}_noise_mean_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Individual Noise Distribution Histograms for each distance metric
    if all_iteration_noise_values and len(all_iteration_noise_values) > 0:
        # Use the first iteration's noise values for detailed distribution analysis
        first_iteration_noise = all_iteration_noise_values[0]
        
        for idx, (dist_name, _) in enumerate(distance_functions[1:]):
            if dist_name in first_iteration_noise and len(first_iteration_noise[dist_name]) > 0:
                plt.figure(figsize=(10, 6))
                
                noise_values = first_iteration_noise[dist_name]
                
                # Create histogram
                plt.hist(noise_values, bins=50, alpha=0.7, color=colors[idx], 
                        density=True, edgecolor='black', linewidth=0.5)
                
                # Calculate and add statistical lines
                mean_val = np.mean(noise_values)
                std_val = np.std(noise_values)
                median_val = np.median(noise_values)
                
                plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {mean_val:.3f}')
                plt.axvline(median_val, color='blue', linestyle='-.', linewidth=2, 
                           label=f'Median: {median_val:.3f}')
                plt.axvline(mean_val + std_val, color='red', linestyle=':', linewidth=1, alpha=0.7)
                plt.axvline(mean_val - std_val, color='red', linestyle=':', linewidth=1, alpha=0.7)
                
                plt.xlabel('Relative Noise Strength', fontweight='bold', fontsize=14)
                plt.ylabel('Density', fontweight='bold', fontsize=14)
                plt.title(f'{dist_name} Relative Noise Distribution', fontweight='bold', fontsize=16)
                plt.legend(fontsize=12)
                plt.grid(alpha=0.3)
                
                # Add statistics text box
                stats_text = (f'Statistics:\n'
                             f'Mean: {mean_val:.4f}\n'
                             f'Std: {std_val:.4f}\n'
                             f'Median: {median_val:.4f}\n'
                             f'Min: {np.min(noise_values):.4f}\n'
                             f'Max: {np.max(noise_values):.4f}')
                
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(f"{DATASET}_noise_distribution_{dist_name.lower().replace('-', '_')}.png", 
                           dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"Generated noise distribution plot for {dist_name}")
    
    # 3. Noise Distribution Boxplot
    plt.figure(figsize=(12, 8))
    noise_data_for_boxplot = []
    labels_for_boxplot = []
    
    for i, (dist_name, _) in enumerate(distance_functions[1:]):
        valid_iterations = noise_means_per_iteration[:, i][~np.isnan(noise_means_per_iteration[:, i])]
        if len(valid_iterations) > 0:
            noise_data_for_boxplot.append(valid_iterations)
            labels_for_boxplot.append(dist_name)
    
    if noise_data_for_boxplot:
        bp = plt.boxplot(noise_data_for_boxplot, labels=labels_for_boxplot, patch_artist=True, 
                        showmeans=True, meanline=True)
        
        for patch, color in zip(bp['boxes'], colors[:len(noise_data_for_boxplot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for meanline in bp['means']:
            meanline.set_color('red')
            meanline.set_linewidth(2)
    
    plt.xlabel('Distance Metric', fontweight='bold', fontsize=14)
    plt.ylabel('Relative Noise Strength Across Iterations', fontweight='bold', fontsize=14)
    plt.title('Relative Noise Strength Distribution Comparison', fontweight='bold', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{DATASET}_noise_boxplot_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Noise trends across iterations
    plt.figure(figsize=(12, 8))
    for i, (dist_name, _) in enumerate(distance_functions[1:]):
        valid_noise = noise_means_per_iteration[:, i][~np.isnan(noise_means_per_iteration[:, i])]
        if len(valid_noise) > 0:
            plt.plot(range(1, len(valid_noise)+1), valid_noise, 
                    marker='o', label=dist_name, color=colors[i], 
                    linewidth=2, markersize=8)
    
    plt.xlabel('Iteration', fontweight='bold', fontsize=14)
    plt.ylabel('Mean Relative Noise Strength', fontweight='bold', fontsize=14)
    plt.title('Relative Noise Strength Consistency Across Iterations', fontweight='bold', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{DATASET}_noise_iteration_trends.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    detailed_results = []
    for i, (dist_name, _) in enumerate(distance_functions[1:]):
        valid_iterations = noise_means_per_iteration[:, i][~np.isnan(noise_means_per_iteration[:, i])]
        detailed_results.append({
            'Distance_Metric': dist_name,
            'Mean_Noise_Strength': mean_noise[i],
            'Std_Noise_Strength': std_noise[i],
            'Min_Noise_Strength': np.min(valid_iterations) if len(valid_iterations) > 0 else np.nan,
            'Max_Noise_Strength': np.max(valid_iterations) if len(valid_iterations) > 0 else np.nan,
            'Median_Noise_Strength': np.median(valid_iterations) if len(valid_iterations) > 0 else np.nan,
            'Valid_Iterations': len(valid_iterations)
        })
    
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{DATASET}_noise_analysis_detailed_results.csv", index=False)
    
    # Print summary
    print("\n=== Experiment Summary ===")
    best_idx = np.argmin(mean_noise)
    worst_idx = np.argmax(mean_noise)
    
    print(f"Lowest noise (most similar): {distance_functions[best_idx + 1][0]} ({mean_noise[best_idx]:.4f} ± {std_noise[best_idx]:.4f})")
    print(f"Highest noise (most different): {distance_functions[worst_idx + 1][0]} ({mean_noise[worst_idx]:.4f} ± {std_noise[worst_idx]:.4f})")
    print(f"Noise range: {np.max(mean_noise) - np.min(mean_noise):.4f}")

def experiment_noise_analysis():
    """
    Relative Noise Strength Analysis Experiment:
    Train a model with cosine similarity and analyze relative noise strength between different distance metrics
    """
    print("\n=== Relative Noise Strength Analysis Experiment ===")
    print(f"Settings: EPOCHS={NUM_EPOCHS}, EPISODES_PER_EPOCH={EPISODES_PER_EPOCH}")
    print(f"TEST_EPISODES={TEST_EPISODES}, ITERATIONS={NUM_ITER_AVG}")
    
    distance_functions = [
        ("Cosine", cosine_similarity),
        ("L2", l2_distance),
        ("L1", l1_distance), 
        ("L-inf", linf_distance),
        ("LSH", lambda a, b: lsh_similarity(a, b, num_bits=OUTPUT_DIM*8))
    ]
    
    # Storage for results
    noise_means_per_iteration = np.zeros((NUM_ITER_AVG, len(distance_functions)-1))
    all_iteration_noise_values = []
    
    for iteration in range(NUM_ITER_AVG):
        print(f"\nIteration {iteration+1}/{NUM_ITER_AVG}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            # Create and train model
            if DATASET in ['omniglot', 'mnist']:
                model = MANN(dataset=DATASET, out_dim=OUTPUT_DIM).to(DEVICE)
            else:
                model = EnhancedMANN(dataset=DATASET, out_dim=OUTPUT_DIM).to(DEVICE)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            print("  Training model...")
            train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, 
                       EPISODES_PER_EPOCH, optimizer, criterion, use_noise=False)
            
            print("  Collecting distance data...")
            model.eval()
            
            # Collect distance data from test episodes
            all_distance_matrices = {name: [] for name, _ in distance_functions}
            
            with torch.no_grad():
                for episode in range(TEST_EPISODES):
                    if episode % 100 == 0:
                        print(f"    Processing episode {episode}/{TEST_EPISODES}")
                    
                    support_imgs, support_labels, query_imgs, query_labels = sample_episode(
                        test_data, N_WAY, K_SHOT, N_QUERY)
                    support_imgs = support_imgs.to(DEVICE)
                    query_imgs = query_imgs.to(DEVICE)
                    
                    # Get embeddings and apply quantization
                    support_emb = model(support_imgs)
                    query_emb = model(query_imgs)
                    support_emb_q, query_emb_q = apply_consistent_quantization(support_emb, query_emb)
                    
                    # Compute all distance metrics
                    for dist_name, dist_fn in distance_functions:
                        if dist_name == "Cosine":
                            similarity_matrix = dist_fn(query_emb_q, support_emb_q)
                            distance_matrix = 1.0 - similarity_matrix
                        else:
                            similarity_matrix = dist_fn(query_emb_q, support_emb_q)
                            distance_matrix = -similarity_matrix
                        
                        distances_flat = distance_matrix.cpu().numpy().flatten()
                        all_distance_matrices[dist_name].extend(distances_flat)
            
            print("  Computing relative noise strength...")
            
            # Convert to numpy arrays and normalize
            normalized_distances = {}
            for dist_name, distances in all_distance_matrices.items():
                distances = np.array(distances)
                min_val = np.min(distances)
                max_val = np.max(distances)
                
                if max_val > min_val:
                    normalized = (distances - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros_like(distances)
                
                normalized_distances[dist_name] = normalized
                print(f"    {dist_name}: Range [{min_val:.4f}, {max_val:.4f}] -> Normalized [{np.min(normalized):.4f}, {np.max(normalized):.4f}]")
            
            # Use Cosine distance as clean signal
            clean_signal = normalized_distances["Cosine"]
            
            # Calculate relative noise strength for each distance metric
            iteration_noise_values = {}
            iteration_noise_means = []
            
            for i, (dist_name, _) in enumerate(distance_functions[1:]):
                noisy_signal = normalized_distances[dist_name]
                
                # Calculate absolute difference (relative noise strength)
                noise_values = np.abs(noisy_signal - clean_signal)
                mean_noise = np.mean(noise_values)
                
                iteration_noise_means.append(mean_noise)
                iteration_noise_values[dist_name] = noise_values
                
                print(f"    Noise |{dist_name} - Cosine|: Mean = {mean_noise:.4f}, Std = {np.std(noise_values):.4f}")
                print(f"      Min noise: {np.min(noise_values):.6f}, Max noise: {np.max(noise_values):.6f}")
            
            noise_means_per_iteration[iteration] = iteration_noise_means
            all_iteration_noise_values.append(iteration_noise_values)
            
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Iteration {iteration+1} error: {str(e)}")
            noise_means_per_iteration[iteration] = np.full(len(distance_functions)-1, np.nan)
            all_iteration_noise_values.append({})
            continue
    
    # Calculate overall statistics
    mean_noise = np.nanmean(noise_means_per_iteration, axis=0)
    std_noise = np.nanstd(noise_means_per_iteration, axis=0)
    
    # Save and visualize results
    save_and_visualize_noise_results(distance_functions, noise_means_per_iteration, 
                                    mean_noise, std_noise, all_iteration_noise_values)
    
    return noise_means_per_iteration, mean_noise, std_noise

# -------------------------------
# Main Program
# -------------------------------
if __name__ == "__main__":
    experiment_noise_analysis()
    print("Relative noise strength analysis experiment completed!")