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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# -------------------------------
# Parameters
# -------------------------------
DATASET = 'omniglot'
N_WAY = 5
K_SHOT = 1
N_QUERY = 5
NUM_EPOCHS = 100
EPISODES_PER_EPOCH = 20
TEST_EPISODES = 300
NUM_ITER_AVG = 10
OUTPUT_DIM = 32 if DATASET in ['omniglot', 'mnist'] else 1024

# Multi-GPU settings
NUM_GPUS = torch.cuda.device_count()
print(f"Available GPUs: {NUM_GPUS}")

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

def l2_distance(a, b):
    dist = torch.norm(a.unsqueeze(1) - b.unsqueeze(0), p=2, dim=2)
    return -dist

def l1_distance(a, b):
    dist = torch.sum(torch.abs(a.unsqueeze(1) - b.unsqueeze(0)), dim=2)
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

def lsh_similarity_wrapper(a, b):
    return lsh_similarity(a, b, num_bits=OUTPUT_DIM*8)

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

def train_model(model, data_by_class, n_way, k_shot, n_query, num_epochs, 
               episodes_per_epoch, optimizer, criterion, device, use_noise=False, noise_std=0.1):
    model.train()
    
    for epoch in range(num_epochs): 
        for episode in range(episodes_per_epoch):
            try:
                support_imgs, support_labels, query_imgs, query_labels = sample_episode(
                    data_by_class, n_way, k_shot, n_query)
                support_imgs, support_labels = support_imgs.to(device), support_labels.to(device)
                query_imgs, query_labels = query_imgs.to(device), query_labels.to(device)
                
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
                                test_episodes, distance_fn, device):
    model.eval()
    total_correct, total_num = 0, 0
    
    with torch.no_grad():
        for _ in range(test_episodes):
            support_imgs, support_labels, query_imgs, query_labels = sample_episode(
                data_by_class, n_way, k_shot, n_query)
            support_imgs, support_labels = support_imgs.to(device), support_labels.to(device)
            query_imgs, query_labels = query_imgs.to(device), query_labels.to(device)
            
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
# Distance Function Mapping
# -------------------------------

def get_distance_function(metric_name):
    """Get distance function by name"""
    distance_funcs = {
        "Cosine": cosine_similarity,
        "L-2": l2_distance,
        "L-1": l1_distance,
        "L-inf": linf_distance,
        "LSH": lsh_similarity_wrapper
    }
    return distance_funcs[metric_name]

# -------------------------------
# Worker Function for Threading
# -------------------------------

def run_single_noise_experiment(gpu_id, noise_idx, noise_std, distance_metric_names, results_lock):
    """Run experiment for one noise value on specific GPU"""
    
    try:
        # Force set the CUDA device at the very beginning
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)  # Use device 0 within the visible devices
        device = torch.device("cuda:0")
        
        # Verify GPU assignment
        actual_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        
        with results_lock:
            print(f"Worker GPU {gpu_id}: Starting noise_std={noise_std:.1f} (noise_idx={noise_idx})")
            print(f"Worker GPU {gpu_id}: Using device {actual_device}, name: {device_name}")
            
    except Exception as e:
        with results_lock:
            print(f"GPU {gpu_id}: Error setting up device: {str(e)}")
        # Fallback to original method
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        
        with results_lock:
            print(f"GPU {gpu_id}: Fallback - Starting noise_std={noise_std:.1f} (noise_idx={noise_idx})")
    
    # Results for this noise level
    noise_results = np.zeros((len(distance_metric_names), NUM_ITER_AVG))
    
    for iteration in range(NUM_ITER_AVG):
        try:
            # Set random seeds for reproducibility
            torch.manual_seed(42 + iteration + noise_idx * 1000)
            np.random.seed(42 + iteration + noise_idx * 1000)
            random.seed(42 + iteration + noise_idx * 1000)
            
            # Create model for this GPU with explicit device setting
            if DATASET in ['omniglot', 'mnist']:
                model = MANN(dataset=DATASET, out_dim=OUTPUT_DIM)
            else:
                model = EnhancedMANN(dataset=DATASET, out_dim=OUTPUT_DIM)
            
            # Explicitly move model to correct GPU
            model = model.to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Verify model is on correct device
            model_device = next(model.parameters()).device
            with results_lock:
                if iteration == 0:  # Only print for first iteration
                    print(f"GPU {gpu_id}: Model on device {model_device}")
            
            # Train model
            use_noise = noise_std > 0.0
            train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, 
                       EPISODES_PER_EPOCH, optimizer, criterion, device, use_noise, noise_std)
            
            # Test with different distance metrics
            for metric_idx, metric_name in enumerate(distance_metric_names):
                metric_fn = get_distance_function(metric_name)
                acc = test_model_with_quantization(
                    model, test_data, N_WAY, K_SHOT, N_QUERY, TEST_EPISODES, metric_fn, device)
                noise_results[metric_idx, iteration] = acc
            
            del model, optimizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            with results_lock:
                print(f"GPU {gpu_id}: Error in iteration {iteration+1} for noise_std={noise_std:.1f}: {str(e)}")
            for metric_idx in range(len(distance_metric_names)):
                noise_results[metric_idx, iteration] = np.nan
    
    with results_lock:
        print(f"GPU {gpu_id}: Completed noise_std={noise_std:.1f} (noise_idx={noise_idx})")
    
    return noise_idx, noise_results

# -------------------------------
# Main Multi-GPU Experiment
# -------------------------------

# -------------------------------
# Alternative Multi-GPU Implementation
# -------------------------------

# -------------------------------
# Threading-based Multi-GPU Implementation
# -------------------------------

def run_gpu_worker_thread(gpu_id, assigned_tasks, distance_metric_names, results_dict, results_lock):
    """Thread-based worker function for multi-GPU processing"""
    
    try:
        # Set up GPU context
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        
        with results_lock:
            print(f"GPU {gpu_id} Worker: Initialized, processing {len(assigned_tasks)} tasks")
            print(f"GPU {gpu_id} Worker: Tasks = {[f'noise_{t[1]:.1f}' for t in assigned_tasks]}")
        
        # Results storage
        worker_results = {}
        
        for task_idx, (noise_idx, noise_std) in enumerate(assigned_tasks):
            with results_lock:
                print(f"GPU {gpu_id}: Starting task {task_idx+1}/{len(assigned_tasks)} - noise_std={noise_std:.1f}")
            
            # Process this noise value
            noise_results = np.zeros((len(distance_metric_names), NUM_ITER_AVG))
            
            for iteration in range(NUM_ITER_AVG):
                try:
                    # Set random seeds
                    torch.manual_seed(42 + iteration + noise_idx * 1000)
                    np.random.seed(42 + iteration + noise_idx * 1000)
                    random.seed(42 + iteration + noise_idx * 1000)
                    
                    # Create and train model
                    if DATASET in ['omniglot', 'mnist']:
                        model = MANN(dataset=DATASET, out_dim=OUTPUT_DIM).to(device)
                    else:
                        model = EnhancedMANN(dataset=DATASET, out_dim=OUTPUT_DIM).to(device)
                    
                    optimizer = optim.Adam(model.parameters(), lr=1e-3)
                    criterion = nn.CrossEntropyLoss()
                    
                    use_noise = noise_std > 0.0
                    train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, 
                               EPISODES_PER_EPOCH, optimizer, criterion, device, use_noise, noise_std)
                    
                    # Test with all metrics
                    for metric_idx, metric_name in enumerate(distance_metric_names):
                        metric_fn = get_distance_function(metric_name)
                        acc = test_model_with_quantization(
                            model, test_data, N_WAY, K_SHOT, N_QUERY, TEST_EPISODES, metric_fn, device)
                        noise_results[metric_idx, iteration] = acc
                    
                    del model, optimizer
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    with results_lock:
                        print(f"GPU {gpu_id}: Error in iteration {iteration+1}: {str(e)}")
                    for metric_idx in range(len(distance_metric_names)):
                        noise_results[metric_idx, iteration] = np.nan
            
            # Store results
            worker_results[noise_idx] = noise_results
            with results_lock:
                print(f"GPU {gpu_id}: Completed task {task_idx+1}/{len(assigned_tasks)} - noise_std={noise_std:.1f}")
        
        # Store results in shared dict
        with results_lock:
            results_dict.update(worker_results)
            print(f"GPU {gpu_id} Worker: All tasks completed!")
            
    except Exception as e:
        with results_lock:
            print(f"GPU {gpu_id} Worker: Fatal error - {str(e)}")

def experiment_noise_sweep():
    """
    Multi-GPU Noise standard deviation sweep experiment using threading
    """
    print("\n=== Multi-GPU Noise Standard Deviation Sweep Experiment ===")
    print(f"Settings: EPOCHS={NUM_EPOCHS}, EPISODES_PER_EPOCH={EPISODES_PER_EPOCH}")
    print(f"TEST_EPISODES={TEST_EPISODES}, ITERATIONS={NUM_ITER_AVG}")
    print(f"Available GPUs: {NUM_GPUS}")
    
    # Define noise range and distance metrics
    noise_stds = np.arange(0.0, 3.05, 0.2)
    distance_metric_names = ["Cosine", "L-2", "L-1", "L-inf", "LSH"]
    
    print(f"Noise std range: {noise_stds[0]:.1f} to {noise_stds[-1]:.1f} ({len(noise_stds)} values)")
    
    if NUM_GPUS < 2:
        print("Warning: Less than 2 GPUs available. Falling back to single GPU.")
        return experiment_noise_sweep_single_gpu()
    
    # Try threading approach
    try:
        # Distribute noise values across GPUs
        noise_indices = list(range(len(noise_stds)))
        gpu_tasks = [[] for _ in range(NUM_GPUS)]
        
        # Round-robin assignment
        for i, noise_idx in enumerate(noise_indices):
            gpu_id = i % NUM_GPUS
            gpu_tasks[gpu_id].append((noise_idx, noise_stds[noise_idx]))
        
        print("\nGPU Task Assignment:")
        for gpu_id, tasks in enumerate(gpu_tasks):
            task_names = [f"noise_{noise_std:.1f}" for _, noise_std in tasks]
            print(f"  GPU {gpu_id}: {len(tasks)} tasks - {task_names}")
        
        # Shared results dictionary and lock
        results_dict = {}
        results_lock = threading.Lock()
        
        # Create and start threads
        threads = []
        for gpu_id in range(NUM_GPUS):
            if gpu_tasks[gpu_id]:  # Only create thread if there are tasks
                thread = threading.Thread(
                    target=run_gpu_worker_thread,
                    args=(gpu_id, gpu_tasks[gpu_id], distance_metric_names, results_dict, results_lock)
                )
                threads.append(thread)
                thread.start()
                print(f"Started thread for GPU {gpu_id}")
        
        # Wait for all threads to complete
        print(f"\nWaiting for {len(threads)} GPU workers to complete...")
        for i, thread in enumerate(threads):
            thread.join()
            print(f"GPU thread {i} completed")
        
        print("All GPU workers completed!")
        
        # Reconstruct results array
        all_results = np.zeros((len(noise_stds), len(distance_metric_names), NUM_ITER_AVG))
        for noise_idx in range(len(noise_stds)):
            if noise_idx in results_dict:
                all_results[noise_idx] = results_dict[noise_idx]
            else:
                print(f"Warning: Missing results for noise_idx {noise_idx}")
                all_results[noise_idx] = np.nan
        
        # Convert back to distance_metrics format for visualization
        distance_metrics = [(name, get_distance_function(name)) for name in distance_metric_names]
        
        # Save and visualize results
        save_and_visualize_results(all_results, noise_stds, distance_metrics)
        
        return all_results
        
    except Exception as e:
        print(f"Threading failed: {str(e)}")
        print("Falling back to single GPU mode...")
        return experiment_noise_sweep_single_gpu()

def experiment_noise_sweep_single_gpu():
    """Fallback single GPU experiment"""
    print("Running single GPU experiment...")
    
    noise_stds = np.arange(0.0, 3.05, 0.2)
    distance_metrics = [
        ("Cosine", cosine_similarity),
        ("L-2", l2_distance), 
        ("L-1", l1_distance),
        ("L-inf", linf_distance),
        ("LSH", lsh_similarity_wrapper),
    ]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_results = np.zeros((len(noise_stds), len(distance_metrics), NUM_ITER_AVG))
    
    for noise_idx, noise_std in enumerate(noise_stds):
        print(f"\nNoise Std: {noise_std:.1f} ({noise_idx+1}/{len(noise_stds)})")
        
        for iteration in range(NUM_ITER_AVG):
            print(f"  Iteration {iteration+1}/{NUM_ITER_AVG}")
            
            try:
                if DATASET in ['omniglot', 'mnist']:
                    model = MANN(dataset=DATASET, out_dim=OUTPUT_DIM).to(device)
                else:
                    model = EnhancedMANN(dataset=DATASET, out_dim=OUTPUT_DIM).to(device)
                
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                criterion = nn.CrossEntropyLoss()
                
                use_noise = noise_std > 0.0
                train_model(model, train_data, N_WAY, K_SHOT, N_QUERY, NUM_EPOCHS, 
                           EPISODES_PER_EPOCH, optimizer, criterion, device, use_noise, noise_std)
                
                for metric_idx, (metric_name, metric_fn) in enumerate(distance_metrics):
                    acc = test_model_with_quantization(
                        model, test_data, N_WAY, K_SHOT, N_QUERY, TEST_EPISODES, metric_fn, device)
                    all_results[noise_idx, metric_idx, iteration] = acc
                
                del model, optimizer
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                for metric_idx in range(len(distance_metrics)):
                    all_results[noise_idx, metric_idx, iteration] = np.nan
    
    save_and_visualize_results(all_results, noise_stds, distance_metrics)
    return all_results

# -------------------------------
# Visualization Functions
# -------------------------------

def save_and_visualize_results(all_results, noise_stds, distance_metrics):
    """Save results and create visualizations"""
    
    # Calculate statistics
    mean_results = np.nanmean(all_results, axis=2)
    std_results = np.nanstd(all_results, axis=2)
    
    # Save to files
    np.savez(f"{DATASET}_noise_sweep_results_multigpu.npz",
             noise_stds=noise_stds,
             distance_metrics=[name for name, _ in distance_metrics],
             results=all_results)
    
    # Create detailed CSV
    detailed_results = []
    for i, noise_std in enumerate(noise_stds):
        for j, (metric_name, _) in enumerate(distance_metrics):
            detailed_results.append({
                'Noise_Std': noise_std,
                'Distance_Metric': metric_name,
                'Mean_Accuracy': mean_results[i, j] if not np.isnan(mean_results[i, j]) else None,
                'Std_Dev': std_results[i, j] if not np.isnan(std_results[i, j]) else None
            })
    
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{DATASET}_detailed_results_multigpu.csv", index=False)
    
    # Create visualization
    create_bar_chart(mean_results, std_results, noise_stds, distance_metrics)
    
    # Print summary
    print_summary(mean_results, std_results, noise_stds, distance_metrics)

def create_bar_chart(mean_results, std_results, noise_stds, distance_metrics):
    """Create bar chart visualization"""
    
    selected_indices = list(range(len(noise_stds)))
    selected_noise_stds = noise_stds[selected_indices]
    selected_means = mean_results[selected_indices, :]
    selected_stds = std_results[selected_indices, :]
    
    plt.figure(figsize=(20, 12))
    
    # Color mapping
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(selected_noise_stds)))
    
    # Bar settings
    num_noise = len(selected_noise_stds)
    num_metrics = len(distance_metrics)
    bar_width = 0.8 / num_noise
    x_pos = np.arange(num_metrics)
    
    # Plot bars
    for metric_idx in range(num_metrics):
        for noise_idx, (noise_std, color) in enumerate(zip(selected_noise_stds, colors)):
            accuracy = selected_means[noise_idx, metric_idx]
            error = selected_stds[noise_idx, metric_idx]
            
            if not np.isnan(accuracy):
                x_position = x_pos[metric_idx] + (noise_idx - num_noise/2 + 0.5) * bar_width
                plt.bar(x_position, accuracy, width=bar_width, color=color, alpha=0.8,
                       yerr=error, capsize=2, edgecolor='black', linewidth=0.3,
                       label=f'Std={noise_std:.1f}' if metric_idx == 0 else "")
    
    # Formatting
    plt.xlabel('Distance Metric', fontweight='bold', fontsize=16)
    plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=16)
    plt.ylim(78,88)
    plt.title('Multi-GPU Performance vs Noise Standard Deviation', fontweight='bold', fontsize=18)
    plt.xticks(x_pos, [name for name, _ in distance_metrics], fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14)
    
    # Legend
    legend_indices = list(range(0, len(selected_noise_stds), max(1, len(selected_noise_stds)//8)))
    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.8) for i in legend_indices]
    legend_labels = [f'Std={selected_noise_stds[i]:.1f}' for i in legend_indices]
    plt.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{DATASET}_noise_sweep_results_multigpu.png", dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(mean_results, std_results, noise_stds, distance_metrics):
    """Print experiment summary"""
    print("\n=== Multi-GPU Experiment Summary ===")
    
    # Find best settings for each metric
    best_settings = {}
    for j, (metric_name, _) in enumerate(distance_metrics):
        valid_mask = ~np.isnan(mean_results[:, j])
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            valid_means = mean_results[valid_mask, j]
            best_idx = valid_indices[np.argmax(valid_means)]
            
            best_settings[metric_name] = {
                'noise_std': noise_stds[best_idx],
                'accuracy': mean_results[best_idx, j],
                'std': std_results[best_idx, j]
            }
            
            print(f"{metric_name}:")
            print(f"  Best noise std: {noise_stds[best_idx]:.1f}")
            print(f"  Best accuracy: {mean_results[best_idx, j]:.2f}% ± {std_results[best_idx, j]:.2f}%")
    
    # Save best settings
    best_df = pd.DataFrame.from_dict(best_settings, orient='index')
    best_df.to_csv(f"{DATASET}_best_settings_multigpu.csv")
    
    # Global best
    global_best_idx = np.unravel_index(np.nanargmax(mean_results), mean_results.shape)
    print(f"\nGlobal best:")
    print(f"  Noise std: {noise_stds[global_best_idx[0]]:.1f}")
    print(f"  Metric: {distance_metrics[global_best_idx[1]][0]}")
    print(f"  Accuracy: {mean_results[global_best_idx]:.2f}% ± {std_results[global_best_idx]:.2f}%")

# -------------------------------
# Main Program
# -------------------------------

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    experiment_noise_sweep()
    print("Multi-GPU experiment completed!")