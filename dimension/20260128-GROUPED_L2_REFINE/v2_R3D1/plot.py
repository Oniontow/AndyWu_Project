import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(csv_file='result_grouped_vs_standard_full.csv'):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run the experiment in the notebook first.")
        return

    df_res = pd.read_csv(csv_file)
    
    # Metrics to plot
    metrics = ['recall@100', 'recall@500', 'recall@1000']
    
    # Setup the figure with 3 subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Define ordered methods
    ordered_methods = ['Direct INT3 (128D)', 'AvgPool (64D)', 'PCA (64D)', 'AutoEncoder (64D)']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Pivot table: Index=Method, Columns=Dist, Values=Metric
        df_plot = df_res.pivot(index='Method', columns='Dist', values=metric)
        
        # Reorder index
        existing_methods = [m for m in ordered_methods if m in df_plot.index]
        df_plot = df_plot.reindex(existing_methods)
        
        # Plot
        # Colors: Blue for Grouped L2, Orange for Standard L2 (default pandas/matplotlib colors)
        # We can map specific colors if needed, but defaults are usually distinct.
        df_plot.plot(kind='bar', ax=ax, width=0.8, color=['skyblue', 'orange'])
        
        ax.set_title(f'{metric} Comparison', fontsize=14)
        ax.set_xlabel("Method", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(title='Distance Metric')
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    plt.suptitle('Method Comparison on Full SIFT1M (Grouped vs Standard L2)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    output_file = 'grouped_vs_standard_comparison_all_recalls.png'
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_results()
