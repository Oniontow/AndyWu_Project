import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_csv(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Plotting {file_path}...")
    df = pd.read_csv(file_path)
    
    # Determine the recall column name
    recall_col = 'Recall@100' if 'Recall@100' in df.columns else 'Recall'
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Results from {os.path.basename(file_path)}')

    # Plot Recall vs Dimension
    if recall_col in df.columns:
        for method in df['Method'].unique():
            subset = df[df['Method'] == method].sort_values('Dimension')
            ax1.plot(subset['Dimension'], subset[recall_col], marker='o', label=method)

        ax1.set_title(f'{recall_col} vs Dimension')
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel(recall_col)
        ax1.legend()
        ax1.grid(True)
    
    # Plot Time vs Dimension
    if 'Time' in df.columns:
        for method in df['Method'].unique():
            subset = df[df['Method'] == method].sort_values('Dimension')
            ax2.plot(subset['Dimension'], subset['Time'], marker='o', label=method)

        ax2.set_title('Time vs Dimension')
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Time (s)')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.show()

# List of files to plot
files = [
    'sift1m_dimension_sweep_results.csv',
    'sift1m_1M_1000at10000_sweep_results.csv'
]

for f in files:
    plot_csv(f)
