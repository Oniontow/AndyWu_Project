import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read the CSV file
csv_path = os.path.join(script_dir, 'omniglot_mann_reduction_results.csv')
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: File not found at {csv_path}")
    exit()

# Filter out Baseline to match the 5 bars in the screenshot
df_filtered = df[df['Method'] != 'Baseline (Float32)'].copy()

# Mapping names to match the screenshot labels
name_mapping = {
    'Direct INT3': 'M1\nDirect INT3',
    'AvgPool + INT3': 'M2\nAvgPooling + INT3',
    'PCA + INT3': 'M3\nPCA + INT3',
    'MaxMagPool + INT3': 'M4\nMax Mag Pooling + INT3',
    'AE + INT3': 'M5\nAutoEncoder + INT3'
}

df_filtered['Label'] = df_filtered['Method'].map(name_mapping)

# If mapping failed (e.g. exact string mismatch), fill with original
df_filtered['Label'] = df_filtered['Label'].fillna(df_filtered['Method'])

# Data for plotting
methods = df_filtered['Label']
accuracy = df_filtered['Accuracy']

# Colors from the screenshot (Standard Matplotlib cycle)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create the plot
plt.figure(figsize=(10, 6))
bars = plt.bar(methods, accuracy, color=colors, width=0.8)

# Add title and labels
plt.title('Omniglot 20-Way-5-shot', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Set Y-axis limit to 0.8 (80%) as requested
plt.ylim(0.8, 1.01)

# Add grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom')

# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()

# Save the plot
output_path = os.path.join(script_dir, 'plot.png')
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")
