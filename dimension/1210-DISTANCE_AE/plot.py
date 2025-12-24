import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Read the Excel file
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'AE_distance_w_table2.xlsx')

try:
    df = pd.read_excel(file_path)
    
    # Extract weight from the label column (assuming it's the second column 'Unnamed: 1')
    # The column name might vary if I read it differently, but based on previous output it is 'Unnamed: 1'
    # Let's rename it for clarity
    df.rename(columns={'Unnamed: 1': 'Label'}, inplace=True)
    
    # Function to extract weight
    def extract_weight(label):
        match = re.search(r'w=([0-9.]+)', str(label))
        if match:
            return float(match.group(1))
        return None

    df['Weight'] = df['Label'].apply(extract_weight)
    
    # Sort by weight just in case
    df = df.sort_values('Weight')
    
    print("Data to plot:")
    print(df[['Label', 'Weight', 'recall@100', 'recall@500', 'recall@1000']])

    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(df['Weight'], df['recall@100'], marker='o', label='Recall@100')
    plt.plot(df['Weight'], df['recall@500'], marker='s', label='Recall@500')
    plt.plot(df['Weight'], df['recall@1000'], marker='^', label='Recall@1000')
    
    plt.title('Recall vs Distance Loss Weight')
    plt.xlabel('Distance Loss Weight')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_path = os.path.join(script_dir, 'ae_distance_weight_comparison2.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

except Exception as e:
    print(f"Error: {e}")
