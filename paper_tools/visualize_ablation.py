import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_dummy_data():
    data = {
        'Method': ['Baseline', 'OT', 'OT+Span', 'OT+Span+Margin', 'Full'],
        'SQuAD_EN': [75.0, 75.5, 76.0, 77.2, 78.5],
        'MLQA_VI': [55.0, 58.2, 60.1, 61.5, 63.0],
        'XQuAD_VI': [60.0, 63.5, 65.2, 66.8, 68.4]
    }
    df = pd.DataFrame(data)
    df.to_csv('ablation.csv', index=False)

def plot_figure4():
    # Setup fonts
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    
    if not os.path.exists('ablation.csv'):
        print("Generating dummy data...")
        generate_dummy_data()
        
    df = pd.read_csv('ablation.csv')
    
    methods = df['Method'].values
    metrics = ['SQuAD_EN', 'MLQA_VI', 'XQuAD_VI']
    
    # Normalize metrics to [0, 1] per column for visualization purposes
    # Or normalize globally. Usually min-max scaling per column is used.
    norm_df = df.copy()
    for col in metrics:
        min_val = norm_df[col].min()
        max_val = norm_df[col].max()
        norm_df[col] = (norm_df[col] - min_val) / (max_val - min_val)
        
    # Radar chart setup
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # Close the loop
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Remove background colors
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Custom ticks
    plt.xticks(angles[:-1], ['SQuAD EN', 'MLQA VI', 'XQuAD VI'])
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.05)
    
    line_styles = ['-', '--', '-.', ':', '-']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, method in enumerate(methods):
        values = norm_df.loc[i, metrics].values.flatten().tolist()
        values += values[:1] # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle=line_styles[i % len(line_styles)], color=colors[i % len(colors)], label=method)
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
        
    # Legend outside
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    
    # Save
    plt.savefig('figure4.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('figure4.svg', format='svg', bbox_inches='tight')
    plt.savefig('figure4.png', format='png', dpi=600, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_figure4()
