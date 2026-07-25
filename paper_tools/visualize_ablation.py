import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_ablation_data():
    # EM data from Table 3 (labeled tab:ablation_study) and Table 1 (tab:main_results)
    em_data = {
        'Method': [
            'M0: Zero-shot (Baseline)', 
            'M1: Vanilla KD', 
            'M2: OT Only (Global)', 
            'M3: OT + Span (Local)', 
            'M4: Dynamic Margin', 
            'M5: Ours (Static)'
        ],
        'SQuAD_EN': [64.26, 64.08, 66.08, 64.58, 66.21, 65.69],
        'MLQA_VI': [46.05, 33.79, 36.08, 44.35, 45.23, 46.0],
        'XQuAD_VI': [46.22, 37.39, 38.66, 46.62, 49.5, 49.7] 
    }
    
    # F1 data from Table 3 (labeled tab:ablation_study) and Table 1 (tab:main_results)
    f1_data = {
        'Method': [
            'M0: Zero-shot (Baseline)', 
            'M1: Vanilla KD', 
            'M2: OT Only (Global)', 
            'M3: OT + Span (Local)', 
            'M4: Dynamic Margin', 
            'M5: Ours (Static)'
        ],
        'SQuAD_EN': [72.84, 72.8, 72.75, 73.09, 74.37, 74.02],
        'MLQA_VI': [67.21, 56.39, 56.06, 65.33, 67.33, 67.6],
        # Note: adjust XQuAD_VI F1 values below if you have different experimental numbers
        'XQuAD_VI': [63.64, 62.55 , 57.63, 66.50, 69.28, 69.73] 
    }
    
    pd.DataFrame(em_data).to_csv('ablation_em.csv', index=False, encoding='utf-8')
    pd.DataFrame(f1_data).to_csv('ablation_f1.csv', index=False, encoding='utf-8')

def plot_bar_chart(csv_path, output_name, ylabel_prefix):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 11
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    methods = df['Method'].values
    metrics = ['SQuAD_EN', 'MLQA_VI', 'XQuAD_VI']
    labels = ['SQuAD-EN (Source)', 'MLQA-VI (Target)', 'XQuAD-VI (Target)']
    
    x = np.arange(len(metrics))
    width = 0.12
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Store M0 values for reference line
    m0_values = df.loc[0, metrics].values
    
    for i, method in enumerate(methods):
        values = df.loc[i, metrics].values
        offset = width * i - (width * len(methods)) / 2 + width / 2
        ax.bar(x + offset, values, width, label=method, color=colors[i], edgecolor='black', alpha=0.9)

    # Add reference lines for M0 (baseline)
    for j in range(len(metrics)):
        ax.hlines(m0_values[j], x[j] - 0.4, x[j] + 0.4, colors='black', linestyles='dashed', alpha=0.7, linewidth=1.5)
        
    ax.set_ylabel(f'{ylabel_prefix} Score', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Configurations", title_fontsize='11')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'{output_name}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'{output_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f'figures/{output_name}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'figures/{output_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_name} to root and figures/")

def main():
    # Force regeneration to include any updates
    generate_ablation_data()
    plot_bar_chart('ablation_f1.csv', 'figure3_grouped_bar', 'F1')
    plot_bar_chart('ablation_em.csv', 'figure4_grouped_bar', 'EM')

if __name__ == '__main__':
    main()