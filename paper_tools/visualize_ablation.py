import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_ablation_data():
    # Các nhãn đã được note rõ ràng chức năng (Baseline, OT, Margin...) 
    # ghép với mã M0-M5 để hội đồng phản biện dễ theo dõi tiến trình
    data = {
        'Method': [
            'M0: Zero-shot (Baseline)', 
            'M1: Vanilla KD', 
            'M2: OT Only (Global)', 
            'M3: OT + Span (Local)', 
            'M4: Static Margin', 
            'M5: Ours (Dynamic)'
        ],
        # Điểm F1 tương ứng từ bảng LaTeX
        'SQuAD_EN': [64.26, 65.65, 66.08, 64.58, 65.64, 64.84],
        'MLQA_VI': [46.05, 41.34, 36.08, 44.35, 45.41, 46],
        
        # TODO: Nhập điểm F1 của XQuAD_VI tương ứng với các cấu hình M0 -> M5 vào list dưới đây
        'XQuAD_VI': [46.22, 45.55 , 38.66 , 46.62 , 51.26 , 51.26] 
    }
    df = pd.DataFrame(data)
    # Lưu file với chuẩn UTF-8 để tránh lỗi font
    df.to_csv('ablation.csv', index=False, encoding='utf-8')

def plot_figure4():
    # Setup fonts
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    
    if not os.path.exists('ablation.csv'):
        print("Generating ablation data...")
        generate_ablation_data()
        
    df = pd.read_csv('ablation.csv', encoding='utf-8')
    
    methods = df['Method'].values
    metrics = ['SQuAD_EN', 'MLQA_VI', 'XQuAD_VI']
    
    # Normalize metrics to [0, 1] per column for visualization purposes
    norm_df = df.copy()
    for col in metrics:
        min_val = norm_df[col].min()
        max_val = norm_df[col].max()
        # Tránh lỗi chia cho 0 nếu data XQuAD_VI chưa được nhập
        if max_val - min_val == 0:
            norm_df[col] = 0.001
        else:
            norm_df[col] = (norm_df[col] - min_val) / (max_val - min_val)
        
    # Radar chart setup
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # Close the loop
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    # Remove background colors
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Custom ticks
    plt.xticks(angles[:-1], ['SQuAD EN (Source)', 'MLQA VI (Target)', 'XQuAD VI (Target)'], size=11, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.05)
    
    # Update styles and colors for 6 methods
    line_styles = ['-', '--', '-.', ':', '-', '--']
    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    linewidths = [1.5, 1.5, 1.5, 1.5, 2.0, 2.5] # Nhấn mạnh M4 và M5
    
    for i, method in enumerate(methods):
        values = norm_df.loc[i, metrics].values.flatten().tolist()
        values += values[:1] # Close the loop
        
        ax.plot(angles, values, linewidth=linewidths[i], linestyle=line_styles[i], color=colors[i], label=method)
        
        # Chỉ fill màu cho phương pháp đề xuất (M5) để làm nổi bật
        if 'Ours' in method:
            ax.fill(angles, values, color=colors[i], alpha=0.15)
        
    # Legend outside
    plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1), title="Configurations", title_fontsize='11')
    
    plt.tight_layout()
    
    # Save
    plt.savefig('figure4_EM.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('figure4_EM.svg', format='svg', bbox_inches='tight')
    plt.savefig('figure4_EM.png', format='png', dpi=600, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_figure4()