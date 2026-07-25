import matplotlib.pyplot as plt

def plot_figure2():
    # Setup fonts
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12
    
    # Data from schedule: Epoch 1 (0.0), Epoch 2-3 (1.0), Epoch 4 (0.7), Epoch 5 (0.5), Epoch 6-8 (0.3)
    epochs = [1, 2, 3, 4, 5, 6]
    margin = [0.0, 1.0, 0.7, 0.5, 0.3, 0.3]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Draw step function (post means the step occurs at the end of the interval, 
    # but here we want values at exact epochs. Let's just plot it as a line with markers or step with 'mid')
    # Actually, a step plot with 'post' makes sense if it holds the value until the next epoch.
    # However, standard line plot with markers is often clearer. Let's use a step plot.
    
    ax.step(epochs, margin, where='post', color='#d62728', linewidth=2.5, marker='o', markersize=6)
    
    # Fill under curve
    ax.fill_between(epochs, margin, step='post', color='#d62728', alpha=0.1)
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Margin Threshold $m(\epsilon)$', fontsize=14)
    
    ax.set_xticks(epochs)
    ax.set_ylim(-0.05, 1.1)
    
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Save
    plt.savefig('figure2_margin_schedule.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('figure2_margin_schedule.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure3b_margin_schedule.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('figures/figure3b_margin_schedule.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved figure2_margin_schedule.pdf and figures/figure3b_margin_schedule.pdf")

if __name__ == '__main__':
    plot_figure2()
