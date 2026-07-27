import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14

    epochs = np.arange(1, 7)
    
    # Static schedule: 1.0 from beginning to end
    static_m = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    # Annealed schedule: 1.0 -> 0.7 -> 0.5 -> 0.3 (held at 0.3)
    annealed_m = [1.0, 0.7, 0.5, 0.3, 0.3, 0.3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Static Strategy
    axes[0].plot(epochs, static_m, marker='s', markersize=8, linewidth=2.5, color='#d62728', label='Static (m=1.0)')
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel(r'Margin Coefficient $m(\epsilon)$', fontweight='bold')
    axes[0].set_title('Static Strategy', fontweight='bold', fontsize=16)
    axes[0].set_ylim(0.2, 1.1)
    axes[0].set_xticks(epochs)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='lower left')

    # Subplot 2: Annealed Strategy
    axes[1].plot(epochs, annealed_m, marker='o', markersize=8, linewidth=2.5, color='#1f77b4', label='Annealed (Dynamic)')
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel(r'Margin Coefficient $m(\epsilon)$', fontweight='bold')
    axes[1].set_title('Dynamic Boundary-Regularization', fontweight='bold', fontsize=16)
    axes[1].set_ylim(0.2, 1.1)
    axes[1].set_xticks(epochs)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='lower left')

    plt.tight_layout()
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('margin_schedule.pdf', bbox_inches='tight')
    plt.savefig('margin_schedule.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/margin_schedule.pdf', bbox_inches='tight')
    plt.savefig('figures/margin_schedule.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved margin_schedule.png and margin_schedule.pdf to root and figures/")

if __name__ == '__main__':
    main()
