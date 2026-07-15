# Paper Tools

This module provides standalone tools to extract intermediate representations from the model without modifying the training logic, and scripts to visualize them for publication.

## Structure

- `exporter.py`: The main entry point. Loads a trained Stage-2 checkpoint, runs one evaluation forward pass, and extracts all representations (Optimal Transport plans, hidden states, logits, margins, tokens, etc.) needed to generate figures. Does NOT compute gradients or modify weights.
- `visualize_ot.py`: Generates Figure 2 (Optimal Transport Plan Heatmap).
- `visualize_layer.py`: Generates Figure 3 (Layer Mixing Weights & Margin Schedule).
- `visualize_ablation.py`: Generates Figure 4 (Radar Chart Ablation Study).
- `visualize_representation.py`: Generates Figure 5 (UMAP/t-SNE of Alignment).

## Usage

### 1. Extract Data

Run the exporter script to extract all required data for a given sample:

```bash
python paper_tools/exporter.py \
    --checkpoint checkpoints/best.pt \
    --config configs/stage2.yaml \
    --dataset data/xquad_vi.json \
    --output_dir paper_figures \
    --sample_index 15
```

All extracted tensors and metadata will be saved to the `paper_figures/` directory.

### 2. Generate Figures

Once the data is extracted, run the visualization scripts. (Note: you may need to copy or link the generated data from `paper_figures/` into the working directory of the visualization scripts or update their loading paths).

```bash
cd paper_tools
python visualize_ot.py
python visualize_layer.py
python visualize_ablation.py
python visualize_representation.py
```

The figures will be exported as `.pdf`, `.svg`, and `.png` (600 dpi) files.
