*Figure 2
Create a publication-quality Python script that visualizes the Sinkhorn Optimal Transport plan γ used in a Cross-Lingual Question Answering model.

Requirements

- Use matplotlib only (no seaborn unless optional).
- Export PDF and SVG.
- Figure size suitable for ACL (about 6 inches wide).
- Font:
    Times New Roman
    fontsize=10

Input

gamma.npy

shape

(T_en, T_vi)

Optional

english_tokens.txt

vietnamese_tokens.txt

Output

A heatmap showing

Y-axis:
English tokens

X-axis:
Vietnamese tokens

Requirements

- Viridis colormap
- Remove unnecessary grid lines
- Show colorbar
- Tick labels rotated 45 degrees
- High resolution
- Tight layout

Highlight

Optionally draw a red rectangle around the transported answer region if answer span indices are provided.

Caption

Visualization of the Optimal Transport plan γ between English and Vietnamese token representations, demonstrating barycentric cross-lingual alignment.

Export

figure2_ot_heatmap.pdf

figure2_ot_heatmap.svg

figure2_ot_heatmap.png (600 dpi)

The code should be modular and publication ready.

*Figure 3:
Create a publication-quality Python script that generates Figure 3 consisting of two subfigures.

=================================

Subfigure (a)

Layer Mixing Weights

=================================

Input

layer_weights.npy

contains four learnable weights corresponding to

Layer 6
Layer 7
Layer 8
Layer 9

Normalize them using softmax.

Plot

Bar chart

X-axis

Layers

Y-axis

Normalized Weight

Use clean monochrome bars.

Annotate values above each bar.

=================================

Subfigure (b)

Dynamic Margin Schedule

=================================

Input

margin_history.csv

Columns

epoch
margin

Plot

Epoch vs Margin

Requirements

Smooth line

Circle markers

Grid disabled

Publication style

=================================

Layout

Two horizontal subfigures

(a)
Layer Mixing

(b)
Margin Schedule

Common font

Times New Roman

fontsize=10

Export

figure3.pdf

figure3.svg

figure3.png

600 dpi

The figure should follow ACL/EMNLP publication style.

*Figure 4:
Create a publication-quality radar chart for an ablation study.

Input

ablation.csv

Rows

Baseline
OT
OT+Span
OT+Span+Margin
Full

Columns

SQuAD_EN
MLQA_VI
XQuAD_VI

Requirements

Create one radar polygon for each method.

Use different line styles.

Light transparent fills.

Legend outside the plot.

Axes

SQuAD EN

MLQA VI

XQuAD VI

Normalize all metrics to [0,1].

Use matplotlib PolarAxes.

No background colors.

Use Times New Roman.

fontsize=10.

Export

figure4.pdf

figure4.svg

figure4.png

600 dpi

The radar chart should resemble figures commonly seen in ACL, EMNLP, or NeurIPS papers.

*Figure 5:
Create a publication-quality figure illustrating how Optimal Transport improves cross-lingual representation alignment.

The figure contains three horizontally arranged panels.

================================================

(a) Before OT

================================================

Input

hidden_before.npy

Shape

(N, D)

Each point has

language

EN or VI

label

answer token or normal token

Project embeddings into two dimensions using UMAP (preferred) or t-SNE.

Plot

English tokens

Blue circles

Vietnamese tokens

Orange triangles

Answer tokens

Red stars

Use equal axis scaling.

No grid.

================================================

(b) OT Transport

================================================

Input

gamma.npy

Visualize the transport matrix γ as a heatmap.

Axes

English Tokens

Vietnamese Tokens

Use Viridis colormap.

Highlight answer token transport if answer spans are available.

================================================

(c) After OT

================================================

Input

hidden_after.npy

Repeat the same projection procedure.

Maintain identical plotting parameters as panel (a).

The projection should preserve global geometry as much as possible.

================================================

Layout

Three panels arranged horizontally.

Titles

(a) Before Alignment

(b) Optimal Transport Plan

(c) After Alignment

Common legend

Blue circle = English

Orange triangle = Vietnamese

Red star = Answer token

Publication style

Times New Roman

fontsize=10

White background

No grid

Minimal ticks

Export

figure5.pdf

figure5.svg

figure5.png

600 dpi

The figure should clearly demonstrate that Optimal Transport reduces the cross-lingual representation gap while preserving answer-aware local structures instead of collapsing the embedding space.