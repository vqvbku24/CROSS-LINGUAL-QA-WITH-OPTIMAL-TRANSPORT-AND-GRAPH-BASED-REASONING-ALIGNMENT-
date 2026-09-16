# Coordinated Alignment Framework

## Overview

This figure illustrates a two-stage cross-lingual QA framework. Stage 1 pretrains a frozen English teacher model, while Stage 2 adapts the target language through a coordinated alignment framework composed of three complementary objectives.

The overall design emphasizes that all alignment modules operate jointly within a shared representation space.

---

# Stage 1: English Teacher Pretraining

**Pipeline**

English SQuAD 2.0
→ XLM-R Encoder
→ Frozen Teacher

The frozen teacher provides:

- Stable source representations
- Source geometry
- Span anchors for knowledge transfer

A single arrow connects the teacher to the Stage 2 framework.

---

# Stage 2: Coordinated Alignment Framework

Main title centered at the top.

Subtitle:

> Jointly optimizes Spatial Coordination · Domain Coordination · Boundary Regularization

The framework is enclosed inside one large rounded rectangle.

The upper portion is surrounded by a dashed border representing the **Shared Representation Space**.

---

# Module 1: Spatial Coordination

Subtitle:

> Reduce the cross-lingual representation gap

This module contains two side-by-side blocks.

## 1. Macro Alignment

Illustrates Optimal Transport aligning target representations toward the English geometry.

Layout:

Before Alignment

EN geometry:
● ● ●

Target geometry:
○ ○ ○

Dashed correspondence lines connect the two sets.

An arrow labeled:

> Optimal Transport

points downward.

After Alignment

EN geometry:
● ● ●

Aligned Target:
● ● ●

Loss label positioned on the right:

L_OT

---

## 2. Micro Alignment

Illustrates span-level projection.

Before Projection

Teacher:

[ answer ]

Target:

[ answer ]

A vertical arrow labeled

γ Projection

points downward.

After Projection

Teacher:

[ answer ]

Target:

[ answer ]

Loss label:

L_span

---

# Module 2: Domain Coordination

Subtitle:

> Keep target adaptation close to source representations

Layout:

Before

Teacher (anchor)

●

Student drift

○

A horizontal arrow labeled

Consistency Constraint

points toward

After

Teacher (anchor)

●

Guided Student

●

Loss label:

L_domain

---

# Module 3: Boundary Regularization

Subtitle:

> Increase confidence between competing spans (Task / Logit Space)

Before

Top-1 ── Top-2

↓

Margin Regularization

↓

After

Top-1 ─────────────── Top-2

The margin becomes visibly larger after regularization.

Loss label:

L_margin

---

# Footer

Remove the previous "Overall Objective" section completely.

Do not include:

L_total = L_task + λ1 L_OT + λ2 L_span + λ3 L_domain + λ4 L_margin

Redistribute the remaining modules vertically so the figure fills the space naturally.

---

# Legend

Place a compact legend at the bottom-right.

Blue filled circle:
Teacher / Source

Green filled circle:
Adapted / Target

White circle:
Before / Drift

---

# Design Style

- Clean ACL/EACL publication style
- Larger typography (20–35% larger)
- Consistent spacing and alignment
- Rounded containers
- Thin blue-gray borders
- Light gray background panels
- Dark blue section headers
- Green used only for adaptation/alignment cues
- Preserve all scientific meaning while improving readability
- Ensure every label remains readable in a two-column conference paper