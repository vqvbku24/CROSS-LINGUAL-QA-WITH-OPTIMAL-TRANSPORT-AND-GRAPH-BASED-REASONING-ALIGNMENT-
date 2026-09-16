# Revision Specification

## Goal

Update the current ACL draft to accommodate the remaining experiments (MAD-X, InfoXLM, additional baselines), improve the paper organization, and prepare the manuscript for the final experimental results without requiring another structural rewrite later.

---

# Task 1 — Expand Main Result Tables

## Objective

Prepare all main experimental tables to include every baseline that will appear in the final paper.

Current paper only reports:

- XLM-R Zero-shot
- Vanilla KD
- MINOTAUR (Vietnamese only)
- Ours

We are currently re-running:

- MAD-X
- InfoXLM

These methods must already appear in every relevant table.

---

## Required Changes

### Table 2 (Main Cross-lingual Results)

Add new rows for:

- MAD-X
- InfoXLM

for every language section:

- Vietnamese
- Arabic
- Hindi

If results are unavailable, fill every metric with:

```
--
```

Example

```
Vietnamese

XLM-R Zero-shot
Vanilla KD
MINOTAUR
MAD-X
InfoXLM
Ours
```

Arabic

```
XLM-R Zero-shot
Vanilla KD
MAD-X
InfoXLM
Ours
```

Hindi

```
XLM-R Zero-shot
Vanilla KD
MAD-X
InfoXLM
Ours
```

---

### Update Table Caption

Remove any wording implying that MAD-X or InfoXLM are intentionally omitted.

Instead state that

- all baselines are evaluated under the same protocol whenever available
- entries marked "--" are experiments currently running and will be filled before submission.

---

# Task 2 — Rewrite MAD-X Discussion

Current manuscript repeatedly states that MAD-X is omitted because it is not directly comparable.

This is no longer correct.

---

## Required Changes

Search for every paragraph mentioning

- MAD-X
- InfoXLM
- Appendix M

Rewrite these paragraphs so that:

Instead of

> MAD-X is omitted...

use wording like

> We additionally evaluate adapter-based transfer (MAD-X) and contrastive representation learning (InfoXLM). Their results are included whenever available. Implementation details and protocol differences are discussed in Appendix M.

Remove every statement claiming numerical comparison is impossible.

Appendix M should now discuss:

- implementation differences
- how we retrained them
- any remaining protocol differences

instead of explaining why they were excluded.

---

# Task 3 — Reserve Space for Pending Figures

Several figures are still under development.

Do NOT attempt to redesign them.

Instead prepare placeholders.

---

## Figure 1

Keep the existing placeholder.

Leave sufficient vertical space.

Do not resize surrounding text.

---

## Additional Pending Figures

Leave placeholder environments for:

- final conceptual framework figure
- remaining baseline comparison figures
- any visualization not yet finalized

Use clearly marked placeholders.

Example

```
[Placeholder – final comparison figure]
```

---

# Task 4 — Move Boundary Schedule Figure to Appendix

Current paper places

Figure:

"The two boundary-regularization schedules"

inside Methodology.

This figure is implementation detail rather than a core contribution.

---

## Required Changes

Move the entire figure into an appropriate appendix.

Suggested appendix:

Implementation Details

or

Hyperparameter Settings

Update all references.

Instead of

> Figure 2

the paper should reference

> Appendix X (Figure X)

or equivalent.

Method section should briefly state

> The complete static and dynamic schedules are shown in Appendix X.

No dangling references should remain.

---

# Task 5 — Move Ablation Table to Appendix

Current Table

"Ablation study isolating the variables of the coordinated alignment framework"

should be moved.

Reason:

This is internal model analysis rather than headline experimental evidence.

---

## Required Changes

Move the entire table into an appendix dedicated to ablations.

Update every in-text reference.

Main text should summarize only the key findings and refer readers to the appendix for the complete table.

Example:

> Detailed ablation results are reported in Appendix X.

---

# Task 6 — Reorganize Source Preservation Table

Current Table:

Analysis of Source Preservation

is organized by configuration.

Reorganize it to match the structure used in

Zero-shot Cross-lingual Transfer Results.

---

## Desired Structure

Split by language.

Example
Teacher
### Vietnamese
```
Vanilla KD
MINOTAUR
MAD-X
InfoXLM
Ours
```

### Arabic

```
Vanilla KD
MINOTAUR
MAD-X
InfoXLM
Ours
```

### Hindi

```
Vanilla KD
MINOTAUR
MAD-X
InfoXLM
Ours
```

Each row should still report

- SQuAD-EN
- XQuAD-en
- MLQA-en

with

Pre

Post

Δsrc

columns unchanged.

Missing experiments should use

```
--
```

Do not remove existing numbers.

Only reorganize the table.

---

# Task 7 — Reserve Appendix Space

Ensure appendices have sufficient room for

- MAD-X discussion
- InfoXLM discussion
- moved boundary schedule figure
- moved ablation table
- remaining baseline analyses

Avoid appendix lettering conflicts after inserting new material.

Renumber figures/tables accordingly.

---

# Task 8 — Consistency Review

After all structural edits are complete,

perform a full manuscript consistency review.

Specifically identify every location where the text is now inconsistent.

Examples include

- text saying a baseline is omitted when it now exists
- references to old figure numbers
- references to old table numbers
- incorrect appendix numbers
- sentences claiming only Vietnamese comparisons when tables now include more baselines
- captions inconsistent with table contents
- dangling citations
- duplicated explanations
- outdated wording about "future experiments"

---

## Deliverable

Produce a checklist containing

For every issue:

- Section
- Current text
- Why it is now inconsistent
- Suggested revision

Do NOT automatically rewrite those paragraphs unless necessary for consistency.

The goal is to let the authors review all wording changes before the final polishing pass.

---

# Constraints

- Preserve scientific claims.
- Preserve writing style.
- Do not modify equations.
- Do not alter reported numerical results.
- Only insert "--" where results are pending.
- Keep formatting ACL-compliant.
- Ensure all cross-references compile correctly after moving figures/tables.