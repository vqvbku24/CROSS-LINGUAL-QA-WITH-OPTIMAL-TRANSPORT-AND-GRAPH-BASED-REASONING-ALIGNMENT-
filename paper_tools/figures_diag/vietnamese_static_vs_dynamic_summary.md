# Vietnamese (Vi) – Static vs Dynamic Summary (Corrected)

**Source**: Clearer tables provided (Static top, Dynamic bottom).  
**Final column used**:
- Seed 42: `epoch 6`
- Seed 43 & 44: `Epoch best after`

---

## 1. Final / Best scores

### Static

| Dataset     | Metric | Seed 42 | Seed 43 | Seed 44 | Mean   |
|-------------|--------|---------|---------|---------|--------|
| XQuAD-hi    | EM     | 50.50   | 49.66   | 50.08   | **50.08** |
| XQuAD-hi    | F1     | 70.38   | 69.37   | 69.89   | **69.88** |
| XQuAD-en    | EM     | 60.59   | 61.76   | 62.02   | **61.46** |
| XQuAD-en    | F1     | 72.37   | 73.12   | 73.74   | **73.08** |
| SQuAD 2.0   | EM     | 66.32   | 66.35   | 66.04   | **66.24** |
| SQuAD 2.0   | F1     | 74.37   | 74.39   | 74.22   | **74.33** |
| MLQA-hi     | EM     | 45.01   | 45.15   | 45.08   | **45.08** |
| MLQA-hi     | F1     | 66.75   | 66.76   | 66.67   | **66.73** |
| MLQA-en     | EM     | 65.80   | 65.80   | 65.42   | **65.67** |
| MLQA-en     | F1     | 79.45   | 79.50   | 79.17   | **79.37** |

### Dynamic

| Dataset     | Metric | Seed 42 | Seed 43 | Seed 44 | Mean   |
|-------------|--------|---------|---------|---------|--------|
| XQuAD-hi    | EM     | 47.98   | 48.32   | 48.57   | **48.29** |
| XQuAD-hi    | F1     | 68.00   | 68.45   | 68.57   | **68.34** |
| XQuAD-en    | EM     | 58.82   | 59.50   | 59.24   | **59.19** |
| XQuAD-en    | F1     | 69.92   | 70.40   | 70.60   | **70.31** |
| SQuAD 2.0   | EM     | 66.53   | 66.70   | 66.62   | **66.62** |
| SQuAD 2.0   | F1     | 74.22   | 74.40   | 74.37   | **74.33** |
| MLQA-hi     | EM     | 44.32   | 44.37   | 44.50   | **44.40** |
| MLQA-hi     | F1     | 66.40   | 66.57   | 66.47   | **66.48** |
| MLQA-en     | EM     | 65.70   | 65.70   | 65.73   | **65.71** |
| MLQA-en     | F1     | 79.47   | 79.44   | 79.54   | **79.48** |

---

## 2. Paired difference (Static − Dynamic) per seed

| Dataset     | Metric | Seed 42 | Seed 43 | Seed 44 | Mean Δ | 
|-------------|--------|---------|---------|---------|--------|
| XQuAD-hi    | EM     | +2.52   | +1.34   | +1.51   | **+1.79** |
| XQuAD-hi    | F1     | +2.38   | +0.92   | +1.32   | **+1.54** |
| XQuAD-en    | EM     | +1.77   | +2.26   | +2.78   | **+2.27** |
| XQuAD-en    | F1     | +2.45   | +2.72   | +3.14   | **+2.77** |
| SQuAD 2.0   | EM     | −0.21   | −0.35   | −0.58   | **−0.38** |
| SQuAD 2.0   | F1     | +0.15   | −0.01   | −0.15   | **−0.00** |
| MLQA-hi     | EM     | +0.69   | +0.78   | +0.58   | **+0.68** |
| MLQA-hi     | F1     | +0.35   | +0.19   | +0.20   | **+0.25** |
| MLQA-en     | EM     | +0.10   | +0.10   | −0.31   | **−0.04** |
| MLQA-en     | F1     | −0.02   | +0.06   | −0.37   | **−0.11** |

---

## 3. Full epoch progression

### Static – Seed 42 (epochs 1 / 2 / 3 / 4)

| Dataset   | Metric | Ep1   | Ep2   | Ep3   | Ep4   |
|-----------|--------|-------|-------|-------|-------|
| XQuAD-hi  | EM     | 50.34 | 50.08 | 49.70 | 50.50 |
| XQuAD-hi  | F1     | 70.47 | 70.28 | 69.73 | 70.38 |
| XQuAD-en  | EM     | 65.04 | 62.86 | 62.44 | 60.59 |
| XQuAD-en  | F1     | 78.41 | 75.63 | 74.78 | 72.37 |
| SQuAD2.0  | EM     | 61.86 | 64.95 | 65.69 | 66.32 |
| SQuAD2.0  | F1     | 70.80 | 73.48 | 74.02 | 74.37 |
| MLQA-hi   | EM     | 44.80 | 45.23 | 46.00 | 45.01 |
| MLQA-hi   | F1     | 66.52 | 67.00 | 67.60 | 66.75 |
| MLQA-en   | EM     | 66.03 | 66.13 | 65.90 | 65.80 |
| MLQA-en   | F1     | 79.70 | 79.60 | 79.50 | 79.45 |

### Static – Seed 43 (Epoch 1 / 2 / 3 / 4)

| Dataset   | Metric | Ep1   | Ep2   | Ep3   | Ep4 |
|-----------|--------|-------|-------|-------|-------|
| XQuAD-hi  | EM     | 50.17 | 50.17 | 50.08 | 49.66 |
| XQuAD-hi  | F1     | 70.04 | 70.38 | 70.39 | 69.37 |
| XQuAD-en  | EM     | 65.04 | 63.00 | 61.85 | 61.76 |
| XQuAD-en  | F1     | 78.39 | 75.70 | 74.21 | 73.12 |
| SQuAD2.0  | EM     | 62.04 | 65.01 | 65.88 | 66.35 |
| SQuAD2.0  | F1     | 71.00 | 73.47 | 74.25 | 74.39 |
| MLQA-hi   | EM     | 45.00 | 45.23 | 45.78 | 45.15 |
| MLQA-hi   | F1     | 66.59 | 66.92 | 67.50 | 66.76 |
| MLQA-en   | EM     | 66.01 | 66.05 | 66.00 | 65.80 |
| MLQA-en   | F1     | 79.52 | 79.54 | 79.60 | 79.50 |

### Static – Seed 44 (Epoch 1 / 2 / 3 / 4)

| Dataset   | Metric | Ep1   | Ep2   | Ep3   | Ep4  |
|-----------|--------|-------|-------|-------|-------|
| XQuAD-hi  | EM     | 50.42 | 50.50 | 50.17 | 50.08 |
| XQuAD-hi  | F1     | 70.35 | 70.36 | 70.39 | 69.89 |
| XQuAD-en  | EM     | 65.12 | 64.53 | 63.03 | 62.02 |
| XQuAD-en  | F1     | 78.69 | 76.05 | 75.70 | 73.74 |
| SQuAD2.0  | EM     | 61.79 | 64.72 | 65.42 | 66.04 |
| SQuAD2.0  | F1     | 70.81 | 73.31 | 73.90 | 74.22 |
| MLQA-hi   | EM     | 45.06 | 45.20 | 45.60 | 45.08 |
| MLQA-hi   | F1     | 66.77 | 66.83 | 67.21 | 66.67 |
| MLQA-en   | EM     | 65.82 | 65.88 | 65.92 | 65.42 |
| MLQA-en   | F1     | 79.41 | 79.45 | 79.50 | 79.17 |

### Dynamic – Seed 42 (epochs 1 / 2 / 3 / 4)

| Dataset   | Metric | Ep1   | Ep2   | Ep3   | Ep4  |
|-----------|--------|-------|-------|-------|-------|
| XQuAD-hi  | EM     | 50.34 | 49.08 | 49.50 | 47.98 |
| XQuAD-hi  | F1     | 70.47 | 69.27 | 69.28 | 68.00 |
| XQuAD-en  | EM     | 65.04 | 63.19 | 61.60 | 58.82 |
| XQuAD-en  | F1     | 78.41 | 75.40 | 73.47 | 69.92 |
| SQuAD2.0  | EM     | 61.86 | 65.32 | 66.21 | 66.53 |
| SQuAD2.0  | F1     | 70.80 | 73.76 | 74.37 | 74.22 |
| MLQA-hi   | EM     | 44.80 | 45.01 | 45.23 | 44.32 |
| MLQA-hi   | F1     | 66.52 | 66.88 | 67.33 | 66.40 |
| MLQA-en   | EM     | 66.03 | 65.90 | 65.70 | 65.70 |
| MLQA-en   | F1     | 79.70 | 79.50 | 79.50 | 79.47 |

### Dynamic – Seed 43 (Epoch 1 / 2 / 3 /4)

| Dataset   | Metric | Ep1   | Ep2   | Ep3   | Ep4 |
|-----------|--------|-------|-------|-------|-------|
| XQuAD-hi  | EM     | 50.17 | 49.24 | 49.24 | 48.32 |
| XQuAD-hi  | F1     | 70.04 | 69.51 | 69.37 | 68.45 |
| XQuAD-en  | EM     | 65.04 | 63.20 | 61.68 | 59.50 |
| XQuAD-en  | F1     | 78.39 | 75.71 | 73.60 | 70.40 |
| SQuAD2.0  | EM     | 62.04 | 65.12 | 66.06 | 66.70 |
| SQuAD2.0  | F1     | 71.00 | 73.60 | 74.26 | 74.40 |
| MLQA-hi   | EM     | 45.00 | 45.26 | 45.46 | 44.37 |
| MLQA-hi   | F1     | 66.59 | 67.00 | 67.40 | 66.57 |
| MLQA-en   | EM     | 66.01 | 65.99 | 65.80 | 65.70 |
| MLQA-en   | F1     | 79.52 | 79.51 | 79.60 | 79.44 |

### Dynamic – Seed 44 (Epoch 1 / 2 / 3 / 4)

| Dataset   | Metric | Ep1   | Ep2   | Ep3   | Ep4  |
|-----------|--------|-------|-------|-------|-------|
| XQuAD-hi  | EM     | 50.42 | 49.08 | 49.16 | 48.57 |
| XQuAD-hi  | F1     | 70.35 | 69.41 | 69.29 | 68.57 |
| XQuAD-en  | EM     | 65.12 | 63.11 | 61.18 | 59.24 |
| XQuAD-en  | F1     | 78.69 | 75.44 | 73.29 | 70.60 |
| SQuAD2.0  | EM     | 61.79 | 65.26 | 66.09 | 66.62 |
| SQuAD2.0  | F1     | 70.81 | 73.70 | 74.30 | 74.37 |
| MLQA-hi   | EM     | 45.06 | 44.90 | 45.35 | 44.50 |
| MLQA-hi   | F1     | 66.77 | 66.66 | 67.44 | 66.47 |
| MLQA-en   | EM     | 65.82 | 65.83 | 65.84 | 65.73 |
| MLQA-en   | F1     | 79.41 | 79.41 | 79.70 | 79.54 |

---

## 4. Quick observation

- **XQuAD-hi / XQuAD-en / MLQA-hi**: Static consistently higher (especially English transfer).
- **SQuAD 2.0 / MLQA-en**: Essentially tied (differences < 0.4 point).
- The gap is largest on the English-side metrics (XQuAD-en), suggesting Static preserves transfer performance better at the selected checkpoint.

*All numbers transcribed directly from the clearer tables you provided.*
