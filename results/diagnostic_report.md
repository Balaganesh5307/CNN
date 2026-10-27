# Phase 1 Deep Diagnostic Report: Error Patterns, Training Dynamics & Root Causes

**Project:** Development of a Convolutional Neural Network (CNN) for Dog-Cat Image Classification Using Three Convolutional Layers  
**Evaluated Model:** `CNN/results/cnn_model.keras` (Epoch 6 restored baseline weights)  
**Evaluated Dataset:** `CNN/dogs_vs_cats_subset/test_labelled/` (200 genuinely unseen photographs: 100 Cat, 100 Dog)  
**Visual Artifact:** `CNN/results/misclassified_images.png` (Visual grid of all 72 misclassified images with true/predicted labels, $P[\text{Dog}]$, confidence, and dimensions)

---

## 1. Quantitative Breakdown of Misclassifications

Out of 200 unseen test photographs:
- **Total Correct:** 128 / 200 (Accuracy: **64.00%**)
- **Total Misclassified:** 72 / 200 (Error Rate: **36.00%**)
  - **Cats Misclassified as Dogs (False Positives):** **51 / 100** (Cat Recall = 49.0%)
  - **Dogs Misclassified as Cats (False Negatives):** **21 / 100** (Dog Recall = 79.0%)

### Confidence & Probability Distribution of Errors:
| Error Category | High Confidence ($P \ge 0.80$ or $P \le 0.20$) | Moderate Confidence ($0.60 \le P < 0.80$ or $0.20 < P \le 0.40$) | Borderline / Low Confidence ($0.40 \le P \le 0.60$) |
| :--- | :---: | :---: | :---: |
| **Cats as Dogs (51 total)** | 7 (13.7%) | 23 (45.1%) | 21 (41.2%) |
| **Dogs as Cats (21 total)** | 2 (9.5%) | 7 (33.3%) | 12 (57.1%) |
| **Combined (72 errors)** | **9 (12.5%)** | **30 (41.7%)** | **33 (45.8%)** |

**Key Finding:** Almost half (45.8%) of all errors are **borderline errors** with predicted probability $P[\text{Dog}]$ lingering between 0.40 and 0.60. The model is severely uncertain because its internal feature separation between feline and canine classes is fragile.

---

## 2. Visual Error Patterns Identified in `misclassified_images.png`

Inspection of the 72 misclassified images revealed several distinct visual clusters:

1. **Severe Feline False Positives (Cats with high $P[\text{Dog}] \ge 0.80$):**
   - `cat.438.jpg` ($P=0.9499$), `cat.423.jpg` ($P=0.9446$), `cat.477.jpg` ($P=0.9140$), `cat.479.jpg` ($P=0.8742$):
     - **Visual Characteristics:** These cats feature complex, multi-textured, outdoor or cluttered domestic backgrounds (e.g., foliage, carpets, furniture). The baseline CNN's unregularized dense layer learned to associate busy spatial textures with dogs.
     - **Squashed Aspect Ratios:** Several high-confidence errors have widescreen dimensions (e.g. $499 \times 346$, Aspect Ratio = 1.44). Resizing directly to $128 \times 128$ compresses wide faces horizontally, distorting pointed cat ears into rounder, dog-like ear profiles.

2. **Canine False Negatives (Dogs with low $P[\text{Dog}] \le 0.20$):**
   - `dog.453.jpg` ($P=0.0895$) and `dog.414.jpg` ($P=0.1200$):
     - **Visual Characteristics:** Small puppies, long-haired/fluffy breeds, or extreme close-up headshots where ears are cropped or lying flat, mimicking feline facial proportions.
     - **Low Lighting:** `dog.414.jpg` is extremely dark (mean brightness = 68.8 vs dataset average 119.4).

3. **Absence of Data Augmentation Impact:**
   - Non-upright postures (sleeping curled up, lying on back) and off-center animals are disproportionately present among the 33 borderline errors. Because training images were only fed in fixed horizontal orientations, the model cannot recognize pets at angled poses.

4. **Ground-Truth Label Validity:**
   - All 200 filenames and visual subjects were audited; **zero mislabeled images** exist in the test split. Every error represents genuine model misclassification.

---

## 3. Verified Epoch-by-Epoch Training & Validation Behavior

The complete training trajectory from execution logs:

| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss | Observation |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 50.13% | 0.7925 | 50.00% | 0.6919 | Random baseline |
| **2** | 50.25% | 0.6914 | 50.50% | 0.6933 | Initial weight adjustment |
| **3** | 53.75% | 0.6904 | 53.50% | 0.6805 | Feature learning starts |
| **4** | 60.87% | 0.6529 | 64.50% | 0.6311 | Rapid feature extraction |
| **5** | 66.12% | 0.5993 | 63.00% | 0.6659 | Temporary plateau |
| **6** | **72.37%** | **0.5275** | **71.00%** | **0.5750** | **OPTIMAL CHECKPOINT (Best Val Loss)** |
| **7** | 78.25% | 0.4619 | 63.00% | 0.7071 | Overfitting begins: val_loss spikes |
| **8** | 82.13% | 0.3994 | 68.50% | 0.6100 | Generalization gap widens |
| **9** | 83.75% | 0.3380 | 68.00% | 0.6765 | Severe divergence |
| **10** | 88.13% | 0.2805 | 71.00% | 0.6565 | Memorization progressing |
| **11** | **91.87%** | **0.1982** | **69.50%** | **0.7296** | **EarlyStopping triggered (patience=5)** |

### Verdict on Underfitting vs. Overfitting:
- **Underfitting:** **REJECTED.** The network readily reached 91.87% training accuracy in 11 epochs; feature capacity in the 3 Conv layers + Dense layer is more than sufficient.
- **Overfitting:** **VERIFIED AND CONFIRMED.** Starting immediately after Epoch 6, training accuracy surged from 72.37% to 91.87% (+19.5%) while validation loss escalated from 0.5750 to 0.7296 (+26.9%). When evaluated on the unseen test set, the model degraded to 64.00%.

---

## 4. Root Cause Ranking

1. **Rank 1: Severe Capacity Bottleneck & Unregularized Dense Layer (97.17% of parameters)**
   - Out of 3,304,769 total parameters, **3,211,392 belong to `Dense(128)`**.
   - With 800 training images, an unregularized dense layer with 3.2M parameters easily memorizes spatial correlations instead of invariant features.
2. **Rank 2: Zero Data Augmentation on a Small Dataset (N=800)**
   - Seeing 800 static images leads to rapid memorization by Epoch 6.
   - Varied poses, tilts, and zoom levels in the test set trigger high uncertainty (45.8% borderline errors).
3. **Rank 3: Unnormalized Intermediate Activations Across Conv Layers**
   - Without Batch Normalization, internal covariate shift across layers causes noisy gradient updates to propagate directly into the 3.2M-weight dense layer.
4. **Rank 4: Aggressive Fixed Learning Rate ($\alpha = 0.001$)**
   - Causes rapid overshoot after Epoch 6, preventing delicate convergence on subtle feline/canine facial features.
