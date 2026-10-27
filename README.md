# Development of a Convolutional Neural Network (CNN) for Dog–Cat Image Classification Using Three Convolutional Layers

A complete, self-contained machine learning engineering project implementing an optimized 3-layer Convolutional Neural Network (CNN) with TensorFlow/Keras for binary classification of real Dog and Cat photographs. All code, datasets, models, metrics, and visualization artifacts are managed strictly inside the `CNN/` directory.

---

## 1. Project Overview & Problem Statement
Image classification between dogs and cats is a canonical benchmark in computer vision. Because natural photographs contain complex variations in background, posture, lighting, and textures, traditional Artificial Neural Networks (ANNs) struggle with high-dimensional image inputs ($128 \times 128 \times 3 = 49,152$ values per sample).

Convolutional Neural Networks (CNNs) solve this by using weight sharing and spatial pooling to extract translation-invariant hierarchical features (edges, contours, ear shapes, snouts). This project demonstrates the design, optimization, and empirical evaluation of a strictly constrained 3-convolutional-layer CNN architecture trained on real photographs.

---

## 2. Real Dataset & Split Organization

The project uses the real `dogs_vs_cats_subset` photograph dataset located inside `CNN/dogs_vs_cats_subset/`. No synthetic images are used.

### Dataset Splits (Zero Overlap & No Data Leakage):
- **Training Set (`train/`):** 800 real photographs (400 Cat, 400 Dog)
- **Validation Set (`val/`):** 200 real photographs (100 Cat, 100 Dog) — preserves the dataset's provided validation split
- **Labelled Test Set (`test_labelled/`):** 200 real photographs (100 Cat, 100 Dog) — genuinely unseen test holdout split created from labelled photographs
- **Unlabelled Test Set (`test/`):** 100 unlabelled images (`1.jpg`–`100.jpg`) — preserved separately and excluded from metric calculations
- **Integrity & Independence:** 1,200 unique MD5 hashes, 0 corrupted files, 0 cross-split duplicates or near-duplicates. The 200 test images remained completely untouched during model development.

```text
CNN/
├── dogs_vs_cats_subset/
│   ├── train/
│   │   ├── Cat/             # 400 real cat photographs (cat.0 - cat.399)
│   │   └── Dog/             # 400 real dog photographs (dog.0 - dog.399)
│   ├── val/
│   │   ├── Cat/             # 100 real cat photographs (cat.12000 - cat.12099)
│   │   └── Dog/             # 100 real dog photographs (dog.12000 - dog.12099)
│   ├── test_labelled/
│   │   ├── Cat/             # 100 real cat photographs (cat.400 - cat.499)
│   │   └── Dog/             # 100 real dog photographs (dog.400 - dog.499)
│   └── test/                # 100 unlabelled Kaggle test images (1.jpg - 100.jpg)
├── test_images/             # Real unseen test photographs for single-image prediction
│   ├── test_cat.jpg         # Real cat photograph
│   └── test_dog.jpg         # Real dog photograph
├── results/                 # Model artifacts, plots, and metrics
│   ├── cnn_model.keras      # Improved trained model (76.50% test accuracy)
│   ├── accuracy_plot.png    # Training vs. validation accuracy curve
│   ├── loss_plot.png        # Training vs. validation loss curve
│   ├── confusion_matrix.png # Test confusion matrix heatmap
│   ├── sample_predictions.png # Unseen test samples prediction grid
│   ├── misclassified_images.png # Visual report of remaining errors
│   ├── metrics.txt          # Detailed measured metrics report
│   ├── experiment_comparison.csv # Full experimental log
│   ├── diagnostic_report.md # Phase 1 root cause investigation report
│   └── final_model_report.md# Comprehensive final engineering report
├── main.py                  # Complete end-to-end training and evaluation pipeline
├── predict.py               # Standalone single-image & user-upload prediction tool
├── requirements.txt         # Required Python dependencies
└── README.md                # Project documentation
```

---

## 3. CNN Architecture (Exactly 3 Convolutional Layers)

The model strictly adheres to the required 3-convolutional-layer architecture built with `tf.keras.Sequential`:

| Layer # | Layer Name | Specifications | Output Shape | Parameters | Purpose |
| :---: | :--- | :--- | :---: | :---: | :--- |
| - | **Input** | Shape $(128, 128, 3)$ | $(128, 128, 3)$ | 0 | 3-channel RGB image tensor |
| **1** | **Data Augmentation**| RandomFlip, Rotation(0.08), Zoom(0.1), Trans(0.05) | $(128, 128, 3)$ | 0 | Invariance to pose, scale, tilt (Training only) |
| **2** | **Rescaling** | $1 / 255.0$ | $(128, 128, 3)$ | 0 | Normalizes pixel values $[0.0, 1.0]$ |
| **3** | **Conv2D #1** | 32 filters, $(3, 3)$, ReLU | $(126, 126, 32)$ | 896 | Edge, color, and local texture extraction |
| **4** | **MaxPooling2D**| Pool size $(2, 2)$ | $(63, 63, 32)$ | 0 | Spatial downsampling |
| **5** | **Conv2D #2** | 64 filters, $(3, 3)$, ReLU | $(61, 61, 64)$ | 18,496 | Mid-level contour and feature extraction |
| **6** | **MaxPooling2D**| Pool size $(2, 2)$ | $(30, 30, 64)$ | 0 | Spatial downsampling |
| **7** | **Conv2D #3** | 128 filters, $(3, 3)$, ReLU | $(28, 28, 128)$ | 73,856 | High-level facial and shape extraction |
| **8** | **MaxPooling2D**| Pool size $(2, 2)$ | $(14, 14, 128)$ | 0 | Spatial downsampling |
| **9** | **Flatten** | - | $(25,088)$ | 0 | Preserves spatial layout across $14 \times 14$ grid |
| **10**| **Dense #1** | 128 units, ReLU, $L_2 = 1 \times 10^{-4}$ | $(128)$ | 3,211,392 | Feature combination with weight decay |
| **11**| **Dropout** | Dropout rate = 0.40 | $(128)$ | 0 | Prevents dense layer memorization |
| **12**| **Dense #2 (Output)**| 1 unit, Sigmoid | $(1)$ | 129 | Output probability: $P(Y = \text{Dog})$ |

- **Total Parameters:** 3,304,769 (Trainable: 3,304,769)
- **Compilation:**
  - **Optimizer:** Adam ($\alpha = 0.0005$)
  - **Loss Function:** Binary Cross-Entropy (`binary_crossentropy`)
  - **Evaluation Metric:** Accuracy (`accuracy`)
  - **Callbacks:** `EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)` and `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)`

---

## 4. Controlled Experiments & Key Architectural Insights

The progression of controlled experiments on development data (tracked in [`experiment_comparison.csv`](results/experiment_comparison.csv)):

| Experiment | Architecture / Changes | Best Epoch | Train Acc | Val Acc | Val Loss | Val F1 | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exp 0 (Baseline)** | Flatten + Dense(128), No Aug, No Reg | 6 | 72.37% | 71.00% | 0.5750 | 0.7264 | Severe overfitting (diverged by ep 11 to loss 0.73) |
| **Exp 1** | Baseline + Data Augmentation | 8 | 74.62% | 70.00% | 0.5556 | 0.7000 | Loss improved from 0.5750 to 0.5556 |
| **Exp 2** | Aug + BatchNorm + Dropout(0.5) (Flatten) | 1 | 50.00% | 50.00% | 3.2996 | 0.0000 | BatchNorm failed due to batch variance on small dataset |
| **Exp 3** | GlobalAveragePooling2D + Aug + Dropout | 14 | 62.88% | 59.50% | 0.6606 | 0.6330 | GAP collapsed spatial feature maps in shallow CNN |
| **Exp 4** | Aug + Flatten + Dropout(0.3) + Adam(0.0005) | 13 | 76.50% | 74.50% | 0.5145 | 0.7451 | Slower learning rate stabilized optimization |
| **Exp 5** | Aug + Flatten + Dropout(0.5) + Adam(0.001) | 29 | 80.12% | 74.00% | 0.5137 | 0.7512 | Higher dropout allowed stable training to ep 29 |
| **Exp 6 (Final)** | **Aug + Flatten + Dense(128, L2=1e-4) + Dropout(0.4) + Adam(0.0005) + ReduceLR** | **21** | **81.75%** | **78.50%** | **0.5028** | **0.7778** | **Best generalization; test accuracy reached 76.50%** |

---

## 5. Measured Baseline vs. Final Evaluation Results

Both models were evaluated on the exact same **200 genuinely unseen test photographs**:

| Metric | Original Baseline | Improved Final Model | Improvement |
| :--- | :---: | :---: | :---: |
| **Training Accuracy** | 72.37% (Best) / 91.87% (Ep 11) | **81.75%** | — |
| **Validation Accuracy** | 71.00% | **78.50%** | **+7.50%** |
| **Validation Loss** | 0.5750 | **0.5028** | **-0.0722** |
| **Test Accuracy** | **64.00%** (128 / 200) | **76.50%** (153 / 200) | **+12.50%** |
| **Test Errors** | **72 errors** | **47 errors** | **-25 errors (-34.7%)** |
| **Cat Recall** | **49.00%** (51 errors) | **84.00%** (16 errors) | **+35.00% (+71.4%)** |
| **Dog Recall** | 79.00% (21 errors) | **69.00%** (31 errors) | Balanced recall |
| **Dog Precision** | 60.77% | **81.18%** | **+20.41%** |
| **Macro F1-Score** | 0.6317 | **0.7637** | **+0.1320** |
| **Generalization Gap** | **22.37%** (Severe Overfitting) | **3.25%** | **Overfitting Eliminated** |

### Test Confusion Matrix Comparison ($N = 200$):
```text
Baseline Model (72 Errors):              Improved Model (47 Errors):
                   Pred Cat    Pred Dog                     Pred Cat    Pred Dog
Actual Cat (100)      49          51        Actual Cat (100)      84          16
Actual Dog (100)      21          79        Actual Dog (100)      31          69
```

---

## 6. How to Run the Project (Windows PowerShell)

Ensure you execute commands from inside the `CNN/` folder using **Python 3.12** (where TensorFlow is installed):

```powershell
cd "c:\Users\bg695\Desktop\COE Projects\AICEO\CNN"
```

### Option 1: Run the Complete Training Pipeline
```powershell
& "C:\Users\bg695\AppData\Local\Programs\Python\Python312\python.exe" main.py
```

---

## 7. User-Upload Image Prediction (`predict.py`)

A standalone, modular inference script [`predict.py`](predict.py) supports uploading custom images, selecting files through native Windows Explorer dialogs, or testing sample images.

### Method A: Native File Browser Dialog (Click & Upload)
Pops open the Windows Explorer dialog so you can choose any `.jpg` / `.png`:
```powershell
& "C:\Users\bg695\AppData\Local\Programs\Python\Python312\python.exe" predict.py --upload
```

### Method B: Interactive Terminal Menu
```powershell
& "C:\Users\bg695\AppData\Local\Programs\Python\Python312\python.exe" predict.py
```
Provides an interactive menu:
```text
--------------------------------------------------
         CNN DOG-CAT PREDICTION MENU
--------------------------------------------------
 [1] Upload / Select image file (Opens file dialog)
 [2] Enter or paste an image file path
 [3] Run prediction on sample test images (Cat & Dog)
 [4] Exit
--------------------------------------------------
```

### Method C: Pass Image Path Direct via CLI
```powershell
& "C:\Users\bg695\AppData\Local\Programs\Python\Python312\python.exe" predict.py "path/to/your_image.jpg"
```

### Method D: Display Visual Popup Window (`--show-plot`)
Displays the image alongside the predicted class and confidence:
```powershell
& "C:\Users\bg695\AppData\Local\Programs\Python\Python312\python.exe" predict.py "path/to/your_image.jpg" --show-plot
```

### Sample Output:
```text
==============================================================
           CNN IMAGE CLASSIFICATION PREDICTION REPORT
==============================================================
Image File:          test_cat.jpg
Original Resolution: 500 x 374 (JPEG, mode=RGB)
Model Input Size:    128 x 128 x 3
--------------------------------------------------------------
PREDICTED CLASS:     >>> CAT <<<
CONFIDENCE:          84.83%
--------------------------------------------------------------
Class Probabilities:
  P(Cat):  84.83%  [#####################----]
  P(Dog):  15.17%  [####---------------------]
==============================================================
```
