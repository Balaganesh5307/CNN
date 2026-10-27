"""
Project Title: Development of a Convolutional Neural Network (CNN) for Dog-Cat 
Image Classification Using Three Convolutional Layers.

Description:
A complete, end-to-end implementation of an optimized 3-layer Convolutional
Neural Network built with TensorFlow/Keras for binary image classification
(Dog vs. Cat) using real photographs from the dogs_vs_cats_subset dataset.
Features data augmentation, Dropout regularization, L2 weight decay,
adaptive learning rate scheduling, and early stopping.
All files and results are managed strictly inside the CNN/ directory.
"""

import os
import sys
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf

# Ensure prompt output is flushed immediately to terminal
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Directory Configuration (relative to this script / CNN root)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dogs_vs_cats_subset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_LABELLED_DIR = os.path.join(DATASET_DIR, "test_labelled")
TEST_UNLABELLED_DIR = os.path.join(DATASET_DIR, "test")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
MAX_EPOCHS = 40
RANDOM_SEED = 42


def ensure_directories():
    """Ensure all required project directories exist."""
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def verify_and_prepare_dataset():
    """
    Verify and prepare the real dogs_vs_cats_subset dataset into class-based
    subdirectories (Cat / Dog) for train, val, and test_labelled splits.
    Preserves the existing val split (200 images: 100 Cat, 100 Dog).
    Extracts an unseen labelled test split (200 images: 100 Cat, 100 Dog) from train,
    leaving 800 training images (400 Cat, 400 Dog).
    Leaves the unlabelled test/ directory (100 images) intact.
    Does NOT use synthetic data or fall back to synthetic generation.
    """
    if not os.path.exists(DATASET_DIR):
        raise RuntimeError(
            f"Dataset directory not found: '{DATASET_DIR}'. "
            "Please ensure 'dogs_vs_cats_subset' is located inside the CNN/ directory."
        )

    train_cat_dir = os.path.join(TRAIN_DIR, "Cat")
    train_dog_dir = os.path.join(TRAIN_DIR, "Dog")
    val_cat_dir = os.path.join(VAL_DIR, "Cat")
    val_dog_dir = os.path.join(VAL_DIR, "Dog")
    test_cat_dir = os.path.join(TEST_LABELLED_DIR, "Cat")
    test_dog_dir = os.path.join(TEST_LABELLED_DIR, "Dog")

    for d in [train_cat_dir, train_dog_dir, val_cat_dir, val_dog_dir, test_cat_dir, test_dog_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Organize validation directory if needed
    val_root_files = [f for f in os.listdir(VAL_DIR) if os.path.isfile(os.path.join(VAL_DIR, f))]
    for fname in val_root_files:
        src = os.path.join(VAL_DIR, fname)
        if fname.lower().startswith("cat."):
            shutil.move(src, os.path.join(val_cat_dir, fname))
        elif fname.lower().startswith("dog."):
            shutil.move(src, os.path.join(val_dog_dir, fname))

    # 2. Organize train directory and create test_labelled split if needed
    train_root_files = [f for f in os.listdir(TRAIN_DIR) if os.path.isfile(os.path.join(TRAIN_DIR, f))]
    if train_root_files:
        cat_files = sorted(
            [f for f in train_root_files if f.lower().startswith("cat.")],
            key=lambda x: int(x.split(".")[1]) if len(x.split(".")) > 2 and x.split(".")[1].isdigit() else 0
        )
        dog_files = sorted(
            [f for f in train_root_files if f.lower().startswith("dog.")],
            key=lambda x: int(x.split(".")[1]) if len(x.split(".")) > 2 and x.split(".")[1].isdigit() else 0
        )

        for f in cat_files[:400]:
            shutil.move(os.path.join(TRAIN_DIR, f), os.path.join(train_cat_dir, f))
        for f in cat_files[400:]:
            shutil.move(os.path.join(TRAIN_DIR, f), os.path.join(test_cat_dir, f))

        for f in dog_files[:400]:
            shutil.move(os.path.join(TRAIN_DIR, f), os.path.join(train_dog_dir, f))
        for f in dog_files[400:]:
            shutil.move(os.path.join(TRAIN_DIR, f), os.path.join(test_dog_dir, f))

    n_train_cat = len(os.listdir(train_cat_dir))
    n_train_dog = len(os.listdir(train_dog_dir))
    n_val_cat = len(os.listdir(val_cat_dir))
    n_val_dog = len(os.listdir(val_dog_dir))
    n_test_cat = len(os.listdir(test_cat_dir))
    n_test_dog = len(os.listdir(test_dog_dir))
    n_unlabelled_test = len(os.listdir(TEST_UNLABELLED_DIR)) if os.path.exists(TEST_UNLABELLED_DIR) else 0

    print("\n--- Real Dataset Inspection & Split Summary ---")
    print(f"Training Set:         {n_train_cat + n_train_dog} images ({n_train_cat} Cat, {n_train_dog} Dog)")
    print(f"Validation Set:       {n_val_cat + n_val_dog} images ({n_val_cat} Cat, {n_val_dog} Dog)")
    print(f"Labelled Test Set:    {n_test_cat + n_test_dog} images ({n_test_cat} Cat, {n_test_dog} Dog)")
    print(f"Unlabelled Test Set:  {n_unlabelled_test} images (in {TEST_UNLABELLED_DIR}, kept separate)")
    print(f"Total Labelled:       {n_train_cat + n_train_dog + n_val_cat + n_val_dog + n_test_cat + n_test_dog} real photographs")

    if n_train_cat == 0 or n_train_dog == 0:
        raise RuntimeError("Training set contains zero images. Dataset loading failed.")

    # Image integrity check
    corrupted_count = 0
    for split_dir in [train_cat_dir, train_dog_dir, val_cat_dir, val_dog_dir, test_cat_dir, test_dog_dir]:
        for img_name in os.listdir(split_dir):
            img_path = os.path.join(split_dir, img_name)
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception as e:
                print(f"[Warning] Corrupted image removed: {img_path} ({e})")
                os.remove(img_path)
                corrupted_count += 1
    if corrupted_count == 0:
        print("Image integrity verification: All images checked and 100% valid.")

    # Ensure real test images exist in test_images/
    test_cat_path = os.path.join(TEST_IMAGES_DIR, "test_cat.jpg")
    test_dog_path = os.path.join(TEST_IMAGES_DIR, "test_dog.jpg")
    sample_cat_test = os.path.join(test_cat_dir, os.listdir(test_cat_dir)[0])
    sample_dog_test = os.path.join(test_dog_dir, os.listdir(test_dog_dir)[0])
    if not os.path.exists(test_cat_path):
        shutil.copyfile(sample_cat_test, test_cat_path)
    if not os.path.exists(test_dog_path):
        shutil.copyfile(sample_dog_test, test_dog_path)

    return {
        "train_count": n_train_cat + n_train_dog,
        "val_count": n_val_cat + n_val_dog,
        "test_count": n_test_cat + n_test_dog,
        "unlabelled_count": n_unlabelled_test
    }


def build_improved_cnn_model():
    """
    Build the optimized 3-Layer Convolutional Neural Network:
    - Retains exactly three Conv2D layers (32, 64, 128)
    - Input shape: (128, 128, 3)
    - Training-only data augmentation (horizontal flips, slight rotations, zoom, translation)
    - Rescaling: 1./255
    - Conv2D: 32 filters, (3, 3), ReLU
    - MaxPooling2D: (2, 2)
    - Conv2D: 64 filters, (3, 3), ReLU
    - MaxPooling2D: (2, 2)
    - Conv2D: 128 filters, (3, 3), ReLU
    - MaxPooling2D: (2, 2)
    - Flatten
    - Dense: 128 neurons, ReLU, L2 regularization (1e-4)
    - Dropout: 0.40 (regularizes the 3.2M-parameter dense layer)
    - Dense: 1 neuron, Sigmoid
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05)
    ], name="data_augmentation")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        data_augmentation,
        tf.keras.layers.Rescaling(1.0 / 255.0),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv2d_1"),
        tf.keras.layers.MaxPooling2D((2, 2), name="maxpool_1"),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_2"),
        tf.keras.layers.MaxPooling2D((2, 2), name="maxpool_2"),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", name="conv2d_3"),
        tf.keras.layers.MaxPooling2D((2, 2), name="maxpool_3"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(
            128, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="dense_128"
        ),
        tf.keras.layers.Dropout(0.4, name="dropout"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output_dense")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_and_save_training_curves(history):
    """Plot and save training and validation accuracy and loss curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    # 1. Accuracy Plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#2ca02c", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#1f77b4", linewidth=2, linestyle="--")
    plt.title("Improved CNN: Training vs. Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    acc_path = os.path.join(RESULTS_DIR, "accuracy_plot.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Accuracy plot saved: {acc_path}")

    # 2. Loss Plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, loss, label="Training Loss", color="#d62728", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    plt.title("Improved CNN: Training vs. Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    loss_path = os.path.join(RESULTS_DIR, "loss_plot.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Loss plot saved: {loss_path}")


def plot_and_save_confusion_matrix(cm, class_names):
    """Plot and save confusion matrix heatmap for test set evaluation."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    acc_val = (cm[0, 0] + cm[1, 1]) / cm.sum()
    plt.title(f"Improved CNN Test Confusion Matrix (Acc: {acc_val*100:.1f}%)", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14, fontweight="bold")

    plt.ylabel("True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved: {cm_path}")


def plot_and_save_sample_predictions(test_ds, model, class_names):
    """Plot and save sample test predictions in an 8-image grid."""
    for img_batch, lbl_batch in test_ds.take(1):
        sample_images = img_batch.numpy()[:8]
        sample_labels = lbl_batch.numpy()[:8]
        break

    sample_preds = model.predict(sample_images, verbose=0).flatten()
    plt.figure(figsize=(14, 7))
    for i in range(len(sample_images)):
        plt.subplot(2, 4, i + 1)
        img = sample_images[i]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        actual_name = class_names[int(sample_labels[i])]
        pred_idx = 1 if sample_preds[i] >= 0.5 else 0
        pred_name = class_names[pred_idx]
        conf = sample_preds[i] if pred_idx == 1 else (1.0 - sample_preds[i])

        color = "green" if actual_name == pred_name else "red"
        plt.imshow(img)
        plt.title(f"True: {actual_name}\nPred: {pred_name} ({conf*100:.1f}%)", color=color, fontsize=10, fontweight="bold")
        plt.axis("off")

    plt.suptitle("Improved CNN: Test Sample Predictions (Green = Correct, Red = Incorrect)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    samples_path = os.path.join(RESULTS_DIR, "sample_predictions.png")
    plt.savefig(samples_path, dpi=300)
    plt.close()
    print(f"Sample predictions plot saved: {samples_path}")


def predict_single_image(image_path, model, class_names):
    """
    Predict whether a single image contains a Dog or a Cat.
    Displays and reports the predicted class and confidence score.
    """
    print(f"\nTarget Image: {image_path}")

    if not os.path.exists(image_path):
        print(f"[Error] Image file not found: {image_path}")
        return None

    try:
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize(IMAGE_SIZE)
        img_array = np.array(img_resized)
        input_tensor = np.expand_dims(img_array, axis=0)

        raw_pred = float(model.predict(input_tensor, verbose=0)[0][0])
        predicted_idx = 1 if raw_pred >= 0.5 else 0
        predicted_label = class_names[predicted_idx]
        confidence = raw_pred if predicted_idx == 1 else (1.0 - raw_pred)

        print(f"  Raw Sigmoid Output (P[Dog]): {raw_pred:.4f}")
        print(f"  Predicted Class:             {predicted_label}")
        print(f"  Confidence:                  {confidence * 100:.2f}%")

        return {
            "path": image_path,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "raw_output": raw_pred
        }
    except Exception as e:
        print(f"[Error] Failed to process image {image_path}: {e}")
        return None


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("=" * 70)
    print("  Development of a Convolutional Neural Network (CNN) for Dog-Cat")
    print("  Image Classification Using Three Convolutional Layers (Optimized)")
    print("=" * 70)

    # Step 1: Ensure directory structure
    ensure_directories()

    # Step 2: Prepare and verify real dataset
    split_info = verify_and_prepare_dataset()

    # Step 3: Load datasets via tf.keras.utils.image_dataset_from_directory
    print(f"\n[Step 2] Loading train, validation, and test datasets...")
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED
    )

    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    raw_test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=TEST_LABELLED_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = raw_train_ds.class_names
    test_file_paths = list(raw_test_ds.file_paths)
    print(f"Class names detected: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = raw_train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = raw_val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = raw_test_ds.prefetch(buffer_size=AUTOTUNE)

    # Step 4: Build Improved 3-Layer CNN Architecture
    print("\n[Step 3] Building Improved 3-Layer CNN Model...")
    model = build_improved_cnn_model()
    model.summary()

    # Step 5: Train Model with EarlyStopping and ReduceLROnPlateau
    print(f"\n[Step 4] Training CNN (Max Epochs: {MAX_EPOCHS}, Batch Size: {BATCH_SIZE})...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - t0
    print(f"Training completed in {training_time:.2f} seconds.")

    # Save trained model
    model_save_path = os.path.join(RESULTS_DIR, "cnn_model.keras")
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Step 6: Threshold Calibration on Validation Set
    print("\n[Step 5] Calibrating Decision Threshold on Validation Set Only...")
    y_val_true, y_val_probs = [], []
    for imgs, lbls in val_ds:
        preds = model.predict(imgs, verbose=0)
        y_val_true.extend(lbls.numpy())
        y_val_probs.extend(preds.flatten())
    y_val_true = np.array(y_val_true)
    y_val_probs = np.array(y_val_probs)

    best_threshold = 0.50
    best_val_acc = 0.0
    for t in np.linspace(0.40, 0.60, 21):
        acc_t = accuracy_score(y_val_true, (y_val_probs >= t).astype(int))
        if acc_t > best_val_acc:
            best_val_acc = acc_t
            best_threshold = float(t)
    print(f"Optimal Validation Threshold: {best_threshold:.2f} (Val Acc: {best_val_acc*100:.2f}%)")

    # Step 7: One-Time Evaluation on Genuinely Unseen Test Set
    print("\n[Step 6] One-Time Final Evaluation on Untouched Test Set (N=200)...")
    train_loss, train_accuracy = model.evaluate(train_ds, verbose=0)
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
    test_loss, test_accuracy_default = model.evaluate(test_ds, verbose=0)

    y_test_true = []
    y_test_probs = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_test_true.extend(labels.numpy())
        y_test_probs.extend(preds.flatten())

    y_test_true = np.array(y_test_true)
    y_test_probs = np.array(y_test_probs)

    # Standard threshold 0.50 evaluation
    y_test_pred_def = (y_test_probs >= 0.50).astype(int)
    cm_def = confusion_matrix(y_test_true, y_test_pred_def)
    prec_def = precision_score(y_test_true, y_test_pred_def, zero_division=0)
    rec_def = recall_score(y_test_true, y_test_pred_def, zero_division=0)
    f1_def = f1_score(y_test_true, y_test_pred_def, zero_division=0)
    report_def = classification_report(y_test_true, y_test_pred_def, target_names=class_names, digits=4)

    # Calibrated threshold evaluation
    y_test_pred_cal = (y_test_probs >= best_threshold).astype(int)
    cm_cal = confusion_matrix(y_test_true, y_test_pred_cal)
    acc_cal = accuracy_score(y_test_true, y_test_pred_cal)
    prec_cal = precision_score(y_test_true, y_test_pred_cal, zero_division=0)
    rec_cal = recall_score(y_test_true, y_test_pred_cal, zero_division=0)
    f1_cal = f1_score(y_test_true, y_test_pred_cal, zero_division=0)

    print("\n" + "=" * 55)
    print("           MEASURED EVALUATION RESULTS")
    print("=" * 55)
    print(f"Training Accuracy:   {train_accuracy * 100:.2f}%  (Loss: {train_loss:.4f})")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%  (Loss: {val_loss:.4f})")
    print(f"\nStandard Threshold (0.50):")
    print(f"  Test Accuracy:     {test_accuracy_default * 100:.2f}% ({int(test_accuracy_default*200)} / 200 correct, {200-int(test_accuracy_default*200)} errors)")
    print(f"  Test Loss:         {test_loss:.4f}")
    print(f"  Test Precision:    {prec_def * 100:.2f}%")
    print(f"  Test Recall:       {rec_def * 100:.2f}%")
    print(f"  Test F1-Score:     {f1_def:.4f}")
    print("  Test Confusion Matrix:")
    print(cm_def)
    print("\n  Test Classification Report:")
    print(report_def)

    print(f"\nCalibrated Threshold ({best_threshold:.2f}):")
    print(f"  Test Accuracy:     {acc_cal * 100:.2f}% ({int(acc_cal*200)} / 200 correct, {200-int(acc_cal*200)} errors)")
    print(f"  Test Precision:    {prec_cal * 100:.2f}%")
    print(f"  Test Recall:       {rec_cal * 100:.2f}%")
    print(f"  Test F1-Score:     {f1_cal:.4f}")

    # Step 8: Save metrics.txt
    metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("CNN DOG-CAT CLASSIFICATION METRICS REPORT (IMPROVED MODEL)\n")
        f.write("=" * 60 + "\n")
        f.write("Project Title: Development of a Convolutional Neural Network (CNN)\n")
        f.write("for Dog-Cat Image Classification Using Three Convolutional Layers\n\n")
        f.write("Dataset Distribution:\n")
        f.write(f"  Training Set:         {split_info['train_count']} images (400 Cat, 400 Dog)\n")
        f.write(f"  Validation Set:       {split_info['val_count']} images (100 Cat, 100 Dog)\n")
        f.write(f"  Labelled Test Set:    {split_info['test_count']} images (100 Cat, 100 Dog)\n")
        f.write(f"  Unlabelled Test Set:  {split_info['unlabelled_count']} images (dogs_vs_cats_subset/test/)\n\n")
        f.write(f"Training Duration:      {training_time:.2f} seconds\n")
        f.write(f"Model Parameters:       {model.count_params():,}\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"  Training Accuracy:    {train_accuracy * 100:.2f}%\n")
        f.write(f"  Training Loss:        {train_loss:.4f}\n")
        f.write(f"  Validation Accuracy:  {val_accuracy * 100:.2f}%\n")
        f.write(f"  Validation Loss:      {val_loss:.4f}\n")
        f.write(f"  Optimal Val Threshold:{best_threshold:.2f}\n\n")
        f.write("Test Evaluation (Standard Threshold 0.50):\n")
        f.write(f"  Test Accuracy:        {test_accuracy_default * 100:.2f}% ({int(test_accuracy_default*200)} / 200 correct, {200-int(test_accuracy_default*200)} errors)\n")
        f.write(f"  Test Loss:            {test_loss:.4f}\n")
        f.write(f"  Test Precision:       {prec_def * 100:.2f}%\n")
        f.write(f"  Test Recall:          {rec_def * 100:.2f}%\n")
        f.write(f"  Test F1-Score:        {f1_def:.4f}\n\n")
        f.write("Test Confusion Matrix (Threshold 0.50):\n")
        f.write(str(cm_def) + "\n\n")
        f.write("Test Evaluation (Calibrated Threshold):\n")
        f.write(f"  Test Accuracy:        {acc_cal * 100:.2f}% ({int(acc_cal*200)} / 200 correct, {200-int(acc_cal*200)} errors)\n")
        f.write(f"  Test Precision:       {prec_cal * 100:.2f}%\n")
        f.write(f"  Test Recall:          {rec_cal * 100:.2f}%\n")
        f.write(f"  Test F1-Score:        {f1_cal:.4f}\n\n")
        f.write("Classification Report (Threshold 0.50):\n")
        f.write(report_def + "\n")
    print(f"Metrics report written to: {metrics_path}")

    # Step 9: Generate and save all plots
    print("\n[Step 7] Generating Diagnostic Graphs in CNN/results/...")
    plot_and_save_training_curves(history)
    plot_and_save_confusion_matrix(cm_def, class_names)
    plot_and_save_sample_predictions(test_ds, model, class_names)

    # Generate misclassified images visual report
    mis_indices = np.where(y_test_pred_def != y_test_true)[0]
    n_mis = len(mis_indices)
    if n_mis > 0:
        cols = 8
        rows = int(np.ceil(n_mis / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.2), dpi=120)
        fig.suptitle(f"Improved CNN: Misclassified Test Images ({n_mis} / 200 Total Errors)", fontsize=15, fontweight="bold", y=0.995)
        for idx, ax in enumerate(np.array(axes).flat):
            if idx < n_mis:
                img_idx = mis_indices[idx]
                fp = test_file_paths[img_idx]
                fn = os.path.basename(fp)
                act_n = class_names[int(y_test_true[img_idx])]
                pr_n = class_names[int(y_test_pred_def[img_idx])]
                p_val = y_test_probs[img_idx]
                cf_val = (p_val if pr_n == "Dog" else (1.0 - p_val)) * 100
                with Image.open(fp) as im_raw:
                    im_thumb = im_raw.resize(IMAGE_SIZE)
                ax.imshow(im_thumb)
                ax.set_title(f"True: {act_n} | Pred: {pr_n}\nP[Dog]={p_val:.2f} ({cf_val:.1f}%)\n{fn}", fontsize=7.5, fontweight="bold", color="firebrick")
                ax.axis("off")
            else:
                ax.axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        mis_path = os.path.join(RESULTS_DIR, "misclassified_images.png")
        plt.savefig(mis_path, bbox_inches="tight")
        plt.close()
        print(f"Misclassified images visual report saved to: {mis_path}")

    # Step 10: Single-image prediction demonstration on test_images/
    print("\n[Step 8] Demonstrating Single-Image Predictions on Unseen Test Images...")
    test_files = sorted([
        os.path.join(TEST_IMAGES_DIR, f)
        for f in os.listdir(TEST_IMAGES_DIR)
        if os.path.splitext(f.lower())[1] in {'.jpg', '.jpeg', '.png', '.bmp'}
    ])
    for tf_path in test_files:
        predict_single_image(tf_path, model, class_names)

    print("\n" + "=" * 70)
    print("  CNN Pipeline Finished Successfully!")
    print(f"  All outputs saved in: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
