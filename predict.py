"""
Single-Image & User-Upload Prediction Pipeline for Dog-Cat CNN Classification.

Features:
- CLI argument inference (python predict.py path/to/image.jpg)
- Interactive user upload via native Windows File Dialog (--upload or interactive menu)
- Detailed prediction report with class probabilities & confidence bars
- Type annotations, clean modular structure, and robust exception handling
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

# Suppress verbose TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH: str = os.path.join(BASE_DIR, "results", "cnn_model.keras")
DEFAULT_TEST_DIR: str = os.path.join(BASE_DIR, "test_images")
IMAGE_SIZE: Tuple[int, int] = (128, 128)
CLASS_NAMES: Tuple[str, str] = ("Cat", "Dog")
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_cnn_model(model_path: str = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """
    Load the trained Keras CNN model from disk.

    Args:
        model_path: Absolute or relative path to .keras model file.

    Returns:
        Loaded tf.keras.Model instance.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: '{model_path}'.\n"
            "Please ensure the model has been trained and saved under CNN/results/cnn_model.keras."
        )
    print(f"Loading trained CNN model from: {model_path}...")
    model = tf.keras.models.load_model(model_path)
    return model


def open_file_dialog() -> Optional[str]:
    """
    Open a native OS file dialog for the user to upload / select an image file.

    Returns:
        Selected file path string, or None if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()  # Hide root window
        root.attributes("-topmost", True)  # Bring dialog to front

        file_path = filedialog.askopenfilename(
            title="Upload / Select Dog or Cat Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as exc:
        print(f"[Warning] Could not launch graphical file dialog: {exc}")
        return None


def predict_image(
    image_path: str,
    model: tf.keras.Model,
    display_plot: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Classify whether an image is a Cat or a Dog with confidence metrics.

    Args:
        image_path: Path to image file on disk.
        model: Loaded Keras CNN model.
        display_plot: Whether to display an image window with prediction overlay.

    Returns:
        Dictionary containing prediction details, or None on error.
    """
    # Clean quotes that might be pasted from Windows paths
    image_path = image_path.strip("\"'")

    if not os.path.exists(image_path):
        print(f"\n[Error] File not found: '{image_path}'")
        return None

    try:
        with Image.open(image_path) as raw_img:
            orig_width, orig_height = raw_img.size
            img_format = raw_img.format or "UNKNOWN"
            img_mode = raw_img.mode
            img_rgb = raw_img.convert("RGB")
            img_resized = img_rgb.resize(IMAGE_SIZE)

        # Shape: (1, 128, 128, 3) with pixel values [0, 255]
        img_array = np.array(img_resized)
        input_tensor = np.expand_dims(img_array, axis=0)

        # Forward inference
        raw_pred = float(model.predict(input_tensor, verbose=0)[0][0])
        pred_idx = 1 if raw_pred >= 0.5 else 0
        predicted_class = CLASS_NAMES[pred_idx]
        confidence = raw_pred if pred_idx == 1 else (1.0 - raw_pred)

        p_cat = (1.0 - raw_pred) * 100.0
        p_dog = raw_pred * 100.0

        # ASCII confidence bar visualization
        cat_bar_len = int(round(p_cat / 4.0))
        dog_bar_len = int(round(p_dog / 4.0))
        cat_bar = "#" * cat_bar_len + "-" * (25 - cat_bar_len)
        dog_bar = "#" * dog_bar_len + "-" * (25 - dog_bar_len)

        print("\n" + "=" * 62)
        print("           CNN IMAGE CLASSIFICATION PREDICTION REPORT")
        print("=" * 62)
        print(f"Image File:          {os.path.basename(image_path)}")
        print(f"Full Path:           {os.path.abspath(image_path)}")
        print(f"Original Resolution: {orig_width} x {orig_height} ({img_format}, mode={img_mode})")
        print(f"Model Input Size:    {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} x 3")
        print("-" * 62)
        print(f"PREDICTED CLASS:     >>> {predicted_class.upper()} <<<")
        print(f"CONFIDENCE:          {confidence * 100.0:.2f}%")
        print("-" * 62)
        print("Class Probabilities:")
        print(f"  P(Cat): {p_cat:6.2f}%  [{cat_bar}]")
        print(f"  P(Dog): {p_dog:6.2f}%  [{dog_bar}]")
        print("=" * 62)

        if display_plot:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(5, 5))
                plt.imshow(img_rgb)
                plt.title(f"Prediction: {predicted_class} ({confidence * 100:.1f}%)", fontsize=14, fontweight="bold")
                plt.axis("off")
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(2)
                plt.close()
            except Exception:
                pass

        return {
            "image_path": image_path,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "p_cat": p_cat / 100.0,
            "p_dog": raw_pred,
            "dimensions": (orig_width, orig_height),
        }

    except Exception as exc:
        print(f"\n[Error] Failed to process image '{image_path}': {exc}")
        return None


def get_sample_images(sample_dir: str = DEFAULT_TEST_DIR) -> List[str]:
    """Retrieve all valid sample images from the test_images directory."""
    if not os.path.exists(sample_dir):
        return []
    return sorted([
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if os.path.splitext(f.lower())[1] in SUPPORTED_EXTENSIONS
    ])


def interactive_user_upload(model: tf.keras.Model) -> None:
    """Prompt user with an interactive upload and testing menu."""
    while True:
        print("\n" + "-" * 50)
        print("         CNN DOG-CAT PREDICTION MENU")
        print("-" * 50)
        print(" [1] Upload / Select image file (Opens file dialog)")
        print(" [2] Enter or paste an image file path")
        print(" [3] Run prediction on sample test images (Cat & Dog)")
        print(" [4] Exit")
        print("-" * 50)

        choice = input("Select an option [1-4] (default: 3): ").strip()

        if choice == "1":
            print("\nOpening file dialog to select image...")
            selected_path = open_file_dialog()
            if selected_path:
                predict_image(selected_path, model)
            else:
                print("No file selected.")

        elif choice == "2":
            user_path = input("\nEnter image path (or drag and drop file here): ").strip()
            if user_path:
                predict_image(user_path, model)
            else:
                print("Path cannot be empty.")

        elif choice in ("", "3"):
            sample_images = get_sample_images()
            if not sample_images:
                print(f"[Warning] No sample images found in '{DEFAULT_TEST_DIR}'.")
            else:
                print(f"\nRunning predictions on {len(sample_images)} sample images:\n")
                for img_path in sample_images:
                    predict_image(img_path, model)
            break

        elif choice == "4":
            print("Exiting prediction tool. Goodbye!")
            break
        else:
            print("Invalid selection. Please choose an option between 1 and 4.")


def main() -> None:
    """Parse arguments and route to prediction or upload pipeline."""
    parser = argparse.ArgumentParser(
        description="Predict whether an image is a Cat or a Dog using the trained 3-Layer CNN model."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to the image to classify (e.g., test_images/test_cat.jpg)",
    )
    parser.add_argument(
        "-u",
        "--upload",
        action="store_true",
        help="Open native file browser dialog to upload and classify a custom image",
    )
    parser.add_argument(
        "-s",
        "--samples",
        action="store_true",
        help="Run inference on all sample test images in test_images/",
    )
    parser.add_argument(
        "--model",
        dest="model_path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained .keras model file (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display image in a popup window with prediction title overlay",
    )

    args = parser.parse_args()

    # Load trained model
    try:
        model = load_cnn_model(args.model_path)
    except Exception as exc:
        print(f"[Fatal Error] {exc}")
        sys.exit(1)

    # 1. Direct file upload flag
    if args.upload:
        print("\nOpening file dialog to upload an image...")
        uploaded_path = open_file_dialog()
        if uploaded_path:
            predict_image(uploaded_path, model, display_plot=args.show_plot)
        else:
            print("No file selected.")
        return

    # 2. Specific image path passed as CLI argument
    if args.image_path is not None:
        predict_image(args.image_path, model, display_plot=args.show_plot)
        return

    # 3. Explicit samples flag
    if args.samples:
        samples = get_sample_images()
        for sample_path in samples:
            predict_image(sample_path, model)
        return

    # 4. Interactive mode or automated fallback
    if sys.stdin.isatty():
        interactive_user_upload(model)
    else:
        # Non-interactive / headless fallback: run sample images
        print("\nNo image specified. Running prediction on sample test images:\n")
        samples = get_sample_images()
        for sample_path in samples:
            predict_image(sample_path, model)


if __name__ == "__main__":
    main()
