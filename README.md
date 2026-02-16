# Endfield-Map-CLS

Endfield-Map-CLS is a Proof of Concept (PoC) for multi-tier open-world map tracking using YOLOv26 classification. It is designed to identify specific map regions or tiers in a game by classifying minimap screenshots against a dataset generated from full map source images.

## ⚠️ Project Status & Disclaimer

This repository is provided strictly as a **proof of concept** and a source of **pretrained weights** to demonstrate the methodology.

1.  **No Dataset Updates**: The `source_images/` directory contains a sample set. **We will not provide updates** to the source images or the generated dataset.
2.  **User Responsibility**: Users must provide their own high-quality map source images to train a model for their specific use case.
3.  **Methodology Only**: The core value of this repo is the preprocessing and training pipeline (`preprocess.py` -> `train.py`), not the specific weights provided.

## Model Parameters & Constraints

- **Model Architecture**: YOLOv26n-cls (Nano Classification Model)
- **Input Resolution**: 128x128 (Immutable)
    - The model input size is fixed at 128x128 to match the specific receptive field required for this task.
- **Source Map Scaling**: **0.16x**
    - **CRITICAL**: Source images in `source_images/` MUST be resized to **0.16x** of the original game map size. The preprocessing pipeline assumes this scale to match the minimap crop factor.
- **Inference Baseline**: **720p**
    - The system is calibrated for game screenshots taken at **1280x720** resolution.
    - `predict.py` automatically scales inputs to this 720p baseline before processing, but the source maps must align with this scale.
- **Training Epochs**: 100 (Default)
- **Performance**: 
    - The provided weights are a demonstration. Accuracy on your specific task will depend entirely on the completeness of your `source_images`.

## Features

- **Automated Dataset Generation**: Slices large map images into training samples with augmentations (rotation, masking, occlusion).
- **YOLOv26 Classification**: Uses the Ultralytics YOLOv26 engine for efficient and accurate classification.
- **Robust Inference**: Preprocessing pipeline ensures input images match the training distribution (circular masking, resizing).

## Installation

1.  Clone the repository.
2.  Install the required Python packages:

    ```bash
    pip install ultralytics opencv-python numpy
    ```

## Usage

### 1. Data Preparation

Place your source map images (e.g., `.png`, `.jpg`) in the `source_images/` directory. The filename will be used as the class name (e.g., `Map01Base.png` -> class `Map01Base`).

Run the preprocessing script to generate the dataset:

```bash
python preprocess.py --input source_images --output dataset
```

This will create a `dataset/` directory with `train` and `val` splits containing augmented map patches.

### 2. Training

Train the YOLO model on the generated dataset:

```bash
python train.py --epochs 100 --batch 128
```

Arguments:
- `--data`: Path to dataset root (default: `dataset`).
- `--model`: Pretrained model path (default: `yolo26n-cls.pt`).
- `--epochs`: Number of training epochs.
- `--imgsz`: Input image size (default: 128).

Training results will be saved to `runs/classify/`.

### 3. Inference

Run predictions on a single image:

```bash
python predict.py path/to/image.jpg
```

Arguments:
- `--model`: Path to a specific model file (optional). If not provided, it automatically finds the latest `best.pt` in `runs/classify`.
- `--debug`: Saves the preprocessed input image as `debug_inference.jpg` for inspection.

## Project Structure

- `preprocess.py`: Generates the dataset from source images.
- `train.py`: Wrapper around YOLOv26 training.
- `predict.py`: Inference script with image preprocessing.
- `dataset/`: Generated training and validation data.
- `source_images/`: Original map images.

---

**This repository is a Proof of Concept (PoC) for solving multi-tier open-world map tracking. It is maintained strictly as a personal research project. No technical support, integration guides, or maintenance will be provided for third-party frameworks.**
