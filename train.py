import argparse

from ultralytics import YOLO

# Configuration
DEFAULT_CONFIG = {
    "model": "yolo26n-cls.pt",  # Base model name (e.g., yolo26n-cls.pt). Will download automatically if not found.
    "imgsz": 128,  # Input image size (width and height) for training
    "batch": 128,  # Batch size for training
    "workers": 8,  # Number of worker threads for data loading
    "patience": 20,  # Epochs to wait for no improvement before early stopping
    "epochs": 100,  # Total number of training epochs
}


def train(args):
    """Executes the training pipeline."""
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        save=True,
        project="runs/classify",
        name=args.name or "train",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument("--data", default="dataset", help="Path to dataset root")
    parser.add_argument(
        "--model",
        default=DEFAULT_CONFIG["model"],
        help="Model name or path (downloads automatically if not found)",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--imgsz", type=int, default=DEFAULT_CONFIG["imgsz"])
    parser.add_argument("--batch", type=int, default=DEFAULT_CONFIG["batch"])
    parser.add_argument("--workers", type=int, default=DEFAULT_CONFIG["workers"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--device", default="0", help="CUDA device (e.g. 0 or 0,1)")
    parser.add_argument("--name", default=None, help="Experiment name")

    args = parser.parse_args()
    train(args)
