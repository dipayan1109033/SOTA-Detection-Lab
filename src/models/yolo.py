import torch
from ultralytics import YOLO
from datasets.dataset_loader import get_dataloader  # Each model loads its dataset

def build_model(cfg):
    """Builds YOLO model."""
    return YOLO(cfg.model.weights)

def train_model(model, cfg):
    """Trains YOLO model."""
    # Load dataset using the prepared temp_dataset_path
    train_loader, _ = get_dataloader(cfg.path.temp_dataset_path, cfg, split="train")
    model.train(data=train_loader, epochs=cfg.train.epoch)

def evaluate_model(model, cfg):
    """Evaluates YOLO model."""
    # Load validation dataset
    _, val_loader = get_dataloader(cfg.path.temp_dataset_path, cfg, split="val")
    results = model.val(data=val_loader)
    print(results)
