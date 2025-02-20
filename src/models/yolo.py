
import os
import math
import shutil
from ultralytics import YOLO

from src.utils.train_utils import *
from src.utils.common_utils import Helper
helper = Helper()


def build_model(cfg):
    """Builds YOLO model."""
    train_root_dir = f"{cfg.path.output_dir}/train"

    # Load a pretrained YOLO model
    if cfg.model.saved_model_folder:
        saved_model_path = f"{train_root_dir}/{cfg.model.saved_model_folder}/weights/best.pt"
        model = YOLO(saved_model_path)
        print(f"Loaded pretrained model from: {saved_model_path}")
    else:
        model = YOLO(cfg.model.weights)
        print(f"Loaded model weights from: {cfg.model.weights}")

    return model

def train_model(model, cfg):
    """Trains YOLO model."""

    # Set random seeds for this process
    set_seeds(cfg.exp.seed)

    # Get output folder number for the experiemnt
    train_root_dir = f"{cfg.path.output_dir}/train"
    cfg.exp.number = get_new_folder_num(train_root_dir, prefix="train")

    # Create dataset for YOLO
    dataset_dir = cfg.path.dataset_root_dir
    dataset_yaml = os.path.join(dataset_dir, "yolo_dataset.yaml")   # Yolo dataset yaml filepath

    # Train the model
    model.train(data=dataset_yaml, 
                seed=cfg.exp.seed, 
                batch=cfg.train.batch_size, 
                epochs=cfg.train.epoch, 
                lr0=cfg.train.lr, 
                optimizer=cfg.train.optimizer, 
                project=train_root_dir, 
                name=f"train{cfg.exp.number}", 
                freeze=cfg.train.freeze_layers,
                save_period=cfg.train.save_interval
                )
    
    # Save config arguments
    train_root_path = f"{train_root_dir}/train{cfg.exp.number}"
    save_OmegaConfig(cfg, train_root_path)

# def evaluate_model(model, cfg):
#     """Evaluates YOLO model."""
#     # Load validation dataset
#     _, val_loader = get_dataloader(cfg.path.temp_dataset_path, cfg, split="val")
#     results = model.val(data=val_loader)
#     print(results)
