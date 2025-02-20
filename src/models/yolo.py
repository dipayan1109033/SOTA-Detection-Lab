
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

    # Get output folder number for the experiemnt
    train_root_dir = f"{cfg.path.output_dir}/train"
    cfg.exp.number = get_new_folder_num(train_root_dir, prefix="train")

    # Get dataset yaml config file
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
                save_period=cfg.train.save_interval,
                device=cfg.exp.device
                )
    
    # Save config arguments
    train_root_path = f"{train_root_dir}/train{cfg.exp.number}"
    save_OmegaConfig(cfg, train_root_path)

def evaluate_model(model, cfg):
    """Evaluates YOLO model."""

    # Check for provided saved model folder
    if not cfg.model.saved_model_folder:
        raise ValueError(f"You need to provide saved_model_folder name to load the trained model")
    
    # Get train details
    train_folder = cfg.model.saved_model_folder
    split = cfg.data.test_split
    cfg.exp.number = int(train_folder[5:])
    model_save_dir = os.path.join(cfg.path.output_dir, "train", train_folder)
    exp_save_name = get_exp_save_name(model_save_dir)
    
    # Get dataset config file
    dataset_dir = cfg.path.dataset_root_dir
    dataset_yaml = os.path.join(dataset_dir, "yolo_dataset.yaml")   # Yolo dataset yaml filepath

    # Prepare output folder
    val_root_dir = f"{cfg.path.output_dir}/validate/val{cfg.exp.number}"
    if os.path.exists(val_root_dir): shutil.rmtree(val_root_dir)
    dataset_folder = f"{helper.get_immediate_folder_name(dataset_dir)}_{split}"

    results = model.val(data=dataset_yaml, 
                        split=split, 
                        save_json= True,
                        project=val_root_dir, 
                        name=dataset_folder,
                        device=cfg.exp.device
                    )

    # Save predicted bounding boxes and some metrics
    output_path = os.path.join(val_root_dir, dataset_folder)
    save_filename = f"{train_folder}_{exp_save_name}__{dataset_folder}.json"
    output_filepath = save_predictions_forYOLO(cfg, results, output_path, dataset_dir, dataset_folder, split, save_filename, extension=".jpg")

    evaluate_predictions(dataset_dir, split, output_filepath, save_dir=cfg.path.output_dir, model_identifier=cfg.model.identifier)

    
def predict(model, cfg):
    """Predict using trained YOLO model."""

    # Check for provided saved model folder
    if not cfg.model.saved_model_folder:
        raise ValueError(f"You need to provide saved_model_folder name to load the trained model")

    # Image folder for predictions
    image_folderpath = cfg.path.predict_image_folder
    if not image_folderpath:
        raise ValueError(f"Image folder path need to be specified through: cfg.path.predict_image_folder")

    # Get train details
    train_folder = cfg.model.saved_model_folder
    cfg.exp.number = int(train_folder[5:])

    # Prepare output folder
    predict_root_dir = f"{cfg.path.output_dir}/predict/predict{cfg.exp.number}"
    image_folder = helper.get_immediate_folder_name(image_folderpath)
    output_folder_path = os.path.join(predict_root_dir, image_folder)
    if os.path.exists(output_folder_path): shutil.rmtree(output_folder_path)


    # Predictions
    results = model.predict(image_folderpath, device=cfg.exp.device, batch=10)
    save_yolo_predictions_to_custom_labels(results, image_folderpath, output_folder_path, draw_scoreThreshold=0.5)
