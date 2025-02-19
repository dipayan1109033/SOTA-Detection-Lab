import torch
from torchvision.ops import nms
from omegaconf import OmegaConf

import os
import copy
import json
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.setup_utils import *
from utils.metrics_utils import Evaluation_withCOCO, writeTo_csv_for_crossval


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Collate image-target pairs into a tuple.
def collate_fn(batch):
    return tuple(zip(*batch))
     
def check_for_cuda_device(device_str):
    """Checks for device 'CUDA', else returns the provided option."""
    if torch.cuda.is_available():
        device = 'cuda'
    elif device_str in ['cpu', 'mps']:
        device = device_str
    else:
        raise ValueError(f"Config device ('{device_str}') is not valid")
    print(f"Using device: {device}")
    return device


def get_new_folder_num(directory, prefix="train"):
    os.makedirs(directory, exist_ok=True)

    # List all folders in the directory
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # Extract the numeric part from folders following the provided prefix
    folder_numbers = []
    for folder in folders:
        if folder.startswith(prefix):
            number_part = folder[len(prefix):]  # Extract everything after prefix string
            if number_part.isdigit():  # Check if the part is a number
                number = int(number_part)
            elif number_part == "":  # If there's no number, assume it's 0
                number = 0
            else:
                continue  # Skip if it's not a valid folder
            folder_numbers.append(number)
    
    # Find the folder with the largest number
    if len(folder_numbers) > 0:
        max_number = max(folder_numbers)
        return max_number+1
    else:
        return 1

def create_unique_experiment_folder(base_dir, prefix="train"):
    folder_num = get_new_folder_num(base_dir, prefix)

    while True:
        folder_path = f"{base_dir}/{prefix}{folder_num}"
        try:
            os.makedirs(folder_path)
            print(f"Directory created: {folder_path}")
            return folder_num
        except FileExistsError:
            print(f"Directory already exists: {folder_path}, trying next...")
            folder_num += 1

def save_OmegaConfig(config, folder_path, filename="my_configs.yaml"):
    # Make a deep copy of the original configuration
    config_copied = copy.deepcopy(config)

    # Resolve interpolations in the copied configuration
    OmegaConf.resolve(config_copied)

    # Save the configuration to a file
    save_path = os.path.join(folder_path, filename)
    OmegaConf.save(config=config_copied, f=save_path)
    print(f"Configuration saved to {save_path}")

def read_OmegaConfig(folder_path, filename="my_configs.yaml"):
    # Construct the full path to the configuration file
    file_path = os.path.join(folder_path, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found at {file_path}")

    # Load the configuration file
    config = OmegaConf.load(file_path)
    print(f"Configuration loaded from {file_path}")
    return config

def get_exp_save_name(train_root_path):
    args_filepath = os.path.join(train_root_path, "args.yaml")
    data = helper.read_from_yaml(args_filepath)
    exp_save_name = f"rs{data['seed']}_yolo11_{data['optimizer']}_b{data['batch']}_e{data['epochs']}_lr{data['lr0']}"
    return exp_save_name


def save_model_checkpoint(model, optimizer, epoch, loss_dict, save_root_dir, save_filename):
    # Define the checkpoint file name
    ckpt_file_name = os.path.join(save_root_dir, "checkpoints", save_filename)
    os.makedirs(os.path.dirname(ckpt_file_name), exist_ok=True)

    # Save the model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss_dict': loss_dict
    }, ckpt_file_name)

    print(f"Model checkpoint saved at: {ckpt_file_name}")

def load_model_checkpoint(ckpt_file_path, model, optimizer=None, device='cuda'):
    """
    Load a model checkpoint for fine-tuning or continued training.

    Args:
        ckpt_file_path (str): Path to the saved checkpoint file (.pt).
        model (torch.nn.Module): The model architecture to load the weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into.
        device (str): The device to map the model and optimizer ('cpu' or 'cuda').
    
    Returns:
        model: The model with loaded state dict.
        optimizer: The optimizer with loaded state dict (if provided).
        epoch: The epoch at which the checkpoint was saved.
        loss_dict: The loss dictionary stored in the checkpoint.
    """
    # Load the checkpoint
    checkpoint = torch.load(ckpt_file_path, map_location=device)

    # Load the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # If optimizer is provided, load its state dict
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Move optimizer state to the correct device if using GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # Load other checkpoint data
    epoch = checkpoint['epoch']
    loss_dict = checkpoint['loss_dict']

    print(f"Model loaded from {ckpt_file_path}, resuming from epoch {epoch}")

    # Return model, optimizer, epoch, and loss dictionary
    return model, optimizer, epoch, loss_dict


# Model train method for one epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch on the provided data.

    Args:
    - model: The PyTorch model to be trained.
    - optimizer: The optimizer used for updating the model parameters.
    - data_loader: DataLoader providing batches of training data.
    - device: The device (CPU or GPU) to perform computations on.
    - epoch: The current epoch number (used for tracking training progress).

    Returns:
    - train_loss_list: A list containing the training losses for each batch.
    """
    model.train()
    model.to(device) 
    train_loss_list = []

    # Initialize a progress bar to show the training progress
    tqdm_bar = tqdm(data_loader, total=len(data_loader))
    
    # Iterate over the DataLoader
    for idx, data in enumerate(tqdm_bar):
        # Reset gradients to zero
        optimizer.zero_grad()
        
        # Unpack data
        images, targets = data

        # Move images and targets to the specified device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # targets = {'boxes'=tensor, 'labels'=tensor}

        # Forward pass through the model to get losses
        losses = model(images, targets)

        # Sum up individual losses into a single loss value
        loss = sum(loss for loss in losses.values())
        loss_val = loss.item()  # Get loss as a scalar value for logging
        train_loss_list.append(loss_val)  # Append loss to list

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Update progress bar description with current loss
        tqdm_bar.set_description(desc=f"Training Loss: {loss:.3f}")

    return train_loss_list

# Model evaluation method with no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluates the model on the provided test data and computes the validation losses.

    Args:
    - model: The PyTorch model to be evaluated.
    - data_loader: DataLoader providing batches of test data.
    - device: The device (CPU or GPU) to perform computations on.

    Returns:
    - val_loss_list: A list containing the validation losses for each batch.
    """
    #model.eval()
    model.to(device) 
    val_loss_list = []

    # Initialize a progress bar to show the evaluation progress
    tqdm_bar = tqdm(data_loader, total=len(data_loader))

    # Iterate over the DataLoader
    for idx, data in enumerate(tqdm_bar):
        images, targets = data

        # Move images and targets to the specified device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Disable gradient computation for evaluation
        with torch.no_grad():
            # Forward pass through the model to get losses
            losses = model(images, targets)

        # Sum up individual losses into a single loss value
        loss = sum(loss for loss in losses.values())
        loss_val = loss.item()  # Get loss as a scalar value
        val_loss_list.append(loss_val)  # Append loss to list

        # Update progress bar description with current loss
        tqdm_bar.set_description(desc=f"Validation Loss: {loss:.4f}")

    return val_loss_list

def predict(model, data_loader, device):
    """
    Perform inference using the provided model on the given data loader.

    Args:
    - model: The trained PyTorch model for object detection.
    - data_loader: DataLoader providing batches of images and their corresponding labels for inference.
    - device: The device (CPU or GPU) to perform computations on.

    Returns:
    - all_predictions: A list of dictionaries containing predicted bounding boxes, labels, and scores for each image.
    - all_labels: A list of ground truth labels corresponding to the images.
    """
    
    # Set the model to evaluation mode
    model.eval()
    model.to(device)

    # Initialize lists to store predictions and labels
    all_predictions = []
    all_labels = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for images, targets in data_loader: 
            images = list(image.to(device) for image in images)

            # Forward pass to get predictions
            predictions = model(images)

            # Extend the all_predictions list with batch predictions
            all_predictions.extend(predictions)

            # Extend the all_labels list with corresponding targets
            all_labels.extend(targets)

    return all_predictions, all_labels


def save_predictions_forYOLO(cfg, results, val_output_path, dataset_dir, dataset_folder, split, save_filename, extension=".jpg", folds=""):
    image_folder_path = os.path.join(dataset_dir, folds, f"{split}/images")
    image_list = helper.get_image_files(image_folder_path)

    dataset_info = DatasetInfo(dataset_dir)
    reverse_label_dict_yolo = dataset_info.get_reverse_label_dict_yolo()
    imagename_to_id_mapping = dataset_info.get_imagename_to_id()

    save_path = os.path.join(val_output_path, "predictions.json")
    coco_predictions = helper.read_from_json(save_path)

    custom_predictions = {}
    for pred in coco_predictions:
        image_name = pred['image_id'] + extension
        image_id = imagename_to_id_mapping[image_name]
        category_id = pred['category_id']
        bbox = pred['bbox']
        score = pred['score']

        # Prepare the custom object entry 
        aDetection = {
            "image_id": image_id,
            "class": reverse_label_dict_yolo[category_id], 
            "score": score,
            "box": bbox 
        }

        # Append the detection to the corresponding image
        if image_name not in custom_predictions:
            custom_predictions[image_name] = []
        custom_predictions[image_name].append(aDetection)

    experiments = {
        "mode": cfg.exp.mode,
        "train_id": save_filename.split("_")[0],
        "train_dataset": cfg.data.folder,
        "test_dataset": dataset_folder,
        "seed": cfg.exp.seed, 
        "batch_size": cfg.train.batch_size,
        "epoch": cfg.train.epoch,
        "learning_rate": cfg.train.lr,
        "optimizer": cfg.train.optimizer,
        "exp_name": f"{cfg.model.identifier}_{cfg.exp.name}",
        "save_filename": cfg.exp.save_name,
        "pretrained_model": cfg.model.pretrained_model
    }
    custom_format = {
        'categories': dataset_info.get_label_dict(),
        'image_files': image_list,
        'predictions': custom_predictions,
        'experiments': experiments,
        'results_ultralytics': results.results_dict
    }
    
    output_filepath = os.path.join(cfg.path.output_dir, "model_predictions", save_filename)
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    helper.write_to_json(custom_format, output_filepath)
    return output_filepath


def filter_predictions(all_predictions, confidence_threshold=0.0, iou_threshold=0.5):
    """
    Filter predictions by applying Non-Maximum Suppression (NMS) and confidence score threshold.

    Args:
    - all_predictions: A list of dictionaries containing predicted bounding boxes, labels, and scores for each image.
    - confidence_threshold: Confidence score threshold for filtering predictions.
    - iou_threshold: IoU threshold for applying NMS.

    Returns:
    - filtered_predictions: A list of dictionaries containing filtered bounding boxes, labels, and scores as NumPy arrays.
    """
    filtered_predictions = []

    for prediction in all_predictions:
        boxes = prediction['boxes']
        scores = prediction['scores']
        labels = prediction['labels']

        # Apply confidence score threshold
        confidence_mask = scores > confidence_threshold
        filtered_boxes = boxes[confidence_mask]
        filtered_scores = scores[confidence_mask]
        filtered_labels = labels[confidence_mask]

        # Remove any overlapping bounding boxes using NMS
        keep_indices = nms(filtered_boxes, filtered_scores, iou_threshold)

        # Filter predictions based on NMS results
        final_boxes = filtered_boxes[keep_indices]
        final_scores = filtered_scores[keep_indices]
        final_labels = filtered_labels[keep_indices]

        # Append filtered predictions to the result list, converting to NumPy arrays directly
        filtered_predictions.append({
            'boxes': final_boxes.cpu().numpy(),
            'scores': final_scores.cpu().numpy(),
            'labels': final_labels.cpu().numpy()
        })

    return filtered_predictions

def save_predictions(cfg, predictions, targets, dataset_dir, dataset_folder, save_filename, singleClass=False):
    image_ids = [target_dict['image_id'].item() for target_dict in targets]

    dataset_info = DatasetInfo(dataset_dir)
    reverse_label_dict = dataset_info.get_reverse_label_dict()
    id_imagename_mapping = dataset_info.get_id_to_imagename()

    image_list = []
    custom_predictions = {}
    for idx, prediction in enumerate(predictions):
        image_id = image_ids[idx]
        image_name = id_imagename_mapping[image_id]
        image_list.append(image_name)

        boxes = prediction['boxes']
        scores = prediction['scores']
        labels = prediction['labels']

        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [round(coordinate, 4) for coordinate in boxes[i].tolist()]
            adetection = {
                'image_id': image_id,
                'class': 1 if singleClass else reverse_label_dict[labels[i]], 
                'score': round(float(scores[i]), 4), 
                'box': [x1, y1, x2-x1, y2-y1]}
            detections.append(adetection)
        custom_predictions[image_name] = detections

    experiments = {
        "mode": cfg.exp.mode,
        "train_id": save_filename.split("_")[0],
        "train_dataset": cfg.data.folder,
        "test_dataset": dataset_folder,
        "seed": cfg.exp.seed, 
        "batch_size": cfg.train.batch_size,
        "epoch": cfg.train.epoch,
        "learning_rate": cfg.train.lr,
        "optimizer": cfg.train.optimizer,
        "exp_name": f"{cfg.model.identifier}_{cfg.exp.name}",
        "save_filename": cfg.exp.save_name,
        "pretrained_model": cfg.model.pretrained_model
    }
    custom_format = {
        'categories': dataset_info.get_label_dict(),
        'image_files': image_list,
        'predictions': custom_predictions,
        'experiments': experiments
    }

    # Save predictions to JSON file
    output_filepath = os.path.join(cfg.path.output_dir, "model_predictions", save_filename)
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    helper.write_to_json(custom_format, output_filepath)
    return output_filepath

def plot_learning_curves(train_loss, valid_loss, model_identifier, save_root_dir):
    """
    Plots the training and validation loss over iterations on the same figure and saves it as a PNG file.

    Args:
    - train_loss: List or array of training loss values, typically one per iteration or epoch.
    - valid_loss: List or array of validation loss values, typically one per iteration or epoch.
    - output_dir: Path to the directory where the loss plot will be saved.

    Saves:
    - 'losses_*.png' in the specified output directory, showing both training and validation loss curves.
    """
    
    # Create a single figure for both losses
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.plot(train_loss, color='blue', label='Training Loss')
    
    # Plot validation loss
    plt.plot(valid_loss, color='red', label='Validation Loss')
    
    # Labeling the axes
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Adding a legend
    plt.legend()
    
    # Saving the figure
    plt.savefig(f"{save_root_dir}/{model_identifier}_learning_curves.png")
    plt.close()  # Close the figure to free memory


def evaluate_predictions(dataset_dir, prediction_json_path, save_dir, model_identifier="yolo", csvFlag=True):

    evalObj = Evaluation_withCOCO(dataset_dir, prediction_json_path, score_threshold=0.5)
    evalObj.compute(save_dir, model_identifier, saveToCSV=csvFlag)
    
    evalObj.filter_coco_predictions(score_threshold=0.0)
    evalObj.compute(save_dir, model_identifier, saveToCSV=csvFlag)

def evaluate_cv_predictions(predictions_filelist, save_dir, model_identifier="yolo", csvFlag=True):
    # Initialize two dictionary
    data = helper.read_from_json(predictions_filelist[0])
    summary_values_0_0 = {key: [] for key, value in data['results_summary_0.0'].items()}
    summary_values_0_5 = {key: [] for key, value in data['results_summary_0.5'].items()}

    for json_filepath in predictions_filelist:
        data = helper.read_from_json(json_filepath)
        result_summary_0_0 = data['results_summary_0.0']
        result_summary_0_5 = data['results_summary_0.5']

        # Accumulate values
        for key1, key2 in zip(result_summary_0_0, summary_values_0_5):
            summary_values_0_0[key1].append(result_summary_0_0[key1])
            summary_values_0_5[key2].append(result_summary_0_5[key2])

    # Save results to csv file
    if csvFlag:
        writeTo_csv_for_crossval(summary_values_0_5, 0.5, predictions_filelist[0], f"{save_dir}/results_{model_identifier}.csv", newcsv=False, debug=True)
        writeTo_csv_for_crossval(summary_values_0_0, 0.0, predictions_filelist[0], f"{save_dir}/results_{model_identifier}.csv", newcsv=False, debug=True)
