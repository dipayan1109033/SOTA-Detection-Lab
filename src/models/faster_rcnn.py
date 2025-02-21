import os
import math
import shutil
import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

# Example utility imports. Adjust to match your actual modules:
from src.utils.train_utils import (
    get_new_folder_num,
    train_one_epoch,
    evaluate,
    # Possibly more utilities here...
)

from src.utils.common_utils import (
    Helper,
    save_OmegaConfig,
    get_exp_save_name,
    evaluate_predictions,
    # Possibly more utilities...
)

helper = Helper()

def build_model(cfg):
    """
    Builds a Faster R-CNN model.
    If cfg.model.saved_model_folder is provided, loads a saved checkpoint. 
    Otherwise, creates a new model (optionally pretrained on COCO).
    """

    train_root_dir = f"{cfg.path.output_dir}/train"
    
    # Create a base Faster R-CNN model (ResNet50 FPN backbone)
    # For a standard pretrained model on COCO, set pretrained=True below
    model = fasterrcnn_resnet50_fpn(pretrained=True) 
    
    # Example: Freeze certain layers if requested
    # (You might implement a more nuanced approach.)
    if cfg.model.freeze_layers:
        for name, parameter in model.named_parameters():
            # Decide which layers to freeze based on your logic
            # For example, freeze the backbone:
            if "backbone" in name:
                parameter.requires_grad = False

    # Load saved checkpoint if a saved folder is specified
    if cfg.model.saved_model_folder:
        saved_model_path = os.path.join(
            train_root_dir, 
            cfg.model.saved_model_folder,
            "weights",
            "fasterrcnn_checkpoint.pth"  # Example filename
        )
        if not os.path.isfile(saved_model_path):
            raise FileNotFoundError(f"Checkpoint not found at: {saved_model_path}")

        checkpoint = torch.load(saved_model_path, map_location=cfg.exp.device)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded pretrained model from: {saved_model_path}")
    else:
        print(f"Loaded base Faster R-CNN with pretrained=True (COCO).")

    # Move model to device
    model.to(cfg.exp.device)

    return model

def train_model(model, cfg):
    """
    Trains a Faster R-CNN model using train_utils helpers.
    """

    # Figure out which experiment folder number to create
    train_root_dir = f"{cfg.path.output_dir}/train"
    cfg.exp.number = get_new_folder_num(train_root_dir, prefix="train")

    # Typically, you'd set up your dataset and dataloader here
    # e.g.:
    #   train_dataset = MyCustomDataset(cfg.path.dataset_root_dir, transforms=..., split='train')
    #   train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, ...)
    #
    # This code snippet just shows placeholders.

    # Example placeholders:
    train_dataset = ...  # TODO: implement or load your dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.model.workers,
        collate_fn=lambda batch: tuple(zip(*batch))  # typical for detection
    )

    # Set up optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.train.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=cfg.train.lr,
            momentum=0.9,
            weight_decay=0.0005
        )
    else:
        # Example: Adam
        optimizer = torch.optim.Adam(params, lr=cfg.train.lr)

    # (Optional) learning rate scheduler
    # Example: StepLR every X epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.train.lr_scheduler_step,
        gamma=cfg.train.lr_scheduler_gamma
    )

    # Training loop
    num_epochs = cfg.train.epoch
    device = cfg.exp.device

    for epoch in range(num_epochs):
        # train for one epoch, printing every iteration
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)

        # update the learning rate
        lr_scheduler.step()

        # Optionally run validation or checkpoint mid-training if desired
        if (epoch + 1) % cfg.model.save_interval == 0:
            checkpoint_dir = os.path.join(train_root_dir, f"train{cfg.exp.number}", "weights")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"fasterrcnn_checkpoint_epoch{epoch+1}.pth")
            torch.save({"model": model.state_dict()}, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1} -> {checkpoint_path}")

    # Final model save
    final_weights_dir = os.path.join(train_root_dir, f"train{cfg.exp.number}", "weights")
    os.makedirs(final_weights_dir, exist_ok=True)
    final_ckpt_path = os.path.join(final_weights_dir, "fasterrcnn_checkpoint.pth")
    torch.save({"model": model.state_dict()}, final_ckpt_path)
    print(f"Final model checkpoint saved -> {final_ckpt_path}")

    # Save config arguments
    train_root_path = f"{train_root_dir}/train{cfg.exp.number}"
    save_OmegaConfig(cfg, train_root_path)

def evaluate_model(model, cfg):
    """
    Evaluates a Faster R-CNN model on a validation/test set.
    """

    # Check for provided saved model folder
    if not cfg.model.saved_model_folder:
        raise ValueError("You need to provide cfg.model.saved_model_folder to load the trained model")
    
    train_folder = cfg.model.saved_model_folder
    cfg.exp.number = int(train_folder[5:])  # e.g., "train12" -> 12
    
    model_save_dir = os.path.join(cfg.path.output_dir, "train", train_folder)
    exp_save_name = get_exp_save_name(model_save_dir)

    # Load the model checkpoint if needed (often you pass `model` pre-loaded, 
    # but you can also reload here)
    checkpoint_path = os.path.join(model_save_dir, "weights", "fasterrcnn_checkpoint.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=cfg.exp.device)
    model.load_state_dict(checkpoint["model"])
    model.to(cfg.exp.device)
    model.eval()

    # Prepare your validation/test dataset and DataLoader
    split = cfg.data.test_split  # e.g., "val" or "test"
    val_dataset = ...  # TODO: your dataset creation
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.model.workers,
        collate_fn=lambda batch: tuple(zip(*batch))
    )

    # Output folder for evaluation results
    val_root_dir = f"{cfg.path.output_dir}/validate/val{cfg.exp.number}"
    if os.path.exists(val_root_dir):
        shutil.rmtree(val_root_dir)
    os.makedirs(val_root_dir, exist_ok=True)

    dataset_folder = f"{helper.get_immediate_folder_name(cfg.path.dataset_root_dir)}_{split}"
    output_path = os.path.join(val_root_dir, dataset_folder)
    os.makedirs(output_path, exist_ok=True)

    # Run evaluation
    eval_results = evaluate(model, val_loader, device=cfg.exp.device)
    print("Evaluation results:", eval_results)

    # (Optional) Save predictions to file
    # e.g., you may have a custom function like 'save_predictions_forRCNN'
    # save_predictions_forRCNN(model, val_loader, output_path)

    # Optionally run further metric calculations or store them
    save_filename = f"{train_folder}_{exp_save_name}__{dataset_folder}.json"
    output_filepath = os.path.join(output_path, save_filename)

    # Example post-processing or custom evaluation
    evaluate_predictions(
        dataset_dir=cfg.path.dataset_root_dir,
        split=split,
        predictions_file=output_filepath,
        save_dir=cfg.path.output_dir,
        model_identifier=cfg.model.identifier
    )

def predict(model, cfg):
    """
    Runs inference with a trained Faster R-CNN model on images in a folder,
    saving predictions in a specified output directory.
    """

    # Check for provided saved model folder
    if not cfg.model.saved_model_folder:
        raise ValueError("You need to provide cfg.model.saved_model_folder to load the trained model")

    # Image folder for predictions
    image_folderpath = cfg.path.predict_image_folder
    if not image_folderpath:
        raise ValueError("You need to specify cfg.path.predict_image_folder for prediction images")

    # Derive experiment number from folder name (e.g., "train12" -> 12)
    train_folder = cfg.model.saved_model_folder
    cfg.exp.number = int(train_folder[5:])

    # Optionally load checkpoint (if model not already loaded)
    checkpoint_path = os.path.join(
        cfg.path.output_dir, 
        "train", 
        train_folder, 
        "weights", 
        "fasterrcnn_checkpoint.pth"
    )
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=cfg.exp.device)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded model checkpoint from: {checkpoint_path}")

    model.to(cfg.exp.device)
    model.eval()

    # Prepare output folder
    predict_root_dir = f"{cfg.path.output_dir}/predict/predict{cfg.exp.number}"
    image_folder = helper.get_immediate_folder_name(image_folderpath)
    output_folder_path = os.path.join(predict_root_dir, image_folder)
    if os.path.exists(output_folder_path):
        shutil.rmtree(output_folder_path)
    os.makedirs(output_folder_path, exist_ok=True)

    # Run inference on each image
    from PIL import Image
    import glob

    image_files = glob.glob(os.path.join(image_folderpath, "*.jpg")) + \
                  glob.glob(os.path.join(image_folderpath, "*.png")) + \
                  glob.glob(os.path.join(image_folderpath, "*.jpeg"))

    for img_file in image_files:
        image = Image.open(img_file).convert("RGB")
        with torch.no_grad():
            prediction = model([torch.tensor(torchvision.transforms.functional.to_tensor(image),
                                             device=cfg.exp.device)])[0]

        # TODO: parse prediction (boxes, labels, scores)
        # Example:
        boxes = prediction["boxes"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()

        # Optionally filter by threshold
        keep_idx = scores >= 0.5
        boxes = boxes[keep_idx]
        labels = labels[keep_idx]
        scores = scores[keep_idx]

        # Draw or save bounding boxes using your custom function
        # e.g.:
        # drawn_image = draw_boxes_on_image(image, boxes, labels, scores)
        # drawn_image.save(os.path.join(output_folder_path, os.path.basename(img_file)))

        # Or simply store them in a file, etc.

    print(f"Predictions saved in: {output_folder_path}")
