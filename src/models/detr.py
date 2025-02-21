import os
import json
import shutil
import math
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Hugging Face Transformers for DETR
from transformers import DetrForObjectDetection, DetrFeatureExtractor

# --------------------------------------------
# UTILITY PLACEHOLDERS (Replace with your own)
# --------------------------------------------
def save_OmegaConfig(cfg, output_path):
    """
    Placeholder for saving your Hydra/OmegaConf config object.
    You can remove or adapt as needed.
    """
    config_path = os.path.join(output_path, "config.yaml")
    os.makedirs(output_path, exist_ok=True)
    # For instance:
    with open(config_path, "w") as f:
        f.write(str(cfg))  # Or OmegaConf.save(cfg, config_path)
    print(f"Saved config to: {config_path}")

def compute_mAP(all_predictions, all_targets):
    """
    Placeholder for computing mAP or other metrics.
    Replace with your actual metric calculation or pycocotools usage.
    """
    # Return a dummy dict for now
    return {"mAP": 0.0}

def post_process_detr_inference(outputs, feature_extractor, orig_image_size):
    """
    Placeholder for converting DETR raw outputs into bounding boxes & labels.
    You might use the built-in postprocessors or your own logic.
    """
    # See huggingface docs:
    # https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/detr#transformers.DetrForObjectDetection.post_process
    # Example:
    # results = model.post_process(outputs, target_sizes=[orig_image_size])
    # ...
    return []

def visualize_and_save(image, predictions, output_image_path):
    """
    Placeholder for drawing bounding boxes on `image` and saving.
    You can also skip visualization and just save JSON results, etc.
    """
    # For example, you could use PIL draw or OpenCV to overlay bounding boxes.
    image.save(output_image_path)

# --------------------------------------------
# DATASET EXAMPLE (Replace with your logic)
# --------------------------------------------
class DetrCustomDataset(Dataset):
    """
    Example dataset for DETR. This is just a skeleton. Replace or adapt to your data.
    Expects .__init__ to load filenames/annotations for one 'split' (train/val/test).
    """
    def __init__(self, root_dir, split, feature_extractor):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.feature_extractor = feature_extractor
        # Load your dataset's image paths & bounding box annotations here
        # e.g., self.image_files = [...]
        #       self.annotations = [...]
        self.image_files = []
        self.annotations = []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        ann = self.annotations[idx]  # Format depends on your annotation approach

        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Suppose ann has "boxes" and "labels" for a single image
        # Format could be: boxes = [[xmin, ymin, xmax, ymax], ...], labels = [class_id, ...]
        boxes = ann["boxes"]
        labels = ann["labels"]

        # Feature extractor step
        # For DETR, you typically provide raw image + annotation dict
        encoded_inputs = self.feature_extractor(
            images=image,
            annotations={"boxes": boxes, "labels": labels},
            return_tensors="pt"
        )
        
        # The feature extractor returns pixel_values, pixel_mask, and labels
        # Squeeze out the batch dimension, since we get shape [1, ...]
        item = {
            "pixel_values": encoded_inputs["pixel_values"].squeeze(0),
            "pixel_mask": encoded_inputs["pixel_mask"].squeeze(0),
            "labels": encoded_inputs["labels"]  # list of dicts with 'boxes' and 'class_labels'
        }
        return item


# --------------------------------------------
# DETR PIPELINE FUNCTIONS
# --------------------------------------------
def build_feature_extractor(cfg):
    """
    Builds the feature extractor for DETR.
    """
    # If you have a local or custom model, adapt this name.
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    return feature_extractor

def build_model(cfg):
    """
    Builds or loads a DETR model.
    """
    if cfg.model.saved_model_folder:
        # Suppose the checkpoint is: <output_dir>/train/<saved_model_folder>/detr_checkpoint.pth
        saved_model_path = os.path.join(
            cfg.path.output_dir, "train", cfg.model.saved_model_folder, "detr_checkpoint.pth"
        )
        print(f"Loading DETR from {saved_model_path}")

        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        model.load_state_dict(torch.load(saved_model_path))
    else:
        print("Loading pretrained DETR from huggingface hub: facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    return model

def train_model(model, cfg):
    """
    Trains the DETR model using a custom training loop.
    """

    # Build dataset & dataloader
    feature_extractor = build_feature_extractor(cfg)
    train_dataset = DetrCustomDataset(
        root_dir=cfg.path.dataset_root_dir,
        split="train",
        feature_extractor=feature_extractor
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.model.workers
    )

    # Choose device
    device = torch.device(cfg.exp.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr)

    # (Optional) If you have a learning rate scheduler, define it here
    # scheduler = ...

    model.train()
    num_epochs = cfg.train.epoch
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels
            )
            loss = outputs.loss  # Summed DETR loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # If using a scheduler, step it here
        # scheduler.step()

    # Save final checkpoint
    train_root_dir = os.path.join(cfg.path.output_dir, "train")
    os.makedirs(train_root_dir, exist_ok=True)

    # You might number your experiment folder the way YOLO does
    # e.g., train{cfg.exp.number}
    experiment_folder = f"train{cfg.exp.number}"
    model_save_dir = os.path.join(train_root_dir, experiment_folder)
    os.makedirs(model_save_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_save_dir, "detr_checkpoint.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"DETR model saved to: {checkpoint_path}")

    # Save the config
    save_OmegaConfig(cfg, model_save_dir)

def evaluate_model(model, cfg):
    """
    Evaluates the DETR model on a validation/test split.
    """
    if not cfg.model.saved_model_folder:
        raise ValueError("You must provide saved_model_folder to load a trained model for evaluation.")

    # Load checkpoint
    saved_model_path = os.path.join(
        cfg.path.output_dir, "train", cfg.model.saved_model_folder, "detr_checkpoint.pth"
    )
    print(f"Loading model from {saved_model_path}")
    model.load_state_dict(torch.load(saved_model_path))

    device = torch.device(cfg.exp.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    feature_extractor = build_feature_extractor(cfg)
    val_dataset = DetrCustomDataset(
        root_dir=cfg.path.dataset_root_dir,
        split=cfg.data.test_split,
        feature_extractor=feature_extractor
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.model.workers
    )

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            # We don't need labels for forward pass in eval mode,
            # but we do have them in batch for metric comparison
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            # Convert raw outputs to final boxes/labels
            predictions = post_process_detr_inference(
                outputs, feature_extractor, None  # or pass original image size
            )
            all_predictions.extend(predictions)

            # Each item in batch["labels"] is a dict with "boxes" and "labels"
            # Move them to CPU or keep them in some standard format
            all_targets.extend(batch["labels"])

    # Evaluate. You can compute mAP or any other detection metric here
    metrics_dict = compute_mAP(all_predictions, all_targets)
    print("Evaluation metrics:", metrics_dict)

    # Optionally save these results somewhere
    val_root_dir = os.path.join(cfg.path.output_dir, "validate")
    experiment_val_folder = f"val{cfg.exp.number}"
    output_dir = os.path.join(val_root_dir, experiment_val_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Example: store them in a JSON file
    results_path = os.path.join(output_dir, "metrics.json")
    with open(results_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved evaluation metrics to {results_path}")

def predict(model, cfg):
    """
    Run inference on a folder of images, saving or visualizing predictions.
    """
    if not cfg.model.saved_model_folder:
        raise ValueError("You must provide saved_model_folder to load a trained model for inference.")

    # Load checkpoint
    saved_model_path = os.path.join(
        cfg.path.output_dir, "train", cfg.model.saved_model_folder, "detr_checkpoint.pth"
    )
    print(f"Loading model from {saved_model_path}")
    model.load_state_dict(torch.load(saved_model_path))

    device = torch.device(cfg.exp.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image_folderpath = cfg.path.predict_image_folder
    if not image_folderpath:
        raise ValueError("Image folder path must be specified in cfg.path.predict_image_folder")

    # Prepare output folder
    predict_root_dir = os.path.join(cfg.path.output_dir, "predict", f"predict{cfg.exp.number}")
    os.makedirs(predict_root_dir, exist_ok=True)

    feature_extractor = build_feature_extractor(cfg)

    # Loop over images in the specified folder
    for img_name in os.listdir(image_folderpath):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        
        img_path = os.path.join(image_folderpath, img_name)
        image = Image.open(img_path).convert("RGB")

        # Preprocess single image
        encoding = feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(device)
        pixel_mask = encoding["pixel_mask"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # Convert to bounding boxes, labels, confidence
        predictions = post_process_detr_inference(outputs, feature_extractor, image.size)

        # Save the visualization or results
        output_image_path = os.path.join(predict_root_dir, img_name)
        visualize_and_save(image, predictions, output_image_path)
        print(f"Saved prediction: {output_image_path}")

# --------------------------------------------
# EXAMPLE USAGE (if running as standalone)
# --------------------------------------------
if __name__ == "__main__":
    class CFG:
        """
        Minimal stub config. 
        In your code, you likely have a Hydra config or OmegaConf object.
        """
        class Path:
            output_dir = "./outputs"
            dataset_root_dir = "./dataset"
            predict_image_folder = "./test_images"

        class Model:
            saved_model_folder = None  # or "train1" if you have a previously trained model
            weights = None
            freeze_layers = None
            save_interval = 1
            workers = 2
            identifier = "detr"

        class Train:
            batch_size = 2
            epoch = 3
            lr = 1e-4
            optimizer = "AdamW"

        class Exp:
            seed = 42
            device = "cuda:0"  # or "cpu"
            number = 1

        class Data:
            test_split = "val"

        class Eval:
            batch_size = 2

        path = Path()
        model = Model()
        train = Train()
        exp = Exp()
        data = Data()
        eval = Eval()

    cfg = CFG()

    # 1) Build model
    model = build_model(cfg)

    # 2) Train model
    train_model(model, cfg)

    # 3) Evaluate model (If you want to evaluate right after training,
    #    set cfg.model.saved_model_folder to "train{cfg.exp.number}" or similar)
    cfg.model.saved_model_folder = f"train{cfg.exp.number}"
    evaluate_model(model, cfg)

    # 4) Predict on a folder of images
    predict(model, cfg)
