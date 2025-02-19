import torch
from torchvision.models.detection import detr_resnet50
from datasets.dataset_loader import get_dataloader

def build_model(cfg):
    """Builds DETR model."""
    model = detr_resnet50(pretrained=cfg.model.pretrained)
    num_classes = cfg.data.num_classes
    model.class_embed = torch.nn.Linear(model.class_embed.in_features, num_classes)
    return model

def train_model(model, cfg):
    """Trains DETR model."""
    train_loader, _ = get_dataloader(cfg.path.temp_dataset_path, cfg, split="train")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    model.train()
    for epoch in range(cfg.train.epoch):
        for images, targets in train_loader:
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
        print(f"Epoch {epoch} completed with loss {losses.item()}")

def evaluate_model(model, cfg):
    """Evaluates DETR model."""
    _, val_loader = get_dataloader(cfg.path.temp_dataset_path, cfg, split="val")
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            print(f"Evaluated batch with {len(outputs)} outputs")
