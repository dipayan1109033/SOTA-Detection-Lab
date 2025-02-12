import torch
import torchvision.models.detection as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_model(cfg):
    model = models.fasterrcnn_resnet50_fpn(pretrained=cfg.model.pretrained)
    
    # Modify classifier for dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.model.num_classes)
    
    return model

def train_model(model, dataloader, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    model.train()
    for epoch in range(cfg.training.epochs):
        for images, targets in dataloader:
            images = [img.to(cfg.device) for img in images]
            targets = [{k: v.to(cfg.device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item()}")

def evaluate_model(model, dataloader, cfg):
    model.eval()
    # Implement evaluation logic here
