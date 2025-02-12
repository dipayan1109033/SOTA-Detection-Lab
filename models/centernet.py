import torch
import os

def build_model(cfg):
    repo = cfg.model.source
    os.system(f"git clone {repo} models/centernet_repo")
    from models.centernet_repo.models.detector import CenterNetDetector
    
    model = CenterNetDetector(cfg.model.num_classes)
    return model

def train_model(model, dataloader, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    model.train()
    for epoch in range(cfg.training.epochs):
        for images, targets in dataloader:
            images = images.to(cfg.device)
            targets = targets.to(cfg.device)

            loss = model(images, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item()}")

def evaluate_model(model, dataloader, cfg):
    model.eval()
    # Implement evaluation logic here
