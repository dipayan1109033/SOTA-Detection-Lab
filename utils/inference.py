import torch
import os

def load_model(cfg):
    model_script = os.path.join("models", f"{cfg.model.model_name}.py")
    if not os.path.exists(model_script):
        raise ValueError(f"Model script {model_script} not found.")

    model_module = __import__(f"models.{cfg.model.model_name}", fromlist=["build_model"])
    model = model_module.build_model(cfg)
    model.load_state_dict(torch.load(cfg.model.checkpoint, map_location=cfg.device))
    model.to(cfg.device)
    model.eval()
    return model

def run_inference(model, image):
    with torch.no_grad():
        predictions = model([image.to(model.device)])
    return predictions
