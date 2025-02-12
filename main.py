import hydra
from omegaconf import DictConfig
import torch
import os

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    print(f"Experiment: {cfg.experiment_name}")

    # Load dataset
    from datasets.dataset_loader import get_dataloader
    train_loader, val_loader = get_dataloader(cfg.dataset)

    # Dynamically load the correct model script
    model_script = os.path.join("models", f"{cfg.model.model_name}.py")
    if not os.path.exists(model_script):
        raise ValueError(f"Model script {model_script} not found.")

    # Import the model script dynamically
    model_module = __import__(f"models.{cfg.model.model_name}", fromlist=["build_model", "train_model", "evaluate_model"])
    
    model = model_module.build_model(cfg)
    model.to(cfg.device)

    # Train or evaluate
    if cfg.task == "train":
        model_module.train_model(model, train_loader, cfg)
    elif cfg.task == "evaluate":
        model_module.evaluate_model(model, val_loader, cfg)
    else:
        print("Invalid task. Choose 'train' or 'evaluate'.")

if __name__ == "__main__":
    main()
