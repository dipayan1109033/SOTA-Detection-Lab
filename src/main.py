
import os
import torch
import hydra
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf



def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocessing(config, debug=False):
    # Set random seeds for reproducibility
    set_seeds(config.exp.seed)

    # # Update dataset and project directory based on local/server
    # if os.path.exists(config.path.project_root_dir.server):
    #     config.exp.on_server = True
    #     config.data.root_dir = config.data.root_dir.server
    #     config.path.project_root_dir = config.path.project_root_dir.server
    # else:
    #     config.exp.on_server = False
    #     config.data.root_dir = config.data.root_dir.local
    #     config.path.project_root_dir = config.path.project_root_dir.local

    # # Add few arguments
    # if config.model.identifier == "yolo":
    #     config.exp.save_name = f"rs{config.exp.seed}_{config.model.identifier}_{config.exp.name}_b{config.train.batch_size}_e{config.train.epoch}_lr{config.train.lr}"
    # else:
    #     config.exp.save_name = f"rs{config.exp.seed}_{config.model.identifier}{config.model.code}_{config.exp.name}_b{config.train.batch_size}_e{config.train.epoch}_lr{config.train.lr}"

    if debug: print(OmegaConf.to_yaml(config))

    return config

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(config: DictConfig):
    print(f"Experiment Name: {config.exp.name}")
    config = preprocessing(config, debug=True)
    return

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
