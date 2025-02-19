
import os
import torch
import hydra
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf



def set_seeds(seed, deterministic=True):
    """Sets random seeds for reproducibility with optional determinism."""
    random.seed(seed)                     # Set Python's built-in random module seed
    np.random.seed(seed)                  # Set NumPy random seed
    torch.manual_seed(seed)               # Set PyTorch seed for CPU operations
    torch.cuda.manual_seed(seed)          # Set PyTorch seed for GPU operations (single-GPU)
    torch.cuda.manual_seed_all(seed)      # Set PyTorch seed for all GPUs (multi-GPU setups)

    torch.backends.cudnn.deterministic = deterministic  # Ensures reproducible operations if True (may slow down training)
    torch.backends.cudnn.benchmark = not deterministic  # Enables faster training if True (less reproducible)

def preprocessing(config, debug=False):
    """Prepares configuration by setting seeds, resolving paths, and generating an experiment name."""
    print(f"Experiment Name: {config.exp.name}")
    set_seeds(config.exp.seed, deterministic=config.train.deterministic)

    # Determine if the script is running locally or on a server
    if os.path.exists(config.path.project_root_dir):
        config.exp.on_server = False    # Running locally
        config.path.dataset_root_dir = os.path.abspath(
            os.path.join(config.path.project_root_dir, config.path.dataset_root_dir.local)
        )
    else:
        config.exp.on_server = True    # Running on a server
        config.path.project_root_dir = hydra.utils.get_original_cwd()  # Hydra-safe project root
        config.path.dataset_root_dir = os.path.abspath(
            os.path.join(config.path.project_root_dir, config.path.dataset_root_dir.server)
        )

    # Generate a unique experiment name for saving models and logs
    if config.model.identifier == "yolo":
        config.exp.save_name = f"rs{config.exp.seed}_{config.model.identifier}_{config.exp.name}_b{config.train.batch_size}_e{config.train.epoch}_lr{config.train.lr}"
    else:
        config.exp.save_name = f"rs{config.exp.seed}_{config.model.identifier}{config.model.code}_{config.exp.name}_b{config.train.batch_size}_e{config.train.epoch}_lr{config.train.lr}"

    # Print the final processed configuration if debug mode is enabled
    if debug: print(OmegaConf.to_yaml(config))

    return config



@hydra.main(version_base=None, config_path="../configs", config_name="default_config")
def main(config: DictConfig):
    """Main script entry point."""

    # Preprocess config settings before running the main pipeline
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
