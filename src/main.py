import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import hydra
import random
import numpy as np
import importlib.util
from omegaconf import DictConfig, OmegaConf

from src.utils.setup_utils import *
from src.utils.common_utils import Helper
helper = Helper()



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

    # Resolve project and dataset paths
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

    # Generate experiment save name
    config.exp.save_name = (
        f"rs{config.exp.seed}_{config.model.identifier}_{config.exp.name}_"
        f"b{config.train.batch_size}_e{config.train.epoch}_lr{config.train.lr}"
    )

    if debug:
        print(OmegaConf.to_yaml(config))

    return config



def dynamic_import(module_path, module_name):
    """Dynamically import a module given its path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Cannot find module {module_name} at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@hydra.main(version_base=None, config_path="../configs", config_name="default_config")
def main(config: DictConfig):
    """Main script entry point."""
    config = preprocessing(config, debug=False)

    # Step 1: Prepare the dataset and update config
    cfg = setup_dataset(config)

    # Step 2: Dynamically load the model script
    model_script_path = os.path.join("src", "models", f"{cfg.model.identifier}.py")
    if not os.path.exists(model_script_path):
        raise FileNotFoundError(f"Model script {model_script_path} not found.")

    model_module = dynamic_import(model_script_path, cfg.model.identifier)

    # Step 3: Build the model
    model = model_module.build_model(cfg)

    # Step 4: Train or evaluate the model
    if cfg.exp.mode == "train":
        model_module.train_model(model, cfg)
    elif cfg.exp.mode == "evaluate":
        model_module.evaluate_model(model, cfg)
    else:
        raise ValueError("Invalid task. Choose 'train' or 'evaluate'.")

if __name__ == "__main__":
    main()
