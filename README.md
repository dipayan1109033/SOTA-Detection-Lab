# 🚀 SOTA-Detection-Lab: A Flexible Framework for Object Detection Training

**SOTA-Detection-Lab** is a **PyTorch-based** framework designed for training, fine-tuning, and evaluating **state-of-the-art object detection models** on **custom datasets**. It supports models like **YOLO**, **Faster R-CNN**, **DETR**, with upcoming additions (e.g., **CenterNet**, **EfficientDet**), and offers **independent training scripts** along with **Hydra-based configuration** for seamless model switching and rapid experimentation.

## 📌 Key Features
- ✅ **Effortless model switching** – train **YOLO**, **Faster R-CNN**, **DETR**, and more with simple configuration changes.  
- ✅ **Standalone training scripts** – no reliance on heavy frameworks like **Detectron2** or **MMDetection**.  
- ✅ **Hydra-powered configuration** – easily customize models, datasets, and hyperparameters.  
- ✅ **Fine-tune on custom datasets** with minimal setup and overhead.  
- ✅ **Unified dataset handling** – a consistent dataset format for all supported models.  
- ✅ **Modular architecture** – easily extend and integrate new models or components.  

---


## 📂 Project Structure

```plaintext
SOTA-DETECTION-LAB/
│
├── configs/                           # Configuration files for models and experiments
│   ├── model/                         # Model-specific configuration files
│   │   ├── detr.yaml                  # Config for DETR model
│   │   ├── faster_rcnn.yaml           # Config for Faster R-CNN model
│   │   └── yolo.yaml                  # Config for YOLO model
│   ├── default_config.yaml            # Global/default settings for the project
│   └── experiment.yaml                # Settings specific to a particular experiment
│
├── data/                              # Dataset storage
│   ├── raw/                           # Unprocessed, original datasets
│   └── processed/                     # Preprocessed datasets ready for use
│       ├── coco8/                     # Example of a processed dataset (custom format)
│       └── ...                        # Additional processed datasets
│
├── experiments/                       # Experiment-related data
│   ├── input/                         # Files used as input during experiments
│   └── output/                        # Outputs from experiments (e.g., logs, results, checkpoints)
│
├── results/                           # Final evaluation results and reports
│
├── src/                               # Source code
│   ├── data_prep/                     # Scripts for dataset preparation and preprocessing
│   │   ├── convert_formats.py         # Convert datasets to custom formats
│   │   └── check_format.py            # Validate and check custom dataset
│   │
│   ├── datasets/                      # Dataset utility functions
│   │   └── transforms.py              # Data augmentation and preprocessing utilities
│   │
│   ├── models/                        # Model implementations
│   │   ├── detr.py                    # DETR model implementation
│   │   ├── faster_rcnn.py             # Faster R-CNN implementation
│   │   └── yolo.py                    # YOLO model implementation
│   │
│   ├── utils/                         # General-purpose utility scripts
│   │   ├── common_utils.py            # Helper functions used across the project
│   │   ├── prepare_utils.py           # Utilities for dataset preprocessing
│   │   ├── dataset_utils.py           # TorchVision Dataset and DataLoader helpers
│   │   ├── setup_utils.py             # Tools for setting up experimental datasets
│   │   ├── train_utils.py             # Training loop helpers, schedulers, etc.
│   │   └── metrics_utils.py           # Functions for evaluation metrics and reporting
│   │
│   └── main.py                        # Main script to run training and evaluation
│
├── venv39/                            # Python 3.9 virtual environment for package management
│
├── requirements.txt                   # List of dependencies for the project
├── .gitignore                         # Specifies files and directories to ignore in Git
├── LICENSE                            # License information for the project
└── README.md                          # Overview, setup instructions, and usage guide
```



## 📖 Setup & Installation

### **1️⃣ Clone the Repositories**

```bash
git clone https://github.com/dipayan1109033/SOTA-Detection-Lab.git
cd SOTA-Detection-Lab
```

To facilitate the calculation of object detection metrics, clone the following repository inside the `src/utils` directory:

```bash
cd src/utils
git clone https://github.com/dipayan1109033/calculate_ODmetrics
```

### **2️⃣ Install Dependencies**

Ensure you have Python 3.9+ and PyTorch installed. Then install the required packages using the `requirements.txt` file:


```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🔹 Training a Model

To train a YOLO model on the `coco8` dataset:

```bash
python src/main.py model="yolo" exp.mode="train" data.folder="coco8" exp.name="yolo_with_coco8"
```

### 🔹 Evaluating a Model

To evaluate a trained YOLO model using the validation split:

```bash
python src/main.py model="yolo" exp.mode="evaluate" data.folder="coco8" data.test_split="val" model.saved_model_folder="train1" exp.name="train1_evaluate_coco8_val"
```

### 🔹 Making Predictions

To make predictions on new images using a trained YOLO model:

```bash
python src/main.py model="yolo" exp.mode="predict" model.saved_model_folder="train1" exp.name="train1_predict_coco8_val" path.predict_image_folder="data/processed/coco8/val/images"
```



## 🛠️ Adding a New Model

To add a new model, follow these steps:

1. Create a new script in the `models/` folder, e.g., `models/my_model.py`.
2. Implement three functions:
   - `build_model(cfg)`: Loads and returns the model.
   - `train_model(model, cfg)`: Implements training logic.
   - `evaluate_model(model, cfg)`: Implements evaluation logic.
   - `predict(model, cfg)`: Implements prediction logic.
3. Create a configuration file under `configs/models/my_model.yaml`.

After that, you can run:

```bash
python src/main.py model="my_model" exp.mode="train" data.folder="coco8" exp.name="my_model_with_coco8"
```




## 📊 Supported Models

| Model         | Source          | Framework                        |
|--------------|----------------|----------------------------------|
| YOLO         | Ultralytics     | PyTorch                          |
| Faster R-CNN | Torchvision     | PyTorch                          |
| DETR         | GitHub Repo     | PyTorch                          |


---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [DETR](https://huggingface.co/facebook/detr-resnet-50)

---

## Contact

For any inquiries or suggestions, please contact [dipayan1109033@gmail.com].


