# 🚀 SOTA-Detection-Lab: A Flexible Object Detection Fine-Tuning Framework

**SOTA-Detection-Lab** is a **PyTorch-based** framework for training, fine-tuning, and evaluating **state-of-the-art object detection models** on **custom datasets**. It supports Faster R-CNN, YOLO, CenterNet, EfficientDet, DETR, and more, with **independent training scripts** and **Hydra-based configuration** to switch models effortlessly for fast experimentation.

## 📌 Key Features
- ✅ **Easily switch between multiple object detection models** (Faster R-CNN, YOLOv5, CenterNet, EfficientDet, DETR, etc.).
- ✅ **Independent training scripts** – no forced dependency on large frameworks like Detectron2 or MMDetection.
- ✅ **Flexible configuration using Hydra** – customize models, datasets, and hyperparameters easily.
- ✅ **Supports fine-tuning on custom datasets** with minimal setup.
- ✅ **Unified dataset handling** – supports COCO, Pascal VOC, and custom datasets.
- ✅ **Modular structure** – extend and add new models easily.

---

## 📂 Project Structure

```plaintext
SOTA-DETECTION-LAB/
│
├── configs/                           # Configuration files
│   ├── model/                         # Model-specific configs
│   │   ├── centernet.yaml
│   │   ├── detr.yaml
│   │   ├── faster_rcnn.yaml
│   │   └── yolo.yaml
│   ├── default_config.yaml            # Global/default configuration
│   └── experiment.yaml                # Experiment-specific configuration
│
├── data/                              # Datasets
│   ├── coco8/
│   ├── coco128/
│   └── VOC2007/
│
├── experiments/                       # Experiment data
│   ├── input/                         # Input files for experiments
│   └── output/                        # Output files (logs, results, checkpoints)
│
├── results/                           # Final results or evaluation outputs
│
├── src/                               # Source code
│   ├── data_prep/                     # Data preparation and preprocessing scripts
│   │   ├── convert_dataset_formats.py # Convert between dataset formats (e.g., COCO to Pascal VOC)
│   │   └── preprocess.py              # Preprocessing routines
│   │
│   ├── datasets/                      # Dataset processing utilities
│   │   ├── dataset_loader.py          # Unified dataset loader
│   │   └── transforms.py              # Data augmentation and preprocessing
│   │
│   ├── models/                        # Model implementations
│   │   ├── centernet.py               # CenterNet model
│   │   ├── faster_rcnn.py             # Faster R-CNN model
│   │   └── yolo.py                    # YOLO model
│   │
│   ├── utils/                         # Utility scripts
│   │   ├── common_utils.py            # General helper functions
│   │   ├── prepare_utils.py           # Raw dataset preprocess related utilities
│   │   ├── dataset_utils.py           # TorchVision Dataset, Dataloader-related utilities
│   │   ├── setup_utils.py             # Experimental Dataset Setup related
│   │   ├── train_utils.py             # Training utilities (loops, schedulers, etc.)
│   │   └── metrics_utils.py           # Evaluation, metrics computation, and reporting
│   │
│   └── main.py                        # Main entry point for running experiments
│
├── venv39/                            # Python virtual environment
│
├── .gitignore                         # Git ignore rules
├── LICENSE                            # License file
└── README.md                          # Project documentation

```



## 📖 Setup & Installation

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/dipayan1109033/SOTA-Detection-Lab.git
cd SOTA-Detection-Lab
```

### **2️⃣ Install Dependencies**

Ensure you have Python 3.9+ and PyTorch installed. Then install the required packages using the `requirements.txt` file:


```bash
pip install -r requirements.txt
```

---

## 🚀 Usage


### 🔹 Training a Model

To train a model, specify the model name and dataset:


```bash
python main.py task=train model=faster_rcnn dataset=coco training.epochs=30
```

Example for CenterNet:

```bash
python main.py task=train model=centernet dataset=custom training.epochs=50
```

### 🔹 Evaluating a Model
```bash
python main.py task=evaluate model=efficientdet dataset=pascal_voc
```




## 🛠️ Adding a New Model

To add a new model, follow these steps:

1. Create a new script in the `models/` folder, e.g., `models/my_model.py`.
2. Implement three functions:
   - `build_model(cfg)`: Loads and returns the model.
   - `train_model(model, dataloader, cfg)`: Implements training logic.
   - `evaluate_model(model, dataloader, cfg)`: Implements evaluation logic.
3. Create a configuration file under `configs/models/my_model.yaml`.

After that, you can run:

```sh
python main.py task=train model=my_model dataset=coco training.epochs=20
```




## 📊 Supported Models

| Model         | Source          | Framework                        |
|--------------|----------------|----------------------------------|
| Faster R-CNN | Torchvision     | PyTorch                          |
| YOLOv5       | Ultralytics     | PyTorch                          |
| YOLOv8       | Ultralytics     | PyTorch                          |
| CenterNet    | GitHub Repo     | PyTorch                          |
| EfficientDet | GitHub Repo     | TensorFlow (converted to PyTorch) |
| DETR         | GitHub Repo     | PyTorch                          |


---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- [DETR](https://github.com/facebookresearch/detectron2)
- [Albumentations](https://albumentations.ai/)

---

## Contact

For any inquiries or suggestions, please contact [dipayan1109033@gmail.com].

