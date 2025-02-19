
import os
import json
import yaml
import shutil
import torch
from sklearn.model_selection import KFold, ShuffleSplit

from utils.common_utils import Helper
helper = Helper()



# Collate image-target pairs into a tuple.
def collate_fn(batch):
    return tuple(zip(*batch))


# Dataset information class
class DatasetInfo:
    def __init__(self, dataset_path):
        """
        Initialize the DatasetInfo class by reading the dataset information from a JSON file.

        Args:
        - dataset_path: Dataset folder path
        """
        self.dataset_path = dataset_path
        self.json_filepath = os.path.join(dataset_path, "dataset_info.json")

        self.label_dict = {}
        self.reverse_label_dict = {}
        self.imagename_to_id = {}
        self.id_to_imagename = {}

        # Load dataset info from JSON file and store in class variables
        self._load_dataset_info()

    def _load_dataset_info(self):
        """
        Load the dataset info from a JSON file and populate class variables.
        """
        # Check if the dataset info file exists
        if not os.path.exists(self.json_filepath):
            raise FileNotFoundError(f"Dataset info file not found at: {self.json_filepath}")

        # Read the dataset info JSON file
        with open(self.json_filepath, 'r') as f:
            data = json.load(f)

        # Read the label dictionary and image name-to-id mappings
        self.label_dict = data['categories']
        self.label_dict_yolo = {name: id-1 for name, id in data['categories'].items()}
        self.imagename_to_id = data['image_id_map']

        # Reverse mappings for labels and image IDs
        self.reverse_label_dict = {id: name for name, id in data['categories'].items()}
        self.reverse_label_dict_yolo = {id-1: name for name, id in data['categories'].items()}
        self.id_to_imagename = {id: name for name, id in data['image_id_map'].items()}

    def get_label_dict(self):
        """Return the label dictionary (categories)."""
        return self.label_dict

    def get_label_dict_yolo(self):
        """Return the label dictionary (categories) for yolo."""
        return self.label_dict_yolo
    
    def get_reverse_label_dict(self):
        """Return the reverse label dictionary (id to label)."""
        return self.reverse_label_dict

    def get_reverse_label_dict_yolo(self):
        """Return the reverse label dictionary (id to label) for yolo."""
        return self.reverse_label_dict_yolo
    
    def get_imagename_to_id(self):
        """Return the image name to id mapping."""
        return self.imagename_to_id

    def get_id_to_imagename(self):
        """Return the image id to name mapping."""
        return self.id_to_imagename



# Following functions are used for YOLO formatted labels

def convert_to_yolo(json_filepath, output_path):
    """
    Converts a JSON file with bounding box annotations to YOLO format.

    Args:
        dataInfo: An object of DatasetInfo class
        json_filepath: Path to the JSON file containing annotations.
        output_path: Folder path to save the YOLO formatted output.
        singleClass: Flag indicating whether to use a single class for all objects.
    """

    json_filename = os.path.basename(json_filepath)
    yolo_txt_filepath = os.path.join(output_path, json_filename[:-4] + "txt")

    # Load JSON data
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    # Extract image dimensions
    image_width = data['asset']['size']['width']
    image_height = data['asset']['size']['height']

    # Write YOLO format data to the output file
    with open(yolo_txt_filepath, 'w') as txt_file:
        for obj in data['objects']:

            class_label_id = obj['class'] - 1      # Yolo class_label_id start from 0 (zero)
            bbox = obj['boundingBox']

            # Calculate normalized center coordinates
            x_center = (bbox['xmin'] + bbox['width'] / 2) / image_width
            y_center = (bbox['ymin'] + bbox['height'] / 2) / image_height

            # Calculate normalized width and height
            width = bbox['width'] / image_width
            height = bbox['height'] / image_height

            # Write object information in YOLO format
            txt_file.write(f"{class_label_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def convert_custom_labels_yolo_labels(custom_labels_path, yolo_labels_path):
    json_filelist = helper.get_files_with_extension(custom_labels_path, extension=".json")
    for filename in json_filelist:
        json_filepath = os.path.join(custom_labels_path, filename)
        convert_to_yolo(json_filepath, yolo_labels_path)



# Following functions are used for metadata creation (json and yaml files)

def create_yaml_from_dataset_info(temp_dataset_dir, dataset_info, save_dir, test_split=True, fold_name=""):    
    reverse_label_dict = {(class_id-1): class_name for class_name, class_id in dataset_info["categories"].items()}

    # Create a dictionary to store the dataset information
    data = {
        "path": temp_dataset_dir,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images" if test_split else None,

        # Classes
        "names": reverse_label_dict
    }

    # Write data to the YAML file
    yaml_filepath = os.path.join(save_dir, f"{fold_name}yolo_dataset.yaml")
    helper.write_to_yaml(data, yaml_filepath)


def create_dataset_metadata_and_yaml(src_dataset_dir, temp_dataset_dir):
    dataset_info_filepath = os.path.join(src_dataset_dir, "dataset_info.json")
    dataset_info = helper.read_from_json(dataset_info_filepath)

    shutil.copy(dataset_info_filepath, temp_dataset_dir)
    create_yaml_from_dataset_info(temp_dataset_dir, dataset_info, save_dir=temp_dataset_dir)

def create_custom_dataset_metadata_and_yaml(metadata, datasets_root_dir, temp_dataset_dir):
    categories = {}
    image_id_mapping = {}

    for dataset_name in metadata["datasets"]:
        dataset_info_path = os.path.join(datasets_root_dir, dataset_name, "dataset_info.json")
        dataset_info = helper.read_from_json(dataset_info_path)
        categories.update(dataset_info["categories"])

        for split_type in ["train", "val", "test"]:
            image_files = metadata["split_data"][dataset_name][split_type]
            for image_file in image_files:
                image_id_mapping[image_file] = dataset_info["image_id_map"][image_file]

    composite_dataset_info = {
        "categories": categories,
        "image_id_map": image_id_mapping
    }
    save_filepath = os.path.join(temp_dataset_dir, "dataset_info.json")
    helper.write_to_json(composite_dataset_info, save_filepath)
    create_yaml_from_dataset_info(temp_dataset_dir, composite_dataset_info, save_dir=temp_dataset_dir)

def create_cv_dataset_metadata_and_yaml(src_dataset_dir, temp_dataset_dir, num_folds):
    dataset_info_filepath = os.path.join(src_dataset_dir, "dataset_info.json")
    dataset_info = helper.read_from_json(dataset_info_filepath)

    shutil.copy(dataset_info_filepath, temp_dataset_dir)

    for fold_idx in range(num_folds):
        temp_dataset_fold_dir = os.path.join(temp_dataset_dir, f"fold{fold_idx}")
        create_yaml_from_dataset_info(temp_dataset_fold_dir, dataset_info, save_dir=temp_dataset_dir, test_split=False, fold_name=f"fold{fold_idx}_")

def create_custom_cv_dataset_metadata_and_yaml(metadata, datasets_root_dir, temp_dataset_dir, num_folds):
    categories = {}
    image_id_mapping = {}

    for dataset_name, dataset in metadata["datasets"].items():
        dataset_info_path = os.path.join(datasets_root_dir, dataset_name, "dataset_info.json")
        dataset_info = helper.read_from_json(dataset_info_path)
        categories.update(dataset_info["categories"])

        if dataset.get("cross_val", False):  # Handle cross-validation folds
            for fold_idx in range(num_folds):
                fold_name = f"fold{fold_idx}"
                for split_type in ["train", "val"]:
                    image_files = metadata["split_data"][dataset_name][fold_name][split_type]
                    for image_file in image_files:
                        image_id_mapping[image_file] = dataset_info["image_id_map"][image_file]
        else:
            for split_type in ["train", "val"]:
                image_files = metadata["split_data"][dataset_name][split_type]
                for image_file in image_files:
                    image_id_mapping[image_file] = dataset_info["image_id_map"][image_file]

    composite_dataset_info = {
        "categories": categories,
        "image_id_map": image_id_mapping
    }
    save_filepath = os.path.join(temp_dataset_dir, "dataset_info.json")
    helper.write_to_json(composite_dataset_info, save_filepath)

    for fold_idx in range(num_folds):
        temp_dataset_fold_dir = os.path.join(temp_dataset_dir, f"fold{fold_idx}")
        create_yaml_from_dataset_info(temp_dataset_fold_dir, composite_dataset_info, save_dir=temp_dataset_dir, test_split=False, fold_name=f"fold{fold_idx}_")


# Following functions are used for dataset creation based on split ratios

def partition_dataset_by_ratio(src_dataset_dir, temp_dataset_dir, split_ratios=[0.8, 0.2, 0.0], seed=42):
    """
    Splits the dataset into train, val, and test based on given split ratios.

    Args:
        src_dataset_dir (str): Path to the source dataset directory.
        temp_dataset_dir (str): Path to the temporary directory where split datasets will be stored.
        split_ratios (list): List of three floats representing the train, val, test split ratios. Should sum to 1.0.
    """
    if os.path.exists(temp_dataset_dir):
        return
    else:
        os.makedirs(temp_dataset_dir, exist_ok=True)

    # Get paths to images and labels subdirectories
    src_images_folder, src_labels_folder = helper.get_subfolder_paths(src_dataset_dir)

    # Get list of images and labels
    images_list = sorted(helper.get_image_files(src_images_folder))
    labels_list = sorted(helper.get_files_with_extension(src_labels_folder, extension=".json"))

    # Ensure that image and label files match
    assert len(split_ratios) == 3, "Split ratio must be a list of three elements: [train_ratio, val_ratio, test_ratio]."
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0."
    assert all([os.path.splitext(img)[0] == os.path.splitext(lbl)[0] for img, lbl in zip(images_list, labels_list)]), "Image and label filenames don't match"

    # Split into train, val, and test
    total = len(images_list)
    val_len = int(split_ratios[1] * total)
    test_len = int(split_ratios[2] * total)
    train_len = total - val_len - test_len  # Remaining for train set

    # Set the random seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_images, val_images, test_images = torch.utils.data.random_split(images_list, [train_len, val_len, test_len], generator=generator)
    generator = torch.Generator().manual_seed(seed)
    train_labels, val_labels, test_labels = torch.utils.data.random_split(labels_list, [train_len, val_len, test_len], generator=generator)

    # Create subdirectories for train, val, and test
    temp_train_dir, temp_val_dir, temp_test_dir = helper.create_subfolders(temp_dataset_dir, folder_list=['train', 'val', 'test'])
    tt_images_folder, tt_labels_folder, tt_clabels_folder = helper.create_subfolders(temp_train_dir, folder_list=['images', 'labels', 'custom_labels'])
    tv_images_folder, tv_labels_folder, tv_clabels_folder = helper.create_subfolders(temp_val_dir, folder_list=['images', 'labels', 'custom_labels'])
    ts_images_folder, ts_labels_folder, ts_clabels_folder = helper.create_subfolders(temp_test_dir, folder_list=['images', 'labels', 'custom_labels'])

    # Transfer training images and labels
    helper.copy_specified_files(src_images_folder, tt_images_folder, file_list=train_images)
    helper.copy_specified_files(src_labels_folder, tt_clabels_folder, file_list=train_labels)
    convert_custom_labels_yolo_labels(tt_clabels_folder, tt_labels_folder)

    # Transfer validation images and labels
    helper.copy_specified_files(src_images_folder, tv_images_folder, file_list=val_images)
    helper.copy_specified_files(src_labels_folder, tv_clabels_folder, file_list=val_labels)
    convert_custom_labels_yolo_labels(tv_clabels_folder, tv_labels_folder)

    # Transfer test images and labels
    helper.copy_specified_files(src_images_folder, ts_images_folder, file_list=test_images)
    helper.copy_specified_files(src_labels_folder, ts_clabels_folder, file_list=test_labels)
    convert_custom_labels_yolo_labels(ts_clabels_folder, ts_labels_folder)

    create_dataset_metadata_and_yaml(src_dataset_dir, temp_dataset_dir)
    print("Dataset setup completed via splits by ratio.")

def create_experimental_dataset_from_metadata(datasets_root_dir, temp_dataset_dir, custom_splits_dir, split_code):
    """Creates a dataset using the information in the provided metadata.
    
    Args:
        datasets_root_dir (str): Root directory containing dataset folders.
        temp_dataset_dir (str): Path to the temporary directory where split datasets will be stored.
        custom_splits_dir (str): Path to the directory containing metadata JSON files.
        split_code (str): Specific split code to identify the metadata JSON filename.
    """
    if os.path.exists(temp_dataset_dir):
        print(f"Temporary dataset directory '{temp_dataset_dir}' already exists. Skipping dataset creation.")
        return
    else:
        os.makedirs(temp_dataset_dir, exist_ok=True)

    custom_splits_filepath = os.path.join(custom_splits_dir, f"{split_code}.json")
    metadata = helper.read_from_json(custom_splits_filepath)

    split_data = metadata["split_data"]
    
    for split_type in ["train", "val", "test"]:
        split_folder = os.path.join(temp_dataset_dir, split_type)
        split_image_folder, split_label_folder, split_clabel_folder = helper.create_subfolders(split_folder, folder_list=['images', 'labels', 'custom_labels'])
        for dataset_name in metadata["datasets"]:
            src_dataset_dir = os.path.join(datasets_root_dir, dataset_name)
            label_folder_name = metadata["datasets"][dataset_name]["label_folder"]
            images_src_path, labels_src_path = helper.get_subfolder_paths(src_dataset_dir, folder_list=['images', label_folder_name])

            image_files = split_data[dataset_name][split_type]
            label_files = [os.path.splitext(file)[0] + ".json" for file in image_files]
            
            helper.copy_specified_files(images_src_path, split_image_folder, file_list=image_files)
            helper.copy_specified_files(labels_src_path, split_clabel_folder, file_list=label_files)
        
        convert_custom_labels_yolo_labels(split_clabel_folder, split_label_folder)
    
    create_custom_dataset_metadata_and_yaml(metadata, datasets_root_dir, temp_dataset_dir)
    print(f"Custom Dataset successfully created at {temp_dataset_dir}")


# Following functions are used for cross-validation dataset creation

def cv_partition_without_replacement(src_dataset_dir, temp_dataset_dir, num_folds=3, seed=42):
    if os.path.exists(temp_dataset_dir):
        return
    else:
        os.makedirs(temp_dataset_dir, exist_ok=True)

    # Get paths to images and labels subdirectories
    src_images_folder, src_labels_folder = helper.get_subfolder_paths(src_dataset_dir)

    # Get list of images and labels
    images_list = sorted(helper.get_image_files(src_images_folder))
    labels_list = sorted(helper.get_files_with_extension(src_labels_folder, extension=".json"))

    # Ensure that image and label files match
    assert all([os.path.splitext(img)[0] == os.path.splitext(lbl)[0] for img, lbl in zip(images_list, labels_list)]), "Image and label filenames don't match"


    # Create K-fold cross-validation splits
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # Iterate over each fold
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(images_list)):
        temp_dataset_fold_dir = os.path.join(temp_dataset_dir, f"fold{fold_idx}")
        temp_train_dir, temp_val_dir = helper.create_subfolders(temp_dataset_fold_dir, folder_list=['train', 'val'])
        tt_images_folder, tt_labels_folder, tt_clabels_folder = helper.create_subfolders(temp_train_dir, folder_list=['images', 'labels', 'custom_labels'])
        tv_images_folder, tv_labels_folder, tv_clabels_folder = helper.create_subfolders(temp_val_dir, folder_list=['images', 'labels', 'custom_labels'])

        # Copy the images and labels for the current fold's training set
        train_images = [images_list[i] for i in train_idx]
        train_labels = [labels_list[i] for i in train_idx]
        helper.copy_specified_files(src_images_folder, tt_images_folder, file_list=train_images)
        helper.copy_specified_files(src_labels_folder, tt_clabels_folder, file_list=train_labels)
        convert_custom_labels_yolo_labels(tt_clabels_folder, tt_labels_folder)

        # Copy the images and labels for the current fold's val set
        val_images = [images_list[i] for i in val_idx]
        val_labels = [labels_list[i] for i in val_idx]
        helper.copy_specified_files(src_images_folder, tv_images_folder, file_list=val_images)
        helper.copy_specified_files(src_labels_folder, tv_clabels_folder, file_list=val_labels)
        convert_custom_labels_yolo_labels(tv_clabels_folder, tv_labels_folder)

    # Create dataset info and yolo YAML file
    create_cv_dataset_metadata_and_yaml(src_dataset_dir, temp_dataset_dir, num_folds)
    print("Dataset setup completed for cross validation (without replacement).")

def cv_partition_with_replacement(src_dataset_dir, temp_dataset_dir, split_ratios=[0.8, 0.2, 0.0], num_folds=3, seed=42):
    if os.path.exists(temp_dataset_dir):
        return
    else:
        os.makedirs(temp_dataset_dir, exist_ok=True)

    # Get paths to images and labels subdirectories
    src_images_folder, src_labels_folder = helper.get_subfolder_paths(src_dataset_dir)

    # Get list of images and labels
    images_list = sorted(helper.get_image_files(src_images_folder))
    labels_list = sorted(helper.get_files_with_extension(src_labels_folder, extension=".json"))

    # Ensure that image and label files match
    assert len(split_ratios) == 3, "Split ratio must be a list of three elements: [train_ratio, val_ratio, test_ratio]."
    assert sum(split_ratios[:2]) == 1.0, "Split ratios [train_ratio, val_ratio] must sum to 1.0."
    assert all([os.path.splitext(img)[0] == os.path.splitext(lbl)[0] for img, lbl in zip(images_list, labels_list)]), "Image and label filenames don't match"


    # ShuffleSplit for cross-validation with replacement
    train_ratio, val_ratio, _ = split_ratios
    shuffle_split = ShuffleSplit(n_splits=num_folds, train_size=train_ratio, test_size=val_ratio, random_state=seed)

    # Create each fold
    for fold_idx, (train_idx, val_idx) in enumerate(shuffle_split.split(images_list)):
        temp_dataset_fold_dir = os.path.join(temp_dataset_dir, f"fold{fold_idx}")
        temp_train_dir, temp_val_dir = helper.create_subfolders(temp_dataset_fold_dir, folder_list=['train', 'val'])
        tt_images_folder, tt_labels_folder, tt_clabels_folder = helper.create_subfolders(temp_train_dir, folder_list=['images', 'labels', 'custom_labels'])
        tv_images_folder, tv_labels_folder, tv_clabels_folder = helper.create_subfolders(temp_val_dir, folder_list=['images', 'labels', 'custom_labels'])

        # Copy the training set for this fold
        train_images = [images_list[i] for i in train_idx]
        train_labels = [labels_list[i] for i in train_idx]
        helper.copy_specified_files(src_images_folder, tt_images_folder, file_list=train_images)
        helper.copy_specified_files(src_labels_folder, tt_clabels_folder, file_list=train_labels)
        convert_custom_labels_yolo_labels(tt_clabels_folder, tt_labels_folder)

        # Copy the validation set for this fold
        val_images = [images_list[i] for i in val_idx]
        val_labels = [labels_list[i] for i in val_idx]
        helper.copy_specified_files(src_images_folder, tv_images_folder, file_list=val_images)
        helper.copy_specified_files(src_labels_folder, tv_clabels_folder, file_list=val_labels)
        convert_custom_labels_yolo_labels(tv_clabels_folder, tv_labels_folder)

    # Create dataset info and yolo YAML file
    create_cv_dataset_metadata_and_yaml(src_dataset_dir, temp_dataset_dir, num_folds)
    print("Dataset setup completed for cross validation (with replacement).")

def create_experimental_cv_dataset_from_metadata(datasets_root_dir, temp_dataset_dir, custom_splits_dir, split_code):
    """Creates a cross-validation dataset using the information in the provided metadata.
    
    Args:
        datasets_root_dir (str): Root directory containing dataset folders.
        temp_dataset_dir (str): Path to the temporary directory where split datasets will be stored.
        custom_splits_dir (str): Path to the directory containing metadata JSON files.
        split_code (str): Specific split code to identify the metadata JSON filename.
    """
    if os.path.exists(temp_dataset_dir):
        print(f"Temporary dataset directory '{temp_dataset_dir}' already exists. Skipping dataset creation.")
        return
    else:
        os.makedirs(temp_dataset_dir, exist_ok=True)

    custom_splits_filepath = os.path.join(custom_splits_dir, f"{split_code}.json")
    metadata = helper.read_from_json(custom_splits_filepath)

    num_folds = metadata["num_folds"]
    split_data = metadata["split_data"]

    for fold_idx in range(num_folds):
        fold_name = f"fold{fold_idx}"
        fold_temp_dir = os.path.join(temp_dataset_dir, fold_name)

        for split_type in ["train", "val"]:
            split_folder = os.path.join(fold_temp_dir, split_type)
            split_image_folder, split_label_folder, split_clabel_folder = helper.create_subfolders(
                split_folder, folder_list=['images', 'labels', 'custom_labels'])

            for dataset_name, dataset in metadata["datasets"].items():
                src_dataset_dir = os.path.join(datasets_root_dir, dataset_name)
                label_folder_name = dataset["label_folder"]
                images_src_path, labels_src_path = helper.get_subfolder_paths(src_dataset_dir, folder_list=['images', label_folder_name])

                if dataset.get("cross_val", False):  # Handle cross-validation folds
                    image_files = split_data[dataset_name][fold_name][split_type]
                else:
                    image_files = split_data[dataset_name][split_type]

                label_files = [os.path.splitext(file)[0] + ".json" for file in image_files]

                helper.copy_specified_files(images_src_path, split_image_folder, file_list=image_files)
                helper.copy_specified_files(labels_src_path, split_clabel_folder, file_list=label_files)

            convert_custom_labels_yolo_labels(split_clabel_folder, split_label_folder)

    create_custom_cv_dataset_metadata_and_yaml(metadata, datasets_root_dir, temp_dataset_dir, num_folds)
    print(f"Custom CV Dataset successfully created at {temp_dataset_dir}")



# Following functions are used for dataset path retrieval
def get_temp_dataset_path(cfg, dataset_folder, mode="train", num_folds=3, use_replacement=False, split_ratios=[0.8, 0.2, 0.0], split_code=None, seed=42):
    # Define the temporary directory for datasets
    temp_datasets_root_dir = os.path.join(cfg.path.project_root_dir, cfg.path.input_dir, "temp_datasets")

    # Choose data partitions
    if cfg.exp.mode == "crossval":      # For cross-val experiement
        if cfg.data.split_code:              # Partition with provided custom splits file
            temp_dataset_path = os.path.join(temp_datasets_root_dir, f"{cfg.data.split_code}")
        elif cfg.data.use_replacement:       # with replacement
            train_split, val_split, test_split = [int(split * 100) for split in cfg.data.split_ratios]
            temp_dataset_path = os.path.join(temp_datasets_root_dir, f"{dataset_folder}_{train_split}_{val_split}_{test_split}_cv{cfg.data.num_folds}")
        else:                               # without replacement
            temp_dataset_path = os.path.join(temp_datasets_root_dir, f"{dataset_folder}_cv{cfg.data.num_folds}_seed{cfg.exp.seed}")
    else:
        if cfg.data.split_code:             # Partition with provided custom splits file
            temp_dataset_path = os.path.join(temp_datasets_root_dir, f"{cfg.data.split_code}")
        else:                               # Partition with splits percentage ratios
            train_split, val_split, test_split = [int(split * 100) for split in cfg.data.split_ratios]
            temp_dataset_path = os.path.join(temp_datasets_root_dir, f"{dataset_folder}_{train_split}_{val_split}_{test_split}_seed{cfg.exp.seed}")

    return temp_dataset_path