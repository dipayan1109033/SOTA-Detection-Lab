
import os
import json
import yaml
import shutil
import torch
from sklearn.model_selection import KFold, ShuffleSplit

from src.utils.common import Helper
helper = Helper()



# Collate image-target pairs into a tuple.
def collate_fn(batch):
    return tuple(zip(*batch))


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

def update_yaml_file(src_dataset_dir, temp_dataset_dir, fold_name=None, test_split=False):
    dataset_folder = helper.get_immediate_folder_name(src_dataset_dir)
    dataset_yaml = os.path.join(src_dataset_dir, "dataset.yaml")

    # Read the YAML file
    with open(dataset_yaml, 'r') as file:
        data = yaml.safe_load(file)

    if not fold_name:
        dataset_dir = temp_dataset_dir
        yaml_filepath = os.path.join(temp_dataset_dir, f"{dataset_folder}.yaml")
        if test_split: data['test'] = "test/images"
    else:
        dataset_dir = os.path.join(temp_dataset_dir, fold_name)
        yaml_filepath = os.path.join(temp_dataset_dir, f"{fold_name}_{dataset_folder}.yaml")

    # Update the root dataset absolute directory
    data['path'] = dataset_dir

    # Write data to the YAML file
    with open(yaml_filepath, 'w') as file:
        yaml.dump(data, file)




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

    # Update and save the YAML config file
    update_yaml_file(src_dataset_dir, temp_dataset_dir, test_split=test_len>0)

    shutil.copy(os.path.join(src_dataset_dir, "dataset_info.json"), temp_dataset_dir)
    print("Dataset setup completed via splits by ratio.")

def partition_dataset_custom(src_dataset_dir, temp_dataset_dir, custom_splits_dir, split_code):
    if os.path.exists(temp_dataset_dir):
        return
    else:
        os.makedirs(temp_dataset_dir, exist_ok=True)

    # Get paths to images and labels subdirectories
    src_images_folder, src_labels_folder = helper.get_subfolder_paths(src_dataset_dir)

    # Read and extract dataset custom splits information
    dataset_folder = helper.get_immediate_folder_name(src_dataset_dir)
    custom_splits_filepath = os.path.join(custom_splits_dir, f"{dataset_folder}_{split_code}.json")
    metadata = helper.read_from_json(custom_splits_filepath)
    train_images = metadata['train_images']
    val_images = metadata['val_images']
    test_images = metadata['test_images']
    train_labels = [imagefile[:-4] + ".json" for imagefile in metadata['train_images']]
    val_labels = [imagefile[:-4] + ".json" for imagefile in metadata['val_images']]
    test_labels = [imagefile[:-4] + ".json" for imagefile in metadata['test_images']]


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

    # Update and save the YAML config file
    update_yaml_file(src_dataset_dir, temp_dataset_dir, test_split=len(test_images)>0)

    shutil.copy(os.path.join(src_dataset_dir, "dataset_info.json"), temp_dataset_dir)
    print("Dataset setup completed via custom splits.")

def cv_partition_custom(src_dataset_dir, temp_dataset_dir, num_folds, custom_splits_dir, split_code):
    if os.path.exists(temp_dataset_dir):
        return
    else:
        os.makedirs(temp_dataset_dir, exist_ok=True)

    # Get paths to images and labels subdirectories
    src_images_folder, src_labels_folder = helper.get_subfolder_paths(src_dataset_dir)

    # Read and extract dataset custom splits information
    dataset_folder = helper.get_immediate_folder_name(src_dataset_dir)
    custom_splits_filepath = os.path.join(custom_splits_dir, f"{dataset_folder}_cv{num_folds}_{split_code}.json")
    metadata = helper.read_from_json(custom_splits_filepath)
    print(custom_splits_filepath)
    print(metadata.keys())

    kfold = metadata["num_folds"]
    assert kfold == num_folds, "Number of folds doesn't match"

    # Iterate over each fold
    for fold_idx in range(num_folds):
        temp_dataset_fold_dir = os.path.join(temp_dataset_dir, f"fold{fold_idx}")
        temp_train_dir, temp_val_dir = helper.create_subfolders(temp_dataset_fold_dir, folder_list=['train', 'val'])
        tt_images_folder, tt_labels_folder, tt_clabels_folder = helper.create_subfolders(temp_train_dir, folder_list=['images', 'labels', 'custom_labels'])
        tv_images_folder, tv_labels_folder, tv_clabels_folder = helper.create_subfolders(temp_val_dir, folder_list=['images', 'labels', 'custom_labels'])

        # Copy the images and labels for the current fold's training set
        train_images = metadata[f'train_images_fold{fold_idx}']
        train_labels = [imagefile[:-4] + ".json" for imagefile in train_images]
        helper.copy_specified_files(src_images_folder, tt_images_folder, file_list=train_images)
        helper.copy_specified_files(src_labels_folder, tt_clabels_folder, file_list=train_labels)
        convert_custom_labels_yolo_labels(tt_clabels_folder, tt_labels_folder)

        # Copy the images and labels for the current fold's val set
        val_images = metadata[f'val_images_fold{fold_idx}']
        val_labels = [imagefile[:-4] + ".json" for imagefile in val_images]
        helper.copy_specified_files(src_images_folder, tv_images_folder, file_list=val_images)
        helper.copy_specified_files(src_labels_folder, tv_clabels_folder, file_list=val_labels)
        convert_custom_labels_yolo_labels(tv_clabels_folder, tv_labels_folder)

        # Update and save the YAML config file
        update_yaml_file(src_dataset_dir, temp_dataset_dir, fold_name=f"fold{fold_idx}")

    shutil.copy(os.path.join(src_dataset_dir, "dataset_info.json"), temp_dataset_dir)
    print("Dataset setup completed for cross validation (without replacement).")


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

        # Update and save the YAML config file
        update_yaml_file(src_dataset_dir, temp_dataset_dir, fold_name=f"fold{fold_idx}")

    shutil.copy(os.path.join(src_dataset_dir, "dataset_info.json"), temp_dataset_dir)
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

        # Update and save the YAML config file
        update_yaml_file(src_dataset_dir, temp_dataset_dir, fold_name=f"fold{fold_idx}")

    shutil.copy(os.path.join(src_dataset_dir, "dataset_info.json"), temp_dataset_dir)
    print("Dataset setup completed for cross validation (with replacement).")



def get_temp_dataset_path(cfg, dataset_folder, mode="train", num_folds=3, use_replacement=False, split_ratios=[0.8, 0.2, 0.0], split_code=None, seed=42):
    # Define the temporary directory for datasets
    temp_dataset_root_dir = os.path.join(cfg.path.project_root_dir, cfg.path.input_dir, "temp_datasets")

    # Choose data partitions
    if mode == "crossval":      # For cross-val experiement
        if use_replacement:             # with replacement
            train_split, val_split, test_split = [int(split * 100) for split in split_ratios]
            temp_dataset_path = os.path.join(temp_dataset_root_dir, f"{dataset_folder}_{train_split}_{val_split}_cv{num_folds}")
        else:                           # without replacement
            temp_dataset_path = os.path.join(temp_dataset_root_dir, f"{dataset_folder}_cv{num_folds}")
    elif split_code:            # Partition with provided custom splits file
        temp_dataset_path = os.path.join(temp_dataset_root_dir, f"{dataset_folder}_{split_code}")
    else:                        # Partition with splits percentage ratios
        train_split, val_split, test_split = [int(split * 100) for split in split_ratios]
        temp_dataset_path = os.path.join(temp_dataset_root_dir, f"{dataset_folder}_{train_split}_{val_split}_{test_split}_seed{seed}")

    return temp_dataset_path