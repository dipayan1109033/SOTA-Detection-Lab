import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import yaml
import json
import shutil
from PIL import Image
from src.utils.prepare_utils import *
from src.utils.common_utils import Helper
helper = Helper()


# Function to find the YAML file in the dataset directory
def find_yaml_in_directory(dataset_dir, debug=False):
    """Find the YAML file in the dataset directory."""
    yaml_files = [f for f in os.listdir(dataset_dir) if f.endswith('.yaml')]
    if not yaml_files:
        raise FileNotFoundError("No YAML file found in the dataset directory.")
    elif len(yaml_files) > 1:
        print("Multiple YAML files found in the dataset directory. Using the first file.")
    yaml_path = os.path.join(dataset_dir, yaml_files[0])
    if debug: print(f"Detected YAML file: {yaml_files[0]}")
    return yaml_path



# Function to update the YAML file for the new dataset structure
def update_yaml_for_new_structure(yaml_path, dest_dir, debug=False):
    """
    Updates the YOLO-style YAML file to reflect the new dataset structure.

    Args:
        yaml_path (str): Path to the original .yaml file.
        dest_dir (str): Path to the new dataset root directory after conversion.
    """
    # Load the existing YAML file
    data = helper.read_from_yaml(yaml_path)

    # Update the paths in the YAML file
    data['path'] = os.path.abspath(dest_dir)
    data['train'] = 'train/images'  # Updated to new structure
    data['val'] = 'val/images'      # Updated to new structure

    # Optional: Update test path if necessary
    if 'test' in data and data['test']:
        data['test'] = 'test/images'  # set to 'test/images' if a test set exists

    # Define the path to save the updated YAML
    updated_yaml_path = os.path.join(dest_dir, os.path.basename(yaml_path))

    # Save the updated YAML
    helper.write_to_yaml(data, updated_yaml_path)
    if debug: print(f"YAML file updated and saved to: {updated_yaml_path}")

# Function to convert COCO dataset structure to custom format
def convert_coco_structure(src_dir, dest_dir, debug=False):
    """
    Converts a COCO-style dataset structure to a new format where train/val/test
    splits are top-level directories containing separate 'images' and 'labels' folders.
    The function automatically detects the YOLO-style YAML file in src_dir.

    -------------------------
    Input Folder Structure:
    -------------------------
    src_dir/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/ (if present in YAML)
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/ (if present in YAML)
    └── coco8.yaml        # YOLO-format YAML file specifying dataset paths and classes

    -------------------------
    Output Folder Structure:
    -------------------------
    dest_dir/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/ (if present in YAML)
        ├── images/
        └── labels/

    -------------------------
    Requirements:
    -------------------------
    - The source directory must contain a YOLO-compatible `.yaml` file (e.g., coco8.yaml) 
      that specifies the paths to the 'train', 'val', and optional 'test' images.

    Args:
        src_dir (str): Path to the original COCO dataset (e.g., coco8/).
        dest_dir (str): Path to save the converted dataset.

    Raises:
        FileNotFoundError: If expected 'images' or 'labels' directories are missing.
    """

    # Automatically detect the YAML file
    yaml_path = find_yaml_in_directory(src_dir)
    yaml_data = helper.read_from_yaml(yaml_path)

    # Get splits from YAML
    splits = {}
    for split in ['train', 'val', 'test']:
        if yaml_data.get(split):
            splits[split] = yaml_data[split]

    if not splits:
        raise ValueError("YAML file does not contain valid 'train', 'val', or 'test' splits.")

    for split_name, split_path in splits.items():
        if debug: print(f"Processing {split_name} split...")

        # Create new directories for train, val, test
        split_dir = os.path.join(dest_dir, split_name)
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Resolve paths relative to src_dir
        src_images = os.path.join(src_dir, split_path)
        src_labels = src_images.replace('images', 'labels')

        if not os.path.exists(src_images):
            raise FileNotFoundError(f"Missing images directory: {src_images}")
        if not os.path.exists(src_labels):
            raise FileNotFoundError(f"Missing labels directory: {src_labels}")

        # Copy images
        for img_file in os.listdir(src_images):
            src_img_path = os.path.join(src_images, img_file)
            dest_img_path = os.path.join(images_dir, img_file)
            shutil.copy2(src_img_path, dest_img_path)

        # Copy labels
        for label_file in os.listdir(src_labels):
            src_label_path = os.path.join(src_labels, label_file)
            dest_label_path = os.path.join(labels_dir, label_file)
            shutil.copy2(src_label_path, dest_label_path)

        if debug: print(f"{split_name.capitalize()} split processed and saved to: {split_dir}")

    # Update the YAML file for the new structure
    update_yaml_for_new_structure(yaml_path, dest_dir)

    print(f"Dataset structure converted and saved to: {dest_dir}")



# Function to convert a single YOLO label file to a custom JSON format
def convert_a_yolo_label_to_custom_json(image_path, yolo_label_path, image_id):
    """
    Converts a single YOLO label file to a custom JSON format with bounding box details.
    
    :param image_path: Path to the corresponding image file
    :param yolo_label_path: Path to the YOLO label .txt file
    :param image_id: Unique image identifier
    :return: JSON object representing the image and its bounding boxes in the custom format
    """
    # Get image dimensions
    with Image.open(image_path) as img:
        image_width, image_height = img.size
    
    # Read YOLO label file
    with open(yolo_label_path, 'r') as file:
        lines = file.readlines()
    
    objects = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"Skipping malformed line: {line}")   
            continue  # Skip malformed lines
        
        class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)
        
        # Convert YOLO normalized coordinates to absolute coordinates
        xmin = truncate_float((x_center - bbox_width / 2) * image_width, 4)
        ymin = truncate_float((y_center - bbox_height / 2) * image_height, 4)
        xmax = truncate_float((x_center + bbox_width / 2) * image_width, 4)
        ymax = truncate_float((y_center + bbox_height / 2) * image_height, 4)
        
        objects.append({
            "class": int(class_id),
            "boundingBox": {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "width": truncate_float(xmax - xmin, 4),
                "height": truncate_float(ymax - ymin, 4)
            }
        })
    
    # Construct JSON output
    json_data = {
        "asset": {
            "name": os.path.basename(image_path),
            "image_id": image_id,
            "size": {
                "width": image_width,
                "height": image_height
            }
        },
        "objects": objects
    }
    
    return json_data

# Function to convert an entire YOLO dataset to a custom JSON format
def convert_yolo_dataset_labels(dataset_dir, class_map=None, debug=False):
    """
    Converts all YOLO labels in a dataset to a custom JSON format, processing each image-label pair.
    
    :param dataset_dir: Path to the restructured dataset directory.
    :param class_map: (Optional) A dictionary mapping class IDs to class names (like yolo-format). If not provided,
                      the mapping try to read from a YAML file in the dataset_dir.
    """

    # Use provided class_map if given; otherwise, read from YAML.
    if class_map is None:
        # Automatically find YAML file in dataset_dir
        yaml_path = find_yaml_in_directory(dataset_dir)
        yaml_data = helper.read_from_yaml(yaml_path)
        class_map = yaml_data.get('names', {})

    # Convert class mapping to desired format (incrementing each id by 1)
    categories = {class_name: int(class_id) + 1 for class_id, class_name in class_map.items()}

    image_id_map = {}
    existing_ids = set()

    # Determine splits based on folders in the dataset directory (train, val, test)
    splits = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d in ['train', 'val', 'test']]

    # Process each split
    for split in splits:
        if debug: print(f"Processing {split} split...")
        
        images_dir = os.path.join(dataset_dir, split, 'images')
        labels_dir = os.path.join(dataset_dir, split, 'labels')
        custom_labels_dir = os.path.join(dataset_dir, split, 'custom_labels')
        os.makedirs(custom_labels_dir, exist_ok=True)
        
        # Iterate over image files
        for image_name in os.listdir(images_dir):
            image_path = os.path.join(images_dir, image_name)
            if not os.path.isfile(image_path):
                print(f"Skipping {image_name} as it is not a file.")
                continue

            # Generate unique image ID
            image_id = generate_unique_image_id(image_name, existing_ids)
            image_id_map[image_name] = image_id
            
            # Locate corresponding YOLO label file
            label_file = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")
            if os.path.exists(label_file):
                json_data = convert_a_yolo_label_to_custom_json(image_path, label_file, image_id)
                output_file = os.path.join(custom_labels_dir, os.path.splitext(image_name)[0] + ".json")
                
                # Save JSON label file
                helper.write_to_json(json_data, output_file)
            else:
                print(f"No label found for image: {image_name}")

    # Save dataset metadata with class mappings and image IDs
    dataset_info_file = os.path.join(dataset_dir, "dataset_info.json")
    helper.write_to_json({"categories": categories, "image_id_map": image_id_map}, dataset_info_file)

    print(f"🎉 Dataset conversion completed. Metadata saved to {dataset_info_file}")


def copy_custom_dataset(src_dir, dest_dir):
    """Copy the custom labels with images to the processed data directory."""

    # Determine splits based on folders in the dataset directory (train, val, test)
    splits = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d)) and d in ['train', 'val', 'test']]

    for split in splits:
        images_dir = os.path.join(src_dir, split, 'images')
        labels_dir = os.path.join(src_dir, split, 'custom_labels')
        dest_images_dir = os.path.join(dest_dir, split, 'images')
        dest_labels_dir = os.path.join(dest_dir, split, 'custom_labels')

        # Copy images and labels
        shutil.copytree(images_dir, dest_images_dir, dirs_exist_ok=True)
        shutil.copytree(labels_dir, dest_labels_dir, dirs_exist_ok=True)

        # Copy the dataset info file
        shutil.copy2(os.path.join(src_dir, "dataset_info.json"), dest_dir)

    print(f"Dataset copied to: {dest_dir}")


def convert_yolo_custom_dataset(src_dataset_dir):
    """ Converts a YOLO-style dataset to a custom JSON format with bounding box details. """
    dataset_name = helper.get_immediate_folder_name(src_dataset_dir)
    raw_dataset_dir = os.path.join(raw_data_dir, f"{dataset_name}_reStructured")

    # Convert the dataset structure
    convert_coco_structure(src_dataset_dir, raw_dataset_dir, debug=True)

    # Convert YOLO labels to custom JSON format
    convert_yolo_dataset_labels(raw_dataset_dir, debug=True)

    # Copy the restructured dataset to the processed data directory
    dest_dataset_dir = os.path.join(processed_data_dir, dataset_name)
    copy_custom_dataset(raw_dataset_dir, dest_dataset_dir)





def check_dataset_consistency(dataset_dir, splits = ['train', 'val']):
    """
    Checks the consistency of a dataset that is structured into train and val splits,
    where each split has 'images' and 'labels' subdirectories.

    Expected folder structure:
    
        dataset_dir/
        ├── train/
        │     ├── images/
        │     └── labels/
        └── val/
              ├── images/
              └── labels/
    
    For each split, the function compares the base filenames (without extensions)
    in the images and labels folders and prints out the files that are missing in one
    folder but present in the other.

    Args:
        dataset_dir (str): Path to the dataset directory.
        splits (list): List of split names to check (default: ['train', 'val']).
    
    Returns:
        dict: A dictionary where keys are the split names ('train', 'val') and the values
              are tuples (missing_images, missing_labels). 'missing_images' is a set of base
              filenames that are in labels but missing in images, and 'missing_labels' is a set
              of base filenames that are in images but missing in labels.
    """
    results = {}

    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        images_folder = os.path.join(split_dir, "images")
        labels_folder = os.path.join(split_dir, "custom_labels")
        
        if not os.path.isdir(images_folder):
            raise FileNotFoundError(f"'images' folder not found in {split_dir}")
        if not os.path.isdir(labels_folder):
            raise FileNotFoundError(f"'labels' folder not found in {split_dir}")

        # Get base filenames (without extensions)
        image_names = {os.path.splitext(f)[0] for f in os.listdir(images_folder)
                       if os.path.isfile(os.path.join(images_folder, f))}
        label_names = {os.path.splitext(f)[0] for f in os.listdir(labels_folder)
                       if os.path.isfile(os.path.join(labels_folder, f))}
        
        # Determine mismatches:
        missing_images = label_names - image_names   # Files in labels missing in images
        missing_labels = image_names - label_names     # Files in images missing in labels
        
        # Print the results for this split
        print(f"--- {split.upper()} SPLIT ---")
        if missing_images:
            print("Files present in labels but missing in images:")
            for name in missing_images:
                print("  ", name)
        else:
            print("No missing images (all labels have corresponding images).")
        
        if missing_labels:
            print("Files present in images but missing in labels:")
            for name in missing_labels:
                print("  ", name)
        else:
            print("No missing labels (all images have corresponding labels).")
        
        print("\n")
        results[split] = (missing_images, missing_labels)
    
    return results



def main():

    # Define the dataset directory
    yolo_dataset_dir = "/Volumes/Works/Projects/datasets/coco128" # coco8, coco128
    #yolo_dataset_dir = "data/raw/VOC2007" 

    #convert_yolo_custom_dataset(yolo_dataset_dir)


    dataset_dir = "data/processed/coco128"
    check_dataset_consistency(dataset_dir)


if __name__ == "__main__":
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"

    main()
