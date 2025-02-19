
import os
import yaml
import json
from PIL import Image


# Function to truncate float values without rounding
def truncate_float(value, decimals=4):
    """
    Truncates a float to a specified number of decimal places without rounding.
    
    :param value: The float value to be truncated
    :param decimals: Number of decimal places to keep
    :return: Truncated float value
    """
    factor = 10.0 ** decimals
    return int(value * factor) / factor

# Function to generate unique image ID ensuring no duplicates
def generate_unique_image_id(image_name, existing_ids):
    """
    Generates a unique, positive image ID using Python's built-in hash function.
    Ensures no duplicate IDs by keeping track of generated values.

    :param image_name: Name of the image file
    :param existing_ids: Set containing already assigned image IDs to avoid duplicates
    :return: Unique positive image ID
    """
    image_id = abs(hash(image_name)) % (10**9)  # 9-digit positive ID
    
    while image_id in existing_ids:
        image_id = abs(hash(image_name + str(image_id))) % (10**9)
    
    existing_ids.add(image_id)
    return image_id

# Function to convert a single YOLO label file to a custom JSON format
def yolo_label_to_json(image_path, yolo_label_path, image_id):
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
def convert_yolo_dataset_labels(yaml_path):
    """
    Converts all YOLO labels in a dataset to a custom JSON format, processing each image-label pair.
    
    :param yaml_path: Path to the dataset YAML file defining dataset structure and class mappings
    """
    # Load dataset configuration from YAML file
    with open(yaml_path, 'r') as file:
        dataset_info = yaml.safe_load(file)
    
    dataset_root = os.path.abspath(os.path.dirname(yaml_path))
    class_map = dataset_info['names']
    image_id_map = {}
    existing_ids = set()

    # Process train, validation, and test sets if available
    for split in ['train', 'val', 'test']:
        if split not in dataset_info or not dataset_info[split]:
            continue
        print(f"Processing {split} split...")
        
        images_dir = os.path.join(dataset_root, dataset_info[split])
        labels_dir = images_dir.replace('images', 'labels')
        custom_labels_dir = labels_dir.replace('labels', 'custom_labels')
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
                json_data = yolo_label_to_json(image_path, label_file, image_id)
                output_file = os.path.join(custom_labels_dir, os.path.splitext(image_name)[0] + ".json")
                
                # Save JSON label file
                with open(output_file, 'w') as json_file:
                    json.dump(json_data, json_file, indent=4)
    
    # Save dataset metadata with class mappings and image IDs
    dataset_info_file = os.path.join(dataset_root, "dataset_info.json")
    categories = {class_name: int(class_id) + 1 for class_id, class_name in class_map.items()}
    with open(dataset_info_file, 'w') as json_file:
        json.dump({"categories": categories, "image_id_map": image_id_map}, json_file, indent=4)





def main():
    
    yaml_filepath = "data/coco8/coco8.yaml"
    #yaml_filepath = "data/coco128/coco128.yaml"
    #yaml_filepath = "data/VOC2007/VOC2007.yaml"

    #convert_yolo_dataset_labels(yaml_filepath)




if __name__ == "__main__":
    main()
