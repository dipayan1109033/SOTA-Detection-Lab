
import os
import cv2
import json

from src.utils.common_utils import Helper
helper = Helper()



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


def get_existing_splits(dataset_dir):
    """
    Returns a list of existing dataset split folders ('train', 'val', 'test') in the dataset directory.
    
    If none of these folders exist, returns None.

    Args:
        dataset_dir (str): Path to the dataset directory.

    Returns:
        list or None: List of existing split folder names ('train', 'val', 'test') if any exist, else None.
    """
    splits = ['train', 'val', 'test']
    existing_splits = [split for split in splits if os.path.isdir(os.path.join(dataset_dir, split))]
    return existing_splits if existing_splits else None




# Check if each label file has a corresponding image file and vice versa across all dataset splits
def check_label_image_correspondence(dataset_path, label_folder="custom_labels"):
    """
    Checks if each label file has a corresponding image file and vice versa across all dataset splits.

    Args:
        dataset_path (str): Path to the root dataset directory.
        label_folder (str): Name of the folder containing label files.

    Returns:
        None
    """
    # Get existing splits using get_existing_splits()
    existing_splits = get_existing_splits(dataset_path)

    if not existing_splits:
        print("No valid dataset splits found.")
        return

    for split in existing_splits:
        split_path = os.path.join(dataset_path, split)
        image_path = os.path.join(split_path, "images")
        label_path = os.path.join(split_path, label_folder)
        
        if not os.path.isdir(label_path) or not os.path.isdir(image_path):
            print(f"Missing folders in split '{split}'. Skipping...")
            continue

        # Get list of label files (.json) and image files
        label_files = helper.get_files_with_extension(label_path, extension=".json")
        image_files = helper.get_files_with_extensions(image_path, extensions=[".jpg", ".jpeg", ".png"])

        label_files_set = {os.path.splitext(f)[0] for f in label_files}
        image_files_set = {os.path.splitext(f)[0] for f in image_files}

        # Check for missing images for label files
        missing_images = label_files_set - image_files_set
        if missing_images:
            print(f"Missing images for the following label files in '{split}':")
            for missing_image in missing_images:
                print(f"  - {missing_image}")
        else:
            print(f"No missing images for labels in '{split}'.")

        # Check for missing labels for image files
        missing_labels = image_files_set - label_files_set
        if missing_labels:
            print(f"Missing labels for the following image files in '{split}':")
            for missing_label in missing_labels:
                print(f"  - {missing_label}")
        else:
            print(f"No missing labels for images in '{split}'.")


# Check for unique image IDs across all dataset splits, validate them against the image_id_map
def check_unique_image_ids_with_metadata(dataset_path, label_folder="custom_labels"):
    """
    Checks for unique image IDs across all dataset splits, validates them against the image_id_map
    in dataset_info.json, and reports duplicates or mismatches.

    Args:
        dataset_path (str): Path to the dataset directory.
        label_folder (str): Name of the label folder within each split.

    Returns:
        None
    """
    # Load dataset_info.json
    dataset_info_json = os.path.join(dataset_path, "dataset_info.json")
    if not os.path.exists(dataset_info_json):
        print(f"dataset_info.json not found in {dataset_path}")
        return

    dataset_info = helper.read_from_json(dataset_info_json)
    image_id_map = dataset_info.get("image_id_map", {})

    existing_splits = get_existing_splits(dataset_path)
    if not existing_splits:
        print("No valid dataset splits found.")
        return set()

    all_image_ids = set()
    duplicate_image_ids = set()
    missing_in_metadata = set()
    mismatched_ids = set()

    for split in existing_splits:
        split_path = os.path.join(dataset_path, split)
        labelfolder_path = os.path.join(split_path, label_folder)
        label_filelist = helper.get_files_with_extension(labelfolder_path, extension=".json")

        print(f"\nChecking image IDs in '{split}' split...")

        for json_file in label_filelist:
            json_filepath = os.path.join(labelfolder_path, json_file)
            data = helper.read_from_json(json_filepath)

            # Extract image ID from the JSON
            image_id = data.get("asset", {}).get("image_id", None)
            image_filename = data.get("asset", {}).get("name", None)

            if image_id and image_filename:
                # Check for duplicate image IDs
                if image_id in all_image_ids:
                    duplicate_image_ids.add(image_id)
                else:
                    all_image_ids.add(image_id)

                # Validate against image_id_map from metadata
                expected_id = image_id_map.get(image_filename)
                if expected_id is None:
                    missing_in_metadata.add(image_filename)
                elif expected_id != image_id:
                    mismatched_ids.add((image_filename, image_id, expected_id))
            else:
                print(f"Warning: Missing image ID or filename in {json_file}")

    print("\nSummary of Image ID Check Across All Splits:")
    print(f"Total Unique Image IDs: {len(all_image_ids)}")

    # Print Duplicate Image IDs
    if duplicate_image_ids:
        print(f"Duplicate Image IDs Found: {len(duplicate_image_ids)}")
        for dup_id in duplicate_image_ids:
            print(f"  - {dup_id}")
    else:
        print("Duplicate Image IDs Found: 0")

    # Print Images Missing in Metadata
    if missing_in_metadata:
        print(f"Images Missing in Metadata (image_id_map): {len(missing_in_metadata)}")
        for missing_file in missing_in_metadata:
            print(f"  - {missing_file}")
    else:
        print("Images Missing in Metadata (image_id_map): 0")

    # Print Mismatched Image IDs Between Labels and Metadata
    if mismatched_ids:
        print(f"Mismatched Image IDs Between Labels and Metadata: {len(mismatched_ids)}")
        for filename, found_id, expected_id in mismatched_ids:
            print(f"  - {filename}: Found ID = {found_id}, Expected ID = {expected_id}")
    else:
        print("Mismatched Image IDs Between Labels and Metadata: 0")


# Validate bounding box coordinates within a single dataset split
def validate_split_bbox_coordinates(split_path, label_folder="custom_labels", fix_flag=False):
    """
    Validates bounding box coordinates within a single dataset split.

    Args:
        split_path (str): Path to the dataset split (e.g., train, val, test).
        label_folder (str): Name of the label folder within the split.
        fix_flag (bool): If True, fixes invalid bounding box coordinates.

    Returns:
        int: Count of bounding boxes with coordinate issues.
    """
    labelfolder_path = os.path.join(split_path, label_folder)
    label_filelist = helper.get_files_with_extension(labelfolder_path, extension=".json")

    issue_count = 0

    for json_file in label_filelist:
        json_filepath = os.path.join(labelfolder_path, json_file)
        data = helper.read_from_json(json_filepath)

        image_width = data['asset']['size']['width']
        image_height = data['asset']['size']['height']

        for obj in data["objects"]:
            bbox = obj['boundingBox']
            xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']

            # Check if bounding box coordinates are out of bounds
            if xmin < 0.0 or ymin < 0.0 or xmax > image_width or ymax > image_height:
                print(f"Issue in {json_file} | Image size: ({image_width:.2f}, {image_height:.2f}) | "
                      f"BBox: ({xmin:.2f}, {ymin:.2f}) - ({xmax:.2f}, {ymax:.2f})")
                issue_count += 1

            # Fix coordinates if fix_flag is True
            if fix_flag:
                bbox['xmin'] = max(0.0, xmin)
                bbox['ymin'] = max(0.0, ymin)
                bbox['xmax'] = min(image_width, xmax)
                bbox['ymax'] = min(image_height, ymax)
                bbox['width'] = bbox['xmax'] - bbox['xmin']
                bbox['height'] = bbox['ymax'] - bbox['ymin']

                helper.write_to_json(data, json_filepath)

    print(f"Total bounding box issues in '{split_path}': {issue_count}")
    return issue_count

# Validate bounding box coordinates across all dataset splits
def validate_dataset_bbox_coordinates(dataset_path, label_folder="custom_labels", fix_flag=False):
    """
    Validates bounding box coordinates across all available dataset splits ('train', 'val', 'test').

    Args:
        dataset_path (str): Path to the dataset directory.
        label_folder (str): Name of the label folder within each split.
        fix_flag (bool): If True, fixes invalid bounding box coordinates.

    Returns:
        None
    """
    existing_splits = get_existing_splits(dataset_path)

    if not existing_splits:
        print("No valid dataset splits found.")
        return

    total_bbox_issues = 0

    for split in existing_splits:
        split_path = os.path.join(dataset_path, split)
        print(f"\nValidating bounding boxes in '{split}' split...")

        # Validate each split using the updated split-level function
        issue_count = validate_split_bbox_coordinates(split_path, label_folder=label_folder, fix_flag=fix_flag)
        total_bbox_issues += issue_count

    print(f"\nTotal bounding box issues across all splits: {total_bbox_issues}")



# Check for all unique class labels in a single split
def check_split_objects_classes(dataset_path, label_folder="custom_labels"):
    """
    Checks for all unique class labels in a single dataset split.

    Args:
        dataset_path (str): Path to the dataset split (e.g., train, val, test).
        label_folder (str): Name of the label folder within the split.

    Returns:
        set: Set of unique class IDs in the split.
    """
    labelfolder_path = os.path.join(dataset_path, label_folder)
    label_filelist = helper.get_files_with_extension(labelfolder_path, extension=".json")

    classes = set()
    for json_file in label_filelist:
        json_filepath = os.path.join(labelfolder_path, json_file)

        data = helper.read_from_json(json_filepath)
        for obj in data["objects"]:
            classes.add(obj["class"])

    print(f"Class IDs in '{dataset_path}': {classes}")
    return classes

# Check for all unique class labels across all splits and map them to names
def check_dataset_objects_classes(dataset_path, label_folder="custom_labels"):
    """
    Checks for all unique class labels across all dataset splits and maps them to class names.
    Also verifies if the class IDs are valid according to dataset_info.json.

    Args:
        dataset_path (str): Path to the dataset directory.
        label_folder (str): Name of the label folder within each split.

    Returns:
        dict: Dictionary mapping class IDs to class names for all detected classes.
    """
    # Load category mappings from dataset_info.json
    dataset_info_json = os.path.join(dataset_path, "dataset_info.json")
    if not os.path.exists(dataset_info_json):
        print(f"dataset_info.json not found in {dataset_path}")
        return

    dataset_info = helper.read_from_json(dataset_info_json)
    id_to_class_name = {v: k for k, v in dataset_info["categories"].items()}
    valid_class_ids = set(id_to_class_name.keys())

    existing_splits = get_existing_splits(dataset_path)

    if not existing_splits:
        print("No valid dataset splits found.")
        return

    all_classes = set()
    invalid_classes = set()

    for split in existing_splits:
        split_path = os.path.join(dataset_path, split)
        print(f"\nChecking classes in '{split}' split...")

        # Use the split-level function to get class IDs
        split_classes = check_split_objects_classes(split_path, label_folder=label_folder)
        all_classes.update(split_classes)

        # Check for invalid class IDs
        for class_id in split_classes:
            if class_id not in valid_class_ids:
                invalid_classes.add(class_id)

    # Map class IDs to class names
    class_id_name_map = {class_id: id_to_class_name.get(class_id, "Unknown") for class_id in all_classes}

    print("\nUnique Classes Across All Splits:")
    for class_id, class_name in class_id_name_map.items():
        print(f"Class ID: {class_id:3d}, Class Name: {class_name}")

    if invalid_classes:
        print("\nInvalid Class IDs Detected:")
        for invalid_id in invalid_classes:
            print(f"Invalid Class ID: {invalid_id}")
    else:
        print("\nNo invalid class IDs detected.")



# Draw bounding boxes over an image from the provided custom JSON label file
def draw_bboxes_on_image(image_path, json_label_path, id_to_class_name):
    """
    Plots bounding boxes over an image from the provided custom JSON label file.

    Args:
        image_path: Path to the image file.
        json_label_path: Path to the corresponding custom JSON label file.
        id_to_class_name: Dictionary mapping class IDs to class names.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Load JSON data
    with open(json_label_path, 'r') as file:
        json_data = json.load(file)

    # Loop through each object in the JSON
    for object_data in json_data["objects"]:
        # Extract bounding box information
        xmin = int(object_data["boundingBox"]["xmin"])
        ymin = int(object_data["boundingBox"]["ymin"])
        xmax = int(object_data["boundingBox"]["xmax"])
        ymax = int(object_data["boundingBox"]["ymax"])

        # Draw the bounding box on the image
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green for boxes

        # Optional: Add class label text (modify as needed)
        confidence_score = f" ({object_data['score']:.2f})" if "score" in object_data else ""
        class_label = f"{id_to_class_name[object_data['class']]} {confidence_score}"

        top = ymin - 10 if ymin > 30 else ymin + 25
        cv2.putText(image, class_label, (xmin + 5, top), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Red text

    return image

# Draw bounding boxes over all images across all dataset splits
def draw_bboxes_across_dataset_splits(dataset_path, label_folder="custom_labels"):
    """
    Draws bounding boxes over images for all splits in the dataset.

    Args:
        dataset_path: Path to the dataset directory.
        label_folder: Name of the folder containing label files within each split.

    Returns:
        None
    """
    # Load category mappings from dataset_info.json
    dataset_info_json = os.path.join(dataset_path, "dataset_info.json")
    if not os.path.exists(dataset_info_json):
        print(f"dataset_info.json not found in {dataset_path}")
        return
    dataset_info = helper.read_from_json(dataset_info_json)
    id_to_class_name = {v: k for k, v in dataset_info["categories"].items()}

    # Get existing splits using get_existing_splits()
    existing_splits = get_existing_splits(dataset_path)

    if not existing_splits:
        print("No valid dataset splits found.")
        return

    for split in existing_splits:
        split_path = os.path.join(dataset_path, split)
        imagefolder_path = os.path.join(split_path, "images")
        labelfolder_path = os.path.join(split_path, label_folder)
        output_folder_path = os.path.join(split_path, "images_withBBoxes")

        os.makedirs(output_folder_path, exist_ok=True)

        # Get list of image files
        image_list = helper.get_image_files(imagefolder_path)

        print(f"\nDrawing bounding boxes for split '{split}'...")

        for filename in image_list:
            image_path = os.path.join(imagefolder_path, filename)
            label_path = os.path.join(labelfolder_path, filename.rsplit(".", 1)[0] + ".json")

            if not os.path.exists(label_path):
                print(f"Missing label for image: {filename}. Skipping...")
                continue

            # Draw bounding boxes
            image_with_bboxes = draw_bboxes_on_image(image_path, label_path, id_to_class_name)

            # Save image with bounding boxes
            save_path = os.path.join(output_folder_path, filename)
            cv2.imwrite(save_path, image_with_bboxes)

        print(f"Saved images with bounding boxes to '{output_folder_path}'")





# Update image IDs for all images across dataset splits, starting from a specified ID
def update_image_ids_sequentially(dataset_path, start_image_id=1, label_folder="custom_labels"):
    """
    Updates image IDs for all images across dataset splits, starting from a specified ID, and updates dataset_info.json.

    Args:
        dataset_path (str): Path to the dataset directory.
        start_image_id (int): Starting image ID (default is 1).
        label_folder (str): Name of the label folder within each split.

    Returns:
        None
    """
    existing_splits = get_existing_splits(dataset_path)

    if not existing_splits:
        print("No valid dataset splits found.")
        return

    current_image_id = start_image_id
    updated_files_count = 0
    image_id_map = {}

    for split in existing_splits:
        split_path = os.path.join(dataset_path, split)
        labelfolder_path = os.path.join(split_path, label_folder)
        label_filelist = helper.get_files_with_extension(labelfolder_path, extension=".json")

        print(f"\nUpdating image IDs in '{split}' split...")

        for json_file in label_filelist:
            json_filepath = os.path.join(labelfolder_path, json_file)
            data = helper.read_from_json(json_filepath)

            # Update the image ID
            if "asset" in data:
                old_image_id = data["asset"]["image_id"]
                image_filename = data["asset"]["name"]
                data["asset"]["image_id"] = current_image_id

                # Save updated JSON
                helper.write_to_json(data, json_filepath)

                # Update image_id_map
                image_id_map[image_filename] = current_image_id

                #print(f"Updated {json_file}: {old_image_id} -> {current_image_id}")
                current_image_id += 1
                updated_files_count += 1
            else:
                print(f"Warning: No 'asset' field found in {json_file}")

    print(f"\nTotal label files updated: {updated_files_count}")
    print(f"New image IDs range from {start_image_id} to {current_image_id - 1}")

    # Update dataset_info.json
    dataset_info_json = os.path.join(dataset_path, "dataset_info.json")
    if os.path.exists(dataset_info_json):
        dataset_info = helper.read_from_json(dataset_info_json)
    else:
        print("No existing dataset_info.json found. Creating a new one.")
        dataset_info = {}

    # Update image_id_map in dataset_info
    dataset_info["image_id_map"] = image_id_map

    # Save updated dataset_info.json
    helper.write_to_json(dataset_info, dataset_info_json)

    print(f"Updated dataset_info.json with new image_id_map.")



# Update object class labels across all dataset splits based on the provided class mapping
def update_objects_classes_across_dataset(dataset_path, class_mapping=None, label_folder="custom_labels"):
    """
    Updates object class labels (using class IDs) across all dataset splits based on the provided class mapping,
    and updates the categories in dataset_info.json.

    Args:
        dataset_path (str): Path to the dataset directory.
        class_mapping (dict): Dictionary mapping old class names to new class names.
        label_folder (str): Name of the label folder within each split.

    Returns:
        None
    """
    if not class_mapping:
        raise ValueError("'class_mapping' is required to update class labels.")

    # Load dataset_info.json to get existing categories
    dataset_info_json = os.path.join(dataset_path, "dataset_info.json")
    if not os.path.exists(dataset_info_json):
        raise FileNotFoundError("dataset_info.json not found in the dataset directory.")

    with open(dataset_info_json, 'r') as f:
        dataset_info = json.load(f)

    old_categories = dataset_info.get("categories", {})

    # Create reverse mapping from class_id to class_name
    id_to_class_name = {v: k for k, v in old_categories.items()}

    # Create mapping from old class_id to new class_id
    new_class_names = set(class_mapping.values())
    new_categories = {name: idx+1 for idx, name in enumerate(new_class_names)}
    class_id_mapping = {}

    for old_id, old_class_name in id_to_class_name.items():
        if old_class_name in class_mapping:
            new_class_name = class_mapping[old_class_name]
            new_class_id = new_categories[new_class_name]
            class_id_mapping[old_id] = new_class_id
        else:
            raise ValueError(f"Class name '{old_class_name}' not found in class_mapping.")

    existing_splits = get_existing_splits(dataset_path)
    if not existing_splits:
        print("No valid dataset splits found.")
        return

    updated_files_count = 0
    unmapped_class_ids = set()

    for split in existing_splits:
        split_path = os.path.join(dataset_path, split)
        labelfolder_path = os.path.join(split_path, label_folder)
        label_filelist = helper.get_files_with_extension(labelfolder_path, extension=".json")

        print(f"\nUpdating class labels in '{split}' split...")

        for json_file in label_filelist:
            json_filepath = os.path.join(labelfolder_path, json_file)
            data = helper.read_from_json(json_filepath)

            updated = False

            for obj in data.get("objects", []):
                old_class_id = obj["class"]
                if old_class_id in class_id_mapping:
                    new_class_id = class_id_mapping[old_class_id]
                    obj["class"] = new_class_id
                    updated = True
                else:
                    unmapped_class_ids.add(old_class_id)

            if updated:
                helper.write_to_json(data, json_filepath)
                print(f"Updated class labels in {json_file}")
                updated_files_count += 1

    print(f"\nTotal label files updated: {updated_files_count}")

    if unmapped_class_ids:
        print("\nWarning: The following class IDs were not mapped and were not updated:")
        for unmapped_id in unmapped_class_ids:
            print(f"  - {unmapped_id}")
    else:
        print("All class IDs were successfully mapped and updated.")

    # Update dataset_info.json with new categories
    dataset_info["categories"] = new_categories

    with open(dataset_info_json, 'w') as f:
        json.dump(dataset_info, f, indent=4)

    print("\nUpdated dataset_info.json with new categories:")
    for class_name, class_id in new_categories.items():
        print(f"  - {class_name}: {class_id}")


