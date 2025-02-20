
import os
import cv2
import json
import yaml
import shutil

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



# Create custom aanotation json file for missing VoTT json (without visual objects)
def create_empty_custom_json(image_path, output_dir, image_id):
    output_images_folder, output_labels_folder = helper.get_subfolder_paths(output_dir)
    image = cv2.imread(image_path)
    custom_json = {
        "asset": {
            "name": os.path.basename(image_path),
            "image_id": image_id,
            "size": {
                "width": image.shape[1],
                "height": image.shape[0]
            },
        },
            "objects": []
    }
    # Write to a new JSON file
    output_jsonfile_path = os.path.join(output_labels_folder, custom_json["asset"]["name"][:-4] + ".json")
    helper.write_to_json(custom_json, output_jsonfile_path)

    # Also copy the image
    shutil.copy(image_path, output_images_folder)

# Create custom aanotation json file for missing VoTT json (without visual objects)
def vott_to_json(json_file_path, image_folder, output_dir, image_id):
    output_images_folder, output_labels_folder = helper.get_subfolder_paths(output_dir)
      
    # Load the JSON data from the file
    data = helper.read_from_json(json_file_path)
    image_path = os.path.join(image_folder, data["asset"]["name"])

    # Extracting key-value pairs
    asset_info = {
        "name": data["asset"]["name"],
        "image_id": image_id,
        "size": data["asset"]["size"],
    }
    
    regions_info = []
    if len(data['regions']) == 0: print(f"-------------- No bbox for: {data['asset']['name']} --------------")
    for region in data["regions"]:      
        if (region["boundingBox"]["width"] <0.05 or region["boundingBox"]["height"] <0.05):
            print(f"Very small boxes found in {data['asset']['name']}")
            continue
        if len(region['tags']) > 1:
            print(f".............. Multiple classes ({len(region['tags'])}) found for a bbox in: {data['asset']['name']} ({os.path.basename(json_file_path)}) ..............")
        
        aBoundingBox = {
            "left": round(region["boundingBox"]["left"], 4),
            "top": round(region["boundingBox"]["top"], 4),
            "width": round(region["boundingBox"]["width"], 4),
            "height": round(region["boundingBox"]["height"], 4),
            "xmin": round(region["boundingBox"]["left"], 4),
            "ymin": round(region["boundingBox"]["top"], 4),
            "xmax": round(region["boundingBox"]["left"] + region["boundingBox"]["width"], 4),
            "ymax": round(region["boundingBox"]["top"] + region["boundingBox"]["height"], 4)
        }

        region_info = {
            "class": region["tags"][0],
            "boundingBox": aBoundingBox
        }
        regions_info.append(region_info)

    # Create a new dictionary to hold the extracted values
    custom_json = {
        "asset": asset_info,
        "objects": regions_info
    }

    # Write the extracted data to a new JSON file
    output_jsonfile_path = os.path.join(output_labels_folder, custom_json["asset"]["name"][:-4] + ".json")
    helper.write_to_json(custom_json, output_jsonfile_path)

    # Also copy the image
    shutil.copy(image_path, output_images_folder)

# Convert VoTT generated json label to custom json formatted label
def convert_VOTT_to_custom_JsonFormat(image_folder, annotation_folder, output_dir, start_image_id=1):
    output_images_folder, output_labels_folder = helper.create_subfolders(output_dir)

    # Convert with json annotation file
    json_files = helper.get_files_with_extension(annotation_folder, extension=".json")
    for ajson_file in json_files:
        input_json_file = os.path.join(annotation_folder, ajson_file)
        vott_to_json(input_json_file, image_folder, output_dir, image_id=start_image_id)
        start_image_id += 1

    # Image files without any annotated VoTT json file
    image_files_all = helper.get_files_with_extensions(image_folder, extensions= [".jpg", ".png"])
    image_files_with_json = helper.get_files_with_extensions(output_images_folder, extensions= [".jpg", ".png"])
    image_files_without_json = list(set(image_files_all) - set(image_files_with_json))

    # Convert without json annotation file 
    for image_file in image_files_without_json:
        src_image_path = os.path.join(image_folder, image_file)
        create_empty_custom_json(src_image_path, output_dir, image_id=start_image_id)
        start_image_id += 1

    # Display some log message
    if len(image_files_without_json) > 0:
        print(f"\nMissing VoTT file(s): {len(image_files_without_json)}")

    print("Conversion complete.")
    return start_image_id




def draw_on_aimage(image_path, json_label_path):
    """
    Plots bounding boxes over an image from provided custom json label json file.

    Args:
        image_path: Path to the image file.
        json_label_path: Path to the corresponding custom json label file.
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
        class_label =  f"{object_data['class']}{confidence_score}"

        buttom = ymin + 25 if ymin < 30 else ymin - 8
        cv2.putText(image, class_label, (xmin+5, buttom), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Green text

    return image

# Draw bounding boxes over all images of a custom dataset
def draw_bboxes_over_images(dataset_path):
    output_folder_path = os.path.join(dataset_path, "images_withBBoxes")
    os.makedirs(output_folder_path, exist_ok=True)

    imagefolder_path, labelfolder_path = helper.get_subfolder_paths(dataset_path)
    image_list = helper.get_image_files(imagefolder_path)

    for filename in image_list:
        image_path = os.path.join(imagefolder_path, filename) 
        label_path = os.path.join(labelfolder_path, filename[:-4] + ".json") 
        imageWithBB = draw_on_aimage(image_path, label_path)
        save_path = os.path.join(output_folder_path, filename)
        cv2.imwrite(save_path, imageWithBB)

# Check and copy images without any bbox
def check_images_withoutBBox(dataset_path, debug=False):
    output_folder_path = os.path.join(dataset_path, "images_withoutBBoxes")
    os.makedirs(output_folder_path, exist_ok=True)

    imagefolder_path, labelfolder_path = helper.get_subfolder_paths(dataset_path)
    label_filelist = helper.get_files_with_extension(labelfolder_path, extension=".json")

    images_withoutBbox = []
    for json_file in label_filelist:
        json_filepath = os.path.join(labelfolder_path, json_file)

        # Load the JSON data from the file
        data = helper.read_from_json(json_filepath)
        if len(data['objects']) == 0: 
            images_withoutBbox.append(data['asset']['name'])
            image_path = os.path.join(imagefolder_path, data['asset']['name'])
            shutil.copy(image_path, output_folder_path)

    
    if len(images_withoutBbox) > 0:
        print(f"Images without bbox: {len(images_withoutBbox)}")
        if debug:
            for image_file in images_withoutBbox:
                print(image_file)

# Check the boundary co-ordinates for each of the bouding box
def check_bbox_coordinates(dataset_path, fixFlag=False):
    labelfolder_path = os.path.join(dataset_path, "labels")
    label_filelist = helper.get_files_with_extension(labelfolder_path, extension=".json")

    count = 0
    for json_file in label_filelist:
        json_filepath = os.path.join(labelfolder_path, json_file)
        data = helper.read_from_json(json_filepath)

        image_width = data['asset']['size']['width']
        image_height = data['asset']['size']['height']

        for object in data["objects"]:
            bbox = object['boundingBox']
            xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']

            if xmin < 0.0 or ymin < 0.0 or xmax > image_width or ymax > image_height:
                print(f"Image width: {image_width:6.2f}: {xmin:6.2f} > {xmax:6.2f} image_height: {image_height:6.2f}: {ymin:6.2f} > {ymax:6.2f}")
                count += 1

            if fixFlag:
                if xmin < 0.0:
                    bbox['xmin'] = 0.0
                if ymin < 0.0:
                    bbox['ymin'] = 0.0
                if xmax > (image_width-1.0):
                    bbox['xmax'] = image_width
                    bbox['width'] = bbox['xmax'] - bbox['xmin']
                if ymax > (image_height-1.0):
                    bbox['ymax'] = image_height
                    bbox['height'] = bbox['ymax'] - bbox['ymin']

                helper.write_to_json(data, json_filepath)
    print(f"Total bbox issue count: {count}")

# Check for all unique class labels present in the dataset
def check_objects_classes(dataset_path):
    labelfolder_path = os.path.join(dataset_path, "labels")
    label_filelist = helper.get_files_with_extension(labelfolder_path, extension=".json")

    classes = set()
    for json_file in label_filelist:
        json_filepath = os.path.join(labelfolder_path, json_file)

        data = helper.read_from_json(json_filepath)
        for object in data["objects"]:
            classes.add(object["class"])

    print(f"Classes: {classes}")
    return classes

def update_objects_classes(dataset_path, class_mappping= None):
    labelfolder_path = os.path.join(dataset_path, "labels")
    label_filelist = helper.get_files_with_extension(labelfolder_path, extension=".json")

    if not class_mappping:
        raise ValueError(f"'class_mappping' is required to update class labels.")

    for json_file in label_filelist:
        json_filepath = os.path.join(labelfolder_path, json_file)

        data = helper.read_from_json(json_filepath)
        for object in data["objects"]:
            object["class"] = class_mappping[object["class"]]

        helper.write_to_json(data, json_filepath)



# Construct dataset info json file for the processed dataset
def construct_and_save_dataset_info(dataset_path, singleClass=True, label_dict=None):
    if not singleClass and not label_dict:
        raise ValueError(f"The 'label_dict' can't be None for multiple classes.")
    labelfolder_path = os.path.join(dataset_path, "labels") 

    image_name_to_id = {}
    json_files = helper.get_files_with_extension(labelfolder_path, extension=".json")
    for json_file in json_files:
        json_path = os.path.join(labelfolder_path, json_file)
        json_data = helper.read_from_json(json_path)

        image_name = json_data["asset"]["name"]
        image_id = json_data["asset"]["image_id"]
        image_name_to_id[image_name] = image_id

    if singleClass:
        dataset_info = {"categories" : {"object": 1}, "image_id_map": image_name_to_id}
    else:
        dataset_info = {"categories" : label_dict, "image_id_map": image_name_to_id}
    
    save_filepath = os.path.join(dataset_path, "dataset_info.json")
    helper.write_to_json(dataset_info, save_filepath)

# Construct yaml file of processed dataset for yolo model 
def write_yaml_file(dataset_path, singleClass=True, label_dict=None):
    if singleClass:
        reverse_label_dict = {0: 'object'}
    else:
        reverse_label_dict = {(v-1): k for k, v in label_dict.items()}

    # Define the content as a Python dictionary
    data = {
        'path': dataset_path,
        'train': 'train/images',
        'val': 'val/images',
        'test': None,  # Optional field, set to None if not present

        # Classes
        'names': reverse_label_dict
    }

    # Write data to the YAML file
    save_filepath = os.path.join(dataset_path, "dataset.yaml")
    with open(save_filepath, 'w') as file:
        yaml.dump(data, file)


def create_dataset_info(dataset_root_dir, dataset_folder, singleClass=True, label_dict=None):
    dataset_path = os.path.join(dataset_root_dir, dataset_folder)
    construct_and_save_dataset_info(dataset_path, singleClass=singleClass, label_dict=label_dict)
    write_yaml_file(dataset_path, singleClass=singleClass, label_dict=label_dict) 
