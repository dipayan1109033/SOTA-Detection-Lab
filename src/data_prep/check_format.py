import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import shutil
from PIL import Image
from src.utils.prepare_utils import *
from src.utils.common_utils import Helper
helper = Helper()




def checking_aCustom_dataset(dataset_path, mapping=None):
    """ Perform various checks on a custom dataset to ensure its consistency and correctness."""
    
    check_label_image_correspondence(dataset_path, label_folder="custom_labels")

    check_unique_image_ids_with_metadata(dataset_path, label_folder="custom_labels")

    #check_dataset_objects_classes(dataset_path, label_folder="custom_labels")

    #validate_dataset_bbox_coordinates(dataset_path, label_folder="custom_labels", fix_flag=False)

    #draw_bboxes_across_dataset_splits(dataset_path, label_folder="custom_labels")
    

def update_dataset_metadata(dataset_path):
    """Update metadata across the dataset."""

    #update_image_ids_sequentially(dataset_path, start_image_id=5000, label_folder="custom_labels")

    class_mapping = {}
    update_objects_classes_across_dataset(dataset_path, class_mapping=class_mapping, label_folder="custom_labels")



    
def main():

    # Define the dataset folder name
    dataset_folder = "coco8"
    dataset_path = os.path.join(processed_data_dir, dataset_folder)

    #checking_aCustom_dataset(dataset_path)

    #update_dataset_metadata(dataset_path)



if __name__ == "__main__":
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"
    main()
