
import os
import json

from utils.common_utils import Helper
helper = Helper()



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

