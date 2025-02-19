
import os
import json
import yaml
import pickle
import shutil
import random
from omegaconf import OmegaConf


class Helper:

    @staticmethod
    def get_default_config(file_path):
        config = OmegaConf.load(file_path)
        return config
    
    @staticmethod
    def read_from_json(file_path):
        """Reads data from a JSON file."""
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"File not found: {file_path}.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from: {file_path}.")
        return None

    @staticmethod
    def write_to_json(data, file_path):
        """Writes data to a JSON file."""
        try:
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error writing to JSON file {file_path}: {e}")

    @staticmethod
    def read_from_yaml(file_path):
        """Reads data from a YAML file."""
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"File not found: {file_path}.")
        except yaml.YAMLError as e:
            print(f"Error reading YAML from {file_path}: {e}")
        return None

    @staticmethod
    def write_to_yaml(data, file_path):
        """Writes data to a YAML file."""
        try:
            with open(file_path, 'w') as file:
                yaml.dump(data, file)
        except Exception as e:
            print(f"Error writing to YAML file {file_path}: {e}")

    @staticmethod
    def read_from_pickle(file_path):
        """Reads data from a pickle file."""
        try:
            with open(file_path, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            print(f"File not found: {file_path}.")
        except pickle.UnpicklingError as e:
            print(f"Error reading from pickle file {file_path}: {e}")
        return None

    @staticmethod
    def write_to_pickle(data, file_path):
        """Writes data to a pickle file."""
        try:
            with open(file_path, "wb") as file:
                pickle.dump(data, file)
        except Exception as e:
            print(f"Error writing to pickle file {file_path}: {e}")




    @staticmethod
    def get_immediate_folder_name(path):
        """Returns the immediate folder name from a given file or directory path.

        Args:
            path: The path to a file or directory.

        Returns:
            The name of the immediate folder containing the file or directory.

        Raises:
            ValueError: If the provided path does not exist or is not a valid file or directory.
        """
        # Check if the provided path exists
        if not os.path.exists(path):
            raise ValueError(f"The provided path '{path}' does not exist.")
        
        # If the path is a file, get the directory name
        if os.path.isfile(path):
            folder_path = os.path.dirname(path)
        else:
            folder_path = path  # Treat the path as a directory if it is not a file

        # Get the immediate folder name
        folder_name = os.path.basename(folder_path.rstrip("/\\"))
        
        return folder_name

    @staticmethod
    def create_folder(folder_path):
        """Creates a folder if it does not already exist.

        Args:
            folder_path: The path of the folder to create.
        """
        try:
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            print(f"Error creating folder {folder_path}: {e}")

    @staticmethod
    def create_subfolders(directory_path, folder_list=["images", "labels"]):
        """Creates specified subfolders within the provided directory (default: 'images' and 'labels').

        If the subfolder already exists, it will be deleted and recreated.

        Args:
            directory_path: The path of the directory
            folder_list: A list of subfolder names to create within the main directory. Defaults to ["images", "labels"].

        Returns:
            A list of paths to the created subfolders.
        """
        try:
            # Remove the main directory if it already exists and recreate it
            if os.path.exists(directory_path):
                shutil.rmtree(directory_path)
            os.makedirs(directory_path, exist_ok=True)
        
            created_folder_paths = []
            for folder_name in folder_list:
                folder_path = os.path.join(directory_path, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                created_folder_paths.append(folder_path)

            return created_folder_paths
        except Exception as e:
            print(f"Error creating subfolders in {directory_path}: {e}")
            return []

    @staticmethod
    def get_subfolder_names(directory_path):
        """Returns the names of all subfolders in the given directory.

        Args:
            directory_path: The path of the directory to search.

        Returns:
            A list of subfolder names.

        Raises:
            ValueError: If the provided path is not a valid directory.
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"The provided path '{directory_path}' is not a valid directory.")

        subfolders = []
        for name in os.listdir(directory_path):
            folder_path = os.path.join(directory_path, name)
            if os.path.isdir(folder_path):
                subfolders.append(name)

        return subfolders

    @staticmethod
    def get_subfolder_paths(directory_path, folder_list=["images", "labels"]):
        """Returns paths of specified subfolders in a directory.

        Args:
            directory_path: The path of the directory to search.
            folder_list: A list of folder names to return paths for. If `None`, all subfolder paths are returned. Defaults to ["images", "labels"].

        Returns:
            A list of paths to the specified subfolders, or all subfolders if `folder_list` is `None`.
        """
        try:
            if folder_list is None:
                # Return paths for all subfolders
                folder_list = Helper.get_subfolder_names(directory_path)
                
            # Return paths for specified subfolders
            return [os.path.join(directory_path, folder_name) for folder_name in folder_list]
        except Exception as e:
            print(f"Error retrieving subfolder paths in {directory_path}: {e}")
            return []




    @staticmethod
    def get_image_files(folder_path):
        """Retrieves a list of image files in the specified folder.

        Args:
            folder_path: The path of the folder to search for image files.

        Returns:
            A list of image files found in the folder, or an empty list if the folder does not exist.
        """
        image_extensions = (".jpg", ".jpeg", ".png")
        if os.path.exists(folder_path):
            return [file for file in os.listdir(folder_path) if file.endswith(image_extensions)]
        else:
            print(f"The folder '{folder_path}' does not exist.")
            return []

    @staticmethod
    def get_files_with_extension(folder_path, extension):
        """Retrieves a list of files with the specified extension in the given folder.

        Args:
            folder_path: The path of the folder to search for files.
            extension: The file extension to filter by (e.g., '.txt').

        Returns:
            A list of files with the specified extension, or an empty list if the folder does not exist.
        """
        if os.path.exists(folder_path):
            return [file for file in os.listdir(folder_path) if file.endswith(extension)]
        else:
            print(f"The folder '{folder_path}' does not exist.")
            return []
        
    @staticmethod
    def get_files_with_extensions(folder_path, extensions):
        """Retrieves a list of files with the specified extensions in the given folder.

        Args:
            folder_path: The path of the folder to search for files.
            extensions: A tuple or list of file extensions to filter by.

        Returns:
            A list of files that match the specified extensions, or an empty list if the folder does not exist.
        """
        if os.path.exists(folder_path):
            if isinstance(extensions, list):
                extensions = tuple(extensions)
            return [file for file in os.listdir(folder_path) if file.endswith(extensions)]
        else:
            print(f"The folder '{folder_path}' does not exist.")
            return []
        
    @staticmethod
    def get_files_without_extension(folder_path, extension):
        """Retrieves a list of file names (without extensions) in the specified folder that end with the given extension.

        Args:
            folder_path: The path of the folder to search for files.
            extension: The file extension to filter by (e.g., '.txt').

        Returns:
            A list of file names without the specified extension, or an empty list if the folder does not exist.
        """
        if os.path.exists(folder_path):
            return [os.path.splitext(file)[0] for file in os.listdir(folder_path) if file.endswith(extension)]
        else:
            print(f"The folder '{folder_path}' does not exist.")
            return []



        
    @staticmethod
    def delete_files_withExtension(folder_path, extension):
        """Deletes all files with the specified extension in the given folder.

        Args:
            folder_path: The path of the folder from which to delete files.
            extension: The file extension to filter by (e.g., '.txt').
        """
        file_list = Helper.get_files_with_extension(folder_path, extension)
        for filename in file_list:
            filepath = os.path.join(folder_path, filename)
            os.remove(filepath)

    @staticmethod
    def delete_all_files_in_folder(folder_path):
        """Deletes all files and subfolders in the specified folder.

        Args:
            folder_path: The path of the folder from which to delete all files and subfolders.

        Raises:
            ValueError: If the provided path is not a valid directory.
        """
        if not os.path.isdir(folder_path):
            raise ValueError(f"The provided path '{folder_path}' is not a valid directory.")

        try:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)         # Delete the file
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)     # Delete the subfolder and its contents
        except Exception as e:
            print(f"Error deleting files in '{folder_path}': {e}")




    @staticmethod
    def copy_specified_files(src_folder, dest_folder, file_list):
        """
        Copies specified files from the source folder to the destination folder.

        Args:
            src_folder (str): The source folder path.
            dest_folder (str): The destination folder path.
            file_list (list): List of filenames to copy.

        Raises:
            OSError: If there is an error during the file copy process.
        """
        # Ensure the destination folder exists
        Helper.create_folder(dest_folder)

        for filename in file_list:
            src_filepath = os.path.join(src_folder, filename)

            # Check if the file exists in the source folder
            if not os.path.isfile(src_filepath):
                print(f"File '{filename}' does not exist in the source folder.")
                continue

            try:
                shutil.copy(src_filepath, dest_folder)
            except OSError as e:
                print(f"Error copying '{filename}': {e}")

    @staticmethod
    def copy_specified_files_withExtension(src_folder, dest_folder, file_list, extension):
        """
            Copies files with the specified extension from the source folder to the destination folder.

        Args:
            src_folder (str): The source folder path.
            dest_folder (str): The destination folder path.
            file_list (list): List of filenames (without extensions) to copy.
            extension (str): File extension to append when looking for files.

        Raises:
            OSError: If there is an error during the file copy process.
        """
        # Ensure the destination folder exists
        Helper.create_folder(dest_folder)

        # Iterate over the file list and copy each file
        for filename in file_list:
            src_file_path = os.path.join(src_folder, filename + extension)

            # Check if the file exists in the source folder
            if not os.path.isfile(src_file_path):
                print(f"File '{filename + extension}' does not exist in the source folder.")
                continue

            try:
                shutil.copy(src_file_path, dest_folder)
            except OSError as e:
                print(f"Error copying '{filename + extension}': {e}")

    @staticmethod
    def copy_files_withExtension(src_folder, dest_folder, extension='.jpg'):
        """
        Copies files with a specified extension from the source folder to the destination folder.

        Args:
            src_folder (str): The source folder path.
            dest_folder (str): The destination folder path.
            extension (str): The file extension to filter files by (default is '.jpg').

        Raises:
            OSError: If there is an error during the file copy process.
        """
        # Get a list of files with the specified extension
        file_list = Helper.get_files_with_extension(src_folder, extension)

        # Copy the specified files to the destination folder
        Helper.copy_specified_files(src_folder, dest_folder, file_list)

    @staticmethod
    def copy_files_withExtensions(src_folder, dest_folder, extensions):
        """
        Copies files with a list of specified extensions from the source folder to the destination folder.

        Args:
            src_folder (str): The source folder path.
            dest_folder (str): The destination folder path.
            extension (list or tuple): A tuple or list of file extensions to filter by.

        Raises:
            OSError: If there is an error during the file copy process.
        """
        # Get a list of files with the specified extension
        file_list = Helper.get_files_with_extensions(src_folder, extensions)

        # Copy the specified files to the destination folder
        Helper.copy_specified_files(src_folder, dest_folder, file_list)

        

    @staticmethod
    def split_files_to_train_val(dataset_path, label_ext = ".txt", val_ratio=0.2, seed=None):
        """ 
        Randomly split a dataset into training and validation subsets.
        """
        # OUTPUT: Create train and val folders if they don't exist
        train_folder, val_folder = Helper.create_subfolders(dataset_path, folder_list=["train", "val"])

        # OUTPUT: Creating subfolders for images and labels
        train_images_folder, train_labels_folder = Helper.create_subfolders(train_folder)
        val_images_folder, val_labels_folder = Helper.create_subfolders(val_folder)

        # INPUT: Get the list of files in the folder
        dataset_images_folder = os.path.join(dataset_path, "images")
        dataset_labels_folder = os.path.join(dataset_path, "labels")
        image_files = Helper.get_files_with_extension(dataset_images_folder, extension=".jpg")

        # Shuffle the list of files randomly
        if seed is not None:
            random.seed(seed)
        random.shuffle(image_files)

        # Calculate the number of files for each partition
        val_count = int(len(image_files) * val_ratio)
        train_count = len(image_files) - val_count

        # Calculate the number of files for training
        train_count = int(len(image_files) * (1.0 - val_ratio))

        # Split the list of files into training and validation subsets
        train_image_files = image_files[:train_count]
        val_image_files = image_files[train_count:]

        # Copy files to the train folder
        for file_name in train_image_files:
            src_image_file = os.path.join(dataset_images_folder, file_name)
            src_label_file = os.path.join(dataset_labels_folder, file_name[:-4] + label_ext)
            shutil.copy(src_image_file, train_images_folder)
            shutil.copy(src_label_file, train_labels_folder)

        # Copy files to the val folder
        for file_name in val_image_files:
            src_image_file = os.path.join(dataset_images_folder, file_name)
            src_label_file = os.path.join(dataset_labels_folder, file_name[:-4] + label_ext)
            shutil.copy(src_image_file, val_images_folder)
            shutil.copy(src_label_file, val_labels_folder)

        print(f"Train samples: {len(train_image_files)} and validation samples: {len(val_image_files)}")

    @staticmethod
    def split_files_to_train_val_test(dataset_path, label_ext = ".txt", val_ratio=0.1, test_ratio=0.2, seed=None):
        """ 
        Randomly split a dataset into training and validation subsets.
        """
        # OUTPUT: Create train, val and test folders if they don't exist
        train_folder, val_folder, test_folder = Helper.create_subfolders(dataset_path, folder_list=["train", "val", "test"])

        # OUTPUT: Creating subfolders for images and labels
        train_images_folder, train_labels_folder = Helper.create_subfolders(train_folder)
        val_images_folder, val_labels_folder = Helper.create_subfolders(val_folder)
        test_images_folder, test_labels_folder = Helper.create_subfolders(test_folder)

        # INPUT: Get the list of files in the folder
        dataset_images_folder = os.path.join(dataset_path, "images")
        dataset_labels_folder = os.path.join(dataset_path, "labels")
        image_files = Helper.get_files_with_extensions(dataset_images_folder, extensions=[".jpg", ".png"])

        # Shuffle the list of files randomly
        if seed is not None:
            random.seed(seed)
        random.shuffle(image_files)


        # Calculate the number of files for each partition
        val_count = int(len(image_files) * val_ratio)
        test_count = int(len(image_files) * test_ratio)
        train_count = len(image_files) - val_count - test_count

        # Split the list of files into training and validation subsets
        train_image_files = image_files[:train_count]
        val_image_files = image_files[train_count:train_count+val_count]
        test_image_files = image_files[train_count+val_count:]

        # Copy files to the train folder
        for file_name in train_image_files:
            src_image_file = os.path.join(dataset_images_folder, file_name)
            src_label_file = os.path.join(dataset_labels_folder, file_name[:-4] + label_ext)
            shutil.copy(src_image_file, train_images_folder)
            shutil.copy(src_label_file, train_labels_folder)

        # Copy files to the val folder
        for file_name in val_image_files:
            src_image_file = os.path.join(dataset_images_folder, file_name)
            src_label_file = os.path.join(dataset_labels_folder, file_name[:-4] + label_ext)
            shutil.copy(src_image_file, val_images_folder)
            shutil.copy(src_label_file, val_labels_folder)

        # Copy files to the val folder
        for file_name in test_image_files:
            src_image_file = os.path.join(dataset_images_folder, file_name)
            src_label_file = os.path.join(dataset_labels_folder, file_name[:-4] + label_ext)
            shutil.copy(src_image_file, test_images_folder)
            shutil.copy(src_label_file, test_labels_folder)

        print(f"Train samples: {len(train_image_files)}, validation samples: {len(val_image_files)}, and  test samples: {len(test_image_files)}")

