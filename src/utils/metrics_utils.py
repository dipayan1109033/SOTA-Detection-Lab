import os, sys
sys.path.append("src/utils/calculate_ODmetrics/")

import cv2
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import scripts.utils.converter as converter
from scripts.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics
from scripts.evaluators.coco_evaluator import get_coco_summary, get_coco_metrics
from scripts.utils.enumerators import BBFormat, BBType, CoordinatesType


from utils.common_utils import Helper
helper = Helper()


def prepare_csv(saveFilePath):
    # Setting header row for the *.csv file
    Header = ['class', '', 'random seed', 'model name', 'batch size', 'epochs', 'lr', 'score threshold', '', 'total_GTs', 'total_DETs', 'Difference', 'TPs counts', 'FPs counts', '', 'Precision(%)', 'Recall(%)', 'F1 Score(%)', '', 'AP50(%)', 'AP75(%)', 'AP(%)', '', 'test-dataset', 'train-id', 'Precision/std', 'Recall/std', 'F1/std', '', 'AP50/std', 'AP75/std', 'AP/std', 'predicted-jsonfile']
    with open(saveFilePath, 'w', newline='') as saveFile:
        write = csv.writer(saveFile)
        write.writerow(Header)

def writeTo_csv(results, score_threshold, prediction_json_path, saveFilePath, newcsv=False):
    if newcsv or not os.path.exists(saveFilePath):
        prepare_csv(saveFilePath)
    data = helper.read_from_json(prediction_json_path)
    info = data['experiments']
    json_filename = os.path.basename(prediction_json_path)

    aRow_data = [results['count'], '', info['seed'], info['exp_name'], info['batch_size'], info['epoch'], info['learning_rate'], score_threshold, '']
    metrics = [results['GTs'], results['DETs'], results['GTs']-results['DETs'], results['TPs'], results['FPs'], '', results['precision'], results['recall'], results['f1_score'], '', results['AP50'], results['AP75'], results['AP'], '', info['test_dataset'], info['train_id'], '', '', '', '', '', '', '', json_filename]
    aRow_data.extend(metrics)

    with open(saveFilePath, 'a', newline='') as saveFile:
        write = csv.writer(saveFile)
        write.writerow(aRow_data)


def writeTo_csv_for_crossval(metrics_dict, score_threshold, prediction_json_path, saveFilePath, newcsv=False, debug=False):
    if newcsv or not os.path.exists(saveFilePath):
        prepare_csv(saveFilePath)
    data = helper.read_from_json(prediction_json_path)
    info = data['experiments']
    json_filename = os.path.basename(prediction_json_path)

    coco_avg = {key: round(sum(values) / len(values), 2) for key, values in metrics_dict.items()}
    coco_std = {key: round(np.std(values, ddof=1), 2) for key, values in metrics_dict.items()}

    aRow_data = [coco_avg['count'], '', info['seed'], info['exp_name'], info['batch_size'], info['epoch'], info['learning_rate'], score_threshold, '']
    metrics = [int(coco_avg['GTs']), int(coco_avg['DETs']), int(coco_avg['GTs'] - coco_avg['DETs']), int(coco_avg['TPs']), int(coco_avg['FPs']), '', coco_avg['precision'], coco_avg['recall'], coco_avg['f1_score'], '', coco_avg['AP50'], coco_avg['AP75'], coco_avg['AP'], '', info['test_dataset'], info['train_id'], coco_std['precision'], coco_std['recall'], coco_std['f1_score'], '', coco_std['AP50'], coco_std['AP75'], coco_std['AP'], json_filename]
    aRow_data.extend(metrics)

    with open(saveFilePath, 'a', newline='') as saveFile:
        write = csv.writer(saveFile)
        write.writerow(aRow_data)

    print(f"scoreTh={score_threshold}         " + '      '.join(f"{key:.5}" for key, value in coco_avg.items()))
    print("metrics_avg:     " + '   '.join(f"{value:7.2f}" for key, value in coco_avg.items()))
    print("metrics_std:     " + '   '.join(f"{value:7.2f}" for key, value in coco_std.items()))
    print()

    return coco_avg, coco_std



def plot_precision_vs_recall_curve(results, title ="Precision-Recall Curve", legend=None, showAP=True, json_path=None):
    """
    Plots a precision-recall curve.

    Args:
        results: pascal_metrics['per_class']
        title: The title of the plot.
        legend: legend for the curve (default: None)
        showAP: whether or not to show the AP value in the title (default: True)
        json_path: json predictions filepath, used to determine saving the ploted curve (default: None)
    """
    output_folder = os.path.join(os.path.dirname(json_path), "plots")
    os.makedirs(output_folder, exist_ok=True)
    save_filepath = f"{output_folder}/ {os.path.basename(json_path)[:-5]}.png"
    for classId, result in results.items():
        precision = result['precision']
        recall = result['recall']
        average_precision = result['AP']

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=legend, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if showAP:
            ap_str = f"{average_precision * 100:.2f}%"
            plt.title(f"{title} \nClass: {classId}, AP: {ap_str}")
        else:
            plt.title(f"{title}")
        plt.grid(True)
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.legend(shadow=True)

        if json_path is not None:
            plt.savefig(save_filepath)
        else:
            plt.show()


class Evaluation_withCOCO:

    def __init__(self, dataset_dir, split, predictions_filepath, iou_threshold=0.5, score_threshold=0.0) -> None:
        self.dataset_dir = dataset_dir
        self.split = split
        self.predictions_filepath = predictions_filepath
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        # Decoding predictions
        data = helper.read_from_json(self.predictions_filepath)
        self.test_dataset = data['experiments']['test_dataset']
        self.image_filelist = data['image_files']
        self.categories = data['categories']

        # loading labels to COCO format
        self.coco_annotations = self.convert_labels_to_coco()
        self.image_dict = self.build_image_dict(self.coco_annotations)

        # load predictions COCO format
        self.coco_predictions_all = self.convert_predictions_to_coco(data)

        # filter COCO predictions
        self.coco_predictions = self.filter_coco_predictions(score_threshold=self.score_threshold)


    def convert_labels_to_coco(self):
        dataset_labels_path = os.path.join(self.dataset_dir, self.split, "custom_labels")

        # Initialize structures for COCO format
        coco_output = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add category information
        for name, id in self.categories.items():
            aCategory = {'id': id, 'name': name}
            coco_output["categories"].append(aCategory)

        annotation_id = 1  # Initialize annotation ID
        for image_filename in self.image_filelist:
            # Construct the JSON filename by replacing the image extension with ".json"
            label_filename = os.path.splitext(image_filename)[0] + ".json"
            label_filepath = os.path.join(dataset_labels_path, label_filename)

            # Load the JSON label file if it exists
            label_data = helper.read_from_json(label_filepath)

            # Extract image information
            image_id = label_data["asset"]["image_id"]
            width = label_data["asset"]["size"]["width"]
            height = label_data["asset"]["size"]["height"]

            # Add image information to COCO output
            coco_output["images"].append({
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height
            })

            # Process each object in the label file
            for obj in label_data["objects"]:
                # Convert to COCO format
                coco_target = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": obj["class"],  # class corresponds to category_id
                    "bbox": [
                        obj["boundingBox"]["xmin"],
                        obj["boundingBox"]["ymin"],
                        obj["boundingBox"]["width"],
                        obj["boundingBox"]["height"]
                    ],
                    "area": obj["boundingBox"]["width"] * obj["boundingBox"]["height"],
                    "iscrowd": 0
                }
                coco_output["annotations"].append(coco_target)
                annotation_id += 1  # Increment annotation ID for the next object

        return coco_output

    def build_image_dict(self, coco_annotations):
        image_dict = {}
        for image_info in coco_annotations['images']:
            image_id = image_info['id']
            image_dict[image_id] = {
                'file_name': image_info['file_name'],
                'width': image_info['width'],
                'height': image_info['height']
            }
        return image_dict

    def convert_predictions_to_coco(self, custom_data):
        # Initialize an empty list for COCO-formatted predictions
        coco_predictions = []

        # Get the category ID for 'object'
        category_dict = custom_data["categories"]

        # Convert custom predictions to COCO format
        for image_name, predictions in custom_data["predictions"].items():

            for pred in predictions:
                score = pred["score"]

                if score >= 0.0:
                    coco_pred = {
                        "image_id": pred["image_id"],                 # already in integer id format
                        "category_id": category_dict[pred["class"]],  # map the class name to category_id
                        "bbox": pred["box"],                          # already in [x, y, width, height] format
                        "score": pred["score"]
                    }
                    coco_predictions.append(coco_pred)

        coco_output = {
                "images": self.coco_annotations["images"],
                "annotations": coco_predictions,
                "categories": self.coco_annotations["categories"]
            }
        
        return coco_output

    def filter_coco_predictions(self, score_threshold=0.0, lowerArea_threshold=0.005):
        # Initialize an empty list for COCO-formatted predictions
        coco_predictions = []

        # Filter COCO predictions 
        for coco_pred in self.coco_predictions_all["annotations"]:
            image_id = coco_pred["image_id"]
            score = coco_pred["score"]

            image_area = self.image_dict[image_id]['width'] * self.image_dict[image_id]['height']
            left, top, width, height = [float(x) for x in coco_pred['bbox']]
            boxArea = width * height

            if score >= score_threshold and (boxArea/image_area) >= lowerArea_threshold:
                coco_predictions.append(coco_pred)

        coco_output = {
                "images": self.coco_annotations["images"],
                "annotations": coco_predictions,
                "categories": self.coco_annotations["categories"]
            }
        self.score_threshold = score_threshold
        self.coco_predictions = coco_output
        return coco_output

    def load_groundtruth_boxes(self):
        ret = converter.coco2bb_withoutPath(self.coco_annotations)
        return ret

    def load_detection_boxes(self):
        ret = converter.coco2bb_withoutPath(self.coco_predictions, bb_type=BBType.DETECTED)
        return ret
    
    def draw_bounding_boxes(self, save_folder, postfix = ""):
        output_folder_path = f"output/{save_folder}/detected_boxes/"
        os.makedirs(output_folder_path, exist_ok=True)

        # Organize predictions by image_id
        predictions_by_image = defaultdict(list)
        for coco_pred in self.coco_predictions["annotations"]:
            image_name = self.image_dict[coco_pred['image_id']]['file_name']
            predictions_by_image[image_name].append(coco_pred)


        # Loop through each image and draw bboxes
        dataset_images_folder = os.path.join(self.dataset_dir, self.split, "images")
        id_category_mapping = {v:k for k,v in self.categories.items()}
        for image_name in self.image_filelist:
            image_path = os.path.join(dataset_images_folder, image_name)
            image = cv2.imread(image_path)

            predictions = predictions_by_image[image_name]
            if len(predictions) > 0:
                for coco_pred in predictions:
                    category_id = coco_pred["category_id"]
                    class_name = id_category_mapping[category_id]
                    left, top, width, height = [int(x) for x in coco_pred['bbox']]
                    score = coco_pred["score"]

                    # Draw the bounding box on the image
                    cv2.rectangle(image, (left, top), (left+width, top+height), (0, 255, 0), 2)  # Green for boxes

                    # Put class name
                    buttom = top + 25 if top < 30 else top - 8
                    cv2.putText(image, f"{class_name}: {score:.2f}", (left+5, buttom), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            save_filepath = f"{output_folder_path}{image_name[:-4]}_{postfix}.jpg"
            cv2.imwrite(save_filepath, image)

    def print_and_save_results(self, coco_summary, pascal_metrics):
        data = helper.read_from_json(self.predictions_filepath)

        # Extract key pascal results
        results_pascal = {}
        for class_id, result in pascal_metrics['per_class'].items():
            results_pascal[class_id] = {
                'total positives': result['total positives'],
                'total TP': result['total TP'],
                'total FP': result['total FP'],
                'AP': float(result['AP']),
                'iou': result['iou'],
                'precision': list(result['precision']),
                'recall': list(result['recall'])
            }
        if self.score_threshold == 0.0:
            data['results_coco'] = coco_summary
        else:
            data['results_pascal'] = results_pascal

        # Calculate and save results summary
        classes_count = len(pascal_metrics['per_class'])
        groundTruths = sum([result['total positives'] for result in pascal_metrics['per_class'].values()])
        TPs = sum([result['total TP'] for result in pascal_metrics['per_class'].values()])
        FPs = sum([result['total FP'] for result in pascal_metrics['per_class'].values()])
        totalDetections = int(TPs+FPs)
        precision = round(100*TPs/(TPs+FPs),2) if (TPs + FPs) != 0 else 0.0
        recall = round(100*TPs/groundTruths,2) if (groundTruths) != 0 else 0.0
        f1_score = round(2*precision*recall/(precision+recall), 2) if (precision+recall) != 0 else 0.0
        coco_AP, coco_AP50, coco_AP75 = [round(100*score, 2) for key, score in coco_summary.items() if key in [ 'AP', 'AP50', 'AP75']]
        rs = {'count':classes_count, 'GTs': groundTruths, 'DETs':totalDetections, 'TPs': int(TPs), 'FPs': int(FPs), 'precision': precision, 'recall': recall, 'f1_score': f1_score, 'AP50': coco_AP50, 'AP75': coco_AP75, 'AP': coco_AP}

        # Print results to console
        print(f"Object classes: {list(pascal_metrics['per_class'].keys())} scoreTh={self.score_threshold}")
        print("total_GTs   total_DETs     TP_count    FP_count    Precision    Recall    F1_Score  |   AP50      AP75      AP")
        print(f" {rs['GTs']:5d}  {rs['DETs']:10d}  {rs['TPs']: 13d}  {rs['FPs']:10d}  {rs['precision']:11.2f}   {rs['recall']:9.2f}  {rs['f1_score']:9.2f}  {rs['AP50']:10.2f}    {rs['AP75']:6.2f}    {rs['AP']:6.2f}\n")

        # Save to prediction json file
        data[f'results_summary_{self.score_threshold}'] = rs
        helper.write_to_json(data, self.predictions_filepath)
        return rs

    def compute(self, save_dir, model_identifier, saveToCSV=False):

        gt_annotations = self.load_groundtruth_boxes()
        det_annotations = self.load_detection_boxes()

        # Calculate the evaluation metrics
        coco_summary = get_coco_summary(gt_annotations, det_annotations)
        pascal_metrics = get_pascalvoc_metrics(gt_annotations, det_annotations, iou_threshold=self.iou_threshold, generate_table=False)

        # Plot precision-recall curve
        results = pascal_metrics['per_class']
        plot_precision_vs_recall_curve(results, title ="Precision-Recall Curve", legend="Yolo11", json_path=self.predictions_filepath)

        # display and save these results in the predictions json file
        results_summary = self.print_and_save_results(coco_summary, pascal_metrics)
        if saveToCSV: writeTo_csv(results_summary, self.score_threshold, self.predictions_filepath, f"{save_dir}/results_{model_identifier}.csv", newcsv=False)

        return results_summary



    def compute_COCO(self):

        # Load the ground truth annotations
        coco_gt = COCO(self.targets_path)

        # Load the predictions
        coco_pred = coco_gt.loadRes(self.coco_predictions['annotations'])

        # Initialize COCO evaluator
        coco_eval = COCOeval(coco_gt, coco_pred, "bbox")

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Access metrics
        metrics = coco_eval.stats
        print(metrics)
        print("AP@[IoU=0.50:0.95]:", metrics[0])
        print("AP@0.50:", metrics[1])
        print("AP@0.75:", metrics[2])



def evaluate_model_crossval():
    model_name = "yolo"     # "yolo", "rcnn", "classic_alg"
    conf_score = 0.0

    metrics_dict = {'count': [], 'GTs': [], 'DETs':[], 'TPs': [], 'FPs': [], 'precision': [], 'recall': [], 'f1_score': [], 'AP50': [], 'AP75': [], 'AP': []}
    for fold_indx in range(3):
        json_filename = f"train13_fold{fold_indx}_rs42_yolov8_SGD_b8_e2_lr0.001__sample_data.json"
        prediction_json_path = f"output/{model_name}/model_predictions/{json_filename}"

        evalObj = Evaluation_withCOCO(prediction_json_path, score_threshold=conf_score)
        results_summary = evalObj.compute(model_name, saveToCSV=True)

        # Use a loop to append values
        for key in metrics_dict:
            metrics_dict[key].append(results_summary[key])
            
    metrics_avg, metrics_std = writeTo_csv_for_crossval(metrics_dict, conf_score, prediction_json_path, f"output/{model_name}/results_cv.csv", newcsv=False)
    print("                    " + '      '.join(f"{key:.5}" for key, value in metrics_avg.items()))
    print("metrics_avg:     " + '   '.join(f"{value:7.2f}" for key, value in metrics_avg.items()))
    print("metrics_std:     " + '   '.join(f"{value:7.2f}" for key, value in metrics_std.items()))


def evaluate_model():
    model_name = "yolo"     # "yolo", "rcnn", "classic_alg"
    
    for dataset_name in ["testdata_1k"]:
    #for dataset_name in ["testdata_215", "testdata_1k", "ldd_vdataset", "lpm_dataset"]:

        json_filename = f"train5_rs42_YOLOv8_SGD_b8_e50_lr0.001__{dataset_name}.json"
        prediction_json_path = f"output/{model_name}/model_predictions/{json_filename}"

        evalObj = Evaluation_withCOCO(prediction_json_path, score_threshold=0.5)
        evalObj.compute(model_name, saveToCSV=True)
        
        evalObj.filter_coco_predictions(score_threshold=0.0)
        evalObj.compute(model_name, saveToCSV=True)
        evalObj.draw_bounding_boxes(save_folder=model_name, postfix="yolo")




if __name__ == "__main__":
    helper = Helper()

    
    #evaluate_model()
    evaluate_model_crossval()

    

