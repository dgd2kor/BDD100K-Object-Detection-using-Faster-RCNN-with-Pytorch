import os
import json
import torch
from typing import List, Tuple, Set, Dict
import numpy as np

from config import Config

class DetectionMetrics:
    """Class for computing object detection evaluation metrics: IoU, Precision, Recall, and F1-score."""

    def __init__(self, iou_threshold: float = 0.5):
        """
        Initializes the DetectionMetrics class.

        Args:
            iou_threshold (float): IoU threshold to consider a detection as True Positive.
        """
        self.iou_threshold = iou_threshold

    @staticmethod
    def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
        """
        Computes the Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (torch.Tensor): Bounding box [x1, y1, x2, y2].
            box2 (torch.Tensor): Bounding box [x1, y1, x2, y2].

        Returns:
            float: IoU score.
        """
        box1 = box1.squeeze(0)  # Remove extra dimension if needed

        # Compute intersection
        x1 = torch.max(box1[0], box2[:, 0])
        y1 = torch.max(box1[1], box2[:, 1])
        x2 = torch.min(box1[2], box2[:, 2])
        y2 = torch.min(box1[3], box2[:, 3])

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        # Compute union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        union = box1_area + box2_area - intersection

        return intersection / (union + 1e-6)  # Avoid division by z
    def compute_metrics(
        self, prediction_data
    ) -> Tuple[float, float, float]:
        """
        Computes precision, recall, and F1-score for a single image.

        Args:
            gt_boxes (torch.Tensor): Ground truth bounding boxes.
            gt_labels (torch.Tensor): Ground truth labels.
            pred_boxes (torch.Tensor): Predicted bounding boxes.
            pred_labels (torch.Tensor): Predicted labels.
            pred_scores (torch.Tensor): Prediction confidence scores.

        Returns:
            Tuple[float, float, float]: Precision, Recall, and F1-score.
        """

        total_precision, total_recall, total_f1_score = [], [], []
        for idx, predictions in enumerate(prediction_data):
            tp, fp, fn = 0, 0, 0
            matched_gt: Set[int] = set()
            pred_boxes = predictions["pred_boxes"]
            pred_labels = predictions["pred_labels"]
            gt_boxes = predictions["gt_boxes"]
            gt_labels = predictions["gt_labels"]
            pred_scores = predictions["pred_scores"]
            for pred_idx, pred_box in enumerate(pred_boxes):
                pred_label = pred_labels[pred_idx]
                best_iou, best_gt_idx = 0, -1

                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_labels[gt_idx] == pred_label:
                        iou = self.compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou, best_gt_idx = iou, gt_idx

                if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn = len(gt_boxes) - len(matched_gt)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            total_precision.append(precision)
            total_recall.append(recall)
            total_f1_score.append(f1_score)
        # Compute final Precision, Recall, F1 Score
        avg_precision = np.mean(total_precision)
        avg_recall = np.mean(total_recall)
        avg_f1_score = np.mean(total_f1_score)

        return avg_precision, avg_recall, avg_f1_score


    def compute_map(self, predictions_data, iou_threshold=0.5):
        """
        Compute mean Average Precision (mAP) for object detection using CUDA.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        average_precisions = []
        
        gt_boxes_list, gt_labels_list, pred_boxes_list, pred_labels_list, pred_scores_list = [], [], [], [], []
        
        # Move data to GPU
        for prediction_data in predictions_data:
            gt_boxes_list.append(torch.tensor(prediction_data["gt_boxes"], device=device))
            gt_labels_list.append(torch.tensor(prediction_data["gt_labels"], device=device))
            pred_boxes_list.append(torch.tensor(prediction_data["pred_boxes"], device=device))
            pred_labels_list.append(torch.tensor(prediction_data["pred_labels"], device=device))
            pred_scores_list.append(torch.tensor(prediction_data["pred_scores"], device=device))

        for class_id in range(10):  # Number of classes in BDD100K
            
            class_gt_boxes = []
            class_pred_boxes = []
            class_pred_scores = []
            
            for gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores in zip(
                gt_boxes_list, gt_labels_list, pred_boxes_list, pred_labels_list, pred_scores_list
            ):
                class_gt_boxes.extend(gt_boxes[gt_labels == class_id])
                class_pred_boxes.extend(pred_boxes[pred_labels == class_id])
                class_pred_scores.extend(pred_scores[pred_labels == class_id])
            
            if not class_gt_boxes:
                continue
            
            class_pred_boxes = torch.stack(class_pred_boxes) if len(class_pred_boxes) > 0 else torch.tensor([], device=device)
            class_pred_scores = torch.tensor(class_pred_scores, device=device)

            # Sort predictions by confidence scores (descending)
            sorted_indices = torch.argsort(class_pred_scores, descending=True)
            class_pred_boxes = class_pred_boxes[sorted_indices]

            tp = torch.zeros(len(class_pred_boxes), device=device)
            fp = torch.zeros(len(class_pred_boxes), device=device)
            matched_gt = torch.zeros(len(class_gt_boxes), device=device)

            for pred_idx, pred_box in enumerate(class_pred_boxes):
                
                if len(class_gt_boxes) == 0:
                    fp[pred_idx] = 1
                    continue

                gt_boxes_tensor = torch.stack(class_gt_boxes)
                ious = self.compute_iou(pred_box.unsqueeze(0), gt_boxes_tensor).squeeze()
                best_iou, best_gt_idx = ious.max(dim=0)

                if best_iou >= iou_threshold and matched_gt[best_gt_idx] == 0:
                    tp[pred_idx] = 1
                    matched_gt[best_gt_idx] = 1
                else:
                    fp[pred_idx] = 1

            tp_cumsum = torch.cumsum(tp, dim=0)
            fp_cumsum = torch.cumsum(fp, dim=0)

            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
            recalls = tp_cumsum / len(class_gt_boxes)

            ap = torch.trapz(precisions, recalls).item()
            average_precisions.append(ap)
            print(f"Class {class_id} AP: {ap}")

        return average_precisions, torch.tensor(average_precisions).mean().item() if average_precisions else 0


def load_predictions_from_json(json_path: str) -> List[Dict[str, torch.Tensor]]:
    """Loads predictions from a JSON file and converts lists back to torch.Tensors.

    Args:
        json_path (str): Path to the JSON file containing saved predictions.

    Returns:
        List[Dict[str, torch.Tensor]]: List of dictionaries with tensorized predictions.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    for pred in data:
        pred["pred_boxes"] = torch.tensor(pred["pred_boxes"], dtype=torch.float32)
        pred["pred_labels"] = torch.tensor(pred["pred_labels"], dtype=torch.int64)
        pred["gt_boxes"] = torch.tensor(pred["gt_boxes"], dtype=torch.float32)
        pred["gt_labels"] = torch.tensor(pred["gt_labels"], dtype=torch.int64)
        pred["pred_scores"] = torch.tensor(pred["pred_scores"], dtype=torch.float32)

    return data


# Example Usage:
if __name__ == "__main__":
    config = Config()

    evaluator = DetectionMetrics(iou_threshold=config.iou_threshold)
    prediction_data = load_predictions_from_json(os.path.join(config.predictions_path,"predictions.json"))

    # Compute precision, recall, and F1 score
    precision, recall, f1 = evaluator.compute_metrics(prediction_data)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    # Compute mAP
    class_aps, mAP = evaluator.compute_map(predictions_data=prediction_data)
    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    for class_id, ap in enumerate(class_aps):
        print(f"Class {class_id} AP: {ap:.4f}")
    # Save metrics to a file
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_average_precision": mAP,
        "class_aps": class_aps
    }
    
    if not os.path.exists(config.predictions_path):
        os.makedirs(config.predictions_path)
    metrics_path = os.path.join(config.predictions_path, "metrics.json")
    # Save the metrics to a JSON file
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

