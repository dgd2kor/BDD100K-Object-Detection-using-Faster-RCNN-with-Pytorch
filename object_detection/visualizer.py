import os
import json
import torch
from typing import Any, List, Dict
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from config import Config

class PredictionVisualizer:
    """Class for loading and visualizing object detection predictions."""

    def __init__(self, json_file: str, image_dir: str, save_dir: str = "output/visualizations"):
        """
        Initializes the PredictionVisualizer.

        Args:
            json_file (str): Path to the saved JSON predictions file.
            image_dir (str): Directory containing the validation images.
            save_dir (str): Directory to save visualized images.
        """
        self.json_file = json_file
        self.image_dir = image_dir
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _convert_to_tensor(self, data: Any) -> Any:
        """Converts lists in a dictionary back to tensors."""
        if isinstance(data, list):
            return torch.tensor(data, dtype=torch.float32)
        return data

    def load_predictions(self) -> List[Dict[str, torch.Tensor]]:
        """Loads and processes the predictions from a JSON file."""
        with open(self.json_file, "r") as f:
            results = json.load(f)

        for item in results:
            item["bboxes_pred"] = self._convert_to_tensor(item.pop("pred_boxes"))
            item["labels_pred"] = self._convert_to_tensor(item.pop("pred_labels")).to(torch.int64)
            item["bboxes_gt"] = self._convert_to_tensor(item.pop("gt_boxes"))
            item["labels_gt"] = self._convert_to_tensor(item.pop("gt_labels")).to(torch.int64)
            item["scores_pred"] = self._convert_to_tensor(item.pop("pred_scores"))

        return results

    def _draw_boxes(self, image_tensor: torch.Tensor, gt_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> torch.Tensor:
        """Draws ground truth and predicted boxes on an image tensor."""
        img = (image_tensor * 255).byte().clone()
        img = draw_bounding_boxes(img, gt_boxes, colors="green", labels=["GT"] * len(gt_boxes), width=3)
        img = draw_bounding_boxes(img, pred_boxes, colors="red", labels=["PRED"] * len(pred_boxes), width=3)
        return img

    def visualize(self, num_samples: int = 10) -> None:
        """Visualizes a subset of predictions."""
        predictions = self.load_predictions()
        
        for i, result in enumerate(predictions[:num_samples]):
            img_path = os.path.join(self.image_dir, result["filename"])
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                image_tensor = F.to_tensor(image)

                annotated_img = self._draw_boxes(image_tensor, result["bboxes_gt"], result["bboxes_pred"])
                annotated_img = F.to_pil_image(annotated_img)

                save_path = os.path.join(self.save_dir, result["filename"])
                annotated_img.save(save_path)

        print(f"Saved visualizations in {self.save_dir}")


if __name__=='__main__':
    config = Config()
    json_path = os.path.join(config.predictions_path, "predictions.json")
    image_folder = config.val_image_path

    visualizer = PredictionVisualizer(json_file=json_path, image_dir=image_folder, save_dir=config.pred_visuals_path)
    visualizer.visualize(num_samples=10)
