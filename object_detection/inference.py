import os
import json
import torch
from tqdm import tqdm
from typing import Any, List, Dict
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import Config
from bdd100k_dataset import BDD100KDataLoader

class ObjectDetectionInference:
    """Class to handle object detection inference."""
    
    def __init__(self, model_path: str, device: str = None):
        """Initializes the model for inference."""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Loads the trained Faster R-CNN model."""
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 12)  # Assuming 12 classes
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def predict(self, dataloader, conf_threshold: float = 0.4, save_path: str = "results") -> List[Dict]:
        """Runs inference on a given DataLoader and saves predictions as JSON."""
        os.makedirs(save_path, exist_ok=True)
        detections = []
        
        for batch_idx, (images, annotations) in enumerate(tqdm(dataloader, desc="Running Inference")):
            images = [img.to(self.device) for img in images]
            
            with torch.no_grad():
                results = self.model(images)
            
            for img, result, annotation in zip(images, results, annotations):
                keep = result["scores"] > conf_threshold
                
                detections.append({
                    "image_id": annotation["image_id"],
                    "pred_boxes": result["boxes"][keep].cpu().tolist(),
                    "pred_scores": result["scores"][keep].cpu().tolist(),
                    "gt_boxes": annotation["boxes"].tolist(),
                    "pred_labels": result["labels"][keep].cpu().tolist(),
                    "gt_labels": annotation["labels"].tolist()
                })
                
        self._save_predictions(detections, save_path)
        return detections
    
    @staticmethod
    def _save_predictions(predictions: Any, save_path: str):
        """Saves predictions in JSON format."""
        file_path = os.path.join(save_path, "predictions.json")
        with open(file_path, "w") as file:
            json.dump(predictions, file, indent=2)
        print(f"Predictions saved to {file_path}")


# Example usage
if __name__ == "__main__":
    config = Config()
    model_weights = config.checkpoint_path
    output_directory = config.predictions_path
    
    bdd_data_loader = BDD100KDataLoader(
        train_json=config.train_annotations,
        val_json=config.val_annotations,
        train_img_dir=config.train_image_path,
        val_img_dir=config.val_image_path,
        batch_size=config.batch_size
    )

    _, val_dataloader, _ = bdd_data_loader.get_dataloaders(subset=True)

    detector = ObjectDetectionInference(model_weights)
    inference_results = detector.predict(val_dataloader, save_path=output_directory)
