import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from config import Config

class FasterRCNNModel:
    """Wrapper class for Faster R-CNN model with custom class handling."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Initialize the Faster R-CNN model.

        :param num_classes: Number of object classes including background.
        :param pretrained: Whether to use pretrained weights.
        """
        self.num_classes = num_classes
        self.model = self._initialize_model(pretrained)

    def _initialize_model(self, pretrained: bool):
        """
        Create a Faster R-CNN model with a modified classifier.

        :param pretrained: Whether to use pretrained weights.
        :return: Faster R-CNN model.
        """
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model.cpu()

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from a checkpoint file.

        :param checkpoint_path: Path to the model checkpoint.
        """
        checkpoint = torch.load(checkpoint_path)

        num_classes_checkpoint = checkpoint["model"]["roi_heads.box_predictor.cls_score.weight"].shape[0]
        print(f"Checkpoint was trained with {num_classes_checkpoint} total classes (including background).")

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def get_model(self):
        """
        Get the modified Faster R-CNN model.

        :return: Faster R-CNN model with updated classifier.
        """
        return self.model


# Example Usage
if __name__ == "__main__":
    config = Config()

    model_instance = FasterRCNNModel(num_classes=config.num_classes)
    model_instance.load_checkpoint(config.checkpoint_path)
    model = model_instance.get_model()
