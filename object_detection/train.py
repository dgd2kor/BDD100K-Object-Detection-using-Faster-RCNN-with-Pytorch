import torch
from faster_rcnn_bdd100k import FasterRCNNModel
from bdd100k_dataset import BDD100KDataLoader
from config import Config

class Trainer:
    """Trainer class for training Faster R-CNN on BDD100K dataset."""
    
    def __init__(self, model, train_loader, val_loader, device, optimizer, lr=0.001, momentum=0.9, weight_decay=0.0005):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = None  # Faster R-CNN has an internal loss calculation

    def train_one_epoch(self):
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for images, targets in self.train_loader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)

    def train(self, epochs):
        """Train the model for given epochs."""
        for epoch in range(epochs):
            loss = self.train_one_epoch()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")


if __name__ == "__main__":
    config = Config()

    bdd_data_loader = BDD100KDataLoader(
        train_json=config.train_annotations,
        val_json=config.val_annotations,
        train_img_dir=config.train_image_path,
        val_img_dir=config.val_image_path,
        batch_size=config.batch_size
    )

    train_dataloader, val_dataloader, sub_train_dataloader = bdd_data_loader.get_dataloaders(subset=True)

    model_instance = FasterRCNNModel(num_classes=config.num_classes)
    model_instance.load_checkpoint(config.checkpoint_path)
    model = model_instance.get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if config.subset_train:
        trainer = Trainer(
            model=model,
            train_loader=sub_train_dataloader,
            val_loader=val_dataloader,
            optimizer=optimizer,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    else:
        trainer = Trainer(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    trainer.train(epochs=config.epochs)