import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import json

from config import Config

class BDD100KDataset(Dataset):
    """Custom Dataset for BDD100K Object Detection."""

    def __init__(
        self,
        image_path: str,
        annotation_file: str,
        transform: transforms.Compose = None
    ) -> None:
        self.image_path = image_path
        self.transform = transform
        config = Config()
        self.category_map = config.class_to_idx
        with open(annotation_file, 'r', encoding='utf-8') as file:
            self.annotation_data = json.load(file)
        
        self.image_data = {
            entry['name']: entry for entry in self.annotation_data
        }

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, idx: int):
        image_name = list(self.image_data.keys())[idx]
        image_path = os.path.join(self.image_path, image_name)
        
        image = Image.open(image_path).convert('RGB')
        annotations = self.image_data[image_name]
        
        bboxes, labels = [], []
        
        for label in annotations['labels']:
            if "box2d" in label and "category" in label:
                bbox = label["box2d"]
                category = label['category']
                
                bboxes.append([
                    bbox['x1'], bbox['y1'],
                    bbox['x2'], bbox['y2']
                ])
                labels.append(self.category_map[category])

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, {"boxes": bboxes, "labels": labels}

class BDD100KDataLoader:
    """Factory class for creating DataLoaders for BDD100K."""

    def __init__(
        self, train_json: str, val_json: str, train_img_dir: str, val_img_dir: str,
        batch_size: int = 8, num_workers: int = 4
    ) -> None:
        self.train_json = train_json
        self.val_json = val_json
        self.train_img_dir = train_img_dir
        self.val_img_dir = val_img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_transform = transforms.Compose([transforms.ToTensor()])

    def get_dataloaders(self, subset: bool = False):
        """Returns train and validation DataLoaders."""
        train_dataset = BDD100KDataset(self.train_img_dir, self.train_json, self.transform)
        val_dataset = BDD100KDataset(self.val_img_dir, self.val_json, self.val_transform)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=lambda x: tuple(zip(*x))
        )

        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=self.num_workers, collate_fn=lambda x: tuple(zip(*x))
        )

        if subset:
            import random
            subset_size = int(0.1 * len(train_dataset))
            subset_indices = random.sample(range(len(train_dataset)), subset_size)

            subset_dataset = Subset(train_dataset, subset_indices)
            train_subset_dataloader = DataLoader(
                subset_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
            )

            return train_loader, val_loader, train_subset_dataloader

        return train_loader, val_loader
