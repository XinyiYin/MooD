import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class EmoSetVADataset(Dataset):
    def __init__(self, image_dir, annotation_dir, image_ids=None, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform

        if image_ids is not None:
            self.image_ids = image_ids
        else:
            self.image_ids = []
            for fname in sorted(os.listdir(annotation_dir)):
                if fname.endswith('.json'):
                    img_id = fname.replace('.json', '')
                    img_path = os.path.join(image_dir, f"{img_id}.jpg")
                    if os.path.exists(img_path):
                        self.image_ids.append(img_id)

        print(f"Dataset: {len(self.image_ids)} samples loaded")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        ann_path = os.path.join(self.annotation_dir, f"{img_id}.json")
        with open(ann_path, 'r') as f:
            ann = json.load(f)

        valence = float(ann['valence'])
        arousal = float(ann['arousal'])
        va = torch.tensor([valence, arousal], dtype=torch.float32)

        return image, va, img_id


def get_transforms(split='train'):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])