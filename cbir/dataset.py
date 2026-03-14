import os

import torch
from torchvision import io, transforms
from torch.utils.data import Dataset


class CbirDataset(Dataset):

    def __init__(self, root_dir: str, transform: transforms.Compose | None = None):
        self.root_dir = root_dir
        self.image_files = sorted(
            file for file in os.listdir(root_dir) 
                if file.endswith(("jpg", "jpeg"))
        )

        self.transform = transform
        if not transform:
            self.transform = transform = transforms.Compose([
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(os.listdir(self.root_dir))
    
    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_files[index]
        image = self.transform(io.read_image(image_path)) # type: ignore
        return image
