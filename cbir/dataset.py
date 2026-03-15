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
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        filename = self.image_files[index]
        image_path = os.path.join(self.root_dir, filename)
        image = self.transform(io.read_image(image_path)) # type: ignore
        image_id = filename.split('.')[0]
        return image, image_id
