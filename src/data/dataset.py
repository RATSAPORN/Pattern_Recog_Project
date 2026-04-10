import json
import os
from PIL import Image
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    def __init__(
        self,
        image_dir,
        annotation_json,
        split_file=None,
        transform=None
    ):
        self.image_dir = image_dir
        self.transform = transform

        with open(annotation_json, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        if split_file:
            with open(split_file) as f:
                valid_images = set(x.strip() for x in f)

            annotations = {
                k: v for k, v in annotations.items()
                if k in valid_images
            }

        self.samples = []

        for image_name, captions in annotations.items():
            for caption in captions:
                self.samples.append((image_name, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_name, caption = self.samples[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption