import os
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import CaptionDataset


def get_flickr8k_dataloaders(
    data_dir,
    batch_size=32,
    num_workers=4
):

    image_dir = os.path.join(
        data_dir,
        "flickr8k/images/Flicker8k_Dataset"
    )

    annotation_json = os.path.join(
        data_dir,
        "flickr8k_annotations.json"
    )

    train_split = os.path.join(
        data_dir,
        "flickr8k/text/Flickr_8k.trainImages.txt"
    )

    val_split = os.path.join(
        data_dir,
        "flickr8k/text/Flickr_8k.devImages.txt"
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = CaptionDataset(
        image_dir,
        annotation_json,
        train_split,
        transform
    )

    val_dataset = CaptionDataset(
        image_dir,
        annotation_json,
        val_split,
        transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader

def get_mscoco_dataloaders(
    data_dir,
    split="train",
    batch_size=32,
    num_workers=4
):

    if split not in ["train", "val"]:
        raise ValueError("split must be 'train' or 'val'")

    image_dir = os.path.join(data_dir, f"mscoco/images/{split}", f"{split}2014")
    annotation_json = os.path.join(data_dir, f"mscoco_{split}_annotations.json")

    split_file = None  

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = CaptionDataset(image_dir, annotation_json, split_file, transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=num_workers)

    return loader

def main():

    flickr8k_train_loader, flickr8k_val_loader = get_flickr8k_dataloaders("data")

    print("flickr8k")
    for images, captions in flickr8k_train_loader:

        print("Images Size :", images.shape)
        print("Example Caption :")
        for caption in captions[:3]:
            print(caption)

        break

    # (mscoco dataloader, need to load mscoco image first)
    
    # mscoco_train_loader = get_mscoco_dataloaders("data", "train")
    # mscoco_val_loader = get_mscoco_dataloaders("data", "val")

    # print("mscoco")
    # for images, captions in mscoco_train_loader:

    #     print("Images Size :", images.shape)
    #     print("Example Caption :")
    #     for caption in captions[:3]:
    #         print(caption)

    #     break

if __name__ == "__main__":
    main()
