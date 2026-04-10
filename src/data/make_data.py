import os
import urllib.request
import zipfile
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATASETS = {
    "Flickr8k_Images": {
        "url": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
        "extract_path": os.path.join(DATA_DIR, "flickr8k", "images")
    },
    "Flickr8k_Text": {
        "url": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
        "extract_path": os.path.join(DATA_DIR, "flickr8k", "text")
    },
    "MSCOCO_Annotations": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        "extract_path": os.path.join(DATA_DIR, "mscoco")
    },
    # "MSCOCO_Train_Images": {
    #    "url": "http://images.cocodataset.org/zips/train2014.zip",
    #     "extract_path": os.path.join(DATA_DIR, "mscoco", "images", "train")
    # }, ## 13GB file size !!
    # "MSCOCO_Val_Images": {
    #    "url": "http://images.cocodataset.org/zips/val2014.zip",
    #     "extract_path": os.path.join(DATA_DIR, "mscoco", "images", "val")
    # } ## 6GB file size !!
}

def download_progress(count, block_size, total_size):
    if total_size == -1:
        return
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading... {percent}%")
    sys.stdout.flush()

def make_data():
    print(f"Starting data preparation in {DATA_DIR}/ ...\n")
    os.makedirs(DATA_DIR, exist_ok=True)

    for name, info in DATASETS.items():
        url = info["url"]
        extract_path = info["extract_path"]
        zip_path = os.path.join(DATA_DIR, f"{name}.zip")

        print(f"[{name}]")
        
        if not os.path.exists(extract_path):
            if not os.path.exists(zip_path):
                print(f"Connecting and downloading...")
                urllib.request.urlretrieve(url, zip_path, reporthook=download_progress)
                print("\nDownload complete!")
            else:
                print("Zip file already exists. Continuing...")

            print(f"Extracting to {extract_path}/ ...")
            os.makedirs(extract_path, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            os.remove(zip_path)
            print(f"Removed {name}.zip\n")
        else:
            print(f"Folder {extract_path} is ready (skipping download)\n")

    print("Data preparation complete!")

if __name__ == "__main__":
    make_data()
