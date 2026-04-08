import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

def preprocess_flickr8k(token_file, output_file):
    print(f"Processing Flickr8k: {token_file}")
    annotations = {}
    if not os.path.exists(token_file):
        print(f"Warning: {token_file} not found.")
        return

    with open(token_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            
            image_id_full = parts[0]
            caption = parts[1]
            image_name = image_id_full.split('#')[0]
            
            if image_name not in annotations:
                annotations[image_name] = []
            annotations[image_name].append(caption)
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=4)
    print(f"Saved {len(annotations)} images to {output_file}")

def preprocess_mscoco(json_file, output_file):
    print(f"Processing MS COCO: {json_file}")
    if not os.path.exists(json_file):
        print(f"Warning: {json_file} not found.")
        return

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    annotations = {}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        filename = id_to_filename.get(img_id)
        if not filename:
            continue
        caption = ann['caption']
        
        if filename not in annotations:
            annotations[filename] = []
        annotations[filename].append(caption)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=4)
    print(f"Saved {len(annotations)} images to {output_file}")

def main():
    # Flickr8k
    flickr_token = os.path.join(DATA_DIR, "flickr8k", "text", "Flickr8k.token.txt")
    flickr_output = os.path.join(DATA_DIR, "flickr8k_annotations.json")
    preprocess_flickr8k(flickr_token, flickr_output)

    # MS COCO Train
    coco_train_json = os.path.join(DATA_DIR, "mscoco", "annotations", "captions_train2014.json")
    coco_train_output = os.path.join(DATA_DIR, "mscoco_train_annotations.json")
    preprocess_mscoco(coco_train_json, coco_train_output)

    # MS COCO Val
    coco_val_json = os.path.join(DATA_DIR, "mscoco", "annotations", "captions_val2014.json")
    coco_val_output = os.path.join(DATA_DIR, "mscoco_val_annotations.json")
    preprocess_mscoco(coco_val_json, coco_val_output)

if __name__ == "__main__":
    main()
