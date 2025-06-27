import os
import random
from pycocotools.coco import COCO
from PIL import Image

base_dir = r"/Volumes/WORK/coco" # Change this to your directory location of where your data is stored
annotations_dir = os.path.join(base_dir, "annotations")
images_dir = os.path.join(base_dir, "train2014")  # Change to "val2014" or "test2014" if needed

annotations_file = os.path.join(annotations_dir, "captions_train2014.json")  # Change to "captions_val2014.json" for validation set captions

# Initializing COCO API
coco = COCO(annotations_file)

# Retrieve image Ids
image_ids = coco.getImgIds()

random_image_id = random.choice(image_ids)

# Loading image information
image_info = coco.loadImgs(random_image_id)[0]

# Loading the image
image_path = os.path.join(images_dir, image_info["file_name"])

# Load captions for the image
caption_ids = coco.getAnnIds(imgIds=random_image_id)
captions = coco.loadAnns(caption_ids)

print(f"Image ID: {random_image_id}")
print(f"Image File: {image_info['file_name']}")
print(f"Image Size: {image_info['width']}x{image_info['height']}")

print("\nGround Truth Captions:")
for i, caption in enumerate(captions):
    print(f"Caption {i + 1}: {caption['caption']}")

try:
    with Image.open(image_path) as img:
        img.verify()
        img.close()
        Image.open(image_path).show()
except Exception as e:
    print(f"Warning: Image file {image_info['file_name']} is corrupted or cannot be opened: {e}")