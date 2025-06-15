import os
import zipfile
import requests
from tqdm import tqdm

base_dir = r"F:/coco2017"  # Change this to your directory location, if you want it on the flash drive, just change this to the flash drive directory location
# this will choose the folder location and where to download and unzip the files at

os.makedirs(base_dir, exist_ok=True)

image_urls = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "test2017": "http://images.cocodataset.org/zips/test2017.zip"
}
annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

def download_with_progress(url, destination):
    """Download a file with a progress bar."""
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 8192
    zip_path = os.path.join(destination, os.path.basename(url))

    with open(zip_path, "wb") as f, tqdm(
        desc=os.path.basename(url),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    return zip_path

def extract_with_progress(zip_path, destination):
    """Extract a zip file with a progress bar."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        total_files = len(zip_ref.infolist())
        with tqdm(total=total_files, desc="Extracting", unit="file") as pbar:
            for file in zip_ref.infolist():
                zip_ref.extract(file, destination)
                pbar.update(1)

    os.remove(zip_path) 

print("Downloading annotations...")
annotation_zip = download_with_progress(annotation_url, base_dir)
extract_with_progress(annotation_zip, base_dir)

for split, url in image_urls.items():
    print(f"Downloading {split} images...")
    image_zip = download_with_progress(url, base_dir)
    extract_with_progress(image_zip, base_dir)

print("Dataset downloaded and extracted successfully!")