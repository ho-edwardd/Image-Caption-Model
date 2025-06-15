import json
import pickle
import torch
from PIL import Image
import clip
from tqdm import tqdm
from transformers import GPT2Tokenizer
import ssl
from pathlib import Path

# Disable SSL verification for CLIP download
ssl._create_default_https_context = ssl._create_unverified_context

class COCOPreprocessor:
    """Processes raw COCO dataset into tokenized captions and CLIP embeddings."""
    
    def __init__(self, data_root: str, output_dir: str, 
                 clip_model_type: str = "ViT-B/32",
                 gpt2_type: str = "gpt2"):
        self.data_root = Path(data_root).absolute()
        self.output_dir = Path(output_dir).absolute()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(clip_model_type, device=self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
    
    def _get_image_path(self, split: str, image_id: int) -> Path:
        """Returns correct path format based on COCO version (2014 vs 2017)"""
        if "2017" in split:
            return self.data_root / split / f"{image_id:012d}.jpg"
        else:  # COCO 2014 format
            return self.data_root / split / f"COCO_{split}_{image_id:012d}.jpg"
    
    def process_dataset(self, split: str):
        """Process a COCO split into tokenized captions and CLIP embeddings."""
        # Load annotations
        ann_file = self.data_root / "annotations" / f"captions_{split}.json"
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        # Get first caption per image
        image_id_to_caption = {}
        for ann in annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in image_id_to_caption:
                image_id_to_caption[image_id] = ann['caption']
        
        # Process images
        results = {
            "clip_embedding": {},
            "captions": [],
            "tokens": []
        }
        
        for image_id, caption in tqdm(image_id_to_caption.items(), desc=f"Processing {split}"):
            image_path = self._get_image_path(split, image_id)
            
            try:
                # Get CLIP embedding
                image = Image.open(str(image_path)).convert("RGB")
                image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.clip_model.encode_image(image_tensor).cpu().float()[0]
                
                # Tokenize caption
                tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
                
                # Store results
                results["clip_embedding"][image_id] = embedding
                results["captions"].append({
                    "image_id": image_id,
                    "caption": caption
                })
                results["tokens"].append(tokens)
                
            except Exception as e:
                print(f"Skipping {image_path}: {str(e)}")
        
        # Save raw data
        output_path = self.output_dir / f"{split}_processed.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Saved {len(results['captions'])} items to {output_path}")

if __name__ == "__main__":
    # Example usage for both versions
    preprocessor = COCOPreprocessor(
        data_root="F:/coco2017",
        output_dir="F:/coco_pickle2017"
    )
    preprocessor.process_dataset("val2017") # change to train2014 or val2014, based on which dataset you want to preprocess, same with replacing for 2017 for each