import os
import json
import pickle
import torch
import numpy as np
import math
import re
from collections import defaultdict, Counter
from tqdm import tqdm
from typing import List, Dict
from transformers import GPT2Tokenizer
from train import GPT2CaptionModel
from inference import beam_search

def print_gpu_info(device):
    if device.type == 'cuda':
        print(f"\n>>> Using GPU: {torch.cuda.get_device_name(0)}")
        print(f">>> CUDA Version: {torch.version.cuda}")
        print(f">>> VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB\n")
    else:
        print("\n>>> Warning: No GPU detected - Using CPU\n")

class CaptionEvaluator:
    def __init__(self, model_weights: str, val_data_path: str, results_dir: str = "results", 
                 device: str = None, beam_k: int = 3, max_length: int = 60, temperature: float = 1.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model_weights = model_weights
        self.val_data_path = val_data_path
        self.results_dir = results_dir
        self.beam_k = beam_k
        self.max_length = max_length
        self.temperature = temperature
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.dataset_name = os.path.splitext(os.path.basename(val_data_path))[0]
        
        self.embeddings_cache = os.path.join(results_dir, f"{self.dataset_name}_embeddings.pkl")
        self.captions_cache = os.path.join(results_dir, f"{self.dataset_name}_captions_beam{beam_k}.pkl")
        self.metadata_cache = os.path.join(results_dir, f"{self.dataset_name}_metadata_beam{beam_k}.json")
    
    def _check_cache_valid(self):
        if not all(os.path.exists(f) for f in [self.embeddings_cache, self.captions_cache, self.metadata_cache]):
            return False
            
        try:
            with open(self.metadata_cache, 'r') as f:
                metadata = json.load(f)
                return (metadata.get('dataset_name') == self.dataset_name and 
                        metadata.get('beam_k') == self.beam_k)
        except:
            return False
    
    def _save_metadata(self):
        metadata = {
            'dataset_name': self.dataset_name,
            'model_weights': self.model_weights,
            'beam_k': self.beam_k,
            'max_length': self.max_length,
            'temperature': self.temperature,
            'device': str(self.device)
        }
        with open(self.metadata_cache, 'w') as f:
            json.dump(metadata, f)
    
    def load_model_and_data(self, subset_size: int = None):
        print("Loading model and data...")
        print_gpu_info(self.device)
        
        with open(self.val_data_path, 'rb') as f:
            val_data = pickle.load(f)
        
        if subset_size is not None:
            clip_embedding = dict(list(val_data['clip_embedding'].items())[:subset_size])
            subset_ids = list(clip_embedding.keys())
            
            captions = defaultdict(list)
            for caption_dict in val_data['captions']:
                img_id = caption_dict['image_id']
                if img_id in subset_ids:
                    captions[img_id].append(caption_dict['caption'])
        else:
            clip_embedding = val_data['clip_embedding']
            captions = defaultdict(list)
            for caption_dict in val_data['captions']:
                captions[caption_dict['image_id']].append(caption_dict['caption'])
        
        model = GPT2CaptionModel().to(self.device)
        model.load_state_dict(torch.load(self.model_weights, map_location=self.device))
        model.eval()
        
        return model, {
            'clip_embedding': clip_embedding,
            'captions': dict(captions)
        }
    
    def generate_captions(self, model, data, batch_size=32):
        if self._check_cache_valid():
            print("Loading cached embeddings and captions...")
            with open(self.embeddings_cache, 'rb') as f:
                all_prefixes, img_ids = pickle.load(f)
            with open(self.captions_cache, 'rb') as f:
                return pickle.load(f)
        
        results = []
        all_embeddings = []
        img_ids = []
        
        for img_id, clip_embed in data['clip_embedding'].items():
            if not isinstance(clip_embed, torch.Tensor):
                clip_embed = torch.tensor(clip_embed, device=self.device)
            all_embeddings.append(clip_embed)
            img_ids.append(img_id)
        
        all_prefixes = []
        for i in tqdm(range(0, len(all_embeddings), batch_size), desc="Computing prefixes"):
            batch_embeddings = torch.stack(all_embeddings[i:i+batch_size]).to(self.device).float()
            
            with torch.no_grad():
                batch_prefixes = model.prefix_projection(batch_embeddings)
                all_prefixes.extend([p for p in batch_prefixes])
        
        print(f"Generating captions with beam search")
        for img_id, prefix in tqdm(zip(img_ids, all_prefixes), 
                                 desc="Generating captions", 
                                 total=len(img_ids)):
            with torch.no_grad():
                caption = beam_search(
                    model, 
                    self.tokenizer, 
                    prefix.unsqueeze(0),
                    beam_k=self.beam_k,
                    max_length=self.max_length,
                    temperature=self.temperature
                )
                
            results.append({
                'image_id': img_id,
                'caption': caption
            })
        
        with open(self.embeddings_cache, 'wb') as f:
            pickle.dump((all_prefixes, img_ids), f)
        with open(self.captions_cache, 'wb') as f:
            pickle.dump(results, f)
        self._save_metadata()
        
        return results

    def preprocess_text(self, text: str) -> List[str]:
        text = re.sub(r'<|endoftext|>', '', text)
        tokens = self.tokenizer.tokenize(text)
        return [t for t in tokens if t not in self.tokenizer.all_special_tokens]
    
    def calculate_bleu(self, refs_list: List[List[str]], hyps: List[str]) -> float:
        bleu_scores = []
        
        for refs, hyp in zip(refs_list, hyps):
            hyp_tokens = self.preprocess_text(hyp)
            refs_tokens = [self.preprocess_text(ref) for ref in refs]
            
            # Brevity penalty
            hyp_len = len(hyp_tokens)
            closest_ref_len = min(len(ref) for ref in refs_tokens)
            bp = 1.0 if hyp_len > closest_ref_len else math.exp(1 - closest_ref_len/hyp_len)
            
            # n-gram precision
            precisions = []
            for i in range(1, 5):  # 1-4 grams
                hyp_ngrams = Counter(zip(*[hyp_tokens[j:] for j in range(i)]))
                max_ref_counts = {}
                
                for ref in refs_tokens:
                    ref_ngrams = Counter(zip(*[ref[j:] for j in range(i)]))
                    for ngram in hyp_ngrams:
                        max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), ref_ngrams.get(ngram, 0))
                
                clipped_counts = sum(min(hyp_ngrams[ngram], max_ref_counts[ngram]) for ngram in hyp_ngrams)
                total_counts = max(1, sum(hyp_ngrams.values()))
                precisions.append(clipped_counts / total_counts)
            
            if min(precisions) > 0:
                p_log_sum = sum(math.log(p) for p in precisions) / 4
                geo_mean = math.exp(p_log_sum)
            else:
                geo_mean = 0
                
            bleu = bp * geo_mean
            bleu_scores.append(bleu)
        
        return np.mean(bleu_scores)

    def compute_metrics(self, data: Dict, generated: List[Dict]) -> Dict[str, float]:
        refs_list, hyps = [], []
        
        for item in generated:
            img_id = item['image_id']
            hyps.append(item['caption'])
            refs = data['captions'].get(img_id, ["<no_caption>"])
            refs_list.append(refs)
        
        return {'BLEU-4': self.calculate_bleu(refs_list, hyps)}

def main():
    config = {
        'model_weights': "F:/checkpoints/transformer_mapper_weights.pt",
        'val_data_path': "F:/coco_pickle2014/val2014_processed.pkl",
        'results_dir': "F:/results",
        'subset_size': None,
        'batch_size': 40,
        'beam_k': 3,
        'max_length': 30,
        'temperature': 1.0
    }
    
    evaluator = CaptionEvaluator(
        model_weights=config['model_weights'],
        val_data_path=config['val_data_path'],
        results_dir=config['results_dir'],
        beam_k=config['beam_k'],
        max_length=config['max_length'],
        temperature=config['temperature']
    )
    
    model, val_data = evaluator.load_model_and_data(config['subset_size'])
    generated = evaluator.generate_captions(model, val_data, batch_size=config['batch_size'])
    scores = evaluator.compute_metrics(val_data, generated)
    
    results_path = os.path.join(config['results_dir'], f"final_metrics_beam{config['beam_k']}.json")
    with open(results_path, 'w') as f:
        json.dump(scores, f, indent=2)
    
    print("\nFinal Metrics:")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()