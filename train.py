import os
import pickle
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Optional

class ClipCocoDataset(Dataset):
    def __init__(self, data_path: str, prefix_length: int = 10, 
                 gpt2_type: str = "gpt2", normalize_clip_embed: bool = False,
                 subset_size: int = None):  # Add subset_size parameter
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_clip_embed = normalize_clip_embed
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create random subset
        if subset_size is not None:
            indices = torch.randperm(len(data["captions"]))[:subset_size].tolist()
            self.captions = [data["captions"][i] for i in indices]
            self.captions_tokens = [data["tokens"][i] for i in indices]
            
            # Get corresponding CLIP embeddings
            img_ids = {c["image_id"] for c in self.captions}
            self.clip_embeddings = {
                k: v for k, v in data["clip_embedding"].items() 
                if k in img_ids
            }
        else:
            self.clip_embeddings = data["clip_embedding"]
            self.captions = data["captions"]
            self.captions_tokens = data["tokens"]
        
        # Calculate max sequence length
        all_len = torch.tensor([len(t) for t in self.captions_tokens]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
    
    def __len__(self):
        return len(self.captions_tokens)
    
    def __getitem__(self, idx):
        # Pad/truncate tokens
        tokens = self.captions_tokens[idx]
        padding = self.max_seq_len - len(tokens)
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        
        # Create mask
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask))
        
        # Get CLIP embedding
        img_id = self.captions[idx]["image_id"]
        clip_embed = self.clip_embeddings[img_id]
        if self.normalize_clip_embed:
            clip_embed = clip_embed / clip_embed.norm(2, -1)
            
        return tokens, mask, clip_embed

class MLPTrans(nn.Module):
    """Multi-layer perceptron (MLP) for transformer layers."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: Optional[int] = None, act = nnf.relu, dropout = 0.):
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias = True, dropout = 0.):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 1/sqrt(k) scaling factor
        self.scale = self.head_dim ** -0.5

        # initialize q,k,v linear matrices for training
        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias = bias)
        
        # initialize W matrix for reprojection to original dimensions
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask = None):
        # in this, sequence length is simply how tall our matrix will be due to linear projection.
        # so if we use 40 images a batch and linear project 1 embed to 10, then we have 40, 10, 512
        # embedding for image is [n, embed_dim]
    
        batch_size, n, _ = x.shape
        
        # This is to produce our q = Q*X, k = K*X, v = V*X
        qkv = self.to_qkv(x).chunk(3, dim=-1) 
        q, k, v = map(lambda t: t.view(batch_size, n, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # ((QX)^T * (KX)) / sqrt(d)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        # maybe dont need this? because this is for cross attention maybe
        if mask is not None:
            if mask.dim() == 2:  # (batch_size, n)
                mask = mask.unsqueeze(1)  # Add head dimension
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Z = A * VX
        out = attn @ v
        
        # Reproject using W^T matrix on Z (W * Z^T)
        out = out.transpose(1, 2).reshape(batch_size, n, self.embed_dim)
        out = self.output_proj(out)
        
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio = 4., bias = False, dropout = 0.,
                 actf = nnf.relu, layer_norm: nn.Module = nn.LayerNorm):
        super().__init__()
        # layer norm
        self.norm1 = nn.LayerNorm(embed_dim)
        # multi-head attention
        self.attn = MultiHeadAttention(embed_dim, num_heads, bias = bias, dropout = dropout)
        # layer norm
        self.norm2 = nn.LayerNorm(embed_dim)
        # final Feed-forward network for output
        self.mlp = MLPTrans(embed_dim, (embed_dim * mlp_ratio), act = actf, dropout = dropout)
        
    def forward(self, x, mask = None):
        # this the residual connection portion, next t_3 + t_1, t_5 + t_3, etc.
        # one residual connection with the initial input and output of the first layer norm/attention step
        x = x + self.attn(self.norm1(x), mask)
        # second residual connection with the output of the first layer norm/attention step and 
        # last layer norm/mlp(ffn) step.
        x = x + self.mlp(self.norm2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        # creates a list of Transformer Blocks
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, embed_dim, num_heads)
                                     for _ in range(num_layers)])
        
    def forward(self, x, mask = None):
        # recursively iterates through the Transformer Blocks
        # the transformer architecture consist of feeding the output of i-1 block as input into block i 
        # and returns the result of the final block as the final output
        for layer in self.layers:
            x = layer(x, mask = mask)
        return x

class MappingNetwork(nn.Module):
    def __init__(self, clip_embed_dim, gpt_embed_dim, prefix_length, num_heads, num_layers):
        super().__init__()
        
        # how long you want your prefix to be
        self.prefix_length = prefix_length
        
        # this is used for linearly projecting the [1 x 512] image embed into a [10 x 768] embed
        self.clip_linear = nn.Linear(clip_embed_dim, prefix_length * gpt_embed_dim)
        
        # initialized a random [10 x 768] embedding for concatenation
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, gpt_embed_dim), requires_grad=True)
        
        # set the transformer architecture to be called
        self.transformer = Transformer(gpt_embed_dim, num_heads, num_layers)
        
    def forward(self, clip_embed):
        
        batch_size = clip_embed.size(0)
        
        # intial image embedding:[batch size, 1, 512] -> intermediate step:[batch size, 1, 7680] -> final shape:[batch size, 10, 768]
        clip_project = self.clip_linear(clip_embed).view(batch_size, self.prefix_length, -1)
        
        # concanetate the constant matrix and clip projection matrix to create a [batch size, 20, 768] dim
        # prefix_const is initially size [10, 768], must manipulate it as [batch size, 10, 768] to fit
        combined = torch.cat([clip_project, 
                              self.prefix_const.unsqueeze(0).expand(batch_size, -1, -1)], dim = 1)
        
        # select only the last 10 "tokens," or the constants.
        prefix = self.transformer(combined)[:, self.prefix_length:]
        return prefix

class GPT2CaptionModel(nn.Module):
    def __init__(self, clip_embed_dim: int = 512, prefix_length: int = 10, num_heads: int = 8, num_layers: int = 8):
        super().__init__()
        # this is how long you want your prefixes, default is 10
        self.prefix_length = prefix_length
        # initialize the pretrained GPT-2 model
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        # this sets the default embedding size of GPT-2 embeddings (default 768)
        self.gpt_embed_size = self.gpt.transformer.wte.weight.shape[1]
        # initialize the Mapping Network (transformer architecture) with parameters for later use
        self.prefix_projection = MappingNetwork(clip_embed_dim, self.gpt_embed_size, prefix_length, num_heads, num_layers)
        
        
    def forward(self, tokens: torch.Tensor, clip_embed: torch.Tensor,
                mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        
        batch_size = tokens.size(0)
        # concatenate the prefix embeddings and truth-label token embeddings
        # this turns each token into embeddings in the GPT-2 space, so [batch, sentence length, 768]
        caption_embed = self.gpt.transformer.wte(tokens)
        # this will turn the clip embedding [batch, 1, 512] into an appropriate prefix embedding [batch, 10, 768]
        prefix_project = self.prefix_projection(clip_embed)
        
        # concatenate the prefixes to the beginning of the token embeddings
        gpt_embedding_cat = torch.cat((prefix_project, caption_embed), dim=1)
        
        # create the labels by filling the labels of prefixes with -100 and tokens with their default positive integers
        if labels is not None:
            dummy_labels = torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device = tokens.device)
            labels = torch.cat((dummy_labels, tokens), dim=1)
        
        # feed the embeddings, labels, and mask into the GPT model
        # this model will skip predict every position including the prefix using teacher forcing, but only the loss
        # for the truth-label tokens position, not prefix position, will considered.
        # labels are also used for positional offset, to ensure that we are predicting the t+1 token using prefix + t tokens
        out = self.gpt(inputs_embeds=gpt_embedding_cat, labels=labels, attention_mask=mask)
        
        return out
    
class CLIPPrefixCaption(GPT2CaptionModel):

    def parameters(self, recurse: bool = True):
        return self.prefix_projection.parameters()

    def train(self, mode: bool = True):
        super(GPT2CaptionModel, self).train(mode)
        self.gpt.eval()
        return self

def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.output_dir, f"{args.output_prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)

def train(dataset: ClipCocoDataset, model: CLIPPrefixCaption, args, output_dir: str = ".", output_prefix: str = ""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    epochs = args.epochs
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, 
        num_training_steps=epochs * len(train_dataloader)
    )
    
    save_config(args)  # save config before training
    
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, clip_embed) in enumerate(train_dataloader):
            tokens = tokens.to(device)
            mask = mask.to(device)
            clip_embed = clip_embed.to(device).float()
            
            optimizer.zero_grad()
            
            # forward pass without labels to get logits
            outputs = model(tokens, clip_embed, mask, labels=None)
            logits = outputs.logits
            
            # calculate loss on caption tokens only
            logits = logits[:, args.prefix_length-1:-1]  # Exclude prefix and last token
            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), 
                tokens.reshape(-1), 
                ignore_index=0
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            
        progress.close()
            
    torch.save(
        model.state_dict(),
        os.path.join(output_dir, "transformer_mapper_weights.pt"),
        _use_new_zipfile_serialization=True
    )
    return model

# We'll do a 80/20 split, we'll train on train2014, train2017, and val2017. However, we will use val2014 as our testing set.
def main():
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument('--data_path', default="F:/coco_pickle2014/train2014_processed.pkl")
    parser.add_argument('--output_dir', default='F:/checkpoints')
    parser.add_argument('--output_prefix', default='training_config')
    parser.add_argument('--subset_size', type=int, default=None)  # leave this as None to use full dataset
    parser.add_argument('--pretrain', type=bool, default=False) # set to False if you want to train from scratch, True if you want to use previously saved weights
    parser.add_argument('--pretrained_weights', default='F:/checkpoints/transformer_mapper_weights.pt') # location of presaved weights
    # Model arguments
    parser.add_argument('--prefix_length', type= int, default=10)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=8)
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--clip_embed_dim', type=int, default=512)
    parser.add_argument('--normalize_clip_embed', type=bool, default=False)
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # initializing model
    model = CLIPPrefixCaption(
        clip_embed_dim=args.clip_embed_dim,
        prefix_length=args.prefix_length,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    if args.pretrain:
        if os.path.exists(args.pretrained_weights):
            print(f"Loading pretrained weights from {args.pretrained_weights}")
            model.load_state_dict(torch.load(args.pretrained_weights))
        else:
            raise FileNotFoundError(
                f"Pretrained weights not found at {args.pretrained_weights} "
                "when pretrain=True was specified"
            )
    else:
        print("Training from scratch (pretrain=False)")
    
    # creating dataset
    dataset = ClipCocoDataset(
        args.data_path, 
        prefix_length=args.prefix_length,
        normalize_clip_embed=args.normalize_clip_embed,
        subset_size=args.subset_size
    )
    
    train(dataset, model, args, output_dir=args.output_dir, output_prefix=args.output_prefix)

if __name__ == '__main__':
    main()