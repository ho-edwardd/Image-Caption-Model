import argparse
import clip
import os
import random
import torch
import json
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
from PIL import Image
from train import GPT2CaptionModel

def greedy_search(model, tokenizer, embed=None, stop_token: str=".",
                  temperature=1.0, max_length: int = 60):
    # puts model in evaluation mode so that no training is performed.
    model.eval()
    # tokenizes the stop token ".", so if the next predicted token is ".", then it'll end the sentence and break the loop
    stop_token_index = tokenizer.encode(stop_token)[0]
    
    with torch.no_grad():
        # initialize the inputs for the GPT model to make predictions (these inputs are our prefixes)
        input = embed
        # initialize an empty sequence of tokens which will be turned into our caption, this will be filled with the predicted tokens
        tokens = None
        
        for _ in range(max_length):
            # processes the prefixes and generates logits for each 10 prefixes and the next 10 + i-th position
            outputs = model.gpt(inputs_embeds=input)
            # select the i-th position/word that is to be predicted
            # and select the logits from it that describe the probability for each word in the English lexicon
            # there will be a respective logit for each token, about 50257 tokens or logits
            logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            # softmax the logits since probabilities can be so small and so close, it can strain the device if not stabilized
            logits = logits.softmax(-1).log()
            
            # select the next token with the highest probability of coming next
            # since this will return the token ID of the highest logit, not the embedding form
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # If we haven't started our sentence, make this the first word of our caption.
            # Otherwise, append the next word to what we have of our generated caption thus far.
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            
            # since next_token is currently a token ID, not embedding, we must return it into the format of an embedding
            # after turning it into an embedding, we can concat it onto prefix to use it to predict the next token using the 10 prefixes and the last predicted token
            # which is now turned into an embedding.
            next_embed = model.gpt.transformer.wte(next_token)
            input = torch.cat((input, next_embed), dim=1)
            
            # check if the next generated word is a period, indicating that the caption is done generating in terms of words
            # therefore, we can break the loop, and the caption is now finished
            if next_token.item() == stop_token_index:
                break
            
        # since our caption appears as token IDs, we have to decode them back into real-word equivalent
        output_text = tokenizer.decode(tokens[0].cpu().numpy())
    
    return output_text

def beam_search(model, tokenizer, embed=None, stop_token: str=".",
                temperature=1.0, max_length: int = 60, beam_k: int = 5):
    # puts model in evaluation mode so that no training is performed
    model.eval()
    # tokenizes the stop token ".", so if the next predicted token is ".", then it'll end the sentence
    stop_token_index = tokenizer.encode(stop_token)[0]
    
    with torch.no_grad():
        # initialize beams with our starting input embeddings and empty sequences
        # each beam tracks (score, token_sequence, input, is_stopped)
        beams = [
            (0.0, torch.empty(0, dtype=torch.long, device=embed.device), 
            embed, 
            False
        )]
        
        for _ in range(max_length):
            # this will store all possible candidate beams we generate this iteration
            candidates = []
            
            # process each beam in our current set of possibilities
            for score, token_seq, input_emb, stopped in beams:
                # skip processing if this beam already reached stop token
                if stopped:
                    candidates.append((score, token_seq, input_emb, True))
                    continue
                
                # processes the current sequence and generates logits for next token
                outputs = model.gpt(inputs_embeds=input_emb)
                # select the logits for the last position only
                logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                # softmax the logits for numerical stability
                log_probs = logits.softmax(-1).log()
                
                # get top-k most likely next tokens for this beam
                topk_log_probs, topk_tokens = torch.topk(log_probs, beam_k, dim=-1)
                
                # create new candidate beams by extending with each top token
                for i in range(beam_k):
                    next_token = topk_tokens[0, i].unsqueeze(0)
                    next_token_log_prob = topk_log_probs[0, i].item()
                    
                    # calculate new score (sum of log probabilities)
                    new_score = score + next_token_log_prob
                    
                    # append token to sequence
                    new_token_seq = torch.cat((token_seq, next_token), dim=0)
                    
                    # get embedding for next token to continue generation
                    next_embed = model.gpt.transformer.wte(next_token.unsqueeze(0))
                    new_input_emb = torch.cat((input_emb, next_embed), dim=1)
                    
                    # check if we hit stop token
                    is_stopped = next_token.item() == stop_token_index
                    
                    # add this candidate to our list
                    candidates.append((new_score, new_token_seq, new_input_emb, is_stopped))
            
            # sort all candidates by their score and select top beam_k
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_k]
            
            # check if all top beams have reached stop token
            # remember, beams are stored as [score, token_sequence, input_embeddings, is_stopped]
            # so if beam[3] is True for all beams, then we can stop the generation loop
            all_finished = all(beam[3] for beam in beams)
            if all_finished:
                break
        
        # select the highest scoring beam
        best_beam = beams[0]  # already sorted by score
        best_tokens = best_beam[1]
        
        # since our caption appears as token IDs, we decode them into text
        # using the same method as greedy_search for consistent formatting
        output_text = tokenizer.decode(best_tokens.cpu().numpy())
    
    return output_text

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}\nTrainable params: {trainable:,} ({trainable/total:.1%})")

def load_config(weights_location: str):
    config_path = os.path.join(os.path.dirname(weights_location), "training_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def display_image_with_caption(image_path, caption, beam_k):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    if beam_k == 1:
        plt.title("Method: Greedy Search", fontsize=12, horizontalalignment='center')
    else:
        plt.title(f'Method: Beam Search k={beam_k}', fontsize=12, horizontalalignment='center')
    plt.figtext(0.5, 0.05, caption, fontsize=12, horizontalalignment='center')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_location", type=str, default="F:/images")
    parser.add_argument("--weights_location", type=str, default="F:/checkpoints/transformer_mapper_weights.pt")
    parser.add_argument("--beam_k", type=int, default=6)
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    config = load_config(args.weights_location)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = GPT2CaptionModel(
        clip_embed_dim=config['clip_embed_dim'],
        prefix_length=config['prefix_length'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers']
    ).to(device)
    model.load_state_dict(torch.load(args.weights_location, map_location=device))
    model.eval()

    if os.path.isdir(args.image_location):
        image_files = [
            f for f in os.listdir(args.image_location) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            and not f.startswith('._')
        ]
        if not image_files:
            raise ValueError(f"No images found in {args.image_location}")
        selected_image = random.choice(image_files)
        image_path = os.path.join(args.image_location, selected_image)
    else:
        image_path = args.image_location
    
    image = Image.open(image_path)
    processed_image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clip_model.encode_image(processed_image).to(device, dtype=torch.float32)
        if config.get('normalize_clip_embed', False):  # Use config value
            prefix = prefix / prefix.norm(dim=-1, keepdim=True)
        clip_embed = model.prefix_projection(prefix).reshape(1, config['prefix_length'], -1)
        
    if args.beam_k == 1:
        caption = greedy_search(model, tokenizer, embed=clip_embed, 
                              max_length=args.max_length, temperature = args.temperature)
    else:
        caption = beam_search(model, tokenizer, embed=clip_embed, beam_k=args.beam_k, 
                            max_length=args.max_length, temperature = args.temperature)
    
    display_image_with_caption(image_path, caption, args.beam_k)
    
    #count_parameters(model)

if __name__ == "__main__":
    main()
    