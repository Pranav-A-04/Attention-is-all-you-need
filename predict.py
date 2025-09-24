import torch
import numpy as np
from model.transformer import Transformer
from preprocess.tokenize import *
from torch.utils.data import DataLoader
import argparse
from datasets import load_dataset
from torch.nn import functional as F
from model.utils import *
parser = argparse.ArgumentParser()

# Inference args
parser.add_argument('--source_sentence', type=str, required=True, help='Source sentence to translate')
parser.add_argument('--source_lang', type=str, choices=['de', 'en'], required=True, help='Source language (de or en)')
parser.add_argument('--target_lang', type=str, choices=['de', 'en'], required=True, help='Target language (de or en)')
parser.add_argument('--emb_dim', type=int, default=512, help='Embedding dimension')
parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for inference')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

# Set random seed for reproducibility
seed_all(args.seed)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
tokenizer.add_special_tokens({"additional_special_tokens": ["<2en>", "<2de>"]})

# Manually adding direction token
direction_token = f"<2{args.target_lang}>"
source_text = f"{direction_token} {args.source_sentence}"

# Load the trained model
model = Transformer(
    vocab_size=len(tokenizer),
    d_model=args.emb_dim,
    num_heads=args.num_heads,
    num_encoder_layers=args.num_encoder_layers,
    num_decoder_layers=args.num_decoder_layers,
    dropout=args.dropout
).to(args.device)

ckpt = torch.load(args.model_path, map_location=args.device)
model.load_state_dict(ckpt)
model.eval()


print(f"Source Text: {source_text}")
max_len = 64  # Maximum length of the target sequence

# Tokenize the input sentence
inputs = tokenizer(
    source_text,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=64
).to(args.device)

input_ids = inputs["input_ids"].to(args.device)
attention_mask = inputs["attention_mask"].to(args.device)
attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

decoder_start_token_id = tokenizer.pad_token_id
generated_ids = torch.tensor([[decoder_start_token_id]], device=args.device)

for _ in range(max_len):
    with torch.no_grad():
        outputs = model(input_ids, generated_ids, src_mask=attention_mask)
    
    next_token_logits = outputs[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
    generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
    
    if next_token_id.item() == tokenizer.eos_token_id:
        break

# Decode the generated tokens to get the translated sentence
translated_sentence = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Translated Sentence: {translated_sentence}\n")