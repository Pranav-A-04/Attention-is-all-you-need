import torch
import torch.nn as nn
import os
import numpy as np
from transformers import AutoTokenizer
from functools import partial
from model.transformer import Transformer
from model.utils import *
from preprocess.tokenize import *
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

#model args
parser.add_argument('--emb_dim', type=int, default=512, help='Embedding dimension')
parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

#training args
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
parser.add_argument('--save_interval', type=int, default=1, help='Interval (in epochs) to save model checkpoints')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

# Set random seed for reproducibility
seed_all(args.seed)

# Dataset preparation & loading

# load IWSLT14 German ↔ English
dataset = load_dataset("iwslt2017", "iwslt2017-en-de")

# taking a pretrained tokenizer for simplicity - one which has already been trained for German <-> English translation
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

# add special tokens for language direction
special_tokens = {"additional_special_tokens": ["<2en>", "<2de>"]}
tokenizer.add_special_tokens(special_tokens)

tokenize_fn = partial(tokenize_func_bidirectional, tokenizer=tokenizer)

# bidirectional tokenization
tokenized_dataset = dataset.map(
    tokenize_fn, 
    batched=True
)

tokenized_dataset.set_format(
                             type="torch",
                             columns=["input_ids", "attention_mask", "labels"]
                            )

train_dataloader = DataLoader(
                    tokenized_dataset["train"], 
                    batch_size=args.batch_size, 
                    shuffle=True
                )
val_dataloader = DataLoader(
                    tokenized_dataset["validation"], 
                    batch_size=args.batch_size, 
                    shuffle=False
                )

def train(batch, model, optimizer, loss_criterion, device, epoch):
    model.train()
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    # Shift labels for teacher forcing
    decoder_input_ids = torch.zeros_like(labels)
    decoder_input_ids[:, 1:] = labels[:, :-1]
    decoder_input_ids[:, 0] = tokenizer.pad_token_id  # Start token can be pad token for simplicity

    outputs = model(input_ids, decoder_input_ids, src_mask=attention_mask, tgt_mask=None)
    
    # Compute loss only on non-padded tokens
    loss = loss_criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
    
    # zero out the optimizer grads
    optimizer.zero_grad()
    
    # backpropagate the loss
    loss.backward()
    
    # update the model parameters
    optimizer.step()

    return loss.item()

def validate(batch, model, loss_criterion, device, epoch):
    model.eval()
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    # Shift labels for teacher forcing
    decoder_input_ids = torch.zeros_like(labels)
    decoder_input_ids[:, 1:] = labels[:, :-1]
    decoder_input_ids[:, 0] = tokenizer.pad_token_id  # Start token can be pad token for simplicity

    with torch.no_grad():
        outputs = model(input_ids, decoder_input_ids, src_mask=attention_mask, tgt_mask=None)
        
        # Compute loss only on non-padded tokens
        loss = loss_criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
    
    return loss.item()

def main():
    # initialize model, optimizer, loss criterion
    vocab_size = len(tokenizer)
    model = Transformer(
        vocab_size=vocab_size,
        d_model=args.emb_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout
    ).to(args.device)
    
    model.resize_token_embeddings(len(tokenizer))


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    print("Start training...")

    # init loss trackers
    avg_train_loss = float('inf')
    avg_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        train_losses = []
        for batch in train_dataloader:
            loss = train(batch, model, optimizer, loss_criterion, args.device, epoch)
            train_losses.append(loss)
            
        avg_train_loss = np.mean(train_losses)
        
        val_losses = []
        if (epoch + 1) % args.save_interval == 0:
            print("Validating...")
            for batch in val_dataloader:
                loss = validate(batch, model, loss_criterion, args.device, epoch)
                val_losses.append(loss)
            
            avg_val_loss = np.mean(val_losses)
        
        print(f"Epoch: {epoch+1}/{args.num_epochs} | Avg Training Loss: {avg_train_loss} | Avg Validation Loss: {avg_val_loss}")
        
        # Save model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            checkpoint_path = os.path.join(args.save_dir, f"transformer_epoch{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")


if __name__ == "__main__":
    main()
