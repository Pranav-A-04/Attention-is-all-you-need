import torch
from torch import nn
from torch.nn import functional as F
from model.encoder_decoder import EncoderBlock, DecoderBlock
from model.utils import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, dropout) for _ in range(num_encoder_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, dropout) for _ in range(num_decoder_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.output_activation = nn.Softmax(dim=-1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        
        # Embed and add positional encoding
        src = self.embedding(src)
        src = self.positional_encoding(src)
        
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        
        # Encoder
        enc_output = src
        for block in self.encoder_blocks:
            enc_output = block(enc_output, src_mask)

        # Decoder
        dec_output = tgt
        for block in self.decoder_blocks:
            dec_output = block(dec_output, enc_output, src_mask, tgt_mask)

        # Final output layer
        output = self.output_layer(dec_output)
        
        # # Get Output Probabilities
        # output = self.output_activation(output)
        # removed since using CrossEntropyLoss which applies LogSoftmax + NLLLoss internally
        
        return output