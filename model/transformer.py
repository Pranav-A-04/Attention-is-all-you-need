import torch
from torch import nn
from torch.nn import functional as F
from model.encoder_decoder import EncoderBlock, DecoderBlock

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(input_dim, output_dim, num_heads, dropout) for _ in range(num_encoder_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(input_dim, output_dim, num_heads, dropout) for _ in range(num_decoder_layers)
        ])
        self.output_layer = nn.Linear(input_dim, output_dim)
        self.output_activation = nn.Softmax(dim=-1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder
        enc_output = src
        for block in self.encoder_blocks:
            enc_output = block(enc_output, src_mask)

        # Decoder
        dec_output = tgt
        for block in self.decoder_blocks:
            dec_output = block(dec_output, enc_output, tgt_mask)

        # Final output layer
        output = self.output_layer(dec_output)
        
        # Get Output Probabilities
        output = self.output_activation(output)
        
        return output