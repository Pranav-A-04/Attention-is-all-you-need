import torch
from torch import nn
from torch.nn import functional as F
from model.utils import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(num_heads, emb_dim, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.self_attention(x, x, x, mask)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_output))
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(num_heads, emb_dim, dropout)
        self.cross_attention = MultiHeadAttention(num_heads, emb_dim, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.layer_norm3 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Self-attention
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.layer_norm1(x + attn_output)

        # Cross-attention
        cross_attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.layer_norm2(x + cross_attn_output)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.layer_norm3(x + self.dropout(ffn_output))

        return x