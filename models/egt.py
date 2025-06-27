import torch
import torch.nn as nn
from .autoencoder import Autoencoder
from .attention import MultiHeadAttention

class EGT(nn.Module):
    def __init__(self, input_dim=4000, encoding_dim=1000, num_heads=8, num_layers=6, dropout=0.1):
        super(EGT, self).__init__()
        
        # Autoencoder for dimensionality reduction
        self.autoencoder = Autoencoder(input_dim, encoding_dim)
        
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, encoding_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            MultiHeadAttention(encoding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoding_dim, 1)
        )
        
        # Initialize weights
        nn.init.normal_(self.pos_embedding, std=0.02)
        
    def forward(self, x):
        # Autoencoder encoding
        x = self.autoencoder.encode(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Add position embedding
        x = x + self.pos_embedding
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # MLP head
        x = self.mlp_head(x)
        return x.squeeze(-1)
    
    def get_attention_maps(self, x):
        """Get attention maps for visualization"""
        attention_maps = []
        
        # Autoencoder encoding
        x = self.autoencoder.encode(x)
        x = x.unsqueeze(1)
        x = x + self.pos_embedding
        
        # Collect attention maps from each layer
        for layer in self.transformer_layers:
            x, attn = layer(x, return_attention=True)
            attention_maps.append(attn)
            
        return attention_maps 