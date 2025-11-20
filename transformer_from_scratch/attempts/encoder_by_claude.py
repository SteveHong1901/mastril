import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Absolute positional encoding using sine and cosine functions.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_length, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        Returns:
            Tensor of shape (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Args:
            Q: (batch_size, num_heads, seq_length, d_k)
            K: (batch_size, num_heads, seq_length, d_k)
            V: (batch_size, num_heads, seq_length, d_k)
            mask: (batch_size, 1, 1, seq_length) or None
        Returns:
            output: (batch_size, num_heads, seq_length, d_k)
            attention_weights: (batch_size, num_heads, seq_length, seq_length)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (for padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute weighted sum of values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        Args:
            x: (batch_size, seq_length, d_model)
        Returns:
            (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """
        Combine the heads back together.
        Args:
            x: (batch_size, num_heads, seq_length, d_k)
        Returns:
            (batch_size, seq_length, d_model)
        """
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_length, d_model)
            key: (batch_size, seq_length, d_model)
            value: (batch_size, seq_length, d_model)
            mask: (batch_size, 1, 1, seq_length) or None
        Returns:
            output: (batch_size, seq_length, d_model)
            attention_weights: (batch_size, num_heads, seq_length, seq_length)
        """
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Apply attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads
        output = self.combine_heads(attn_output)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, d_model)
        Returns:
            (batch_size, seq_length, d_model)
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    Consists of:
    1. Multi-head self-attention
    2. Add & Norm
    3. Feed-forward network
    4. Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_length, d_model)
            mask: (batch_size, 1, 1, seq_length) or None
        Returns:
            output: (batch_size, seq_length, d_model)
            attention_weights: (batch_size, num_heads, seq_length, seq_length)
        """
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, attention_weights


class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder.
    Consists of:
    1. Input embedding
    2. Positional encoding
    3. N encoder layers
    """
    def __init__(
        self, 
        vocab_size, 
        d_model=512, 
        num_layers=6, 
        num_heads=8, 
        d_ff=2048, 
        max_seq_length=5000,
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Input embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Stack of encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input token indices of shape (batch_size, seq_length)
            mask: Padding mask of shape (batch_size, 1, 1, seq_length) or None
        Returns:
            output: (batch_size, seq_length, d_model)
            attention_weights: List of attention weights from each layer
        """
        # Embedding with scaling
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through each encoder layer
        all_attention_weights = []
        for encoder_layer in self.encoder_layers:
            x, attention_weights = encoder_layer(x, mask)
            all_attention_weights.append(attention_weights)
        
        return x, all_attention_weights
    
    def create_padding_mask(self, seq, pad_token=0):
        """
        Create padding mask for the input sequence.
        Args:
            seq: (batch_size, seq_length)
            pad_token: The token index used for padding
        Returns:
            mask: (batch_size, 1, 1, seq_length)
        """
        mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
        return mask


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 10000
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1
    batch_size = 32
    seq_length = 50
    
    # Create encoder
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout
    )
    
    # Create sample input (random token indices)
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Create padding mask (assume last 10 tokens are padding)
    mask = encoder.create_padding_mask(x)
    
    # Forward pass
    output, attention_weights = encoder(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of attention weight tensors: {len(attention_weights)}")
    print(f"Each attention weight shape: {attention_weights[0].shape}")
    
    # Verify output
    print(f"\nEncoder has {sum(p.numel() for p in encoder.parameters()):,} parameters")

