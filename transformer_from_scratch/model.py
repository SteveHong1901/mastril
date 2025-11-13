from curses import KEY_SUSPEND
import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    ''' Embedding of size (vocab_size, d_model) '''
    def __init__(self, d_model: int, vocab_size: int): 
        super().__init__() # constuctor
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # Map each numbers to 1x512 vectors. Vectors learnt by the model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    ''' Positional Encoding: Vector of size (1 x d_model) to add on each of the input embedding. One for each position. '''
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1), can also use .reshape(-1, 1) or .view(-1, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))

        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        # register_buffer: Registers 'pe' as a buffer (non-trainable parameter)
        # - Automatically moved to GPU/CPU when module.to(device) is called
        # - Saved/loaded with model state_dict
        # - Not updated during backprop (positional encodings are fixed)
        # Equivalent to self.pe = pe, but properly integrated with PyTorch's module system
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encodings to input embeddings
        # self.pe shape: (1, seq_len, d_model)
        # x shape: (batch_size, seq_len, d_model)
        # We slice self.pe to match x's sequence length (in case x is shorter than max seq_len)
        # .requires_grad_(False) ensures gradients don't flow through positional encodings
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        
        # Apply dropout for regularization during training
        # This helps prevent overfitting by randomly zeroing some values
        return self.dropout(x)

class LayerNormalization(nn.Module):
    ''' 
    output = alpha * (x - mean) / sqrt(variance + eps) + bias 
    nn.Parameter is a special wrapper that tells PyTorch: "This tensor should be learned during training."
    '''
    def __init__(self, eps: float = 10e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Paramter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # Taking mean across the last dim, and avoid collapsing the dim
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias

# class FeedForwardBlock(nn.Module):

#     def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
#         super().__init__()
#         self.linear_1 = nn.Linear(d_model, d_ff) # Has bias by default
#         self.dropout = nn.Dropout(dropout)
#         self.linear_2 = nn.Linear(d_ff, d_model)

#     def forward(self, x):
#         # (batch, Seq_len, d_model) -> (Batch, Seq_len, d_ff) -> (Batch, Seq_len, d_model)
#         return self.linear_2(self.dropout(torch.linear_1(x)))


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),  # or nn.GELU() for GPT-style transformers
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.net(x)
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod #static method means we can call the method without an instance of the class
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        # (B, h, l, d_k) @ (B, h, d_k, L) -> (B, h, L, L)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # tranposing the second to last with the last (seq_len and d_k)
        if mask is not None:
            attention_score.masked_filled(mask == 0, 1e-9)
        attention_scores = attention_scores.softmax(dim = -1) # (B, h, L, L) For each query position, softmax normalizes across all key positions, each query has a probability distribution over all keys
        
        return (attention_scores @ value), attention_scores
        
        
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model, d_model) --> (batch, seq_len, d_model, d_model)
        key  = self.w_k(k)
        value = self.w_v(v)
        
        # Split the embedding dim not the token length
        # (batch, seq_len, d_model, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # Splitting d_model up into h strips of d_k each. then we swap the second dimension with the first
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)    
        
        # (B, h, L, d_k) -> (B, L, h, d_k) -> (B, L, d)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # why -1 here and x.shape[1] above?
        
        # (B, L, d) @ (d, d) -> (B, L, d)
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))
    

class Encoder(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_conections = nn.ModuleList([ResidualConnection(dropout) for i in range(2)]) # nn.ModuleList is a PyTorch container that holds a list of nn.Module objects. It's like a regular Python list, but PyTorch-aware.
    
        # soruce mask is used to prevent the padding words from interacting with other words
    def forward(self, x: torch.Tensor, src_mask):
        x = self.residual_conections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) #SA -> norm -> residual 
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x