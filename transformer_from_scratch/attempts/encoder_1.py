from turtle import forward
import torch
import torch.nn as nn
import math


# token embedding
class TokenEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # think about rows and columns

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x L
        return self.embedding(x) * math.sqrt(self.d_model) #x: B x L x d

# ablosute positional encoding


# layer norm
class LayerNormalization(nn.Module):
    def __init__(self, eps: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x L x d
        mean = x.mean(dim=-1, keepdim=True) # B x L x 1 instead of B x L
        var = x.var(dim=-1, keepdim=True) # B x L x 1 instead of B x L
        return (self.alpha * (x-mean)) / torch.sqrt(var + self.eps) + self.bias # # B x L x d 

# FFN
class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x L x d
        return self.net(x) # B x L x d

# MHA

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int):
        super().__init__()
        self.h = h
        self.d_model = d_model
        assert d_model % h == 0

        self.d_k: int = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attention_score = (q @ k.tranpose(-2, -1)).softmax(dim=-1) / math.sqrt(q.shape[-1])
        return attention_score @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x L x d
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(2, 1) # B x h x L x d
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(2, 1)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(2, 1)

        x = MultiHeadAttention(query, key, value)

        # B x h x L x d_k -> B x L x h x d_k -> B x L x d
        x = x.transpose(2, 1).view(x.shape[0], x.shape[1], self.h, self.d_k)

        return self.w_o(x)

        
    
# Reisudal connection
class ResidualConnection(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        return x + sublayer(self.norm(x))
        

# Encoder block
class Encoder(nn.Module):
    def __init__(self, attention_block: MultiHeadAttention, feedforward_network: FeedForwardLayer):
        self.attention_block = attention_block
        self.feedforward_network = feedforward_network
        self.residual_stack = nn.ModuleList([ResidualConnection() for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x L x d
        x = self.residual_stack[0](x, lambda: self.attention_block(x))
        x = self.residual_stack[1](x, self.feedforward_network(x))
        return x
        

# Encoder
class EncoderStack(nn.Module):
    def __init__(self, num_layers: int, d_model: int, h: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.ModuleList([Encoder(
            MultiHeadAttention(d_model, h),
            FeedForwardLayer(d_model, d_ff)
        ) 
        for _ in range(num_layers)
    ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x