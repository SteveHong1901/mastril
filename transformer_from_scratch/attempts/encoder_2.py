import torch
import torch.nn as nn
import math

# TokenEmbeddings
# Mapping from the tokens to the embedding space. Same tokkens gets the same embeddings.

class TokenEmbeddings(nn.Modules):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x L x 1 -> B x L x d
        return self.embedding(x) + math.sqrt(self.d_model)

# Positional Encoding

# LayerNorm
class LayerNorm(nn.Modules):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x L x d -> B x L x d
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        return (self.alpha * (x-mean) / torch.sqrt(var + self.eps)) + self.bias

# FFN
class FeedForwardLayer(nn.Modules):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x L x d
        return self.net(x)

# Residual Connection
class ResidualConnection(nn.Modules):
    '''A residual connection adds the original input back to the transformed output (x + sublayer(x))'''
    def __init__(self):
        super().__init__()
        self.norm = LayerNorm(eps=1e-5)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        # x: B x L x d -> B x L x d
        return x + sublayer(self.norm(x))

# MultiHeadAttention

# define all the Q, K, V weights
# head and dim per head
# scaled dot product attention ops
# splitting and concating heads

class MultiHeadAttention(nn.Modules):
    def __init__(self, d_model: int, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h, 'd_model is not divisible by h'

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        # q: B x h x L x d, k: B x h x d x L -> B x h x L x L
        QK_T = (q @ k.tranpose(-2, -1))

        if mask is not None:
            QK_T.mask_filled(mask==0, 1e-9)
        
        attention_scores = QK_T.softmax(dim=-1)

        return attention_scores @ v # B x h x L x d

    def head_split(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x L x d -> B x L x h x d_k -> B x h x L x d_k
        batch_size, seq_len, d_model = x.shape
        return x.view(batch_size, seq_len, self.h, self.d_k).tranpose(2, 1)

    def head_combine(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x h x L x d_k -> B x L x h x d_k -> B x L x d
        batch_size, seq_len, d_model = x.shape
        return x.transpose(2, 1).view(batch_size, seq_len, self.h * self.d_k)

    def forward(self, x: torch.Tensor):

        # x: B, L, h, d -> B, L, h, d

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        Q_h = self.head_split(Q)
        K_h = self.head_split(K)
        V_h = self.head_split(V)

        attention = self.scaled_dot_product_attention(Q_h, K_h, V_h)

        attention_cb = self.head_combine(attention)

        return self.w_o(attention_cb)

# Encoder Block

class EncoderLayer(nn.Module):
    def __init__(self, attention_block: MultiHeadAttention, feedfoward_block: FeedForwardLayer) -> None:
        super().__init__()
        self.attention_block = attention_block
        self.feedfoward = feedfoward_block
        self.residual_block = nn.ModuleList([ResidualConnection() for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x L x d
        x = self.residual_block[0](x, lambda: self.attention_block(x))
        x = self.residual_block[1](x, self.feedfoward(x))
        return x


# TransformerEncoder
class Encoder(nn.Module):
    def __init__(self, num_layers: int, h: int, d_model: int, d_ff: int) -> None:
        super().__int__()
        self.layers = nn.ModuleList(
            [EncoderLayer(
                MultiHeadAttention(d_model, h),
                FeedForwardLayer(d_model, d_ff)
            )] for _ in range(num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x L x d
        for layer in self.layers:
            x = layer(x)
        return x