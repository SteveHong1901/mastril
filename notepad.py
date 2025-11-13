import torch.nn as nn
import torch 

class LayerNorm(nn.Module):
    def __init__(self, eps: int = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        mean = x.mean(dims=-1, keepdim=True)
        std = x.mean(dims=-1, keepdim=True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias
    

        
class FFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),  # Use the dropout parameter
            nn.Linear(d_hidden, d_model)
        )
        
    def forward(self, x: torch.Tensor):
        return self.net(x)