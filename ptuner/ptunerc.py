import torch
import torch.nn as nn

class ptuner(nn.Module):
    def __init__(self) -> None:
        super(ptuner,self).__init__()
        self.b = nn.Parameter(torch.zeros((1, 77, 1)))  
        self.a = nn.Parameter(torch.randn((1, 1, 768)))

    
    def forward(self,x,beta=1.0):
        
        return x + beta * (self.b @ self.a)