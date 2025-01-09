import torch
from torch import nn
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        #定义两个可学习参数
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        
    def forward(self, input):
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1,  keepdim=True)
        out = (input - mean)/(std + self.eps)
        return self.gamma*out + self.beta