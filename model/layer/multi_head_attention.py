import torch
from torch import nn
import math


class MultiHeadAttention(nn.Module()):
    def __init__(self, hidden_size, head_num):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.head_num = head_num

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)
        self.output_linear = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, q, k, v, mask=None):
        """
        q: [batch_size, seq_len, hidden_size]
        k: [batch_size, seq_len, hidden_size]
        v: [batch_size, seq_len, hidden_size]
        mask: [batch_size, seq_len, seq_len]，这里是为了实现mask机制，避免未来信息泄露
        """
        # q, k, v: [batch_size, seq_len, hidden_size]
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        #分解多头
        q, k, v = [self.split_heads(tensor) for tensor in [q, k, v]]
        
        #计算注意力
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hidden_size//self.head_num)#k转置后是[batch_size, head_num, hidden_size//head_num, seq_len]，这样计算后是[batch_size, head_num, seq_len, seq_len]
        
        if mask is not None:
           score = score.masked_fill(mask == 0, -10000)#将 mask 中为 0 的位置对应的 score 值替换为一个非常小的值（如 -10000），以便在后续的 softmax 操作中，这些值几乎为零（数值上接近于无效）。
        
        score = self.softmax(score)
        output = torch.matmul(score, v)
        output = output.transpose(1, 2)# [batch_size, head_num, seq_len, hidden_size//head_num] -> [batch_size, seq_len, head_num, hidden_size//head_num]
        batch_size, seq_len, _, _ = output.size()
        output = output.view(batch_size, seq_len, self.hidden_size)#-> [batch_size, seq_len, hidden_size]
        
        output = self.output_linear(output)#最后再过一个线性层
        
        return output
        
        
    def split_heads(self, tensor):
        """
        tensor: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = tensor.size()
        tensor = tensor.view(batch_size, seq_len, self.head_num, hidden_size//self.head_num)
        return tensor.permute(0, 2, 1, 3)# permute允许同时重排多个维度，这里是将第2和第3维度交换，[batch_size, head_num, seq_len, hidden_size//head_num]方便后续计算
    
        