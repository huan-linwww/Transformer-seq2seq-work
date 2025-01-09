from torch import nn
from model.layer.layer_norm import LayerNorm
from model.layer.FFN import FFN
from model.layer.multi_head_attention import MultiHeadAttention
class Encoder_b(nn.Module):
    def __init__(self, hidden_size, feedforward_size, head_num, dropout=0.1):
        super(Encoder_b, self).__init__()
        self.multiheadatt = MultiHeadAttention(hidden_size, head_num)
        self.ffn = FFN(hidden_size, feedforward_size,dropout)
        self.layer_norm1 = LayerNorm(hidden_size)
        self.layer_norm2 = LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, input, src_mask):
        out = self.multiheadatt(input, input, input, src_mask)# src_mask是为了屏蔽掉padding的位置
        out = self.dropout1(out)
        out = self.layer_norm1(input + out)#残差连接
        out_ffn = self.ffn(out)
        out_ffn = self.dropout2(out_ffn)
        out = self.layer_norm2(out_ffn + out)
        
        return out