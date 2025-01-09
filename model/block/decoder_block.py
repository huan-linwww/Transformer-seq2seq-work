from torch import nn
from model.layer.layer_norm import LayerNorm
from model.layer.FFN import FFN
from model.layer.multi_head_attention import MultiHeadAttention
class Decoder_b(nn.Module):
    def __init__(self, hidden_size, feedforward_size, head_num, dropout=0.1):
        super(Decoder_b, self).__init__()
        self.attention_m = MultiHeadAttention(hidden_size, head_num)
        self.attention = MultiHeadAttention(hidden_size, head_num)
        self.ffn = FFN(hidden_size, feedforward_size,dropout)
        self.layer_norm1 = LayerNorm(hidden_size)
        self.layer_norm2 = LayerNorm(hidden_size)
        self.layer_norm3 = LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, input,srcmusk, mask, input_enc):
        out_musked = self.attention_m(input, input, input, mask)
        out_musked = self.dropout1(out_musked)
        out_musked = self.layer_norm1(input + out_musked)#残差连接
        
        out = self.attention(out_musked, input_enc, input_enc, srcmusk)
        out = self.dropout2(out)
        out = self.layer_norm2(out_musked + out)
        
        out_ffn = self.ffn(out)
        out_ffn = self.dropout2(out_ffn)
        out = self.layer_norm3(out_ffn + out)
        
        return out