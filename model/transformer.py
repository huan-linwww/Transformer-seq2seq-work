from model import encoder
from model import decoder
from torch import nn
import torch

class Transformer(nn.Module()):
    def __init__(self, vocab_size, dec_vocab_size, embed_size, max_len, hidden_size, feedforward_size, head_num, dropout, device, layer_num=6, src_pad_idx=0, trg_pad_idx=0, trg_sos_idx=1):
        super(Transformer, self).__init__()
        self.encoder = encoder.Encoder(vocab_size, embed_size, max_len, hidden_size, feedforward_size, head_num, dropout, device, layer_num)
        self.decoder = decoder.Decoder(dec_vocab_size, embed_size, max_len, hidden_size, feedforward_size, head_num, dropout, device, layer_num)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
    def forward(self, input, tgt):
        #src_mask
        src_mask = (input != self.src_pad_idx).unsqueeze(1).unsqueeze(2)#输出是一个布尔张量，形状为 [batch_size, src_len],之后维度扩展为 [batch_size, 1, 1, src_len]，多头注意力需要广播掩码到 [batch_size, n_head, seq_len, seq_len] 的维度
        
        #trg_mask
        trg_mask_pad_musk = (tgt != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.shape[1]
        trg_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()#创建一个下三角矩阵，用于确保解码器只能关注生成到当前词之前的词
        trg_mask = trg_mask_pad_musk & trg_mask
        out_enc = self.encoder(input, src_mask)
        out = self.decoder(tgt, src_mask, trg_mask, out_enc)
        
        return out
    