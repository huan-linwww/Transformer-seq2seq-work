from torch import nn
from model.embedding.embedding import Embeddings
from model.block.decoder_block import Decoder_b

class Decoder(nn.Module):
    def __init__(self, dec_vocab_size,  embed_size, max_len, hidden_size, feedforward_size, head_num, dropout, device, layer_num=6):
        super(Decoder, self).__init__()
        self.embedding = Embeddings(dec_vocab_size, embed_size, max_len, device, dropout)#这里的dec_vocab_size是目标语言的词表大小
        self.Decoder = nn.ModuleList([Decoder_b(hidden_size, feedforward_size, head_num, dropout) for _ in range(layer_num)])
        
        self.linear = nn.Linear(hidden_size, dec_vocab_size)

        
    def forward(self, input, src_mask, mask, input_enc):
        out = self.embedding(input)
        
        
        for decoder in self.Decoder:
            out = decoder(out, src_mask, mask, input_enc)
            
        out = self.linear(out)

        return out