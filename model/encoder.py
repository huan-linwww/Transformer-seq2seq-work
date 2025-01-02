from torch import nn
from model.embedding.embedding import Embeddings
from model.block.encoder_block import Encoder_b

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, hidden_size, feedforward_size, head_num, dropout, device, layer_num=6):
        super(Encoder, self).__init__()
        self.embedding = Embeddings(vocab_size, embed_size, max_len, device, dropout)
        self.Encoder = nn.ModuleList([Encoder_b(hidden_size, feedforward_size, head_num, dropout) for _ in range(layer_num)])
        
    def forward(self, input, src_mask):
        out = self.embedding(input)
        for encoder in self.Encoder:
            out = encoder(out, src_mask)
        return out