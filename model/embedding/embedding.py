from torch import nn
from model.embedding.posembedding import Posencoder
from model.embedding.tokenembedding import TokenEmbedding

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, device, dropout=0.1):
        """
        vocab_size: int 词表大小
        embed_size: int embedding维度
        max_len: int 最大长度
        device: torch.device
        """
        super(Embeddings, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_size)
        self.pos_embedding = Posencoder(embed_size, max_len, device)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input):
        return self.dropout(self.token_embedding(input) + self.pos_embedding(input))