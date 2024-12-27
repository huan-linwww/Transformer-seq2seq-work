from torch import nn
# 为了实现词的embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__(vocab_size, embed_size, padding_idx=1)#padding_idx=1 表示词表中索引为0的位置用于padding，此外，注意是索引，不是词，根据索引进行embedding