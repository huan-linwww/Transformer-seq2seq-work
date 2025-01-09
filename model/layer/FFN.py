from torch import nn

class FFN(nn.Module):
    def __init__(self, hidden_size, feedforward_size,droupout=0.1):
        """
        hidden_size: int
        feedforward_size: int
        """
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(hidden_size, feedforward_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(droupout)
        self.linear2 = nn.Linear(feedforward_size, hidden_size)
        
        
    def forward(self, input):
        out = self.linear1(input)
        out = self.relu(out)
        out = self.dropout(out)
        return self.linear2(out)