import torch

class Posencoder(torch.nn.Module()):
    def __init__(self, input_dim, max_len,device):
        """
        input_dim: int
        max_len: int 方便处理不同长度的句子
        device: torch.device
        """
        
        super(Posencoder, self).__init__()
        self.encoding = torch.zeros(max_len, input_dim, device=device)# [max_len, input_dim]
        self.encoding.requires_grad = False  # 这里只是计算编码，因此不需要梯度

        self.dim = input_dim
        self.device = device
        pos = torch.arange(0, max_len, device=self.device).unsqueeze(1)#在指定维度增加一个大小为1的维度[max_len,1]
        _2i = torch.arange(0, input_dim, 2, device=self.device).float()
        self.encoder[:,1::2] = torch.sin(pos/(10000**(_2i/input_dim))) # 这里10000是标量，会广播到与_2i/input_dim相同维度
        self.encoder[:,0::2] = torch.cos(pos/(10000**(_2i/input_dim)))

    def forward(self, input):
        return self.encoder[:input.size(1), :]# [seq_len, input_dim]，注意nn自动处理batch_first，所以此处不考虑batch维度
        