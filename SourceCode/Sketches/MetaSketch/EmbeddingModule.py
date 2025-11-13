import torch.nn as nn


class EmbeddingNet(nn.Module):
    """
    嵌入网络，将输入特征映射到低维嵌入空间
    对应论文中的特征嵌入模块，生成用于寻址的初始表示
    """

    def __init__(self, input_dim, output_dim, hidden_layer_size):
        super().__init__()
        self.in_dim = input_dim  # 输入维度，对应论文中的特征维度
        self.out_dim = output_dim  # 输出维度，对应论文中的嵌入维度l_z
        self.hidden_layer_size = hidden_layer_size  # 隐藏层大小

        # 定义三层前馈网络结构
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_layer_size),
            nn.BatchNorm1d(self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(
                self.hidden_layer_size, (self.hidden_layer_size + self.out_dim) // 2
            ),
            nn.BatchNorm1d((self.hidden_layer_size + self.out_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.hidden_layer_size + self.out_dim) // 2, self.out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.float()
        return self.net(x)
