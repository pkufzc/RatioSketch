import torch.nn as nn


class WeightDecodeNetResidual(nn.Module):
    """
    权重解码网络 - 残差连接版本
    功能：通过残差连接增强梯度流动，提升深层网络性能
    对应论文中的改进版g_dec，通过残差结构优化频率估计精度
    """

    def __init__(self, input_dim, weight_decode_hidden_layer_size, output_dim=1):
        super().__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.hidden_layer_size = weight_decode_hidden_layer_size

        # 定义网络层（含残差连接）
        self.hidden_1 = nn.Linear(self.in_dim, self.hidden_layer_size)
        self.activate_1 = nn.ReLU()  # 使用ReLU确保非负中间表示

        self.hidden_2 = nn.Linear(self.hidden_layer_size, self.in_dim)
        self.activate_2 = nn.ReLU()  # 残差连接后使用ReLU

        self.hidden_3 = nn.Linear(self.in_dim, (self.in_dim + self.out_dim) // 2)
        self.activate_3 = nn.ReLU()

        self.out = nn.Linear((self.in_dim + self.out_dim) // 2, self.out_dim)

    def forward(self, x):
        """前向传播：含残差连接的频率解码"""
        y = self.hidden_1(x)
        y = self.activate_1(y)
        y = self.activate_2(self.hidden_2(y) + x)
        y = self.activate_3(self.hidden_3(y))
        y = self.out(y)
        return y
