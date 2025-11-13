import torch
import torch.nn as nn
from .SparseSoftmax import Sparsemax


class AttentionMatrix(nn.Module):
    def __init__(self, refined_dim, slot_dim, depth_dim=1):
        super().__init__()
        self.refined_dim = refined_dim  # 精炼向量维度（输入维度）
        self.slot_dim = slot_dim  # 槽位维度（输出维度，即每个哈希表的槽位数）
        self.depth_dim = depth_dim  # 深度维度（模拟多哈希函数的数量）
        self.attention_matrix = torch.nn.Parameter(
            torch.rand(depth_dim, refined_dim, slot_dim, requires_grad=True)
        )
        self.normalize()  # 初始化后立即归一化矩阵参数
        self.sparse_softmax = Sparsemax(slot_dim)  # 稀疏Softmax层（实现稀疏寻址）

    def forward(self, refined_vec):
        product_tensor = refined_vec.matmul(self.attention_matrix)
        return self.sparse_softmax(product_tensor)

    def normalize(self):
        with torch.no_grad():  # 归一化操作不计算梯度
            matrix_pow_2 = torch.square(self.attention_matrix)
            matrix_base = torch.sqrt(matrix_pow_2.sum(dim=1, keepdim=True))
            self.attention_matrix.data = self.attention_matrix.div(matrix_base)
