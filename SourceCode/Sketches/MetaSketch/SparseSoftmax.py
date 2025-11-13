import numpy as np
import torch
import torch.nn as nn


class Sparsemax(nn.Module):
    """
    Sparsemax激活函数类，实现论文中的稀疏Softmax操作
    数学原理：通过欧式投影将输入向量映射到概率单纯形（probability simplex），并强制输出稀疏性
    参考：论文II-B节"we utilize Sparse SoftMax [36], [37] instead of SoftMax"
    """
    def __init__(self, input_size, dim=None):
        """
        初始化Sparsemax层
        :param input_size: 输入向量的维度大小（即槽位数量d2，对应论文中的slot_dim）
        :param dim: 应用Sparsemax的维度，默认最后一维
        """
        super(Sparsemax, self).__init__()
        # 创建固定参数：[1, 2, ..., input_size]，用于计算稀疏性边界
        self.range = nn.Parameter(
            torch.arange(start=1, end=input_size + 1, step=1, dtype=torch.float).view(1, -1),
            requires_grad=False  # 不需要梯度更新
        )
        self.dim = -1 if dim is None else dim  # 默认在最后一维应用稀疏Softmax

    def forward(self, input):
        """
        前向传播函数：计算稀疏Softmax
        :param input: 输入张量，形状需包含批次维度，例如 [batch_size, d1, d2]（对应论文中的a_i）
        :return: 稀疏化后的地址向量，形状与输入一致，非零元素仅占少数
        """
        # ---------------------- 维度调整 ----------------------
        # 将目标维度移至第1维（方便后续计算）
        input = input.transpose(0, self.dim)
        original_size = input.size()  # 保存原始形状
        # 展平为 [d2, batch_size * d1] 以便处理
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)  # 形状变为 [batch_size * d1, d2]
        dim = 1  # 操作维度固定为列（每个样本的d2维度）

        # ---------------------- 数值稳定性处理 ----------------------
        # 减去每行最大值，避免指数运算溢出（同Softmax的稳定性技巧）
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # ---------------------- 排序与稀疏性判断 ----------------------
        # 对输入按降序排序，得到zs: [batch_size*d1, d2]
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        # 扩展range张量以匹配zs的形状，用于计算边界
        range = self.range.expand_as(zs)  # shape: [1, d2] → [batch_size*d1, d2]

        # 计算边界条件：1 + k*zs_k > sum_{i=1}^k zs_i
        bound = 1 + range * zs  # 公式中的1 + k*z_{(k)}
        cumulative_sum_zs = torch.cumsum(zs, dim=dim)  # 前k项累加和
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())  # 指示函数：1 + k*z_{(k)} > sum_{i=1}^k z_i ?

        # 找到最大的k值，使得条件成立（即稀疏度）
        k = torch.max(is_gt * range, dim=dim, keepdim=True)[0]  # 每行最大的k值

        # ---------------------- 计算阈值tau ----------------------
        # 提取前k个非零元素（利用布尔索引）
        zs_sparse = is_gt * zs  # 仅保留满足条件的元素，其余为0
        # 计算阈值tau = (sum(zs_sparse) - 1) / k
        taus = (torch.sum(zs_sparse, dim=dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)  # 扩展至与输入同形

        # ---------------------- 生成稀疏输出 ----------------------
        # Sparsemax公式：max(0, z_i - tau)
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # ---------------------- 恢复原始维度 ----------------------
        # 逆序调整维度，恢复输入的原始形状
        output = self.output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """
        反向传播函数：计算梯度
        利用稀疏性掩码（nonzeros）仅对非零元素传播梯度
        """
        dim = 1
        # 找到输出中的非零元素（稀疏掩码）
        nonzeros = torch.ne(self.output, 0)  # 非零元素为True，其余为False
        # 计算每行的平均梯度（仅对非零元素）
        sum_grad = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim, keepdim=True)
        # 梯度仅在非零元素上有效，其余为0（对应论文中的链式法则推导）
        self.grad_input = nonzeros * (grad_output - sum_grad.expand_as(grad_output))

        return self.grad_input