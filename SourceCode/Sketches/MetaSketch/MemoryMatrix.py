from torch import nn
import torch


class BasicMemoryMatrix(nn.Module):
    def __init__(self, depth_dim, slot_dim, embedding_dim):
        super().__init__()
        self.slot_dim = slot_dim  # 槽位维度（d_2，对应存储矩阵的列数）
        self.depth_dim = depth_dim  # 深度维度（d_1，对应多哈希表数量）
        self.embedding_dim = embedding_dim  # 嵌入向量维度（l_z）
        self.memory_matrix = torch.zeros(
            self.depth_dim,
            self.slot_dim,
            self.embedding_dim,
            requires_grad=True,
        )

    def to(self, device):
        super().to(device)
        self.memory_matrix = self.memory_matrix.to(device)
        return self

    def write(self, address, embedding, frequency):
        frequency_embedding = embedding * frequency.view(-1, 1)
        write_matrix = address.transpose(1, 2).matmul(frequency_embedding)
        self.memory_matrix = self.memory_matrix + write_matrix

    def read(self, address, embedding):
        read_info_tuple = self.basic_read_attention_sum(address, embedding)
        return torch.cat(read_info_tuple, dim=-1)

    def basic_read_attention_sum(self, address, embedding, read_compensate=True):
        batch_size = address.shape[1]
        # 地址与存储矩阵相乘，获取基础读取矩阵（形状：[batch_size, d1, l_z]）
        basic_read_matrix = address.matmul(self.memory_matrix)

        if read_compensate:
            # 地址稀疏性补偿：除以地址的L2范数平方，缓解稀疏地址的权重偏差
            norm = address.square().sum(
                dim=-1, keepdim=True
            )  # 形状：[batch_size, d1, 1]
            basic_read_matrix = basic_read_matrix / norm.clamp(min=1e-8)  # 防止除零

        # 处理嵌入向量中的零值（避免除法溢出）
        cm_embedding = torch.where(
            embedding > 0.00001, embedding, torch.zeros_like(embedding) + 0.00001
        )
        zero_add_vec = torch.where(
            abs(embedding) < 0.0001,
            torch.zeros_like(embedding) + 10000,
            torch.zeros_like(embedding),
        )

        # 应用两种Count-Min风格的读取头（CM Read Head）
        cm_read_info_1 = self.cm_read_1(basic_read_matrix, cm_embedding, zero_add_vec)
        cm_read_info_2 = self.cm_read_2(basic_read_matrix, cm_embedding, zero_add_vec)

        # 调整基础读取矩阵维度，便于后续拼接
        basic_read_info = basic_read_matrix.transpose(0, 1).reshape(batch_size, -1)

        return basic_read_info, cm_read_info_1, cm_read_info_2

    def cm_read_1(self, basic_read_matrix, cm_embedding, zero_add_vec):
        cm_basic_read_matrix = basic_read_matrix + zero_add_vec
        # 归一化：除以嵌入向量（论文中的gating机制，抑制无关特征）
        cm_read = cm_basic_read_matrix.div(cm_embedding)  # 形状：[batch_size, d1, l_z]
        # 沿嵌入维度取最小值，模拟Count-Min的多哈希最小值策略
        min_cm_read, _ = cm_read.min(dim=-1)  # 形状：[batch_size, d1]
        return min_cm_read.transpose(0, 1)  # 形状：[d1, batch_size] → [batch_size, d1]

    def cm_read_2(self, basic_read_matrix, cm_embedding, zero_add_vec):
        """
        第二种Count-Min读取头：先减去最小值再归一化
        进一步抑制噪声，提升低频率项的鲁棒性
        """
        # 沿嵌入维度取最小值，作为噪声基准
        min_info, _ = basic_read_matrix.min(
            dim=-1, keepdim=True
        )  # 形状：[batch_size, d1, 1]
        # 减去最小值，突出相对差异
        basic_read_minus_min = basic_read_matrix - min_info
        # 处理极小值，避免除以零
        basic_read_minus_min = torch.where(
            abs(basic_read_minus_min) < 0.0001,
            torch.zeros_like(basic_read_minus_min) + 100000,
            basic_read_minus_min,
        )
        # 归一化并取最小值
        cm_read = (basic_read_minus_min + zero_add_vec).div(cm_embedding)
        min_cm_read, _ = torch.min(cm_read, dim=-1)  # 形状：[batch_size, d1]

        # 调整维度并拼接最小值信息
        min_info = min_info.squeeze().transpose(0, 1)  # 形状：[batch_size, d1]
        min_cm_read = min_cm_read.transpose(0, 1)  # 形状：[batch_size, d1]
        return torch.cat((min_info, min_cm_read), dim=-1)  # 形状：[batch_size, 2*d1]
