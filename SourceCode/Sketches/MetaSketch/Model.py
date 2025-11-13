# 导入PyTorch库
import os
import torch
import torch.nn as nn
from SourceCode.Sketches.MetaSketch.EmbeddingModule import EmbeddingNet
from SourceCode.Sketches.MetaSketch.MemoryMatrix import BasicMemoryMatrix
from SourceCode.Sketches.MetaSketch.DecodeModule import WeightDecodeNetResidual
from SourceCode.Sketches.MetaSketch.RefineModule import RefineNet
from SourceCode.Sketches.MetaSketch.AttentionMatrix import AttentionMatrix


class MetaSketch(nn.Module):
    """
    元草图模型类，继承自抽象模型基类
    实现论文中提出的四大核心模块：嵌入、精炼、稀疏寻址、存储与解码
    """

    def __init__(self, slot_num):
        super(MetaSketch, self).__init__()
        self.embedding_net = EmbeddingNet(1, 23, 64)
        self.refine_net = RefineNet(23, 5, 32)
        self.memory_matrix = BasicMemoryMatrix(2, slot_num, 23)
        self.decode_net = WeightDecodeNetResidual(76, 256)
        self.attention_matrix = AttentionMatrix(5, slot_num, 2)

        model_dir = os.path.join(os.path.dirname(__file__), "meta_model")
        model_path = os.path.join(model_dir, f"meta_{slot_num}")
        full_model = torch.load(model_path, map_location="cpu")
        # filtered = {k: v for k, v in full_model.items() if k in self.state_dict()}
        missing, unexpected = self.load_state_dict(full_model, strict=False)
        del full_model

    def to(self, device):
        self.embedding_net = self.embedding_net.to(device)
        self.refine_net = self.refine_net.to(device)
        self.memory_matrix = self.memory_matrix.to(device)
        self.decode_net = self.decode_net.to(device)
        self.attention_matrix = self.attention_matrix.to(device)

    def clear(self, mem_size):
        self.memory_matrix.clear(mem_size)

    def dec_query(self, query_x, weight_sum_tensor):
        embedding = self.embedding_net(query_x)
        refined = self.refine_net(embedding)
        address = self.attention_matrix(refined)
        read_info = self.memory_matrix.read(address, embedding)
        decode_info = torch.cat((read_info, embedding, weight_sum_tensor), dim=1)
        dec_pred = self.decode_net(decode_info)
        return dec_pred

    def write(self, input_x, input_y):
        embedding = self.embedding_net(input_x)
        refined = self.refine_net(embedding)
        address = self.attention_matrix(refined)
        self.memory_matrix.write(address, embedding, input_y.squeeze(-1))
