import torch
import torch.nn as nn
from SourceCode.Sketches.LegoSketch.EmbeddingModule import EmbeddingModule
from SourceCode.Sketches.LegoSketch.DecodeModule import SetDecodeModule
from SourceCode.Sketches.LegoSketch.AddressModule import RandomAddressModule
from SourceCode.Sketches.LegoSketch.MemoryModule import MemoryModule
import os

class LegoSketch(nn.Module):
    def __init__(self):
        super(LegoSketch, self).__init__()
        self.embedding_module = EmbeddingModule()
        self.memory_module = MemoryModule()
        self.decode_module = SetDecodeModule()
        self.address_module = RandomAddressModule()
        model_path = os.path.join(os.path.dirname(__file__), "Base_seed0_model")
        full_model = torch.load(model_path)
        # 只加载模型中有的参数
        filtered = {k: v for k, v in full_model.items() if k in self.state_dict()}
        self.load_state_dict(filtered, strict=False)
        del full_model

    def clear(self, batch_size=1):
        self.memory_module.clear(batch_size)

    def get_embedding(self, x):
        embedding = self.embedding_module(x)
        return embedding

    def get_address(self, x):
        address = self.address_module(x)
        return address

    def write(self, batch_input_x, batch_input_y, item_size_list=None):
        if item_size_list is None:
            item_size_list = [batch_input_x.shape[0]]
        batch_embedding = self.get_embedding(batch_input_x)
        batch_address = self.get_address(batch_input_x)
        self.memory_module.write(
            batch_address, batch_embedding, batch_input_y, item_size_list
        )

    def dec_query(self, batch_query_x, weight_sum_tensor, item_size_list=None):
        if item_size_list is None:
            item_size_list = [batch_query_x.shape[0]]
        batch_embedding = self.get_embedding(batch_query_x)
        batch_address = self.get_address(batch_query_x)
        batch_cm_readhead, batch_cm_info, batch_basic_read_info = (
            self.memory_module.read(batch_address, batch_embedding, item_size_list)
        )
        set_info = torch.stack((batch_basic_read_info, batch_embedding), dim=-1)
        batch_dec_pred, stream_info1, stream_info2 = self.decode_module(
            set_info,
            weight_sum_tensor,
            batch_cm_readhead,
            self.memory_module.memory_matrix.data,
            item_size_list,
        )
        # print(stream_info1.mean(), stream_info2.mean())
        if stream_info2.mean() > 10000 and stream_info1.mean() < 1.0:
            return batch_dec_pred
        else:
            return batch_cm_readhead
        # return batch_dec_pred
