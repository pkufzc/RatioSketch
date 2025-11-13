import numpy as np
import torch
from . import tcms_cpp


class TCMS:
    @property
    def sketch(self):
        return self.create_sketch()

    @property
    def data(self):
        return self.create_sketch()

    def __init__(self, d, w, device="cpu"):
        self.device = device
        self.d = d
        self.w = int(w * 4 / 3)
        self.cpp = tcms_cpp.TCMSCpp(self.d, self.w)

    def update_batch(self, x_batch, c_batch):
        x_np = np.asarray(
            x_batch.cpu() if hasattr(x_batch, "cpu") else x_batch, dtype=np.int64
        ).reshape(-1)
        c_np = np.asarray(
            c_batch.cpu() if hasattr(c_batch, "cpu") else c_batch, dtype=np.int32
        ).reshape(-1)
        self.cpp.update_batch(x_np, c_np)

    def query_all_hashes(self, x_batch):
        x_np = np.asarray(
            x_batch.cpu() if hasattr(x_batch, "cpu") else x_batch, dtype=np.int64
        ).reshape(-1)
        est = self.cpp.query_all_hashes(x_np)
        result = torch.from_numpy(np.array(est)).to(self.device)
        # print('result[:5]: ',result[:5])
        return result

    def to(self, device):
        self.device = device

    def clear(self, d, w):
        self.d = d
        self.w = int(w * 4 / 3)
        self.cpp = tcms_cpp.TCMSCpp(self.d, self.w)

    def create_sketch(self):
        arr = self.cpp.create_sketch()
        arr_np = np.array(arr).reshape(self.d, -1)
        return torch.from_numpy(arr_np).to(self.device)

    def query_final(self, x_batch, sketch_est=None):
        N = len(x_batch)
        if sketch_est is None:
            sketch_est = self.query_all_hashes(x_batch)
        min_est = torch.min(sketch_est, dim=1, keepdim=True)[0]
        flags = torch.ones((N, 1), dtype=torch.bool, device=self.device)
        return min_est, flags
