import torch
import torch.nn as nn
from SourceCode.ModelModule.Hash import RandomHash


class CMS:
    def __init__(self, d, w, device="cpu"):
        self.d = d
        self.w = w
        self.device = device
        self.sketch = torch.zeros((d, w), dtype=torch.float32).to(device)
        self.hash_functions = nn.ModuleList(
            [RandomHash(hash_seed=i, hash_range=w) for i in range(d)]
        ).to(device)

    def to(self, device):
        self.device = device
        self.sketch = self.sketch.to(self.device)
        self.hash_functions = self.hash_functions.to(self.device)

    def clear(self, d, w):
        self.d = d
        self.w = w
        self.sketch = torch.zeros((d, w), dtype=torch.float32).to(self.device)
        self.hash_functions = nn.ModuleList(
            [RandomHash(hash_seed=i, hash_range=w) for i in range(d)]
        ).to(self.device)

    def update_batch(self, x_batch, c_batch):
        batch_size = x_batch.shape[0]
        x_batch = x_batch.to(self.device)
        c_batch = c_batch.to(self.device)

        for i in range(self.d):
            idx_tensor = (
                self.hash_functions[i](x_batch, batch_size, hash_num=1)
                .squeeze(1)
                .long()
            )
            self.sketch[i].scatter_add_(
                dim=0,
                index=idx_tensor,
                src=c_batch.squeeze(1).to(torch.float32),
            )

    def query_all_hashes(self, x_batch):
        batch_size = x_batch.shape[0]
        x_batch = x_batch.to(self.device)
        estimates = torch.zeros((batch_size, self.d), device=self.device)

        for i in range(self.d):
            idx_tensor = (
                self.hash_functions[i](x_batch, batch_size, hash_num=1)
                .squeeze(1)
                .long()
            )
            sketch_vals = self.sketch[i, idx_tensor]
            estimates[:, i] = sketch_vals

        return estimates

    def query_final(self, x_batch, sketch_est=None):
        N = len(x_batch)
        if sketch_est is None:
            x_batch = x_batch.to(self.device)
            sketch_est = self.query_all_hashes(x_batch)
        min_est = torch.min(sketch_est, dim=1, keepdim=True)[0]  # [N, 1]
        flags = torch.ones((N, 1), dtype=torch.bool, device=self.device)
        return min_est, flags
