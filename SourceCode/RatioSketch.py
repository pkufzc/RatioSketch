import torch
import torch.nn as nn


class RatioSketch(nn.Module):
    def __init__(self, decode_module, base_sketch, loss_func):
        super(RatioSketch, self).__init__()
        self.decode_module = decode_module
        self.base_sketch = base_sketch
        self.loss_func = loss_func
        self.device = None

    def to(self, device=None):
        if device:
            self.device = device
        self.decode_module.to(self.device)
        self.base_sketch.to(self.device)
        self.loss_func.to(self.device)

    def clear(self, d=None, w=None):
        with torch.no_grad():
            if d == None and w == None:
                self.base_sketch.clear(self.base_sketch.d, self.base_sketch.w)
            elif d == None:
                self.base_sketch.clear(self.base_sketch.d, w)
            else:
                self.base_sketch.clear(d, w)

    def write(self, batch_x, batch_supprt_y):
        self.base_sketch.update_batch(batch_x, batch_supprt_y)

    def dec_query(self, batch_query_x, batch_freqs_sum):
        batch_read_info = self.base_sketch.query_all_hashes(batch_query_x)
        read_freqs, heap_flags = self.base_sketch.query_final(
            batch_query_x, batch_read_info
        )
        dec_pred, dec_is = self.decode_module(
            batch_read_info,
            batch_freqs_sum,
            read_freqs,
            heap_flags,
            self.base_sketch.sketch.data,
        )
        return dec_pred, read_freqs, heap_flags, batch_read_info, dec_is
