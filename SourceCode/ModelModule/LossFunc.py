import torch
from torch import nn


class AutoWeightLoss(nn.Module):
    def __init__(self, num=2, weights=None, device=None):
        self.weights = weights
        learnable = True
        if weights is not None:
            learnable = False
        self.learnable = learnable
        super(AutoWeightLoss, self).__init__()
        if learnable:
            params = torch.ones(num, requires_grad=learnable)
            self.params = torch.nn.Parameter(params)
        else:
            self.params = torch.tensor(weights, requires_grad=learnable, device=device)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(
                1 + self.params[i] ** 2
            )
        return loss_sum


class RatioLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_loss = AutoWeightLoss(2)

    def print_real_ratio_max(self, real_ratio, confidence=0.95):
        real_ratio = real_ratio.flatten()
        sorted_vals, _ = torch.sort(real_ratio)
        N = len(sorted_vals)
        low_idx = int((1.0 - confidence) / 2 * N)
        high_idx = int((1.0 + confidence) / 2 * N)
        max_in_ci = sorted_vals[low_idx:high_idx].max().item()
        print(f"real_ratio {confidence*100:.1f}%置信区间最大值: {max_in_ci:.6f}")

    def forward_dec(self, batch_pred, batch_read_head, heap_flags, batch_y, dec_is):
        batch_y = batch_y.clamp(min=1)
        batch_read_head = batch_read_head.clamp(min=1)
        heap_mask = heap_flags.view(-1)
        tail_mask = ~heap_mask

        # heap部分

        ratio_loss_heap = 0
        heap_pred = batch_pred[heap_mask]
        heap_y = batch_y[heap_mask]
        heap_weight = torch.log2(heap_y + 1)
        # heap_weight = 1
        if heap_mask.any():
            dec_ratio_heap = heap_pred / batch_read_head[heap_mask]
            real_ratio_heap = heap_y / batch_read_head[heap_mask]
            ratio_loss_heap = (
                (heap_weight * (dec_ratio_heap - real_ratio_heap)).abs().mean()
            )

        # 此处没有loss_tail
        # ratio_loss_tail = 0
        # tail_pred = batch_pred[tail_mask]
        # tail_y = batch_y[tail_mask]
        # tail_weight = 1
        # if tail_mask.any():
        #     dec_ratio_tail = tail_pred / batch_read_head[tail_mask]
        #     real_ratio_tail = tail_y / batch_read_head[tail_mask]
        #     ratio_loss_tail = (
        #         (tail_weight * (dec_ratio_tail - real_ratio_tail)).abs().mean()
        #     )

        real_is = (
            torch.tensor(len(batch_pred), device=dec_is.device).view(-1, 1).float()
        )
        item_size_loss = (dec_is / real_is - 1).abs().mean()

        return self.final_loss(ratio_loss_heap, item_size_loss)
