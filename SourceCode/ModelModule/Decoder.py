"""Decoder that predicts a coefficient applied to base-sketch reads.

The decoder outputs a per-item coefficient (in (0,1)) which is
multiplicatively applied to the raw read frequencies from the base
sketch to produce final frequency estimates.
"""

import torch
import torch.nn as nn


class RatioDecoder(nn.Module):
    """Learned decoder that outputs a coefficient per item.

    The network encodes global sketch features and local read statistics,
    then produces a coefficient in (0,1) which is multiplied by the
    raw read frequencies to yield final predictions.
    """

    def __init__(self):
        super().__init__()
        self.filter_slot_size = 64
        # encode a small slice of memory as global feature
        self.global_enc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        # produce a load ratio scalar from global encoding
        self.load_enc = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU(),
        )
        # encode local per-item statistics
        self.local_info_enc = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        # final MLP that outputs coefficient in (0,1)
        self.ratio_dec = nn.Sequential(
            nn.Linear(19, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        read_values,
        sum_info,
        read_freqs,
        heap_flags,
        memory,
    ):
        # derive basic structural numbers
        slot_num = memory.shape[-1]
        d = memory.shape[-2]
        device = memory.device
        batch_size = read_values.shape[0]

        # global features from a small slice of memory
        mem_feat = 1000 * memory[:, : self.filter_slot_size] / sum_info.mean()
        global_info = self.global_enc(mem_feat.unsqueeze(-1)).mean(-2)
        load_ratio = self.load_enc(global_info).mean(-2)
        is_num = load_ratio * slot_num
        global_info = global_info.sum(-2)
        sl_num = sum_info[0].item()
        global_info = global_info.expand(batch_size, -1)

        # structural info vector
        struct_info = (
            torch.tensor(
                [load_ratio.item() / d, d * slot_num / sl_num, is_num.item() / sl_num],
                device=device,
            )
            .float()
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # local statistics from read values and read freqs
        read_freqs = read_freqs.clamp(min=1)
        sort_vals, _ = torch.sort(read_values, dim=-1, descending=False)
        sort_ratios = (sort_vals - read_freqs) / read_freqs
        sort_mean = sort_ratios.mean(dim=-1, keepdim=True)
        sort_std = sort_ratios.std(dim=-1, keepdim=True)
        sort_median = sort_ratios.median(dim=-1, keepdim=True)[0]
        sort_ratios = torch.cat((sort_ratios, heap_flags.float()), dim=-1)
        freq_weights = slot_num * read_freqs / sum_info

        local_info = self.local_info_enc(
            torch.cat(
                (sort_ratios, sort_mean, sort_std, sort_median, freq_weights), dim=-1
            )
        )

        input_features = torch.cat([local_info * global_info, struct_info], dim=-1)
        coefficient = self.ratio_dec(input_features)
        dec_pred = coefficient * read_freqs.detach()
        return dec_pred, is_num
