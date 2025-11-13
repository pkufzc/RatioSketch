import torch
import torch.nn as nn


class RatioDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter_slot_size = 64
        # memory/sketch全局特征编码
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
        self.is_enc = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU(),
        )
        self.local_info_enc = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
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
        # 结构参数
        slot_num = memory.shape[-1]
        d = memory.shape[-2]
        device = memory.device
        batch_size = read_values.shape[0]
        mem_feat = 1000 * memory[:, : self.filter_slot_size] / sum_info.mean()
        global_info = self.global_enc(mem_feat.unsqueeze(-1)).mean(-2)
        # print('global_info: ',global_info)
        is_ratio = self.is_enc(global_info).mean(-2)
        is_num = is_ratio * slot_num
        global_info = global_info.sum(-2)
        sl_num = sum_info[0].item()
        global_info = global_info.expand(batch_size, -1)
        # 结构特征

        struct_info = (
            torch.tensor(
                [is_ratio.item() / d, d * slot_num / sl_num, is_num.item() / sl_num],
                device=device,
            )
            .float()
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

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


if __name__ == "__main__":
    model = RatioDecoder()
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = total_params * 4  # float24 占4字节
    total_MB = total_bytes / 1024

    print(f"参数总数: {total_params}")
    print(f"占用内存: {total_MB:.2f} KB")
