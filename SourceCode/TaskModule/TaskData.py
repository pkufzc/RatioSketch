import torch


class TaskData:

    def __getitem__(self, index):
        return self.items[index], self.freqs[index]

    def __init__(self, items, freqs, zipf_info, device, ds_name="None"):
        super().__init__()
        self.ds_name = ds_name
        self.freqs = freqs
        self.items = items
        self.zipf_info = zipf_info
        self.device = device
        self.item_size_list = torch.tensor([len(items)], dtype=torch.long)
        self.freqs_sum_list = torch.tensor([sum(self.freqs)], dtype=torch.float32)
        self.to_device(self.device)

    def to_device(self, device):
        self.device = device
        self.items = self.items.to(self.device)
        self.freqs = self.freqs.to(self.device)
        self.zipf_info = self.zipf_info.to(self.device)
        self.item_size_list = self.item_size_list.to(self.device)
        self.freqs_sum_list = self.freqs_sum_list.to(self.device)

    def to_np(self):
        self.items = self.items.numpy()
        self.freqs = self.freqs.numpy()
        self.zipf_info = self.zipf_info.numpy()
        self.item_size_list = self.item_size_list.numpy()
        self.freqs_sum_list = self.freqs_sum_list.numpy()

    def to_tensor(self):
        self.items = torch.tensor(self.items)
        self.freqs = torch.tensor(self.freqs)
        self.zipf_info = torch.tensor(self.zipf_info)
        self.item_size_list = torch.tensor(self.item_size_list)
        self.freqs_sum_list = torch.tensor(self.freqs_sum_list)

    def __len__(self):
        return self.freqs.shape[0]
