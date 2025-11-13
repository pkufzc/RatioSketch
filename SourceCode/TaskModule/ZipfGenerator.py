"""Zipf-based sample generator.

Generates item id samples and integer weights following a Zipf-like
distribution for synthetic training and testing.
"""

import torch
import random
import numpy as np
from scipy.stats import linregress


class ZipfGenerator:
    """Sample item sets and integer weights from a Zipf family.

    This class supports both linear and log sampling of item sizes.
    """

    def __init__(
        self,
        zipf_param_lower,
        zipf_param_upper,
        sl_lower,
        sl_upper,
        is_lower,
        is_upper,
        device="cpu",
        sample_IS_scale="linear",
    ):
        super().__init__()

        self.samples_ndarray = None
        self.samples_tensor = None
        self.sample_index = 0
        self.zipf_param_upper = zipf_param_upper
        self.zipf_param_lower = zipf_param_lower
        self.is_lower = is_lower
        self.is_upper = is_upper
        self.sl_lower = sl_lower
        self.sl_upper = sl_upper
        self.device = device
        assert sample_IS_scale in ["linear", "log"]
        self.sample_IS_scale = sample_IS_scale
        self.sample_IS_func = (
            self.sample_by_linear_scale
            if sample_IS_scale == "linear"
            else self.sample_by_log_scale
        )
        print("sample_IS_scale:", sample_IS_scale)
        print("sample_IS_func:", self.sample_IS_func.__name__)
        self.prepare()

    def get_item_size(self):
        item_size = self.sample_IS_func(1)
        return item_size

    def transfer_device(self):
        self.samples_tensor = torch.tensor(
            self.samples_ndarray, device=self.device
        ).long()
        self.shuffle()

    def prepare(self):
        # pre-generate a large pool of unique ids
        self.samples_ndarray = np.arange(1000000).reshape(-1, 1)
        self.transfer_device()

    def shuffle(self):
        index = torch.randperm(self.samples_tensor.shape[0], device=self.device)
        self.samples_tensor = self.samples_tensor[index]

    def get_zipf_by_is(self, zipf_param, item_size, stream_length):
        with torch.no_grad():
            x = torch.arange(item_size + 1, 1, step=-1, device=self.device).float()
            x = x ** (-zipf_param)
            x = x / x.sum()
            vector = x * (1 / x[1])
            index = torch.randperm(vector.shape[0], device=self.device)
            vector = vector[index]
            return vector

    def sample_by_log_scale(self, upper_ratio):
        uniform_ratio = random.random() * upper_ratio
        item_size = int(
            ((self.is_upper / self.is_lower) ** uniform_ratio) * self.is_lower
        )
        return item_size

    def sample_by_linear_scale(self, upper_ratio):
        uniform_ratio = random.random() * upper_ratio
        item_size = int(uniform_ratio * (self.is_upper - self.is_lower) + self.is_lower)
        return item_size

    def sample_train_support(self, item_size=None, sl_size=None, zipf_param=None):
        # produce a random training sample: ids, integer weights, and zipf param
        assert item_size is None and sl_size is None and zipf_param is None
        upper_ratio = 1
        item_size = self.sample_IS_func(upper_ratio)
        sl_size = random.random() * (self.sl_upper - self.sl_lower) + self.sl_lower
        zipf_param = (
            random.random() * (self.zipf_param_upper - self.zipf_param_lower)
            + self.zipf_param_lower
        )
        weights = (self.get_zipf_by_is(zipf_param, item_size, sl_size)).int() + 1
        if item_size + self.sample_index >= self.samples_tensor.shape[0]:
            self.sample_index = 0
            self.shuffle()
        samples = self.samples_tensor[self.sample_index : item_size + self.sample_index]
        self.sample_index += item_size
        return (
            samples,
            weights.unsqueeze(-1),
            torch.tensor(zipf_param, device=self.device),
        )

    def sample_test_support(self, item_size=None, sl_size=None, zipf_param=None):
        # deterministic sampling for test: requires explicit params
        assert item_size is not None and sl_size is not None and zipf_param is not None
        weights = (self.get_zipf_by_is(zipf_param, item_size, sl_size)).int() + 1
        if item_size + self.sample_index >= self.samples_tensor.shape[0]:
            self.sample_index = 0
            self.shuffle()
        samples = self.samples_tensor[self.sample_index : item_size + self.sample_index]
        self.sample_index += item_size
        return (
            samples,
            weights.unsqueeze(-1),
            torch.tensor(zipf_param, device=self.device),
        )
