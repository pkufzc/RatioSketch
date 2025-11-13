"""Data loading helpers.

This module expects precomputed .npz files with 'items' and 'freqs'.
The default base_dir is set to the original author's path and should be
updated to point to local dataset files before running experiments.
"""

import numpy as np
from scipy.stats import linregress

# simple in-memory cache to avoid repeated IO
_DATASET_CACHE = {}


def load_dataset_cache(ds_name, train=True):
    """Load dataset npz into the global cache."""
    # WARNING: user should update base_dir to point to local dataset files
    base_dir = "./Dataset"
    if train:
        npz_path = f"{base_dir}/{ds_name}_freqs_train.npz"
    else:
        npz_path = f"{base_dir}/{ds_name}_freqs_test.npz"
    data = np.load(npz_path)
    items = data["items"]
    freqs = data["freqs"]
    _DATASET_CACHE[(ds_name, train)] = (items, freqs)
    return items, freqs


def estimate_zipf_param(freq):
    """Estimate Zipf parameter via linear regression on log(rank)-log(freq)."""
    freq_sorted = np.array(sorted(freq, reverse=True))
    ranks = np.arange(1, len(freq_sorted) + 1)
    log_ranks = np.log(ranks)
    log_freqs = np.log(freq_sorted)
    slope, *_ = linregress(log_ranks, log_freqs)
    zipf_param = -slope
    return zipf_param


def read_dat(item_size=None, max_is=None, train=True, ds_name="caida"):
    """Return (items, freqs, zipf_param). Loads and caches when needed."""
    key = (ds_name, train)
    if key not in _DATASET_CACHE:
        items, freqs = load_dataset_cache(ds_name, train)
    else:
        items, freqs = _DATASET_CACHE[key]
    items = items.copy()
    freqs = freqs.copy()
    total_num = len(freqs)
    if item_size:
        idx = np.random.choice(total_num, size=item_size, replace=True)
        items = items[idx]
        freqs = freqs[idx]
    if max_is:
        items = items[:max_is]
        freqs = freqs[:max_is]
    zipf_param = estimate_zipf_param(freqs)
    return items, freqs, zipf_param


if __name__ == "__main__":
    for name in [
        "caida",
        "mawi",
        "dc",
        "aol",
        "criteo",
        "webdocs",
    ]:
        _, freqs, zipf = read_dat(train=True, ds_name=name)
        print(
            f"Train {name} item size {len(freqs)}  stream length: {sum(freqs)}  zipf:  {zipf}"
        )
        _, freqs, zipf = read_dat(train=False, ds_name=name)
        print(
            f"Test {name} item size {len(freqs)}  stream length: {sum(freqs)}  zipf:  {zipf}"
        )
