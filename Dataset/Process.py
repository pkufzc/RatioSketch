import numpy as np
import pandas as pd
import random
import csv
from pathlib import Path

random.seed(0)


def process_and_save_split(items, max_items, train_path, test_path):

    items = np.array(items)
    if items.ndim > 1:
        items = [tuple(row) for row in items]
    split_idx = len(items) // 2
    train_items = items[:split_idx]
    test_items = items[split_idx:]
    print(len(train_items), len(test_items))
    train_freq = pd.Series(train_items).value_counts()
    test_freq = pd.Series(test_items).value_counts()

    train_item_list = train_freq.index.tolist()
    test_item_list = test_freq.index.tolist()
    if max_items:
        train_item_list = train_item_list[:max_items]
        test_item_list = test_item_list[:max_items]

    print(f"Train items count: {len(train_item_list)}")
    print(f"Test items count: {len(test_item_list)}")
    intersection_count = len(set(train_item_list) & set(test_item_list))
    print(f"Items in both train and test: {intersection_count}")
    all_items = list(set(train_item_list) | set(test_item_list))
    item_to_id = {item: 1 + idx for idx, item in enumerate(all_items)}
    train_ids = np.array([item_to_id[item] for item in train_item_list])
    train_freqs = train_freq.values[:max_items]
    test_ids = np.array([item_to_id[item] for item in test_item_list])
    test_freqs = test_freq.values[:max_items]
    train_stream = np.array([item_to_id[item] for item in train_items])
    np.savez(train_path, items=train_ids, freqs=train_freqs, stream=train_stream)
    test_stream = np.array([item_to_id[item] for item in test_items])
    print(len(train_stream), len(test_stream))
    np.savez(test_path, items=test_ids, freqs=test_freqs, stream=test_stream)


if __name__ == "__main__":
    pass
