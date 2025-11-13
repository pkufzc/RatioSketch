"""Task generator and multiprocessing helpers.

This module spawns a background process that provides
training tasks and also loads test datasets in parallel for evaluation.
"""

from SourceCode.TaskModule.TaskData import TaskData
import torch
import numpy as np
import multiprocessing as mp
from Dataset.DataLoader import read_dat
import time
import os
import random

mp.set_start_method("spawn", force=True)
os.environ["OMP_NUM_THREADS"] = "35"
os.environ["MKL_NUM_THREADS"] = "35"


# top-level helper for multiprocessing dataset load (avoids pickle issues)
def load_one_dataset_for_mp(ds_name):
    print(f"[Child PID {os.getpid()}] Loading dataset: {ds_name}")
    from Dataset.DataLoader import read_dat

    items, freqs, zipf_info = read_dat(item_size=None, train=False, ds_name=ds_name)
    items = torch.tensor(items, dtype=torch.long).unsqueeze(-1)
    freqs = torch.tensor(freqs, dtype=torch.float32).unsqueeze(-1)
    zipf_info = torch.tensor(zipf_info, dtype=torch.float32)
    taskdata = TaskData(items, freqs, zipf_info, "cpu", ds_name)
    print(f"[Child PID {os.getpid()}] Loading dataset: {ds_name} over")
    return ds_name, taskdata


class TaskGeneraotr:
    """Produces training tasks and preloads test tasks.

    A background process fills a queue with training TaskData objects.
    produce_test_task reads each dataset once (in parallel) and prepares
    test groups used by the Logger for evaluation.
    """

    def __init__(self, zipf_generator, device, dataset_mem_dic):
        self.dataset_mem_dic = dataset_mem_dic
        self.zipf_generator = zipf_generator
        self.device = device
        self.dataset_is_dic = {
            "caida": 12300000,
            "mawi": 2000000,
            "dc": 4000000,
            "aol": 850000,
            "criteo": 4500000,
            "webdocs": 400000,
        }
        self.mp_queue = mp.Queue(maxsize=100)
        self._stop_event = mp.Event()
        self._producer_proc = mp.Process(target=self._task_producer, daemon=True)
        self._producer_proc.start()

    def get_train_task(self):
        # blocking get from the multiprocessing queue
        return self.mp_queue.get()

    def _task_producer(self):

        while not self._stop_event.is_set():
            if not self.mp_queue.full():
                if random.random() < 0:
                    # unused branch: sample synthetic ids from ZipfGenerator
                    self.zipf_generator.transfer_device()
                    items, freqs, zipf_info = self.zipf_generator.sample_train_support()
                else:
                    item_size = self.zipf_generator.get_item_size()
                    dataset_name = random.choice(list(self.dataset_mem_dic.keys()))
                    items, freqs, zipf_info = read_dat(
                        item_size, None, True, dataset_name
                    )
                    items = torch.tensor(items, dtype=torch.long).unsqueeze(-1)
                    freqs = torch.tensor(freqs, dtype=torch.float32).unsqueeze(-1)
                    zipf_info = torch.tensor(zipf_info, dtype=torch.float32)
                train_taskdata = TaskData(items, freqs, zipf_info, self.device)
                self.mp_queue.put(train_taskdata)
            else:
                time.sleep(0.01)

    def produce_test_task(self):
        """Load each dataset once in parallel and prepare test groups.

        This reduces repeated IO and prepares a list of (task_group, desc)
        pairs consumed by the Logger for evaluation.
        """
        print("start generating test task (optimized, no repeated file IO)...")
        meta_task_group_discribe_list = []
        test_meta_task_group_list = []

        # 1. parallel load all datasets using multiprocessing pool
        ds_names = list(self.dataset_mem_dic.keys())
        print(f"[Main PID {os.getpid()}] ds_names: {ds_names}")
        print(f"[Main PID {os.getpid()}] mp.cpu_count(): {mp.cpu_count()}")
        num_proc = min(len(ds_names), max(1, mp.cpu_count() - 1, 4))
        print(
            f"[Main PID {os.getpid()}] Parallel loading datasets with {num_proc} processes..."
        )
        with mp.Pool(processes=num_proc) as pool:
            results = pool.map(load_one_dataset_for_mp, ds_names)
        for ds_name, task_data in results:
            task_data.to_device(self.device)
            for memory_size in self.dataset_mem_dic[ds_name]:
                info = f"{ds_name.upper()}_MS_{memory_size}_SL_{task_data.freqs_sum_list[0].item()}"
                meta_task_group_discribe_list.append(info)
                test_meta_task_group_list.append([task_data])

        print(
            f"end generating test task... Created {len(test_meta_task_group_list)} task groups"
        )
        return test_meta_task_group_list, meta_task_group_discribe_list
