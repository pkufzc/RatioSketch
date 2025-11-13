
from SourceCode.TaskModule.TaskData import TaskData
import torch
import random
import numpy as np
import multiprocessing as mp
from functools import partial
from Dataset.DataLoader import read_dat
mp.set_start_method('spawn', force=True)
import os
os.environ["OMP_NUM_THREADS"] = "35"
os.environ["MKL_NUM_THREADS"] = "35"
# 顶层定义多进程数据集加载函数，避免pickle问题
def load_one_dataset_for_mp(args):
    ds_name, use_stream = args
    print(f"[Child PID {os.getpid()}] Loading dataset: {ds_name}")
    from Dataset.DataLoader import read_dat
    items, freqs, zipf_info, stream = read_dat(
        item_size=None, train=False, ds_name=ds_name, use_stream=use_stream
    )
    items = torch.tensor(items, dtype=torch.long).unsqueeze(-1)
    freqs = torch.tensor(freqs, dtype=torch.float32).unsqueeze(-1)
    zipf_info = torch.tensor(zipf_info, dtype=torch.float32)
    stream = torch.tensor(stream, dtype=torch.long).unsqueeze(-1)
    taskdata = TaskData(items, freqs, zipf_info, stream, 'cpu', ds_name)
    print(f"[Child PID {os.getpid()}] Loading dataset: {ds_name} over")
    return ds_name, taskdata


class TaskGeneraotr:
    def __init__(
        self,
        zipf_generator,
        device,
        test_task_group_size,
        test_zipf_param_list,
        dataset_mem_dic,
        use_stream
    ):
        self.use_stream=use_stream
        self.dataset_mem_dic = dataset_mem_dic
        self.zipf_generator = zipf_generator
        self.device = device
        self.test_zipf_param_list = test_zipf_param_list
        self.dataset_is_dic = {
            "caida": 12300000,
            "mawi": 2000000,
            "dc": 4000000,
            "kosarak": 35000,
            "chainstore": 40000,
            "aol": 850000,
            "pcb": 4500000,
            "webdocs": 400000,
        }
        self.mp_queue = mp.Queue(maxsize=100)
        self._stop_event = mp.Event()
        self._producer_proc = mp.Process(target=self._task_producer, daemon=True)
        self._producer_proc.start()

    def get_train_task(self):
        # 从多进程队列阻塞获取任务
        return self.mp_queue.get()

    def _task_producer(self):
        # 后台进程持续生成任务
        import random
        while not self._stop_event.is_set():
            if not self.mp_queue.full():
                if random.random() < 0:
                    self.zipf_generator.transfer_device()
                    items, freqs, zipf_info = self.zipf_generator.sample_train_support()
                    stream = zipf_info
                else:
                    item_size = self.zipf_generator.get_item_size()
                    dataset_name = random.choice(list(self.dataset_mem_dic.keys()))
                    ds_is = self.dataset_is_dic.get(dataset_name)
                    items, freqs, zipf_info, stream = read_dat(item_size, None, True, dataset_name)
                    items = torch.tensor(items, dtype=torch.long).unsqueeze(-1)
                    freqs = torch.tensor(freqs, dtype=torch.float32).unsqueeze(-1)
                    zipf_info = torch.tensor(zipf_info, dtype=torch.float32)
                    stream = torch.tensor(stream, dtype=torch.long).unsqueeze(-1)
                train_taskdata = TaskData(
                    items,
                    freqs,
                    zipf_info,
                    stream,
                    self.device,
                )
                self.mp_queue.put(train_taskdata)
            else:
                import time
                time.sleep(0.01)


        
    def produce_test_task(self):
        """
        优化版：每个数据集只读取一次文件，缓存后批量生成任务，极大减少IO和内存消耗。
        """
        print("start generating test task (optimized, no repeated file IO)...")
        meta_task_group_discribe_list = []
        test_meta_task_group_list = []

        # 1. 多进程并行读取所有数据集的数据（函数已移到模块顶层）
        ds_names = list(self.dataset_mem_dic.keys())
        print(f"[Main PID {os.getpid()}] ds_names: {ds_names}")
        print(f"[Main PID {os.getpid()}] mp.cpu_count(): {mp.cpu_count()}")
        num_proc = min(len(ds_names), max(1, mp.cpu_count()-1, 4))
        print(f"[Main PID {os.getpid()}] Parallel loading datasets with {num_proc} processes...")
        with mp.Pool(processes=num_proc) as pool:
            results = pool.map(load_one_dataset_for_mp, [(ds_name, self.use_stream) for ds_name in ds_names])
        for ds_name, task_data in results:
            task_data.to_device(self.device)
            for memory_size in self.dataset_mem_dic[ds_name]:
                info = f"{ds_name.upper()}_MS_{memory_size}_SL_{task_data.freqs_sum_list[0].item()}"
                meta_task_group_discribe_list.append(info)
                test_meta_task_group_list.append([task_data])

        print(f"end generating test task... Created {len(test_meta_task_group_list)} task groups")
        return test_meta_task_group_list, meta_task_group_discribe_list
