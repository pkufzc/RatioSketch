import torch
import time
import threading
import queue
import os


class Trainer:

    def __init__(self, task_generator, model, device, main_optimizer, logger):
        self.task_generator = task_generator
        self.device = device
        self.model = model
        self.main_optimizer = main_optimizer
        self.logger = logger
        self.task_queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()

        self._producer_thread = threading.Thread(
            target=self._task_producer, daemon=True
        )
        self._producer_thread.start()

    def _task_producer(self):
        while not self._stop_event.is_set():
            if not self.task_queue.full():
                task = self.task_generator.get_train_task()
                self.task_queue.put(task)
            else:
                time.sleep(0.01)

    def train(self, train_task_nums):
        self.model.to(self.device)
        for task_index in range(train_task_nums):
            print(f"Task: {task_index}")
            self.main_optimizer.zero_grad()
            start_time = time.time()
            train_taskdata = self.task_queue.get()  # 阻塞直到有任务
            end_time = time.time()
            print("训练任务获取耗时为: ", end_time - start_time)
            if task_index % self.logger.eval_gap == 0:
                self.logger.logging(self.model, task_index)
            freqs_sum_list = train_taskdata.freqs_sum_list.repeat(
                train_taskdata.item_size_list, 1
            ).view(-1, 1)
            items = train_taskdata.items
            freqs = train_taskdata.freqs

            item_size = len(items) // 16
            memory_size = torch.randint(item_size, item_size * 32, (1,)).item()

            self.model.to()
            self.model.clear(
                d=self.model.base_sketch.d,
                w=memory_size // (4 * self.model.base_sketch.d),
            )
            self.model.write(items, freqs)
            dec_pred, read_head, heap_flags, _, dec_is = self.model.dec_query(
                items, freqs_sum_list
            )
            dec_loss = self.model.loss_func.forward_dec(
                dec_pred, read_head, heap_flags, freqs, dec_is
            )
            dec_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.main_optimizer.step()
        self._stop_event.set()
        self.logger.close_all_file()
