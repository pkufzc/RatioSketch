import csv
import json
import os
import sys
import time
import shutil
import re
import glob
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from SourceCode.Sketches.TCMS.TCMS import TCMS
from SourceCode.Sketches.CMS import CMS
from SourceCode.Sketches.LegoSketch.Model import LegoSketch
from SourceCode.Sketches.MetaSketch.Model import MetaSketch
import random

mpl.use("Agg")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["legend.fontsize"] = 12


class Logger:
    def __init__(
        self,
        task_group_list,
        task_discribe_list,
        task_comment,
        config,
        eval_gap,
        eval_metrics,
        use_stream,
    ):
        self.use_stream = use_stream
        self.metrics = eval_metrics
        self.eval_gap = eval_gap
        self.config = config
        self.model_path = None
        self.task_discribe_list = task_discribe_list
        self.task_group_list = task_group_list
        self.task_comment = task_comment
        self.project_root = self._find_project_root()
        self.path = os.getcwd()
        self.py_file_path = sys.argv[0]
        self.dataset_name = "Ratio_" + time.strftime("%m%d", time.localtime())
        self.csv_writer_list = None
        self.log_file_list = None
        self.file_header = None
        self.model_dir = None
        self.init_log_file()

    def _find_project_root(self):
        """递归向上查找包含 SourceCode 目录的项目根路径"""
        cur_dir = os.path.abspath(os.getcwd())
        while True:
            if os.path.isdir(os.path.join(cur_dir, "SourceCode")):
                return cur_dir
            parent = os.path.dirname(cur_dir)
            if parent == cur_dir:
                break
            cur_dir = parent
        raise FileNotFoundError(
            "未找到包含 SourceCode 目录的项目根路径，请检查项目结构。"
        )

    def init_log_file(self):
        """初始化日志文件和相关目录"""
        # 生成时间戳和基础路径
        timestamp = time.strftime("_%m_%d_%H_%M_%S", time.localtime())
        base_dir = os.path.join(
            self.project_root,
            f"LogDir/{self.dataset_name}/{self.task_comment}{timestamp}",
        )
        os.makedirs(base_dir, exist_ok=True)
        self._save_config(base_dir)
        self._init_log_writers(base_dir)
        # self._backup_tasks(base_dir, timestamp)
        self._backup_code(base_dir)
        self.model_path = os.path.join(base_dir, f"{self.task_comment}_model")
        self.model_dir = base_dir

    def _save_config(self, base_dir):
        """保存配置信息到配置文件"""
        with open(os.path.join(base_dir, "config"), "w", encoding="utf-8") as f:
            json.dump(self.config, f)

    def _init_log_writers(self, base_dir):
        """初始化日志文件和CSV写入器"""
        self.log_file_list = []
        self.csv_writer_list = []
        for desc in self.task_discribe_list:
            log_path = os.path.join(base_dir, f"{desc}.csv")
            file = open(log_path, "w", newline="", encoding="utf-8")
            self.log_file_list.append(file)
            self.csv_writer_list.append(csv.writer(file))

    def _backup_tasks(self, base_dir, timestamp):
        """备份测试任务数据"""
        tasks_dir = os.path.join(base_dir, f"tasks_{self.task_comment}{timestamp}")
        os.makedirs(tasks_dir, exist_ok=True)

        for i, task_group in enumerate(self.task_group_list):
            group_dir = os.path.join(tasks_dir, self.task_discribe_list[i])
            os.makedirs(group_dir, exist_ok=True)

            for j, task in enumerate(task_group):
                task_path = os.path.join(group_dir, f"{j}.npz")
                self.save_taskdata(task, task_path)

        print("测试任务数据备份完成")

    def _backup_code(self, base_dir):
        """备份源代码和实验代码"""
        # 自动查找项目根目录
        self.project_root = self._find_project_root()
        source_dir = os.path.join(self.project_root, "SourceCode")
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"源代码目录不存在: {source_dir}")
        # 备份源代码
        shutil.copytree(
            source_dir,
            os.path.join(base_dir, "SourceCode"),
            dirs_exist_ok=True,
        )

        # 标准化路径分隔符并备份实验代码
        self.py_file_path = self.py_file_path.replace("\\", "/")
        self.path = self.path.replace("\\", "/")

        exp_dir = os.path.join(
            base_dir, f"ExpCode/{{}}".format(self.path.split("/")[-1])
        )
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy(self.py_file_path, exp_dir)

        print("代码备份完成")

    def save_taskdata(self, taskdata, path):
        """保存任务数据到文件"""
        np.savez(
            path,
            items=taskdata.items.cpu().numpy(),
            freqs=taskdata.freqs.cpu().numpy(),
            zipf_info=taskdata.zipf_info.cpu().numpy(),
            item_size_list=taskdata.item_size_list.cpu().numpy(),
            freqs_sum_list=taskdata.freqs_sum_list.cpu().numpy(),
        )

    def close_all_file(self):
        """关闭所有日志文件"""
        for file in self.log_file_list:
            file.close()

    def logging(self, model, step):
        print(f"{step} step 开始记录日志...")
        model.eval()  # 设为评估模式
        for i in range(len(self.log_file_list)):
            log_file = self.log_file_list[i]
            csv_writer = self.csv_writer_list[i]
            group_results = self.eval_on_one_group(
                model,
                self.task_group_list[i],
                self.task_discribe_list[i],
            )
            if self.file_header is None:
                self.file_header = list(group_results.keys())
                self.file_header.insert(0, "task_num")
                for writer in self.csv_writer_list:
                    writer.writerow(self.file_header)
            group_results["task_num"] = step
            row_content = [group_results[key] for key in self.file_header]
            csv_writer.writerow(row_content)
            log_file.flush()

        model.train()
        torch.save(model.state_dict(), self.model_path)
        print("模型保存成功")
        self.plot_metrics()
        self.plot_avg_throughput_bar()
        # self.plot_top_f1score_bar()
        print(f"{step} step 日志记录完成")

    def eval_on_one_group(self, model, taskdata_group, group_desc):
        group_dict_list = []
        for taskdata in taskdata_group:
            task_results = self.get_basic_eval_info_on_one_task(
                taskdata, model, group_desc
            )
            group_dict_list.append(task_results)
        group_merged_results = {}
        for key in group_dict_list[0].keys():
            avg_value = sum(dic[key] for dic in group_dict_list) / len(group_dict_list)
            group_merged_results[key] = avg_value
        return group_merged_results

    def get_basic_eval_info_on_one_task(self, taskdata, model, task_desc):
        info_dict = {}
        seed = int(time.time() * 1000) % (2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 对freqs每个元素乘以[0.8,1.2]的随机数，并更新freqs_sum_list
        temp_taskdata = copy.deepcopy(taskdata)
        with torch.no_grad():
            # 取原始freqs
            freqs = temp_taskdata.freqs
            # 生成同shape的随机数
            rand_scale = torch.empty_like(freqs).uniform_(0.95, 1.05)
            freqs = freqs * rand_scale
            temp_taskdata.freqs = freqs
            # 更新freqs_sum_list
            temp_taskdata.freqs_sum_list = torch.tensor(
                [freqs.sum().item()], dtype=torch.float32, device=freqs.device
            )
            match = re.search(r"_MS_(\d+)_", task_desc)
            mem_size = int(match.group(1)) * 1024
            sketch_map = {
                "CMS": CMS,
                "Tower": TCMS,
            }
            for metric in self.metrics:
                name, _, device_list = metric
                if name in ["CMS+RS", "Tower+RS"]:
                    info_dict.update(
                        self._eval_ratio(
                            model, temp_taskdata, mem_size, device_list, name
                        )
                    )
                elif name == "Lego":
                    info_dict.update(
                        self._eval_lego(mem_size, temp_taskdata, device_list)
                    )
                elif name == "Meta":
                    info_dict.update(
                        self._eval_meta(mem_size, temp_taskdata, device_list)
                    )
                else:
                    model_class = sketch_map.get(name)
                    info_dict.update(
                        self._eval_sketch(
                            mem_size, temp_taskdata, model_class, name, device_list
                        )
                    )
        return info_dict

    def get_f1score(self, pred_mask, true_mask):
        tp = (pred_mask & true_mask).sum().item()
        fp = (pred_mask & (~true_mask)).sum().item()
        fn = ((~pred_mask) & true_mask).sum().item()
        precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall + 1e-8)
            if (precision + recall) > 0
            else float("nan")
        )
        return f1

    def _eval_ratio(self, ratio_model, taskdata, mem_size, device_list, model_name):
        results = {}
        items = taskdata.items
        freqs = taskdata.freqs
        zipf_info = taskdata.zipf_info
        ds_name = taskdata.ds_name
        results["Zipf"] = zipf_info.cpu().item()
        results["Item_size"] = len(items)
        results["Label_sum"] = freqs.sum().cpu().item()
        temp_base_sketch = ratio_model.base_sketch
        logdir = os.path.join(os.path.dirname(__file__), "../LogDir")
        if model_name == "CMS+RS":
            model_path = os.path.join(
                logdir, f"模型记录/效果CMS d=3/10kb_{ds_name}/10kb_model"
            )
            state_dict = torch.load(model_path)
            ratio_model.load_state_dict(state_dict, strict=False)
            del state_dict
            ratio_model.base_sketch = CMS(d=3, w=mem_size // (4 * 3))
        elif model_name == "Tower+RS":
            model_path = os.path.join(
                logdir, f"模型记录/效果TCMS d=3/10kb_{ds_name}/10kb_model"
            )
            state_dict = torch.load(model_path)
            ratio_model.load_state_dict(state_dict, strict=False)
            del state_dict
            ratio_model.base_sketch = TCMS(d=3, w=mem_size // (4 * 3))

        for device_type in device_list:
            dev = "GPU" if device_type == "cuda" else "CPU"
            taskdata.to_device(device_type)
            ratio_model.to(device_type)

            items = taskdata.items
            freqs = taskdata.freqs
            zipf_info = taskdata.zipf_info
            freqs_sum = taskdata.freqs_sum_list.repeat(taskdata.item_size_list, 1)

            torch.cuda.synchronize()
            store_start = time.perf_counter()
            ratio_model.write(items, freqs)
            torch.cuda.synchronize()
            store_end = time.perf_counter()
            store_elapsed = store_end - store_start

            torch.cuda.synchronize()
            query_start = time.perf_counter()
            ratio_pred, *_ = ratio_model.dec_query(items, freqs_sum, False)
            torch.cuda.synchronize()
            query_end = time.perf_counter()
            query_elapsed = query_end - query_start
            results[f"{model_name}_STORE_{dev}_THROUGHPUT"] = (
                items.numel() / store_elapsed
            )
            results[f"{model_name}_QUERY_{dev}_THROUGHPUT"] = (
                items.numel() / query_elapsed
            )

        results.update(self._calculate_metrics(ratio_pred, freqs, f"{model_name}"))
        top_mask = self._get_top_mask(freqs)
        top_freqs = freqs[top_mask]
        top_ratio_pred = ratio_pred[top_mask]
        results.update(
            self._calculate_metrics(top_ratio_pred, top_freqs, f"TOP_{model_name}")
        )
        top_ratio_pred_mask = self._get_top_mask(ratio_pred)
        f1 = self.get_f1score(top_ratio_pred_mask, top_mask)
        results[f"{model_name}_TOP_F1"] = f1

        ratio_model.base_sketch = temp_base_sketch
        return results

    def _eval_meta(self, mem_size, taskdata, device_list):
        results = {}
        for device_type in device_list:
            taskdata.to_device(device_type)
            items = taskdata.items
            freqs = taskdata.freqs
            dev = "GPU" if device_type == "cuda" else "CPU"

            group_num = mem_size // 102400
            total_items = len(items)
            base_size = total_items // group_num
            remainder = total_items % group_num
            item_size_list = [
                base_size + 1 if i < remainder else base_size for i in range(group_num)
            ]
            support_y_np = freqs.cpu().numpy()
            weight_sum_list = []
            start = 0
            for size in item_size_list:
                group_sum = float(np.sum(support_y_np[start : start + size]))
                weight_sum_list.extend([group_sum] * size)
                start += size
            freqs_sum = torch.tensor(
                weight_sum_list, dtype=freqs.dtype, device=freqs.device
            ).unsqueeze(1)

            store_elapsed = 0
            query_elapsed = 0
            meta_pred_list = []
            start_index = 0
            for item_size in item_size_list:
                temp_items = items[start_index : start_index + item_size]
                temp_freqs = freqs[start_index : start_index + item_size]
                temp_freqs_sum = freqs_sum[start_index : start_index + item_size]
                torch.cuda.synchronize()
                store_start = time.perf_counter()
                # 这里固定使用100kb的模型
                meta = MetaSketch(556)
                meta.to(device_type)
                meta.write(temp_items, temp_freqs)
                torch.cuda.synchronize()
                store_end = time.perf_counter()
                store_elapsed += store_end - store_start

                torch.cuda.synchronize()
                query_start = time.perf_counter()
                meta_pred = meta.dec_query(temp_items, temp_freqs_sum)
                torch.cuda.synchronize()
                query_end = time.perf_counter()
                query_elapsed += query_end - query_start
                meta_pred_list.append(meta_pred)
                start_index += item_size
            meta_pred = torch.cat(meta_pred_list, dim=0)
            # 在这里结束
            results[f"Meta_STORE_{dev}_THROUGHPUT"] = items.numel() / store_elapsed
            results[f"Meta_QUERY_{dev}_THROUGHPUT"] = items.numel() / query_elapsed

        results.update(self._calculate_metrics(meta_pred, freqs, "Meta"))
        top_mask = self._get_top_mask(freqs)
        top_freqs = freqs[top_mask]
        top_meta_pred = meta_pred[top_mask]
        results.update(self._calculate_metrics(top_meta_pred, top_freqs, "TOP_Meta"))
        return results

    def _eval_lego(self, mem_size, taskdata, device_list):
        results = {}
        for device_type in device_list:
            taskdata.to_device(device_type)
            items = taskdata.items
            freqs = taskdata.freqs
            dev = "GPU" if device_type == "cuda" else "CPU"
            group_num = mem_size // 102400
            total_items = len(items)
            base_size = total_items // group_num
            remainder = total_items % group_num
            item_size_list = [
                base_size + 1 if i < remainder else base_size for i in range(group_num)
            ]
            support_y_np = freqs.cpu().numpy()
            weight_sum_list = []
            start = 0
            for size in item_size_list:
                group_sum = float(np.sum(support_y_np[start : start + size]))
                weight_sum_list.extend([group_sum] * size)
                start += size
            weight_sum = torch.tensor(
                weight_sum_list, dtype=freqs.dtype, device=freqs.device
            ).unsqueeze(1)

            store_elapsed = 0
            query_elapsed = 0
            lego_pred_list = []
            start_index = 0
            for item_size in item_size_list:
                temp_items = items[start_index : start_index + item_size]
                temp_freqs = freqs[start_index : start_index + item_size]
                temp_weight_sum = weight_sum[start_index : start_index + item_size]
                lego = LegoSketch()
                lego = lego.to(device_type)
                lego.clear(1)
                torch.cuda.synchronize()
                store_start = time.perf_counter()
                lego.write(temp_items, temp_freqs, [item_size])
                torch.cuda.synchronize()
                store_end = time.perf_counter()
                store_elapsed += store_end - store_start

                torch.cuda.synchronize()
                query_start = time.perf_counter()
                lego_pred = lego.dec_query(temp_items, temp_weight_sum, [item_size])
                torch.cuda.synchronize()
                query_end = time.perf_counter()
                query_elapsed += query_end - query_start
                lego_pred_list.append(lego_pred)
                start_index += item_size
            lego_pred = torch.cat(lego_pred_list, dim=0)

            results[f"Lego_STORE_{dev}_THROUGHPUT"] = items.numel() / store_elapsed
            results[f"Lego_QUERY_{dev}_THROUGHPUT"] = items.numel() / query_elapsed

        results.update(self._calculate_metrics(lego_pred, freqs, "Lego"))
        top_mask = self._get_top_mask(freqs)
        top_freqs = freqs[top_mask]
        top_lego_pred = lego_pred[top_mask]
        results.update(self._calculate_metrics(top_lego_pred, top_freqs, "TOP_Lego"))
        return results

    def _eval_sketch(self, mem_size, taskdata, model_class, model_name, device_list):
        results = {}
        for device_type in device_list:
            taskdata.to_device(device_type)
            items = taskdata.items
            freqs = taskdata.freqs
            dev = "GPU" if device_type == "cuda" else "CPU"
            print("model_name: ", model_name)
            model = model_class(d=3, w=mem_size // (4 * 3), device=device_type)
            torch.cuda.synchronize()
            if self.use_stream and model_name in []:
                input_items = taskdata.stream
                input_freqs = torch.ones_like(input_items)
            else:
                input_items = items
                input_freqs = freqs
            store_start = time.perf_counter()
            model.update_batch(input_items, input_freqs)
            torch.cuda.synchronize()
            store_end = time.perf_counter()
            store_elapsed = store_end - store_start

            torch.cuda.synchronize()
            query_start = time.perf_counter()
            pred = model.query_final(items)[0]
            torch.cuda.synchronize()
            query_end = time.perf_counter()
            query_elapsed = query_end - query_start

            results[f"{model_name}_STORE_{dev}_THROUGHPUT"] = (
                items.numel() / store_elapsed
            )
            results[f"{model_name}_QUERY_{dev}_THROUGHPUT"] = (
                items.numel() / query_elapsed
            )

        results.update(self._calculate_metrics(pred, freqs, model_name))
        top_mask = self._get_top_mask(freqs)
        top_freqs = freqs[top_mask]
        top_pred = pred[top_mask]
        results.update(
            self._calculate_metrics(top_pred, top_freqs, f"TOP_{model_name}")
        )
        return results

    def _get_top_mask(self, freqs, top_percent=0.2):
        freqs = freqs.flatten()  # 保证一维
        n = len(freqs)
        k = int(n * top_percent)
        sorted_indices = torch.argsort(freqs, descending=True)
        topk_indices = sorted_indices[:k]
        mask = torch.zeros_like(freqs, dtype=torch.bool)
        mask[topk_indices] = True
        return mask

    def _calculate_metrics(self, pred, true, name_prefix=""):
        metrics = {}
        # 计算基本指标
        aae = (pred - true).abs().mean()
        are = ((pred - true) / true).abs().mean()
        aoe = ((pred - true).abs().sum() / true.abs().sum()).mean()

        metrics[f"{name_prefix}_ARE"] = are.cpu().item()
        metrics[f"{name_prefix}_AAE"] = aae.cpu().item()
        metrics[f"{name_prefix}_AOE"] = aoe.cpu().item()
        return metrics

    def plot_metrics(self):
        """绘制所有方法的评估指标图"""
        # 定义要显示的方法和颜色
        metrics = self.metrics

        # 解析任务描述，分类为合成数据集和真实数据集
        zipf_datasets, real_datasets = self._parse_task_descriptions()

        # 绘制合成数据集指标图
        for zipf_value, ms_list in zipf_datasets.items():
            self._plot_dataset_metrics(
                ms_list, metrics, f"ZIPF_{zipf_value}", f"Metrics for ZIPF {zipf_value}"
            )

        # 绘制真实数据集指标图
        for ds_name, ms_list in real_datasets.items():
            self._plot_dataset_metrics(
                ms_list, metrics, ds_name, f"Metrics for {ds_name}"
            )

        print("指标图保存完成")

    def _parse_task_descriptions(self):
        """解析任务描述，将其分类为合成数据集和真实数据集

        Returns:
            tuple: (合成数据集字典, 真实数据集字典)
        """
        zipf_to_ms_map = {}
        real_dataset_map = {}

        for desc in self.task_discribe_list:
            # 处理合成数据集
            match = re.search(r"ZIPF_(\d+\.\d+)_MS_(\d+)", desc)
            if match:
                zipf_value = float(match.group(1))
                mem_size = int(match.group(2))
                if zipf_value not in zipf_to_ms_map:
                    zipf_to_ms_map[zipf_value] = []
                zipf_to_ms_map[zipf_value].append((mem_size, desc))
                continue

            # 处理真实数据集
            REAL_DATASETS = [
                "CAIDA",
                "MAWI",
                "DC",
                "WEBDOCS",
                "AOL",
                "PCB",
            ]
            match_real = re.search(f"^({'|'.join(REAL_DATASETS)})_MS_(\d+)", desc)
            if match_real:
                ds_name = match_real.group(1)
                mem_size = int(match_real.group(2))
                if ds_name not in real_dataset_map:
                    real_dataset_map[ds_name] = []
                real_dataset_map[ds_name].append((mem_size, desc))

        return zipf_to_ms_map, real_dataset_map
