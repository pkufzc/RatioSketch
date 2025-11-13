from SourceCode.Trainer import Trainer
from SourceCode.RatioSketch import *
from SourceCode.TaskModule.ZipfGenerator import ZipfGenerator
from SourceCode.ModelModule.Decoder import *
from SourceCode.ModelModule.LossFunc import *
from SourceCode.Logger import *
from SourceCode.TaskModule.TaskGenerator import TaskGeneraotr
import os
import random
import numpy as np
from SourceCode.ModelModule.Hash import *
from SourceCode.Sketches.CMS import CMS
from SourceCode.Sketches.TCMS.TCMS import TCMS


class Factory:
    def __init__(self, config):
        self.config = config
        # self.seed_everything(1)

    def seed_everything(self, seed=0):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def init_trainer(self):
        print("init ratiosketch....")
        device = self.init_device()
        ratiosketch = self.init_ratiosketch(device)
        task_generator = self.init_task_generator(device)
        test_task_group_list, task_discribe_list = task_generator.produce_test_task()
        logger = self.init_logger(test_task_group_list, task_discribe_list)
        main_optimizer = self.init_optimizer(ratiosketch)
        trainer = Trainer(task_generator, ratiosketch, device, main_optimizer, logger)
        print("init trainer done!")
        return trainer

    def init_task_generator(self, device):
        test_task_group_size = self.config["logger_config"]["test_task_group_size"]
        zipf_param_upper = self.config["data_config"]["zipf_param_upper"]
        zipf_param_lower = self.config["data_config"]["zipf_param_lower"]
        test_zipf_param_list = self.config["logger_config"]["test_zipf_param_list"]
        is_upper = self.config["data_config"]["is_upper"]
        is_lower = self.config["data_config"]["is_lower"]
        sl_upper = self.config["data_config"]["sl_upper"]
        sl_lower = self.config["data_config"]["sl_lower"]
        sample_IS = self.config["factory_config"]["sample_IS"]
        dataset_mem_dic = self.config["logger_config"]["dataset_mem_dic"]
        use_stream = self.config["logger_config"]["use_stream"]
        zipf_support_generator = ZipfGenerator(
            zipf_param_lower,
            zipf_param_upper,
            sl_lower,
            sl_upper,
            is_lower,
            is_upper,
            device,
            sample_IS,
        )
        # set zipf_decorate True due to zipf basic SketchCode
        task_generator = TaskGeneraotr(
            zipf_support_generator,
            device,
            test_task_group_size,
            test_zipf_param_list,
            dataset_mem_dic,
            use_stream,
        )
        return task_generator

    def init_logger(self, test_task_group_list, task_discribe_list):
        logger = Logger(
            test_task_group_list,
            task_discribe_list,
            self.config["data_config"]["task_comment"],
            self.config,
            self.config["logger_config"]["eval_gap"],
            self.config["logger_config"]["eval_metrics"],
            self.config["logger_config"]["use_stream"],
        )
        return logger

    def init_device(self):
        cuda_num = self.config["train_config"]["cuda_num"]
        if cuda_num == -1:
            device = torch.device("cpu")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def init_loss_func(self, loss_class):
        return eval(loss_class + "()")

    def init_optimizer(self, ratiosketch):
        lr = self.config["train_config"]["lr"]
        main_optimizer = torch.optim.Adam(ratiosketch.parameters(), lr=lr)
        return main_optimizer

    def init_ratiosketch(self, device):
        memory_matrix = self.init_base_sketch(
            self.config["factory_config"]["MemoryModule_class"],
            self.config["dim_config"]["dep_dim"],
            self.config["dim_config"]["slot_dim"],
            device,
        )
        decode_net = self.init_decode_module(
            self.config["factory_config"]["DecodeModule_class"]
        )
        loss_func = self.init_loss_func(self.config["factory_config"]["loss_class"])
        ratiosketch = RatioSketch(decode_net, memory_matrix, loss_func)
        return ratiosketch

    def init_base_sketch(self, memory_class, dep_dim, slot_dim, device):
        base_sketch = eval(memory_class + "(dep_dim, slot_dim, device)")
        return base_sketch

    def init_decode_module(
        self,
        decode_class,
    ):
        weight_decode_net = eval(decode_class + "()")
        return weight_decode_net
