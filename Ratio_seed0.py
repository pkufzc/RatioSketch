import numpy as np
import os

os.sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from SourceCode.Factory import Factory


def init_config():
    data_config = {
        "task_comment": "10kb",
        "sl_lower": 1,
        "sl_upper": 20,
        "is_lower": 10000,
        "is_upper": 150000,
        "zipf_param_lower": 0.1,
        "zipf_param_upper": 2.5,
    }
    dim_config = {
        "dep_dim": 3,
        "slot_dim": 64,
    }
    train_config = {
        "train_step": 5010,
        "lr": 0.001,
        "cuda_num": 0,
    }
    logger_config = {
        "eval_gap": 1,
        "dataset_mem_dic": {
            # "dc": list(map(int, np.linspace(3000, 8000, 6))),
            "aol": list(map(int, np.linspace(500, 3000, 6))),
            # "caida": list(map(int, np.linspace(5000, 30000, 6))),
            # "mawi": list(map(int, np.linspace(1000, 3500, 6))),
            # "criteo": list(map(int, np.linspace(3000, 8000, 6))),
            # "webdocs": list(map(int, np.linspace(300, 800, 6))),
        },
        # Only one model can be trained per run. If you select "CMS+RS" below,
        # do not also select "Tower+RS" in the same run, which will cause a conflict.
        "eval_metrics": [
            # ("Lego", "#D7263D", ["cuda"]),  # 深红
            # ("Meta", "#1B98E0", ["cuda"]),  # 高亮蓝
            ("CMS", "#2E294E", ["cuda"]),  # 深紫蓝
            ("CMS+RS", "#F4D35E", ["cuda"]),  # 明亮黄
            # ("Tower", "#06D6A0", ["cuda"]),  # 青绿
            # ("Tower+RS", "#FF7F11", ["cuda"]),  # 高亮橙
        ],
    }
    factory_config = {
        "loss_class": "RatioLoss",
        # CMS for using CMS+RS, TCMS for using Tower+RS
        "MemoryModule_class": "CMS",
        "sample_IS": "linear",
        "DecodeModule_class": "RatioDecoder",
    }
    config = {
        "train_config": train_config,
        "factory_config": factory_config,
        "dim_config": dim_config,
        "data_config": data_config,
        "logger_config": logger_config,
    }
    return config


if __name__ == "__main__":
    config = init_config()
    train_config = config["train_config"]
    factory = Factory(config)
    trainer = factory.init_trainer()
    trainer.train(train_config["train_step"])
