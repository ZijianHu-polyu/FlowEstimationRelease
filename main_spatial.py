import torch
import random
import argparse
import numpy as np
import datetime as dt
from spatial_inference.model_name import ModelName

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1)


cur_time = dt.datetime.now().isoformat()
config = {
    "data_filedir": "processed_data/urban_data/",
    "cache_filedir": "processed_data/urban_data/cache",
    "city": "Sacramento",
    "year": 2021,
    "begin_month": 7,
    "end_month": 8,
    "n_history": 12,
    "time_interval": 5,
    "zoom_level": 16,
    "pixels": 512,
    "savedir": "results/%s" % cur_time,
    "epoch": 30,
    "lr": 3e-4,
    "opt_opt": 1,
    "num_workers": 1,
    "device": "cuda:0",
    "subgraph_scale": 200,
    "model":{
        "embedding_dim": 256,
        "graph": {
            "graph_multi_head": 3,
            "graph_dropout": 0.2,
            "graph_alpha": 0.2,
        },
        "image": {
            "encoder_hidden_dim": 4,
            "encoder_output_dim": 8,
            "dense_input_dim": 48,
            "dense_hidden_dim": 96,
            "dense_output_dim": 192,
            "decoder_hidden_dim": 1024
        }
    }
}

parser = argparse.ArgumentParser(description='命令行中传入一个数字')
parser.add_argument('--epoch', type=int, default=10, help='epoch')
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
parser.add_argument('--embedding_dim', type=int, default=256, help='embedding')
parser.add_argument('--graph_multi_head', type=int, default=3, help='epsilon_start')
parser.add_argument('--graph_dropout', type=float, default=0.2, help='epsilon_end')
parser.add_argument('--graph_alpha', type=float, default=0.2, help='epsilon_decay')
parser.add_argument('--zoom_level', type=int, default=15, help='epsilon_decay')
parser.add_argument('--pixels', type=int, default=1024, help='epsilon_decay')
parser.add_argument('--num_workers', type=int, default=10, help='multiprocessing worker number')
parser.add_argument('--opt_opt', type=int, default=2, help='multiprocessing worker number')
parser.add_argument('--device', type=str, default="cuda:0", help='multiprocessing worker number')
parser.add_argument('--city', type=str, default="birmingham", help='estimation city')
parser.add_argument('--year', type=int, default=2017, help='data year')
parser.add_argument('--begin_month', type=int, default=1, help='data begin month')
parser.add_argument('--end_month', type=int, default=12, help='data end month')


args = parser.parse_args()

config["epoch"] = args.epoch
config["lr"] = args.lr
config["zoom_level"] = args.zoom_level
config["pixels"] = args.pixels
config["city"] = args.city
config["year"] = args.year
config["begin_month"] = args.begin_month
config["end_month"] = args.end_month
config["num_workers"] = args.num_workers
config["opt_opt"] = args.opt_opt
config["device"] = args.device
config["model"]["embedding_dim"] = args.embedding_dim
config["model"]["graph"]["graph_multi_head"] = args.graph_multi_head
config["model"]["graph"]["graph_dropout"] = args.graph_dropout
config["model"]["graph"]["graph_alpha"] = args.graph_alpha

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy('file_system')
    m = ModelName(config)
    m.run()
    # m.show()