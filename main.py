import os
# 在导入 numpy, sklearn 之前设置
os.environ['OPENBLAS_NUM_THREADS'] = '64'  # 限制为 64 个线程，安全且高效
os.environ['OMP_NUM_THREADS'] = '64'       # 防止 OpenMP 也产生冲突
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
import argparse
import utils
import shutil
import logging
from logging.handlers import RotatingFileHandler
import json

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[
        RotatingFileHandler(
            "zzzz.log",
            maxBytes=100_000,
            backupCount=1,
            encoding='utf-8'
        ),
        logging.StreamHandler()  # 同时输出到控制台
    ],
)

logger = logging.getLogger(__name__)

def init(default_config="config/visual/swat.yaml"):
    parser = argparse.ArgumentParser()

    # allow specifying a YAML configuration file; defaults to config/config.yaml
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="path to YAML config file (values overridden by CLI arguments)",
    )
    initial_args, _ = parser.parse_known_args()
    """
    Hyperparameters
    """
    timestamp = utils.readable_timestamp()

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    
    parser.add_argument("--time_recon", type=json.loads, default=None)
    parser.add_argument("--freq_recon", type=json.loads, default=None)
    parser.add_argument("--normal_flow", type=dict, default=None)

    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--dataset",  type=str)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--metrics",
        type=str,
        default="auc",
        help="Metrics to report: auc, vus, or a combination such as 'auc vus'.",
    )
    parser.add_argument("--time_ngram_order", type=int, default=3)
    parser.add_argument("--var_ngram_order", type=int, default=2)
    parser.add_argument("--report_variant", type=str, default="minmax_vcpm ecdf_vcpm minmax_tree ecdf_tree")
    parser.add_argument("--ablation", action="store_true", dest="ablation", default=False)
    parser.add_argument("--no_use_freq", action="store_false", dest="use_freq", default=True)
    parser.add_argument("--no_use_temp", action="store_false", dest="use_temp", default=True)
    parser.add_argument(
        "--frequency_representation",
        choices=["amplitude", "real_imag"],
        default="amplitude",
    )

    # whether or not to save model
    parser.add_argument("-save", action="store_true")
    parser.add_argument("--filename",  type=str, default=timestamp)
    parser.add_argument("--visual_channel", type=list, default=None, help="Channel to visualize; if not specified, will select [0,1,2]")

    

    config_values = {}
    DICT_KEYS = ("time_recon", "freq_recon") 
    if initial_args.config:
        config_values = utils.load_config(initial_args.config)
        # config_values = dic_to_namespace(config_values)  # convert dict to Namespace for set_defaults
    config_dicts = {k: config_values.get(k, None) for k in DICT_KEYS}
    parser.set_defaults(**config_values)
    args = parser.parse_args()

    for key in DICT_KEYS:
        base = config_dicts.get(key) or {}
        cli_val = getattr(args, key, None) or {}
        setattr(args, key, utils.dic_to_namespace(utils.deep_merge(base, cli_val)))
        print(getattr(args, key, None))

    if not args.filename:
        args.filename = timestamp
    if args.save:
        logger.info('Results will be saved in ./results/' + args.filename + '.pth')
    
    utils.print_args(args)
    return args

if __name__ == "__main__":
    from solver import Solver

    args = init()

    shutil.rmtree("./data/indices", ignore_errors=True)
    shutil.rmtree(f"./visual/{args.filename}", ignore_errors=True)
    args.mode = args.mode.split()
    for seed in args.seeds:
        utils.set_seed(seed)
        args.seed = seed
        solver = Solver(args)
        if "train" in args.mode:
            if not solver.load_model():
                solver.train()
        if "test" in args.mode:
            solver.test()
        if "visual" in args.mode:
            solver.visual_train()
            solver.visual_test()
        if "visual_proto" in args.mode:
            solver.draw_indices()

        if "visual_anomaly" in args.mode:
            solver.draw_anomaly()
    print("finish")
