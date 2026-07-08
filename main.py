import os
# 在导入 numpy, sklearn 之前设置
os.environ['OPENBLAS_NUM_THREADS'] = '64'  # 限制为 64 个线程，安全且高效
os.environ['OMP_NUM_THREADS'] = '64'       # 防止 OpenMP 也产生冲突
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第一块 GPU
import argparse
import utils
import shutil
import logging
from logging.handlers import RotatingFileHandler

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

    """
    Hyperparameters
    """
    timestamp = utils.readable_timestamp()

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    
    parser.add_argument("--time_recon", type=dict, default={})
    parser.add_argument("--freq_recon", type=dict, default={})
    parser.add_argument("--normal_flow", type=dict, default={})

    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--dataset",  type=str)

    # whether or not to save model
    parser.add_argument("-save", action="store_true")
    parser.add_argument("--filename",  type=str, default=timestamp)
    parser.add_argument("--visual_channel", type=list, default=None, help="Channel to visualize; if not specified, will select [0,1,2]")

    initial_args, _ = parser.parse_known_args()

    config_values = {}
    if initial_args.config:
        config_values = utils.load_config(initial_args.config)
        config_values = dic_to_namespace(config_values)  # convert dict to Namespace for set_defaults

    # del config_values.config
    # config_values.pop('config', None)

    # apply config as defaults; CLI arguments will override when we parse again
    parser.set_defaults(**config_values)

    # final argument parsing with config defaults applied
    args = parser.parse_args()

    # if filename wasn't provided via CLI or config (empty string), use timestamp
    if not args.filename:
        args.filename = timestamp

    

    if args.save:
        logger.info('Results will be saved in ./results/' + args.filename + '.pth')
    
    utils.print_args(args)
    return args

def dic_to_namespace(config_values):
    assert isinstance(config_values, dict), "Expected config_values to be a dict"
    for key in config_values.keys():
        if config_values[key] is not None and type(config_values[key]) is dict:
            config_values[key] = argparse.Namespace(**config_values[key])
    
    return config_values



if __name__ == "__main__":
    from solver import Solver

    utils.set_seed(0)  
    args = init()
    shutil.rmtree("./data/indices", ignore_errors=True)
    shutil.rmtree(f"./visual/{args.filename}", ignore_errors=True)
    solver = Solver(args)
    args.mode = args.mode.split()
    if "train" in args.mode:
        solver.train()
    if "train_nf" in args.mode:
        solver.train_nf()
    if "test_nf" in args.mode:
        solver.test_nf()
    if "test_channel" in args.mode:
        solver.test_channel()
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