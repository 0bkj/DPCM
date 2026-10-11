import torch
import time
import os
import yaml
import numpy as np
import random
import logging
import argparse

logger = logging.getLogger(__name__)

def set_seed(seed):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def truncate(s: str, width: int) -> str:
    """将字符串截断为指定宽度，超长则末尾加 '...'"""
    if len(s) <= width:
        return s
    if width < 4:
        return s[:width]  # 容不下 '...'，直接截断
    return s[:width - 3] + '...'

def print_args(args):
    col_width = 30
    total_width = col_width * 2 + 3  # │field│value│ → 30 + 1 + 30 + 2 边框 = 63

    logger.info('┌' + '─' * (total_width - 2) + '┐')
    
    for arg in vars(args):
        field = truncate(str(arg), col_width)
        value = truncate(str(getattr(args, arg)), col_width)
        logger.info(f"│{field:<{col_width}}│{value:<{col_width}}│")
    
    logger.info('└' + '─' * (total_width - 2) + '┘')

def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(
    model,
    results,
    hyperparameters,
    timestamp,
    *,
    epoch=None,
    global_step=None,
    seed=None,
):
    SAVE_MODEL_PATH = os.getcwd() + '/results'
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters,
        'epoch': epoch,
        'global_step': global_step,
        'seed': seed,
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/' + timestamp + '.pth')


def load_config(path):
    """Read a YAML configuration file and return its contents as a dict.

    If the file does not exist or cannot be parsed an empty dict is returned.
    """
    if not path:
        logger.info('No config file provided.')
        return {}
    try:
        logger.info(f'Loading config from {path}')
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f'Failed to load configuration from {path}') from exc

def compute_metrics(pred, labels, slidingWindow=100, metrics="auc"):
    from vus.metrics import get_metrics
    results = {}
    if isinstance(metrics, str):
        metric_names = metrics.replace(",", " ").replace("+", " ").split()
    else:
        metric_names = list(metrics)
    metric_names = [name.lower() for name in metric_names]
    if "all" in metric_names or "both" in metric_names:
        metric_names = ["auc", "vus"]
    unknown = set(metric_names) - {"auc", "vus"}
    if unknown:
        raise ValueError(f"Unsupported metrics: {sorted(unknown)}")
    if not metric_names:
        raise ValueError("At least one metric must be selected.")
    if "auc" in metric_names:
        results.update(get_metrics(pred, labels, metric="auc"))
    if "vus" in metric_names:
        results.update(
            get_metrics(pred, labels, metric="vus", slidingWindow=slidingWindow)
        )
    return results


def evaluate(pred,labels,slidingWindow=100,metrics="auc"):
    results = compute_metrics(pred, labels, slidingWindow, metrics)
    # for metric in results.keys():
    #     logger.info(f"{metric} : {results[metric]}")
    return results
        
        
def get_anomaly_segments(labels_array, anomaly_val=1):
    """
    将二值标签数组转换为连续异常区间的列表 [(start, end), ...]
    例如: [0, 1, 1, 0, 1] -> [(1, 3), (4, 5)] (注意:end是开区间，对应切片逻辑)
    """
    segments = []
    in_anomaly = False
    start_idx = 0
    
    # 在末尾补一个0，确保最后一段异常能被正确闭合
    extended_labels = np.append(labels_array, 0) 
    
    for t, val in enumerate(extended_labels):
        if val == anomaly_val and not in_anomaly:
            # 异常开始
            in_anomaly = True
            start_idx = t
        elif val != anomaly_val and in_anomaly:
            # 异常结束
            in_anomaly = False
            segments.append((start_idx, t))
            
    return segments


def build_memory_index(val_vectors, use_gpu=False):
    """
    val_vectors: np.ndarray, shape (N_val, c), dtype 可以是整数
    use_gpu: 是否放到 GPU 上
    返回 faiss index
    """
    import faiss

    # FAISS 的 L1 距离需要 float32
    vectors = val_vectors.astype(np.float32)
    dim = vectors.shape[1]
    
    # 使用 L1 距离 (metric=1 即曼哈顿距离，等价于汉明距离)
    index = faiss.IndexFlat(dim, faiss.METRIC_L1)  # 或直接 IndexFlatL1(dim)
    
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.add(vectors)
    return index


def write_metric(f,model,seed,eval_results, write_head=False):
    if write_head:
        f.write("seed,model")
        for metric in eval_results.keys():
            f.write(f",{metric}")
        f.write("\n")
    f.write(f"{seed},{model}")
    for value in eval_results.values():
        f.write(f",{value:.3f}")
    f.write("\n")

def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def dic_to_namespace(config_values):
    assert isinstance(config_values, dict), "Expected config_values to be a dict"
    config_values = argparse.Namespace(**config_values)
    return config_values