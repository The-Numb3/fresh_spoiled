# meatscan/utils.py
import os, yaml, random, numpy as np, torch

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # 성능 우선

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def optimize_threshold(y_true, y_prob, metric="f1"):
    # 0..1 구간에서 간단 스윕
    import numpy as np
    from .metrics import compute_metrics
    best_th, best_val = 0.5, -1.0
    for th in np.linspace(0.05, 0.95, 19):
        m = compute_metrics(y_true, y_prob, threshold=th)
        val = m.get(metric, 0.0)
        if val > best_val:
            best_val, best_th = val, th
    return float(best_th), float(best_val)
