import torch, random, numpy as np
from src.config import cfg

def set_seed(seed=cfg.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)