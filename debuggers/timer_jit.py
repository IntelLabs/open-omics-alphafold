import torch
import time


@torch.jit.ignore
def read_time():
  return time.time()