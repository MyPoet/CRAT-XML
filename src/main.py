import sys
import random
import argparse

import torch
import numpy as np



def init_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=False, default=2022)
args = parser.parse_args()

if __name__ == '__main__':
    init_seed(args.seed)