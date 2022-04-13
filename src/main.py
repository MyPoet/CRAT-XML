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
parser.add_argument('--batch', type=int, required=False, default=16)
parser.add_argument('--lr', type=float, required=False, default=0.0001)
parser.add_argument('--epoch', type=int, required=False, default=20)
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--max_len', type=int, required=False, default=512)
parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')

parser.add_argument('--bert', type=str, required=False, default='bert-base')

args = parser.parse_args()

if __name__ == '__main__':
    init_seed(args.seed)
    print(str(args.dataset, args.bert))




    