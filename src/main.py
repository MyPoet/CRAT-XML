import sys
import random
import datetime
import argparse
from model import CRATXML

import torch
import numpy as np
from sklearn.model_selection import train_test_split

from dataset import createDataCSV, CRATXML
from log import Logger

def init_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def load_group(dataset, group_tree=0):
    if dataset == 'wiki500k':
        return np.load(f'./data/Wiki-500K/label_group{group_tree}.npy', allow_pickle=True)
    elif dataset == 'amazon670k':
        return np.load(f'./data/Amazon-670K/label_group{group_tree}.npy', allow_pickle=True)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=False, default=2022)
parser.add_argument('--batch', type=int, required=False, default=16)
parser.add_argument('--lr', type=float, required=False, default=0.0001)
parser.add_argument('--epoch', type=int, required=False, default=20)
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--max_len', type=int, required=False, default=512)
parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')

parser.add_argument('--valid', action='store_true')
parser.add_argument('--bert', type=str, required=False, default='bert-base')

# cluster arguments
parser.add_argument('--group_y_group', type=int, default=0) # 聚类编号
parser.add_argument('--group_y_candidate_num', type=int, required=False, default=3000)
parser.add_argument('--group_y_candidate_topk', type=int, required=False, default=10)


args = parser.parse_args()

if __name__ == '__main__':
    init_seed(args.seed)
    print(str(args.dataset+args.bert))
    LOG = Logger('log_'+str(args.dataset)+str(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')))

    print(f'load {args.dataset} dataset...')
    df, label_map = createDataCSV(args.dataset)
    if args.valid:
        train_df, valid_df = train_test_split(df[df['dataType'] == 'train'],
                                              test_size=0.1,
                                              random_state=2022)
        df.iloc[valid_df.index.values, 2] = 'valid'
        print('valid size', len(df[df['dataType'] == 'valid']))
    print(f'load {args.dataset} dataset with '
          f'{len(df[df.dataType =="train"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')

    if args.dataset in ['wiki500k', 'amazon670k']:
        group_y = load_group(args.dataset, args.group_y_group)
        _group_y = []
        for idx, labels in enumerate(group_y):
            _group_y.append([])
            for label in labels:
                _group_y[-1].append(label_map[label])
            _group_y[-1] = np.array(_group_y[-1])
        print("The length of label_map: ", len(label_map), "The length of label_map", len(_group_y))

        model = CRATXML()