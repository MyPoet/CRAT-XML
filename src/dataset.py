import tqdm
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def createDataCSV(dataset):
    labels = []
    texts = []
    dataType = []
    label_map = {} # store all label set

    name_map = {'wiki31k': 'Wiki10-31K',
                'wiki500k': 'Wiki-500K',
                'amazoncat13k': 'AmazonCat-13K',
                'amazon670k': 'Amazon-670K',
                'eurlex4k': 'Eurlex-4K'}

    assert dataset in name_map
    dataset = name_map[dataset]

    fext = '_texts.txt' if dataset == 'Eurlex-4K' else '_raw_texts.txt'
    with open(f'./data/{dataset}/train{fext}') as f:
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('train')

    with open(f'./data/{dataset}/test{fext}') as f:
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('test')

    with open(f'./data/{dataset}/train_labels.txt') as f:
        for i in tqdm.tqdm(f):
            for l in i.replace('\n', '').split():
                label_map[l] = 0
            labels.append(i.replace('\n', ''))

    with open(f'./data/{dataset}/test_labels.txt') as f:
        for i in tqdm.tqdm(f):
            for l in i.replace('\n', '').split():
                label_map[l] = 0
            labels.append(i.replace('\n', ''))

    assert len(texts) == len(labels) == len(dataType)

    df_row = {'text': texts, 'label': labels, 'dataType': dataType}

    for i, k in enumerate(sorted(label_map.keys())):
        label_map[k] = i
    df = pd.DataFrame(df_row)

    print('The length of label map is', len(label_map))

    return df, label_map


class CRATDataset(Dataset):
    def __init__(self, df, mode, tokenizer, label_map, max_length,
                 token_type_ids=None, group_y=None, candidates_num=None):
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.df, self.num_labels, self.label_map = df[df.dataType == self.mode], len(label_map), label_map
        self.len = len(self.df)
        self.tokenizer, self.max_length, self.group_y = tokenizer, max_length, group_y
        self.token_type_ids = token_type_ids
        self.candidates_num = candidates_num

        if group_y is not None:
            self.candidates_num, self.group_y, self.n_group_y_labels = candidates_num, [], group_y.shape[0]
            self.map_group_y = np.empty(len(label_map), dtype=np.long)
            for idx, labels in enumerate(group_y):
                self.group_y.append([])
                for label in labels:
                    self.group_y[-1].append(label_map[label])
                self.map_group_y[self.group_y[-1]] = idx
                self.group_y[-1]  = np.array(self.group_y[-1])
            self.group_y = np.array(self.group_y)

    def __len__(self):
        return self.len 
    