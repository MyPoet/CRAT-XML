import tqdm
import pandas as pd
import numpy as np

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