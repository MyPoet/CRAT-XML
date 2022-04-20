import time
from tokenize import group

import tqdm
import numpy as np
from apex import amp
from torch import nn
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

from tokenizers import BertWordPieceTokenizer
from transformers import RobertaTokenizerFast

def get_bert(bert_name):
    if 'roberta' in bert_name:
        print('load roberta-base')
        model_config = RobertaConfig.from_pretrained('roberta-base')
        model_config.output_hidden_states = True
        bert = RobertaModel.from_pretrained('roberta-base', config=model_config)
    elif 'xlnet' in bert_name:
        print('load xlnet-base-cased')
        model_config = XLNetConfig.from_pretrained('xlnet-base-cased')
        model_config.output_hidden_states = True
        bert = XLNetModel.from_pretrained('xlnet-base-cased', config=model_config)
    else:
        print('load bert-base-uncased')
        model_config = BertConfig.from_pretrained('bert-base-uncased')
        model_config.output_hidden_states = True
        bert = BertModel.from_pretrained('bert-base-uncased', config=model_config)
    return bert

class CRATXML(nn.Module):
    def __init__(self, num_labels, group_y=None, bert='bert-base', dropout=0.5, candidates_topk=10, hidden_dim=300, device_id=0):
        super(CRATXML, self).__init__()
        print("CRATXML")

        self.candidates_topk = candidates_topk
        self.bert_name, self.bert = bert, get_bert(bert)
        self.drop_out = nn.Dropout(dropout)

        self.group_y = group_y
        if self.group_y is not None:
            self.group_y_labels = group_y.shape[0]
            print('hidden dim:',  hidden_dim)
            print('label group numbers:',  self.group_y_labels)

            self.l0 = nn.Linear(self.bert.config.hidden_size, self.group_y_labels)
            # hidden bottle layer
            self.l1 = nn.Linear(self.bert.config.hidden_size, hidden_dim)
            self.embed = nn.Embedding(num_labels, hidden_dim)
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            self.l0 = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.device_id = device_id

    def get_fast_tokenizer(self):
        if 'roberta' in self.bert_name:
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case=True)
        elif 'xlnet' in self.bert_name:
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased') 
        else:
            tokenizer = BertWordPieceTokenizer(
                "data/.bert-base-uncased-vocab.txt",
                lowercase=True)
        return tokenizer

    def get_tokenizer(self):
        if 'roberta' in self.bert_name:
            print('load roberta-base tokenizer')
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        elif 'xlnet' in self.bert_name:
            print('load xlnet-base-cased tokenizer')
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        else:
            print('load bert-base-uncased tokenizer')
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        return tokenizer


    # train_loss_1 表示本体句子的loss
    # train_loss_2 表示对抗句子的loss
    # cl_loss 表示对比训练的loss
    def one_epoch(self, epoch, dataloader, optimizer, mode='train', eval_loader=None, eval_step=20000, log=None):
        bar = tqdm.tqdm(total=len(dataloader))
        p1, p3, p5 = 0, 0, 0
        g_p1, g_p3, g_p5 = 0, 0, 0
        total, acc1, acc3, acc5 = 0, 0, 0, 0
        g_acc1, g_acc3, g_acc5 = 0, 0, 0
        train_loss_1, train_loss_2, cl_loss = 0, 0, 0

        if mode == 'train':
            self.train()
        else:
            self.eval()
        
        pred_scores, pred_labels = [], []
        bar.set_description(f'{mode}-{epoch}')

        with torch.set_grad_enabled(mode == 'train'):
            for step, data in enumerate(dataloader):
                batch = tuple(t for i in data)
                inputs = {'input_ids': batch[0].to(self.device_id),
                            'attention_mask': batch[1].to(self.device_id),
                            'token_type_ids': batch[2].to(self.device_id)}
                if mode == 'train':
                    inputs['labels'] = batch[3].to(self.device_id)
                    if self.group_y is not None:
                        inputs['group_labels'] = batch[4].to(self.device_id)
                        inputs['candidates'] = batch[5].to(self.device_id)

                