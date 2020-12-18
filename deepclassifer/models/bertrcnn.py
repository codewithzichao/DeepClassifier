# -*- coding:utf-8 -*-
'''

Author:
    Zichao Li,2843656167@qq.com

'''
from __future__ import print_function

import torch
import torch.nn as nn
from transformers import *
import numpy as np

class BertRCNN(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_size,
                 dropout_rate,
                 num_class,
                 bert_path,
                 rnn_type="lstm",
                 num_layers=1,
                 requires_grads=False):
        '''
        initialization
        :param embedding_dim:embedding dim
        :param hidden_size: rnn hidden size
        :param dropout_rate: dropout rate
        :param num_class: the number of label
        :param bert_path: bert path
        :param rnn_type: rnn type. Default:lstm
        :param num_layers: the number of rnn layer
        :param requires_grads: whether to update gradient of Bert in training stage
        '''
        super(BertRCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.bert_path=bert_path
        self.requires_grads=requires_grads

        self.bert = AutoModel.from_pretrained(self.bert_path)
        if self.requires_grads is False:
            for p in self.bert.parameters():
                p.requires_grads = False

        if self.rnn_type == "lstm":
            self.birnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                                 num_layers=self.num_layers, batch_first=True, bidirectional=True)
        elif self.rnn_type == "gru":
            self.birnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, batch_first=True, bidirectional=True)
        else:
            raise ValueError("rnn type must be one of {lstm,gru}.")

        self.W = nn.Linear(in_features=self.embedding_dim + self.hidden_size * 2 * self.num_layers,
                           out_features=self.hidden_size * 2)

        self.global_max_pool1d = nn.AdaptiveMaxPool1d(output_size=1)
        self.classifier = nn.Linear(in_features=self.hidden_size * 2, out_features=self.num_class)

    def forward(self, input_ids, attention_mask=None):
        '''
        forard propagation
        :param params: input_ids:[batch_size,max_length]
        :param params: attention_mask:[batch_size,max_length]
        :return: logits:[batch_size,num_class]
        '''

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_output.last_hidden_state

        rnn_output, _ = self.birnn(x)
        x = torch.cat((x, rnn_output), dim=-1)
        x = torch.tanh(self.W(x))
        x = x.permute(0, 2, 1)
        x = self.global_max_pool1d(x).squeeze(dim=-1)
        outputs = self.classifier(x)

        return outputs
