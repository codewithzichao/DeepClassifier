# -*- coding:utf-8 -*-
'''

Author:
    Zichao Li,2843656167@qq.com

'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

class HAN(nn.Module):
    def __init__(self,
                 embedding_dim,
                 word_hidden_size,
                 seq_hidden_size,
                 dropout_rate,
                 num_class,
                 vocab_size=0,
                 seq_length=0,
                 rnn_type="lstm",
                 embedding_matrix=None,
                 requires_grads=False):
        '''
        initialization
        ⚠️In default,the way to initialize embedding is loading pretrained embedding look-up table!
        :param embedding_dim: embedding dim
        :param word_hidden_size: word hidden size
        :param seq_hidden_size: seq hidden size
        :param dropout_rate: dropout rate
        :param num_class: the number of label
        :param vocab_size: vocabulary size
        :param seq_length: sequence length
        :param rnn_type: rnn type,which must be lstm or gru.
        :param embedding_matrix: pretrained embedding lookup table,shape is [vocab_size,embedidng_dim]
        :param requires_grads: whether to update gradient of embedding in training stage
        '''
        super(HAN,self).__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.word_hidden_size = word_hidden_size
        self.seq_hidden_size=seq_hidden_size
        self.dropout_rate=dropout_rate
        self.num_class=num_class
        self.rnn_type = rnn_type
        self.embedding_matrix = embedding_matrix
        self.requires_grads=requires_grads

        if self.embedding_matrix is None:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.embedding_dim,
                                          padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=self.requires_grads)
            self.vocab_size = self.embedding_matrix.shape[0]

        if self.rnn_type == "lstm":
            self.word_rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.word_hidden_size,
                                    batch_first=True, num_layers=1, bidirectional=True)
            self.seq_rnn = nn.LSTM(input_size=self.word_hidden_size * 2, hidden_size=self.seq_hidden_size,
                                   batch_first=True, num_layers=1, bidirectional=True)
        elif self.rnn_type == "gru":
            self.word_rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.word_hidden_size,
                                   batch_first=True, num_layers=1, bidirectional=True)
            self.seq_rnn = nn.GRU(input_size=self.word_hidden_size * 2, hidden_size=self.seq_hidden_size,
                                  batch_first=True, num_layers=1, bidirectional=True)
        else:
            raise Exception("wrong rnn type,must be one of [lstm,gru].")

        self.fc1 = nn.Linear(in_features=self.word_hidden_size * 2, out_features=self.word_hidden_size * 2)
        self.U_w = nn.Parameter(torch.Tensor(self.word_hidden_size * 2, self.word_hidden_size * 2))
        self.fc2 = nn.Linear(in_features=self.word_hidden_size * 2, out_features=self.word_hidden_size * 2)
        self.U_s = nn.Parameter(torch.Tensor(self.word_hidden_size * 2, self.word_hidden_size * 2))

        self.dropout=nn.Dropout(p=self.dropout_rate)
        self.classifer = nn.Linear(in_features=self.word_hidden_size * 2, out_features=self.num_class)

    def forward(self,inputs):

        word_length=inputs.size()[-1]
        seq_length=inputs.size()[1]
        inputs=inputs.view(-1,word_length)
        x=self.embedding(inputs)
        x, _ = self.word_rnn(x)

        # char attention
        temp = torch.tanh(self.fc1(x))
        char_score = torch.matmul(temp, self.U_w)
        char_weights = F.softmax(char_score, dim=1)
        x = torch.mul(char_weights, x)
        x = torch.sum(x, dim=1)

        x = x.view(-1, seq_length, x.shape[-1])
        x, _ = self.seq_rnn(x)

        # word attention
        temp = torch.tanh(x)
        word_score = torch.matmul(temp, self.U_s)
        word_weights = F.softmax(word_score, dim=1)
        x = torch.mul(word_weights, x)
        x = torch.sum(x, dim=1)

        x=self.dropout(x)
        outputs = self.classifer(x)

        return outputs




