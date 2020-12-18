import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import *

class BertHAN(nn.Module):
    def __init__(self,
                 embedding_dim,
                 word_hidden_size,
                 seq_hidden_size,
                 dropout_rate,
                 num_class,
                 bert_path,
                 rnn_type="lstm",
                 requires_grads=False):
        '''
        initialization
        :param embedding_dim: embedding dim
        :param word_hidden_size: word hidden size
        :param seq_hidden_size: sequence hidden size
        :param dropout_rate: dropout rate
        :param num_class: the number of label
        :param bert_path: bert path
        :param rnn_type: rnn type. default:lstm
        :param requires_grads: whether to update gradient of Bert in training
        '''
        super(BertHAN,self).__init__()
        self.embedding_dim = embedding_dim
        self.word_hidden_size = word_hidden_size
        self.seq_hidden_size = seq_hidden_size
        self.rnn_type = rnn_type
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.bert_path=bert_path
        self.requires_grads=requires_grads

        self.bert=BertModel.from_pretrained(self.bert_path)
        if self.requires_grads is False:
            for p in self.bert.parameters():
                p.requires_grads=False

        if self.rnn_type == "lstm":
            self.word_rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.word_hidden_size,
                                    batch_first=True, num_layers=1, bidirectional=True)
            self.seq_rnn = nn.LSTM(input_size=self.word_hidden_size * 2, hidden_size=self.seq_hidden_size,
                                   batch_first=True, num_layers=1, bidirectional=True)
        elif self.rnn_type=="gru":
            self.word_rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.word_hidden_size,
                                   batch_first=True, num_layers=1, bidirectional=True)
            self.seq_rnn = nn.GRU(input_size=self.word_hidden_size * 2, hidden_size=self.seq_hidden_size,
                                  batch_first=True, num_layers=1, bidirectional=True)
        else:
            raise Exception("wrong rnn type,must be one of [lstm,gru].")

        self.fc1=nn.Linear(in_features=self.word_hidden_size * 2, out_features=self.word_hidden_size * 2)
        self.U_w=nn.Parameter(torch.Tensor(self.word_hidden_size * 2, self.word_hidden_size * 2))
        self.fc2=nn.Linear(in_features=self.word_hidden_size * 2, out_features=self.word_hidden_size * 2)
        self.U_s=nn.Parameter(torch.Tensor(self.word_hidden_size*2,self.word_hidden_size*2))

        self.dropout=nn.Dropout(p=self.dropout_rate)
        self.classifer=nn.Linear(in_features=self.word_hidden_size*2,out_features=self.num_class)

    def forward(self, input_ids, attention_mask=None):
        '''
        forard propagation
        :param params: input_ids:[batch_size,max_seq_length,max_word_length]
        :param params: attention_mask:[batch_size,max_seq_length,max_word_length]
        :return: logits:[batch_size,num_class]
        '''

        seq_length=input_ids.size()[1]
        word_length=input_ids.size()[2]
        input_ids=input_ids.view(-1,word_length)
        if attention_mask is not None:
            attention_mask=attention_mask.view(-1,word_length)

        # bert encoding
        bert_output=self.bert(input_ids=input_ids,attention_mask=attention_mask)
        x=bert_output.last_hidden_state
        x,_=self.word_rnn(x)

        # char attention
        temp=torch.tanh(self.fc1(x))
        char_score=torch.matmul(temp,self.U_w)
        char_weights=F.softmax(char_score,dim=1)
        x=torch.mul(char_weights,x)
        x=torch.sum(x,dim=1)

        x=x.view(-1,seq_length,x.shape[-1])
        x,_=self.seq_rnn(x)

        # word attention
        temp=torch.tanh(x)
        word_score=torch.matmul(temp,self.U_s)
        word_weights=F.softmax(word_score,dim=1)
        x=torch.mul(word_weights,x)
        x=torch.sum(x,dim=1)

        x=self.dropout(x)
        outputs=self.classifer(x)

        return outputs
