import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import *
import numpy as np

class BertDPCNN(nn.Module):
    def __init__(self,
                 embedding_dim,
                 dropout_rate,
                 num_class,
                 bert_path,
                 num_blocks=3,
                 kernel_sizes=3,
                 num_filters=250,
                 requires_grads=False):
        '''
        initialization
        :param embedding_dim: embedding dim
        :param dropout_rate: dropout rate
        :param num_class: the number of label
        :param bert_path: bert path
        :param num_blocks: the number of block ,default:3
        :param kernel_sizes: kernel size
        :param num_filters: the number of filter
        :param requires_grads: whether to update gradient of Bert in training
        '''
        super(BertDPCNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.bert_path=bert_path
        self.requires_grads=requires_grads

        self.bert = BertModel.from_pretrained(self.bert_path)
        if self.requires_grads is False:
           for p in self.bert.parameters():
               p.requires_grads=False

        # text region embedding
        self.region_embedding = nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                                          stride=1, kernel_size=(self.kernel_sizes, self.embedding_dim))

        # two conv
        self.conv2d1 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters,
                                 stride=2, kernel_size=(self.kernel_sizes, 1), padding=0)
        self.conv2d2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters,
                                 stride=2, kernel_size=(self.kernel_sizes, 1), padding=0)
        self.padding1 = nn.ZeroPad2d((0, 0,(self.kernel_sizes-1)//2, (self.kernel_sizes-1)-((self.kernel_sizes-1)//2)))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, self.kernel_sizes-2))  # bottom

        # one block
        self.block_max_pool = nn.MaxPool2d(kernel_size=(self.kernel_sizes, 1), stride=2)
        self.conv2d3 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters,
                                 stride=1, kernel_size=(self.kernel_sizes, 1), padding=0)
        self.conv2d4 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters,
                                 stride=1, kernel_size=(self.kernel_sizes, 1), padding=0)

        # final pool and softmax

        self.flatten = nn.Flatten()
        self.dropout=nn.Dropout(p=self.dropout_rate)
        self.classifier = nn.Linear(in_features=self.num_filters, out_features=self.num_class)

    def forward(self, input_ids, attention_mask=None):
        '''
        forard propagation
        :param params: input_ids:[batch_size,max_length]
        :param params: attention_mask:[batch_size,max_length]
        :return: logits:[batch_size,num_class]
        '''

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embedding = bert_output.last_hidden_state.unsqueeze(dim=1)

        x = self.region_embedding(embedding)

        x = self.padding1(x)
        x = torch.relu(self.conv2d1(x))
        x = self.padding1(x)
        x = torch.relu(self.conv2d2(x))
        for i in range(self.num_blocks):
            x = self._block(x)

        x = self.flatten(x)
        x=self.dropout(x)
        outputs = self.classifier(x)

        return outputs

    def _block(self, x):

        x = self.padding2(x)
        pool_x = self.block_max_pool(x)

        x = self.padding1(pool_x)
        x = F.relu(self.conv2d3(x))
        x = self.padding1(x)
        x = F.relu(self.conv2d4(x))

        return x + pool_x
