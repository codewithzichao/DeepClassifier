import torch
import torch.nn as nn
from transformers import *

class BertTextCNN(nn.Module):
    def __init__(self,
                 embedding_dim,
                 dropout_rate,
                 num_class,
                 bert_path,
                 num_layers=3,
                 kernel_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 strides=[1, 1, 1],
                 paddings=[0, 0, 0],
                 requires_grads=False):
        '''
        initialization
        ⚠⚠️In default,the way to initialize embedding is loading pretrained embedding look-up table!
        :param dropout_rate: dropout rate
        :param num_class: the number of label
        :param bert_path: bert config path
        :param embedding_dim: embedding dim
        :param num_layers: the number of cnn layer
        :param kernel_sizes: list of conv kernel size
        :param num_filters: list of conv filters
        :param strides: list of conv strides
        :param paddings: list of conv padding
        :param requires_grads: whther to update gradient of Bert in training stage
        '''
        super(BertTextCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.strides = strides
        self.paddings = paddings
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.bert_path=bert_path
        self.requires_grads=requires_grads

        self.bert = AutoModel.from_pretrained(self.bert_path)
        if self.requires_grads is False:
            for p in self.bert.parameters():
                p.requires_grads = False

        if self.num_layers != len(self.kernel_sizes) or self.num_layers != len(self.num_filters):
            raise Exception("The number of num_layers and num_filters must be equal to the number of kernel_sizes!")

        self.conv1ds = []
        self.global_max_pool1ds = []
        final_hidden_size = sum(self.num_filters)
        for i in range(self.num_layers):
            conv1d = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_filters[i],
                               kernel_size=self.kernel_sizes[i],
                               stride=self.strides[i], padding=self.paddings[i])
            global_max_pool1d = nn.AdaptiveMaxPool1d(output_size=1)
            self.conv1ds.append(conv1d)
            self.global_max_pool1ds.append(global_max_pool1d)

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.classifier = nn.Linear(in_features=final_hidden_size, out_features=self.num_class)

    def forward(self, input_ids, attention_mask=None):
        '''
        forard propagation
        :param params: input_ids:[batch_size,max_length]
        :param params: attention_mask:[batch_size,max_length]
        :return: logits:[batch_size,num_class]
        '''

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_output.last_hidden_state

        x = x.permute(0, 2, 1)
        cnn_pool_result = []
        for i in range(self.num_layers):
            temp = torch.relu(self.conv1ds[i](x))
            temp = self.global_max_pool1ds[i](temp).squeeze(dim=-1)
            cnn_pool_result.append(temp)

        x = torch.cat(cnn_pool_result, dim=-1)
        x = self.dropout(x)
        outputs = self.classifier(x)

        return outputs


