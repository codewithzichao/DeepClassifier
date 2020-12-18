import torch
import torch.nn as nn
import numpy as np

class TextCNN(nn.Module):
    def __init__(self,
                 embedding_dim,
                 dropout_rate,
                 num_class,
                 vocab_size=0,
                 seq_length=0,
                 num_layers=3,
                 kernel_sizes=[3, 4, 5],
                 strides=[1, 1, 1],
                 paddings=[0, 0, 0],
                 num_filters=[100, 100, 100],
                 embedding_matrix=None,
                 requires_grads=False):
        '''
        initialization
        ⚠️In default,the way to initialize embedding is loading pretrained embedding look-up table!
        :param embedding_dim: embedding dim
        :param dropout_rate: drouput rate
        :param num_class: the number of label
        :param vocab_size: vocabulary size
        :param seq_length: max length of sequence after padding
        :param num_layers: the number of cnn
        :param kernel_sizes: list of conv kernel size
        :param strides: list of conv strides
        :param paddings: list of padding
        :param num_filters: list of num filters
        :param embedding_matrix: pretrained embedding look-up table,shape is:[vocab_size,embedding_dim]
        :param requires_grads: whether to update gradient of embedding in training
        '''
        super(TextCNN, self).__init__()

        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.embedding_matrix = embedding_matrix
        self.requires_grads = requires_grads

        if self.num_layers != len(self.kernel_sizes) or self.num_layers != len(self.num_filters):
            raise ValueError("The number of num_layers and num_filters must be equal to the number of kernel_sizes!")

        # embedding
        if self.embedding_matrix is None:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.embedding_dim,
                                          padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=self.requires_grads)
            self.vocab_size = self.embedding_matrix.shape[0]

        # conv layers
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

        # dropout
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.classifier = nn.Linear(in_features=final_hidden_size, out_features=self.num_class)

    def forward(self, input_ids):
        '''
        forward propagation
        :param inputs: [batch_size,seq_length]
        :return: [batch_size,num_class]
        '''
        x = self.embedding(input_ids)
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



