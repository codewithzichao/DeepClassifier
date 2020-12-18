import torch
import torch.nn as nn
import torch.nn.functional as F


class DPCNN(nn.Module):
    def __init__(self,
                 embedding_dim,
                 dropout_rate,
                 num_class,
                 vocab_size=0,
                 seq_length=0,
                 num_blocks=3,
                 num_filters=250,
                 kernel_sizes=3,
                 embedding_matrix=None,
                 requires_grads=False):
        '''
        initialization
        ⚠️In default,the way to initialize embedding is loading pretrained embedding look-up table!
        :param embedding_dim: embedding dim
        :param num_class: the number of label
        :param dropout_rate: dropout rate
        :param vocab_size: vocabulary size
        :param seq_length: max length of sequence after padding
        :param num_blocks: the number of block in DPCNN model
        :param num_filters: the number of filters of conv kernel
        :param kernel_sizes: conv kernel size
        :param embedding_matrix: pretrained embedding look up table
        :param requires_grads: whether to update gradient of embedding in training stage
        '''
        super(DPCNN, self).__init__()

        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.dropout_rate=dropout_rate
        self.num_blocks = num_blocks
        self.num_class = num_class
        self.kernel_sizes = kernel_sizes
        self.embedding_matrix = embedding_matrix
        self.requires_grads = requires_grads

        # embedding
        if self.embedding_matrix is None:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.embedding_dim,
                                          padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=self.requires_grads)
            self.vocab_size = self.embedding_matrix.shape[0]

        # text region embedding
        self.region_embedding = nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                                          stride=1, kernel_size=(self.kernel_sizes, self.embedding_dim))

        # two conv
        self.conv2d1 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters,
                                 stride=2, kernel_size=(self.kernel_sizes, 1), padding=0)
        self.conv2d2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters,
                                 stride=2, kernel_size=(self.kernel_sizes, 1), padding=0)
        self.padding1 = nn.ZeroPad2d((0, 0, (self.kernel_sizes-1)//2, (self.kernel_sizes-1)-((self.kernel_sizes-1)//2)))  # top bottom
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

    def forward(self, inputs):
        '''
        forward propagation
        :param inputs: [batch_size,seq_length]
        :return: [batch_size,num_class]
        '''

        embedding=self.embedding(inputs).unsqueeze(dim=1)
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


