TextCNN model
==========

I will show you that the pramamters of textcnn.🤩

**Initialization**
===========================

class TextCNN(nn.Module):

    def __init__(self,embedding_dim,dropout_rate,
        num_class,vocab_size=0,seq_length=0,
        num_layers=3,kernel_sizes=[3, 4, 5],
        strides=[1, 1, 1],paddings=[0, 0, 0],
        num_filters=[100, 100, 100],
        embedding_matrix=None,
        requires_grads=False):



In default,the way to initialize embedding is loading pretrained embedding look-up table!

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

