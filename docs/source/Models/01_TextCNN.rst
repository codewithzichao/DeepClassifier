deepclassifier.models.TextCNN
==========

I will show you that the parameters of TextCNN model.

.. code-block:: python


 class TextCNN(self,embedding_dim,dropout_rate,
        num_class,vocab_size=0,seq_length=0,
        num_layers=3,kernel_sizes=[3, 4, 5],
        strides=[1, 1, 1],paddings=[0, 0, 0],
        num_filters=[100, 100, 100],
        embedding_matrix=None,
        requires_grads=False):

Initialize TextCNN model.


.. important:: We strongly recommand you to use pre-trained embedding such as GloVe.

Parameters:
 - embedding_dim: embedding dim
 - dropout_rate: drouput rate
 - num_class: the number of label
 - vocab_size: vocabulary size
 - seq_length: max length of sequence after padding
 - num_layers: the number of cnn
 - kernel_sizes: list of conv kernel size
 - strides: list of conv strides
 - paddings: list of padding
 - num_filters: list of num filters
 - embedding_matrix: pretrained embedding look-up table,shape is:[vocab_size,embedding_dim]
 - requires_grads: whether to update gradient of embedding in training

.. code-block:: python

 forward(self, input_ids)
Parameters:
 - input_ids: [batch_size,seq_length]

**Reference**

.. code-block::

 @inproceedings{kim-2014-convolutional,
    title = "Convolutional Neural Networks for Sentence Classification",
    author = "Kim, Yoon",
    booktitle = "Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing ({EMNLP})",
    month = oct,
    year = "2014",
    address = "Doha, Qatar",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D14-1181",
    doi = "10.3115/v1/D14-1181",
    pages = "1746--1751",
}

