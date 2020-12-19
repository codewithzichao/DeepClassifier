import torch
import numpy as np
import codecs


def load_pretrained_embedding(pretrained_embedding_file_path):
    '''
    # 加载预训练的词向量，这里使用GloVe
    :param pretrained_embedding_file_path: 预训练词向量文件存放的路径
    :return:
    '''
    embedding_matrix = []
    word2idx = dict()
    word2idx["__PAD__"] = 0
    word2idx["__UNK__"] = 1
    index = 2

    with codecs.open(pretrained_embedding_file_path, "r", encoding="utf-8") as f:
        temp = f.readline().strip().split(" ")
        embedding_dim = len(temp) - 1

        embedding_matrix.append(np.zeros(shape=(embedding_dim,)))
        embedding_matrix.append(np.random.randn(embedding_dim))

        f.seek(0)
        for line in f:
            line = line.strip().split(" ")
            word, emd = line[0], line[1:]
            emd = [float(x) for x in emd]

            word2idx[word] = index
            index += 1
            embedding_matrix.append(np.array(emd))

    assert word2idx.__len__() == embedding_matrix.__len__()
    embedding_matrix = torch.from_numpy(np.array(embedding_matrix)).float()

    return word2idx, embedding_matrix



def text_convert_to_ids(text, word2idx):
    '''
    # 将一个样本向量化
    :param text: text
    :param word2idx: 词典
    :return: 向量化的文本
    '''
    input_ids = []
    tokenizered_text = text.strip().split(" ")
    for item in tokenizered_text:
        id = word2idx.get(item, 1)
        input_ids.append(id)

    return np.array(input_ids)


def texts_convert_to_ids(texts, word2idx):
    '''
    # 将样本的list向量化
    :param texts: list of text,[text1,text2,...]
    :param word2idx: 词典
    :return: 向量化的文本
    '''
    input_ids = []
    for text in texts:
        temp = text_convert_to_ids(text, word2idx)
        input_ids.append(temp)

    return np.array(input_ids)


def pad_sequences(input_id, max_length=200):
    '''
    padding
    :param input_id: 向量化的文本
    :param max_length: 最大长度
    :return: 向量化的文本
    '''
    input_ids = []
    for i in range(input_id.shape[0]):
        if input_id[i].__len__() >= max_length:
            input_ids.append(input_id[i][:max_length])
        else:
            extra = max_length - input_id[i].__len__()
            input_ids.append(np.append(input_id[i], [0] * extra))

    return np.array(input_ids)
