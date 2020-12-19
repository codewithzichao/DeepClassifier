# -*- coding:utf-8 -*-
'''

Author:
    Zichao Li,2843656167@qq.com

'''
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from deepclassifier.models import TextCNN
from deepclassifier.trainers import Trainer
from tensorboardX import SummaryWriter
from preprocessing import load_pretrained_embedding, texts_convert_to_ids,pad_sequences
from sklearn.model_selection import train_test_split

# 数据路径
base_path = "/Users/codewithzichao/MyGithubProjects/nlp_programs/DeepClassifier/examples/sentiment-analysis-on-movie-reviews/"
train_data_path = base_path + "train.tsv"
test_data_path = base_path + "test.tsv"

# 获取数据
train_data_df = pd.read_csv(train_data_path, sep="\t")
train_data_df, dev_data_df = train_test_split(train_data_df, test_size=0.2)
test_data_df = pd.read_csv(test_data_path, sep="\t")

train_data = train_data_df.iloc[:, -2].values
train_label = train_data_df.iloc[:, -1].values
dev_data = dev_data_df.iloc[:, -2].values
dev_label = dev_data_df.iloc[:, -1].values
test_data = test_data_df.iloc[:, -1].values

# 获取词典与词向量
pretrained_embedding_file_path = "/Users/codewithzichao/MyGithubProjects/nlp_programs/DeepClassifier/examples/glove/glove.6B.50d.txt"
word2idx, embedding_matrix = load_pretrained_embedding(pretrained_embedding_file_path=pretrained_embedding_file_path)

# 文本向量化
train_data = texts_convert_to_ids(train_data, word2idx)
dev_data = texts_convert_to_ids(dev_data, word2idx)
test_data = texts_convert_to_ids(test_data, word2idx)

train_data=torch.from_numpy(pad_sequences(train_data))
dev_data=torch.from_numpy(pad_sequences(dev_data))
test_data=torch.from_numpy(pad_sequences(test_data))

# 产生batch data
class my_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item_data = self.data[item]
        item_label = self.label[item]

        return item_data, item_label


class my_dataset1(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item_data = self.data[item]

        return item_data


# 训练集
batch_size = 20
my_train_data = my_dataset(train_data, train_label)
train_loader = DataLoader(my_train_data, batch_size=batch_size, shuffle=True,drop_last=True)
# 验证集
my_dev_data = my_dataset(dev_data, dev_label)
dev_loader = DataLoader(my_dev_data, batch_size=batch_size, shuffle=True,drop_last=True)
# 测试集
pred_data = my_dataset1(test_data)
pred_data = DataLoader(pred_data, batch_size=1)

# 定义模型
my_model = TextCNN(embedding_dim=embedding_matrix.shape[1], dropout_rate=0.2, num_class=5,
                   embedding_matrix=embedding_matrix, requires_grads=False)
optimizer = optim.Adam(my_model.parameters())
loss_fn = nn.CrossEntropyLoss()
save_path = "best.ckpt"

writer = SummaryWriter("logfie/1")
my_trainer = Trainer(model_name="textcnn", model=my_model, train_loader=train_loader, dev_loader=dev_loader,
                     test_loader=None, optimizer=optimizer, loss_fn=loss_fn, save_path=save_path, epochs=100,
                     writer=writer, max_norm=0.25, eval_step_interval=10, device='cpu')

# 训练
my_trainer.train()
# 测试
p, r, f1 = my_trainer.test()
print(p, r, f1)
# 打印在验证集上最好的f1值
print(my_trainer.best_f1)

# 预测
prd_label = my_trainer.predict(pred_data)
print(prd_label.shape)
