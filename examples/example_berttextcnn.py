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
from deepclassifier.models import BertTextCNN
from deepclassifier.trainers import Trainer
from tensorboardX import SummaryWriter
from transformers import BertTokenizer
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


# 产生batch data
class my_dataset(Dataset):
    def __init__(self, data, label, max_length, tokenizer=None):
        self.data = data
        self.label = label
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item_data = self.data[item]
        item_label = [self.label[item]]

        item_data = item_data.strip().split()
        c = ["[CLS]"] + item_data + ["SEP"]
        input_ids = self.tokenizer.convert_tokens_to_ids(c)
        if len(input_ids) >= self.max_length:
            input_ids = input_ids[:self.max_length]
        attention_mask = [1.0] * len(input_ids)
        extra = self.max_length - len(input_ids)
        if extra > 0:
            input_ids += [0] * extra
            attention_mask += [0.0] * extra

        return torch.LongTensor(input_ids), torch.FloatTensor(attention_mask), torch.LongTensor(item_label)


class my_dataset1(Dataset):
    def __init__(self, data, max_length, tokenizer=None):
        self.data = data

        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item_data = self.data[item]

        item_data = item_data.strip().split()
        c = ["[CLS]"] + item_data + ["SEP"]
        input_ids = self.tokenizer.convert_tokens_to_ids(c)
        if len(input_ids) >= self.max_length:
            input_ids = input_ids[:self.max_length]
        attention_mask = [1.0] * len(input_ids)
        extra = self.max_length - len(input_ids)
        if extra > 0:
            input_ids += [0] * extra
            attention_mask += [0.0] * extra

        return input_ids, attention_mask


tokenizer = BertTokenizer(vocab_file="/Users/codewithzichao/Desktop/开源的库/bert-base-uncased/vocab.txt")
# 训练集
batch_size = 20
my_train_data = my_dataset(train_data, train_label, 200, tokenizer)
train_loader = DataLoader(my_train_data, batch_size=batch_size, shuffle=True, drop_last=True)
# 验证集
my_dev_data = my_dataset(dev_data, dev_label, 200, tokenizer)
dev_loader = DataLoader(my_dev_data, batch_size=batch_size, shuffle=True, drop_last=True)
# 测试集
pred_data = my_dataset1(test_data, 200, tokenizer)
pred_data = DataLoader(pred_data, batch_size=1)

# 定义模型
my_model = BertTextCNN(embedding_dim=768, dropout_rate=0.2, num_class=5,
                       bert_path="/Users/codewithzichao/Desktop/开源的库/bert-base-uncased/")

optimizer = optim.Adam(my_model.parameters())
loss_fn = nn.CrossEntropyLoss()
save_path = "best.ckpt"

writer = SummaryWriter("logfie/1")
my_trainer = Trainer(model_name="berttextcnn", model=my_model, train_loader=train_loader, dev_loader=dev_loader,
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
