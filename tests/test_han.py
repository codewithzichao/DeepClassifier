import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from deepclassifer.models import HAN
from deepclassifer.trainers import Trainer
from tensorboardX import SummaryWriter


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
train_data = np.random.randint(0, 100, (100, 60,10))
train_label = torch.from_numpy(np.array([int(x > 0.5) for x in np.random.randn(100)]))
my_train_data = my_dataset(train_data, train_label)
final_train_data = DataLoader(my_train_data, batch_size=batch_size, shuffle=True)

# 验证集
dev_data = np.random.randint(0, 100, (100, 60,10))
dev_label = torch.from_numpy(np.array([int(x > 0.5) for x in np.random.randn(100)]))
my_dev_data = my_dataset(dev_data, dev_label)
final_dev_data = DataLoader(my_dev_data, batch_size=batch_size, shuffle=True)

# 测试集
test_data = np.random.randint(0, 100, (100, 60,10))
test_label = torch.from_numpy(np.array([int(x > 0.5) for x in np.random.randn(100)]))
my_test_data = my_dataset(test_data, dev_label)
final_test_data = DataLoader(my_test_data, batch_size=batch_size, shuffle=True)

my_model = HAN(10,100,100,0.2,2,100,60)
optimizer = optim.Adam(my_model.parameters())
loss_fn = nn.CrossEntropyLoss()
save_path = "best.ckpt"

writer = SummaryWriter("logfie/1")
my_trainer = Trainer(model_name="han", model=my_model, train_loader=final_train_data, dev_loader=final_dev_data,
                     test_loader=final_test_data, optimizer=optimizer, loss_fn=loss_fn,
                     save_path=save_path, epochs=100, writer=writer, max_norm=0.25, eval_step_interval=10)

# 训练
my_trainer.train()
# 测试
p, r, f1 = my_trainer.test()
print(p, r, f1)
# 打印在验证集上最好的f1值
print(my_trainer.best_f1)

# 预测
pred_data = np.random.randint(0, 100, (100, 60))
pred_data=my_dataset1(pred_data)
pred_data=DataLoader(pred_data,batch_size=1)
prd_label=my_trainer.predict(pred_data)
print(prd_label.shape)
