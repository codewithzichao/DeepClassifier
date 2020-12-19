# Example
 In this part, I will show you how to use DeepClassifier to carry text classification task.🥰

## Dataset 
   **kaggle dataset:** [sentiment-analysis-on-movie-reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)

   **Pretrained embedding:** GloVe [download](https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.6B.zip)

   **BERT pretrained weights:** [download](https://huggingface.co/bert-base-uncased)

## EDA

Before processing the data,you can do data analysis：

![image](https://github.com/codewithzichao/DeepClassifier/blob/master/examples/len.png)

![image](https://github.com/codewithzichao/DeepClassifier/blob/master/examples/label.png)

## Preprocessing

This step is mainly divided into two steps: 

* load_pretrained_embedding
* texts_convert_to_ids 
  

Details can be seen **proprecessing.py**.

## Training
The core code is below:🥰

**TextCNN**

Details can be seen **example_textcnn.py**.

```python
from deepclassifier.models import TextCNN
from deepclassifier.trainers import Trainer
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
pred_label = my_trainer.predict(pred_data)
print(pred_label.shape)
```

**BertTextCNN**

Details can be seen **example_berttextcnn.py**.
```python
from deepclassifier.models import BertTextCNN
from deepclassifier.trainers import Trainer
# 定义模型
my_model = BertTextCNN(embedding_dim=768, dropout_rate=0.2, num_class=5,
                       bert_path=bert_path)

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
pred_label = my_trainer.predict(pred_data)
print(pred_label.shape)

```

if you want to run  **example_textcnn.py** or **example_berttextcnn.py**, please **download datasets and glove, and replace the data dir.** Have fun!🥰

> Your file dir must be like that:👇
```shell

├── bert-base-uncased
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── rust_model.ot
│   ├── tf_model.h5
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── example_berttextcnn.py
├── example_textcnn.py
├── glove
│   ├── glove.6B.100d.txt
│   ├── glove.6B.200d.txt
│   ├── glove.6B.300d.txt
│   ├── glove.6B.300d.txt.pt
│   ├── glove.6B.50d.txt
│   └── glove.6B.zip
├── preprocessing.py
└── sentiment-analysis-on-movie-reviews
    ├── sampleSubmission.csv
    ├── test.tsv
    └── train.tsv
```

