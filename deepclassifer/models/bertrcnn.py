import torch
import torch.nn as nn
from transformers import *
import numpy as np

class BertRCNN(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_size,
                 dropout_rate,
                 num_class,
                 bert_path,
                 rnn_type="lstm",
                 num_layers=1,
                 requires_grads=False):
        '''
        initialization
        :param embedding_dim:embedding dim
        :param hidden_size: rnn hidden size
        :param dropout_rate: dropout rate
        :param num_class: the number of label
        :param bert_path: bert path
        :param rnn_type: rnn type. Default:lstm
        :param num_layers: the number of rnn layer
        :param requires_grads: whether to update gradient of Bert in training stage
        '''
        super(BertRCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.bert_path=bert_path
        self.requires_grads=requires_grads

        self.bert = AutoModel.from_pretrained(self.bert_path)
        if self.requires_grads is False:
            for p in self.bert.parameters():
                p.requires_grads = False

        if self.rnn_type == "lstm":
            self.birnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                                 num_layers=self.num_layers, batch_first=True, bidirectional=True)
        elif self.rnn_type == "gru":
            self.birnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, batch_first=True, bidirectional=True)
        else:
            raise ValueError("rnn type must be one of {lstm,gru}.")

        self.W = nn.Linear(in_features=self.embedding_dim + self.hidden_size * 2 * self.num_layers,
                           out_features=self.hidden_size * 2)

        self.global_max_pool1d = nn.AdaptiveMaxPool1d(output_size=1)
        self.classifier = nn.Linear(in_features=self.hidden_size * 2, out_features=self.num_class)

    def forward(self, input_ids, attention_mask=None):
        '''
        forard propagation
        :param params: input_ids:[batch_size,max_length]
        :param params: attention_mask:[batch_size,max_length]
        :return: logits:[batch_size,num_class]
        '''

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_output.last_hidden_state

        rnn_output, _ = self.birnn(x)
        x = torch.cat((x, rnn_output), dim=-1)
        x = torch.tanh(self.W(x))
        x = x.permute(0, 2, 1)
        x = self.global_max_pool1d(x).squeeze(dim=-1)
        outputs = self.classifier(x)

        return outputs

if __name__=="__main__":

    from torch.utils.data import Dataset, DataLoader

    class my_dataset(Dataset):
        def __init__(self, data,mask,label):
            self.data = data
            self.label = label
            self.mask=mask

        def __len__(self):
            return len(self.data)

        def __getitem__(self, item):
            item_data = self.data[item]
            item_mask=self.mask[item]
            item_label = self.label[item]

            return item_data, item_mask,item_label


    data = np.random.randint(0, 100, (100, 60))
    mask=np.random.randint(0,1,(100,60))
    label = torch.from_numpy(np.array([int(x > 0.5) for x in np.random.randn(100)]))

    batch_size = 20
    my_data = my_dataset(data, mask, label)
    final_data = DataLoader(my_data, batch_size=batch_size, shuffle=True)

    my_model=BertRCNN(768,100,0.2,2,"/Users/codewithzichao/Desktop/开源的库/DeepClassifier/bert-base-uncased/")

    import torch.optim as optim
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(my_model.parameters())


    def train_step(params,batch_label):
        preds = my_model(input_ids=params["input_ids"],attention_mask=params["attention_mask"])
        loss = loss_fn(preds, batch_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return preds, loss


    epochs = 5
    for epoch in range(1, epochs + 1):

        for batch_index, (batch_data, batch_mask,batch_label) in enumerate(final_data):
            preds, loss = train_step({"input_ids":batch_data,"attention_mask":batch_mask}, batch_label)
            accuracy = 1.0 * sum(torch.argmax(preds, dim=-1) == batch_label) / batch_size
            print(f"epoch:{epoch},loss:{loss},accuracy:{accuracy}")

