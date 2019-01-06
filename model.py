import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
class BiLSTMNet(nn.Module):
    def __init__(self, vocab_size, embed_size,weight,word_to_idx,idx_to_word,labels,
            num_hiddens = 100,num_layers = 2,bidirectional = True,**kwargs):
        super(BiLSTMNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0.1)
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 2, labels)
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)
        return outputs
class BiGRUNet(nn.Module):
    def __init__(self, vocab_size, embed_size,weight,word_to_idx,idx_to_word,labels,
            num_hiddens = 100,num_layers = 2,bidirectional = True,**kwargs):
        super(BiGRUNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.GRU(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0.1)
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 2, labels)
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)
        return outputs
class textCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_len, labels, weight,word_to_idx,idx_to_word,**kwargs):
        super(textCNN, self).__init__(**kwargs)
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.conv1 = nn.Conv2d(1, 1, (3, embed_size))
        self.conv2 = nn.Conv2d(1, 1, (4, embed_size))
        self.conv3 = nn.Conv2d(1, 1, (5, embed_size))
        self.pool1 = nn.MaxPool2d((seq_len - 3 + 1, 1))
        self.pool2 = nn.MaxPool2d((seq_len - 4 + 1, 1))
        self.pool3 = nn.MaxPool2d((seq_len - 5 + 1, 1))
        self.linear = nn.Linear(3, labels)
    def forward(self, inputs):
        inputs = self.embedding(inputs).view(inputs.shape[0], 1, inputs.shape[1], -1)
        x1 = F.relu(self.conv1(inputs))
        x2 = F.relu(self.conv2(inputs))
        x3 = F.relu(self.conv3(inputs))

        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)

        x = torch.cat((x1, x2, x3), -1)
        x = x.view(inputs.shape[0], 1, -1)

        x = self.linear(x)
        x = x.view(-1, self.labels)

        return(x)
