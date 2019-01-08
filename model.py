import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
class EMA:
    def __init__(self,decay):
        self.decay = decay
        self.shadows = {}
        self.devices = {}
    def __len__(self):
        return len(self.shadows)
    def get(self,name):
        return self.shadows[name].to(self.devices[name])
    def set(self, name: str, param: nn.Parameter):
        self.shadows[name] = param.data.to('cpu').clone()
        self.devices[name] = param.data.device
    def update_parameter(self, name: str, param: nn.Parameter):
        if name in self.shadows:
            data = param.data
            new_shadow = self.decay * data + (1.0 - self.decay) * self.get(name)
            param.data.copy_(new_shadow)
            self.shadows[name] = new_shadow.to('cpu').clone()
class BiLSTMNet(nn.Module):
    def __init__(self, vocab_size, embed_size,weight,labels,
            num_hiddens = 100,num_layers = 2,bidirectional = True,**kwargs):
        super(BiLSTMNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
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
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        del hidden
        encoding = torch.cat([states[0], states[-1]], dim=1)
        hidden = self.decoder(encoding)
        outputs = self.sigmoid(hidden)
        return outputs
class BiGRUNet(nn.Module):
    def __init__(self, vocab_size, embed_size,weight,labels,
            num_hiddens = 100,num_layers = 2,bidirectional = True,**kwargs):
        super(BiGRUNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
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
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        hidden = self.decoder(encoding)
        outputs = self.sigmoid(hidden)
        return outputs
class textCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_len, labels, weight,**kwargs):
        super(textCNN, self).__init__(**kwargs)
        self.labels = labels
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.conv1 = nn.Conv2d(1, 1, (3, embed_size))
        self.conv2 = nn.Conv2d(1, 1, (4, embed_size))
        self.conv3 = nn.Conv2d(1, 1, (5, embed_size))
        self.pool1 = nn.MaxPool2d((seq_len - 3 + 1, 1))
        self.pool2 = nn.MaxPool2d((seq_len - 4 + 1, 1))
        self.pool3 = nn.MaxPool2d((seq_len - 5 + 1, 1))
        self.linear = nn.Linear(3, labels)
        self.sigmoid = nn.Sigmoid()
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
        outputs = self.sigmoid(x)
        return(outputs)
# Multiway Attention Networks for Modeling Sentence Pairs
class MANNet(nn.Module):
    def __init__(self,vocab_size,embed_size,encoder_size,labels,weight,dropout = 0.2):
        super(MANNet,self).__init__()
        self.drop_out = dropout
        self.labels = labels
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.l_encoder = nn.GRU(input_size=embed_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.r_encoder = nn.GRU(input_size=embed_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        # Concat Attention
        self.Wc1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wc2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vc = nn.Linear(encoder_size, 1, bias=False)
        # Bilinear Attention
        self.Wb = nn.Linear(2 * encoder_size, 2 * encoder_size, bias=False)
        # Dot Attention :
        self.Wd = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vd = nn.Linear(encoder_size, 1, bias=False)
        # Minus Attention :
        self.Wm = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vm = nn.Linear(encoder_size, 1, bias=False)

        self.Ws = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vs = nn.Linear(encoder_size, 1, bias=False)

        self.gru_agg = nn.GRU(12 * encoder_size, encoder_size, batch_first=True, bidirectional=True)            
        # Mixed Attention
        self.Wp = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vp = nn.Linear(encoder_size, 1, bias=False)
        self.Wc1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wc2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vc = nn.Linear(encoder_size, 1, bias=False)
        self.prediction = nn.Linear(2 * encoder_size, self.labels, bias=False)
        # initialize the weights
        self.initiation()
    def initiation(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, 0.1)
    def forward(self,inputs):
        # Encoding Layer
        left_embedding = self.embedding(inputs)
        right_embedding = self.embedding(inputs)
        hl,_ = self.l_encoder(left_embedding)
        hl=F.dropout(hl,self.drop_out)
        hr,_ = self.r_encoder(right_embedding)
        hr =F.dropout(hr,self.drop_out)
        # print(hl.size(),hr.size())
        # Multiway Matching
        # Concat Attention
        _s1 = self.Wc1(hl).unsqueeze(1)
        _s2 = self.Wc2(hr).unsqueeze(2)
        sjt = self.vc(torch.tanh(_s1 + _s2)).squeeze()
        ait = F.softmax(sjt, 2)
        ptc = ait.bmm(hl)
        # print(ptc.size())
        # BiLinear Attention
        _s1 = self.Wb(hl).transpose(2, 1)
        sjt = hr.bmm(_s1)
        ait = F.softmax(sjt, 2)
        ptb = ait.bmm(hl)
        # print(ptb.size())
        # Dot Attention
        _s1 = hl.unsqueeze(1)
        _s2 = hr.unsqueeze(2)
        sjt = self.vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        ptd = ait.bmm(hl)
        # print(ptd.size())
        # Minus Attention
        sjt = self.vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        ptm = ait.bmm(hl)
        # print(ptm.size())
        # Weighted Sum Attention
        _s1 = hr.unsqueeze(1)
        _s2 = hr.unsqueeze(2)
        sjt = self.vs(torch.tanh(self.Ws(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        pts = ait.bmm(hr)
        # print(pts.size())
        # Inside Aggregation
        aggregation = torch.cat([hr, pts, ptc, ptd, ptb, ptm], 2)
        aggregation_representation, _ = self.gru_agg(aggregation)
        # print(aggregation_representation.size())
        # Mixed Aggregation
        sj = self.vp(torch.tanh(self.Wp(hl))).transpose(2, 1)
        rl = F.softmax(sj, 2).bmm(hl)
        sj = F.softmax(self.vc(self.Wc1(aggregation_representation) + self.Wc2(rl)).transpose(2, 1), 2)
        rr = sj.bmm(aggregation_representation)
        score = torch.sigmoid(self.prediction(rr.squeeze()))
        return score
