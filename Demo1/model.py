import torch
import torch.nn as nn
import torch.nn.functional as F
import torchnlp

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
class BiGRUNet(nn.Module):
    def __init__(self, vocab_size, embed_size,weight=None,seq_len = 1000,first_labels = 6,second_labels = 20,
            statement_labels = 4,num_hiddens = 100,num_layers = 2,bidirectional = True,use_gpu=False,**kwargs):
            super(BiGRUNet,self).__init__(**kwargs)
            # model args
            self.vocab_size = vocab_size
            self.embed_size = embed_size
            self.first_labels =  first_labels
            self.second_labels = second_labels
            self.statement_labels = statement_labels
            self.num_hiddens = num_hiddens
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.use_gpu = use_gpu
            self.seq_len = seq_len
            # model of inner 
            if weight is None:
                self.embedding = nn.Embedding(vocab_size,embed_size)
            else:
                self.embedding = nn.Embedding.from_pretrained(weight)
                self.embedding.weight.requires_grad = False
            self.encoder = nn.GRU(input_size = embed_size,hidden_size = num_hiddens,
                            num_layers = num_layers,bidirectional = bidirectional,dropout = 0.1)
            t_sizea = num_hiddens//2
            # seq*num_hiddens or seq*num_hiddens*2-->seq*t_sizea
            if bidirectional:
                self.decoder = nn.Linear(num_hiddens*2,t_sizea)
            else:
                self.decoder = nn.Linear(num_hiddens,t_sizea)
            # seq*t_sizea -->first_labels*t_sizea
            self.Wa = nn.Linear(seq_len,first_labels)
            # first_labels*t_sizea --> first_labels*second_labels
            self.Wb = nn.Linear(t_sizea,second_labels)
            # first_labels*second_labels --> statement_labels*second_labels
            self.Wc = nn.Linear(first_labels,statement_labels)
    def forward(self,inputs,run = True):
        if self.use_gpu and run:
            embeddings = self.embedding(inputs.cuda())
        else:
            embeddings = self.embedding(inputs) # btc*seq*emb
        states, hidden = self.encoder(embeddings.permute([1, 0, 2])) # btc*seq*hid
        
        hidden = self.decoder(states.permute([1,0,2])) # btc*seq*t_sizea
        ajt = self.Wa(hidden.permute([0,2,1])).permute([0,2,1]) #btc*first_labels*t_sizea
        bjt = self.Wb(ajt) # btc*first_labels*second_labels
        ojt = self.Wc(bjt.permute([0,2,1])).permute([0,2,1])# btc*statement_labels*second_labels
        predict = F.softmax(ojt,dim = 1)
        return predict
class BiLSTMNet(nn.Module):
    def __init__(self, vocab_size, embed_size,weight=None,seq_len = 1000,first_labels = 6,second_labels = 20,
            statement_labels = 4,num_hiddens = 100,num_layers = 2,bidirectional = True,use_gpu=False,**kwargs):
            super(BiLSTMNet,self).__init__(**kwargs)
            # model args
            self.vocab_size = vocab_size
            self.embed_size = embed_size
            self.first_labels =  first_labels
            self.second_labels = second_labels
            self.statement_labels = statement_labels
            self.num_hiddens = num_hiddens
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.use_gpu = use_gpu
            self.seq_len = seq_len
            # model of inner 
            if weight is None:
                self.embedding = nn.Embedding(vocab_size,embed_size)
            else:
                self.embedding = nn.Embedding.from_pretrained(weight)
                self.embedding.weight.requires_grad = False
            self.encoder = nn.LSTM(input_size = embed_size,hidden_size = num_hiddens,
                            num_layers = num_layers,bidirectional = bidirectional,dropout = 0.1)
            t_sizea = num_hiddens//2
            # seq*num_hiddens or seq*num_hiddens*2-->seq*t_sizea
            if bidirectional:
                self.decoder = nn.Linear(num_hiddens*2,t_sizea)
            else:
                self.decoder = nn.Linear(num_hiddens,t_sizea)
            # seq*t_sizea -->first_labels*t_sizea
            self.Wa = nn.Linear(seq_len,first_labels)
            # first_labels*t_sizea --> first_labels*second_labels
            self.Wb = nn.Linear(t_sizea,second_labels)
            # first_labels*second_labels --> statement_labels*second_labels
            self.Wc = nn.Linear(first_labels,statement_labels)
    def forward(self,inputs,run = True):
        if self.use_gpu and run:
            embeddings = self.embedding(inputs.cuda())
        else:
            embeddings = self.embedding(inputs) # btc*seq*emb
        states,_ = self.encoder(embeddings.permute([1, 0, 2])) # btc*seq*hid
        
        hidden = self.decoder(states.permute([1,0,2])) # btc*seq*t_sizea
        ajt = self.Wa(hidden.permute([0,2,1])).permute([0,2,1]) #btc*first_labels*t_sizea
        bjt = self.Wb(ajt) # btc*first_labels*second_labels
        ojt = self.Wc(bjt.permute([0,2,1])).permute([0,2,1])# btc*statement_labels*second_labels
        predict = F.softmax(ojt,dim = 1)
        return predict.cpu()
class textCNN(nn.Module):
    def __init__(self, vocab_size, embed_size,weight=None,seq_len = 1000,first_labels = 6,second_labels = 20,
            statement_labels = 4,num_layers = 2,bidirectional = True,use_gpu=False,**kwargs):
        super(textCNN, self).__init__(**kwargs)
        # model args
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.first_labels = first_labels
        self.second_labels = second_labels
        self.statement_labels = statement_labels
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_gpu = use_gpu
        # model defination
        if weight is None:
            self.embedding = nn.Embedding(vocab_size,embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(weight)
            self.embedding.weight.requires_grad = False
        self.conv_lists = []
        self.pool_lists = []
        for k in range(self.first_labels):
            convd = nn.Conv2d(1, first_labels, (5, embed_size))
            setattr(self,"convd%d"%k,convd)
            self.conv_lists.append(convd)
        for k in range(self.first_labels):
            maxpool = nn.MaxPool2d((seq_len,5),padding = 2)
            setattr(self,"maxpool%d"%k,maxpool)
            self.pool_lists.append(maxpool)
        t_sizea = 200
        # first_labels*t_sizea --> first_labels*second_labels
        self.Wb = nn.Linear(t_sizea,second_labels)
        # first_labels*second_labels --> statement_labels*second_labels
        self.Wc = nn.Linear(first_labels,statement_labels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs,run = True):
        if self.use_gpu:
            inputs = self.embedding(inputs.cuda()).view(inputs.shape[0], 1, inputs.shape[1], -1)
        else:
            inputs = self.embedding(inputs).view(inputs.shape[0], 1, inputs.shape[1], -1)
        cat_list = []
        for k in range(self.first_labels):
            p_out = F.relu(self.conv_lists[k](inputs).permute([0,3,1,2]))
            out = self.pool_lists[k](p_out)
            cat_list.append(out)
        ajt = torch.cat(cat_list,dim = 2).squeeze()
        bjt = self.Wb(ajt) # btc*first_labels*second_labels
        ojt = self.Wc(bjt.permute([0,2,1])).permute([0,2,1])# btc*statement_labels*second_labels
        outputs = F.softmax(ojt,dim = 1)
        return outputs
