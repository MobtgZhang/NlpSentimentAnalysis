import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import BiLSTM,BiGRU,BiSRU
from .layers import SelfAttentiveEncoder,Aggregation
from .highway import RecurrentHighway
class RNNs_AttNone(nn.Module):
    def __init__(self,config):
        super(RNNs_AttNone,self).__init__()
        if config["rnn_model"] == "BiLSTM":
            self.encoder = BiLSTM(config)
        elif config["rnn_model"] == "BiGRU":
            self.encoder = BiGRU(config)
        elif config["rnn_model"] == "BiSRU":
            self.encoder = BiSRU(config)
        else:
            raise Exception('Error when initializing Classifier')
        self.rnn_name = config["rnn_model"]
        self.drop_att = nn.Dropout(config['drop_att'])
        self.drop_out = nn.Dropout(config['drop_out'])
        self.gru_agg = nn.GRU(input_size=config['emb_size'] * 2, hidden_size=config['emb_size'],
                              bidirectional=True)
        """
            prediction layer
        """
        encoder_size = config['emb_size']
        self.Wp = nn.Linear(encoder_size, encoder_size, bias=False)
        self.vp = nn.Linear(encoder_size,config['seperate_hops'], bias=False)
        self.vq = nn.Linear(config['text_length'], config['seperate_hops'], bias=False)
        self.Wc1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wc2 = nn.Linear(encoder_size, encoder_size, bias=False)
        self.vc = nn.Linear(encoder_size, config['labels'], bias=False)

        self.prediction = nn.Linear(6,config["class_number"], bias=False)
    def forward(self, inp,hidden):
        aggregation,embedding = self.encoder(inp,hidden)
        # text_length*batch_size*(num_directions * hidden_size)
        # batch_size*text_length*hidden_size
        aggregation = self.drop_out(aggregation)
        attention = self.drop_att(embedding)
        aggregation_representation, _ = self.gru_agg(aggregation)

        # batch_size*text_length*(num_directions * hidden_size)
        sj = self.vp(torch.tanh(self.Wp(attention))).transpose(2, 1)
        rp = F.softmax(sj, 2).bmm(attention)

        aggregation_representation = aggregation_representation.permute([1,2,0])
        rq = self.vq(aggregation_representation).permute([0,2,1])
        out = self.Wc2(rp) + self.Wc1(rq)

        rc = F.softmax(self.vc(out).transpose(2, 1), 2)

        score = F.softmax(self.prediction(rc).transpose(2, 1), dim=1)
        attention = None
        return score, attention
    def init_hidden(self,bsz):
        return self.encoder.init_hidden(bsz)
class SABiLSTMOLD(nn.Module):
    def __init__(self, config):
        super(SABiLSTMOLD, self).__init__()
        self.encoder = SelfAttentiveEncoder(config)
        self.fc = nn.Linear(config['emb_size'] * 2 * config['attention-hops'], config['nfc'])
        # predict layer
        self.Agg_layer = Aggregation(config)
    def init_weights(self, init_range=0.1):
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)
    def forward(self, inp, hidden):
        outp, attention = self.encoder.forward(inp, hidden)
        typename = type(self.encoder)
        score, attention =  self.Agg_layer(outp, attention, typename)
        return score, attention
    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)
class SABiRNN(nn.Module):
    def __init__(self,config):
        super(SABiRNN,self).__init__()
        self.encoder = SelfAttentiveEncoder(config)
        self.fc = nn.Linear(config['emb_size'] * 2 * config['attention-hops'], config['nfc'])
        self.drop_att = nn.Dropout(config['drop_att'])
        self.drop_out = nn.Dropout(config['drop_out'])
        self.attention_hops = self.encoder.attention_hops
        self.gru_agg = nn.GRU(input_size=self.attention_hops, hidden_size=config['seperate_hops'],
                              bidirectional=True)
        """
            prediction layer
        """
        self.att = nn.Linear(config['emb_size'] * config['nlayers'],config['seperate_hops'],bias=False)
        self.Wp = nn.Linear(config['seperate_hops']*2,config['seperate_hops'], bias=False)
        self.vp = nn.Linear(config['seperate_hops'], config['seperate_hops'], bias=False)
        self.Wc = nn.Linear(config['seperate_hops']*2, config['labels'], bias=False)
        self.prediction = nn.Linear(config['seperate_hops'], config["class_number"], bias=False)
    def forward(self, inp,hidden):
        attention, out_hidden = self.encoder.forward(inp, hidden)
        # text_length*attention_hops*(emb_size*nlayers)
        # text_length*attention_hops*batch_size
        # dropout layer
        attention = self.drop_out(attention)
        attention = self.att(attention)
        inputs = self.drop_att(out_hidden).transpose(1,2)
        # GRU Aggregation layer
        agg_rep, _ = self.gru_agg(inputs)
        # predict layer
        agg_rep = agg_rep.transpose(0,1)
        sj = self.vp(torch.tanh(self.Wp(agg_rep))).transpose(2, 1)
        rp = F.softmax(sj, 2).bmm(agg_rep)
        out = self.Wc(rp)
        rc = F.softmax(out.transpose(2, 1),2)
        score = F.softmax(self.prediction(rc).transpose(2, 1), dim=1)
        return score, attention
    def init_hidden(self,bsz):
        return self.encoder.init_hidden(bsz)

class HighAtt(nn.Module):
    def __init__(self,config):
        super(HighAtt,self).__init__()
        self.encoder = nn.Embedding(config['ntoken'], config['emb_size'])
        self.higway = RecurrentHighway(input_size=config['emb_size'],hidden_size=config['emb_size'],
                                       recurrence_length=5,vocab_size=config['ntoken'])

    def forward(self, inp,hidden):
        pass
class BertAtt(nn.Module):
    def __init__(self):
        super(BertAtt,self).__init__()
    def forward(self, inp,hidden):
        pass