import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import model.layers as layers
class Highway(nn.Module):
    def __init__(self, layer_num: int, size: int):
        super(Highway,self).__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])

    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x.transpose(1, 2)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super(DepthwiseSeparableConv,self).__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))
class Embedding(nn.Module):
    def __init__(self,char_dim,word_dim,dropout_char,dropout_word):
        super(Embedding,self).__init__()
        self.char_dim = char_dim
        self.word_dim = word_dim
        self.dropout_char = dropout_char
        self.dropout_word = dropout_word
        self.conv2d = DepthwiseSeparableConv(char_dim, char_dim, 5, dim=2)
        self.high = Highway(2, word_dim+word_dim)

    def forward(self, ch_emb, wd_emb):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_char, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)
        ch_emb = ch_emb.squeeze()
        wd_emb = F.dropout(wd_emb, p=self.dropout_word, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.high(emb)
        return emb
class SelfAttentiveEncoder(nn.Module):
    def __init__(self,vocab_size,embedding_dim,
                      char_size,char_embedding_dim,hidden_size,
                      attention_hops,attention_unit,
                      dropout_emb,dropout_rnn,
                      rnn_type,use_gpu):
        super(SelfAttentiveEncoder, self).__init__()
        # Word embeddings (+1 for padding)
        self.attention_hops = attention_hops
        self.attention_unit = attention_unit
        self.rnn_type = rnn_type
        self.use_gpu = use_gpu
        self.dropout_emb = dropout_emb
        self.dropout_rnn = dropout_rnn
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      padding_idx=0)

        # Char embeddings (+1 for padding)
        self.char_embedding = nn.Embedding(char_size,
                                           char_embedding_dim,
                                           padding_idx=0)
        # encode embeddings
        self.embedding = layers.Embedding(char_dim =char_embedding_dim,
                                          word_dim =embedding_dim,
                                          dropout_char =dropout_emb,
                                          dropout_word =dropout_emb)
        in_size = char_embedding_dim + embedding_dim
        self.bilstm = self.RNN_TYPES[rnn_type](in_size,hidden_size,
                                                    bidirectional=True,
                                                    dropout = dropout_rnn)

        self.ws1 = nn.Linear(hidden_size * 2,attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, attention_hops, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.init_weights()

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        if hidden is None:
            batch_size = inp[0]
            hidden = self.init_hidden(batch_size)
        if self.use_gpu:
            hidden = hidden.cuda()
        outp = self.bilstm.forward(inp,hidden)
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
        transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]
        if self.dropout_emb > 0:
            emb = F.dropout(compressed_embeddings,p = self.dropout_emb)
        hbar = self.tanh(self.ws1(emb))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == self.dictionary.word2idx['<pad>']).float())
            # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def init_hidden(self, bsz):
        if self.rnn_type == 'lstm':
            weight = next(self.parameters()).data
            return (Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()))
class SentimentSelfAtt(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
    def __init__(self,args,normalize= True):
        super(SentimentSelfAtt,self).__init__()
        self.self_att = SelfAttentiveEncoder(vocab_size = args.vocab_size,
            embedding_dim = args.embedding_dim,char_size = args.char_size,
            char_embedding_dim = args.char_embedding_dim,hidden_size = args.hidden_size,
            attention_hops = args.attention_hops,attention_unit = args.attention_unit,
            dropout_emb = args.dropout_emb,dropout_rnn = args.dropout_rnn,
            rnn_type = args.rnn_type,use_gpu = args.cuda)
        self.match_att = layers.MatchNetwork()
    def forward(self, *input):
        pass