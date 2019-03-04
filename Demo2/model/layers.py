from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchnlp.nn as nnlp
import torch.nn.functional as F
from sru import SRU
import os
class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.encoder = nn.Embedding(config['ntoken'], config['emb_size'])
        self.bilstm = nn.LSTM(config['emb_size'], config['emb_size'], config['nlayers'],
                              dropout=config['dropout'],bidirectional=True,batch_first=True)
        self.nlayers = config['nlayers']
        self.nhid = config['emb_size']
        self.pooling = config['pooling']
        self.dictionary = config['dictionary']
        self.init_weights()
        self.encoder.weight.data[self.dictionary.word2idx['<unk>']] = 0

        if os.path.exists(config['word-vector']):
            print('Loading word vectors from', config['word-vector'])
            self.encoder.weight.data = torch.FloatTensor(config['embedding']['weight'])
            print('%d words from external word vectors loaded.' % config['embedding']['loaded_cnt'])
        # note: init_range constraints the value of initial weights
    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)
    def forward(self, inp, hidden):
        emb = self.drop(self.encoder(inp))
        outp,chidden = self.bilstm(emb, hidden)
        if self.pooling == 'mean':
            outp = torch.mean(outp, 0).squeeze()
        elif self.pooling == 'max':
            outp = torch.max(outp, 0)[0].squeeze()
        elif self.pooling == 'all' or self.pooling == 'all-word':
            outp = torch.transpose(outp, 0, 1).contiguous()
        return outp, emb

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()))
class BiGRU(nn.Module):
    def __init__(self,config):
        super(BiGRU,self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.encoder = nn.Embedding(config['ntoken'], config['emb_size'])
        self.bigru = nn.GRU(config['emb_size'], config['emb_size'], config['nlayers'],
                              dropout=config['dropout'], bidirectional=True, batch_first=True)
        self.nlayers = config['nlayers']
        self.nhid = config['emb_size']
        self.pooling = config['pooling']
        self.dictionary = config['dictionary']
        self.init_weights()
        self.encoder.weight.data[self.dictionary.word2idx['<unk>']] = 0

        if os.path.exists(config['word-vector']):
            print('Loading word vectors from', config['word-vector'])
            self.encoder.weight.data = torch.FloatTensor(config['embedding']['weight'])
            print('%d words from external word vectors loaded.' % config['embedding']['loaded_cnt'])
    def forward(self,inp,hidden):
        emb = self.drop(self.encoder(inp))
        outp, chidden = self.bigru(emb,hidden)
        if self.pooling == 'mean':
            outp = torch.mean(outp, 0).squeeze()
        elif self.pooling == 'max':
            outp = torch.max(outp, 0)[0].squeeze()
        elif self.pooling == 'all' or self.pooling == 'all-word':
            outp = torch.transpose(outp, 0, 1).contiguous()
        return outp,emb
    def init_hidden(self,bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_())
    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)
class BiSRU(nn.Module):
    def __init__(self,config):
        super(BiSRU,self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.encoder = nn.Embedding(config['ntoken'], config['emb_size'])
        self.bisru = SRU(config['emb_size'], config['emb_size'], config['nlayers'],
            dropout=config['dropout'], bidirectional=True)
        self.nlayers = config['nlayers']
        self.nhid = config['emb_size']
        self.pooling = config['pooling']
        self.dictionary = config['dictionary']
        self.init_weights()
        self.encoder.weight.data[self.dictionary.word2idx['<unk>']] = 0

        if os.path.exists(config['word-vector']):
            print('Loading word vectors from', config['word-vector'])
            self.encoder.weight.data = torch.FloatTensor(config['embedding']['weight'])
            print('%d words from external word vectors loaded.' % config['embedding']['loaded_cnt'])
    def forward(self,inp,hidden):
        emb = self.drop(self.encoder(inp))
        outp, chidden = self.bisru(emb, hidden)
        if self.pooling == 'mean':
            outp = torch.mean(outp, 0).squeeze()
        elif self.pooling == 'max':
            outp = torch.max(outp, 0)[0].squeeze()
        elif self.pooling == 'all' or self.pooling == 'all-word':
            outp = torch.transpose(outp, 0, 1).contiguous()
        return outp, emb
    def init_hidden(self,bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_())
    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)

class SelfAttentiveEncoder(nn.Module):
    def __init__(self, config):
        super(SelfAttentiveEncoder, self).__init__()
        self.rnn_type = config['rnn_type']
        if config['rnn_type'] == "lstm":
            self.rnn = BiLSTM(config)
        else:
            self.rnn = BiGRU(config)
        self.drop = nn.Dropout(config['dropout'])
        self.ws1 = nn.Linear(config['emb_size'] * 2, config['attention-unit'], bias=False)
        self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dictionary = config['dictionary']
        self.init_weights()
        self.attention_hops = config['attention-hops']

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        outp, emb = self.rnn.forward(inp, hidden)
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
        transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == self.dictionary.word2idx['<unk>']).float())
            # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas
    def init_hidden(self, bsz):
        return self.rnn.init_hidden(bsz)

# refer to 《Multiway Attention Networks for Modeling Sentence Pairs》
class Aggregation(nn.Module):
    def __init__(self,config):
        super(Aggregation,self).__init__()
        self.drop_out = nn.Dropout(config['dropout'])
        self.drop_att = nn.Dropout(config['dropout'])
        self.gru_agg = nn.GRU(input_size=config['emb_size'] * 2, hidden_size=config['labels'],
                                bidirectional=False)
        self.att_layer = nnlp.Attention(config['text_length'], attention_type='general')
        self.drop_out = nn.Dropout(config['drop_out'])
        self.drop_att = nn.Dropout(config['drop_att'])
        self.pred = nn.Linear(config['text_length'], config['class_number'])
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(config['dropout'])
        self.dictionary = config['dictionary']
        self.init_weights()
    def init_weights(self,init_range=0.1):
        self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)
    def forward(self, aggregation,attention,typename):
        aggregation = self.drop_out(aggregation)  # text_length*att_hops*hidden
        attention = self.drop_att(attention)  # text_length*att_hops*btz
        outp, chidden = self.gru_agg(aggregation)

        outputs, weights = self._packed_pad(outp, attention)
        outputs = self.pred(outputs)
        predict = weights.bmm(outputs)
        predict = self.tanh(self.drop(predict))
        predict = predict.permute([0, 2, 1])
        out = F.softmax(predict, dim=1)
        if typename == BiLSTM:
            attention = None
        return out, attention
    def _packed_pad(self, outp, attention):
        query = attention.permute([1, 2, 0])
        context = outp.permute([1, 2, 0])
        outputs, weights = self.att_layer(query, context)
        outputs = outputs.permute([1, 0, 2])
        weights = weights.permute([1, 2, 0])
        return outputs, weights

# refer to 《Reinforced Mnemonic Reader for Machine Reading Comprehension》
class StackedBRNN(nn.Module):
    """Stacked Bi-directional RNNs.
    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.
        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        if x_mask.data.sum() == 0 or x_mask.data.eq(1).long().sum(1).min() == 0:
            # No padding necessary.
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval.
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care.
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        """
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj
