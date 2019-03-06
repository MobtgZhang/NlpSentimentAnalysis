import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------


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

class SelfAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * x_j) for i in X
    * alpha_j = softmax(x_j * x_i)
    """

    def __init__(self, input_size, identity=False, diag=True):
        super(SelfAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None
        self.diag = diag

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len1 * dim1
            x_mask: batch * len1 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * dim1
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
        else:
            x_proj = x

        # Compute scores
        scores = x_proj.bmm(x_proj.transpose(2, 1))
        if not self.diag:
            x_len = x.size(1)
            for i in range(x_len):
                scores[:, i, i] = 0

        # Mask padding
        x_mask = x_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(x_mask.data, -float('inf'))

        # Normalize with softmax
        alpha = F.softmax(scores, dim=2)

        # Take weighted average
        matched_seq = alpha.bmm(x)
        return matched_seq

# ------------------------------------------------------------------------------
# Functional Units
# ------------------------------------------------------------------------------

class Gate(nn.Module):
    """Gate Unit
    g = sigmoid(Wx)
    x = g * x
    """
    def __init__(self, input_size):
        super(Gate, self).__init__()
        self.linear = nn.Linear(input_size, input_size, bias=False)

    def forward(self, x):
        """
        Args:
            x: batch * len * dim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            res: batch * len * dim
        """
        x_proj = self.linear(x)
        gate = torch.sigmoid(x)
        return x_proj * gate


class MatchNetwork(nn.Module):
    def __init__(self,
                 doc_attn_hidden_size, hidden_size,
                 first_grained_size,second_grained_size ,class_size,
                 dropout_rate=0,
                 normalize=True):
        super(MatchNetwork, self).__init__()
        self.normalize = normalize
        self.doc_attn_hidden_size = doc_attn_hidden_size
        self.hidden_size = hidden_size
        self.first_grained_size = first_grained_size
        self.second_grained_size = second_grained_size
        self.class_size = class_size
        self.dp = dropout_rate

        self.self_attn = NonLinearSeqAttn(doc_attn_hidden_size, hidden_size,)
        grained_hidden_size = doc_attn_hidden_size*2

        # Coarse granularity layer
        self.fine_layer1 = MultiMatch(in_size = grained_hidden_size,
                                      hidden_size =hidden_size,
                                      label_size = first_grained_size,
                                      training=self.training,
                                      dropout_rate = dropout_rate)
        self.fine_layer2 = MultiMatch(in_size = grained_hidden_size,
                                      hidden_size =hidden_size,
                                      label_size = second_grained_size,
                                      training=self.training,
                                      dropout_rate = dropout_rate)

        self.weights = MultiMatch(in_size=first_grained_size,
                                  hidden_size= hidden_size,
                                  label_size=class_size,
                                  training=self.training,
                                  dropout_rate = dropout_rate)

    def init_hiddens(self, y, y_mask):
        attn = self.self_attn(y, y_mask)
        res = attn.unsqueeze(1).bmm(y).squeeze(1)  # [B, I]
        return res
    def forward(self, x,x_mask):
        hiddens = self.init_hiddens(x, x_mask)
        x_ = torch.cat([x, hiddens.unsqueeze(1).repeat(1, x.size(1), 1)], 2)  # bsz*length*(x_size*2)
        f_c = self.fine_layer1(x_)
        s_c = self.fine_layer2(x_)

        o_c = s_c.bmm(f_c.transpose(2, 1))
        rp = self.weights(f_c.transpose(2, 1))
        weight = rp.bmm(f_c.transpose(2, 1))

        predict = weight.bmm(o_c.transpose(2, 1))
        score = F.log_softmax(predict,dim=1)
        return score

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
class NonLinearSeqAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(function(Wx_i)) for x_i in X.
    """

    def __init__(self, input_size, hidden_size):
        super(NonLinearSeqAttn, self).__init__()
        self.FFN = FeedForwardNetwork(input_size, hidden_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * dim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        scores = self.FFN(x).squeeze(2)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores,dim=1)
        return alpha
class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size,hidden_size):
        super(LinearSeqAttn, self).__init__()
        self.att_in = nn.GRU(input_size=input_size,hidden_size=hidden_size,bidirectional=True,batch_first=True)
        self.linear = nn.Linear(2*hidden_size, input_size)
    def forward(self, x):
        """
        Args:
            x: batch * len * hdim
        Output:
            alpha: batch * len * hdim
        """
        out,c = self.att_in(x)
        alpha = self.linear(out)
        return alpha

class MultiMatch(nn.Module):
    def __init__(self,in_size,hidden_size,label_size,
                 training= True,
                 dropout_rate=0):
        super(MultiMatch,self).__init__()
        self.label_size = label_size
        self.in_size = in_size
        self.dp = dropout_rate
        self.training = training
        self.linear = nn.Linear(in_size,hidden_size, bias=False)
        self.weights = nn.Linear(hidden_size,label_size, bias=False)

    def forward(self, inpx):
        s0 = torch.tanh(self.linear(inpx))# bsz*length*hidden_size
        sj = self.weights(s0).transpose(2, 1)
        rp = F.softmax(sj, 2).bmm(s0)
        rp = F.dropout(rp, p=self.dp, training=self.training)
        return rp

class RecurrentHighway(nn.Module):
    def __init__(self, input_size,hidden_size,layers):
        super(RecurrentHighway,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.L = layers
        self.highways = nn.ModuleList()
        self.highways.append(HighwayBlock(self.input_size, self.hidden_size, first_layer=True))
        for _ in range(self.L - 1):
            self.highways.append(HighwayBlock(self.input_size, self.hidden_size, first_layer=False))
        self.out_embedding = nn.Linear(hidden_size,hidden_size)

    def init_state(self, batch_size):
        hidden = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
        return hidden

    def forward(self, embed_batch, hidden=None):
        batch_size = embed_batch.size(0)
        max_time = embed_batch.size(1)
        if hidden is None:
            hidden = self.init_state(batch_size)
        # Loop over all times steps
        layer_outputs = []
        for step in range(max_time):
            # Loop over L times for each time step
            layer_list = []
            for tick in range(self.L):
                hidden = self.highways[tick](embed_batch[:, step, :], hidden)
                out = self.out_embedding(hidden)
                layer_list.append(out.unsqueeze(1))
            out = torch.cat(layer_list,2)
            layer_outputs.append(out)
        return torch.cat(layer_outputs,1)
# Highway block for each recurrent 'tick'
class HighwayBlock(nn.Module):
    def __init__(self, input_size, hidden_size, first_layer):
        super(HighwayBlock,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.first_layer = first_layer

        # input weight matrices
        if self.first_layer is True:
            self.W_H = nn.Linear(input_size, hidden_size)
            self.W_T = nn.Linear(input_size, hidden_size)
            self.W_C = nn.Linear(input_size, hidden_size)

        # hidden weight matrices
        self.R_H = nn.Linear(hidden_size, hidden_size)
        self.R_T = nn.Linear(hidden_size, hidden_size)
        self.R_C = nn.Linear(hidden_size, hidden_size)


    def forward(self, _input, prev_hidden):
        if self.first_layer:
            hl = self.W_H(_input) + self.R_H(prev_hidden)
            tl = self.W_C(_input) + self.R_C(prev_hidden)
            cl = self.W_T(_input) + self.R_T(prev_hidden)
        else:
            hl = self.R_H(prev_hidden)
            tl = self.R_C(prev_hidden)
            cl = self.R_T(prev_hidden)

        # Core recurrence operation
        _hidden = (hl * tl) + (prev_hidden * cl)
        return _hidden
class SFU(nn.Module):
    """Semantic Fusion Unit
    The ouput vector is expected to not only retrieve correlative information from fusion vectors,
    but also retain partly unchange as the input vector
    """
    def __init__(self, input_size, fusion_size):
        super(SFU, self).__init__()
        self.linear_r = nn.Linear(input_size + fusion_size, input_size)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size)

    def forward(self, x, fusions):
        r_f = torch.cat([x, fusions], 2)
        r = torch.tanh(self.linear_r(r_f))
        g = torch.sigmoid(self.linear_g(r_f))
        o = g * r + (1-g) * x
        return o
