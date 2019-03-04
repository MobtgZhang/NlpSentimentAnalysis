import torch
import torch.nn as nn
from torch.autograd import Variable
# highway block
class RecurrentHighway(nn.Module):
    def __init__(self, input_size, hidden_size, recurrence_length, vocab_size):
        super(RecurrentHighway).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.L = recurrence_length
        self.highways = nn.ModuleList()
        self.highways.append(HighwayBlock(self.input_size, self.hidden_size, first_layer=True))
        for _ in range(self.L - 1):
            self.highways.append(HighwayBlock(self.input_size, self.hidden_size, first_layer=False))

        self.out_embedding = nn.Linear(self.hidden_size, vocab_size)

    def init_state(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(bsz, self.hidden_size).zero_())

    def forward(self, _input, hidden=None):
        batch_size = _input.size(0)
        max_time = _input.size(1)
        if hidden is None:
            hidden = self.init_state(batch_size)
        embed_batch = _input
        # Loop over all times steps
        layer_outputs = []
        for step in range(max_time):
            # Loop over L times for each time step
            for tick in range(self.L):
                hidden = self.highways[tick](embed_batch[:, step, :], hidden)
                out = self.out_embedding(hidden)
                layer_outputs.append(out)

        return torch.cat(layer_outputs)


# Highway block for each recurrent 'tick'
class HighwayBlock(nn.Module):
    def __init__(self, input_size, hidden_size, first_layer):
        super(HighwayBlock).__init__()
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