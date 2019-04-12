import torch
import torch.nn as nn
import torch.nn.functional as F
import model.layers as layers
class RnnDocReader(nn.Module):
    """
    RnnDocReader model for multilayers sentiment analaysis.
    """
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
    def __init__(self,args, normalize= True):
        super(RnnDocReader,self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)
        # Char embeddings (+1 for padding)
        self.char_embedding = nn.Embedding(args.char_size,
                                           args.char_embedding_dim,
                                           padding_idx=0)

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = args.embedding_dim + args.num_features

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers

        # document merging
        self.self_attn = layers.LinearSeqAttn(input_size=doc_hidden_size,
                                              hidden_size=args.hidden_size)
        # Multimatch layer
        self.match_net = layers.MatchNetwork(
                doc_attn_hidden_size=doc_hidden_size,
                hidden_size=args.hidden_size,
                first_grained_size=args.first_grained_size,
                second_grained_size=args.second_grained_size,
                class_size=args.class_size,
                dropout_rate=args.dropout_rnn,
                normalize=normalize
            )
    def forward(self, x, x_c, x_f, x_mask):
        """Inputs:
        x = document word indices             [batch * len_d]
        x_f = document word features indices  [batch * len_d * nfeat]
        x_mask = document padding mask        [batch * len_d]
        """
        # Embed both document
        x_emb = self.embedding(x)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x_emb = F.dropout(x_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Form document encoding inputs
        drnn_input = [x_emb]
        # Add manual features
        if self.args.num_features > 0:
            drnn_input.append(x_f)

        # Encode document with RNN
        doc_hidden = self.doc_rnn(torch.cat(drnn_input, 2), x_mask)
        doc_out = self.self_attn(doc_hidden)
        score = self.match_net(doc_out,x_mask)
        return score