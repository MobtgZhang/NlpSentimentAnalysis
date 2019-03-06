import torch
import torch.nn as nn
import torch.nn.functional as F
import model.layers as layers

class R_Net(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
    def __init__(self,args, normalize= True):
        super(R_Net,self).__init__()
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

        # Char rnn to generate char features
        self.char_rnn = layers.StackedBRNN(
            input_size=args.char_embedding_dim,
            hidden_size=args.char_hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=False,
        )

        doc_input_size = args.embedding_dim + args.char_hidden_size * 2

        # Encoder
        self.encode_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoder
        doc_hidden_size = 2 * args.hidden_size


        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers

        attn_hidden_size = 2 * doc_hidden_size

        # Self-matching-attention-baed RNN of the whole doc
        self.doc_self_attn = layers.SelfAttnMatch(doc_hidden_size, identity=False)

        self.doc_self_attn_gate = layers.Gate(attn_hidden_size)


        self.doc_self_attn_rnn = layers.StackedBRNN(
            input_size=attn_hidden_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        doc_self_attn_hidden_size = 2 * args.hidden_size

        self.doc_self_attn_rnn2 = layers.StackedBRNN(
            input_size=doc_self_attn_hidden_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )
        # Multimatch layer
        self.match_net = layers.MatchNetwork(
            doc_attn_hidden_size=doc_self_attn_hidden_size,
            hidden_size=args.hidden_size,
            first_grained_size = args.first_grained_size,
            second_grained_size = args.second_grained_size,
            class_size = args.class_size,
            dropout_rate=args.dropout_rnn,
            normalize=normalize
        )
    def forward(self,x,x_c,x_f,x_mask):
        """Inputs:
                x = document word indices             [batch * len_d]
                x_c = document char indices           [batch * len_d]
                x_f = document word features indices  [batch * len_d * nfeat]
                x_mask = document padding mask        [batch * len_d]
                """
        # Embed both document and question
        x_emb = self.embedding(x)
        x_c_emb = self.char_embedding(x_c)
        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x_emb = F.dropout(x_emb, p=self.args.dropout_emb, training=self.training)
            x_c_emb = F.dropout(x_c_emb, p=self.args.dropout_emb, training=self.training)
        # Generate char features
        x_c_features = self.char_rnn(
            x_c_emb.reshape((x_c_emb.size(0) * x_c_emb.size(1), x_c_emb.size(2), x_c_emb.size(3))),
            x_mask.unsqueeze(2).repeat(1, 1, x_c_emb.size(2)).reshape(
                (x_c_emb.size(0) * x_c_emb.size(1), x_c_emb.size(2)))
        ).reshape((x_c_emb.size(0), x_c_emb.size(1), x_c_emb.size(2), -1))[:, :, -1, :]
        # Combine input
        crnn_input = [x_emb, x_c_features]  # embedding_dim + char_hidden_size * 2
        # Encode document with RNN
        c = self.encode_rnn(torch.cat(crnn_input, 2), x_mask) # hidden_size*2*doc_layers
        # Match documents to themselves
        doc_self_attn_hiddens = self.doc_self_attn(c, x_mask) # hidden_size*2*doc_layers

        rnn_input = self.doc_self_attn_gate(torch.cat([c,doc_self_attn_hiddens], 2))
        c = self.doc_self_attn_rnn(rnn_input, x_mask)
        c = self.doc_self_attn_rnn2(c, x_mask)

        # Predict
        score = self.match_net(c,x_mask)

        return score









