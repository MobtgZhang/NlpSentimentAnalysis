import torch
import torch.nn as nn
import torch.nn.functional as F
import model.layers as layers


class MnemonicReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
    def __init__(self,args, normalize= True):
        super(MnemonicReader,self).__init__()
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

        doc_input_size = args.embedding_dim + args.num_features

        # Encoder
        self.encoding_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )
        doc_hidden_size = args.hidden_size*args.doc_layers
        sfu_hidden_size = doc_hidden_size + args.char_hidden_size*2
        # Interactive aligning, self aligning and aggregating
        self.interactive_aligners = nn.ModuleList()
        self.interactive_SFUs = nn.ModuleList()
        self.self_aligners = nn.ModuleList()
        self.self_SFUs = nn.ModuleList()
        self.aggregate_rnns = nn.ModuleList()
        for i in range(args.hop):
            # interactive aligner
            self.interactive_SFUs.append(layers.SFU(doc_hidden_size, sfu_hidden_size))
            # self aligner
            self.self_aligners.append(layers.SelfAttnMatch(doc_hidden_size, identity=True, diag=False))
            self.self_SFUs.append(layers.SFU(doc_hidden_size, 3 * doc_hidden_size))
            # aggregating
            self.aggregate_rnns.append(
                layers.StackedBRNN(
                    input_size=doc_hidden_size,
                    hidden_size=args.hidden_size,
                    num_layers=1,
                    dropout_rate=args.dropout_rnn,
                    dropout_output=args.dropout_rnn_output,
                    concat_layers=False,
                    rnn_type=self.RNN_TYPES[args.rnn_type],
                    padding=args.rnn_padding,
                )
            )
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

        # Generate features hidden

        # Combine input
        crnn_input = [x_emb]
        # Add manual features
        if self.args.num_features > 0:
            crnn_input.append(x_f)
            crnn_input = torch.cat(crnn_input, 2)
        else:
            crnn_input = x_emb

        # Encode document with RNN
        document = self.encoding_rnn(crnn_input, x_mask)
        # Align and aggregate
        c_check = document
        for i in range(self.args.hop):
            #print(c_check.size(), x_c_features.size())
            # 300 200
            c_bar = self.interactive_SFUs[i].forward(c_check,torch.cat([c_check,x_c_features],2))
            c_tilde = self.self_aligners[i].forward(c_bar, x_mask)
            c_hat = self.self_SFUs[i].forward(c_bar, torch.cat([c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
            c_check = self.aggregate_rnns[i].forward(c_hat, x_mask)

        score = self.match_net(c_check,x_mask)
        return score