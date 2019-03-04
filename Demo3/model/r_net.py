import torch
import torch.nn as nn

class R_Net(nn.Module):
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
    def forward(self, *input):
        pass