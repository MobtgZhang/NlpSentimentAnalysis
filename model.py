import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import gensim
from absl import app
import time

from utils import LoadVocabs
from config import config
# Making token datasets
def MakeSets(Vocabs):
    word_to_idx = {word:k+1 for k,word in enumerate(Vocabs)}
    word_to_idx['<unk>'] = 0
    idx_to_word = {k+1:word for k,word in enumerate(Vocabs)}
    idx_to_word[0] = '<unk>'
    return idx_to_word,word_to_idx
# Encoding the tokens
def encode_samples(tokenized_samples,word_to_idx):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features
# Encoding the features
def pad_samples(features, maxlen=1000, PAD=0):
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while(len(padded_feature) < maxlen):
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features

class SentimentNet(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels, use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0.1)
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 2, labels)
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)
        return outputs
def main(_):
    # Make a dictionary
    Vocabs = LoadVocabs(config.vocab_file)
    idx_to_word,word_to_idx = MakeSets(Vocabs)
    vocab_size = len(Vocabs)
    # To bulid the datatsets
    datasets = np.load(config.train_npz)
    train_tokenized = list(datasets['sentences'][()])
    labels = datasets['labels'].tolist()
    print("Preparing datasets ... ...")
    train_features = torch.tensor(pad_samples(encode_samples(train_tokenized,word_to_idx),config.text_length))
    train_labels = torch.tensor(labels)
    # test_features = torch.tensor(pad_samples(encode_samples(test_tokenized, vocab)))
    # test_labels = torch.tensor([score for _, score in test_data])
    print("Datasets has been Loaded!")
    # To make the embeddings
    print("Loading word2vecs ... ...")
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(config.wordembedding_file,binary=False, encoding='utf-8')
    print("Word2vecs has been Loaded!")
    weight = torch.zeros(vocab_size + 1,config.word_dim)
    for i in range(len(wvmodel.index2word)):
        try:
            index = word_to_idx[wvmodel.index2word[i]]
        except:
            continue
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(idx_to_word[word_to_idx[wvmodel.index2word[i]]]))
    # Preparing the model
    num_epochs = 5
    num_hiddens = 100
    num_layers = 2
    bidirectional = True
    batch_size = 64
    labels = 2
    lr = 0.8
    device = torch.device(config.device)
    use_gpu = torch.cuda.is_available()

    net = SentimentNet(vocab_size=(vocab_size+1), embed_size=config.word_dim,num_hiddens=num_hiddens, num_layers=num_layers,
                   bidirectional=bidirectional, weight=weight,labels=labels, use_gpu=use_gpu)
    net.to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    # test_set = torch.utils.data.TensorDataset(test_features, test_labels)
    # test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True)
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, test_losses = 0, 0
        train_acc, test_acc = 0, 0
        n, m = 0, 0
        exit()
        for feature, label in train_iter:
            n += 1
            net.zero_grad()
            feature = Variable(feature.cuda())
            label = Variable(label.cuda())
            score = net(feature)
            loss = loss_function(score, label)
            loss.backward()
            optimizer.step()
            train_acc += accuracy_score(torch.argmax(score.cpu().data,dim=1), label.cpu())
            train_loss += loss
        '''
        with torch.no_grad():
            for test_feature, test_label in test_iter:
                m += 1
                test_feature = test_feature.cuda()
                test_label = test_label.cuda()
                test_score = net(test_feature)
                test_loss = loss_function(test_score, test_label)
                test_acc += accuracy_score(torch.argmax(test_score.cpu().data,
                                                    dim=1), test_label.cpu())
                test_losses += test_loss
        '''
        end = time.time()
        runtime = end - start
        # print('epoch: %d, train loss: %.4f, train acc: %.2f, test loss: %.4f, test acc: %.2f, time: %.2f' %
        #  (epoch, train_loss.data / n, train_acc / n, test_losses.data / m, test_acc / m, runtime))
        print('epoch: %d, train loss: %.4f, train acc: %.2f, time: %.2f' %(epoch, train_loss.data / n, train_acc / n, runtime))
        exit()
if __name__ == "__main__":
    app.run(main)
