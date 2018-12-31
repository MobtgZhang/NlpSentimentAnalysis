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
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from utils import LoadVocabs
from config import config
from model import SentimentNet

from test_preparmodel import GetVocabs,sepData
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
def prepare_vocab(Vocabs):
    # Make a dictionary
    idx_to_word,word_to_idx = MakeSets(Vocabs)
    return idx_to_word,word_to_idx
def prepare_train_dataset(sentences_train,labels_train,word_to_idx):
    print("Preparing train datasets ... ...")
    train_features = torch.LongTensor(pad_samples(encode_samples(sentences_train,word_to_idx),config.text_length))
    train_labels = torch.FloatTensor(labels_train)
    print("Train datasets has been Loaded!")
    return train_features,train_labels
def prepare_validate_dataset(sentences_validate,labels_validate,word_to_idx):
    print("Preparing test datasets ... ...")
    validate_features = torch.LongTensor(pad_samples(encode_samples(sentences_validate,word_to_idx)))
    validate_labels = torch.FloatTensor(labels_validate)
    print("Test datasets has been Loaded!")
    return validate_features,validate_labels
def prepare_embedding(vocab_size,word_to_idx,idx_to_word):
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
    return weight
def prepare_train(train_features,train_labels,validate_features,validate_labels,weight,word_to_idx,idx_to_word,vocab_size,model_save_file,
                    num_epochs,batch_size,labels,learning_rate,num_hiddens = 100,num_layers = 2,bidirectional = True):
    device = torch.device(config.device)
    use_gpu = torch.cuda.is_available()
    net = SentimentNet(vocab_size=(vocab_size+1), embed_size=config.word_dim,num_hiddens=num_hiddens, num_layers=num_layers,
                   bidirectional=bidirectional, weight=weight,word_to_idx = word_to_idx,idx_to_word = idx_to_word,labels=labels, use_gpu=use_gpu)
    net.to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    validate_set = torch.utils.data.TensorDataset(validate_features, validate_labels)
    validate_iter = torch.utils.data.DataLoader(validate_set, batch_size=batch_size,shuffle=False)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True)
    train_loss_list = []
    validate_loss_list = []
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, validate_losses = 0, 0
        train_acc, validate_acc = 0, 0
        n, m = 0, 0
        for feature, label in train_iter:
            n += 1
            net.zero_grad()
            feature = Variable(feature.cuda())
            label = Variable(label.cuda())
            score = net(feature)
            loss = loss_function(score, label)
            loss.backward()
            optimizer.step()
            train_acc += accuracy_score(torch.argmax(score.cpu().data,dim=1),torch.argmax(label.cpu().data,dim=1))
            train_loss += loss
            train_loss_list.append(loss.cpu().data.numpy().tolist())
        with torch.no_grad():
            for validate_feature, validate_label in validate_iter:
                m += 1
                validate_feature = validate_feature.cuda()
                validate_label = validate_label.cuda()
                validate_score = net(validate_feature)
                validate_loss = loss_function(validate_score, validate_label)
                validate_acc += accuracy_score(torch.argmax(validate_score.cpu().data,dim=1), torch.argmax(validate_label.cpu().data,dim=1))
                validate_losses += validate_loss
                validate_loss_list.append(loss.cpu().data.numpy().tolist())
        end = time.time()
        runtime = end - start
        print('epoch: %d, train loss: %.4f, train acc: %.2f, validate loss: %.4f, tevalidate acc: %.2f, time: %.2f' %
                (epoch, train_loss.data / n, train_acc / n, validate_losses.data / m, validate_acc / m, runtime))
    # 保存整个网络和参数
    torch.save(net,model_save_file)
    return train_loss_list,validate_loss_list
def load_train_dataset(filename):
    data = np.load(filename)
    sentences = list(data['sentences'])
    labels = data['label_list']
    return sentences,labels
def load_validate_dataset(filename):
    data = np.load(filename)
    sentences = list(data['sentences'])
    labels = data['label_list']
    return sentences,labels
def train_entry():
    # combine three vocabulary
    Vocabs = set()
    # Make a dictionary
    Vocabs_train = LoadVocabs(config.train_vocab_file)
    Vocabs_validate = LoadVocabs(config.validate_vocab_file)
    Vocabs.update(Vocabs_train)
    Vocabs.update(Vocabs_validate)
    # Make a dictionary
    vocab_size = len(Vocabs)
    idx_to_word,word_to_idx = prepare_vocab(Vocabs)
    # To bulid the datatsets
    sentences_train,labels_train,sentences_test,labels_test = sepData(filename)
    train_features,train_labels = load_train_dataset(config.train_npz)
    validate_features,validate_labels = prepare_validate_dataset(config.validate_npz)
    # To make the embeddings
    weight = prepare_embedding(vocab_size,word_to_idx,idx_to_word)
    # training the model
    train_loss_list,validate_loss_list = prepare_train(train_features,train_labels,validate_features,validate_labels,weight,word_to_idx,idx_to_word,vocab_size,config.model_save_file,
                    config.num_epochs,config.batch_size,config.labels,config.learning_rate)
    # draw pictures
    x1 = np.linspace(0,len(train_loss_list)-1,len(train_loss_list))
    x2 = np.linspace(0,len(validate_loss_list)-1,len(validate_loss_list))
    plt.plot(x1,train_loss_list)
    plt.plot(x2,validate_loss_list)
    plt.savefig(config.picture_save_file)
    plt.show()
def test_entry():
    pass
def main(_):
    # Make a dictionary
    filename = "/home/asus/AI_Challenger2018/NewCode5/testData/910b77ca580b4afd.csv.npz"
    Vocabs = GetVocabs(filename)
    vocab_size = len(Vocabs)
    idx_to_word,word_to_idx = prepare_vocab(Vocabs)
    # To bulid the datatsets
    sentences_train,labels_train,sentences_test,labels_test = sepData(filename)
    train_features,train_labels = prepare_train_dataset(sentences_train,labels_train,word_to_idx)
    validate_features,validate_labels = prepare_validate_dataset(sentences_test,labels_test,word_to_idx)
    # To make the embeddings
    weight = prepare_embedding(vocab_size,word_to_idx,idx_to_word)
    # Preparing the model
    num_epochs = 5
    num_hiddens = 100
    num_layers = 2
    bidirectional = True
    batch_size = 50
    labels = 20
    learning_rate = 0.8
    model_save_file = "model.pt"
    train_loss_list,validate_loss_list = prepare_train(train_features,train_labels,validate_features,validate_labels,weight,vocab_size,model_save_file,
                    num_epochs,batch_size,labels,learning_rate,num_hiddens,num_layers,bidirectional)
    # draw pictures
    x1 = np.linspace(0,len(train_loss_list)-1,len(train_loss_list))
    x2 = np.linspace(0,len(validate_loss_list)-1,len(validate_loss_list))
    plt.plot(x1,train_loss_list)
    plt.plot(x2,validate_loss_list)
    plt.savefig(config.picture_save_file)
    plt.show()
if __name__ == "__main__":
    app.run(main)
