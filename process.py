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
import os
from utils import GetVocabs,MakeSets,encode_samples,pad_samples,prepare_vocab
from config import config
from model import SentimentNet

from test_preparmodel import sepData
def prepare_datasets(sentences,labels,word_to_idx,mode):
    print("Preparing "+mode+" datasets ... ...")
    train_features = torch.LongTensor(pad_samples(encode_samples(sentences,word_to_idx),config.text_length))
    train_labels = torch.FloatTensor(labels)
    print("The "+mode+" datasets has been loaded!")
    return train_features,train_labels
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
def prepare_train(train_features,train_labels,validate_features,validate_labels,weight,net,
        vocab_size,model_save_file,num_epochs,batch_size,learning_rate):
    device = torch.device(config.device)
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
        print('epoch: %d, train loss: %.4f, train acc: %.2f, validate loss: %.4f, validate acc: %.2f, time: %.2f' %
                (epoch, train_loss.data / n, train_acc / n, validate_losses.data / m, validate_acc / m, runtime))
    # 保存整个网络和参数
    torch.save(net,model_save_file)
    return train_loss_list,validate_loss_list
def load_datasets(filename):
    data = np.load(filename)
    sentences = list(data['sentences'])
    labels = data['labels']
    return sentences,labels
def train_entry(modelname):
    if os.path.exists(config.model_save_file):
        print("model is exisits!")
    
    # combine three vocabulary
    Vocabs = set()
    # Make a dictionary
    print("Loading vocabulary ...")
    Vocabs_train = GetVocabs(config.train_npz)
    Vocabs_validate = GetVocabs(config.validate_npz)
    Vocabs.update(Vocabs_train)
    Vocabs.update(Vocabs_validate)
    vocab_size = len(Vocabs)
    idx_to_word,word_to_idx = prepare_vocab(Vocabs)
    print("Vocabulary loaded !")
    # To bulid the datatsets
    # Train datasets
    print("Preparing datasets ...")
    sentences_train,labels_train = load_datasets(config.train_npz)
    train_features,train_labels = prepare_datasets(sentences_train,labels_train,word_to_idx,"train")
    
    sentences_validate,labels_validate = load_datasets(config.validate_npz)
    validate_features,validate_labels = prepare_datasets(sentences_validate,labels_validate,word_to_idx,"validate")

    # To make the embeddings
    weight = prepare_embedding(vocab_size,word_to_idx,idx_to_word)
    # training the model
    if not os.path.exisits(config.save_statics_file):
        os.mkdir(config.save_statics_file)
    if modelname == "SentimentNet":
        net = SentimentNet(vocab_size=(vocab_size+1), embed_size=config.word_dim,
            weight=weight,word_to_idx = word_to_idx,idx_to_word = idx_to_word,labels=config.labels, use_gpu=config.use_gpu)
    
    train_loss_list,validate_loss_list = prepare_train(train_features,train_labels,validate_features,validate_labels,weight,net,
        vocab_size,config.model_save_file,config.num_epochs,config.batch_size,config.learning_rate)
    # draw pictures
    x1 = np.linspace(0,len(train_loss_list)-1,len(train_loss_list))
    plt.plot(x1,train_loss_list)
    plt.title('Train loss',fontsize=12)
    num = 0
    while True:
        if not os.path.exists(config.pic_trainloss_savefile):
            plt.savefig(config.pic_trainloss_savefile)
        else:
            filename = config.pic_trainloss_savefile + str(num)
            plt.savefig(filename)
            num += 1
    plt.show()

    x2 = np.linspace(0,len(validate_loss_list)-1,len(validate_loss_list))
    plt.plot(x2,validate_loss_list)
    plt.title('validate loss',fontsize=12)
    while True:
        if not os.path.exists(config.pic_validateloss_savefile):
            plt.savefig(config.pic_validateloss_savefile)
        else:
            filename = config.pic_validateloss_savefile + str(num)
            plt.savefig(filename)
            num += 1
    plt.show()
def test_entry():
    # loading model
    net = torch.load(config.model_save_file)
    word_to_idx = net.word_to_idx
    # preparing test datasets
    sentences_test,labels_test = load_datasets(config.test_npz)
    test_features,test_labels = prepare_datasets(sentences_test,labels_test,word_to_idx,"test")

    test_set = torch.utils.data.TensorDataset(test_features, test_labels)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size,shuffle=False)
    start = time.time()
    test_losses = 0
    test_acc = 0
    m = 0
    test_loss_list = []
    loss_function = nn.MSELoss()
    with torch.no_grad():
        for test_feature, test_label in test_iter:
            m += 1
            test_feature = test_feature.cuda()
            test_label = test_label.cuda()
            test_score = net(test_feature)
            test_loss = loss_function(test_score, test_label)
            test_acc += accuracy_score(torch.argmax(test_score.cpu().data,dim=1), torch.argmax(test_label.cpu().data,dim=1))
            test_losses += validate_loss
            test_loss_list.append(loss.cpu().data.numpy().tolist())
    # draw pictures
    x = np.linspace(0,len(test_loss_list)-1,len(test_loss_list))
    plt.plot(x,train_loss_list,label = "train loss")
    num = 0
    while True:
        if not os.path.exists(config.pic_testloss_savefile):
            plt.savefig(config.pic_testloss_savefile)
        else:
            filename = config.pic_testloss_savefile + str(num)
            plt.savefig(filename)
            num += 1
    plt.show()
