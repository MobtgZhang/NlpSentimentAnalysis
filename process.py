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
import math
from tqdm import tqdm
from utils import GetVocabs,MakeSets,encode_samples,pad_samples,prepare_vocab,prepare_labels
from config import config
from model import BiLSTMNet,textCNN,BiGRUNet,EMA,MANNet

from test_preparmodel import sepData
def prepare_datasets(sentences,labels,word_to_idx,mode):
    print("Preparing "+mode+" datasets ... ...")
    features = torch.LongTensor(pad_samples(encode_samples(sentences,word_to_idx),config.text_length))
    scaler = prepare_labels(labels)
    labels = torch.FloatTensor(scaler.transform(labels))
    print("The "+mode+" datasets has been loaded!")
    return features,labels,scaler
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
def prepare_train(train_features,train_labels,validate_features,validate_labels,train_scaler,validate_scaler,weight,net,
        vocab_size,model_save_file,num_epochs,batch_size,learning_rate):
    device = torch.device(config.device)
    net.to(device)
    loss_function = nn.MSELoss()

    base_lr = 1.0
    warm_up = config.lr_warm_up_num

    ema = EMA(config.ema_decay)
    for name, p in net.named_parameters():
        if p.requires_grad: ema.set(name, p)
    params = filter(lambda param: param.requires_grad, net.parameters())
    optimizer = optim.Adam(lr=base_lr, betas=(config.beta1, config.beta2), eps=1e-7, weight_decay=3e-7, params=params)
    cr = learning_rate / math.log2(warm_up)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: cr * math.log2(ee + 1) if ee < warm_up else learning_rate)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    validate_set = torch.utils.data.TensorDataset(validate_features, validate_labels)
    validate_iter = torch.utils.data.DataLoader(validate_set, batch_size=batch_size,shuffle=False)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True)
    train_loss_list = []
    validate_loss_list = []
    for epoch in tqdm(range(num_epochs),"epoches: "):
        start = time.time()
        train_loss, validate_losses = 0, 0
        train_acc, validate_acc = 0, 0
        n, m = 0, 0
        for feature, label in tqdm(train_iter,"train: "):
            n += 1
            net.zero_grad()
            feature = Variable(feature.cuda())
            label = Variable(label.cuda())
            score = net(feature)
            loss = loss_function(score, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            for name,p in net.named_parameters():
                if p.requires_grad:ema.update_parameter(name,p)
            torch.nn.utils.clip_grad_norm_(net.parameters(),config.grad_clip)
            t_score = train_scaler.inverse_transform(score.cpu().data.numpy())
            t_score = np.squeeze(np.ceil(t_score.reshape(1,-1)))
            t_label = train_scaler.inverse_transform(label.cpu().data.numpy())
            t_label = np.squeeze(t_label.reshape(1,-1))

            train_acc += accuracy_score(t_label,t_score)
            train_loss += loss
            train_loss_list.append(loss.cpu().data.numpy().tolist())
        with torch.no_grad():
            for validate_feature, validate_label in tqdm(validate_iter,"validate: "):
                m += 1
                validate_feature = validate_feature.cuda()
                validate_label = validate_label.cuda()
                validate_score = net(validate_feature)
                validate_loss = loss_function(validate_score, validate_label)
                v_score = validate_scaler.inverse_transform(validate_score.cpu().data.numpy())
                v_label = validate_scaler.inverse_transform(validate_label.cpu().data.numpy())

                v_score = validate_scaler.inverse_transform(score.cpu().data.numpy())
                v_score = np.squeeze(np.ceil(v_score.reshape(1,-1)))
                v_label = validate_scaler.inverse_transform(label.cpu().data.numpy())
                v_label = np.squeeze(v_label.reshape(1,-1))

                validate_acc += accuracy_score(v_label,v_score)
                validate_losses += validate_loss
                validate_loss_list.append(validate_loss.cpu().data.numpy().tolist())
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
        print("model is existed!")
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
    train_features,train_labels,train_scaler = prepare_datasets(sentences_train,labels_train,word_to_idx,"train")
    
    sentences_validate,labels_validate = load_datasets(config.validate_npz)
    validate_features,validate_labels,validate_scaler = prepare_datasets(sentences_validate,labels_validate,word_to_idx,"validate")

    # To make the embeddings
    weight = prepare_embedding(vocab_size,word_to_idx,idx_to_word)
    # training the model
    if not os.path.exists(config.save_statics_file):
        os.mkdir(config.save_statics_file)
    if modelname == "BiLSTMNet":
        net = BiLSTMNet(vocab_size=(vocab_size+1), embed_size=config.word_dim,
            weight=weight,word_to_idx = word_to_idx,idx_to_word = idx_to_word,labels=config.labels)
    elif modelname == "textCNN":
        net = textCNN(vocab_size, embed_size = config.word_dim, seq_len = config.text_length, labels= config.labels, 
                weight= weight,word_to_idx = word_to_idx,idx_to_word= idx_to_word)
    elif modelname == "BiGRUNet":
        net = BiGRUNet(vocab_size, embed_size = config.word_dim,labels= config.labels, 
                weight= weight,word_to_idx = word_to_idx,idx_to_word= idx_to_word)
    elif modelname == "MANNet":
        net = MANNet(vocab_size, embed_size = config.word_dim,encoder_size = 600,labels= config.labels, 
                weight= weight,word_to_idx = word_to_idx,idx_to_word= idx_to_word)
    else:
        raise Exception("unknown model")
    train_loss_list,validate_loss_list = prepare_train(train_features,train_labels,validate_features,validate_labels,train_scaler,validate_scaler,weight,net,
        vocab_size,config.model_save_file,config.num_epochs,config.batch_size,config.learning_rate)
    # draw pictures
    x1 = np.linspace(0,len(train_loss_list)-1,len(train_loss_list))
    plt.plot(x1,train_loss_list)
    plt.title('Train loss',fontsize=12)
    num = 0
    while True:
        if not os.path.exists(config.pic_trainloss_savefile):
            plt.savefig(config.pic_trainloss_savefile)
            break
        else:
            (filepath,filename) = os.path.split(config.pic_trainloss_savefile)
            file_a = filename.split(".")[0]
            file_b = filename.split(".")[1]
            filename = file_a + str(num) + file_b
            config.pic_trainloss_savefile = filename
            num += 1
    plt.show()

    x2 = np.linspace(0,len(validate_loss_list)-1,len(validate_loss_list))
    plt.plot(x2,validate_loss_list)
    plt.title('validate loss',fontsize=12)
    while True:
        if not os.path.exists(config.pic_validateloss_savefile):
            plt.savefig(config.pic_validateloss_savefile)
            break
        else:
            (filepath,filename) = os.path.split(config.pic_validateloss_savefile)
            file_a = filename.split(".")[0]
            file_b = filename.split(".")[1]
            filename = file_a + str(num) + file_b
            config.pic_validateloss_savefile = filename
            num += 1
    plt.show()
def test_entry():
    # loading model
    if os.path.exists(config.model_save_file):
        net = torch.load(config.model_save_file)
    else:
        print("There is no model in file: "+config.save_statics_file)
        return 
    word_to_idx = net.word_to_idx
    # preparing test datasets
    sentences_test,labels_test = load_datasets(config.test_npz)
    test_features,test_labels,test_scaler = prepare_datasets(sentences_test,labels_test,word_to_idx,"test")

    test_set = torch.utils.data.TensorDataset(test_features, test_labels)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size,shuffle=False)
    start = time.time()
    test_losses = 0
    test_acc = 0
    m = 0
    test_loss_list = []
    loss_function = nn.MSELoss()
    with torch.no_grad():
        for test_feature, test_label in tqdm(test_iter,"test: "):
            m += 1
            test_feature = test_feature.cuda()
            test_label = test_label.cuda()
            test_score = net(test_feature)
            test_loss = loss_function(test_score, test_label)
            te_score = test_scaler.inverse_transform(test_score.cpu().data.numpy())
            te_label = test_scaler.inverse_transform(test_label.cpu().data.numpy())

            te_score = test_scaler.inverse_transform(te_score)
            te_score = np.squeeze(np.round(te_score.reshape(1,-1)))
            te_label = test_scaler.inverse_transform(te_label)
            te_label = np.squeeze(te_label.reshape(1,-1))

            test_acc += accuracy_score(te_label,te_score)
            test_losses += test_loss
            test_loss_list.append(test_loss.cpu().data.numpy().tolist())
    end = time.time()
    runtime = end - start
    print('test loss: %.4f, test acc: %.2f, time: %.2f' %(test_losses.data / m, test_acc / m, runtime))
    # draw pictures
    x = np.linspace(0,len(test_loss_list)-1,len(test_loss_list))
    plt.plot(x,test_loss_list,label = "train loss")
    num = 0
    while True:
        if not os.path.exists(config.pic_testloss_savefile):
            plt.savefig(config.pic_testloss_savefile)
            break
        else:
            (filepath,filename) = os.path.split(config.pic_testloss_savefile)
            file_a = filename.split(".")[0]
            file_b = filename.split(".")[1]
            filename = file_a + str(num) + file_b
            config.pic_testloss_savefile = filename
            num += 1
    plt.show()
