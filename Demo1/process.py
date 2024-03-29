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
from model import BiLSTMNet,textCNN,BiGRUNet,EMA

def prepare_datasets(sentences,labels,word_to_idx,mode):
    print("Preparing "+mode+" datasets ... ...")
    features = torch.LongTensor(pad_samples(encode_samples(sentences,word_to_idx),config.text_length))
    labels = prepare_labels(labels)
    labels = torch.LongTensor(labels)
    print("The "+mode+" datasets has been loaded!")
    return features,labels
def prepare_embedding(vocab_size,word_to_idx,idx_to_word):
    print("Loading word2vecs ... ...")
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(config.wordembedding_file,binary=False, encoding='utf-8')
    print("Word2vecs has been Loaded!")
    weight = torch.FloatTensor(np.random.randn(vocab_size + 1,config.word_dim))
    for i in range(len(wvmodel.index2word)):
        try:
            index = word_to_idx[wvmodel.index2word[i]]
        except:
            continue
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(idx_to_word[word_to_idx[wvmodel.index2word[i]]]))
    return weight
def prepare_train(train_features,train_labels,validate_features,validate_labels,weight,net,
        vocab_size,modelname,num_epochs,batch_size,learning_rate):
    device = torch.device(config.device)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()

    base_lr = 1.0
    warm_up = config.lr_warm_up_num

    ema = EMA(config.ema_decay)
    for name, p in net.named_parameters():
        if p.requires_grad: ema.set(name, p)
    params = filter(lambda param: param.requires_grad, net.parameters())
    optimizer = optim.Adam(lr=base_lr,params=params)
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
            score = net(feature)
            print(score)
            print(label)
            loss = loss_function(score, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            for name,p in net.named_parameters():
                if p.requires_grad:ema.update_parameter(name,p)
            torch.nn.utils.clip_grad_norm_(net.parameters(),config.grad_clip)

            # append the data 
            t_score = train_scaler.inverse_transform(score.data.numpy())
            t_score = np.squeeze(np.round(t_score.reshape(-1))).astype(int)

            # append the data 
            t_label = train_scaler.inverse_transform(label.data.numpy())
            t_label = np.squeeze(np.round(t_label.reshape(-1))).astype(int)
            
            train_acc += accuracy_score(t_label,t_score)
            train_loss += loss
            train_loss_list.append(loss.data.numpy().tolist())
        with torch.no_grad():
            for validate_feature, validate_label in tqdm(validate_iter,"validate: "):
                m += 1
                validate_score = net(validate_feature)
                validate_loss = loss_function(validate_score, validate_label)
                v_score = validate_score.data.numpy()
                v_label = validate_label.data.numpy()

                # append the data
                v_score = validate_scaler.inverse_transform(v_score)
                v_score = np.squeeze(np.round(v_score.reshape(1,-1))).astype(int)

                # append the data
                v_label = validate_scaler.inverse_transform(v_label)
                v_label = np.squeeze(np.round(v_label.reshape(1,-1))).astype(int)

                validate_acc += accuracy_score(v_label,v_score)
                validate_losses += validate_loss
                validate_loss_list.append(validate_loss.data.numpy().tolist())
        end = time.time()
        runtime = end - start
        print('epoch: %d, train loss: %.4f, train acc: %.2f, validate loss: %.4f, validate acc: %.2f, time: %.2f' %
                (epoch, train_loss.data / n, train_acc / n, validate_losses.data / m, validate_acc / m, runtime))
    # make a dir
    save_file = os.path.join(config.save_statics_file,modelname)
    if not os.path.exists(save_file):
        os.mkdir(save_file)
    # 保存整个网络和参数
    model_save_file = os.path.join(save_file,modelname + ".pkl")
    net.to("cpu")
    torch.save(net,model_save_file)
    return train_loss_list,validate_loss_list
def load_datasets(filename):
    data = np.load(filename)
    sentences = list(data['sentences'])
    labels = data['labels']
    return sentences,labels
def train_entry(modelname):
    # combine three vocabulary
    Vocabs = set()
    # Make a dictionary
    print("Loading vocabulary ...")
    Vocabs_train = GetVocabs(config.train_npz)
    Vocabs_validate = GetVocabs(config.validate_npz)
    Vocabs_test = GetVocabs(config.test_npz)
    Vocabs.update(Vocabs_train)
    Vocabs.update(Vocabs_validate)
    Vocabs.update(Vocabs_test)
    vocab_size = len(Vocabs)
    idx_to_word,word_to_idx = prepare_vocab(Vocabs)
    np.savez(config.vocab_file,idx_to_word = idx_to_word,word_to_idx = word_to_idx)
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
    if not os.path.exists(config.save_statics_file):
        os.mkdir(config.save_statics_file)
    if modelname == "BiLSTMNet":
        net = BiLSTMNet(vocab_size=(vocab_size+1), embed_size=config.word_dim,weight=weight,use_gpu = config.use_gpu)
    elif modelname == "textCNN":
        net = textCNN(vocab_size, embed_size = config.word_dim, seq_len = config.text_length, labels= config.labels,weight= weight,use_gpu = config.use_gpu)
    elif modelname == "BiGRUNet":
        net = BiGRUNet(vocab_size, embed_size = config.word_dim,labels= config.labels,weight= weight,use_gpu = config.use_gpu)
    else:
        raise Exception("unknown model:" + modelname)
    train_loss_list,validate_loss_list = prepare_train(train_features,train_labels,validate_features,validate_labels,weight,net,
        vocab_size,modelname,config.num_epochs,config.batch_size,config.learning_rate)
    # draw pictures
    pic_save_file = os.path.join(config.save_statics_file,modelname)

    pic_train = os.path.join(pic_save_file,modelname+"_train"+".png")
    x1 = np.linspace(0,len(train_loss_list)-1,len(train_loss_list))
    plt.plot(x1,train_loss_list)
    plt.title('Train loss',fontsize=12)
    plt.savefig(pic_train)
    plt.show()

    pic_validate = os.path.join(pic_save_file,modelname+"_validate"+".png")
    x2 = np.linspace(0,len(validate_loss_list)-1,len(validate_loss_list))
    plt.plot(x2,validate_loss_list)
    plt.title('validate loss',fontsize=12)
    plt.savefig(pic_validate)
    plt.show()
