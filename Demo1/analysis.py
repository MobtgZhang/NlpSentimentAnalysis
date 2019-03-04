import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
import os
from absl import app
import pyltp as ltp
from sklearn.metrics import accuracy_score
from utils import GetVocabs,MakeSets,encode_samples,pad_samples,prepare_vocab,prepare_labels
from config import config
from process import load_datasets,prepare_datasets
class SentimentAnalysis:
    def __init__(self):
        self.ltpsplitfile = None
        self.sentimentfile = None
        self.Sengmentor = None
        self.SentimentModel = None
        self.Vocabs = None
    def load(self,ltpsplitfile,sentimentfile,vocabfile):
        self.ltpsplitfile = ltpsplitfile
        self.sentimentfile = sentimentfile
        # load the PyLtp model
        self.Sengmentor = ltp.Segmentor()
        self.Sengmentor.load(ltpsplitfile)
        # load the trained model
        self.SentimentModel = torch.load(sentimentfile)
        # load the vocabulary
        data = np.load(vocabfile)
        word_to_idx = data['word_to_idx'][()]
        self.Vocabs = word_to_idx
    def calculdate(self,sent):
        # sengment the words
        feature = list(self.Sengmentor.segment(sent))
        word_to_indexlist = self.pad_samples(self.encode_samples(sent),maxlen=1000)
        # change word to indexes
        feature = torch.autograd.Variable(torch.LongTensor(word_to_indexlist).view(1,1000))
        score = self.SentimentModel(feature,run = False)
        return (torch.argmax(score,dim = 1).cpu().data.numpy() - 2)
    def release(self):
        self.Sengmentor.release()
        self.ltpsplitfile = None
        self.sentimentfile = None
        self.Sengment = None
        self.SentimentModel = None
        self.Vocabs = None
    # Encoding the tokens
    def encode_samples(self,tokenized_samples):
        feature = []
        for token in tokenized_samples:
            if token in self.Vocabs:
                feature.append(self.Vocabs[token])
            else:
                feature.append(0)
        return feature
    # Encoding the features
    def pad_samples(self,feature, maxlen=1000, PAD=0):
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while(len(padded_feature) < maxlen):
                padded_feature.append(PAD)
        return padded_feature
def collect_data(net,loadfilename,mode):
    # preparing vocabulary
    data = np.load(config.vocab_file)
    word_to_idx = data['word_to_idx'][()]
    # preparing test datasets
    sentences_test,labels_test = load_datasets(loadfilename)
    features,labels = prepare_datasets(sentences_test,labels_test,word_to_idx,mode)

    data_set = torch.utils.data.TensorDataset(features, labels)
    data_iter = torch.utils.data.DataLoader(data_set, batch_size=config.batch_size,shuffle=False)

    start = time.time()
    losses = 0
    accuracy = 0
    m = 0
    list_true = []
    list_pred = []
    loss_function = nn.CrossEntropyLoss()
    with torch.no_grad():
        for feature, label in tqdm(data_iter,mode+": "):
            m += 1
            score = net(feature)
            if config.use_gpu:
                label = label.cuda()
            loss = loss_function(score, label)

            score = torch.argmax(score,dim = 1).cpu().data.numpy()
            # append the data
            list_pred.append(score.astype(int))
            score = np.squeeze(score.reshape(-1).astype(int))

            # append the data
            label = label.cpu().data.numpy()
            list_true.append(label.astype(int))
            label = np.squeeze(label.reshape(-1).astype(int))

            accuracy += accuracy_score(label,score)
            losses += loss
    end = time.time()
    runtime = end - start
    print(mode+' loss: %.4f, test acc: %.2f, time: %.2f' %(losses.data / m, accuracy / m, runtime))
    return list_true,list_pred
def test_entry(modelname):
    # loading model
    model_save_file = os.path.join(config.save_statics_file,modelname,modelname + ".pkl")
    if os.path.exists(model_save_file):
        net = torch.load(model_save_file)
        net.to(config.device)
    else:
        print("There is no model in file: "+config.save_statics_file)
        return 
    train = {}
    test = {}
    validate = {}
    # get dataset
    train['true'],train['predict'] = collect_data(net,config.train_npz,"train")
    validate['true'],validate['predict'] = collect_data(net,config.validate_npz,"validate")
    test['true'],test['predict'] = collect_data(net,config.test_npz,"test")
    savefile = os.path.join(config.save_statics_file,modelname,"results.npz")
    # save dataset
    np.savez(savefile,train = train,validate = validate,test = test)
