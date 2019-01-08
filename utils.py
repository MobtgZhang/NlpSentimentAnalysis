from tqdm import tqdm
import uuid
import pandas as pd
import numpy as np
from config import config
from sklearn import preprocessing
# five score

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
def prepare_labels(labels):
    scaler = preprocessing.MinMaxScaler().fit(labels)
    return scaler
# random creates a file for every data file
def getRadomNum():
    res = str(uuid.uuid4())
    res = res.replace('-', '')
    return res[:16]
# Open a file for thread
def GetData(filename,seplength):
    print("Seperate the dataset...")
    out = pd.read_csv(filename)
    Length = len(out)
    All_Sep = Length//seplength
    DataList = []
    for k in tqdm(range(All_Sep)):
        DataList.append(out.loc[k*seplength:(k+1)*seplength])
    return DataList
# LoadVocabs
def GetVocabs(filename):
    Vocabs = set()
    data = np.load(filename)
    sentences = list(data['sentences'])
    for sent in sentences:
        for word in sent:
            Vocabs.add(word)
    return Vocabs
