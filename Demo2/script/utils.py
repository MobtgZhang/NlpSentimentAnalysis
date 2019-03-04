import pandas as pd
import uuid
from tqdm import tqdm
import numpy as np
import os
import gensim
import torch.nn as nn
import process.config as config
def prepare_dictionary():
    if os.path.exists(config.vocab_npz):
        print("Dictionary has beed built!")
        print("Files: " + config.vocab_npz + " exists!")
        return
    # combine three vocabulary
    Vocabs = set()
    # Make a dictionary
    print("Making vocabulary ...")
    Vocabs_train = GetVocabs(config.train_npz)
    Vocabs_validate = GetVocabs(config.validate_npz)
    Vocabs_test = GetVocabs(config.test_npz)
    Vocabs.update(Vocabs_train)
    Vocabs.update(Vocabs_validate)
    Vocabs.update(Vocabs_test)
    idx_to_word, word_to_idx = MakeSets(Vocabs)
    np.savez(config.vocab_npz, idx_to_word=idx_to_word, word_to_idx=word_to_idx)
    print("Dicitonary Build! ")
class Dictionary(object):
    def __init__(self, path=''):
        self.word2idx = None
        self.idx2word = None
        if path != '':
            # load an external dictionary
            prepare_dictionary()
            outData = np.load(path)
            self.word2idx = outData['word_to_idx'].tolist()
            self.idx2word = outData['idx_to_word'].tolist()
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    def __len__(self):
        return len(self.idx2word)
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
# prepare the wordembedding
def prepare_embedding(word_embFile,dictionary,emb_save,embed_size = 300):
    loaded_cnt = 0
    # 使用gensim载入word2vec词向量
    # wvmodel = gensim.models.KeyedVectors.load_word2vec_format(word_embFile, binary=False, encoding='utf-8')
    wvmodel = gensim.models.word2vec.Word2Vec.load(word_embFile).wv
    assert wvmodel.vector_size >= embed_size
    vocab_size = len(dictionary.word2idx) + 1
    vocabulary = wvmodel.index2word
    weight = np.random.randn(vocab_size,embed_size)
    for word in tqdm(dictionary.word2idx):
        if word not in vocabulary:
            continue
        real_id = dictionary.word2idx[word]
        loaded_id = vocabulary.index(word)
        weight[real_id] = wvmodel.vectors[loaded_id][:embed_size]
        loaded_cnt += 1
    np.savez(emb_save,weight=weight,loaded_cnt = loaded_cnt)


# for bert model
def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)