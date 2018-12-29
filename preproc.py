from tqdm import tqdm
from absl import app
import pyltp as ltp
import pandas as pd
import numpy as np
import csv 
import os
from config import config
from utils import DataSetWordsLabel,DataSetEmddingLabel
from zhon import hanzi
import string
def MakeVocabs(task_file,save_file = "vocab.txt"):
    print("Make vocabulary,Loading File:\n{}".format(task_file))
    Vocabs = set()
    with open(task_file,mode = "r",encoding = "utf-8") as fp:
        DataSet = csv.reader(fp)
        Headers =  next(DataSet)
        for line in  tqdm(DataSet):
            # id,sentence,labels
            id_index = int(line[0])
            sentence = line[1]
            labels_index = line[2:-1]
            # This is the English and Chinese punctuations
            # print(string.punctuation)  //!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
            # print(hanzi.punctuation)   //＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。
            # segment the sentences
            segment = ltp.Segmentor()
            segment.load(os.path.join(config.segment_model_file,"cws.model"))
            sent = list(segment.segment(sentence))
            segment.release()
            for word in sent:
                Vocabs.add(word.strip())
    with open(save_file,mode = "w",encoding = "utf-8") as fp:
        for _,word in enumerate(Vocabs):
            fp.write(word + "\n")
    Length = len(Vocabs)
    del Vocabs
    print("The vocabulary length:{}".format(Length))
    print("Saved File:\n{}".format(save_file))
# The first step we must make a vocabulary to modify the task
def LoadVocabs(save_file = "vocab.txt"):
    print("Loading vocabulary:\n{}".format(save_file))
    Vocabs = set()
    with open(save_file,mode = "r",encoding = "utf-8") as fpLoad:
        while True:
            line = fpLoad.readline()
            if not line:
                break
            line = line.strip()
            if line =="":
                break
            else:
                Vocabs.add(line)
    return Vocabs
# Get word embedding 
def GetEmbeddings(Vocabs,embedding_file):
    print("Loading Embedding Model .......")
    word_vecs = {}
    # loads 300x1 word vectors from file.
    with open(embedding_file, "r") as fp:
        header = fp.readline()
        vocab_size, layer_size = map(int, header.split()) # 3000000 300
        for line in tqdm(range(vocab_size)):
            line = fp.readline()
            vectors = line.split()
            word = vectors[0]
            try:
                vec  = list(map(float, vectors[1:-1]))
            except:
                pass
            if word in Vocabs:
                word_vecs[word] = np.array(vec, dtype='float32')
    # add random vectors of unknown words which are not in pre-trained vector file.
    # if pre-trained vectors are not used, then initialize all words in vocab with random value.
    for word in Vocabs:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, config.word_dim)
    return word_vecs
def preproc():
    # Generate the vocabulary
    task_file = "/home/asus/AI_Challenger2018/TestData/testfile.csv"
    save_file = "vocab.txt"
    if not os.path.exists(save_file):
        MakeVocabs(task_file,save_file)
    Vocabs = LoadVocabs()
    # load embeddings
    word_vecs = GetEmbeddings(Vocabs,config.wordembedding_file)
    # MakeDataSet
    datasetswords = DataSetWordsLabel(task_file)
    # ChangeToEmbedding
    datasetemdding = DataSetEmddingLabel(datasetswords,word_vecs)
    if not os.path.exists(config.Save_datafile):
        os.mkdir(config.Save_datafile)
    np.savez(config.train_npz,datasetemdding = datasetemdding.Sentences)
def main(_):
    preproc()
if __name__ == "__main__":
    app.run(main)
