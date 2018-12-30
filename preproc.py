from tqdm import tqdm
from absl import app
import pyltp as ltp
import pandas as pd
import numpy as np
import csv 
import os
import threading
from config import config
from utils import getRadomNum
import thread_sepwords as thsepword
import thread_sepsentences as thsepsent
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
            if word in Vocabs and len(vec) == config.word_dim:
                word_vecs[word] = np.array(vec, dtype='float32')
    # add random vectors of unknown words which are not in pre-trained vector file.
    # if pre-trained vectors are not used, then initialize all words in vocab with random value.
    for word in Vocabs:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, config.word_dim)
    return word_vecs
def preproc(task_file,save_file,seplength,delay):
    if not os.path.exists(config.save_datafile):
        os.mkdir(config.save_datafile)
    # Generate the vocabulary
    dictionary = thsepword.Dictionary(task_file,config.vocab_file,"file",seplength,delay)
    # MakeDataSet
    datasetswords = thsepsent.DataSetWords(task_file,save_file,"file",seplength,delay)
def main(_):
    task_file = "/home/asus/AI_Challenger2018/TestData/testfile.csv"
    # save_file = "/home/asus/AI_Challenger2018/TestData/sent.npz"
    seplength = 200
    delay = 3
    preproc(task_file,config.train_npz,seplength,delay)
    data = np.load(config.train_npz)
    
if __name__ == "__main__":
    app.run(main)
