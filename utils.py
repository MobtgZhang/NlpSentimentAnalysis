from config import config
import csv
from absl import app
from tqdm import tqdm
import pyltp as ltp
import os
import numpy as np
import uuid
import pandas as pd
# random creates a file for every data file
def getRadomNum():
    res = str(uuid.uuid4())
    res = res.replace('-', '')
    return res[:16]
# Open a file for thread
def GetData(filename,seplength):
    out = pd.read_csv(filename)
    Length = len(out)
    All_Sep = Length//seplength
    DataList = []
    for k in range(All_Sep):
        DataList.append(out.loc[k*seplength:(k+1)*seplength])
    return DataList
# LoadVocabs
def LoadVocabs(save_filename):
    Vocabs = set()
    with open(save_filename,mode = "r",encoding = "utf-8") as fpLoad:
        while True:
            line = fpLoad.readline()
            if not line:
                break
            line = line.strip()
            if line =="":
                continue
            else:
                Vocabs.add(line)
    return Vocabs
def WordsToEmbedding(wordslist,embedding):
    sent_len = len(wordslist)
    emb_len = config.word_dim
    Emb = np.zeros((sent_len,emb_len),dtype = "float32")
    for k,word in enumerate(wordslist):
        if word in embedding:
            Emb[k] = embedding[word.strip()]
        else:
            Emb[k] = np.random.uniform(-0.25, 0.25, config.word_dim)
    return Emb
