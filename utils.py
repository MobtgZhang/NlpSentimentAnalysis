
from config import config
import csv
from absl import app
from tqdm import tqdm
import pyltp as ltp
import os
import numpy as np
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
class DataSetEmddingLabel:
    def __init__(self,datasetswords,embedding):
        self.Sentences = []
        for index in range(len(datasetswords)):
            wordslist = datasetswords[index]
            Emb = WordsToEmbedding(wordslist[0],embedding)
            Label = wordslist[1]
            self.Sentences.append([Emb,Label])
        self.DataSetLength = len(self.Sentences)
    def __getitem__(self,index):
        return self.Sentences[index]
    def __len__(self):
        return self.DataSetLength
class DataSetWordsLabel:
    def __init__(self,task_file):
        self.Sentences = []
        with open(task_file,mode = "r",encoding = "utf-8") as fp:
            print("Building DataSetWords ...")
            DataSet = csv.reader(fp)
            Headers =  next(DataSet)
            for line in tqdm(DataSet):
                # id,sentence,labels
                id_index = int(line[0])
                sentence = line[1]
                labels_index = line[2:-1]
                labels_index = list(map(int, labels_index[1:-1]))
                # This is the English and Chinese punctuations
                # print(string.punctuation)  //!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
                # print(hanzi.punctuation)   //＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。
                # segment the sentences
                segment = ltp.Segmentor()
                segment.load(os.path.join(config.segment_model_file,"cws.model"))
                sent = list(segment.segment(sentence))[0:config.text_length]
                segment.release()
                self.Sentences.append([sent,labels_index])
        self.DataSetLength = len(self.Sentences)
    def __getitem__(self,item):
        return self.Sentences[item]
    def __len__(self):
        return self.DataSetLength
def main(_):
    datasets = DataSet("/home/asus/AI_Challenger2018/TestData/testfile.csv")
    for line in datasets:
        print(line)
if __name__ == "__main__":
    app.run(main)





def process_file(filename,data_type,word_counter,char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename,"r") as fh:
        source = pd.read_csv(filename)
        print(source[0:2]["content"].loc[0])
def preproc(config):
    word_counter,char_counter = Counter() , Counter()
    process_file(config.train_file,"train",word_counter,char_counter)
    # process_file(config.validation_file,"validation",word_counter,char_counter)
    # process_file(config.test_file,"test",word_counter,char_counter)
    print("preproc")
