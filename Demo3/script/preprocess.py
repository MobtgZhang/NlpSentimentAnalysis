import os
import pandas as pd
import codecs

from tqdm import tqdm

from functools import partial
from multiprocessing import Pool
from multiprocessing.util import Finalize

from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import Text8Corpus
import logging
import pyltp as ltp
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from .tokenizer import JieBaTokenizer,LtpTokenizer

from config import LTP_MODEL_PATH,ANNTOTORS
def init(tokenizer):
    global TOK
    if tokenizer == "jieba":
        TOK = JieBaTokenizer(annotators=ANNTOTORS)
    elif tokenizer == "pyltp":
        TOK = LtpTokenizer(annotators=ANNTOTORS,model_path=LTP_MODEL_PATH)
    Finalize(TOK, TOK.shutdown, exitpriority=100)
def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)

    output = {
        'words': tokens.words(),
        'chars': tokens.chars(),
        'pos': tokens.pos(),
        'ner': tokens.entities(),
    }
    return output
def process_dataset(data, tokenizer, workers=None):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""
    make_pool = partial(Pool, workers, initializer=init)

    workers = make_pool(initargs=(tokenizer,))
    c_tokens = workers.map(tokenize, data['sentences'])
    workers.close()
    workers.join()

    for idx in tqdm(range(len(data["sentences"]))):
        document = c_tokens[idx]['words']
        document_char = c_tokens[idx]['chars']
        cpos = c_tokens[idx]['pos']
        cner = c_tokens[idx]['ner']
        yield {
            'document': document,
            'document_char': document_char,
            'cpos': cpos,
            'cner': cner,
            'labels':data['label_list'][idx]
        }
def load_dataset(path):
    """Load csv file and store fields separately."""
    outData = pd.read_csv(path)
    sentences_raw = outData["content"]
    sentences = []
    for k in tqdm(range(len(sentences_raw))):
        sent = sentences_raw[k][1:-1].replace("\n", "")
        sentences.append(sent)
    datalist = outData.columns[2:]
    indexes = list(datalist)
    label_list = outData[indexes].values.tolist()
    output = {
        "sentences": sentences,
        "indexes": indexes,
        "label_list": label_list
    }
    return output
def prepare_embeddings(args):

    sentences = Text8Corpus(args.sep_AI_RAW)
    model = Word2Vec(sentences=sentences,
                         sg=args.sg,
                         size=args.emb_size,
                         window=5,
                         min_count=1,
                         negative=3,
                         sample=0.001,
                         hs=True,
                         workers=6)
    out_root = os.path.join(args.emb_path,args.type)
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    if args.sg:
        out_file = os.path.join(out_root,args.type+"_emb_skipGram.emb")
    else:
        out_file = os.path.join(out_root,args.type+"_emb_COBW.emb")
    model.save(out_file)
def preparing_raw(file_list,save_path,type="words"):
    list_concat = []
    for file in file_list:
        print("file: "+file)
        outData = pd.read_csv(file)
        list_concat.append(outData)
    print("Done!")
    res = pd.concat(list_concat, axis=0)
    sentences = res['content'].tolist()
    seg_sentences = []
    if type == "words":
        segmentor = ltp.Segmentor()
        cws_model_path = os.path.join(LTP_MODEL_PATH,"cws.model")
        segmentor.load(cws_model_path)
        for k in tqdm(range(len(sentences)),"segmenting:"):
            # segment the words.
            clean_text = sentences[k].replace("\n"," ")
            tokens = list(segmentor.segment(clean_text))
            seg_sentences.append(tokens)
        segmentor.release()
    else:
        for k in tqdm(range(len(sentences)),"segmenting:"):
            # segment the words.
            clean_text = sentences[k].replace("\n"," ")
            tokens = list(clean_text)
            seg_sentences.append(tokens)
    with codecs.open(save_path, mode="w", encoding="utf-8") as file:
        for k in tqdm(range(len(seg_sentences)), "writing :"):
            file.write(" ".join(seg_sentences[k]) + "\n")
    print("File: %s saved!"%save_path)