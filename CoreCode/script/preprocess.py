import os
import pandas as pd
import codecs

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from multiprocessing.util import Finalize
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from .tokenizer import LtpTokenizer

from utils.config import LTP_MODEL_PATH,ANNTOTORS
def init(tokenizer):
    global TOK
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
