#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Machine Comprehension predictor"""

import logging

from functools import partial
from multiprocessing import Pool
from multiprocessing.util import Finalize
from tqdm import tqdm
import codecs
import json
import time
from collections import Counter

from model.model import DocReader
import utils.util as util
from script.tokenizer import LtpTokenizer
from utils.config import LTP_MODEL_PATH,SFD_MODEL_PATH,ANNTOTORS
from utils.config import LABEL_LILST
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
# ------------------------------------------------------------------------------
# Tokenize + annotate
# ------------------------------------------------------------------------------

TOK = None

def init(tokenizer,annotators):
    global TOK
    TOK = LtpTokenizer(annotators=annotators,model_path=LTP_MODEL_PATH)
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

def get_annotators_for_model(model):
    annotators = set()
    if model.args.use_pos:
        annotators.add('pos')
    if model.args.use_ner:
        annotators.add('ner')
    return annotators


# ------------------------------------------------------------------------------
# Predictor class.
# ------------------------------------------------------------------------------


class Predictor(object):
    """Load a pretrained DocReader model and predict inputs on the fly."""

    def __init__(self, model, normalize=True,tokenizer = "pylyp",
                 embedding_file=None, char_embedding_file=None, num_workers=None):
        """
        Args:
            model: path to saved model file.
            normalize: squash output score to 0-1 probabilities with a softmax.
            embedding_file: if provided, will expand dictionary to use all
              available pretrained vectors in this file.
            num_workers: number of CPU processes to use to preprocess batches.
        """
        logger.info('Initializing model...')
        self.model = DocReader.load(model, normalize=normalize)
        self.label_list = LABEL_LILST.split(",")
        if embedding_file:
            logger.info('Expanding dictionary...')
            words = util.index_embedding_words(embedding_file)
            added_words = self.model.expand_dictionary(words)
            self.model.load_embeddings(added_words, embedding_file)
        if char_embedding_file:
            logger.info('Expanding dictionary...')
            chars = util.index_embedding_chars(char_embedding_file)
            added_chars = self.model.expand_char_dictionary(chars)
            self.model.load_char_embeddings(added_chars, char_embedding_file)

        logger.info('Initializing tokenizer...')
        annotators = get_annotators_for_model(self.model)
        if num_workers is None or num_workers > 0:
            make_pool = partial(Pool, num_workers, initializer=init)
            self.workers = make_pool(initargs=(tokenizer,annotators,))

        else:
            self.workers = None
            if tokenizer == "jieba":
                self.tokenizer = JieBaTokenizer(annotators=annotators)
            elif tokenizer == "pyltp":
                self.tokenizer = LtpTokenizer(annotators=annotators, model_path=LTP_MODEL_PATH)
            elif tokenizer == "stanfordnlp":
                self.tokenizer = StanfordcorenlpTokenizer(annotators=annotators, model_path=SFD_MODEL_PATH)
            else:
                raise Exception("Unknown tokenizer: " + tokenizer)
    def process_text(self,data_file,save_predict):
        t0 = time.time()
        logger.info('Loading dataset %s' % data_file)
        dataset = self.load_dataset(data_file)
        logger.info('Will write to file %s' % save_predict)
        c_tokens = self.workers.map(tokenize, dataset)
        self.workers.close()
        self.workers.join()
        with codecs.open(save_predict, mode='w',encoding="utf-8") as f:
            for idx in tqdm(range(len(c_tokens))):
                document = c_tokens[idx]['words']
                document_char = c_tokens[idx]['chars']
                cpos = c_tokens[idx]['pos']
                cner = c_tokens[idx]['ner']
                dictionary = {
                    'document': document,
                    'document_char': document_char,
                    'cpos': cpos,
                    'cner': cner,
                }
                f.write(json.dumps(dictionary) + '\n')
        logger.info('Total time: %.4f (s)' % (time.time() - t0))
        logger.info("File: %s saved!" % save_predict)
    def load_dataset(self,path):
        """Load csv file and store fields separately."""
        sentences = []
        with codecs.open(path,mode="r",encoding="utf-8") as f:
            while True:
                sent = f.readline()
                if not sent:
                    break
                sent = sent.replace("\n"," ")
                sentences.append(sent)
        return sentences
    def load_processed_data(self,args,saved_predict):
        """Load examples from preprocessed file.
            One example per line, JSON encoded.
            """
        # Load JSON lines
        examples = []
        with open(saved_predict) as f:
            for line in f:
                ex = json.loads(line)
                examples.append(ex)
        return examples
    def predict(self,ex):
        """Predict a single document ."""
        # Eval mode
        self.model.network.eval()
        # Transfer to GPU
        if self.model.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True)) for e in ex]
        else:
            inputs = [e if e is None else Variable(e) for e in ex]
        # Run forward
        scores = self.model.network(*inputs)
        predicts = torch.argmax(scores, dim=1)
        return predicts.numpy()
    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------
class PredictorDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.examples[index])

    def lengths(self):
        return [len(ex['document']) for ex in self.examples]

def vectorize(ex, model):
    """Torchify a single example."""
    args = model.args
    word_dict = model.word_dict
    char_dict = model.char_dict
    feature_dict = model.feature_dict

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    document_char = [torch.LongTensor([char_dict[c] for c in cs]) for cs in ex['document_char']]

    # Create extra features vector
    if len(feature_dict) > 0:
        c_features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        c_features = None

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['cpos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                c_features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['cner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                c_features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            c_features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l
    return document, document_char,c_features
def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    docs = [ex[0] for ex in batch]
    doc_chars = [ex[1] for ex in batch]
    c_features = [ex[2] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    max_char_length = max([c.size(0) for cs in doc_chars for c in cs])

    x = torch.LongTensor(len(docs), max_length).zero_()
    x_c = torch.LongTensor(len(docs), max_length, max_char_length).zero_()
    x_mask = torch.ByteTensor(len(docs), max_length).fill_(1)

    if c_features[0] is None:
        x_f = None
    else:
        x_f = torch.zeros(len(docs), max_length, c_features[0].size(1))
    for i, d in enumerate(docs):
        x[i, :d.size(0)].copy_(d)
        x_mask[i, :d.size(0)].fill_(0)
        if x_f is not None:
            x_f[i, :d.size(0)].copy_(c_features[i])
    for i, cs in enumerate(doc_chars):
        for j, c in enumerate(cs):
            c_ = c[:max_char_length]
            x_c[i, j, :c_.size(0)].copy_(c_)
    return x, x_c, x_f, x_mask

class SentencePredict:
    ANNTOTORS = {"pos", "ner"}
    def __init__(self,model_file,ltp_model_path):
        self.model = DocReader.load(model_file, normalize=True)
        self.ltp_model_path = ltp_model_path
    def predict(self,text):
        outline = self.get_tokens(text)
        vector = vectorize(outline, self.model)
        batch = batchify([vector])
        """Predict a single document ."""
        # Eval mode
        self.model.network.eval()
        # Transfer to GPU
        if self.model.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True)) for e in batch]
        else:
            inputs = [e if e is None else Variable(e) for e in batch]
        # Run forward
        scores = self.model.network(*inputs)
        predicts = torch.argmax(scores, dim=1)
        return predicts.numpy().squeeze()
    def get_tokens(self,sentence, annotators=ANNTOTORS):
        TOK = LtpTokenizer(annotators=annotators,model_path=self.ltp_model_path)
        tokens = TOK.tokenize(sentence)
        output = {
            'document': tokens.words(),
            'document_char': tokens.chars(),
            'cpos': tokens.pos(),
            'cner': tokens.entities(),
        }
        return output