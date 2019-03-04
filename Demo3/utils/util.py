import json
from collections import Counter
import logging
logger = logging.getLogger(__name__)
from gensim.models.word2vec import Word2Vec

from utils.data import Dictionary

def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    model = Word2Vec.load(embedding_file).wv
    for w in model.index2word:
        w = Dictionary.normalize(w)
        words.add(w)
    return words
def index_embedding_chars(char_embedding_file):
    """Put all the chars in char_embedding_file into a set."""
    chars = set()
    model = Word2Vec.load(char_embedding_file).wv
    for w in model.index2word:
        w = Dictionary.normalize(w)
        chars.add(w)
    return chars
def load_data(args,filename):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    examples = []
    if filename == args.train_file:
        num = 0
    with open(filename) as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)
            if filename == args.train_file:
                num +=1
                if num >= 30000:
                    break
    return examples
def load_words(args, examples):
    """Iterate and index all the words in examples (documents + questions)."""
    def _insert(iterable):
        for w in iterable:
            w = Dictionary.normalize(w)
            if valid_words and w not in valid_words:
                continue
            words.add(w)

    if args.restrict_vocab and args.words_embedding_file:
        logger.info('Restricting to words in %s' % args.words_embedding_file)
        valid_words = index_embedding_words(args.words_embedding_file)
        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None

    words = set()
    for ex in examples:
        _insert(ex['document'])
    return words
def load_chars(args, examples):
    """Iterate and index all the chars in examples (documents + questions)."""
    def _insert(iterable):
        for cs in iterable:
            for c in cs:
                c = Dictionary.normalize(c)
                if valid_chars and c not in valid_chars:
                    continue
                chars.add(c)

    if args.restrict_vocab and args.char_embedding_file:
        logger.info('Restricting to chars in %s' % args.char_embedding_file)
        valid_chars = index_embedding_chars(args.char_embedding_file)
        logger.info('Num chars in set = %d' % len(valid_chars))
    else:
        valid_chars = None

    chars = set()
    for ex in examples:
        _insert(ex['document_char'])
    return chars
def build_feature_dict(args, examples):
    """Index features (one hot) from fields in examples and options."""
    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    feature_dict = {}

    # Part of speech tag features
    if args.use_pos:
        for ex in examples:
            for w in ex['cpos']:
                _insert('pos=%s' % w)

    # Named entity tag features
    if args.use_ner:
        for ex in examples:
            for w in ex['cner']:
                _insert('ner=%s' % w)

    # Term frequency feature
    if args.use_tf:
        _insert('tf')

    return feature_dict
def build_word_dict(args, examples):
    """Return a word dictionary from question and document words in
    provided examples.
    """
    word_dict = Dictionary()
    for w in load_words(args, examples):
        word_dict.add(w)
    return word_dict
def build_char_dict(args, examples):
    """Return a char dictionary from question and document words in
    provided examples.
    """
    char_dict = Dictionary()
    for c in load_chars(args, examples):
        char_dict.add(c)
    return char_dict

def top_words(args, examples, word_dict):
    """Count and return the most common words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['question']:
            w = Dictionary.normalize(w)
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)