import copy
import os
import pyltp as ltp
import jieba.posseg as posseg
from stanfordcorenlp import StanfordCoreNLP
class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    CHAR = 1
    POS = 2
    NER = 3

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT] for t in self.data]).strip()

    def chars(self):
        """Returns a list of the first character of each token

        Args:
            uncased: lower cases characters
        """
        return [[c for c in t[self.CHAR]] for t in self.data]

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        return [t[self.TEXT] for t in self.data]
    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]
    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]
class LtpTokenizer(object):

    def __init__(self,annotators,model_path,**kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: Pyltp model to use .
        """
        # path defination

        self.annotators = annotators
        self.model_path = model_path
        cws_model_path = os.path.join(self.model_path, 'cws.model')
        pos_model_path = os.path.join(self.model_path, 'pos.model')
        ner_model_path = os.path.join(self.model_path, 'ner.model')
        self.segmentor = ltp.Segmentor()
        self.segmentor.load(cws_model_path)
        self.postagger = ltp.Postagger()
        self.postagger.load(pos_model_path)
        self.recognizer = ltp.NamedEntityRecognizer()
        self.recognizer.load(ner_model_path)

    def tokenize(self, text):

        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')

        # segment the words
        tokens = list(self.segmentor.segment(clean_text))
        # postag the words
        postags = list(self.postagger.postag(tokens))
        # ner recognition
        netags = list(self.recognizer.recognize(tokens,postags))

        data = []
        Length = len(tokens)
        for i in range(Length):
            # Get whitespace
            data.append((
                tokens[i],
                list(tokens[i]),
                postags[i],
                netags[i]
            ))

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={'non_ent': ''})

    def shutdown(self):
        self.segmentor.release()
        self.postagger.release()
        self.recognizer.release()

    def __del__(self):
        self.shutdown()
class JieBaTokenizer(object):
    def __init__(self,annotators, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: Pyltp model to use .
        """
        self.annotators = annotators
    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')

        # segment and postag the words
        seg_list = posseg.cut(clean_text)

        data = []
        for w in seg_list:
            # Get whitespace
            data.append((
                w.word,
                list(w.word),
                w.flag,
                ""
            ))
        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={'non_ent': ''})

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()
class StanfordcorenlpTokenizer(object):
    def __init__(self,annotators,model_path, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: Stanfordcorenlp model to use .
        """
        self.annotators = annotators
        self.nlp = StanfordCoreNLP(model_path,lang='zh',memory="2g")
        self.model_path = model_path
    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')

        # segment and postag the words
        seg_list = self.nlp.word_tokenize(clean_text)
        # postag the words
        pos_list = self.nlp.pos_tag(clean_text)
        # ner recognition
        ner_list = self.nlp.ner(clean_text)
        # Constituency Parsing
        #con_list = self.nlp.parse(clean_text)
        # dependency parse
        #dep_list = self.nlp.dependency_parse(clean_text)

        data = []
        for k in range(len(seg_list)):
            # Get whitespace
            data.append((
                seg_list[k],
                list(seg_list[k]),
                pos_list[k][1],
                ner_list[k][1]
            ))
        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={'non_ent': ''})

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()
