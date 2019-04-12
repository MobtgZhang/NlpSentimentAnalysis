import logging
import copy
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import gensim


from .m_reader import MnemonicReader
from .rnn_reader import RnnDocReader
from .r_net import R_Net
from utils.data import Dictionary
from utils.config import override_model_args

class DocReader(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, word_dict, char_dict, feature_dict,
                 state_dict=None, normalize=True):
        # Book-keeping.
        self.args = args
        self.word_dict = word_dict
        self.char_dict = char_dict
        self.args.vocab_size = len(word_dict)
        self.args.char_size = len(char_dict)
        self.feature_dict = feature_dict
        self.args.num_features = len(feature_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        if args.model_type == 'rnn':
            self.network = RnnDocReader(args, normalize)
        elif args.model_type == 'r_net':
            self.network = R_Net(args, normalize)
        elif args.model_type == 'mnemonic':
            self.network = MnemonicReader(args, normalize)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)

        # Load saved state
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)
    def expand_dictionary(self, words):
        """Add words to the DocReader dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).

        Args:
            words: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        to_add = {self.word_dict.normalize(w) for w in words
                  if w not in self.word_dict}

        # Add words to dictionary and expand embedding layer
        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                self.word_dict.add(w)
            self.args.vocab_size = len(self.word_dict)
            logger.info('New vocab size: %d' % len(self.word_dict))

            old_embedding = self.network.embedding.weight.data
            self.network.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                        self.args.embedding_dim,
                                                        padding_idx=0)
            new_embedding = self.network.embedding.weight.data
            new_embedding[:old_embedding.size(0)] = old_embedding

        # Return added words
        return to_add

    def expand_char_dictionary(self, chars):
        """Add chars to the DocReader dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).

        Args:
            chars: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        to_add = {self.char_dict.normalize(w) for w in chars
                  if w not in self.char_dict}

        # Add chars to dictionary and expand embedding layer
        if len(to_add) > 0:
            logger.info('Adding %d new chars to dictionary...' % len(to_add))
            for w in to_add:
                self.char_dict.add(w)
            self.args.char_size = len(self.char_dict)
            logger.info('New char size: %d' % len(self.char_dict))

            old_char_embedding = self.network.char_embedding.weight.data
            self.network.char_embedding = torch.nn.Embedding(self.args.char_size,
                                                             self.args.char_embedding_dim,
                                                             padding_idx=0)
            new_char_embedding = self.network.char_embedding.weight.data
            new_char_embedding[:old_char_embedding.size(0)] = old_char_embedding

        # Return added chars
        return to_add
    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}

        model=gensim.models.KeyedVectors.load_word2vec_format(embedding_file)
        loaded_dim = model.vector_size
        assert (loaded_dim == embedding.size(1))
        words_list = model.index2word
        for word in words_list:
            w = self.word_dict.normalize(word)
            if w in words:
                vec = torch.Tensor(model.get_vector(word))
                if w not in vec_counts:
                    vec_counts[w] = 1
                    embedding[self.word_dict[w]].copy_(vec)
                else:
                    logging.warning(
                        'WARN: Duplicate embedding found for %s' % w
                    )
                    vec_counts[w] = vec_counts[w] + 1
                    embedding[self.word_dict[w]].add_(vec)
        del model
        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def load_char_embeddings(self, chars, char_embedding_file):
        """Load pretrained embeddings for a given list of chars, if they exist.

        Args:
            chars: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            char_embedding_file: path to text file of embeddings, space separated.
        """
        chars = {w for w in chars if w in self.char_dict}
        logger.info('Loading pre-trained embeddings for %d chars from %s' %
                    (len(chars), char_embedding_file))
        char_embedding = self.network.char_embedding.weight.data

        # When normalized, some chars are duplicated. (Average the embeddings).
        vec_counts = {}

        model=gensim.models.KeyedVectors.load_word2vec_format(char_embedding_file)
        loaded_dim = model.vector_size
        assert (loaded_dim == char_embedding.size(1))
        chars_list = model.index2word
        for char in chars_list:
            w = self.char_dict.normalize(char)
            if w in chars:
                vec = torch.Tensor(model.get_vector(char))
                if w not in vec_counts:
                    vec_counts[w] = 1
                    char_embedding[self.char_dict[w]].copy_(vec)
                else:
                    logging.warning(
                        'WARN: Duplicate char embedding found for %s' % w
                    )
                    vec_counts[w] = vec_counts[w] + 1
                    char_embedding[self.char_dict[w]].add_(vec)
        del model
        for w, c in vec_counts.items():
            char_embedding[self.char_dict[w]].div_(c)
        logger.info('Loaded %d char embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(chars)))

    def tune_embeddings(self, words):
        """Unfix the embeddings of a list of words. This is only relevant if
        only some of the embeddings are being tuned (tune_partial = N).

        Shuffles the N specified words to the front of the dictionary, and saves
        the original vectors of the other N + 1:vocab words in a fixed buffer.

        Args:
            words: iterable of tokens contained in dictionary.
        """
        words = {w for w in words if w in self.word_dict}

        if len(words) == 0:
            logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(words) == len(self.word_dict):
            logger.warning('Tuning ALL embeddings in dictionary')
            return

        # Shuffle words and vectors
        embedding = self.network.embedding.weight.data
        for idx, swap_word in enumerate(words,self.word_dict.START):
            # Get current word + embedding for this index
            curr_word = self.word_dict[idx]
            curr_emb = embedding[idx].clone()
            old_idx = self.word_dict[swap_word]

            # Swap embeddings + dictionary indices
            embedding[idx].copy_(embedding[old_idx])
            embedding[old_idx].copy_(curr_emb)
            self.word_dict[swap_word] = idx
            self.word_dict[idx] = swap_word
            self.word_dict[curr_word] = old_idx
            self.word_dict[old_idx] = curr_word

        # Save the original, fixed embeddings
        self.network.register_buffer(
            'fixed_embedding', embedding[idx + 1:].clone()
        )

    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network.

        Args:
            state_dict: network parameters
        """
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, lr=self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(parameters, lr=self.args.learning_rate,
                                            rho=self.args.rho, eps=self.args.eps,
                                            weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)
    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()
        # Transfer to GPU
        # ex : x,x_c,x_f,x_mask,targets
        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True)) for e in ex[:-1]]
            targets =  Variable(ex[-1].cuda(async=True))
        else:
            inputs = [e if e is None else Variable(e) for e in ex[:-1]]
            targets = Variable(ex[-1])
        # Run forward
        score = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.cross_entropy(score,targets)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.args.grad_clipping)

        # Update parameters
        if not self.parallel:
            self.optimizer.step()
        else:
            self.optimizer.module.step() 
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return loss.data

    def reset_parameters(self):
        """Reset any partially fixed parameters to original states."""

        # Reset fixed embeddings to original value
        if self.args.tune_partial > 0:
            # Embeddings to fix are indexed after the special + N tuned words
            offset = self.args.tune_partial + self.word_dict.START
            if self.parallel:
                embedding = self.network.module.embedding.weight.data
                fixed_embedding = self.network.module.fixed_embedding
            else:
                embedding = self.network.embedding.weight.data
                fixed_embedding = self.network.fixed_embedding
            if offset < embedding.size(0):
                embedding[offset:] = fixed_embedding
    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex,async_pool = None):
        """Forward a batch of examples only to get predictions.

        Args:
            ex: the batch
        Output:
            scores: batch * fine_second_size * class_number predict matrix
            targets: batch * fine_second_size  labeled data
        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else Variable(e.cuda(async=True)) for e in ex[:-1]]
            targets = Variable(ex[-1].cuda(async=True))
        else:
            inputs = [e if e is None else Variable(e) for e in ex[:-1]]
            targets = Variable(ex[-1])
        # Run forward
        scores = self.network(*inputs)
        del inputs

        # Decode predictions
        scores = scores.data.cpu()
        targets = targets.data.cpu()
        return scores,targets
    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'char_dict': self.char_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        params = {
            'state_dict': self.network.state_dict(),
            'word_dict': self.word_dict,
            'char_dict': self.char_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')
    @staticmethod
    def load(filename, new_args=None, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        try:
            char_dict = saved_params['char_dict']
        except KeyError as e:
            char_dict = Dictionary()

        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        return DocReader(args, word_dict, char_dict, feature_dict, state_dict, normalize)

    @staticmethod
    def load_checkpoint(filename, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        char_dict = saved_params['char_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = DocReader(args, word_dict, char_dict, feature_dict, state_dict, normalize)
        model.init_optimizer(optimizer)
        return model, epoch

        # --------------------------------------------------------------------------
        # Runtime
        # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
            This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
        self.optimizer = torch.nn.DataParallel(self.optimizer)
