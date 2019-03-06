import os
import subprocess
import logging
logger = logging.getLogger()
import torch
from sklearn.metrics import f1_score,accuracy_score
from utils import config
import utils.util as util
from model.model import DocReader
import torch.nn.functional as F
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')
def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=15,
                         help='Batch size for training')
    runtime.add_argument('--dev-batch-size', type=int, default=32,
                         help='Batch size during validation/testing')
    runtime.add_argument('--test-batch-size', type=int, default=32,
                         help='Batch size during validation/testing')
    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=config.MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--out-dir', type=str, default=config.OUT_DIR,
                       help='Directory of training/validation/test data')
    files.add_argument('--train-file', type=str,
                       default='train-processed-pyltp.json',
                       help='Preprocessed train file')
    files.add_argument('--dev-file', type=str,
                       default='validation-processed-pyltp.json',
                       help='Preprocessed dev file')
    files.add_argument('--test-file', type=str,
                       default='test-processed-pyltp.json',
                       help='Preprocessed test file')
    files.add_argument('--embed-dir', type=str, default=config.EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--words-embedding-file', type=str,
                       default='words_emb_skipGram.emb',
                       help='Space-separated pretrained embeddings file')
    files.add_argument('--char-embedding-file', type=str,
                       default='chars_emb_skipGram.emb',
                       help='Space-separated pretrained embeddings file')
    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')
    save_load.add_argument('--expand-dictionary', type='bool', default=False,
                           help='Expand dictionary of pretrained model to ' +
                                'include training/dev words of new data')
    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--document', type='bool', default=False,
                            help='Document words')
    preprocess.add_argument('--restrict-vocab', type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')
    # General
    general = parser.add_argument_group('General')
    general.add_argument('--official-eval', type='bool', default=True,
                         help='Validate with official SQuAD eval')
    general.add_argument('--valid-metric', type=str, default='exact_match',
                         help='The evaluation metric used for model selection: None, exact_match, f1')
    general.add_argument('--display-iter', type=int, default=25,
                         help='Log state after every <display_iter> epochs')
    general.add_argument('--sort-by-len', type='bool', default=True,
                         help='Sort batches by length for speed')
def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist

    args.train_file = os.path.join(args.out_dir, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)
    args.dev_file = os.path.join(args.out_dir, args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError('No such file: %s' % args.dev_file)
    args.test_file = os.path.join(args.out_dir, args.test_file)
    if not os.path.isfile(args.test_file):
        raise IOError('No such file: %s' % args.test_file)
    if args.words_embedding_file:
        args.words_embedding_file = os.path.join(args.embed_dir,"words",args.words_embedding_file)
        if not os.path.isfile(args.words_embedding_file):
            raise IOError('No such file: %s' % args.words_embedding_file)
    if args.char_embedding_file:
        args.char_embedding_file = os.path.join(args.embed_dir,"chars",args.char_embedding_file)
        if not os.path.isfile(args.char_embedding_file):
            raise IOError('No such file: %s' % args.char_embedding_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])


    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    # Embeddings options
    if args.words_embedding_file:
        from gensim.models.word2vec import Word2Vec
        dim = Word2Vec.load(args.words_embedding_file).wv.vector_size
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')
    if args.char_embedding_file:
        from gensim.models.word2vec import Word2Vec
        dim = Word2Vec.load(args.char_embedding_file).wv.vector_size
        args.char_embedding_dim = dim
    elif not args.char_embedding_dim:
        raise RuntimeError('Either char_embedding_file or char_embedding_dim '
                           'needs to be specified.')

    # Make sure tune_partial and fix_embeddings are consistent.
    if args.tune_partial > 0 and args.fix_embeddings:
        logger.warning('WARN: fix_embeddings set to False as tune_partial > 0.')
        args.fix_embeddings = False

    # Make sure fix_embeddings and embedding_file are consistent
    if args.fix_embeddings:
        if not (args.words_embedding_file or args.pretrained):
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    return args
# ------------------------------------------------------------------------------
# Validation loops. Includes both "unofficial" and "official" functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------


def validate_official(args, data_loader, model, global_stats,saver,mode):
    """Run one full unofficial validation.
    Unofficial = doesn't use SQuAD script.
    """
    eval_time = util.Timer()
    f1_score_avg = util.AverageMeter()
    accuracy_score_avg = util.AverageMeter()
    exact_match = util.AverageMeter()

    # Make predictions
    examples = 0
    for ex in data_loader:
        batch_size = ex[0].size(0)
        scores,targets = model.predict(ex)
        loss = F.cross_entropy(scores,targets)
        predicts = torch.argmax(scores,dim=1)
        # We get metrics for independent start/end and joint start/end
        accuracies = eval_accuracies(predicts,targets)
        f1_score_avg.update(accuracies[0], batch_size)
        accuracy_score_avg.update(accuracies[1], batch_size)
        exact_match.update(accuracies[2],batch_size)

        # If getting train accuracies, sample max 10k
        examples += batch_size
        if mode == 'train' and examples >= 1e4:
            break
    saver.loss_saver.add(loss.data)
    saver.f1_saver.add(f1_score_avg.avg)
    saver.acc_saver.add(accuracy_score_avg.avg)
    saver.em_saver.add(exact_match.avg)
    logger.info('%s valid unofficial: Epoch = %d | f1_score = %.2f | ' %
                (mode, global_stats['epoch'], f1_score_avg.avg) +
                'accuracy_score = %.2f | exact = %.2f | examples = %d | ' %
                (accuracy_score_avg.avg, exact_match.avg, examples) +
                'valid time = %.2f (s)' % eval_time.time())
    return {'exact_match': exact_match.avg}

def eval_accuracies(predicts,targets,average="macro"):

    """An unofficial evalutation helper.
    Compute exact predicts,targets match accuracies for a batch.
    type: binary,micro,macro
    """
    # Convert 1D tensors to lists of lists (compatibility)
    if torch.is_tensor(predicts):
        predicts = predicts.data.numpy()
        targets = targets.data.numpy()
    shape = predicts.shape
    batch_size = shape[0]
    class_number = shape[1]
    f1 = util.AverageMeter()
    em = util.AverageMeter()
    acc = util.AverageMeter()
    for k in range(class_number):
        y_pred = predicts[:, k:k + 1].squeeze()
        y_true = targets[:, k:k + 1].squeeze()
        # f1_score matches
        f1_val = f1_score(y_true, y_pred, average=average)
        f1.update(f1_val)
        # accuracy matches
        acc_val = accuracy_score(y_true, y_pred)
        acc.update(acc_val)
        # extract match
        for k in range(batch_size):
            for _p, _t in zip(y_pred,y_true):
                if _p == _t:
                    em.update(1)
                else:
                    em.update(0)
    return f1.avg * 100, acc.avg * 100, em.avg * 100

def init_from_scratch(args, train_exs, dev_exs,test_exs):
    """New model, new data, new dictionary."""
    # Create a feature dict out of the annotations in the data
    logger.info('-' * 100)
    logger.info('Generate features')
    feature_dict = util.build_feature_dict(args, train_exs)
    logger.info('Num features = %d' % len(feature_dict))
    logger.info(feature_dict)
    # Build a dictionary from the data questions + documents (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build word dictionary')
    word_dict = util.build_word_dict(args, train_exs + dev_exs + test_exs)
    logger.info('Num words = %d' % len(word_dict))
    # Build a char dictionary from the data questions + documents (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build char dictionary')
    char_dict = util.build_char_dict(args, train_exs + dev_exs + test_exs)
    logger.info('Num chars = %d' % len(char_dict))

    # Initialize model
    model = DocReader(config.get_model_args(args), word_dict, char_dict, feature_dict)

    # Load pretrained embeddings for words in dictionary
    if args.words_embedding_file:
        model.load_embeddings(word_dict.tokens(), args.words_embedding_file)
    if args.char_embedding_file:
        model.load_char_embeddings(char_dict.tokens(), args.char_embedding_file)

    return model
# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats,train_saver):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = util.AverageMeter()
    epoch_time = util.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        train_loss.update(model.update(ex))
        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.4f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()
    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)