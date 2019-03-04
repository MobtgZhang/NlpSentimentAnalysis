import os
import subprocess
import logging
logger = logging.getLogger()

import config

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
    runtime.add_argument('--batch-size', type=int, default=45,
                         help='Batch size for training')
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

