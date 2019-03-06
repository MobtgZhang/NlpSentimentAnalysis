import os
import argparse
import sys
import logging
import json
logger = logging.getLogger(__name__)
import numpy as np
import torch

import utils.data as data
import utils.vertor as vector
from model.model import DocReader
import utils.util as util
from script.train_utils import validate_official,add_train_args,set_defaults
from utils import config


def test(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = util.load_data(args, args.train_file)
    logger.info('Num train examples = %d' % len(train_exs))
    dev_exs = util.load_data(args, args.dev_file)
    logger.info('Num dev examples = %d' % len(dev_exs))
    test_exs = util.load_data(args, args.test_file)
    logger.info('Num test examples = %d' % len(test_exs))
    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        # Just resume training, no modifications.
        logger.info('Found a checkpoint...')
        checkpoint_file = args.model_file + '.checkpoint'
        model, start_epoch = DocReader.load_checkpoint(checkpoint_file, args)
    else:
        # Training starts fresh. But the model state is either pretrained or
        # newly (randomly) initialized.
        if args.pretrained:
            logger.info('Using pretrained model...')
            model = DocReader.load(args.pretrained, args)
            if args.expand_dictionary:
                logger.info('Expanding dictionary for new data...')
                # Add words in training + dev examples
                words = util.load_words(args, train_exs + dev_exs)
                added_words = model.expand_dictionary(words)
                # Load pretrained embeddings for added words
                if args.words_embedding_file:
                    model.load_embeddings(added_words, args.words_embedding_file)
                logger.info('Expanding char dictionary for new data...')
                # Add words in training + dev examples

                chars = util.load_chars(args, train_exs + dev_exs)
                added_chars = model.expand_char_dictionary(chars)
                # Load pretrained embeddings for added words
                if args.char_embedding_file:
                    model.load_char_embeddings(added_chars, args.char_embedding_file)
        else:
            logger.info('No model pretrained...\nexit')
            exit()
        test_dataset = data.ReaderDataset(test_exs, model)
        if args.sort_by_len:
            test_sampler = data.SortedBatchSampler(test_dataset.lengths(),
                                               args.test_batch_size,
                                               shuffle=False)
        else:
            test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=args.test_batch_size,
                    sampler=test_sampler,
                    num_workers=args.data_workers,
                    collate_fn=vector.batchify,
                    pin_memory=args.cuda,
        )
        # -------------------------------------------------------------------------
        # PRINT CONFIG
        logger.info('-' * 100)
        logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))
        # --------------------------------------------------------------------------
        # TRAIN/VALID LOOP
        logger.info('-' * 100)
        logger.info('Starting testing...')


        stats = {'timer': util.Timer(), 'epoch': 0, 'best_valid': 0}

        # Validate unofficial (dev)
        result = validate_official(args, test_loader, model, stats, mode='dev')

        # Save best valid
        if args.valid_metric is None or args.valid_metric == 'None':
            model.save(args.model_file)
        elif result[args.valid_metric] > stats['best_valid']:
            logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                        (args.valid_metric, result[args.valid_metric],
                         stats['epoch'], model.updates))
            model.save(args.model_file)
            stats['best_valid'] = result[args.valid_metric]
if __name__ == "__main__":
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'WRMCQA Document Reader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    print(args)
    test(args)