import argparse

import numpy as np
import torch

import logging
logger = logging.getLogger(__name__)
import sys
import os
import json

from script.train_utils import add_train_args,set_defaults,init_from_scratch
from script.train_utils import train,validate_official
from model.model import DocReader
import utils.data as data
from utils import config
import utils.util as util
import utils.vector as vector
# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = util.load_data(args,args.train_file)
    logger.info('Num train examples = %d' % len(train_exs))
    dev_exs = util.load_data(args,args.dev_file)
    logger.info('Num dev examples = %d' % len(dev_exs))
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
            logger.info('Training model from scratch...')
            model = init_from_scratch(args, train_exs, dev_exs)
            # Set up partial tuning of embeddings
            if args.tune_partial > 0:
                logger.info('-' * 100)
                logger.info('Counting %d most frequent question words' %
                            args.tune_partial)
                top_words = util.top_words(
                    args, train_exs, model.word_dict
                )
                for word in top_words[:5]:
                    logger.info(word)
                logger.info('...')
                for word in top_words[-6:-1]:
                    logger.info(word)
                model.tune_embeddings([w[0] for w in top_words])

            # Set up optimizer
            model.init_optimizer()
    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Three datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')
    train_dataset = data.ReaderDataset(train_exs, model)

    if args.sort_by_len:
        train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                args.batch_size,
                                                shuffle=True)
    else:
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        #pin_memory= args.cuda,
    )
    dev_dataset = data.ReaderDataset(dev_exs, model)
    if args.sort_by_len:
        dev_sampler = data.SortedBatchSampler(dev_dataset.lengths(),
                                              args.dev_batch_size,
                                              shuffle=False)
    else:
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.dev_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        #pin_memory=args.cuda,
    )
    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))
    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer': util.Timer(), 'epoch': 0, 'best_f1_score': 0,'best_em_score': 0}
    train_saver = util.DataSaver(args.model_name, "train")
    dev_saver = util.DataSaver(args.model_name, "dev")
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch
        # Train
        train(args, train_loader, model, stats,train_saver)

        # Validate unofficial (train)
        validate_official(args, train_loader, model, stats, train_saver)

        # Validate unofficial (dev)
        result = validate_official(args, dev_loader, model, stats,dev_saver)

        # Save best valid
        if args.valid_metric is None or args.valid_metric == 'no':
            model.save(args.model_file)
        # {'exact_match': exact_match.avg,"f1_score":f1_score_avg.avg}
        if result['exact_match'] > stats['best_em_score']:
            stats['best_em_score'] = result['exact_match']
        if result['f1_score'] > stats['best_f1_score']:
            stats['best_f1_score'] = result['f1_score']
        logger.info('Best f1_score = %.2f Best em_score: = %.2f (epoch %d, %d updates)' %
                    (stats['best_f1_score'], stats['best_em_score'],stats['epoch'], model.updates))
        model.save(args.model_file)
    # save trained data
    train_saver.save(args.model_dir)
    dev_saver.save(args.model_dir)
    return model
if __name__ == '__main__':
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
    # Run!
    model = main(args)
