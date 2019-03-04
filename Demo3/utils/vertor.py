from collections import Counter
import torch


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

    return document, document_char, c_features

def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    NUM_INPUTS = 6
    NUM_TARGETS = 20
    NUM_EXTRA = 0

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




    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x, x_c, x_f, x_mask

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][NUM_INPUTS]):
            y_s = torch.cat([ex[NUM_INPUTS] for ex in batch])
            y_e = torch.cat([ex[NUM_INPUTS+1] for ex in batch])
        else:
            y_s = [ex[NUM_INPUTS] for ex in batch]
            y_e = [ex[NUM_INPUTS+1] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x, x_c, x_f, x_mask