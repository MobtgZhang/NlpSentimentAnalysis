import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from torch.autograd import Variable
import random
import time
import math
import os
from sklearn.metrics import f1_score,accuracy_score

from script.utils import pad_samples,encode_samples,Dictionary
from process.config import config
from model.model import SABiLSTMOLD,RNNs_AttNone,SABiRNN

def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix

        ret = (torch.sum(torch.sum(torch.sum((mat ** 2), 1), 1)).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')
def load_datasets(filename):
    data = np.load(filename)
    sentences = list(data['sentences'])
    labels = data['labels']
    return sentences,labels
def prepare_labels(labels):
    labels = labels + np.ones(labels.shape)*2
    return labels
def prepare_datasets(sentences,labels,word_to_idx,mode,text_length = 1000):
    print("Preparing "+mode+" datasets ... ...")
    features = torch.LongTensor(pad_samples(encode_samples(sentences,word_to_idx),text_length))
    labels = prepare_labels(labels)
    labels = torch.LongTensor(labels)
    print("The "+mode+" datasets has been loaded!")
    return features,labels
def save_model_data(args,mode):
    net = args["model"]
    # save model
    if not os.path.exists(config.save_statics_file):
        os.mkdir(config.save_statics_file)
    save_path = os.path.join(config.save_statics_file, args['modelname'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, args['modelname'] + ".pkt")
    net.to("cpu")
    torch.save(net, save_file)

    # save loss list
    total_loss_list = args['total_loss_list']
    total_pure_loss_list = args['total_pure_loss_list']

    save_file = os.path.join(config.save_statics_file,args["modelname"],mode)
    if not os.path.exists(save_file):
        os.mkdir(save_file)
    loss_file = os.path.join(save_file,"loss.npz")

    np.savez(loss_file,total_loss_list = total_loss_list,total_pure_loss_list = total_pure_loss_list)

    # save one epoch predict list
    label_pred_list = args["label_pred_list"]
    label_true_list = args["label_pred_list"]
    label_pred_list = np.concatenate(label_pred_list, axis=0)
    label_true_list = np.concatenate(label_true_list, axis=0)
    label_pred_npz = os.path.join(config.save_statics_file, args["modelname"], mode, "label_pred_npz.npz")
    np.savez(label_pred_npz, label_pred_list=label_pred_list, label_true_list=label_true_list)

    # save f1_score and accuracy score
    acc_mean_list = args["acc_mean_list"]
    f1_mean_list = args["f1_mean_list"]
    np.savez(label_pred_npz, acc_mean_list=acc_mean_list, f1_mean_list=f1_mean_list)
def validate(args,epoch):
    net = args["model"]
    criterion = args["criterion"]
    I = args['I']
    if config.use_gpu:
        net = net.cuda()
    net.eval()
    dictionary = Dictionary(path=config.vocab_npz)
    sentences_validate, labels_validate = load_datasets(config.validate_npz)
    validate_features, validate_labels = prepare_datasets(sentences_validate, labels_validate, dictionary.word2idx,
                                                          "validate", config.text_length)
    validation_set = torch.utils.data.TensorDataset(validate_features,validate_labels)
    validation_iter = torch.utils.data.DataLoader(validation_set, config.validate_batch_size, shuffle=True)

    total_pure_loss = 0
    total_loss = 0
    m = 0
    label_pred_list = []
    label_true_list = []
    with torch.no_grad():
        for batch, (validate_feature, validate_label) in enumerate(validation_iter):
            m += 1
            if config.use_gpu:
                validate_feature = Variable(validate_feature).cuda()
                validate_label = Variable(validate_label).cuda()
            if len(validate_feature) < config.validate_batch_size:
                validate_batch_size = len(validate_feature)
            else:
                validate_batch_size = config.validate_batch_size
            hidden = net.init_hidden(validate_batch_size)
            validate_score, attention = net.forward(validate_feature, hidden)

            label_pred = torch.argmax(validate_score, dim=1)
            label_true = validate_label

            label_pred = label_pred.cpu().data.numpy()
            label_pred_list.append(label_pred)
            label_true = label_true.cpu().data.numpy()
            label_true_list.append(label_true)

            # calculate the F1 score
            f1_mean, acc_mean = evalate(label_pred, label_true)

            loss = criterion(validate_score, validate_label)
            total_pure_loss += loss.data

            if attention is not None:  # add penalization term
                attentionT = torch.transpose(attention, 1, 2).contiguous()
                extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])
                loss += config.penalization_coeff * extra_loss
            total_loss += loss.data
            print('validation: |epoch {:3d}| loss {:5.4f} | pure loss {:5.4f}|f1_score {:5.2f}|accuracy_score {:5.2f}'.format(epoch,
                total_loss / m, total_pure_loss / m, f1_mean * 100, acc_mean * 100))
    # save one epoch predict list
    label_pred_list = np.concatenate(label_pred_list, axis=0)
    label_true_list = np.concatenate(label_true_list, axis=0)
    f1_mean, acc_mean = evalate(label_pred_list, label_true_list)

    save_file = os.path.join(config.save_statics_file, args["modelname"],"validation")
    if not os.path.exists(save_file):
        os.mkdir(save_file)
    label_pred_npz = os.path.join(save_file,"label_pred_npz.npz")
    np.savez(label_pred_npz, label_pred_list=label_pred_list, label_true_list=label_true_list)
    return f1_mean,acc_mean,total_loss.cpu().data.numpy()/m,total_pure_loss.cpu().data.numpy()/m

def test_entry(args):
    # load model
    save_path = os.path.join(config.save_statics_file, args['modelname'])
    save_file = os.path.join(save_path, args['modelname'] + ".pkt")
    criterion = args["criterion"]
    I = args["I"]

    net = torch.load(save_file)
    if config.use_gpu:
        net = net.cuda()
    net.eval()
    dictionary = Dictionary(path=config.vocab_npz)
    # preparing test data
    sentences_test, labels_test = load_datasets(config.test_npz)
    test_features, test_labels = prepare_datasets(sentences_test, labels_test,dictionary.word2idx, "test",config.text_length)
    test_set = torch.utils.data.TensorDataset(test_features,test_labels)
    test_iter = torch.utils.data.DataLoader(test_set, config.test_batch_size, shuffle=True)

    total_pure_loss = 0
    total_loss = 0
    m = 0
    label_pred_list = []
    label_true_list = []
    acc_mean_list = []
    f1_mean_list = []
    with torch.no_grad():
        for batch, (test_feature, test_label) in enumerate(test_iter):
            m += 1
            if config.use_gpu:
                test_feature = Variable(test_feature).cuda()
                test_label = Variable(test_label).cuda()
            if len(test_feature) < config.validate_batch_size:
                test_batch_size = len(test_feature)
            else:
                test_batch_size = config.validate_batch_size
            hidden = net.init_hidden(test_batch_size)
            test_score, attention = net.forward(test_feature, hidden)

            label_pred = torch.argmax(test_score, dim=1)
            label_true = test_label

            label_pred = label_pred.cpu().data.numpy()
            label_pred_list.append(label_pred)
            label_true = label_true.cpu().data.numpy()
            label_true_list.append(label_true)

            # calculate the F1 score
            f1_mean, acc_mean = evalate(label_pred, label_true)
            acc_mean_list.append(acc_mean)
            f1_mean_list.append(f1_mean)

            loss = criterion(test_score, test_label)
            total_pure_loss += loss.data

            if attention is not None:  # add penalization term
                attentionT = torch.transpose(attention, 1, 2).contiguous()
                extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])
                loss += config.penalization_coeff * extra_loss
            total_loss += loss.data
            print('test:  | loss {:5.4f} | pure loss {:5.4f}|f1_score {:5.2f}|accuracy_score {:5.2f}'.format(total_loss / m,total_pure_loss / m,f1_mean*100,acc_mean*100))
        # save one epoch predict list
        label_pred_list = np.concatenate(label_pred_list, axis=0)
        label_true_list = np.concatenate(label_true_list, axis=0)
        save_file = os.path.join(config.save_statics_file, args["modelname"],"test")
        if not os.path.exists(save_file):
            os.mkdir(save_file)
        save_label_npz = os.path.join(save_file,"label_pred_npz.npz")
        np.savez(save_label_npz, label_pred_list=label_pred_list, label_true_list=label_true_list)
def evalate(label_pred,label_true,average = "macro"):
    # binary,micro,macro
    f1_sum = 0
    acc_sum = 0
    for k in range(20):
        y_pred = label_pred[:, k:k + 1].squeeze()
        y_true = label_true[:, k:k + 1].squeeze()
        f1_sum += f1_score(y_true,y_pred,average=average)
        acc_sum += accuracy_score(y_true,y_pred)
    f1_mean = f1_sum/20
    acc_mean = acc_sum/20
    return f1_mean,acc_mean
def train(args,epoch_number):
    # make the torch module train mode
    net = args["model"]
    net.train()

    criterion = args['criterion']
    optimizer = args['optimizer']
    scheduler = args['scheduler']
    I = args['I']
    # ema = args['EMA']
    total_loss_list = args['total_loss_list']
    total_pure_loss_list = args['total_pure_loss_list']

    total_loss = 0
    total_pure_loss = 0  # without the penalization term
    start_time = time.time()
    train_set = torch.utils.data.TensorDataset(args["train_features"], args["train_labels"])
    train_iter = torch.utils.data.DataLoader(train_set, config.batch_size, shuffle=True)

    label_pred_list = args["label_pred_list"]
    label_true_list = args["label_pred_list"]
    acc_mean_list = args["acc_mean_list"]
    f1_mean_list = args["f1_mean_list"]

    m = 0
    best_f1_score = 0
    best_acc_score = 0
    for batch,(feature,label) in enumerate(train_iter):
        m += 1
        net.zero_grad()
        if config.use_gpu:
            feature = Variable(feature).cuda()
            label = Variable(label).cuda()

        if len(feature) <config.batch_size:
            batch_size = len(feature)
        else:
            batch_size = config.batch_size
        hidden = net.init_hidden(batch_size)
        score,attention = net.forward(feature,hidden)

        label_pred = torch.argmax(score,dim=1)
        label_true = label

        label_pred = label_pred.cpu().data.numpy()
        label_pred_list.append(label_pred)
        label_true = label_true.cpu().data.numpy()
        label_true_list.append(label_true)

        # calculate the F1 score
        f1_mean,acc_mean = evalate(label_pred, label_true)
        best_f1_score = f1_mean if f1_mean >best_f1_score else best_f1_score
        best_acc_score = f1_mean if acc_mean > best_acc_score else best_acc_score

        acc_mean_list.append(acc_mean)
        f1_mean_list.append(f1_mean)
        loss = criterion(score, label)
        total_pure_loss += loss.data

        if attention is not None:  # add penalization term
            attentionT = torch.transpose(attention, 1, 2).contiguous()
            extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])
            loss += config.penalization_coeff * extra_loss
        optimizer.zero_grad()
        loss.backward()
        '''
        for name, p in net.named_parameters():
            if p.requires_grad: ema.update_parameter(name, p)
        '''
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.data

        end_time = time.time()
        elapsed = end_time - start_time
        total_loss_list.append(total_loss/m)
        total_pure_loss_list.append(total_pure_loss/m)

        out_str = 'training:  | epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f} | pure loss {:5.4f}|' \
                  ' f1_score: {:5.2f} | accuracy: {:5.2f}'
        print(out_str.format(epoch_number, batch, len(args["train_features"])//config.batch_size,
                                 elapsed * 1000 / config.log_interval, total_loss / m,
                                 total_pure_loss / m,f1_mean*100,acc_mean*100))
    # save data
    save_model_data(args,mode="train")
    return best_acc_score,best_f1_score
def train_entry(modelname):
    # Set the random seed manually for reproducibility.
    config.use_gpu = torch.cuda.is_available()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        if not config.use_gpu:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    # loading the dictionary
    print("Begin loading vocabulary.")
    dictionary = Dictionary(path=config.vocab_npz)
    print("Begin preparing embedding files.")
    if config.type_gram == "COBW":
        embedding = np.load(config.emb_save_COBW)
    else:
        embedding = np.load(config.emb_save_SkipGram)
    n_token = len(dictionary)

    # load model
    save_path = os.path.join(config.save_statics_file, modelname)
    save_file = os.path.join(save_path,modelname + ".pkt")
    if not os.path.exists(save_file):
        # preparing model
        if modelname == "SABiLSTMOLD":
            args_config = {
            'drop_out': config.drop_out,
            'drop_att':config.drop_att,
            'dropout':config.dropout,
            'ntoken': n_token,
            'nlayers': config.nlayers,
            'emb_size': config.emb_size,
            'pooling': 'all',
            'attention-unit': config.attention_unit,
            'attention-hops': config.attention_hops,
            'nfc': config.nfc,
            'dictionary': dictionary,
            'word-vector': config.word_embFile,
            'class_number': config.class_number,
            'labels': config.labels,
            'text_length': config.text_length,
            'seperate-hops': config.seperate_hops,
            'word_embedding_file':config.word_embFile,
            'word_dim':config.emb_size,
            "embedding": embedding,
            }
            model = SABiLSTMOLD(args_config)
        elif modelname == "SABiLSTM" or modelname == "SABiGRU":
            args_config = {
            'drop_out': config.drop_out,
            'drop_att':config.drop_att,
            'dropout':config.dropout,
            'ntoken': n_token,
            'nlayers': config.nlayers,
            'emb_size': config.emb_size,
            'pooling': 'all',
            'attention-unit': config.attention_unit,
            'attention-hops': config.attention_hops,
            'nfc': config.nfc,
            'dictionary': dictionary,
            'word-vector': config.word_embFile,
            'class_number': config.class_number,
            'labels': config.labels,
            'text_length': config.text_length,
            'seperate_hops': config.seperate_hops,
            'word_embedding_file':config.word_embFile,
            'word_dim':config.emb_size,
            "embedding": embedding,
            "aggregation_hid":config.aggregation_hid
            }
            if modelname == "SABiLSTM":
                args_config['rnn_type'] = "lstm"
            else:
                args_config['rnn_type'] = "gru"
            model = SABiRNN(args_config)
        elif modelname == "BiLSTM" or modelname == "BiGRU" or modelname == "BiSRU":
            args_config = {
            'drop_out':config.drop_out,
            'drop_att':config.drop_att,
            'dropout': config.dropout,
            'text_length': config.text_length,
            'class_number': config.class_number,
            'dictionary': dictionary,
            'ntoken': n_token,
            'nlayers': config.nlayers,
            'emb_size': config.emb_size,
            'pooling': 'all',
            'word-vector': config.word_embFile,
            'class-number': config.class_number,
            'labels': config.labels,
            'word_embedding_file': config.word_embFile,
            'word_dim': config.emb_size,
            "embedding": embedding,
            "rnn_model":modelname,
            "seperate_hops":config.seperate_hops
            }
            model = RNNs_AttNone(args_config)
        else:
            raise Exception("unknown model:" + modelname)
    else:
        print("model exists! Loading ...")
        model = torch.load(save_file)
    if config.use_gpu:
        model = model.cuda()

    I = Variable(torch.zeros(config.text_length,config.attention_hops,config.attention_hops))
    for i in range(config.batch_size):
        for j in range(config.attention_hops):
            I.data[i][j][j] = 1
    if config.use_gpu:
        I = I.cuda()
    criterion = nn.CrossEntropyLoss()
    params = filter(lambda param: param.requires_grad, model.parameters())
    base_lr = 1.0
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(lr=base_lr, params=params)
    elif config.optimizer == 'SGD':
        optimizer = optim.SGD(lr=base_lr, params=params)
    else:
        raise Exception('For other optimizers, please add it yourself. '
                        'supported ones are: SGD and Adam.')
    '''
    ema = EMA(config.ema_decay)
    for name, p in model.named_parameters():
        if p.requires_grad: ema.set(name, p)
    '''
    cr = config.learning_rate / math.log2(config.warm_up)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: cr * math.log2(ee + 1) if ee < config.warm_up else config.learning_rate)

    # train datasets and validation datasets preparing
    print("Begin preparing datatsets.")
    sentences_train, labels_train = load_datasets(config.train_npz)
    train_features, train_labels = prepare_datasets(sentences_train, labels_train, dictionary.word2idx, "train",config.text_length)

    total_loss_list = []
    total_pure_loss_list = []
    label_pred_list = []
    label_true_list = []
    f1_mean_list = []
    acc_mean_list = []
    train_args = {
        "train_features":train_features,
        "train_labels":train_labels,
        "model":model,
        "modelname":modelname,
        "optimizer":optimizer,
        "criterion":criterion,
        "scheduler":scheduler,
        #"EMA":ema,
        "I":I,
        "total_loss_list":total_loss_list,
        "total_pure_loss_list":total_pure_loss_list,
        "label_pred_list":label_pred_list,
        "label_true_list":label_true_list,
        "f1_mean_list":f1_mean_list,
        "acc_mean_list":acc_mean_list
    }
    try:
        best_acc_list = []
        best_f1_list = []
        for epoch in range(config.epochs):
            best_acc_score,best_f1_score = train(train_args,epoch)
            best_acc_list.append(best_acc_score)
            best_f1_list.append(best_f1_score)
            save_file = os.path.join(config.save_statics_file, modelname)
            if not os.path.exists(save_file):
                os.mkdir(save_file)
            save_npz = os.path.join(save_file,"best.npz")
            np.savez(save_npz,best_f1_list = best_f1_list,best_acc_score = best_acc_score)
            # validate the dataset
            validation_args = {
                "model": model,
                "modelname": modelname,
                "criterion": criterion,
                "I": I
            }
            validation_f1 = []
            validation_acc = []
            f1_mean,acc_mean,total_mean_loss,total_pure_mean_loss = validate(validation_args,epoch)
            validation_f1.append(f1_mean)
            validation_acc.append(acc_mean)

            evaluate_data = os.path.join(config.save_statics_file, modelname,"validation")
            if not os.path.exists(evaluate_data):
                os.mkdir(evaluate_data)
            save_file = os.path.join(evaluate_data,"evaluate.npz")
            np.savez(save_file,f1_score = validation_f1,accuracy_score = validation_acc,
                     total_mean_loss=total_mean_loss,total_pure_mean_loss = total_pure_mean_loss)
        # test model
        args = {
                "modelname": modelname,
                "criterion": criterion,
                "I": I
            }
        test_entry(args)
    except KeyboardInterrupt:
        pass
