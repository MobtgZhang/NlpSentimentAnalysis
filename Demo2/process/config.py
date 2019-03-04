import os
import absl.flags as flags
import torch.backends.cudnn as cudnn

home = os.path.expanduser(".")
# The model defination of the file
segment_file = os.path.join(home,"ltp_data_v3.4.0")
# The defination of the raw data ,the first data,named raw_data
raw_train = os.path.join(home,"raw_data","ai_challenger_sentiment_analysis_trainingset_20180816","sentiment_analysis_trainingset.csv")
raw_validation = os.path.join(home,"raw_data","ai_challenger_sentiment_analysis_validationset_20180816","sentiment_analysis_validationset.csv")
raw_test = os.path.join(home,"raw_data","ai_challenger_sentiment_analysis_testa_20180816","sentiment_analysis_testseta.csv")
flags.DEFINE_string("raw_train", raw_train, "")
flags.DEFINE_string("raw_validation", raw_validation, "")
flags.DEFINE_string("raw_test", raw_test, "")

# The processed data (labeled)
processed_csv_root = os.path.join(home,"AI_Challenger2018")
train_csv = os.path.join(home,"AI_Challenger2018","sentiment_analysis_trainingset.csv")
validation_csv = os.path.join(home,"AI_Challenger2018","sentiment_analysis_validationset.csv")
test_csv = os.path.join(home,"AI_Challenger2018","sentiment_analysis_testset.csv")
flags.DEFINE_string("processed_csv_root", processed_csv_root, "")
flags.DEFINE_string("train_csv", train_csv, "location of the training data, should be a csv file")
flags.DEFINE_string("validation_csv", validation_csv, "location of the validation data, should be a csv file")
flags.DEFINE_string("test_csv", test_csv, "location of the test data, should be a csv file")
# The processed data (segmented and labeled npz files)
# Save data files
save_datafile = os.path.join(home,"data_processed")
train_npz = os.path.join(home,save_datafile,"train_npz.npz")
test_npz = os.path.join(home,save_datafile,"test_npz.npz")
validate_npz = os.path.join(home,save_datafile,"validation_npz.npz")
flags.DEFINE_string("save_datafile", save_datafile, "")
flags.DEFINE_string("train_npz", train_npz, "")
flags.DEFINE_string("test_npz", test_npz, "")
flags.DEFINE_string("validate_npz", validate_npz, "")
flags.DEFINE_integer("validate_batch_size",60,"")
flags.DEFINE_integer("test_batch_size",60,"")
# The Dictionary file
vocab_npz = os.path.join(home,save_datafile,"vocab_npz.npz")
flags.DEFINE_string("vocab_npz",vocab_npz,"")

# word embedding file
word_embFile = os.path.join(home,"PretrainedWordEmb","sgns.wiki.bigram-char")
flags.DEFINE_string("word_embFile",word_embFile,"")

WordPretrained = os.path.join(home,"PretrainedEMB")
flags.DEFINE_string("WordPretrained",WordPretrained,"")
# model save files
save_statics_file = os.path.join(home,"log")
flags.DEFINE_string("save_statics_file",save_statics_file,"")

# cache file
cache_file = os.path.join(home,"tmp")
flags.DEFINE_string("cache_file",cache_file,"")
# Embedding save file
emb_save_COBW = os.path.join(home,"data_processed","WordEmbCOBW.npz")
emb_save_SkipGram = os.path.join(home,"data_processed","WordEmbSkipGram.npz")
flags.DEFINE_string("emb_save_COBW",emb_save_COBW,"")
flags.DEFINE_string("emb_save_SkipGram",emb_save_SkipGram,"")
# Some of values of the model defination
flags.DEFINE_string("type_gram",default="COBW",help="this is the method")
flags.DEFINE_integer("seed",default=111,help="random seed")
flags.DEFINE_integer("emb_size",default=300,help="size of word embeddings")
flags.DEFINE_integer("nhid",default=100,help="number of hidden units per layer")
flags.DEFINE_integer("nlayers",default=2,help="number of layers in BiLSTM")
flags.DEFINE_integer("attention_unit",default=25,help="number of attention unit")
flags.DEFINE_integer("attention_hops",default=15,help="number of attention hops, for multi-hop attention model")
flags.DEFINE_float("drop_out",default=0.2,help="dropout applied to layers (0 = no dropout)")
flags.DEFINE_float("drop_att",default=0.2,help="dropout applied to layers (0 = no dropout)")
flags.DEFINE_float("dropout",default=0.2,help="dropout applied to layers (0 = no dropout)")
flags.DEFINE_float("grad_clip",default=0.5,help="clip to prevent the too large grad in LSTM")
flags.DEFINE_float("learning_rate",default=0.01,help="initial learning rate")
flags.DEFINE_integer("nfc",default=100,help="hidden (fully connected) layer size for classifier MLP")
flags.DEFINE_integer("epochs",default=1,help="upper epoch limit")
flags.DEFINE_integer("batch_size",default=32,help="batch size for training")
flags.DEFINE_integer("aggregation_hid",default=100,help="aggregation hidden layer number")
flags.DEFINE_integer("class_number",default=4,help="number of classes")
flags.DEFINE_integer("seperate_hops",default=6,help="number of hops")
flags.DEFINE_string("optimizer",default="Adam",help="type of optimizer")
flags.DEFINE_float("penalization_coeff",default=1.0,help="the penalization coefficient")
flags.DEFINE_integer("labels",default=20,help="the number of labels")
flags.DEFINE_integer("text_length",default=300,help="the length of the text")
flags.DEFINE_float("ema_decay",default=0.9999,help="Exponential moving average decay")
flags.DEFINE_integer("warm_up",default=1000,help="This is the warm_up rate")
flags.DEFINE_integer("log_interval",default=200,help="report interval")
flags.DEFINE_bool("use_gpu",default=False,help="the cuda usage")

config = flags.FLAGS
cudnn.enabled = False
