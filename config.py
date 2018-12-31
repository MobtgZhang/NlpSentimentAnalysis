import os
import absl.flags as flags
import torch
import torch.backends.cudnn as cudnn

# home = os.path.expanduser(".")
home = "/home/asus"
# The model defination of the file
segment_model_file = os.path.join(home,"ltp_data_v3.4.0")
# The defination of the raw data ,the first data,named AI_Challenger2018
train_file = os.path.join(home,"AI_Challenger2018","NewCode5","AI-Challenger2018","sentiment_analysis_trainingset.csv")
validation_file = os.path.join(home,"AI_Challenger2018","NewCode5","AI-Challenger2018","sentiment_analysis_validationset.csv")
test_file = os.path.join(home,"AI_Challenger2018","NewCode5","AI-Challenger2018","sentiment_analysis_testset.csv")
# Save data files
save_datafile = os.path.join(home,"AI_Challenger2018","NewCode5","data_processed")
train_npz = os.path.join(home,save_datafile,"train_npz.npz")
test_npz = os.path.join(home,save_datafile,"test_npz.npz")
validate_npz = os.path.join(home,save_datafile,"validation_npz.npz")

train_vocab_file = os.path.join(home,save_datafile,"train_vocab.txt")
validate_vocab_file = os.path.join(home,save_datafile,"validate_vocab.txt")
test_vocab_file = os.path.join(home,save_datafile,"test_vocab.txt")
# The defination of cache file
cache_file = os.path.join(home,"AI_Challenger2018","NewCode5","tmp")
# The defination of the word embedding 
wordembedding_file = os.path.join(home,"Word2Vec","sgns.baidubaike.bigram-char")
flags.DEFINE_string("mode", "train", "train/data/debug/test")

# The flag Data
flags.DEFINE_string("train_file", train_file, "")
flags.DEFINE_string("validate_file", validation_file, "")
flags.DEFINE_string("test_file", test_file, "")
flags.DEFINE_string("cache_file",cache_file,"")

flags.DEFINE_string("train_npz", train_npz, "")
flags.DEFINE_string("test_npz", test_npz, "")
flags.DEFINE_string("validate_npz", validate_npz, "")
flags.DEFINE_string("save_datafile", save_datafile, "")


flags.DEFINE_string("train_vocab_file", train_vocab_file, "")
flags.DEFINE_string("validate_vocab_file", validate_vocab_file, "")
flags.DEFINE_string("test_vocab_file", test_vocab_file, "")

flags.DEFINE_string("segment_model_file", segment_model_file, "")
flags.DEFINE_string("wordembedding_file", wordembedding_file, "")

flags.DEFINE_integer("word_dim",300, "Embedding dimension for word")
flags.DEFINE_integer("char_dim",300, "Embedding dimension for char")
flags.DEFINE_integer("text_length",1000, "the length of the text")
flags.DEFINE_integer("labels",20, "the labels of the text")

flags.DEFINE_integer("train_word_sep_size",1000, "the batchsize to seperate raw data for word")
flags.DEFINE_integer("train_delay",1000, "train raw dataset delay time")
flags.DEFINE_integer("validation_delay",500, "validation raw dataset delay time")
flags.DEFINE_integer("test_delay",500, "test raw dataset delay time")

# model definations
flags.DEFINE_integer("num_epochs",15, "the numbers for training")
flags.DEFINE_integer("batch_size",50, "the batchsize to seperate raw data for trainig data")
flags.DEFINE_float("learning_rate",0.8, "the learningrate for trainig data")

# model save files
save_statics_file = os.path.join(home,"AI_Challenger2018","NewCode5")
model_save_file = os.path.join(home,save_statics_file,"model.pt")
picture_save_file = os.path.join(home,save_statics_file,"pic.png")

flags.DEFINE_string("picture_save_file", picture_save_file, "")
flags.DEFINE_string("model_save_file", model_save_file, "")

config = flags.FLAGS
device = "cuda" if torch.cuda.is_available() else "cpu"
flags.DEFINE_string("device", device, "")

cudnn.enabled = False
