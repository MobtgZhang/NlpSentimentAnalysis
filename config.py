import os
import absl.flags as flags

# home = os.path.expanduser(".")
home = "/home/asus"
# The model defination of the file
segment_model_file = os.path.join(home,"ltp_data_v3.4.0")
# The defination of the raw data ,the first data,named AI_Challenger2018
train_file = os.path.join(home,"AI_Challenger2018","ai_challenger_sentiment_analysis_trainingset_20180816","sentiment_analysis_trainingset.csv")
validation_file = os.path.join(home,"AI_Challenger2018","ai_challenger_sentiment_analysis_validationset_20180816","sentiment_analysis_validationset.csv")
test_file = os.path.join(home,"AI_Challenger2018","ai_challenger_sentiment_analysis_testa_20180816","sentiment_analysis_testa.csv")
# The defination of the word embedding 
wordembedding_file = os.path.join(home,"Word2Vec","sgns.baidubaike.bigram-char")
flags.DEFINE_string("mode", "train", "train/debug/test")
# The defination of cache file
cache_file = os.path.join(home,"tmp")
# The Processed Data

flags.DEFINE_string("segment_model_file", segment_model_file, "")
flags.DEFINE_string("wordembedding_file", wordembedding_file, "")

flags.DEFINE_string("train_file", train_file, "")
flags.DEFINE_string("validation_file", validation_file, "")
flags.DEFINE_string("test_file", test_file, "")

flags.DEFINE_integer("word_dim",300, "Embedding dimension for word")
flags.DEFINE_integer("char_dim",300, "Embedding dimension for char")
flags.DEFINE_integer("text_length",1000, "the length of the text")


config = flags.FLAGS
