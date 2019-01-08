from preproc import preproc
from absl import app
from config import config
from process import train_entry,test_entry
def main(_):
    # load embeddings
    if config.mode == "train":
        train_entry("BiLSTMNet")
    elif config.mode == "data":
        seplength = 1000
        delay = 10
        # train dataset has 90000 
        preproc(config.train_file,config.train_npz,seplength,delay)
        # validate dataset has 15000 
        preproc(config.validate_file,config.validate_npz,seplength,delay)
        # test dataset has 15000 
        preproc(config.test_file,config.test_npz,seplength,delay)
    elif config.mode == "debug":
        train_entry("BiLSTMNet")
    elif config.mode == "test":
        test_entry()
    else:
        print("Unknown mode")
        exit(0)
if __name__ == "__main__":
    app.run(main)
