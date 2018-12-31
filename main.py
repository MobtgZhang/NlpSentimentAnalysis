from preproc import preproc
from absl import app
from config import config
from process import train_entry
def test_entry(config):
    print("test_entry")
def main(_):
    # load embeddings
    if config.mode == "train":
        train_entry()
    elif config.mode == "data":
        seplength = 1000
        delay = 50
        # train dataset has 90000 
        preproc(config.train_file,config.train_npz,config.train_vocab_file,seplength,delay)
        # validate dataset has 15000 
        preproc(config.validation_file,config.validation_npz,config.validate_vocab_file,seplength,delay)
        # test dataset has 15000 
        preproc(config.test_file,config.test_npz,config.test_vocab_file,seplength,delay)
    elif config.mode == "debug":
        train_entry()
    elif config.mode == "test":
        test_entry(config)
    else:
        print("Unknown mode")
        exit(0)
if __name__ == "__main__":
    app.run(main)
