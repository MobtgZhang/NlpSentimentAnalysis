from config import config
from preproc import preproc
from absl import app
def train_entry(config):
    print("train_entry")
def test_entry(config):
    print("test_entry")
def main(_):
    # load embeddings
    if config.mode == "train":
        train_entry(config)
    elif config.mode == "data":
        preproc(config)
    elif config.mode == "debug":
        train_entry(config)
    elif config.mode == "test":
        test_entry(config)
    else:
        print("Unknown mode")
        exit(0)
if __name__ == "__main__":
    app.run(main)
