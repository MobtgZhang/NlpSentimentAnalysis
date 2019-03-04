from absl import app
from process.preproc import preproc
from script.train import train_entry
def main(_):
    preproc()
    train_entry("SABiGRU")
if __name__ == "__main__":
    app.run(main)