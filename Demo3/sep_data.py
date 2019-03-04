import os
import argparse

from config import DATA_DIR,OUT_DIR
from script.preprocess import preparing_raw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default="words", type=str, help='type of the segmenting sentences ')
    parser.add_argument('--data_dir',default=DATA_DIR,type=str,help="Path to AI data directory")
    parser.add_argument('--out_dir', default=OUT_DIR, type=str, help="Path to AI data directory")
    args = parser.parse_args()
    out_file = "raw_%s.csv" % args.type
    save_file = os.path.join(args.out_dir, out_file)
    if os.path.exists(save_file):
        print("file: "+save_file+" exists!")
    else:
        file_list = []
        for file in os.listdir(args.data_dir):
            out_file = os.path.join(args.data_dir,file)
            file_list.append(out_file)
        preparing_raw(file_list,save_file,args.type)
if __name__ == "__main__":
    main()