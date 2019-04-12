import argparse
import time
import os
import json

from script.preprocess import load_dataset,process_dataset
from utils.config import DATA_DIR,OUT_DIR,EMBED_DIR
def main():
    # -----------------------------------------------------------------------------
    # Commandline options
    # -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',default=DATA_DIR, type=str, help='Path to AI data directory')
    parser.add_argument('--out_dir',default=OUT_DIR,type=str, help='Path to output file dir')
    parser.add_argument('--split',default="",type=str, help='Filename for train/dev split')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--tokenizer', type=str, default='pyltp')
    args = parser.parse_args()
    if args.split in ["training","validation"]:
        out_file = os.path.join(args.out_dir, '%s-processed-%s.json' % (args.split, args.tokenizer))
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        if not os.path.exists(out_file):
            t0 = time.time()
            in_file = os.path.join(args.data_dir, "sentiment_analysis_"+args.split + 'set.csv')
            print('Loading dataset %s' % in_file)
            dataset = load_dataset(in_file)
            print('Will write to file %s' % out_file)
            with open(out_file, 'w') as f:
                for ex in process_dataset(dataset, args.tokenizer, args.num_workers):
                    f.write(json.dumps(ex) + '\n')
            print('Total time: %.4f (s)' % (time.time() - t0))
        else:
            print("file:" + out_file + " exists!")
if __name__ == "__main__":
    main()