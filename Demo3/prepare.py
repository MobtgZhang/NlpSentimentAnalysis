import argparse
import time
import os
import json

from script.preprocess import load_dataset,process_dataset,prepare_embeddings
from utils.config import DATA_DIR,OUT_DIR,EMBED_DIR
def main():
    # -----------------------------------------------------------------------------
    # Commandline options
    # -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',default=DATA_DIR, type=str, help='Path to AI data directory')
    parser.add_argument('--out_dir',default=OUT_DIR,type=str, help='Path to output file dir')
    parser.add_argument('--split',default="",type=str, help='Filename for train/dev split')
    parser.add_argument('--emb_size', default=300, type=int, help='embedding size')
    parser.add_argument("--emb_path",default=EMBED_DIR,type=str,help="word embedding root file")
    parser.add_argument("--type", default="words", type=str, help="word embedding root file")
    parser.add_argument("--sep_raw",default="",type=str,help="raw data embedding file training")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--tokenizer', type=str, default='pyltp')
    parser.add_argument('--sg', default=1, type=int, help='0 for COBW or 1 for SkipGram')
    args = parser.parse_args()
    if args.split in ["train","test","validation"]:
        out_file = os.path.join(args.out_dir, '%s-processed-%s.json' % (args.split, args.tokenizer))
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        if not os.path.exists(out_file):
            t0 = time.time()
            in_file = os.path.join(args.data_dir, args.split + 'set.csv')
            print('Loading dataset %s' % in_file)
            dataset = load_dataset(in_file)

            print('Will write to file %s' % out_file)
            with open(out_file, 'w') as f:
                for ex in process_dataset(dataset, args.tokenizer, args.num_workers):
                    f.write(json.dumps(ex) + '\n')
            print('Total time: %.4f (s)' % (time.time() - t0))
        else:
            print("file:" + out_file + " exists!")
    if not args.sep_raw == "":
        if not os.path.exists(args.sep_raw):
            print("file: "+args.sep_raw +" not exists!")
        else:
            out_root = os.path.join(args.emb_path, args.type)
            if args.sg == 1:
                out_file = os.path.join(out_root, args.type + "_emb_skipGram.emb")
            else:
                out_file = os.path.join(out_root, args.type + "_emb_COBW.emb")
            if not os.path.exists(out_file):
                prepare_embeddings(args)
            else:
                print("file:" + out_file + " exists!")
if __name__ == "__main__":
    main()