import os
import shutil
import time
import pandas as pd
import json
import codecs
import argparse

from utils import config
from script.preprocess import load_dataset,process_dataset
def sepData(filename,length):
    outData = pd.read_csv(filename)
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    set_len = len(outData) // length
    for k in range(length):
        start_id = k*set_len
        end_id = (k+1)*set_len
        outData[start_id:end_id].to_csv(os.path.join("tmp", "data%d.csv"%k), index=None)
def process(filename,tokenizer,num_workers = 5):
    t0 = time.time()
    print('Loading dataset %s' % filename)
    dataset = load_dataset(filename)
    out_file = os.path.basename(filename).split(".")[0]+"-processed-%s.json"%tokenizer
    out_file = os.path.join("tmp",out_file)
    print('Will write to file %s' % out_file)
    with open(out_file, 'w') as f:
        for ex in process_dataset(dataset, tokenizer,num_workers):
            f.write(json.dumps(ex) + '\n')
    print('Total time: %.4f (s)' % (time.time() - t0))
def combine(file_list,save_path):
    num = 0
    with codecs.open(save_path,mode="w",encoding="utf-8") as fileW:
        for file in file_list:
            with codecs.open(file,mode="r",encoding="utf-8") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    fileW.write(line)
                    num+= 1
    print(num)
    exit()
def is_exists(file_list):
    flag = True
    for file in file_list:
        outputfile = os.path.join("tmp", file)
        if not os.path.exists(outputfile):
            flag = False
            break
    return flag
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', default="pyltp", type=str, help='Path to AI data directory')
    parser.add_argument('--len', default=5, type=int, help='length of seprating data')
    parser.add_argument('--dataset',default="train",type=str,help='dir of seprating data')
    args = parser.parse_args()
    outputfile = os.path.join(config.OUT_DIR, args.dataset + "-processed-%s.json" % args.tokenizer)
    if os.path.exists(outputfile):
        print("file: " + outputfile + " exists!")
        exit()
    data_list = ["data%d.csv" % i for i in range(0, args.len)]
    flag = is_exists(data_list)
    if flag:
        str_line = " ".join(data_list)
        print("file: "+str_line + " exists!")
    else:
        print("seperating ...")
        sepData(os.path.join(config.DATA_DIR, args.dataset + "set.csv"), args.len)

    if os.path.exists(outputfile):
        print("file: " + outputfile + " exists!")
    else:
        file_list = [os.path.basename(f).split(".")[0] + "-processed-%s.json" % args.tokenizer for
                     f in data_list]
        flag = is_exists(file_list)
        if flag:
            str_line = " ".join(file_list)
            print("file: " + str_line + " exists!")
        else:
            print("segmenting ...")
            for file in data_list:
                filename = file.split(".")[0] + "-processed-%s.json"%args.tokenizer
                inputfile = os.path.join("tmp",file)
                process(inputfile,args.tokenizer)
        print("combining ...")
        file_list = [os.path.join("tmp",file) for file in file_list]
        combine(file_list,outputfile)
    shutil.rmtree("tmp", True)
    print("Done!")