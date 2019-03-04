import os
import shutil
import time
import pandas as pd
import json
import codecs
import argparse

import config
from script.preprocess import load_dataset,process_dataset
def sepData(filename):
    outData = pd.read_csv(filename)
    Length1 = len(outData)//3
    Length2 = Length1*2
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    outData[:Length1].to_csv(os.path.join("tmp","data1.csv"),index=None)
    outData[Length1:Length2].to_csv(os.path.join("tmp","data2.csv"),index=None)
    outData[Length2:].to_csv(os.path.join("tmp","data3.csv"),index=None)
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
    with codecs.open(save_path,mode="w",encoding="utf-8") as fileW:
        for file in file_list:
            with codecs.open(file,mode="r",encoding="utf-8") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    fileW.write(line + "\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', default="pyltp", type=str, help='Path to AI data directory')
    args = parser.parse_args()
    data_list = ["data%d.csv" % i for i in range(1, 4)]
    flag = True
    for file in data_list:
        outputfile = os.path.join("tmp",file)
        if not os.path.exists(outputfile):
            flag = False
            break
    if flag:
        str_line = " ".join(data_list)
        print("file: "+str_line + " exists!")
    else:
        print("seperating ...")
        sepData(os.path.join(config.DATA_DIR, "trainset.csv"))
    outputfile = os.path.join(config.OUT_DIR,"train-processed-%s.json"%args.tokenizer)
    if os.path.exists(outputfile):
        print("file: " + outputfile + " exists!")
    else:
        print("segmenting ...")
        for file in data_list:
            filename = file.split(".")[0] + "-processed-%s.json"%args.tokenizer
            inputfile = os.path.join("tmp",file)
            process(inputfile,args.tokenizer)
        print("combining ...")
        data_list = [os.path.basename(f).split(".")[0]+"-processed-%s.json"%args.tokenizer for f in data_list]
        outputfile = os.path.join(config.OUT_DIR,"train-processed-%s.json"%args.tokenizer)
        combine(data_list,outputfile)
    shutil.rmtree("tmp", True)
    print("Done!")