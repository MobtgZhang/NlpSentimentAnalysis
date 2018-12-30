from config import config
from tqdm import tqdm
import uuid
import pandas as pd
# random creates a file for every data file
def getRadomNum():
    res = str(uuid.uuid4())
    res = res.replace('-', '')
    return res[:16]
# Open a file for thread
def GetData(filename,seplength):
    print("Seperate the dataset...")
    out = pd.read_csv(filename)
    Length = len(out)
    All_Sep = Length//seplength
    DataList = []
    for k in tqdm(range(All_Sep)):
        DataList.append(out.loc[k*seplength:(k+1)*seplength])
    return DataList
# LoadVocabs
def LoadVocabs(save_filename):
    Vocabs = set()
    with open(save_filename,mode = "r",encoding = "utf-8") as fpLoad:
        while True:
            line = fpLoad.readline()
            if not line:
                break
            line = line.strip()
            if line =="":
                continue
            else:
                Vocabs.add(line)
    return Vocabs
