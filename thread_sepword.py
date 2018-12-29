import threading
import _thread
import time
import pandas as pd
import pyltp as ltp
from absl import app
from config import config
import uuid
import os
# Define a function for a thread
def print_time( threadName, delay):
    time.sleep(delay)
    print ("%s: %s" % (threadName, time.ctime(time.time())))
# random creates a file for every data file
def getRadomNum():
    res = str(uuid.uuid4())
    res = res.replace('-', '')
    return res[:16]
# Open a file for thread
def GetData(filename,seplength):
    out = pd.read_csv(filename)
    Length = len(out)
    All_Sep = Length//seplength
    DataList = []
    for k in range(All_Sep):
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
# Inherit father theard creates a thread that manage the file data
class FolderThread(threading.Thread):
    def __init__(self,theardID,theardName,task_file,save_file):
        threading.Thread.__init__(self)
        self.theardID = theardID
        self.theardName = theardName
        self.task_file = task_file
        self.save_file = save_file
    def run(self):
        pass
    def create_thread(self,delay):
        pass
class FileThread(threading.Thread):
    def __init__(self,threadID,threadName,task_filename,save_filename,seplength,delay,threadlockfile):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.threadName = threadName
        self.save_file = save_filename
        self.task_file = task_filename
        self.datalist = seplength
        self.delay = delay
        self.threadLockFile = threadlockfile
        self.seplength = seplength
    def run(self):
        if not os.path.exists(config.cache_file):
            os.mkdir(config.cache_file)
        print("Starting " + self.threadID + " Name: "+self.threadName)
        # 获得锁，成功获得锁定后返回True
        # 可选的timeout参数不填时将一直阻塞直到获得锁定
        # 否则超时后将返回False
        self.threadLockFile.acquire()
        thread_mode_list = self.create_thread(self.task_file,self.seplength,self.delay)
        # 等待所有线程完成
        for t in thread_mode_list:
            t.join()
        # Here Insert a thread,that thread synchronization
        Vocabs = set()
        (filepath,tmp_file) = os.path.split(self.task_file)
        rootfile = os.path.join(config.cache_file,tmp_file.split(".")[0])
        for files in os.listdir(rootfile):
            filename = os.path.join(rootfile,files)
            vob = LoadVocabs(filename)
            Vocabs.update(vob)
        # Save cache file
        with open(self.save_file,mode = "w",encoding = "utf-8") as fp:
            for word in Vocabs:
                fp.write(word + "\n")
        # remove files
        for files in os.listdir(rootfile):
            filename = os.path.join(rootfile,files)
            os.remove(filename)
        os.rmdir(rootfile)
        # 释放锁
        self.threadLockFile.release()
    # Create a data thread for a file 
    def create_thread(self,task_filename,seplength,delay):
        if not os.path.exists(config.cache_file):
            os.mkdir(config.cache_file)
        # creates a file for data work
        (filepath,filename) = os.path.split(task_filename)
        filename = os.path.join(config.cache_file,filename.split(".")[0])
        if not os.path.exists(filename):
            os.mkdir(filename)
        outData = GetData(task_filename,seplength)
        fp_namelist = []
        ListLength = len(outData)
        fp_count = 0
        # random creates a file for every data file
        while fp_count <= ListLength:
            fname = os.path.join(filename,getRadomNum()+".csv")
            if fname not in fp_namelist:
                fp_namelist.append(fname)
                fp_count += 1
        thread_count = 0
        thread_mode_list = []
        # create threads
        while thread_count < ListLength:
            threadID = getRadomNum()
            threadName = "file:" + str(thread_count)
            save_filename = fp_namelist[thread_count]
            datalist = outData[thread_count]
            # Deination in ModeThread
            threadLockMode = threading.Lock()
            thread_mode = ModeThread(threadID,threadName,save_filename,datalist,threadLockMode)
            # 开启新线程
            time.sleep(delay)
            thread_mode.start()
            # 添加线程到线程列表
            thread_mode_list.append(thread_mode)
            thread_count += 1
        return thread_mode_list
class ModeThread(threading.Thread):
    def __init__(self,threadID,threadName,save_filename,datalist,modethreadlock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.threadName = threadName
        self.save_file = save_filename
        self.datalist = datalist
        self.threadLockMode = modethreadlock
    def run(self):
        # 获得锁，成功获得锁定后返回True
        # 可选的timeout参数不填时将一直阻塞直到获得锁定
        # 否则超时后将返回False
        print("Starting " + self.threadID + " Name: "+self.threadName)
        self.threadLockMode.acquire()
        # MakeVocabs
        Vocabs = set()
        for k in self.datalist.index:
            sentence = self.datalist.loc[k]['content']
            segment = ltp.Segmentor()
            segment.load(os.path.join(config.segment_model_file,"cws.model"))
            sent = list(segment.segment(sentence))
            segment.release()
            for word in sent:
                Vocabs.add(word.strip())
        # Save cache file
        with open(self.save_file,mode = "w",encoding = "utf-8") as fp:
            for word in Vocabs:
                fp.write(word + "\n")
        # 释放锁
        self.threadLockMode.release()
def main(_):
    task_filename = "/home/asus/AI_Challenger2018/TestData/testfile.csv"
    threadID = getRadomNum()
    threadName = "ModeName"
    save_filename = "/home/asus/AI_Challenger2018/NewCode/vocabfile.csv"
    seplength = 200
    delay = 1
    # Deination in ModeThread
    threadLockFile = threading.Lock()
    thread = FileThread(threadID,threadName,task_filename,save_filename,seplength,delay,threadLockFile)
    thread.start()
    print("exit thread")
if __name__ == "__main__":
    app.run(main)
