import threading
import time
import pandas as pd
import pyltp as ltp
from absl import app
from config import config
from tqdm import tqdm
import os
from utils import getRadomNum,GetData,LoadVocabs
class Dictionary:
    def __init__(self,task_file,save_file,mode,seplength,delay):
        self.task_file = task_file
        self.save_file = save_file
        if not os.path.exists(save_file):
            threadID = getRadomNum()
            (filepath,threadName) = os.path.split(self.task_file)
            # Deination in ModeThread
            print("start thread")
            if mode == "block":
                threadLockBlock = threading.Lock()
                thread = BlockThread(theardID,threadName,self.save_file,datalist,threadLockBlock)
            elif mode == "file":
                threadLockFile = threading.Lock()
                thread = FileThread(threadID,threadName,self.task_file,self.save_file,seplength,delay,threadLockFile)
            else:
                pass
            thread.start()
            # Util finish the thread
            thread.join()
            print("exit thread")
        self.vob_length = 0
        self.Vocabs = None
    def getVocabs(self):
        if self.Vocabs is None:
            Vocabs = LoadVocabs(self.save_file)
            self.vob_length = len(Vocabs)
            self.Vocabs = Vocabs
        return self.Vocabs
    def __len__(self):
        return self.vob_length

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
        print("Starting " + self.threadID + " Name: "+self.threadName+" Time:[%s]"%time.ctime(time.time()))
        # 获得锁，成功获得锁定后返回True
        # 可选的timeout参数不填时将一直阻塞直到获得锁定
        # 否则超时后将返回False
        self.threadLockFile.acquire()
        thread_mode_list = self.create_thread(self.task_file,self.seplength,self.delay)
        # 等待所有线程完成
        for k in range(len(thread_mode_list)):
            thread_mode_list[k].join()
            print("End " + thread_mode_list[k].threadID + " Name: "+thread_mode_list[k].threadName+" Time:[%s]"%time.ctime(time.time()))
        print("End " + self.threadID + " Name: "+self.threadName+" Time:[%s]"%time.ctime(time.time()))
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
            # Deination in BlockThread
            threadLockBlock = threading.Lock()
            thread_mode = BlockThread(threadID,threadName,save_filename,datalist,threadLockBlock)
            # 开启新线程
            time.sleep(delay)
            thread_mode.start()
            # 添加线程到线程列表
            thread_mode_list.append(thread_mode)
            thread_count += 1
        return thread_mode_list
class BlockThread(threading.Thread):
    def __init__(self,threadID,threadName,save_filename,datalist,blockthreadlock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.threadName = threadName
        self.save_file = save_filename
        self.datalist = datalist
        self.threadLockMode = blockthreadlock
    def run(self):
        # 获得锁，成功获得锁定后返回True
        # 可选的timeout参数不填时将一直阻塞直到获得锁定
        # 否则超时后将返回False
        print("Starting " + self.threadID + " Name: "+self.threadName+" Time:[%s]"%time.ctime(time.time()))
        self.threadLockMode.acquire()
        # MakeVocabs
        Vocabs = set()
        for k in tqdm(self.datalist.index,self.threadName + " block"):
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
