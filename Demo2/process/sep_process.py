import process.config as config
import os
import pandas as pd
import shutil
def process_csv():
    if not (os.path.exists(config.train_csv) and
            os.path.exists(config.test_csv) and
            os.path.exists(config.validation_csv)):
        if os.path.exists(config.processed_csv_root):
            shutil.rmtree(config.processed_csv_root)
            os.mkdir(config.processed_csv_root)
        else:
            os.mkdir(config.processed_csv_root)
    else:
        print("Files processed ")
        print(config.train_csv)
        print(config.validation_csv)
        print(config.test_csv)
        print("Now exists!")
        return
    print("Processing raw data ......")
    outData = pd.read_csv(config.raw_train)
    outData[:15000].to_csv(config.test_csv, index = None) # test_data
    outData[15000:].to_csv(config.train_csv, index = None) # train_data
    pd.read_csv(config.raw_validation).to_csv(config.validation_csv, index = None) # validation
    print("Raw data processed!")