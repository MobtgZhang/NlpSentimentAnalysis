# Multi-way matching based fine-grained sentiment analysis for user reviews 
This project is used for the paper:[Guo X , Zhang G , Wang S , et al. Multi-waymatchingbased fine-grained sentiment analysis for user reviews[J]. Neural Computing and Applications, 2020, 32(10):5409-5423.](http://link.springer.com/article/10.1007/s00521-019-04686-9)
## 1. DataSet instructions
This model is used for AI-Challenger2018 fine-grained user comment sentiment classification, and the dataset is based on the AI-Challenger2018 dataset. 
## 2. Chinese language segment instructions 
Before training the model, download the Pyltp word segmentation model of Harbin Institute of Technology and unzip it to the folder. The download address is [here](http://pan.baidu.com/share/link?shareid=1988562907&uk=2738088569.)
## 3. Chinese language word vector instructions
Download the Chinese word vector model before training. This experiment is based on the following address of the [Chinese word vector](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz)
## 4.model training process
[1] split training data:
```bash
python3 prepare.py --split training
```
[2] split validation data: 
```bash
python3 prepare.py --split validation
```
[3] training data
```bash
python3 train.py
```
Addtionally, we can change the parameters in file config.py.
