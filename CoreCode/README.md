1.本模型用于AI-Challenger2018 细粒度用户评论情感分类,数据集基于AI-Challenger2018数据集。
2.训练模型之前下载哈工大pyltp分词模型，并解压到文件夹下，下载地址为：http://pan.baidu.com/share/link?shareid=1988562907&uk=2738088569 。
3.训练之前下载中文词向量模型，本实验基于中文词向量如下地址：https://github.com/Embedding/Chinese-Word-Vectors 。
4.训练模型运行过程：
[1] python3 prepare.py --split training
[2] python3 prepare.py --split validation
[3] python3 train.py 
可以设置参数训练模型
