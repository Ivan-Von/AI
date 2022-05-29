# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import operator
import pandas as pd
import csv
import numpy as np
 
"""
函数说明:创建数据样本
Returns:
    dataset - 实验样本切分的词条
    classVec - 类别标签向量
"""

data_train = pd.read_csv('AI\\lab6\\train_set.csv')
data_test = pd.read_csv('AI\\lab6\\validation_set.csv')

col_word = data_train['Words']
words = np.array(col_word)
words_train = []
for i in range(len(words)):
    index = words[i].split()
    words_train.append(index)

anger_train = np.array(data_train['anger'])
disgust_train = np.array(data_train['disgust'])
fear_train = np.array(data_train['fear'])
joy_train = np.array(data_train['joy'])
sad_train = np.array(data_train['sad'])
surprise_train = np.array(data_train['surprise'])

emotion_train = []
N = len(words_train)
for i in range(N):
    emotion_train.append([anger_train[i],disgust_train[i],fear_train[i],joy_train[i],sad_train[i],surprise_train[i]])

emo_array = []
for i in range(N):
    index = emotion_train[i].index(max(emotion_train[i]))
    emo_array.append(index)

 
"""
函数说明：特征选择TF-IDF算法
Parameters:
     list_words:词列表
Returns:
     dict_feature_select:特征选择词字典
"""
def feature_select(list_words):
    #总词频统计
    doc_frequency=defaultdict(int)
    #print(doc_frequency)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i]+=1

    #计算每个词的TF值
    word_tf={}  #存储没个词的tf值
    for i in doc_frequency:
        word_tf[i]=doc_frequency[i]/sum(doc_frequency.values())
 
    #计算每个词的IDF值
    doc_num=len(list_words)
    word_idf={} #存储每个词的idf值
    word_doc=defaultdict(int) #存储包含该词的文档数
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i]+=1
    for i in doc_frequency:
        word_idf[i]=math.log(doc_num/(word_doc[i]+1))
 
    #计算每个词的TF*IDF的值
    word_tf_idf={}
    for i in doc_frequency:
        word_tf_idf[i]=word_tf[i]*word_idf[i]
 
    # 对字典按值由大到小排序
    dict_feature_select=sorted(word_tf_idf.items(),key=operator.itemgetter(1),reverse=True)
    return dict_feature_select
 






if __name__=='__main__':
    features=feature_select(words_train) #所有词的TF-IDF值
    # print(features)
    # print(len(features))