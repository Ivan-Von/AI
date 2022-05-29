from typing import Tuple
import pandas as pd
import numpy as np
import csv
emotion = {0:"anger",1:"disgust",2:"fear",3:"joy",4:"sad",5:"surprise"}
# a full set inluding 'test_set.csv' and 'validation_set.csv'
data_train = pd.read_csv('AI\\lab6\\final_set.csv')
# test_set.csv
data_test = pd.read_csv('AI\\lab6\\test_set.csv')

N_word = 0

col_word = data_train['Words']
words = np.array(col_word)
words_train = []
for i in range(len(words)):
    index = words[i].split()
    # 2 demisions
    words_train.append(index)
    # length for Polynomial
    N_word += len(index)
# the number of sentences
N = len(words_train)

def get_emotion(data_train,N):
    # get emotions
    anger_train = np.array(data_train['anger'])
    disgust_train = np.array(data_train['disgust'])
    fear_train = np.array(data_train['fear'])
    joy_train = np.array(data_train['joy'])
    sad_train = np.array(data_train['sad'])
    surprise_train = np.array(data_train['surprise'])
    # to find emotion
    emotion_train = []
    for i in range(N):
        emotion_train.append([anger_train[i],disgust_train[i],fear_train[i],joy_train[i],sad_train[i],surprise_train[i]])
    # to find emotion type [0,5]
    emo_array = []
    for i in range(N):
        index = emotion_train[i].index(max(emotion_train[i]))
        emo_array.append(index)
    return emo_array

def Bernoulli(N,str):
    P = []
    P_ei = []
    Nei = [0,0,0,0,0,0]
    Nei_xk = []
    emo_array = get_emotion(data_train,N)
    for i in range(N):
        Nei[emo_array[i]]  += 1
    for j in range(6):
        P_ei.append(Nei[j]*1.0/N)
    for m in range(len(str)):
        index = [0,0,0,0,0,0]
        for n in range(N):
            if str[m] in words_train[n]:
                index[emo_array[n]] += 1
        Nei_xk.append(index)
    for i in range(6):
        index_P = P_ei[i]
        for k in range(len(str)):
            # Laplace smoothing
            P_Xk_Ei = (Nei_xk[k][i]*1.0 + 1) / (Nei[i]*1.0+2)
            index_P = index_P * P_Xk_Ei
        P.append(index_P)
    return P

#N_word 单词总长度
def Polynomial(N,str):
    P = []
    P_ei = []
    NWei = [0,0,0,0,0,0]
    Nei = [0,0,0,0,0,0]
    NWei_xk = []
    V = 0
    emo_array = get_emotion(data_train,N)
    for i in range(N):
        NWei[emo_array[i]] += len(words_train[i])
        V += len(words_train[i])
        Nei[emo_array[i]]  += 1
    for j in range(6):
        P_ei.append(Nei[j]*1.0/N)
    for m in range(len(str)):
        index = [0,0,0,0,0,0]
        for n in range(N):
            if str[m] in words_train[n]:
                index[emo_array[n]] += 1
        NWei_xk.append(index)
    for i in range(6):
        index_P = P_ei[i]
        for k in range(len(str)):
            # Laplace smoothing
            P_Xk_Ei = (NWei_xk[k][i]*1.0 + 1) / (Nei[i]*1.0 + V)
            index_P = index_P*P_Xk_Ei
        P.append(index_P)
    return P

def compare_(emo_array1,emo_array2):
    index = 0
    for i in range(len(emo_array1)):
        if emo_array1[i] == emo_array2[i]:
            index += 1
    print("The accuracy is :",round(index*1.0/len(emo_array1),10)) 

if __name__ == '__main__':
    col_word = data_test['Words']
    words = np.array(col_word)
    words_test = []
    for i in range(len(words)):
        index = words[i].split()
        words_test.append(index)
    N1 = len(words_test)
    emotion_new = []
    emo_array_new = []
    for i in range(N1):#N1
        emotion_new.append(Polynomial(N,words_test[i]))
        index = emotion_new[i].index(max(emotion_new[i]))
        emo_array_new.append(index)
    header = ['Words','emotion']
    with open('20337268_zhangwenqin_NB_classification.csv', 'w',newline='') as f:
       # 创建writer
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(words)):
            writer.writerow([words[i],emotion[emo_array_new[i]]])