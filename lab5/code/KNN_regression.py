import pandas as pd
import numpy as np
import math
import csv
emotion = {0:"anger",1:"disgust",2:"fear",3:"joy",4:"sad",5:"surprise"}
# to readd files 
data_train = pd.read_csv('AI\\lab6\\train_set.csv')
data_test = pd.read_csv('AI\\lab6\\test_set.csv')
data_validation = pd.read_csv('AI\\lab6\\validation_set.csv')

# to get words and turn it into list
# words_train [] is the words list for train_set.csv, and according to the demand, it should be train_set && validation_set, anyway, just change the fle source
col_word = data_train['Words']
words = np.array(col_word)
words_train = []
for i in range(len(words)):
    index = words[i].split()
    words_train.append(index)

Len_words = len(words)

# the baseline
K = int(math.sqrt(Len_words))
# to create corpus
corpus = []
for i in range(Len_words):
    for j in range(len(words_train[i])):
        if words_train[i][j] not in corpus:
            corpus.append(words_train[i][j])
len_corpus = len(corpus)

# One-hot
# Noromally, it should include test_set's corpus, but i didn't use the COS to cauclate the distance so i simplified it.
One_hot = []
for i in range(len(words)):
    index = [0 for n in range(len_corpus)]
    for j in range(len(words_train[i])):
        index_position = corpus.index(words_train[i][j])
        index[index_position] += 1
    One_hot.append(index)

# to get emotion
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
    return emotion_train,emo_array
    # the 1st return is 6 emotions, 2nd return is sentences' emotions

# The regressive KNN with Eulic distance
def Eucli_distance_KNN_regression(str):
    lable = [0,0,0,0,0,0]
    distance = []
    not_in_corpus = 0
    index = [0 for n in range(len_corpus)]
    for j in range(len(str)):
        if str[j] in corpus:
            index_position = corpus.index(str[j])
            index[index_position] += 1
        else:
            not_in_corpus += 1
    # in regressive KNN, we just need all 6 emotions
    all_emotion,_ = get_emotion(data_train,Len_words)
    for i in range(Len_words):
        Eulic = not_in_corpus
        for j in range(len_corpus):
            Eulic += math.pow(One_hot[i][j] - index[j],2)
        index_distance = math.sqrt(Eulic)
        distance.append(index_distance)
    distance.sort()
    # to get final P
    for i in range(6):
        index = 0
        for j in range(K):
            index = index + (all_emotion[j][i]/distance[j])
        lable[i] = index
    # the possibilities for 6 emotions
    return lable

if __name__=='__main__':
    col_word = data_test['Words']
    words = np.array(col_word)
    words_test = []
    for i in range(len(words)):
        index = words[i].split()
        words_test.append(index)
    N1 = len(words_test)

    header = ['textid','Words','anger','disgust','fear','joy','sad','surprise']
    with open('20337268_zhangwenqin_KNN_regression.csv', 'w',newline='') as f:
       # 创建writer
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(words)):
            index = Eucli_distance_KNN_regression(words_test[i])
            sum = index[0] + index[1] + index[2] + index[3] + index[4] + index[5]
            writer.writerow([i+1,words[i],round(index[0]/sum,6),round(index[1]/sum,6),round(index[2]/sum,6),round(index[3]/sum,6),round(index[4]/sum,6),round(index[5]/sum,6)])