import pandas as pd
import numpy as np
import csv
data_train = pd.read_csv('AI\\lab7\\train.csv')
test_train = pd.read_csv('AI\\lab7\\test.csv')
x = np.array(data_train)
answer = data_train['answer']
LEN = 7000
# n = 1
# w = [0 for i in range(40)] # 初始化
# b = 0
# need = [0 for i in range(LEN)]

# 如果是真，则需要更新
def identify(w,i):
    index = [0 for i in range(40)]
    for j in range(40):
        index[j] += w[j]*x[i][j]
    sum_index = sum(index) + b
    sum_index = sum_index * answer[i]

    if (sum_index != 0 and answer[i] == 0) or (sum_index <= 0 and answer[i] == 1):
        return 1
    elif (sum_index == 0 and answer[i] == 0) or (sum_index > 0 and answer[i] == 1):
        return 0

def update():
    for i in range(LEN):
        if identify(w,i):
            need[i] = 1
        else:
            need[i] = 0

def __PLA__():
    update()
    sum = 0
    while 1 in need:
        sum == 1
        if sum > 100:
            break
        i = need.index(1)
        for j in range(40):
            w[j] += n*x[i][j]*answer[i]
        global b
        b += n*answer[i]
        update()
        print(w,b)

def result():
    y = np.array(test_train)
    for i in range(len(test_train)):
        index = 0
        for j in range(40):
            index += x[i][j]*w[j]
            
n = 1
w = [0 for i in range(40)] # 初始化
b = 0
need = [0 for i in range(LEN)]

if __name__ ==  '__main__':
    __PLA__()
    right = 0
    for i in range(1000):
        if not identify(w,i + LEN) :
            right += 1
    print(right*1.0/1000)