import numpy as np
import pandas as pd
import math
data_train = pd.read_csv('AI\\lab7\\train.csv')
test_train = pd.read_csv('AI\\lab7\\test.csv')
x = np.array(data_train)
for i in range(6000):
    x[i][40] = 1
answer = data_train['answer']   
need = [1 for i in range(7000)]
w = [0 for i in range(41)]
n = 1

def pi(j):
    sum = 0
    for i in range(41):
        sum += w[i]*x[j][i]
    
    P = 1.0/(1+np.exp(-sum))
    return P

def update_w():
    update = [0 for i in range(41)]
    rate = 0
    for i in range(7000):
        rate = answer[i] - pi(i)
        for j in range(41):
            update[j] += rate * x[i][j]
    for i in range(41):
        w[i] += n * update[i]

def update_need():
    for i in range(7000):
        if (pi(i) > 1/2 and answer[i] == 1) or (pi(i) < 1/2 and answer[i] == 0):
            need[i] = 0
        else:
            need[i] = 1

def L():
    L = 0
    for i in range(7000):
        sum = 0
        for j in range(41):
            sum += x[i][j]*w[j]
        L += (answer[i]*sum - math.log(1 + np.exp(sum)))
    return L

if __name__ ==  '__main__':
    update_need()
    i = 0
    while(1 in need):
        i += 1
        if i > 8:#(正确率最高)
            break
        update_w()
        update_need()
    
    right = 0
    for i in range(1000):
        if (pi(i+7000) > 1/2 and answer[i+7000] == 1) or (pi(i+7000) < 1/2 and answer[i+7000] == 0):
            right += 1
    print(format(right*1.0/1000,'.5f'))