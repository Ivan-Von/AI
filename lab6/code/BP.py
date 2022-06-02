import time
from numpy import *
import pandas as pd
import numpy as np
 
######## 数据集 ########

data_train = pd.read_csv('AI\\lab7\\train.csv')
test_train = pd.read_csv('AI\\lab7\\test.csv')

answer = data_train['answer']
train = np.array(data_train)

LEN = 7000
p_s = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for i in range(LEN)]
print(len(p_s[0]))
t_s = [[0] for i in range(LEN)]
p_t = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for i in range(len(answer)-LEN)]
t_t = [[0] for i in range(len(answer) - LEN)]

for i in range(LEN):
    t_s[i][0] = answer[i]
    for j in range(40):
        p_s[i] = train[i][j]

for i in range(len(answer) - LEN):
    t_t[i][0] = answer[i+LEN]
    for j in range(40):
        p_t[i] = train[i+LEN][j]                                                                     
 
 
n_epoch = 2000           # 训练次数
HNum = 2                   # 各层隐藏层节点数
HCNum = 2                  # 隐藏层层数
AFKind = 3                 # 激励函数种类
emax = 0.01                # 最大允许均方差根
LearnRate = 0.1           # 学习率
 
TNum = 7                   # 特征层节点数 (特征数)
 
SNum = len(p_s)            # 样本数
 
# INum = len(p_s[0])         # 输入层节点数（每组数据的维度）
INum = 40
ONum = len(t_s[0])         # 输出层节点数（结果的维度）
StudyTime = 0              # 学习次数
KtoOne = 0.0               # 归一化系数
e = 0.0                    # 均方差跟
 
I = zeros(INum) 
Ti = zeros(TNum)
To = zeros(TNum)
Hi = zeros((HCNum,HNum))
Ho = zeros((HCNum,HNum))
Oi = zeros(ONum)
Oo = zeros(ONum)
Teacher = zeros(ONum)
u = 0.2*ones((TNum,HNum))                  # 初始化 权值矩阵u
w = 0.2*ones(((HCNum-1,HNum,HNum)))        # 初始化 权值矩阵w
v = 0.2*ones((HNum,ONum))                  # 初始化 权值矩阵v
dw = zeros((HCNum-1,HNum,HNum))
Hb = zeros((HCNum,HNum))
Ob = zeros(ONum)
He = zeros((HCNum,HNum))
Oe = zeros(ONum)
p_s = array(p_s)
t_s = array(t_s)
p_t = array(p_t)

def Calcu_KtoOne(p,t):                         # 确定归一化系数
	p_max = p.max()
	t_max = t.max()
	return max(p_max,t_max)
	
def trait(p):                                  # 特征化
	t = zeros((p.shape[0],TNum))
	for i in range(0,p.shape[0],1):
		t[i,0] = p[0]*p[1]*p[2]
		t[i,1] = p[0]*p[1]
		t[i,2] = p[0]*p[2]
		t[i,3] = p[1]*p[2]
		t[i,4] = p[0]
		t[i,5] = p[1]
		t[i,6] = p[2]
	
	return t
	
def AF(p,kind):   # 激励函数
	t = []
	if kind == 1:   # sigmoid
		pass
	elif kind == 2:   # tanh
		pass
	elif kind == 3:    # ReLU
 
		return where(p<0,0,p)
	else:
		pass
		
def dAF(p,kind):   # 激励函数导数
	t = []
	if kind == 1:   # sigmoid
		pass
	elif kind == 2:   # tanh
		pass
	elif kind == 3:    # ReLU
		return where(p<0,0,1) 
	else:
		pass	
		
def nnff(p,t):
	pass
	
def nnbp(p,t):
	pass

def train(p,t):                                # 训练
	global e
	global v
	global w
	global dw
	global u	
	global I 
	global Ti 
	global To 
	global Hi 
	global Ho 
	global Oi 
	global Oo 
	global Teacher 
	global Hb 
	global Ob 
	global He 
	global Oe
	global StudyTime
	global KtoOne	
	
	e = 0.0
	p = trait(p)
		
	KtoOne = Calcu_KtoOne(p,t)
	for isamp in range(0,SNum,1):
		To = p[isamp]/KtoOne
		Teacher = t[isamp]/KtoOne			
		time_start2 = time.perf_counter()
		#计算各层隐藏层输入输出 Hi Ho 
		for k in range(0,HCNum,1):
			if k == 0:
				Hi[k] = dot(To,u)
				Ho[k] = AF(add(Hi[k],Hb[k]),AFKind)
			else:
				Hi[k] = dot(Ho[k-1],w[k-1])
				Ho[k] = AF(add(Hi[k],Hb[k]),AFKind)
		#计算输出层输入输出 Oi Oo 
		Oi = dot(Ho[HCNum-1],v)
		Oo = AF(add(Oi,Ob),AFKind)		
		#反向 nnbp 
		
		#反向更新 v 
		Oe = subtract(Teacher,Oo)
		Oe = multiply(Oe,dAF(add(Oi,Ob),AFKind))
						
		e += sum(multiply(Oe,Oe))		
		#v梯度 		
		dv = dot(array([Oe]),array([Ho[HCNum-1]])).transpose()			  # v 的梯度 
		v = add(v,dv*LearnRate) 
		#反向更新 w 
		He = zeros((HCNum,HNum))
	
		for c in range(HCNum-2,-1,-1):
			if c == HCNum-2:
				He[c+1] = dot(v,Oe)
				He[c+1] = multiply(He[c+1],dAF(add(Hi[c+1],Hb[c+1]),AFKind))
				dw[c] = dot(array([Ho[c]]).transpose(),array([He[c+1]]))				
				w[c] = add(w[c],LearnRate*dw[c])
			else:
				He[c+1] = dot(w[c+1],He[c+2])
				He[c+1] = multiply(He[c+1],dAF(add(Hi[c+1],Hb[c+1]),AFKind))
				dw[c] = dot(array([Ho[c]]).transpose(),array([He[c+1]]))	
				w[c] = add(w[c],LearnRate*dw[c])

		#反向更新u
		He[0] = dot(w[0],He[1])
		He[0] = multiply(He[0],dAF(add(Hi[0],Hb[0]),AFKind))	
		du = dot(array([To]).transpose(),array([He[0]]))			
		u = add(u,du)
		#更新阈值 b
		Ob = Ob + Oe*LearnRate				
		Hb = Hb + He*LearnRate
	e = sqrt(e)
	
def predict(p):
				
	p = trait(p)
	p = p/KtoOne
	p_result = zeros((p.shape[0],1))
 
	for isamp in range(0,p.shape[0],1):
		for k in range(0,HCNum,1):
			if k == 0:
				Hi[k] = dot(p[isamp],u)
				Ho[k] = AF(add(Hi[k],Hb[k]),AFKind)
			else:
				Hi[k] = dot(Ho[k-1],w[k-1])
				Ho[k] = AF(add(Hi[k],Hb[k]),AFKind)
			
			
		#计算输出层输入输出 Oi Oo
		Oi = dot(Ho[HCNum-1],v)
		Oo = AF(add(Oi,Ob),AFKind)
		Oo = Oo*KtoOne
		p_result[isamp] = Oo
	return p_result
 
result = predict(p_t)
right = 0
for i in range(len(answer) - LEN):
	if result[i] == t_t[i]:
		right += 1
print("准确率",right*1.0/1000)