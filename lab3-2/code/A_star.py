import heapq
import numpy as np
import time

# 方向数组
dx = [0, -1, 0, 1]
dy = [1, 0, -1, 0]

OPEN = []

CLOSE = set() # close表，用于判重

path = []

# 最终状态
node_num  = 0
end = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)

# 初始状态测例集
init = [
     (1, 2, 4, 8, 5, 7, 11, 10, 13, 15, 0, 3, 14, 6, 9, 12),
     (5,1,3,4,2,7,8,12,9,6,11,15,0,13,10,14),
     (14, 10, 6, 0, 4, 9, 1, 8, 2, 3, 5, 11, 12, 13, 7, 15),
     (6, 10, 3, 15, 14, 8, 7, 11, 5, 1, 0, 2, 13, 12, 9, 4)
    ]

# 状态结点
class Node(object):
    def __init__(self, gn=0, hn=0, state=None, parent=None):
       self.gn = gn
       self.hn = hn
       self.fn = self.gn + self.hn
       self.state = state
       self.parent = parent

    def __lt__(self, node): # heapq的比较函数
        if self.fn == node.fn:
            return self.gn > node.gn
        return self.fn < node.fn

# 曼哈顿函数
def manhattan(state):
    M = 0 
    for t in range(16):
        if state[t] == end[t] or state[t]== 0:
            continue
        else:
            x =  (state[t] - 1) // 4   # 最终坐标
            y =  state[t] - 4 * x - 1
            dx = t // 4 # 实际坐标
            dy = t % 4
            M += (abs(x - dx) + abs(y - dy))
    return M

def generateChild():  # 生成子结点
   movetable = [] # 针对数码矩阵上每一个可能的位置，生成其能够移动的方向列表
   for i in range(16):
       x, y = i % 4, i // 4
       moves = []
       if x > 0: moves.append(-1)    # 左移
       if x < 3: moves.append(+1)    # 右移
       if y > 0: moves.append(-4)    # 上移
       if y < 3: moves.append(+4)    # 下移
       movetable.append(moves) 
   def children(state):
       idxz = state.index(0) # 寻找数码矩阵上0的坐标
       l = list(state) # 将元组转换成list，方便进行元素修改
       for m in movetable[idxz]:
           l[idxz] = l[idxz + m] # 数码交换位置
           l[idxz + m] = 0
           yield(1, tuple(l)) # 临时返回
           l[idxz + m] = l[idxz]
           l[idxz] = 0
   return children


def A_star(start, Fx): # start 为起始结点，Fx为启发式函数
    root = Node(0, 0, start, None) #gn, hn,state, parent

    OPEN.append(root)
    heapq.heapify(OPEN)

    CLOSE.add(start)
    
    while len(OPEN) != 0: #存在未拓展的节点
        top = heapq.heappop(OPEN)
        global node_num  # 扩展的结点数
        node_num += 1
        if top.state == end: # 目标检测
            return print_path(top)

        generator = generateChild() #生成子结点
        for cost, state in generator(top.state): 
            if state in CLOSE: # CLOSE表为set容器，这里进行环检测
                continue
            CLOSE.add(state)
            child = Node(top.gn+cost, Fx(state), state,top)
            heapq.heappush(OPEN, child) # 将child加入优先队列中

def print_path(node):
    if node.parent != None:
        print_path(node.parent)
    path.append(node.state)
    return path

if __name__ == '__main__':
    for idx, test in enumerate(init):
        with open('result.txt','a')as file:
            time1 = time.time()
            PATH = np.asarray(A_star(test, manhattan))
            time2 = time.time()
            test = np.asarray(test)
            for i, p in enumerate(PATH):
                if i == 0:
                    continue
                else:
                    file.write('Step: %d \n' %(i))
                    file.write("%s \n" %(str(p)))
            print('Test %d, Step %d' %(idx+1, len(path)-1))
            print("Time %f" %(time2-time1))
            file.write('Test %d, Step %d \n' %(idx+1, len(path)-1))
            file.write("Time %f\n" %(time2-time1))
        OPEN.clear()
        CLOSE.clear()
        path.clear()
