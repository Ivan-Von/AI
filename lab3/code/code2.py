import os
import sys
start = []# 记录起点
end = []# 记录终点
matrix = [] # 储存迷宫数据
state = []# 访问状态标记 0为未访问，1为正向队列访问，2为逆向队列访问
action = []# 父节点生成此节点时所采取的动作

def read(file_path):
    global matrix
    global state
    global action
    with open(file_path,encoding = 'utf-8') as file_obj:
        for line in file_obj:
            matrix.append(list(line.strip()))
    state = [[0 for i in range(len(matrix[0]))] for j in range(len(matrix))]            
    action = [[[0,0] for i in range(len(matrix[0]))] for j in range(len(matrix))]

def __dbfs__(start,end):
    global matrix # 储存迷宫数据
    global action #父节点生成此节点时所采取的动作
    dirx = [0,0,1,-1]#参考代码，同第一个拆开
    diry = [1,-1,0,0]
    path1 = [start] # BFS中以start为起点的正向队列
    path2 = [end] # BFS中以end为起点的逆向队列
    flag = 0 # flag == 1, 拓展正向队列，flag == 0，拓展逆向队列

    while(len(path1) and len(path2)):
        state[start[0]][start[1]] = 1
        state[end[0]][end[1]] = 2

        if(len(path1) <= len(path2)):
            flag = 1
            temp = path1[0]
            del path1[0]
        else:
            flag = 0
            temp = path2[0]
            del path2[0]

        for i in range(4):
                dx = temp[0] + dirx[i]
                dy = temp[1] + diry[i]
                if(dx >= 0 and dx < len(matrix) and dy >= 0 and dy < len(matrix[0]) and matrix[dx][dy]!='1'):
                    if(state[dx][dy] == 0):
                        state[dx][dy] = state[temp[0]][temp[1]]
                        if(flag):
                            action[dx][dy] = [dirx[i],diry[i]] # 正向序列，action记为正向
                            path1.append([dx,dy])
                        else:
                            action[dx][dy] = [-dirx[i],-diry[i]] # 逆向序列，action记为逆向
                            path2.append([dx,dy])
                    else:
                        if state[temp[0]][temp[1]] + state[dx][dy] == 3: # 相遇
                            if(flag):
                                path = []
                                dx = temp[0]
                                dy = temp[1]
                                while(action[dx][dy] != [0,0]): # 如果没有遍历到起点（在path路径上，只有起点的action是[0,0]）
                                    path.append(action[dx][dy]) # 将父节点产生此节点的动作记录到path中
                                    dx -= action[dx][dy][0]
                                    dy -= action[dx][dy][1]
                                path.reverse()
                                
                                path2 = []
                                dx = temp[0]+dirx[i]
                                dy = temp[1]+diry[i]
                                while(action[dx][dy] != [0,0]): # 如果没有遍历到终点（在path路径上，只有终点的action是[0,0]）
                                    path2.append(action[dx][dy]) # 将父节点产生此节点的动作记录到path中
                                    dx += action[dx][dy][0]
                                    dy += action[dx][dy][1]

                                path.append([dirx[i],diry[i]])
                                path += path2
                                return path
                            
                            else:
                                path = []
                                dx = temp[0]
                                dy = temp[1]
                                while(action[dx][dy] != [0,0]): # 如果没有遍历到起点（在path路径上，只有起点的action是[0,0]）
                                    path.append(action[dx][dy]) # 将父节点产生此节点的动作记录到path中
                                    dx += action[dx][dy][0]
                                    dy += action[dx][dy][1]

                                path2 = []
                                dx = temp[0]+dirx[i]
                                dy = temp[1]+diry[i]
                                while(action[dx][dy] != [0,0]): # 如果没有遍历到终点（在path路径上，只有终点的action是[0,0]）
                                    path2.append(action[dx][dy]) # 将父节点产生此节点的动作记录到path中
                                    dx -= action[dx][dy][0]
                                    dy -= action[dx][dy][1]
                                path2.reverse()

                                path2.append([dirx[i],diry[i]])
                                path2 += path
                                return path2

def search_path():
    global start
    global end
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 'S':
                start = [i,j]
                state[i][j] = 1
            if matrix[i][j] == 'E':
                end = [i,j]
    path = __dbfs__(start,end)
    return path
#打印出路径,可视化，同第一个
def see_path(matrix,path):#可视化
    global start
    dx = start[0]
    dy = start[1]

    while(len(path)):
        dx += path[0][0]
        dy += path[0][1]
        matrix[dx][dy] = '2'
        del path[0]
    
    for i,p in enumerate(path):
        if i==0:
            matrix[p[0]][p[1]] ="E"
        elif i==len(path)-1:
            matrix[p[0]][p[1]]="S"
        else:
            matrix[p[0]][p[1]] =3
    print("\n")
    for r in matrix:
        for c in r:
            if c=='2':#路径，绿色
                print('\033[0;32m'+"*"+" "+'\033[0m',end="")
            elif c=="S" or c=="E":#开始和结尾，蓝色
                print('\033[0;34m'+c+" " + '\033[0m', end="")
            elif c==3:#探索过的路径，红色
                print('\033[0;35m'+"*"+" "+'\033[0m',end="")
            elif c=='1':#墙
                print('\033[0;40m'+" "*2+'\033[0m',end="")
            else:
                print(" "*2,end="")
        print()

if __name__ == '__main__':
    filePath =r'MazeData.txt'
    read(filePath)
    path = search_path()
    see_path(matrix,path)