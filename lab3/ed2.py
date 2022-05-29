direction=[(0,1),(1,0),(0,-1),(-1,0)] #当前位置四个方向的偏移量
path=[]              #存找到的路径
matrix = []
start = (0,0)
end=(0,0)

def read(filePath):
    with open(filePath, encoding='utf-8') as file_obj:
        for line in open(filePath,encoding='utf-8'):
            line = line.strip()
            line = list(line)
            matrix.append(line)

def mark(matrix,pos):  #给迷宫matrix的位置pos标"2"表示“遍历过”
    matrix[pos[0]][pos[1]]=2
 
def passable(matrix,pos): #检查迷宫matrix的位置pos是否可通行
    return matrix[pos[0]][pos[1]]==0
 
def find_path(matrix,pos,end):
    mark(matrix,pos)
    if pos==end:
        print(pos,end=" ")  #已到达出口，输出这个位置。成功结束
        path.append(pos)
        return True
    for i in range(4):      #否则按四个方向顺序检查
        nextp=pos[0]+direction[i][0],pos[1]+direction[i][1]
        #考虑下一个可能方向
        if passable(matrix,nextp):        #不可行的相邻位置不管
            if find_path(matrix,nextp,end):#如果从nextp可达出口，输出这个位置，成功结束
                print(pos,end=" ")
                path.append(pos)
                return True
    return False
 
def see_path(matrix,path):     #使寻找到的路径可视化
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
            if c==3:
                print('\033[0;31m'+"*"+" "+'\033[0m',end="")
            elif c=="S" or c=="E":
                print('\033[0;34m'+c+" " + '\033[0m', end="")
            elif c==2:
                print('\033[0;32m'+"#"+" "+'\033[0m',end="")
            elif c==1:
                print('\033[0;;40m'+" "*2+'\033[0m',end="")
            else:
                print(" "*2,end="")
        print()
 
if __name__ == '__main__':
    filePath = r'MazeData.txt'
    read(filePath)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 'S':
                matrix[i][j] = 0
                start = (i,j)
            elif matrix[i][j] == 'E':
                matrix[i][j] = 0
                end = (i,j)
            else:
                matrix[i][j] = int(matrix[i][j])
    #print(matrix)
    #start=(1,34)
    #end=(16,1)
    find_path(matrix,start,end)
    see_path(matrix,path)