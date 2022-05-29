matrix = []
path1 = []
path2 = []
direction = [(0,1),(1,0),(0,-1),(-1,0)]#表示上下左右四个方向
start = (0,0)
end = (0,0)
def read(filePath):
    with open(filePath, encoding='utf-8') as file_obj:
        for line in open(filePath,encoding='utf-8'):
            line = line.strip()
            line = list(line)
            matrix.append(line)

def find_path(matrix,start,end):
    path1.append(start)
    path2.append(end)
    while(path1 & path2):
        now_start = path1[-1]
        now_end = path2[-1]
        if now_start == now_end:
            print(path1)
            print(path2)
            break
        row1,clo1 = now_start
        row2,clo2 = now_end
        matrix[row1][clo1] = 2
        matrix[row2][clo2] = 2
        func(path1,row1,clo1)
        func(path2,row2,clo2)
        continue
    
def func(path,row,clo):
    if matrix[row-1][clo] == 0 : #上面
        path.append((row-1,clo))
    elif matrix[row + 1][clo] == 0: #下面
        path.append((row + 1,clo))
    elif matrix[row][clo - 1] == 0: #左边
        path.append((row,clo - 1))
    elif matrix[row][clo + 1] == 0: #右边
        path.append((row,clo + 1))
    else:
        path.pop()

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
    find_path(matrix,start,end)