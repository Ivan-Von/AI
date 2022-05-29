'''
评估函数使用
    1. 权值表：边、角、中心的相对重要程度
    2. 行动力：棋手合法的可能棋步数量
    3. 稳定子：绝对不会被翻转的棋子。例如四角
    事实证明稳定子对程序影响不大，下面进行了删减
'''
from new_board import*
import numpy
# eight directions
direction = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

# 1.
# evaluation map, for showing the importance
# I don't know how to get the chart, it comes from Internet. I guess it's a kind of experience
Vmap = numpy.array(
    [[500,-25,10,5,5,10,-25,500],
    [-25,-45,1,1,1,1,-45,-25],
    [10,1,3,2,2,3,1,10],
    [5,1,2,1,1,2,1,5],
    [5,1,2,1,1,2,1,5],
    [10,1,3,2,2,3,1,10],
    [-25,-45,1,1,1,1,-45,-25],
    [500,-25,10,5,5,10,-25,500]]
)

# to cauclate the position's weight 
def position_weight(board,color):
    return sum(sum(board * Vmap))*color

# 2.
# The possible directions
def execution(board,color):
    # accessible choices
    moves = []
    # possible moved boards
    ValidBoardList = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                newboard = board.copy()
                if new_board(newboard,i,j,color):
                    moves.append((i,j))
                    ValidBoardList.append(newboard)
    return moves, ValidBoardList

#3. 
# The pieces that never will be reversed
def getstable(board, color): #稳定子
    stable = [0,0,0]
    # 角, 边, 八个方向都无空格
    cind1 = [0,0,7,7]
    cind2 = [0,7,7,0]
    inc1 = [0,1,0,-1]
    inc2 = [1,0,-1,0]
    stop = [0,0,0,0]
    for i in range(4):
        if board[cind1[i]][cind2[i]] == color:
            stop[i] = 1
            stable[0] += 1
            for j in range(1,7):
                if board[cind1[i]+inc1[i]*j][cind2[i]+inc2[i]*j] != color:
                    break
                else:
                    stop[i] = j + 1
                    stable[1] += 1
    for i in range(4):
        if board[cind1[i]][cind2[i]] == color:
            for j in range(1,7-stop[i-1]):
                if board[cind1[i]-inc1[i-1]*j][cind2[i]-inc2[i-1]*j] != color:
                    break
                else:
                    stable[1] += 1
    colfull = numpy.zeros((8, 8), dtype=numpy.int)
    colfull[:,numpy.sum(abs(board), axis = 0) == 8] = True
    rowfull = numpy.zeros((8, 8), dtype=numpy.int)
    rowfull[numpy.sum(abs(board), axis = 1) == 8,:] = True
    diag1full = numpy.zeros((8, 8), dtype=numpy.int)
    for i in range(15):
        diagsum = 0
        if i <= 7:
            sind1 = i
            sind2 = 0
            jrange = i+1
        else:
            sind1 = 7
            sind2 = i-7
            jrange = 15-i
        for j in range(jrange):
            diagsum += abs(board[sind1-j][sind2+j])
        if diagsum == jrange:
            for k in range(jrange):
                diag1full[sind1-j][sind2+j] = True
    diag2full = numpy.zeros((8, 8), dtype=numpy.int)
    for i in range(15):
        diagsum = 0
        if i <= 7:
            sind1 = i
            sind2 = 7
            jrange = i+1
        else:
            sind1 = 7
            sind2 = 14-i
            jrange = 15-i
        for j in range(jrange):
            diagsum += abs(board[sind1-j][sind2-j])
        if diagsum == jrange:
            for k in range(jrange):
                diag2full[sind1-j][sind2-j] = True
    stable[2] = sum(sum(numpy.logical_and(numpy.logical_and(numpy.logical_and(colfull, rowfull), diag1full), diag2full)))
    return stable

# get evaluation
def evaluation(moves,board,color):
    moves_avaiable, _ = execution(board,-color)
    stable = getstable(board, color)
    value = position_weight(board,color) + 15* (len(moves)-len(moves_avaiable)) + 10*sum(stable)
    return value