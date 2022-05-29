from shutil import move
from evaluation import*
from new_board import*
import numpy
import math
from random import random
from SA import*



# Alpha_Beta Pruning algorithm
def Alpha_Beta(board,depth,alpha,beta,actcolor,mycolor,maxdepth,T=100,Tf=1):
    # to get vaild positions
    moves,ValidBoardList = execution(board, actcolor)
    # nowhere to lay pieces, return an illegal number, and it's rival's turn
    if len(moves) == 0:
        return evaluation(moves,board,mycolor),(-1,-1)
    # run out of depth, end search and return
    if depth == 0:
        return evaluation(moves,board,mycolor),[]
    # beginning
    if depth == maxdepth:
        for i in range(len(moves)):
            # if it is a stable pieces, we have to choose it
            if Vmap[moves[i][0]][moves[i][1]] == Vmap[0][0] and actcolor == mycolor:
                return 1000,moves[i]
        if depth >= 4:
            V_value = []
            for i in range(len(ValidBoardList)):
                value, bestmove = Alpha_Beta(ValidBoardList[i], 1, -10000, 10000, -actcolor, mycolor, maxdepth,T,Tf)
            V_value.append(value)
        # sort the data in an ascending order
        ind = numpy.argsort(V_value)
        maxN = 6
        moves = [moves[i] for i in ind[0:maxN]]
        ValidBoardList = [ValidBoardList[i] for i in ind[0:maxN]]

    bestmove = []
    bestscore = -10000
    for i in range(len(ValidBoardList)):
        score, move = Alpha_Beta(ValidBoardList[i], depth-1, -beta, -max(alpha, bestscore), -actcolor, mycolor, maxdepth)
        score = -score
        if score > bestscore:
            bestscore = score
            bestmove = moves[i]
            # cut off
            if bestscore > beta:
                return bestscore, bestmove
    return bestscore, bestmove

# to decide the best choice for AI
# pick different depths on different situations
def AI_best(board, color):
    stage = sum(sum(abs(board)))
    if stage <= 9:
        depth = 5
    elif stage >= 49:
        depth = 6
    else:
        depth = 4
    _, bestmove = Alpha_Beta(board, depth, -10000, 10000, color, color, depth)
    return bestmove

# if there is 0
def end(board):
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                return True
    return False

# if there are all 1 or all -1
def special(board,color):
    for i in range(8):
        for j in range(8):
            if board[i][j] == -color:
                return False
    return True