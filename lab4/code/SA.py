from evaluation import*
from new_board import*
import math
from random import random

# get new x,y
def generate_new(x, y, T):
    x_new = int(x + T * (random() - random())*9)
    y_new = int(y + T * (random() - random())*9)
    return x_new,y_new

# to judge if is's acceptable
def Metrospolis(f, f_new,T):
    if(f_new > f):
        return 1
    else:
        p = math.exp((f-f_new)/T)
        if random() < p:
            return 1
        else:
            return 0

# def run(board,actcolor,mycolor,T=100,Tf=1):
#     bestmove = []
#     while T > Tf:
#         moves,ValidBoardList = execution(board, actcolor)
#         for i in range(len(ValidBoardList)):
#             score = evaluation(moves,ValidBoardList[i],mycolor)
#             x_new,y_new = generate_new(moves[i][0],moves[i][1])
#             score_new = generate_new(x_new,y_new)
#             if Metrospolis(score,score_new,T):
#                 bestmove.append((x_new,y_new))
#                 best_score = score_new
#         T = T * 0.1
#     return best_score,bestmove

def SA_A_B(board,depth,actcolor,T=100,Tf=0):#T=depth
    moves,ValidBoardList = execution(board,actcolor)
    # 2 special examples
    if len(moves) == 0:
        return evaluation(moves,board,actcolor),(-1,-1)
    if depth == 0:
        return evaluation(moves,board,actcolor),[]
    # when it's still have tem
    while(T > Tf):
        x_index,y_index = moves[0][0],moves[0][1]
        board_index = ValidBoardList[0]
        for i in range(len(moves)):
            # to get len(moves) random numbers
            new_board(board_index,x_index,y_index,actcolor)
            score = sum(sum(board * Vmap))*actcolor
            # new value
            x_new,y_new = generate_new(moves[i][0],moves[i][1],T)
            index,next_board = new_board(board_index,x_new,y_new,actcolor)
            if(index):
                score_new = sum(sum(board * Vmap))*actcolor
                # if it's acceptable
                if Metrospolis(score,score_new,T): 
                    bestmove = (x_new,y_new)
                    bestscore = score_new
            # update the baseline
            x_index = x_new
            y_index = y_new
            board_index = next_board
        # SA 
        T = T * 0.1
    return bestscore,bestmove

def AI_best(board, color):
    stage = sum(sum(abs(board)))
    if stage <= 9:
        depth = 5
    elif stage >= 49:
        depth = 6
    else:
        depth = 4
    _, bestmove = SA_A_B(board, depth, color, color)
    return bestmove