from alpha_beta import *
#from SA import*
import numpy
from evaluation import execution
from new_board import*

board = numpy.zeros((8,8),dtype=numpy.int)
board[3][4] = board[4][3] = 1 #白
board[3][3] = board[4][4] = -1 #黑
AIColor = 1

print("Begin. Good Luck\n",board)
while(end(board)):
    if(special(board,AIColor)):
        print("No place to lay. You lost")
        break
    elif(special(board,-AIColor)):
        print("AI has no place to lay. You win")
        break
    move = AI_best(board,AIColor)
    x,y = move
    print("The AI chose :",move)
    new_board(board,x,y,1)
    print("After AI, the score is:",sum(sum(board)),"\nand the board is:")
    print(board)
    moves = execution(board,-AIColor)
    print("Accesssible choice:")
    print(moves[0])
    if moves[0] == []:
        input("No place to lay")
        continue
    else:
        i = input("Where to lay")
        # the order can not be changed
        if not i.isdigit() or int(i) >= len(moves[0]) or int(i) < 0:
            print("WARNING! IILEGAL INPUT. YOU LOST AN CHANCE!")
        else:
            yourmove = moves[0][int(i)]
            new_board(board,yourmove[0],yourmove[1],-AIColor)
            print("After Your choice, the score is:",sum(sum(board)),"\nand the board is:")
            print(board)

score = sum(sum(board))
if score < 0:
    print("You keep more than AI. You win")
elif score == 0:
    print("DEUCE")
else:
    print("AI keeps more than you. You lost")