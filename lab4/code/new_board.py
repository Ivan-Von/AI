direction = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

# to make sure the coord is reasonable
def reasonable(x,y):
    return x>=0 and x<8 and y>=0 and y<8

# to place new chess pieces to the chosen place and update the board
def new_board(board,x,y,color):
    if x < 0 or y < 0 or x > 8 or y > 8: 
        return False
    board[x][y] = color
    valid = False
    for d in range(8):
        # search the eight directions one by one 
        i = x + direction[d][0]
        j = y + direction[d][1]
        # find the last piece with opposite color
        while reasonable(i,j) and board[i][j] == -color:
            i += direction[d][0]
            j += direction[d][1]
        # reverse the pieces with the opposite color
        if reasonable(i,j) and board[i][j] == color:
            while True:
                i -= direction[d][0]
                j -= direction[d][1]
                if i == x and j == y:
                    break
                valid = True
                board[i][j] = color
    return valid
