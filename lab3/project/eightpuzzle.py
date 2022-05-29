import search
import random

# Module Classes

class EightPuzzleState:
 def __init__( self, numbers ):
   self.cells = []
   numbers = numbers[:] # Make a copy so as not to cause side-effects.
   numbers.reverse()
   for row in range( 3 ):
     self.cells.append( [] )
     for col in range( 3 ):
       self.cells[row].append( numbers.pop() )
       if self.cells[row][col] == 0:
         self.blankLocation = row, col

 def isGoal( self ):
   current = 0
   for row in range( 3 ):
    for col in range( 3 ):
      if current != self.cells[row][col]:
        return False
      current += 1
   return True

 def legalMoves( self ):
   moves = []
   row, col = self.blankLocation
   if(row != 0):
     moves.append('up')
   if(row != 2):
     moves.append('down')
   if(col != 0):
     moves.append('left')
   if(col != 2):
     moves.append('right')
   return moves

 def result(self, move):
   row, col = self.blankLocation
   if(move == 'up'):
     newrow = row - 1
     newcol = col
   elif(move == 'down'):
     newrow = row + 1
     newcol = col
   elif(move == 'left'):
     newrow = row
     newcol = col - 1
   elif(move == 'right'):
     newrow = row
     newcol = col + 1
   else:
     raise "Illegal Move"

   # Create a copy of the current eightPuzzle
   newPuzzle = EightPuzzleState([0, 0, 0, 0, 0, 0, 0, 0, 0])
   newPuzzle.cells = [values[:] for values in self.cells]
   # And update it to reflect the move
   newPuzzle.cells[row][col] = self.cells[newrow][newcol]
   newPuzzle.cells[newrow][newcol] = self.cells[row][col]
   newPuzzle.blankLocation = newrow, newcol

   return newPuzzle

 # Utilities for comparison and display
 def __eq__(self, other):
   for row in range( 3 ):
      if self.cells[row] != other.cells[row]:
        return False
   return True

 def __hash__(self):
   return hash(str(self.cells))

 def __getAsciiString(self):
   lines = []
   horizontalLine = ('-' * (13))
   lines.append(horizontalLine)
   for row in self.cells:
     rowLine = '|'
     for col in row:
       if col == 0:
         col = ' '
       rowLine = rowLine + ' ' + col.__str__() + ' |'
     lines.append(rowLine)
     lines.append(horizontalLine)
   return '\n'.join(lines)

 def __str__(self):
   return self.__getAsciiString()

# TODO: Implement The methods in this class

class EightPuzzleSearchProblem(search.SearchProblem):
  def __init__(self,puzzle):
    self.puzzle = puzzle
    self.numNodesExpanded = 0
    self.expandedNodeSet = {}

  def getStartState(self):
    return self.puzzle    # Your code here
      
  def isGoalState(self,state):
    return state.isGoal() # Your code here
   
  def getSuccessors(self,state):
    # Leave these lines in.  They keep track of the search progress
    self.numNodesExpanded += 1
    self.expandedNodeSet[state] = 1

    # Your code here
    return [(state.result(move), 1.0) for move in state.legalMoves()]
  # Search Stats

  def displaySearchStats(self):
    print 'Number of nodes expanded:',self.numNodesExpanded
    print 'Number of unique nodes expanded:', len(self.expandedNodeSet)

  def resetSearchStats(self):
    self.numNodesExpanded = 0
    self.expandedNodeSet = {}

# Heuristics

def misplacedTiles(state, eightPuzzleSearchProblem):
  current = 0
  total = 0.0
  for row in range( 3 ):
      for col in range( 3 ):
          if current and current != state.cells[row][col]:
              total += 1.0
          current += 1
  return total

def manhattanDistance(state, eightPuzzleSearchProblem):
  goalCoordinates = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
  total = 0.0
  for row in range( 3 ):
      for col in range( 3 ):
          actual = state.cells[row][col]
          goalrow, goalcol = goalCoordinates[actual]
          dist = abs(goalrow-row) + abs(goalcol-col)
          if actual:
              total += dist
  return total

def gaschnig(state, eightPuzzleSearchProblem):
    total = 0.0
    currentCoordinates = range(9)
    for row in range( 3 ):
        for col in range( 3 ):
            currentCoordinates[state.cells[row][col]] = (row,col)
    goalCoordinates = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    while currentCoordinates != goalCoordinates:
        blankLocation = currentCoordinates[0]
        if blankLocation != (0,0):
            tiletomove = goalCoordinates.index(blankLocation)
            currentCoordinates[0] = currentCoordinates[tiletomove]
            currentCoordinates[tiletomove] = blankLocation
        else:
            tiletomove = 1
            while currentCoordinates[tiletomove] == goalCoordinates[tiletomove]:
                tiletomove += 1
            currentCoordinates[0] = currentCoordinates[tiletomove]
            currentCoordinates[tiletomove] = blankLocation
        total += 1.0
    return total

def max_heuristic(state, eightPuzzleSearchProblem):
    return max([manhattanDistance(state,None), gaschnig(state, None)])

# Module Methods

EIGHT_PUZZLE_DATA = [[1, 0, 2, 3, 4, 5, 6, 7, 8], 
                     [1, 7, 8, 2, 3, 4, 5, 6, 0], 
                     [4, 3, 2, 7, 0, 5, 1, 6, 8], 
                     [5, 1, 3, 4, 0, 2, 6, 7, 8], 
                     [1, 2, 5, 7, 6, 8, 0, 4, 3], 
                     [0, 3, 1, 6, 8, 2, 7, 5, 4]]

def loadEightPuzzle(puzzleNumber):
  """
    puzzleNumber: The number of the eight puzzle to load.
    
    Returns an eight puzzle object generated from one of the
    provided puzzles in EIGHT_PUZZLE_DATA.
    
    puzzleNumber can range from 0 to 5.
    
    >>> print loadEightPuzzle(0)
    -------------
    | 1 |   | 2 |
    -------------
    | 3 | 4 | 5 |
    -------------
    | 6 | 7 | 8 |
    -------------
  """
  return EightPuzzleState(EIGHT_PUZZLE_DATA[puzzleNumber])

def createRandomEightPuzzle(moves=100):
 """
   moves: number of random moves to apply

   Creates a random eight puzzle by applying
   a series of 'moves' random moves to a solved
   puzzle.
 """
 puzzle = EightPuzzleState([0,1,2,3,4,5,6,7,8])
 for i in range(moves):
   # Execute a random legal move
   puzzle = puzzle.result(random.sample(puzzle.legalMoves(), 1)[0])
 return puzzle
