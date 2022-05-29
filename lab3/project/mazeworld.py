# -*- coding: cp936 -*-
import search
import util

## Module Classes

class Maze:
  def __init__(self,grid):
    self.grid = grid
    self.numRows = len(grid)
    self.numCols = len(grid[0])
    for i in range(self.numRows):
      for j in range(self.numCols):
        if len(grid[i]) != self.numCols:
          raise "迷宫不是一个规则的矩形"
        if grid[i][j] == 'S':
          self.startCell = (i,j)
        if grid[i][j] == 'E':
          self.exitCell= (i,j)
    if self.exitCell == None:
      raise "未设置出发点"
    if self.startCell == None:
      raise "迷宫未设置出口"
   
  def isPassable(self, x, y):
    """
      x: 行位置
      y: 列位置
    
    若单元(x,y)是可以通过的，即 ' ' 或 '~'，则返回true
    """
    return self.isWater(x, y) or self.isClear(x, y)
  
  def isWater(self, x, y):
    """
      x: 行位置
      y: 列位置
    
    若单元(x,y)为水，即 '~'，则返回true
    """
    return self.grid[x][y] == '~'
    
  def isClear(self, x, y):
    """
      x: 行位置
      y: 列位置
    
    若单元(x,y)为空（畅通），即 ' '，则返回true
    """
    return self.grid[x][y] == ' '
    
  def isBlocked(self, x,y):
    """
      x: 行位置
      y: 列位置
      
    若单元(x,y)为障碍物，即'#'，则返回true
    """
    return self.grid[x][y] == '#'   
        
  def setBlock(self, x, y):  
    """
      x: 行位置
      y: 列位置
      
     在(x,y)位置设置障碍物，不过前提是该位置原来是为空的 
    """
    if(self.grid[x][y] != 'S' and self.grid[x][y] != 'E'):
      self.grid[x][y] = '#'
  
  def setClear(self, x, y):
    """
      x: 行位置
      y: 列位置
      
    设置(x,y)位置为空，前提是该位置即不是起点也不是出口
    """
    if(self.grid[x][y] != 'S' and self.grid[x][y] != 'E'):
      self.grid[x][y] = ' '
      
  def setWater(self, x, y):
    """
      x: 行位置
      y: 列位置
      
    设置(x,y)位置为水，前提是该位置即不是起点也不是出口
    """
    if(self.grid[x][y] != 'S' and self.grid[x][y] != 'E'):
      self.grid[x][y] = '~'
        
  def getNumRows(self):
    """
      获取迷宫的行数
    """
    return self.numRows
  
  def getNumCols(self):
    """
      获取迷宫的列数
    """
    return self.numCols  
   
  def getStartCell(self):
    """
      获取起点的(x,y)位置
    """
    return self.startCell
  
  def getExitCell(self):
    """
      获取迷宫出口的(x,y)位置
    """
    return self.exitCell

  def __getAsciiString(self):
    """
      获取以字符串形式描述的迷宫布局
    """
    lines = []
    headerLine = ' ' + ('- ' * (self.numCols)) + ' '
    lines.append(headerLine)
    for row in self.grid:
      rowLine = '|' + ' ' .join(row)  + '|'
      lines.append(rowLine)
    lines.append(headerLine)
    return '\n'.join(lines)

  def __str__(self):
    return self.__getAsciiString()


class MazeSearchProblem(search.SearchProblem):
    """
      Implementation of a SearchProblem for the 
      Maze World domain
      
      Each state is encoded as a (x,y) pair for the 
      position in the grid. The start state is the 
      start cell for the maze and the only goal is the
      Maze exit cell. 
    """                                                                                    
    def __init__(self,maze):
      """
      
      """
      self.maze = maze     
      self.numNodesExpanded = 0        
      self.expandedNodeSet = {}                                                                       
                                                                                        
    def getStartState(self):
      """
      
      """
      return self.maze.getStartCell()
        
    def isGoalState(self,state):
      """
      
      """
      return state == self.maze.getExitCell()
    
    def __isValidState(self,state):
      """
        state: Cell position
      
      Returns true is the given state corresponds
      to an unblocked and valid maze position
      """
      x,y = state
      if x < 0 or x >= self.maze.getNumRows():
        return False
      if y < 0 or y >= self.maze.getNumCols():
        return False
      return not self.maze.isBlocked(x,y)
          
    def getSuccessors(self,state):
        """
          state: Cell position 
        
        Returns list of (successor,cost) pairs where
        each succesor is either left, right, up, or down 
        from the original state and the cost is 1.0 for each
        """
        # Update Search Stats
        self.numNodesExpanded += 1
        self.expandedNodeSet[state] = 1
        states = []
        x,y = state
        # Right
        states.append((x,y+1))       
        # Down
        states.append((x+1,y))
        # Left
        states.append((x,y-1)) 
        # Up 
        states.append((x-1,y))        
          
        # So successors appear in order (Right,Down,Left,Up)
        states.reverse()  
        return [(x,self.getCost(x)) for x in states if self.__isValidState(x)]        
        
    def getCost(self, state):
      """
        Returns the step cost of entering each terrain type.
        
        Blank spaces have cost 1.
        Water spaces have cost 5.
      """
      x, y = state
      if self.maze.isClear(x, y):
        return 1
      elif self.maze.isWater(x, y):
        return 5
      elif state == self.maze.getStartCell() or state == self.maze.getExitCell():
        return 1
      else:
        raise "The cost of an impassable cell is undefined." 
        
    def getMaze(self):
        return self.maze

    # Search Stats

    def displaySearchStats(self):
        """
          Display number of nodes expanded by 'getSuccessors'
        """
        print 'Number of nodes expanded:',self.numNodesExpanded
        print 'Number of unique nodes expanded:', len(self.expandedNodeSet)
    
    def resetSearchStats(self):        
       self.numNodesExpanded = 0
       self.expandedNodeSet = {}
       
## Simple Maze Agent

class SimpleMazeAgent(search.SearchAgent):
  """
    一直往右走，直到走到了迷宫的出口，或者无法再往右了；
    这是一个最简单的走迷宫的智能体，如果起点与出口同在一
    行上，而且两者之间没有障碍物，该智能体才可能找到问题
    的解决方案，否则必定失败
  """
  def solve(self, mazeSearchProblem):
    solution = []
    cost = 0.0
    cell = mazeSearchProblem.getStartState()
    # Move to the right as long as we can
    while True:  
      solution.append(cell)
      if mazeSearchProblem.isGoalState(cell):
        return solution, cost
      nextCell, nextCost = None, 0.0
      for successor, stepCost in mazeSearchProblem.getSuccessors(cell):
        delta_y = successor[1]-cell[1]
        if delta_y == 1: # Right 
          nextCell = successor
          nextCost = stepCost
          break  
      if nextCell != None:
        cell = nextCell
        cost += nextCost
      else: 
        print "SimpleMazeAgent: 不能再往右了，我不知道该怎么办了，放弃！"
        return (None, 0.0)
              
## Maze Heuristic Function Classes

def manhattanDistance(state, mazeSearchProblem):
    """
      Returns the Manhattan distance between the state
      and the goal for the provided maze.
      
      The manhattan distance between points (x0,y0)
      and (x1,y1) is |x0-x1| + |y0-y1|
    """   
    maze = mazeSearchProblem.maze
    delta_x = maze.getExitCell()[0] - state[0] 
    delta_y = maze.getExitCell()[1] - state[1]
    return abs(delta_x) + abs(delta_y)

## Module Methods

def testAgentOnMaze(agent, maze, verbose=True):
  """
     Test the search agent 'agent' on the 'maze'
     and prints the cost of the solution and also
     displays the maze along with 'x' for the cells
     used in the solution and 'o' for cells expanded
     but not used in the solution
  """
  problem = MazeSearchProblem(maze) # 生成MazeSearchProblem类的一个实例problem，迷宫布局为maze
  solution, cost = agent.solve(problem)
  if solution == None:
    print '找不到解决方案!'
    problem.displaySearchStats()
    problem.resetSearchStats()
    return 
  if verbose:
    grid_copy = []
    for row in maze.grid:
      grid_copy.append([x for x in row]) 
    for x,y in problem.expandedNodeSet:
      ch = maze.grid[x][y]
      if ch != 'S' and ch != 'E': grid_copy[x][y] = 'o' 
    for x,y in solution:  
      ch = maze.grid[x][y]
      if ch != 'S' and ch != 'E': grid_copy[x][y] = 'x'    
    maze_copy = Maze(grid_copy)
    print maze_copy
    print "x - 解路径上的单元格"
    print "o - 搜索过程扩展过的单元格"
    print "-------------------------------" 
  print '解的耗散:', cost  
  problem.displaySearchStats()
                                          
def readMazeFromFile(file):
    """
      file: Name of file containing maze
      
     Returns a Maze instance 
    """
    fin = open(file)
    lines = fin.read().splitlines()
    grid = []
    for line in lines: 
      grid.append(list(line))
    return Maze(grid)

def createEmptyMaze(rows, cols):
  """ 
  Returns an empty (rows x cols) maze with a central entrace
  and an exit at the right 
  """
  grid = []
  for i in range(rows):
    grid.append([' ' for x in range(cols)])
  grid[rows/2][cols/2] = 'S'
  grid[rows/2][cols-1] = 'E'
  return Maze(grid)