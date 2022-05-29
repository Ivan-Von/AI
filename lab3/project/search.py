import util

## Abstract Search Classes
class SearchProblem:
  def getStartState(self):
     abstract    
  def getSuccessors(self, state):
     abstract
  def isGoalState(self, state):
    abstract
  def displaySearchStats(self):
    abstract
  def resetSearchStats(self):
    abstract
class SearchAgent:
  def solve(self, searchProblem):
    abstract

## Arlo's solution starts here

class Node:
  def __init__(self, state, parent, pathcost):
    self.state = state
    self.parent = parent
    self.pathcost = pathcost

def tracepath(node):
  if node.parent:
    return tracepath(node.parent) + [node.state]
  else:
    return [node.state]

class DepthFirstSearchAgent(SearchAgent):
  def solve(self,searchProblem):
    fringe = util.Stack()
    fringe.push(Node(searchProblem.getStartState(), None, 0.0))
    closed = set() # faster than list for membership test

    while not fringe.isEmpty():
      node = fringe.pop()
      if searchProblem.isGoalState(node.state):
        return (tracepath(node), node.pathcost)
      elif node.state not in closed:
          closed.add(node.state)
          for (nextstate, nextcost) in searchProblem.getSuccessors(node.state):
            fringe.push(Node(nextstate, node, node.pathcost + nextcost))
    return (None, 0.0)

class BreadthFirstSearchAgent(SearchAgent):
  def solve(self,searchProblem):
    fringe = util.Queue()
    fringe.enqueue(Node(searchProblem.getStartState(), None, 0.0))
    closed = set() # faster than list for membership test

    while not fringe.isEmpty():
      node = fringe.dequeue()
      if searchProblem.isGoalState(node.state):
        return (tracepath(node), node.pathcost)
      elif node.state not in closed:
          closed.add(node.state)
          for (nextstate, nextcost) in searchProblem.getSuccessors(node.state):
            fringe.enqueue(Node(nextstate, node, node.pathcost + nextcost))
    return (None, 0.0)

class UniformCostSearchAgent(SearchAgent):
  def solve(self,searchProblem):
    fringe = util.PriorityQueue()
    fringe.setPriority(Node(searchProblem.getStartState(), None, 0.0), 0.0)
    closed = set() # faster than list for membership test

    while not fringe.isEmpty():
      node = fringe.dequeue()
      if searchProblem.isGoalState(node.state):
        return (tracepath(node), node.pathcost)
      elif node.state not in closed:
          closed.add(node.state)
          for (nextstate, nextcost) in searchProblem.getSuccessors(node.state):
            g = node.pathcost + nextcost
            fringe.setPriority(Node(nextstate, node, g), g)
    return (None, 0.0)
      
class AStarSearchAgent(SearchAgent):
  def __init__(self, heuristicFn):
    self.heuristicFn = heuristicFn 

  def solve(self,searchProblem):
    fringe = util.PriorityQueue()

    startstate = searchProblem.getStartState()
    g = 0.0
    h = self.heuristicFn(startstate, searchProblem)

    fringe.setPriority(Node(startstate, None, 0.0), g+h)
    closed = set() # faster than list for membership test

    while not fringe.isEmpty():
      node = fringe.dequeue()
      if searchProblem.isGoalState(node.state):
        return (tracepath(node), node.pathcost)
      elif node.state not in closed:
          closed.add(node.state)
          for (nextstate, nextcost) in searchProblem.getSuccessors(node.state):
            g = node.pathcost + nextcost
            h = self.heuristicFn(nextstate, searchProblem)
            fringe.setPriority(Node(nextstate, node, g), g+h)
    return (None, 0.0)
  
