import heapq

"""
 Utility class
 
 Data structures useful for implementing SearchAgents 
"""

class Stack:
  """
   Data structure that implements a last-in-first-out (LIFO)
  queue policy. 
  """
  def __init__(self):
    self.list = []
    
  def push(self,item):
    """
        Push 'item' onto the stack
    """
    self.list.append(item)

  def pop(self):
    """
       Pop the most recently pushed item from
       the stack
    """
    return self.list.pop()

  def isEmpty(self):
    """
        Returns true if the stack is empty
    """
    return len(self.list) == 0

class Queue:
  """
    Data structure that implements a first-in-first-out (FIFO)
  queue policy. 
  """
  def __init__(self):
    self.list = []
  
  def enqueue(self,item):
    """
      Enqueue the 'item' into the queue
    """
    self.list.insert(0,item)

  def dequeue(self):
    """
      Dequeue the earliest enqued item still in the queue. This
      operation removes the item from the queue.
    """
    return self.list.pop()

  def isEmpty(self):
    """
        Returns true if the queue is empty.
    """
    return len(self.list) == 0

class PriorityQueue:
  """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
  """
    
  def __init__(self):
    """
      heap: A binomial heap storing [priority,item]
      lists. 
      
      dict: Dictionary storing item -> [priorirty,item]
      maps so we can reach into heap for a given 
      item and update the priorirty and heapify
    """
    self.heap = []
    self.dict = {}
      
  def setPriority(self,item,priority):
    """
        Sets the priority of the 'item' to
    priority. If the 'item' is already
    in the queue, then its key is changed
    to the new priority, regardless if it
    is higher or lower than the current 
    priority.
    """
    if item in self.dict:
      self.dict[item][0] = priority
      heapq.heapify(self.heap)
    else:
      pair = [priority,item]
      heapq.heappush(self.heap,pair)
      self.dict[item] = pair
      
  def getPriority(self,item):
    """
        Get priority of 'item'. If 
    'item' is not in the queue returns None
    """
    if not item in self.dict:
      return None
    return self.dict[item][0]
      
  def dequeue(self):
    """
      Returns lowest-priority item in priority queue, or
      None if the queue is empty
    """
    if self.isEmpty(): return None
    (priority,item) = heapq.heappop(self.heap)
    del self.dict[item]
    return item  
  
  def isEmpty(self):
    """
        Returns True if the queue is empty
    """
    return len(self.heap) == 0
