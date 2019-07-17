import numpy as np
import math
import sys

# this is implementation of the simplest reinforcement learning algorithm; namely, q learning

sys.setrecursionlimit(10000)

"""
Some info about problem setup:
  -we assume n grid cells (n x n matrix)
  -impossible moves will have -1 reward
  -possible moves but no reward implies reward of 0
  -each episode ends with reaching reward
  -epsilon decreases incrementally with each iteration eventually ending up with 0 (at the start, it's equal to 1)
  -with each iteration, agent is likely to prefer non-random actions; maximizing reward based on current information
  -different values of learning rate and discount_factor can be experimented, if no idea start with default values
  -MAIN DISADVANTAGE of q-learning is: q table can be very large (n x n matrix) if number of states are large (n) so it will require too much iterations
"""

class qModel():
  # initialize variables
  def __init__(self,reward_table,learning_rate,discount_factor,max_iterations,epsilon,targetState):
    self.q_table = np.zeros((len(reward_table),len(reward_table)))
    self.reward_table = reward_table
    self.learning_rate = learning_rate
    self.discount_factor = discount_factor
    self.max_iterations = max_iterations
    self.epsilon = 1
    self.targetState = targetState
    self.iterations = max_iterations
    self.actionsPerEpisode = 0

  # this function iterates through states
  def iterate(self,currentState=-1):
    if currentState == -1:
      self.actionsPerEpisode = 0
      currentState = round(np.random.uniform(low=0,high=len(reward_table)-1))
      while currentState == self.targetState: # ensure that starting point is not target point
        currentState = round(np.random.uniform(low=0,high=len(reward_table)-1))
    while True:
      prob = np.random.rand(1)
      if self.epsilon > prob[0]:
        nextState = round(np.random.uniform(low=0,high=len(reward_table)-1))
        if nextState != currentState:  # ensure that next state is not itself
          break
      else:
        nextState = np.argmax(self.q_table[currentState])
        if nextState != currentState:  # ensure that next state is not itself
          break
    self.actionsPerEpisode += 1
    #print(self.epsilon,prob[0])
    #print(currentState,nextState)
    self.q_table[currentState][nextState] += self.learning_rate * (self.reward_table[currentState][nextState] +
                                                        self.discount_factor * np.max(self.q_table[currentState]) - self.q_table[currentState][nextState])
    self.epsilon -= 1 / self.max_iterations # epsilon decreases incrementally
    self.iterations -= 1 # when agent moves, iteration decreases by one
    if self.iterations > 0:
      if (nextState == self.targetState):
        print(self.actionsPerEpisode)
        print("episode ends...")  # agent reached the highest reward and episode restarts
        currentState = -1
        self.iterate(currentState)
      else:
        self.iterate(nextState)
    return self.q_table

reward_table = [[-1,0,-1,0.1,-1,-1,-1,-1,-1],[0,-1,0,-1,0.1,-1,-1,-1,-1],[-1,0,-1,-1,-1,0,-1,-1,-1],
                [0,-1,-1,-1,0.1,-1,0,-1,-1],[-1,0,-1,0,-1,0.1,-1,0,-1],[-1,-1,0,-1,0,-1,-1,-1,+1],
                [-1,-1,-1,0,-1,-1,-1,0.1,-1],[-1,-1,-1,-1,0,-1,0,-1,+1],[-1,-1,-1,-1,-1,0,-1,0,-1]]

qModel = qModel(reward_table,learning_rate=0.1,discount_factor=0.8,max_iterations=5000,targetState=8,epsilon=1)

print(qModel.iterate())                 
