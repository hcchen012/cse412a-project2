# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    score = 0
    foodList = newFood.asList()
    ghostDist = []
    foodDist = []

    for i in newGhostStates:
      ghostDist.append(manhattanDistance(newPos, i.getPosition()))

    for i in foodList:
      foodDist.append(manhattanDistance(newPos, i))

    if len(newGhostStates) > 0:
      score += min(ghostDist)

    if len(foodDist) > 0:
      score -= min(foodDist)

    return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    agentIndex = self.index

    def getMaxValue(gameState,depth):
      #reach terminal state
      if gameState.isWin() or gameState.isLose() or depth == self.depth or not gameState.getLegalActions(agentIndex):
        return self.evaluationFunction(gameState)

      value = -float("inf")
      actions = gameState.getLegalActions(agentIndex)
      for action in actions:
        if action != Directions.STOP:
          successor = gameState.generateSuccessor(agentIndex, action)
          value = max(value, getMinValue(successor, depth, agentIndex+1))
      return value

    def getMinValue(gameState,depth,ghostIndex):
      #reach terminal state
      if gameState.isWin() or gameState.isLose() or depth == self.depth or not gameState.getLegalActions(ghostIndex):
        return self.evaluationFunction(gameState)

      value = float("inf")
      actions = gameState.getLegalActions(ghostIndex)
      for action in actions:
        if action != Directions.STOP:
          successor = gameState.generateSuccessor(ghostIndex, action)
          #at last ghost
          if ghostIndex == gameState.getNumAgents() - 1:
            value = min(value, getMaxValue(successor, depth + 1))
          else:
            value = min(value, getMinValue(successor, depth, ghostIndex + 1))
      return value
  
    max_utility = -float("inf")
    actions = gameState.getLegalActions(agentIndex)
    best_action = actions[agentIndex]

    for action in actions:
      if action != Directions.STOP:
        successor = gameState.generateSuccessor(agentIndex, action)
        result = getMinValue(successor, 0, agentIndex+1)
        if result >= max_utility:
          max_utility = result
          best_action = action
    return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    agentIndex = self.index

    def getMaxValue(gameState,depth, alpha, beta):
      #reach terminal state
      if gameState.isWin() or gameState.isLose() or depth == self.depth or not gameState.getLegalActions(agentIndex):
        return self.evaluationFunction(gameState)

      value = -float("inf")
      actions = gameState.getLegalActions(agentIndex)
      for action in actions:
        if action != Directions.STOP:
          successor = gameState.generateSuccessor(agentIndex, action)
          value = max(value, getMinValue(successor, depth, agentIndex+1, alpha, beta))
          alpha = max (alpha, value)
          if alpha >= beta: 
            return value #prune
      return value

    def getMinValue(gameState,depth,ghostIndex, alpha, beta):
      #reach terminal state
      if gameState.isWin() or gameState.isLose() or depth == self.depth or not gameState.getLegalActions(ghostIndex):
        return self.evaluationFunction(gameState)

      value = float("inf")
      actions = gameState.getLegalActions(ghostIndex)
      for action in actions:
        if action != Directions.STOP:
          successor = gameState.generateSuccessor(ghostIndex, action)
          #at last ghost
          if ghostIndex == gameState.getNumAgents() - 1:
            value = min(value, getMaxValue(successor, depth + 1, alpha, beta))
          else:
            value = min(value, getMinValue(successor, depth, ghostIndex + 1, alpha, beta))
          beta = min(beta, value)
          if alpha >= beta: 
            return value
      return value

    max_utility = -float("inf")
    actions = gameState.getLegalActions(agentIndex)
    best_action = actions[agentIndex]

    for action in actions:
      if action != Directions.STOP:
        successor = gameState.generateSuccessor(agentIndex, action)
        result = getMinValue(successor, 0, agentIndex+1, -float("inf"), float("inf"))
        if result >= max_utility:
          max_utility = result
          best_action = action
    return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    agentIndex = self.index

    def getMaxValue(gameState,depth):
      #reach terminal state
      if gameState.isWin() or gameState.isLose() or depth == self.depth or not gameState.getLegalActions(agentIndex):
        return self.evaluationFunction(gameState)

      value = -float("inf")
      actions = gameState.getLegalActions(agentIndex)
      for action in actions:
        if action != Directions.STOP:
          successor = gameState.generateSuccessor(agentIndex, action)
          value = max(value, getExpectedValue(successor, depth, agentIndex+1))
      return value

    def getExpectedValue(gameState,depth,ghostIndex):
      #reach terminal state
      if gameState.isWin() or gameState.isLose() or depth == self.depth or not gameState.getLegalActions(ghostIndex):
        return self.evaluationFunction(gameState)

      value = float("inf")
      actions = gameState.getLegalActions(ghostIndex)
      probability = 1.0 / len(actions)
      for action in actions:
        if action != Directions.STOP:
          successor = gameState.generateSuccessor(ghostIndex, action)
          #at last ghost
          if ghostIndex == gameState.getNumAgents() - 1:
            value = probability * getMaxValue(successor, depth + 1)
          else:
            value = probability * getExpectedValue(successor, depth, ghostIndex + 1)
      return value
  
    max_utility = -float("inf")
    actions = gameState.getLegalActions(agentIndex)
    best_action = actions[agentIndex]

    for action in actions:
      if action != Directions.STOP:
        successor = gameState.generateSuccessor(agentIndex, action)
        result = getExpectedValue(successor, 0, agentIndex+1)
        if result >= max_utility:
          max_utility = result
          best_action = action
    return best_action


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: For the evaluation function, I considered the food, capsules, regular ghosts, and scared ghosts currently in the game state. 
    Food pellets closer to pacman should increase the score more, which is why I added 10/min(foodDist). 
    Also, if pacman is close to a capsule, there should be a larger incentive to eat the capsule which is why I added 18/min(capsuleDist) to the score. 
    Being closer to a scared ghost should increase the score while being close to a normal ghost should negatively affect the score.
  """
  "*** YOUR CODE HERE ***"
  #util.raiseNotDefined()
  if currentGameState.isWin():
    return 9999999
  if currentGameState.isLose():
    return -9999999

  currPosition = currentGameState.getPacmanPosition()
  foodList = currentGameState.getFood().asList()

  numGhosts = []
  numScaredGhosts = []
  foodDist = []
  ghostDist = []
  scaredGhostDist = []
  capsuleDist = []
  foodScore = 0
  ghostScore = 0
  capScore = 0

  #food pellets closer to current position should increase the food score more
  for i in foodList:
    foodDist.append(manhattanDistance(currPosition, i))
  if len(foodDist) is not 0:
    foodScore += 10.0/ float(min(foodDist))

  #bigger score for capsules; want pacman to eat them as soon as possible
  for i in currentGameState.getCapsules():
    capsuleDist.append(manhattanDistance(currPosition, i))
  if len(capsuleDist) is not 0:
    capScore += 17/float(min(capsuleDist))

  #ghost scores
  for i in currentGameState.getGhostStates():
    if i.scaredTimer > 0:
      numScaredGhosts.append(i)
    else:
      numGhosts.append(i)
  
  for i in numGhosts:
    ghostDist.append(manhattanDistance(currPosition, i.getPosition()))
  for i in numScaredGhosts:
    scaredGhostDist.append(manhattanDistance(currPosition, i.getPosition()))


  if len(scaredGhostDist) == 0:
    ghostScore -= 10/min(ghostDist) #if regular ghost, decrease score
  elif len(ghostDist) == 0:
    ghostScore += 10/min(scaredGhostDist) #if scared ghost, increase score

  return 1.4*currentGameState.getScore() + foodScore + 1.4*ghostScore + capScore

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

