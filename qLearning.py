import numpy as np
import api
import visualization as v
from matplotlib import pyplot

import utils


def initializeQtable():
    return np.zeros((40, 40, 4))


def moveDirection(number):
    # translates the index returned from np.argmax() to direction for API
    if number == 0:
        return 'N'
    elif number == 1:
        return 'S'
    elif number == 2:
        return 'E'
    elif number == 3:
        return 'W'
    else:
        return 'Wrong Input'


def updateQtable(location, qTable, reward, gamma, newLocation, learningRate, move_Direction):
    # bellman equation- new Q(s,a) = Q(s,a) + learningRate * [R(s,a) + gamma * maxQ'(s',a') - Q(s,a)]
    # calculating new qValue
    value = reward + gamma * qTable[newLocation[0], newLocation[1], :].max() - qTable[
        location[0], location[1], move_Direction]
    newQTable = qTable[location[0], location[1], move_Direction] + learningRate * value
    qTable[location[0], location[1], move_Direction] = newQTable


def qLearning(qTable, worldId=0, mode='train', learningRate=0.5, gamma=0.9, epsilon=0.5, goodStates=[], badStates=[],
              epoch=0, obstacles=[], runs=0, visibility=True):
    a = api.API(worldId=worldId)
    w_res = a.enter_world()
    if visibility:
        print("w_res: ", w_res)
    terminal_state = False
    # create a var to track the type of terminal state
    good = False
    rewardsGained = []
    # find out where we are
    getLocation = a.getLocation()
    # create a list of everywhere we've been for the viz
    visited = []
    if visibility:
        print("getLocation", getLocation)
    if getLocation["code"] != "OK":
        print(f"error! Location: {getLocation}")
        return -1
    currentLocation = int(getLocation["state"].split(':')[0]), int(
        getLocation["state"].split(':')[1])
    # pyplot for visibility
    pyplot.figure(1, figsize=(10, 10))
    currentBoard = [[float('-inf')] * 40 for _ in range(40)]
    visited.append(currentLocation)
    while True:
        # Start Visibility
        currentBoard[currentLocation[1]][currentLocation[0]] = 1
        for i in range(len(currentBoard)):
            for j in range(len(currentBoard)):
                if currentBoard[i][j] != 0:
                    currentBoard[i][j] -= .1
        for obstacle in obstacles:
            if obstacle in visited:
                obstacles.remove(obstacle)
        v.updateVisualization(currentBoard, goodStates, badStates, obstacles, runs, epoch, worldId, currentLocation,
                              visibility)
        # End Visibility
        if mode == 'train':
            # epsilon greedy approach
            if np.random.uniform() < epsilon:
                unexploredLocations = np.where(qTable[currentLocation[0]][currentLocation[1]].astype(int) == 0)[0]
                exploredLocations = np.where(qTable[currentLocation[0]][currentLocation[1]].astype(int) != 0)[0]

                if unexploredLocations.size != 0:
                    moveNumber = int(np.random.choice(unexploredLocations))
                else:
                    moveNumber = int(np.random.choice(exploredLocations))
            else:
                moveNumber = np.argmax(qTable[currentLocation[0]][currentLocation[1]])

        else:
            # exploit mode
            moveNumber = np.argmax(qTable[currentLocation[0]][currentLocation[1]])
        moveMade = a.makeMove(move=moveDirection(moveNumber))
        if visibility:
            print("moveMade", moveMade)
        if moveMade["code"] != "OK":
            print(f"error! move: {moveMade}")
            move_failed = True
            while move_failed:
                moveMade = a.makeMove(move=moveDirection(moveNumber))
                if moveMade["code"] == 'OK':
                    move_failed = False
        # check for terminal state
        if moveMade["newState"] is not None:
            newLocation = int(moveMade["newState"]["x"]), int(moveMade["newState"]["y"])
            expectedLocation = list(currentLocation)
            latestMove = moveDirection(moveNumber)
            if latestMove == "N":
                expectedLocation[1] += 1
            elif latestMove == "S":
                expectedLocation[1] -= 1
            elif latestMove == "E":
                expectedLocation[0] += 1
            elif latestMove == "W":
                expectedLocation[0] -= 1
            expectedLocation = tuple(expectedLocation)
            if visibility:
                print(f"New Location {newLocation}")
            if visibility:
                print(f"Expected Location {expectedLocation}")

            if mode == "train":
                obstacles.append(expectedLocation)
            visited.append(newLocation)
            for obstacle in obstacles:
                if obstacle in visited:
                    obstacles.remove(obstacle)
        else:
            terminal_state = True
        reward = float(moveMade["reward"])
        rewardsGained.append(reward)

        # update qtable
        currentLocation = newLocation
        if mode == "train":
            updateQtable(currentLocation, qTable, reward, gamma, newLocation, learningRate, moveNumber)
        if terminal_state:
            if reward > 0:
                good = True
            if not (currentLocation in goodStates) and not (currentLocation in badStates):
                if good:
                    goodStates.append(currentLocation)
                else:
                    badStates.append(currentLocation)
            v.updateVisualization(currentBoard, goodStates, badStates, obstacles, runs, epoch, worldId, currentLocation,
                                  visibility)
            break
    pyplot.figure(2, figsize=(5, 5))
    # cumulative average
    cumulativeAverage = np.cumsum(rewardsGained) / (np.arange(len(rewardsGained)) + 1)
    utils.plotQlearning(worldId, epoch, cumulativeAverage, runs)
    return qTable, goodStates, badStates, obstacles
