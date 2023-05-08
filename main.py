import qLearning
import numpy as np
import os
import utils


def main():
    mode = str(input(
        "\noption 't' is train (default)\noption 'e' is explore\n\nENTER OPTION: ") or "t")

    if mode == "t":
        world = int(input("\nwhich World would you like to train\nWORLD: ") or "0")
        epochs = int(input(
            f"\nhow many epochs\nEPOCHS: ") or "1")
        visibility = True
        epsilon = 0.9
        q_table = model.initializeQtable()
        if not (os.path.exists(f"./runs/world_{world}/")):
            os.makedirs(f"./runs/world_{world}/")
        runNumber = len([i for i in os.listdir(f"runs/world_{world}")])
        file_path = f"./runs/Q-table_world_{world}"
        goodStates = []
        badStates = []
        obstacles = []

        for epoch in range(epochs):
            print("Current EPOCH" + str(epoch) + ":\n\n")
            q_table, goodStates, badStates, obstacles = model.qLearning(
                q_table, world, 'train', learningRate=0.5, gamma=0.9, epsilon=epsilon,
                goodStates=goodStates, badStates=badStates,
                epoch=epoch, obstacles=obstacles, runs=runNumber, visibility=visibility)

            epsilon = utils.epsilonDecayFunction(epsilon, epoch)

            np.save(file_path, q_table)
        np.save(f"./runs/obstacles_world_{world}", obstacles)
        np.save(f"./runs/good_term_states_world_{world}", goodStates)
        np.save(f"./runs/bad_term_states_world_{world}", badStates)

    elif mode == "e":

        world = int(input("\nwhich World would you like to explore\nWORLD: ") or "0")
        epochs = int(input(
            f"\nhow many epochs\nEPOCHS: ") or "1")
        visibility = True
        file_path = f"./runs/Q-table_world_{world}"
        q_table = np.load(file_path + ".npy")
        obstacles = np.load(f"./runs/obstacles_world_{world}" + ".npy")
        goodStates = np.load(f"./runs/good_term_states_world_{world}" + ".npy")
        badStates = np.load(f"./runs/bad_term_states_world_{world}" + ".npy")
        obstacles = obstacles.tolist()
        goodStates = goodStates.tolist()
        badStates = badStates.tolist()
        epsilon = 0.6
        runNumber = len([i for i in os.listdir(f"runs/world_{world}")])
        for epoch in range(epochs):
            print("EPOCH #" + str(epoch) + ":\n\n")
            q_table, goodStates, badStates, obstacles = model.qLearning(
                q_table, worldId=world, mode='expl', learningRate=0.1, gamma=0.9, epsilon=epsilon,
                goodStates=goodStates, badStates=badStates,
                epoch=epoch, obstacles=obstacles, runs=runNumber, visibility=visibility)
    else:
        print("Incorrect option")
        exit()


if __name__ == "__main__":
    main()
