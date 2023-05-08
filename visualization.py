from matplotlib import pyplot
import os


def updateVisualization(data, goodStates, badStates, obstacles, currentRun, epoch, world, location, visibility):
    pyplot.figure(1)
    pyplot.clf()
    pyplot.imshow(data)
    pyplot.draw()
    pyplot.title(f'WORLD: {world} EPOCH: {epoch}')
    pyplot.ylim(-1, 41)
    pyplot.xlim(-1, 41)
    for z in obstacles:
        pyplot.plot(z[0], z[1], marker="x", color='k')
    for x in goodStates:
        pyplot.plot(x[0], x[1], marker="x", color='g')
    for y in badStates:
        pyplot.plot(y[0], y[1], marker="x", color='r')
    pyplot.plot(location[0], location[1], marker="*", color='blue')
    if visibility:
        pyplot.show(block=False)
        pyplot.pause(0.01)

    if not os.path.exists("./runs/world_{}/attempt_{}".format(world, currentRun)):
        os.makedirs("./runs/world_{}/attempt_{}".format(world, currentRun))
    pyplot.savefig("./runs/world_{}/attempt_{}/epoch_{}.png".format(world, currentRun, epoch))
