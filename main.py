import gym
import random
import numpy as np
from keras import Sequential
from keras.layers import Dense
import keras

env = gym.make("MountainCar-v0")
observation = env.reset()

threshold = 50


# for _ in range(1):
#     env.render()
#     action = env.action_space.sample()  # your agent here (this takes random actions)
#     print(action)
#     observation, reward, done, info = env.step(action)
#
#     print(reward)
#
#     if done:
#         observation = env.reset()
# env.close()


def gatherData(dataSize):
    dataX = []
    dataY = []
    env.reset()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    while len(dataX) < dataSize:
        env.reset()
        currX = []
        currY = []
        score = 0
        sequenceSize = 10
        seqIndex = 0
        while True:


            if seqIndex >= sequenceSize:
                seqIndex = 0
                action = [0,2][random.randrange(0, 2)]
            else:
                seqIndex += 1

            observation, reward, done, info = env.step(action)
            currX.append(observation)
            y = [.0, .0, .0]
            y[action] = 1.0
            currY.append(y)
            score += reward
            # env.render()
            if done:
                env.reset()
                break
        if (score <= threshold):
            dataX += currX
            dataY += currY

    return np.array(dataX), np.array(dataY)


def createModel():
    model = Sequential()
    model.add(Dense(2, input_shape=(2,), activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model


def runModel():
    while 1 == 1:
        env.reset()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        while not done:
            action = model.predict(np.array([observation]))
            decision = np.argmax(action)
            observation, reward, done, info = env.step(decision)
            env.render()


dataX, dataY = gatherData(100000)

print(dataY)

model = createModel()
model.fit(dataX, dataY, epochs=10)
runModel()
