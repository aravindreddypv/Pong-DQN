import gym
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import random
import datetime
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
# from gym import wrappers
from keras.models import load_model
import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer


def QNetwork():
    print("Started building the model")
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), padding='same',input_shape=(img_rows, img_cols, img_channels)))  # 80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(6))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    print("finished building the model")
    return model


def getAction(s_t):
    random.seed(datetime.datetime.now())
    r = random.uniform(0, 1)
    if r > epsilon:
        return random.randint(0, action_dimen - 1)
    else:
        q = model.predict(s_t)  # input a stack of 4 images, get the prediction
	print q
        max_Q = np.argmax(q)
	# print max_Q
        return max_Q


# Batch train from memory -- Experience replay

def train():
    len_memory = len(memory)
    num_actions = action_dimen
    bsize = min(len_memory, batch_size)

    inputs = np.zeros((bsize, 80,80,4))
    targets = np.zeros((bsize, num_actions))

    mini_batch = random.sample(memory, bsize)

    for i in range(bsize):
        entry = mini_batch[i]
        state_t = entry[0]
        action_t = entry[1]  # This is action index
        reward_t = entry[2]
        state_t1 = entry[3]
        t_done = entry[4]

        inputs[i:i + 1] = state_t

        targets[i] = model.predict(state_t)
        Q_sa = model.predict(state_t1)

        if t_done:
            targets[i, action_t] = reward_t  # Bellman Equation
        else:
            targets[i, action_t] = reward_t + gamma * np.max(Q_sa)  # Bellman Equation

    model.train_on_batch(inputs, targets)


# Storing into the memory

def remember(param):
    memory.append(param)
    if len(memory) > memory_size:
        del memory[0]


OBSERVE = 0
EXPLORE = 2000000
FINAL_EPSILON = 0.9 # final value of epsilon
INITIAL_EPSILON = 0.9
memory_size = 50000
batch_size = 100
gamma = 0.99
memory = []
LEARNING_RATE = 1e-7
img_rows, img_cols = 80, 80
# Convert image into Black and white
img_channels = 4  # We stack 4 frames

env = gym.make('Pong-v0')
# env = wrappers.Monitor(env, 'PongDQ-experiment-4')
input_layer = 6400
output_layer = env.action_space.n
action_dimen = env.action_space.n
model = load_model('model2test2.h5')  # Neural Network

epsilon=INITIAL_EPSILON
x_t = env.reset()
x_t = skimage.color.rgb2gray(x_t)
x_t = skimage.transform.resize(x_t, (80, 80))
x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
# print (s_t.shape)

# In Keras, need to reshape
s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4
total_reward = 0
t=0
while True:
    t+=1
    # print t,epsilon
    # env.render()
    action = getAction(s_t)
    if epsilon<=FINAL_EPSILON:
        epsilon+=(FINAL_EPSILON-INITIAL_EPSILON)/EXPLORE
    x_t1_colored, reward, done, info = env.step(action)
    print action,reward
    total_reward+=reward
    x_t1 = skimage.color.rgb2gray(x_t1_colored)
    x_t1 = skimage.transform.resize(x_t1, (80, 80))
    x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
    x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
    s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
    remember([s_t, action, reward, s_t1, done])
    if t>OBSERVE:
        train()
    if done:
	print int(total_reward),epsilon,t
        total_reward=0
	x_t = env.reset()
	x_t = skimage.color.rgb2gray(x_t)
	x_t = skimage.transform.resize(x_t, (80, 80))
	x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

	s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
	# print (s_t.shape)

	# In Keras, need to reshape
	s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4
    else:
    	s_t=s_t1
    if t % 1000 == 0:
        model.save('model2test4.h5')
env.close()
