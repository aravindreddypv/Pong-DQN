import gym
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import random
import datetime
# from gym import wrappers
from keras.models import load_model

def prepro(I):
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()
    


def QNetwork(input_layer, output_layer):
    model = Sequential()
    model.add(Dense(1024, input_dim=input_layer, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(output_layer))
    model.compile(optimizer='adam', loss='mse')
    return model


def getAction(t_observation):
    random.seed(datetime.datetime.now())
    r = random.uniform(0, 1)
    if r > epsilon:
        return random.randint(0, action_dimen-1)
    else:
        observation_array = np.array(t_observation)
        observation_array = observation_array.reshape(1, 6400)
        return np.argmax(model.predict(observation_array)[0])


# Batch train from memory -- Experience replay

def train():
    len_memory = len(memory)
    num_actions = action_dimen
    bsize = min(len_memory, batch_size)

    inputs = np.zeros((bsize, 6400))
    targets = np.zeros((bsize, num_actions))

    mini_batch = random.sample(memory, bsize)

    for i in range(bsize):
        entry = mini_batch[i]
        state_t = entry[0]
        action_t = entry[1]  # This is action index
        reward_t = entry[2]
        state_t1 = entry[3]
        t_done = entry[4]

        s_t = np.array(state_t)
        s_t = s_t.reshape(1, 6400)

        s_t1 = np.array(state_t1)
        s_t1 = s_t1.reshape(1, 6400)

        inputs[i] = s_t
        targets[i] = model.predict(s_t)
        Q_s = np.max(model.predict(s_t1))

        if t_done:
            targets[i, action_t] = reward_t  # Bellman Equation
        else:
            targets[i, action_t] = reward_t + gamma * Q_s  # Bellman Equation

    model.train_on_batch(inputs, targets)


# Storing into the memory

def remember(param):
    memory.append(param)
    if len(memory) > memory_size:
	del memory[0]




OBSERVE = 100000
EXPLORE = 2000000
FINAL_EPSILON = 0.9 # final value of epsilon
INITIAL_EPSILON = 0
memory_size = 50000
batch_size = 32
gamma = 0.99
memory = []


env = gym.make('Pong-v0')
# env = wrappers.Monitor(env, 'PongDQ-experiment-2')
input_layer = 6400
output_layer = env.action_space.n
action_dimen=env.action_space.n
model = QNetwork(input_layer, output_layer)  # Neural Network

epsilon=INITIAL_EPSILON
observation = env.reset()
prev_x = None
total_reward = 0
t=0
while True:
	t+=1
	# env.render()
	if prev_x is None:
		cur_x = prepro(observation)
	else:
		cur_x=temp_x
	x = cur_x - prev_x if prev_x is not None else np.zeros(6400)
	prev_x = cur_x
	temp_observation = x[:]
	action = getAction(temp_observation)
	if epsilon<=FINAL_EPSILON:
        	epsilon+=(FINAL_EPSILON-INITIAL_EPSILON)/EXPLORE
	observation, reward, done, info = env.step(action)
	temp_x=prepro(observation)
	total_reward += reward
	remember([x, action, reward, temp_x-cur_x, done])
	if t>OBSERVE:
        	train()
	if done:
		print int(total_reward),epsilon
		total_reward=0
		observation = env.reset()
		prev_x = None  
	if t%1000==0:
		model.save('model01.h5')
env.close()

