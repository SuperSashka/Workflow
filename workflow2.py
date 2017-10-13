# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import workflowenv2 as wf



EPISODES = 10000

def leaderboardcompare(lbrd,score):
    if score>=lbrd[0]:
        lbrd.extend([score]);
        lbrd.sort()
        del lbrd[0];
        rew=lbrd.index(score)+1;
        rew=0.5*rew
    else:
        rew=-0.01;
    return [lbrd,rew];


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.01  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    state_size = 33
    action_size = 16
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 7
    lbrd=[0,0,0,0,0]
    cumulativereward=0
    scoreavg=0
    learning1=[]        
    for e in range(EPISODES):
        comptime=np.random.randint(20, size=(3, 5))+1
        chns=wf.treegen(5)
        wfl=wf.workflow(chns,comptime)
        done=wfl.completed
        state = wfl.state
        state = np.reshape(state, [1, state_size])
        for time in range(5000):
            action = agent.act(state)
            total_time,_=wfl.act(action)
            next_state = wfl.state
            done=wfl.completed
            #score+=reward
            reward=0
            if not(done): reward=total_time
            if done and total_time>scoreavg/(e+1):
                reward=total_time
            #if done: [lbrd,reward]=leaderboardcompare(lbrd,score-1)  
            cumulativereward+=reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                scoreavg+=total_time
                print("episode: {}/{}, time: {}, e: {:.2}, score: {}, score avg: {:.4}, rew. avg, {:.4}"
                      .format(e, EPISODES, time, agent.epsilon, total_time, scoreavg/(e+1),cumulativereward/(e+1)))
                learning1.append([scoreavg/(e+1)])
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        #if e % 250 ==0:
            #lbrd=[0,0,0,0,0]
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
    import matplotlib.pyplot as plt 
    plt.figure(figsize=(20,10))
    plt.plot(learning1[100:], '-')
    plt.ylabel('avg reward')
    plt.xlabel('episodes')
    plt.show()