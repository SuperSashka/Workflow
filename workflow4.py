# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:55:05 2017

@author: user
"""

# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import workflowenv2 as wf



EPISODES = 60000

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
        self.gamma = 1.0   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, s_t, a_t, r_t,s_t1,a_t1,r_t1, done):
        self.memory.append((s_t, a_t, r_t, s_t1,a_t1,r_t1, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for s_t, a_t, r_t, s_t1,a_t1,r_t1, done in minibatch:
            target = self.model.predict(s_t1)
            if done:
                target[0][a_t1] = r_t1
            else:
                a = self.model.predict(s_t1)[0]
                t = self.target_model.predict(s_t1)[0]
                target[0][a_t] = r_t+self.gamma*r_t1 + (self.gamma)**2 * t[np.argmax(a)]
            self.model.fit(s_t, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    state_size = 39
    action_size = 15
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 256
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
        s_t1 = np.reshape(state, [1, state_size])
        s_t=0
        a_t=0
        r_t=0
        for time in range(5000):
            a_t1 = agent.act(s_t1)
            total_time,_=wfl.act(a_t1)
            s_t1 = wfl.state
            done=wfl.completed
            #score+=reward
            #reward=0
            #if not(done): reward=total_time
            #if done and total_time>scoreavg/(e+1):
                #reward=total_time
            r_t1=total_time
            #if done: [lbrd,reward]=leaderboardcompare(lbrd,score-1)  
            cumulativereward+=r_t1
            s_t1 = np.reshape(s_t1, [1, state_size])
            if time>1: 
                agent.remember(s_t,a_t,r_t,s_t1,a_t1,r_t1, done)
            s_t=s_t1
            a_t=a_t1
            r_t=r_t1
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