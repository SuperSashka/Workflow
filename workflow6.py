#!/usr/bin/env python
from __future__ import print_function


import random
import numpy as np
from collections import deque

import json
#from keras import initializations
#from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D,Reshape
from keras.optimizers import SGD , Adam
import tensorflow as tf
import workflowenv3 as wf




class DQNAgent:
    def __init__(self, state_size, action_size):
        self.STATE=state_size
        self.ACTIONS = action_size # number of valid actions
        self.GAMMA = 0.99 # decay rate of past observations
        self.OBSERVATION = 20. # timesteps to observe before training
        self.EXPLORE = 80000. # frames over which to anneal epsilon
        self.FINAL_EPSILON = 0.01 # final value of epsilon
        self.INITIAL_EPSILON = 0.1 # starting value of epsilon
        self.REPLAY_MEMORY = 50000 # number of previous transitions to remember
        self.FRAME_PER_ACTION = 1
        self.LEARNING_RATE = 1e-4
        self.D = deque(maxlen=self.REPLAY_MEMORY)
        self.model=self.buildmodel()
        self.epsilon=self.INITIAL_EPSILON

    def buildmodel(self):
        model = Sequential()
        model.add(Dense(1024, input_dim=self.STATE, activation='relu'))
#        model.add(Reshape((32, 32, 1)))
#        model.add(Conv2D(10, (3, 3), activation='relu', input_shape=(32, 32, 1)))
#        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(self.ACTIONS, activation='linear'))
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        return model
    
    
    def remember(self,SARSA):
        self.D.append(SARSA)
        
    def act(self,state, time):
        a_t = np.zeros([self.ACTIONS])
        #choose an action epsilon greedy
        if random.random() <= self.epsilon:
                #print("----------Random Action----------")
                action_index = random.randrange(self.ACTIONS)
                a_t[action_index] = 1
        else:
                q = self.model.predict(state)       #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1 
        return a_t,action_index
    
    def replay(self,batch_size,ep):
        model_loss=0
        if ep>self.OBSERVATION:
            #print('Explore')
            minibatch = random.sample(self.D, batch_size)
            inputs = np.zeros((batch_size, self.STATE))  
            #print (inputs.shape)
            targets = np.zeros((inputs.shape[0], self.ACTIONS))                     
            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward
                inputs[i:i + 1] = state_t    #I saved down s_t
                targets[i] = self.model.predict(state_t)  # Hitting each buttom probability
                Q_sa = self.model.predict(state_t1)
                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + self.GAMMA * np.max(Q_sa)
            # targets2 = normalize(targets)
            model_loss = self.model.train_on_batch(inputs, targets)
            if self.epsilon > self.FINAL_EPSILON:
                #print('Decrease epsilon')
                self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE 
        return model_loss
    
    def save(self):
        print("Now we save model")
        self.model.save_weights("model.h5", overwrite=True)
        with open("model.json", "w") as outfile:
                json.dump(self.model.to_json(), outfile)
    def load(self):
        self.epsilon = self.FINAL_EPSILON 
        print ("Now we load weight")
        self.model.load_weights("model.h5")
        adam = Adam(lr=self.LEARNING_RATE)
        self.model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")   


def playGame():
    # open up a game state to communicate with emulator
    state_size = 121
    action_size = 50
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 256
    cumulativereward=0
    scoreavg=0
    timeavg=0
    EPISODES=500000
    loss=0

    for e in range(EPISODES):
        time_delay=0;
        if e<time_delay: 
            scoreavg=0
            timeavg=0
        comptime=wf.compgen(10,5)
        chns=wf.treegen(10)
        wfl=wf.workflow(chns,comptime)
        done=wfl.completed
        state = wfl.state
        state = np.reshape(state, [1, state_size])
        for time in range(5000):
            _,action = agent.act(state,e)
            if e<time_delay:
                total_time,_=wfl.act_mode(action,'time')
            else:
                total_time,_=wfl.act_mode(action,'score')
            next_state = wfl.state
            done=wfl.completed
            #score+=reward
            #reward=0
            #if not(done): reward=total_time
            #if done and total_time>scoreavg/(e+1):
                #reward=total_time
            reward=total_time
            #if done: [lbrd,reward]=leaderboardcompare(lbrd,score-1)  
            cumulativereward+=reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember((state, action, reward, next_state, done))
            state = next_state
            if done:
                scoreavg+=total_time
                timeavg+=time
                neps=1
                if e>time_delay:neps=e+1-time_delay
                print("episode: {}/{}, time: {}, time avg: {:.4}, score: {}, score avg: {:.4}, rew. avg, {:.4}"
                      .format(e, EPISODES, time, timeavg/neps, total_time, scoreavg/neps,cumulativereward/neps))
                learning1.append([scoreavg/(e+1)])
                time1.append([timeavg/(e+1)])
                break
        if len(agent.D) > batch_size:
            loss+=agent.replay(batch_size,e)


    
if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    learning1=[] 
    time1=[]
    playGame()
    import matplotlib.pyplot as plt 
    plt.figure(figsize=(20,10))
    plt.plot(learning1[100:], '-')
    plt.ylabel('avg reward')
    plt.xlabel('episodes')
    plt.show()

