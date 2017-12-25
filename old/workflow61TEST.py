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
from keras.optimizers import SGD , Adam
import tensorflow as tf
import workflowenv3 as wf




class DQNAgent:
    def __init__(self, state_size, action_size):
        self.STATE=state_size
        self.ACTIONS = action_size # number of valid actions
        self.GAMMA = 0.99 # decay rate of past observations
        self.OBSERVATION = 10. # timesteps to observe before training
        self.EXPLORE = 80000. # frames over which to anneal epsilon
        self.FINAL_EPSILON = 0.01 # final value of epsilon
        self.INITIAL_EPSILON = 0.1 # starting value of epsilon
        self.REPLAY_MEMORY = 50000 # number of previous transitions to remember
        self.FRAME_PER_ACTION = 1
        self.LEARNING_RATE = 1e-4
        self.D = deque(maxlen=self.REPLAY_MEMORY)
        self.model=self.buildmodel()
        self.epsilon=0.1*0.25

    def buildmodel(self):
        model = Sequential()
        model.add(Dense(1024, input_dim=self.STATE, activation='relu'))
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
            inputs = np.zeros((len(minibatch), self.STATE))  
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
    
    def save(self,name):
        print("Now we save model")
        self.model.save_weights(name, overwrite=True)
        with open("model.json", "w") as outfile:
                json.dump(self.model.to_json(), outfile)
    def load(self,name):
        print ("Now we load weight")
        self.model.load_weights(name)
        adam = Adam(lr=self.LEARNING_RATE)
        self.model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")   


    
if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    # open up a game state to communicate with emulator
    state_size = 99
    action_size = 30
    time_delay=3500
    agent = DQNAgent(state_size, action_size)
    agent.load('model61t99999.h5')
    done = False

    comptimeTEST=np.asarray([[ 9, 12,  7,  4,  4,  5, 19,  7, 16, 19],
       [12, 11,  4,  9, 17, 14, 12, 11,  5, 12],
       [12, 10,  2, 18, 19, 13, 18, 19, 15, 18]])
    chnsTEST=np.asarray([[ 0,  1,  1,  1,  0,  0,  0,  1,  1,  1],
       [-1,  0,  0,  1,  0,  0,  1,  0,  0,  0],
       [-1,  0,  0,  1,  0,  0,  1,  1,  1,  0],
       [-1, -1, -1,  0,  1,  1,  1,  0,  1,  0],
       [ 0,  0,  0, -1,  0,  0,  1,  0,  0,  0],
       [ 0,  0,  0, -1,  0,  0,  1,  0,  0,  1],
       [ 0, -1, -1, -1, -1, -1,  0,  1,  0,  1],
       [-1,  0, -1,  0,  0,  0, -1,  0,  0,  1],
       [-1,  0, -1, -1,  0,  0,  0,  0,  0,  1],
       [-1,  0,  0,  0,  0, -1, -1, -1, -1,  0]])
    wfl=wf.workflow(chnsTEST,comptimeTEST)
    done=wfl.completed
    state = wfl.state
    state = np.reshape(state, [1, state_size])
    for time in range(5000):
        _,action = agent.act(state,100500)
        total_time,_=wfl.act_mode(action,'score')
        next_state = wfl.state
        done=wfl.completed
        if done:
             print(" time: {}, score: {}".format(time,total_time))
             break
    Sconv=[[],[],[]]
    for i in range(len(wfl.shdl)):
        Sconv[wfl.shdl[i][0]].append([wfl.shdl[i][1],wfl.shdl[i][2],wfl.comp_times[wfl.shdl[i][0]][wfl.shdl[i][1]]])
        

