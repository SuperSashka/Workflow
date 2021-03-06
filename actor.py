# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:58:10 2017
@author: user
"""

from __future__ import print_function


import random
from collections import deque
import numpy as np
import json
from keras import initializers
#from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.STATE=state_size
        self.ACTIONS = action_size # number of  actions
        self.GAMMA = 0.99 # decay rate of past observations
        self.OBSERVATION = 500. # timesteps to observe before training
        self.EXPLORE = 20000. # frames over which to anneal epsilon
        #как правило мы используем e-greedy policy и хотим, чтобы в начале действия были
        #более рандомны, чем в начале, поэтому мы снижаем e от INITIAL до FINAL за EXPLORE шагов
        self.FINAL_EPSILON = 0.01 # final value of epsilon
        self.INITIAL_EPSILON = 0.02 # starting value of epsilon
        self.REPLAY_MEMORY = 50000 # number of previous transitions to remember
        self.LEARNING_RATE = 1e-4
        self.D = deque(maxlen=self.REPLAY_MEMORY)
        self.model=self.buildmodel()
        self.epsilon=self.INITIAL_EPSILON

    #структура сети
    def buildmodel(self):
        model = Sequential()
        """
        model.add(Dense(2048, input_dim=self.STATE, activation='relu',kernel_initializer=initializers.RandomNormal(stddev=0.001)))
        model.add(Dense(2048, activation='relu',kernel_initializer=initializers.RandomNormal(stddev=0.1)))
        model.add(Dense(self.ACTIONS, activation='relu',kernel_initializer=initializers.RandomNormal(stddev=0.1)))
        """
        model.add(Dense(2048, input_dim=self.STATE, activation='relu'))
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(self.ACTIONS, activation='relu',kernel_initializer=initializers.RandomUniform(minval=0, maxval=0.1, seed=None)))        
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        return model
    
    #дека для запоминания действий для обучения
    def remember(self,SARSA):
        self.D.append(SARSA)
        
    def act(self,state,mask):
        #choose an action epsilon greedy
        if random.random() <= self.epsilon:
                #print("----------Random Action----------")
                q = np.random.random(self.ACTIONS)
                if mask is not 'none': q*=mask
                action_index = np.argmax(q)
        else:
                q = self.model.predict(state)
                if mask is not 'none': q*=mask
                action_index = np.argmax(q)
        return action_index
    
    #для обучения используется SARSA
    def replay(self,batch_size,ep):
        model_loss=0
        q_loss=0
        #observation используется для того чтобы заполнить память рандомными действиями
        if ep>self.OBSERVATION:
            #выбираем batch_size действий из памяти
            minibatch = random.sample(self.D, batch_size)
            #мы оптимизируем функцию Q(s,a), соотвественно - вход сети s, выход a
            inputs = np.zeros((len(minibatch), self.STATE))  
            targets = np.zeros((inputs.shape[0], self.ACTIONS))
            #преобразуем каждую строку из памяти, чтобы использовать sarsa                     
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
                Q_sc=self.model.predict(state_t)
                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    #Q(s,a) это предположительная награда, если мы перейдём из состояния s в s_1
                    #c помощью дейтсивя a, поэтому мы меняем Q по алгоритму SARSA для действий из
                    #памяти
                    targets[i, action_t] = reward_t + self.GAMMA * np.max(Q_sa)
                q_loss+=(Q_sc[0,action_t]-targets[i, action_t])**2
            #обучаем сеть
            model_loss = self.model.train_on_batch(inputs, targets)
            if ep < self.EXPLORE:
                #print('Decrease epsilon')
                self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE 
            if ep==self.EXPLORE: self.epsilon=self.FINAL_EPSILON
        return model_loss,q_loss
    
    #функции для загрузки/сохранения весов
    def save(self, name):
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