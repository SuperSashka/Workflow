#!/usr/bin/env python
from __future__ import print_function



import numpy as np

import enviroment as wf
import tensorflow as tf
import actor 
    
if __name__ == "__main__":
    taskn=60
    procn=5
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    learning61=[] 
    time61=[]
    # open up a game state to communicate with emulator
    state_size = 2375
    action_size = 300
    agent = actor.DQNAgent(state_size, action_size)
    #agent.load()
    done = False
    batch_size = 32
    cumulativereward=0
    scoreavg=0
    EPISODES=100000
    loss=0

    for e in range(EPISODES):
        comptime=wf.compgen(taskn,procn)
        chns=wf.treegen(taskn)
        wfl=wf.workflow(chns,comptime)
        done=wfl.completed
        state = wfl.state
        state = np.reshape(state, [1, state_size])
        for time in range(5000):
            action = agent.act(state,wfl.val_mask)
            total_time,_=wfl.act(action)
            next_state = wfl.state
            done=wfl.completed
            reward=total_time
            cumulativereward+=reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember((state, action, reward, next_state, done))
            state = next_state
            if done:
                scoreavg+=total_time
                neps=e+1
                print("episode: {}/{}, score: {}, score avg: {:.4}, rew. avg, {:.4}"
                      .format(e, EPISODES, total_time, scoreavg/neps,cumulativereward/neps))
                learning61.append([scoreavg/neps])
                break
        if len(agent.D) > batch_size:
            loss+=agent.replay(batch_size,e)
    import matplotlib.pyplot as plt 
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(20,10))
    plt.plot(learning61[100:], '-')
    plt.ylabel('avg reward')
    plt.xlabel('episodes')
    plt.show()

