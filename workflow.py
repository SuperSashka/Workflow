# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:58:10 2017
@author: user
"""
import numpy as np
import tensorflow as tf
import workflowenv as wf
import actor


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    learning61=[] 
    time61=[]
    task_par=60
    proc_par=5
    state_size = 2075
    action_size = task_par*proc_par
    agent =actor.DQNAgent(state_size, action_size)
    #agent.load("model61.h5")
    done = False
    batch_size = 256
    cumulativereward=0
    scoreavg=0
    timeavg=0
    EPISODES=100000
    loss=0

    for e in range(EPISODES):
        comptime=wf.compgen(task_par,proc_par)
        chns=wf.treegen(task_par)
        wfl=wf.workflow(chns,comptime)
        done=wfl.completed
        state = wfl.state
        state = np.reshape(state, [1, state_size])
        for time in range(5000):
            mask=wfl.get_mask()
            #mask="none"
            action = agent.act(state,mask)
            total_time,_=wfl.act(action,"mask")
            #total_time,_=wfl.act(action,"random")
            next_state = wfl.state
            done=wfl.completed
            reward=total_time
            cumulativereward+=reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember((state, action, reward, next_state, done))
            state = next_state
            if done:
                scoreavg+=total_time
                timeavg+=time
                neps=e+1
                #print("episode: {}/{}, time: {}, time avg: {:.4}, score: {}, score avg: {:.4}, rew. avg, {:.4}"
                 #     .format(e, EPISODES, time+1, timeavg/neps, total_time, scoreavg/neps,cumulativereward/neps))
                print("episode: {}/{}, score: {}, score avg: {:.4}".format(e, EPISODES, total_time, scoreavg/neps))
                learning61.append([scoreavg/neps])
                time61.append([timeavg/neps])
                break
        if len(agent.D) > batch_size:
            loss+=agent.replay(batch_size,e)
    import matplotlib.pyplot as plt 
    plt.figure(figsize=(10,5))
    plt.plot(learning61[100:], '-')
    plt.ylabel('avg reward')
    plt.xlabel('episodes')
    plt.show()
