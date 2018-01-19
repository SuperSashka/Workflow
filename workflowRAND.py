# -*- coding: utf-8 -*-
"""
Файл с закоментированными отличиями с обычным workflow которым генерирует
Расписания рандомно
"""
import numpy as np
#import tensorflow as tf
import workflowenv as wf
#import actor


if __name__ == "__main__":
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)
    #from keras import backend as K
    #K.set_session(sess)
    learningRAND=[] 
    timeRAND=[]
    task_par=60
    proc_par=5
    #state_size = 78
    action_size = task_par*proc_par
    #agent =actor.DQNAgent(state_size, action_size)
    #agent.load("model61.h5")
    done = False
    #batch_size = 256
    #cumulativereward=0
    scoreavg=0
    timeavg=0
    EPISODES=100000
    #loss=0

    for e in range(EPISODES):
        comptime=wf.compgen(task_par,proc_par)
        chns=wf.treegen(task_par)
        wfl=wf.workflow(chns,comptime)
        done=wfl.completed
        #state = wfl.state
        #state = np.reshape(state, [1, state_size])
        for time in range(5000):
            #mask=wfl.get_mask()
            #mask="none"
            action = np.random.randint(action_size)
            #total_time,_=wfl.act(action,"mask")
            total_time,_=wfl.act(action,"random")
            #next_state = wfl.state
            done=wfl.completed
            #reward=total_time
            #cumulativereward+=reward
            #next_state = np.reshape(next_state, [1, state_size])
            #agent.remember((state, action, reward, next_state, done))
            #state = next_state
            if done:
                scoreavg+=total_time
                timeavg+=time
                neps=e+1
                print("episode: {}/{}, time: {}, time avg: {:.4}, score: {}, score avg: {:.4}"
                      .format(e, EPISODES, time+1, timeavg/neps, total_time, scoreavg/neps))
                #print("episode: {}/{}, score: {}, score avg: {:.4}".format(e, EPISODES, total_time, scoreavg/neps))
                learningRAND.append([scoreavg/neps])
                timeRAND.append([timeavg/neps])
                break
        #if len(agent.D) > batch_size:
            #loss+=agent.replay(batch_size,e)
    import matplotlib.pyplot as plt 
    plt.figure(figsize=(10,5))
    plt.plot(learningRAND[100:], '-')
    plt.ylabel('avg reward')
    plt.xlabel('episodes')
    plt.show()
