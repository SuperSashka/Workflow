 # -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:06:51 2017

@author: user
"""

# -*- coding: utf-8 -*-
import numpy as np
import workflowenv2 as wf
import workflowenv4 as wf4


EPISODES = 100000


if __name__ == "__main__":
    cumulativereward=0
    learningRAND=[]        
    for e in range(EPISODES):
        comptime=wf4.compgen(60,5)
        tree=wf.treegen(60)
        wfl=wf.workflow(tree,comptime)
        done=wfl.completed
        for time in range(5000):
            action = np.random.randint(300)
            reward,_=wfl.act(action)
            done=wfl.completed
            #score+=reward
            #reward=0
            #if done: [lbrd,reward]=leaderboardcompare(lbrd,score-1)  
            if done:
                cumulativereward+=reward
                print("episode: {}/{}, time: {}, score: {}, awerage reward: {:.4}"
                      .format(e, EPISODES, time, reward, cumulativereward/(e+1)))
                learningRAND.append([cumulativereward/(e+1)])
                break
        #if e % 250 ==0:
            #lbrd=[0,0,0,0,0]
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
    import matplotlib.pyplot as plt 
    import matplotlib
    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(figsize=(20,10))
    plt.plot(learningRAND[100:], '-')
    plt.ylabel('avg reward')
    plt.xlabel('episodes')
    plt.show()