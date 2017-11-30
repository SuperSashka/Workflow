 # -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:06:51 2017

@author: user
"""

# -*- coding: utf-8 -*-
import numpy as np
import workflowenv4 as wf



EPISODES = 100000

if __name__ == "__main__":
    cumulativereward=0
    learningRAND=[]        
    for e in range(EPISODES):
        comptime=wf.compgen(5,3)
        tree=wf.treegen(5)
        wfl=wf.workflow(tree,comptime)
        done=wfl.completed
        for time in range(5):
            wfl2=wf.wfl_copy(wfl)
            for time1 in range(time,5):
                done1=wfl2.completed
                va=wfl2.valid_actions_mask()
                action_dist = np.random.random(15)*va
                action=(action_dist.tolist()).index(max(action_dist))
                pr_reward1,_=wfl2.act(action)
                done1=wfl2.completed
                if done1: break
            #if time>0: print(pr_reward1-pr_reward2)
            va=wfl.valid_actions_mask()
            #print(va)
            action_dist = np.random.random(15)*va
            #print(va[action])
            action=(action_dist.tolist()).index(max(action_dist))
            score,_=wfl.act(action)
            done=wfl.completed
            #score+=reward
            #reward=0
            #if done: [lbrd,reward]=leaderboardcompare(lbrd,score-1)  
            pr_reward2=pr_reward1
            if done:
                cumulativereward+=score
                print("episode: {}/{}, score: {}, score avg: {:.4}"
                      .format(e, EPISODES, score, cumulativereward/(e+1)))
                learningRAND.append([cumulativereward/(e+1)])
                break
            
        #if e % 250 ==0:
            #lbrd=[0,0,0,0,0]
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
    import matplotlib.pyplot as plt 
    import matplotlib
    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,5))
    plt.plot(learningRAND[100:], '-')
    plt.ylabel('avg reward')
    plt.xlabel('episodes')
    plt.show()