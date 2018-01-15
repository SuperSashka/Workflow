 # -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:06:51 2017

@author: user
"""

# -*- coding: utf-8 -*-
import numpy as np
import enviroment as wf
from datetime import datetime




EPISODES = 10000

if __name__ == "__main__":
    cumulativereward=0
    learningRAND=[]  
    dta_start=datetime.now()  
    for e in range(EPISODES):
        comptime=wf.compgen(60,5)
        tree=wf.treegen(60)
        wfl=wf.workflow(tree,comptime)
        done=wfl.completed
        random_dist=np.random.random(300)
        for time in range(60):
#            wfl2=wf.wfl_copy(wfl)
#            for time1 in range(time,5):
#                done1=wfl2.completed
#                va=wfl2.valid_actions_mask()
#                action_dist = np.random.random(15)*va
#                action=(action_dist.tolist()).index(max(action_dist))
#                pr_reward1,_=wfl2.act(action)
#                done1=wfl2.completed
#                if done1: break
            #if time>0: print(pr_reward1-pr_reward2)
            #va=wfl.val_mask
            #print(va)
            #action_dist =random_dist *va
            #print(va[action])
            #action=(action_dist.tolist()).index(max(action_dist))
            #print(action)
            action=np.random.randint(60*5)
            done=wfl.isvalid(wfl.actions[action][0],1)
            while not(done):
                action=np.random.randint(60*5)
                done=wfl.isvalid(wfl.actions[action][0],1)
                #print(action,done)
                
            score,_=wfl.act(action)
            done=wfl.completed
            #score+=reward
            #reward=0
            #if done: [lbrd,reward]=leaderboardcompare(lbrd,score-1)  
#            pr_reward2=pr_reward1
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
    dta_end=datetime.now()
    print(str(dta_end-dta_start))
    import matplotlib.pyplot as plt 
    import matplotlib
    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,5))
    plt.plot(learningRAND[100:], '-')
    plt.ylabel('avg reward')
    plt.xlabel('episodes')
    plt.show()