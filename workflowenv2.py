# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:58:10 2017

@author: user
"""
import numpy as np
from collections import Iterable

def flatten(items):
    """Yield items from any nested iterable; see REF."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x
            
def chgen(ntask):
    import random
    chn=[[],[],[],[],[]]
    for i in range(ntask):
        chn[np.random.randint(5)].append(i)
    chn.sort()
    while chn[0]==[]: del chn[0]
    for chain in chn : random.shuffle(chain)
    return chn;


class workflow:
    def __init__(self, tasks,comp_times):
        #tasks in chains form [[first,second,...],[first,second,...],...[first,...,last]]
        #example [[0,4],[2,3,1]], i.e. 0 should be completed before 4 and 2 before 3 before 1
        self.tsk = tasks
        #initial chain memory and some werid magic blocks to memorize it
        self.tsk_mem=[]
        for i in range(len(self.tsk)): self.tsk_mem.append([])
        for i in range(len(self.tsk)):
            for j in range(len(self.tsk[i])):
                self.tsk_mem[i].append(self.tsk[i][j])
        #matrix (i,j), i=#procs,j=#task
        self.comp_times = comp_times
        #shedule has form [[processor,task],...[processor,task]]
        self.shdl=[]
        #for terminal state we need to know is scheduling is completed
        self.completed=False
        #number of chains of tasks
        self.nchains=len(self.tsk)
        #number of processors
        self.nprocs=len(comp_times)
        #number of 'worst case' computational time for every chain
        self.chainslength=self.chain_length_min(self.tsk)
        #minimal time of computing of first process in every chain
        self.chainfirst=self.chain_proc(self.tsk)
        #sum of all worst case chain times, i.e. 'maximal worst case' shedule length
        self.maxlength=np.sum(self.chain_length_max(self.tsk))
        self.load=np.zeros(3)
        #for DQN we need to have all actions as a number, every action here is (i,j), shedule first task in chain
        #j on i-th processor
        self.actions=[]
        for i in range(5):
            for j in range(3):
                self.actions.append([i,j])
        #for reward we need 'delta' time between two consequent sheduling        
        self.time1=self.shedule_length(self.shdl)
        self.time2=self.shedule_length(self.shdl)
        self.astarttime=self.AST(self.shdl,comp_times)
        self.afinishtime=self.AFT(self.shdl,comp_times)
        self.state=list(flatten([self.chainslength/self.maxlength,self.chainfirst/self.maxlength,self.load/self.maxlength]))
        self.state2=list(flatten([self.chainslength/self.maxlength,self.chainfirst/self.maxlength,self.load/self.maxlength,self.astarttime/self.maxlength,self.afinishtime/self.maxlength]))
        self.state3=list(flatten([self.astarttime/self.maxlength,self.afinishtime/self.maxlength]))
    #schedule (possibly incomplete) computation time    
    def shedule_length(self,shdl):
        length=np.zeros(3);
        for item in shdl:
            length[item[0]]+=self.comp_times[item[0],item[1]]
        totallength=np.sum(length)
        return totallength;
    
    #'worst case' computational time for every chain
    def chain_length_max(self,chns):
        clength=np.zeros(5)
        curchain=0;
        for chain in chns:
            for link in chain:
                clength[curchain]+=np.amax(self.comp_times[:,link])
            curchain+=1;
        return clength;
    
    #'best case' computational time for every chain
    def chain_length_min(self,chns):
        clength=np.zeros(5)
        curchain=0;
        for chain in chns:
            for link in chain:
                clength[curchain]+=np.amin(self.comp_times[:,link])
            curchain+=1;
        return clength;
    
    #minimal time of computing of first process in every chain on every processor
    def chain_proc(self,chns):
        cmin=np.zeros(5*3)
        curchain=0;
        for chain in chns:
            if chain!=[]:
                firstlink=chain[0]
                for proc in range(3): cmin[3*curchain+proc]=self.comp_times[proc,firstlink]
            else:
                cmin[curchain]=0
            curchain+=1;
        return cmin;
    
    def actual_proc_load(self, shdl):
        load=np.zeros(3);
        for item in shdl:
            load[item[0]]+=self.comp_times[item[0],item[1]]
        return load;
    
    def AFT(self,shdl,comptime):
        aft=[comptime.max()+1,comptime.max()+1,comptime.max()+1,comptime.max()+1,comptime.max()+1];
        prload=[0,0,0];
        for item in shdl:
            prload[item[0]]+=comptime[item[0],item[1]]
            aft[item[1]]=prload[item[0]]
        return aft;

    def AST(self,shdl,comptime):
        ast=[comptime.max()+1,comptime.max()+1,comptime.max()+1,comptime.max()+1,comptime.max()+1];
        prload=[0,0,0];
        for item in shdl:
            ast[item[1]]=0
            ast[item[1]]+=prload[item[0]]
            prload[item[0]]+=comptime[item[0],item[1]]
        return ast;
        
    def AFT_chain(self,tsks,aft):
        aft_chain=[]
        for i in range(len(tsks)): aft_chain.append([])
        for i in range(len(tsks)):
            for j in range(len(tsks[i])):
                aft_chain[i].append(aft[tsks[i][j]])
        return aft_chain;

    def AST_chain(self,tsks,ast):
        ast_chain=[]
        for i in range(len(tsks)): ast_chain.append([])
        for i in range(len(tsks)):
            for j in range(len(tsks[i])):
                ast_chain[i].append(ast[tsks[i][j]])
        return ast_chain;

    def order_violation(self,ast_chain,aft_chain):
        violation=False
        for i in range(len(ast_chain)):
            if len(ast_chain[i])>1:
                for j in range(len(ast_chain[i])-1):
                    if ast_chain[i][j+1]<aft_chain[i][j]:
                        print('oder violation, chain {} element {} starts before {} ends'.format(i+1,j+2,j+1))
                        violation=True
        return violation;
    
    
    
    # shedule first task in chain j on i-th processor
    def schedule_task(self, nproc, nchain):
        reward=0
        #if all chains are empty - scheduling is completed
        if self.tsk!=[] and self.tsk!=[[],[]] and self.tsk!=[[],[],[]] and self.tsk!=[[],[],[],[]] and self.tsk!=[[],[],[],[],[]]:    
            #checking if action is valid and we can shedule task
            if nchain<self.nchains and nproc<self.nprocs and not(self.completed) :
                #if current chain is not empty
                if self.tsk[nchain] != []:
                    #length before
                    self.time1=self.shedule_length(self.shdl)
                    #schedule
                    self.shdl.append([nproc,self.tsk[nchain][0]])
                    #delete task scheduled
                    del self.tsk[nchain][0]
                    #print('Shedule:', self.shdl)
                    #print('Task queue:', self.tsk)
                    #time after
                    self.time2=self.shedule_length(self.shdl)
                    #print('Current length:', self.shedule_length(self.shdl),'delta',self.time2-self.time1
                     #     ,'Max length:',self.maxlength)
                    #less delta - more reward, actually here we want 'negative reward', so less actions is more rewarding
                    #print('Reward',(self.time1-self.time2)/self.maxlength)
                    #reward=(self.time2-self.time1);
                    self.nchains=len(self.tsk)
                    self.chainslength=self.chain_length_min(self.tsk)
                    self.chainfirst=self.chain_proc(self.tsk)
                    self.load=self.actual_proc_load(self.shdl)
                    #print('Chain lengths:', self.chainslength)
                    #print('Chain first item min tme:', self.chainfirst)
                    self.state=list(flatten([self.chainslength/self.maxlength,self.chainfirst/self.maxlength,self.load/self.maxlength]))
                    self.astarttime=self.AST(self.shdl,self.comp_times)
                    self.afinishtime=self.AFT(self.shdl,self.comp_times)
                    self.state2=list(flatten([self.chainslength/self.maxlength,self.chainfirst/self.maxlength,self.load/self.maxlength,self.astarttime/self.maxlength,self.afinishtime/self.maxlength]))
                    self.state3=list(flatten([self.astarttime/self.maxlength,self.afinishtime/self.maxlength]))
                else:
                    #if current chain is empty we want to delete it
                    #print('Chain', nchain, 'is empty, deleting')
                    del self.tsk[nchain]
                    self.nchains=len(self.tsk)
                    #reward=0
            else:
                #print('Restricted action, nothing changed')
                #print('Reward',-0.01)
                reward=-1
        else:
            #print('Scheduling completed, total length is', self.shedule_length(self.shdl), 'out of max possible', self.maxlength )
            #print('Reward',self.maxlength/self.shedule_length(self.shdl))
            reward=self.maxlength-self.shedule_length(self.shdl)
            #print('with schedule', self.shdl )
            self.completed=True
        return reward,self.state;
    
    def act(self, action): 
        reward,state=self.schedule_task(self.actions[action][1],self.actions[action][0])
        return reward,state;

