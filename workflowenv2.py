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
            
def treegen(tree_length):
    tree=np.zeros((5,5),dtype=np.int)
    for j in range(len(tree)): tree[j,(j+1):len(tree)]=np.random.randint(0,2,len(tree[j,(j+1):len(tree)]))
    tree=tree-np.transpose(tree)
    return tree;

def uppertriangle(tree):
    upper=[]
    for i in range(len(tree)):
        for j in range(i+1,len(tree)):
            upper.append(tree[i,j])
    return upper;
            

class workflow:
    def __init__(self, tree,comp_times):
        self.tree = tree
        self.comp_times = comp_times
        self.shdl=[]
        self.current_time=0
        #for terminal state we need to know is scheduling is completed
        self.completed=False
        #number of chains of tasks
        self.ntasks=len(self.tree)
        #number of processors
        self.nprocessors=len(comp_times)
        #sum of all worst case chain times, i.e. 'maximal worst case' shedule length
        self.maxlength=0
        for i in range(self.ntasks): self.maxlength+=np.amax(comp_times[:,i])
        self.load=np.zeros(self.nprocessors)
        #for DQN we need to have all actions as a number, every action here is (i,j), shedule first task in chain
        #j on i-th processor
        self.actions=[[-1,-1]]
        for i in range(self.ntasks):
            for j in range(self.nprocessors):
                self.actions.append([i,j])
        self.scheduled=[]
        self.state=list(flatten([uppertriangle(self.tree),self.comp_times/self.maxlength,self.load/self.maxlength,self.ifscheduled(),self.ifbusy()]))
    
    
    def ifscheduled(self):
        ifshdl=np.zeros(self.ntasks)
        for task in self.scheduled:
            ifshdl[task]=int(1)
        return ifshdl
    
    def prequesites(self,task):
        preq=[]
        for i in range(len(self.tree)): 
            if self.tree[task,i]==-1: preq.append(i)
        return(preq)

    def AFT(self,shdl):
        comptime=self.comp_times;
        #maxtime=self.schedule_length(shdl);
        maxtime=100500
        aft=[maxtime+1,maxtime+1,maxtime+1,maxtime+1,maxtime+1];
        for item in shdl:
            aft[item[1]]=(comptime[item[0],item[1]]+item[2])
        return aft;

    def AST(self,shdl):
        #maxtime=self.schedule_length(shdl);
        maxtime=100500
        ast=[maxtime+1,maxtime+1,maxtime+1,maxtime+1,maxtime+1];
        for item in shdl:
            ast[item[1]]=item[2]
        return ast;

    def violation(self,task,current_time):
        vltn=False
        aft=self.AFT(self.shdl);
        preq=self.prequesites(task)
        for item in preq: 
            if aft[item]>current_time: 
                vltn=True
                #print('Prequesites are not yet computed')
        return vltn;
    
    def schedule_length(self,shdl):
        prload=[0,0,0];
        for item in shdl:
            if prload[item[0]]<(self.comp_times[item[0],item[1]]+item[2]): 
                prload[item[0]]=(self.comp_times[item[0],item[1]]+item[2])
        shdl_length=np.amax(prload)
        return shdl_length
    
    def processor_load(self, time):
        load=[0,0,0];
        for item in self.shdl:
            if load[item[0]]<(self.comp_times[item[0],item[1]]+item[2]): 
                load[item[0]]=(self.comp_times[item[0],item[1]]+item[2])
        for i in range(len(load)): load[i]=load[i]-time
        return load
    
    
    def ifbusy(self):
        ifload=np.zeros(self.nprocessors)
        load=self.processor_load(self.current_time)
        for i in range(self.nprocessors):
            if load[i]>0: ifload[i]=int(1)
        return ifload
    # shedule first task in chain j on i-th processor
    def schedule_task(self, nproc,ntsk,time):
        reward=0
        pr_load=self.processor_load(time)
        if ntsk in self.scheduled:
            #print('Process is already sheduled')
            reward=-1
        else:
            if self.violation(ntsk,time):
                #print('Prequesites are not yet computed and task cannot be scheduled')
                reward=-1
            else:
                    self.scheduled.append(ntsk)
                    self.shdl.append([nproc,ntsk,time+pr_load[nproc]])
                    self.load=self.processor_load(self.current_time)
                    self.state=list(flatten([uppertriangle(self.tree),self.comp_times/self.maxlength,self.load/self.maxlength,self.ifscheduled(),self.ifbusy()]))
                    reward=0
        if len(self.scheduled)==self.ntasks:
            self.completed=True
            reward=self.maxlength-self.schedule_length(self.shdl)
        return reward,self.state;
    
    def act(self, action): 
        if action==0: 
            self.current_time+=1
            reward=-1
            state=self.state;
        if action>0: reward,state=self.schedule_task(self.actions[action][1],self.actions[action][0],self.current_time)
        return reward,state;

