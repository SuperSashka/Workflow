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
    tree=np.zeros((tree_length,tree_length),dtype=np.int)
    for j in range(len(tree)): tree[j,(j+1):len(tree)]=np.random.randint(0,2,len(tree[j,(j+1):len(tree)]))
    tree=tree-np.transpose(tree)
    return tree;

def compgen(ntask,nprocessors):
    c_gen=np.zeros((nprocessors,ntask))
    mainline=np.random.randint(20, size=(1, ntask))+1
    c_gen[0,:]=mainline
    for i in range(1,nprocessors):
        c_gen[i,:]=np.random.uniform(low=0.5, high=2.0)*mainline
    c_gen=np.int64(c_gen)
    return c_gen

def uppertriangle(tree):
    upper=[]
    for i in range(len(tree)):
        for j in range(i+1,len(tree)):
            upper.append(tree[i,j])
    upper=np.asarray(upper)
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
        self.maxlength=np.amax(self.max_comp_length())
        #self.maxlength=np.int(100)
        self.load=np.zeros(self.nprocessors)
        #for DQN we need to have all actions as a number, every action here is (i,j), shedule first task in chain
        #j on i-th processor
        self.actions=[]
        for i in range(self.ntasks):
            for j in range(self.nprocessors):
                self.actions.append([i,j])
        self.scheduled=[]
        #self.state=list(flatten([uppertriangle(self.tree),self.comp_times/self.maxlength,self.load/self.maxlength,self.ifscheduled(),self.schedule_length(self.shdl)/self.maxlength,self.npreqs_notcomputed()]))
        #self.state=list(flatten([uppertriangle(self.tree),self.comp_times/self.maxlength,self.load/self.maxlength]))
        self.val_mask=self.valid_actions_update()
        #self.state=list(flatten([uppertriangle(self.tree),self.comp_times/self.maxlength,self.load/self.maxlength,self.ifscheduled(),self.schedule_length(self.shdl)/self.maxlength,self.npreqs_notcomputed(),self.val_mask]))
        self.state=list(flatten([uppertriangle(self.tree),self.comp_times/self.maxlength,self.load/self.maxlength,self.val_mask]))
        
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
        
    
    def process_chain(self,task):
        subtasks=[task]
        if self.prequesites(task)!=[]:
            for subtask in self.prequesites(task):
                if subtask not in subtasks:
                    subtasks.append(subtask)
            if self.prequesites(subtask)!=[]: subtasks.append(self.process_chain(subtask))
        subtasks=list(flatten(subtasks))
        subtasks=list(set(subtasks))
        return subtasks
    
    def max_comp_length(self):
        max_length=np.zeros(self.ntasks)
        for i in range(self.ntasks): 
            for task in self.process_chain(i):
                max_length[i]+=int(np.amax(self.comp_times[:,task]))
        return max_length
            

    def AFT(self,shdl):
        comptime=self.comp_times;
        #maxtime=self.schedule_length(shdl);
        maxtime=self.maxlength
        aft=[maxtime+1,maxtime+1,maxtime+1,maxtime+1,maxtime+1];
        for item in shdl:
            aft[item[1]]=(comptime[item[0],item[1]]+item[2])
        return aft;

    def AST(self,shdl):
        #maxtime=self.schedule_length(shdl);
        maxtime=self.maxlength
        ast=[maxtime+1,maxtime+1,maxtime+1,maxtime+1,maxtime+1];
        for item in shdl:
            ast[item[1]]=item[2]
        return ast;

    def violation(self,task,current_time):
        vltn=False
        preq=self.prequesites(task)
        for item in preq: 
            if item not in self.scheduled: 
                vltn=True
                #print('Prequesites are not yet computed')
        return vltn;
    
    def schedule_length(self,shdl):
        prload=np.zeros(self.nprocessors);
        for item in shdl:
            if prload[item[0]]<(self.comp_times[item[0],item[1]]+item[2]): 
                prload[item[0]]=(self.comp_times[item[0],item[1]]+item[2])
        shdl_length=np.amax(prload)
        return shdl_length
    
    def processor_load(self, time):
        load=np.zeros(self.nprocessors);
        for item in self.shdl:
            if load[item[0]]<(self.comp_times[item[0],item[1]]+item[2]): 
                load[item[0]]=(self.comp_times[item[0],item[1]]+item[2])
        #for i in range(len(load)): load[i]=load[i]-time
        return load
    
    def processor_time(self):
        time=np.zeros(self.nprocessors);
        for item in self.shdl:
            time[item[0]]+=(self.comp_times[item[0],item[1]]) 
        return time
    
    
    
    def npreqs_notcomputed(self):
        notcomp=np.zeros(self.ntasks)
        for i in range(self.ntasks):
            for task in self.prequesites(i):
                if task not in self.scheduled:
                    notcomp[i]+=1
        notcomp=notcomp/self.ntasks
        return notcomp;

    def time_to_start(self):
        TTS=np.zeros(self.ntasks)
        for i in range(self.ntasks):
            for task in self.prequesites(i):
                if task not in self.scheduled:
                    TTS[i]+=np.amin(self.comp_times[:,task])
        TTS=TTS/self.ntasks
        return TTS; 
       
    
    def ifbusy(self):
        ifload=np.zeros(self.nprocessors)
        load=self.processor_load(self.current_time)
        for i in range(self.nprocessors):
            if load[i]>0: ifload[i]=int(1)
        return ifload
    
    def isvalid(self,ntsk,nproc):
        vld=True
        if (ntsk in self.scheduled) or (self.violation(ntsk,nproc)): vld=False
        return vld
    
    def valid_actions_update(self):
        valid_mask=np.zeros(len(self.actions),dtype=int)
        for action in range(len(self.actions)):
            if self.isvalid(self.actions[action][0],self.actions[action][1]): valid_mask[action]=1
        return valid_mask

    def schedule_task(self, ntsk,nproc):
        reward=0
        pr_load=self.processor_time()
        #if not(self.isvalid(ntsk,nproc)):
            #print('Process is already sheduled')
        #   reward=-0.2
        #else:
        self.scheduled.append(ntsk)
        self.shdl.append([nproc,ntsk,pr_load[nproc]])
        self.load=self.processor_time()
        self.val_mask=self.valid_actions_update()
        self.state=list(flatten([uppertriangle(self.tree),self.comp_times/self.maxlength,self.load/self.maxlength,self.val_mask]))
        if len(self.scheduled)==self.ntasks:
            self.completed=True
            reward=self.maxlength-self.schedule_length(self.shdl)
        return reward,self.state;
  
    def act(self, action): 
        reward,state=self.schedule_task(self.actions[action][0],self.actions[action][1])
        return reward,state;
    

