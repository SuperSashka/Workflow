# -*- coding: utf-8 -*-
"""
Модуль, который содержит расписание и правила его составления
"""
import numpy as np
from collections import Iterable
from itertools import chain

#переводит двумерные массивы в одномерные
def flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

#генерируем дерево зависимостей задач            
def treegen(tree_length):
    tree=np.zeros((tree_length,tree_length),dtype=np.int)
    #т.к. это орграф без петель, то матрица кососимметричная
    for j in range(len(tree)): tree[j,(j+1):len(tree)]=np.random.randint(0,2,len(tree[j,(j+1):len(tree)]))
    tree=tree-np.transpose(tree)
    return tree;

#генерируем матрицу ресурсов
def compgen(ntask,nprocessors):
    c_gen=np.zeros((nprocessors,ntask))
    mainline=np.random.randint(20, size=(1, ntask))+1
    #генерируем одну строчку (для одного процессора)
    c_gen[0,:]=mainline
    #остальные строчки отличаются лишь на множитель от 0.5 до 2
    for i in range(1,nprocessors):
        c_gen[i,:]=np.random.uniform(low=0.5, high=2.0)*mainline
    c_gen=np.int64(c_gen)
    return c_gen

#т.к. дерево - кососимметричная матрица - нам нужны только значения над диагональю (или под)
#по ним мы полностью можем восстановить дерево    
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
        self.load=np.zeros(self.nprocessors)
        #for DQN we need to have all actions as a number, every action here is (i,j), shedule first task in chain
        #j on i-th processor
        self.actions=[]
        for i in range(self.ntasks):
            for j in range(self.nprocessors):
                self.actions.append([i,j])
        self.scheduled=[]
        #self.state=list(flatten([uppertriangle(self.tree),self.comp_times/self.maxlength,self.load/self.maxlength,self.ifscheduled(),self.schedule_length(self.shdl)/self.maxlength,self.npreqs_notcomputed()]))
        #self.state=list(chain.from_iterable([uppertriangle(self.tree),list(chain.from_iterable(self.comp_times/self.maxlength)),self.load/self.maxlength,self.ifscheduled(),self.schedule_length(self.shdl)/self.maxlength]))
        self.state=list(chain.from_iterable([uppertriangle(self.tree),list(chain.from_iterable((self.comp_times/self.maxlength))),self.load/self.maxlength]))
        #self.state=list(flatten([uppertriangle(self.tree),self.comp_times/self.maxlength,self.load/self.maxlength]))
    
    #выводит вектор распределённых задач, 1-в расписании, 0 - ещё нет    
    def ifscheduled(self):
        ifshdl=np.zeros(self.ntasks)
        for task in self.scheduled:
            ifshdl[task]=int(1)
        return ifshdl
    
    #выводит непосредственных предшественников i-той задачи
    def prequesites(self,task):
        preq=[]
        for i in range(len(self.tree)): 
            if self.tree[task,i]==-1: preq.append(i)
        return(preq)
        
    #выводит полную цепочку задач до i-той, начиная с первой
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
    
    #служит для оценки "наихудшего" расписания
    def max_comp_length(self):
        max_length=np.zeros(self.ntasks)
        for i in range(self.ntasks): 
            for task in self.process_chain(i):
                max_length[i]+=int(np.amax(self.comp_times[:,task]))
        return max_length
            
    #стандартная HEFT метрика (legacy)
    def AFT(self,shdl):
        comptime=self.comp_times;
        #maxtime=self.schedule_length(shdl);
        maxtime=self.maxlength
        aft=[maxtime+1,maxtime+1,maxtime+1,maxtime+1,maxtime+1];
        for item in shdl:
            aft[item[1]]=(comptime[item[0],item[1]]+item[2])
        return aft;
    
    #стандартная HEFT метрика (legacy)
    def AST(self,shdl):
        #maxtime=self.schedule_length(shdl);
        maxtime=self.maxlength
        ast=[maxtime+1,maxtime+1,maxtime+1,maxtime+1,maxtime+1];
        for item in shdl:
            ast[item[1]]=item[2]
        return ast;

    #проверка, распределены ли все предшественники для данной задачи
    def violation(self,task,current_time):
        vltn=False
        preq=self.prequesites(task)
        for item in preq: 
            if item not in self.scheduled: 
                vltn=True
                break
                #print('Prequesites are not yet computed')
        return vltn;
    
    #выводит длину расписания
    def schedule_length(self,shdl):
        prload=np.zeros(self.nprocessors);
        for item in shdl:
            if prload[item[0]]<(self.comp_times[item[0],item[1]]+item[2]): 
                prload[item[0]]=(self.comp_times[item[0],item[1]]+item[2])
        shdl_length=np.amax(prload)
        return shdl_length
    
    #выводит загрузку процессора - время завершения последней распределённой задачи
    def processor_load(self, time):
        load=np.zeros(self.nprocessors);
        for item in self.shdl:
            if load[item[0]]<(self.comp_times[item[0],item[1]]+item[2]): 
                load[item[0]]=(self.comp_times[item[0],item[1]]+item[2])
        #for i in range(len(load)): load[i]=load[i]-time
        return load
    
    #выводит общее время распределённых задач для каждого процессора
    def processor_time(self):
        time=np.zeros(self.nprocessors);
        for item in self.shdl:
            time[item[0]]+=(self.comp_times[item[0],item[1]]) 
        return time
    
    
    #выводит число непосчитанных предшественников для данной задачи (legacy метрика)
    def npreqs_notcomputed(self):
        notcomp=np.zeros(self.ntasks)
        for i in range(self.ntasks):
            for task in self.prequesites(i):
                if task not in self.scheduled:
                    notcomp[i]+=1
        notcomp=notcomp/self.ntasks
        return notcomp;

     #выводит минимальное время, нужное для того, чтобы распределить задачу (legacy метрика)
    def time_to_start(self):
        TTS=np.zeros(self.ntasks)
        for i in range(self.ntasks):
            for task in self.prequesites(i):
                if task not in self.scheduled:
                    TTS[i]+=np.amin(self.comp_times[:,task])
        TTS=TTS/self.ntasks
        return TTS; 
       
    #в какой-то момент показывало, свободен ли процессов в данный момент времени t (legacy метрика)
    #но потом я отказался от явного времени
    def ifbusy(self):
        ifload=np.zeros(self.nprocessors)
        load=self.processor_load(self.current_time)
        for i in range(self.nprocessors):
            if load[i]>0: ifload[i]=int(1)
        return ifload
    
    #показывает валидно ли распределить задау i на процессор j
    def isvalid(self,ntsk,nproc):
        vld=True
        if (ntsk in self.scheduled): vld=False
        if vld:
            if (self.violation(ntsk,nproc)): vld=False
        return vld
    
    #маска валидных задач
    def get_mask(self):
        valid_mask=np.zeros(len(self.actions),dtype=int)
        for action in range(len(self.actions)):
            if self.actions[action][0] in self.scheduled: continue
            if self.isvalid(self.actions[action][0],self.actions[action][1]): valid_mask[action]=1
        return valid_mask
    
    #запланировать задачу i на процессор j
    def schedule_task(self, ntsk,nproc,mode):
        reward=0
        pr_load=self.processor_time()
        #mode "random" подразумевает, что действие не прошло через маску
        #и мы дополнительно должны проверить его на валидность
        if mode=="random":
            if not(self.isvalid(ntsk,nproc)):
                #print('Invalid action selected')
                reward=-0.2
            else:
                self.scheduled.append(ntsk)
                #добавляем i задачу на j процессор
                self.shdl.append([nproc,ntsk,pr_load[nproc]])
                #обновляем метрики и состояние
                self.load=self.processor_time()
                self.state=list(chain.from_iterable([uppertriangle(self.tree),list(chain.from_iterable((self.comp_times/self.maxlength))),self.load/self.maxlength]))
                if len(self.scheduled)==self.ntasks:
                    self.completed=True
                    reward=self.maxlength-self.schedule_length(self.shdl)
        #mode "mask" подразумевает, что действие изначально валидно
        if mode=="mask":
            self.scheduled.append(ntsk)
            self.shdl.append([nproc,ntsk,pr_load[nproc]])
            self.load=self.processor_time()
            self.state=list(chain.from_iterable([uppertriangle(self.tree),list(chain.from_iterable((self.comp_times/self.maxlength))),self.load/self.maxlength]))
            if len(self.scheduled)==self.ntasks:
                self.completed=True
                reward=self.maxlength-self.schedule_length(self.shdl)
        return reward,self.state;
  
    def act(self, action,mode): 
        #действия в нейронке хранятся в виде списка длиной n_proc*n_task - тут мы его дешифруем
        reward,state=self.schedule_task(self.actions[action][0],self.actions[action][1],mode)
        return reward,state;