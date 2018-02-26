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
            
#возвращает индексы i,j элемента v в матрице
def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return i, x.index(v)

#генерируем дерево зависимостей задач            
def treegen(length_from,length_to):
    if length_from<length_to:
        tree_length=np.random.randint(length_from,length_to)
    else:
        tree_length=length_to
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

def out_gen(ntask):
    out=np.random.randint(5,100, size=ntask)
    return out

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
    def __init__(self, tree,comp_times,max_taskn,output):
        self.internet_speed=50
        self.tree = tree
        self.maxtree=np.zeros((max_taskn,max_taskn))
        self.maxtree[0:len(self.tree),0:len(self.tree)]=self.tree
        self.comp_times = comp_times
        self.shdl=[]
        #for terminal state we need to know is scheduling is completed
        self.completed=False
        #number of chains of tasks
        self.max_task=max_taskn
        self.ntasks=len(self.tree)
        self.out=np.zeros(max_taskn)
        self.out[0:self.ntasks]=output[0:self.ntasks]
        self.inp=self.input_gen()
        self.preqset=[]
        for i in range(self.ntasks):
            self.preqset.append(set(self.parents(i)))
        #number of processors
        self.nprocessors=len(comp_times)
        self.maxlength=np.amax(self.max_comp_length())
        self.load=np.zeros(self.nprocessors)
        #for DQN we need to have all actions as a number, every action here is (i,j), schedule first task in chain
        #j on i-th processor
        self.actions=[]
        for i in range(max_taskn):
            for j in range(self.nprocessors):
                self.actions.append([i,j])
        self.scheduled=[]
        self.utrig=uppertriangle(self.maxtree)
        #self.state=list(chain.from_iterable([self.utrig,list(chain.from_iterable((self.comp_times/self.maxlength))),self.load/self.maxlength]))
        #распределение задач по уровням
        self.lvls=self.levels() 
        self.height=len(self.lvls)
        self.w=[]
        for lvl in self.lvls:
            self.w.append(len(lvl))
        self.max_width=max(self.w)
        self.avg_width=sum(self.w)/len(self.w)
        self.procinf=np.zeros((4,max_taskn))
        self.procinf[:,0:self.ntasks]=self.task_lvl()
        self.wflinfo=np.asarray([self.height/self.ntasks,self.max_width/self.ntasks,self.avg_width/self.ntasks])
        self.node_parents=np.zeros((self.max_task,self.nprocessors))
        self.node_dict={}
        self.valid_tsk=set()
        self.state=[]
        self.state_update()
        
    def state_update(self):
        self.state=list(chain.from_iterable([self.wflinfo,list(chain.from_iterable(self.procinf)),self.ifscheduled(),list(chain.from_iterable((self.comp_times/self.maxlength))),self.load/self.maxlength,list(chain.from_iterable(self.node_parents/self.ntasks))]))
        return
    
        
    #выводит вектор распределённых задач, 1-в расписании, 0 - ещё нет    
    def ifscheduled(self):
        ifshdl=np.zeros(self.max_task)
        for task in self.scheduled:
            ifshdl[task]=int(1)
        return ifshdl
    

    
    #выводит непосредственных предшественников i-той задачи
    def parents(self,task):
        parents, = np.where(self.tree[task]==-1 )
        return parents

    def parents_on_node(self,task):
        nodes=np.zeros(self.nprocessors)
        for parent in self.parents(task):
            if parent in self.scheduled: 
                nodes[self.node_dict[parent]]+=1
        return nodes
    
    def parents_update(self):
        for task in self.valid_tsk:
            self.node_parents[task,:]=self.parents_on_node(task)
        return
    
    def childs(self,task):
        childs, = np.where(self.tree[task]==1 )
        return childs
    
    def levels(self):
        lvl=[];
        currlvl=[];
        for taskn in range(len(self.tree)):
            if self.parents(taskn).size==0: currlvl.append(taskn)
        lvl.append(currlvl);
        for _ in range(self.ntasks):
            currlvl=[];
            for t in lvl[-1]:
                currlvl+=(self.childs(t).tolist())
            for lowerlvl in range(len(lvl)):
                currlvl=list(set(currlvl)-set(lvl[lowerlvl]))
            if len(currlvl)==0 :break
            lvl.append(currlvl)
        return lvl

    def task_lvl(self):
        tsklvl=np.zeros(len(self.tree),dtype=int)
        tsk_on_lvl=np.zeros(len(self.tree),dtype=int)
        n_parents=np.zeros(len(self.tree),dtype=int)
        n_childs=np.zeros(len(self.tree),dtype=int)
        for tsk in range(len(self.tree)):
            tsklvl[tsk],_=index_2d(self.lvls,tsk)
            tsk_on_lvl[tsk]=len(self.lvls[tsklvl[tsk]])
            n_parents[tsk]=len(self.parents(tsk))
            n_childs[tsk]=len(self.childs(tsk))
        return tsklvl,tsk_on_lvl,n_parents,n_childs

    def input_gen(self):
        inp_gen=np.zeros(self.max_task)
        for task in range(self.ntasks):
            for parent in self.parents(task):
                inp_gen[task]+=self.out[parent]
        return inp_gen

    #выводит полную цепочку задач до i-той, начиная с первой
    def process_chain(self,task):
        subtasks=[task]
        if self.parents(task)!=[]:
            for subtask in self.parents(task):
                if subtask not in subtasks:
                    subtasks.append(subtask)
            if self.parents(subtask)!=[]: subtasks.append(self.process_chain(subtask))
        subtasks=list(flatten(subtasks))
        subtasks=list(set(subtasks))
        return subtasks
    
    #служит для оценки "наихудшего" расписания
    def max_comp_length(self):
        max_length=np.sum(self.comp_times,axis=1)
        return max_length
            

    #проверка, распределены ли все предшественники для данной задачи
    def violation(self,task):
        vltn=True
        preq=self.preqset[task]
        sdl=set(self.scheduled)
        if (preq.intersection(sdl)==preq): vltn=False
                #print('Prequesites are not yet computed')
        return vltn

    
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
    
    
    #показывает валидно ли распределить задау i на процессор j
    def isvalid(self,ntsk,nproc):
        vld=True
        if (ntsk in self.scheduled): vld=False
        if (ntsk>=self.ntasks): vld=False
        if vld:
            if (self.violation(ntsk)): vld=False
        return vld
    
    #маска валидных задач
    def get_mask(self):
        self.valid_tsk=set()
        valid_mask=np.zeros(len(self.actions),dtype=int)
        for action in range(len(self.actions)):
            if self.actions[action][0] in self.scheduled: continue
            if self.isvalid(self.actions[action][0],self.actions[action][1]): 
                valid_mask[action]=1
                self.valid_tsk.add(self.actions[action][0])
        self.parents_update()
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
                """
                for child in self.childs(ntsk):
                    self.comp_times[nproc,child]-=self.out[ntsk]
                """
                #обновляем метрики и состояние
                self.load=self.processor_time()
                self.state_update()
                if len(self.scheduled)==self.ntasks:
                    self.completed=True
                    reward=self.maxlength-self.schedule_length(self.shdl)
        #mode "mask" подразумевает, что действие изначально валидно
        if mode=="mask":
            self.scheduled.append(ntsk)
            self.shdl.append([nproc,ntsk,pr_load[nproc]])
            """
            for child in self.childs(ntsk):
                self.comp_times[nproc,child]-=self.out[ntsk]
            """
            self.parents_update()
            self.load=self.processor_time()
            self.state_update()
            self.node_dict[ntsk]=nproc
            if len(self.scheduled)==self.ntasks:
                self.completed=True
                reward=(self.maxlength-self.schedule_length(self.shdl))/self.ntasks
                #reward=self.maxlength-self.schedule_length(self.shdl)
        return reward,self.state;
  
    def act(self, action,mode): 
        #действия в нейронке хранятся в виде списка длиной n_proc*n_task - тут мы его дешифруем
        reward,state=self.schedule_task(self.actions[action][0],self.actions[action][1],mode)
        return reward,state;