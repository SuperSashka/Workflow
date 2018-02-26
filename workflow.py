# -*- coding: utf-8 -*-
"""
все используемые пакеты

import random
from collections import deque
import numpy as np
import json
#from keras import initializations
#from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from collections import Iterable
from itertools import chain
import numpy as np
import tensorflow as tf
#модуль с правилами для составления расписания
import workflowenv as wf
#модуль с агентом и сетью
import actor


"""

import numpy as np
import tensorflow as tf
#модуль с правилами для составления расписания
import workflowenv as wf
#модуль с агентом и сетью
import actor


if __name__ == "__main__":
    #инициализируем Tensorflow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #bybwbfkbpbhetv Keras
    from keras import backend as K
    K.set_session(sess)
    learning61=[] 
    time61=[]
    #число задач
    task_par=10
    #минимальное число задач
    task_par_min=10
    #число процессоров
    proc_par=3
    #длинна вектора состояния (вылезает ошибка с числом которое надо подставить при первом запуске с новыми параметрами)
    state_size = 116
    #число действий
    action_size = task_par*proc_par
    #инициализируем агента
    agent =actor.DQNAgent(state_size, action_size)
    #функция загрузки весов (сохраняются agent.save(name))
    #agent.load("weights90000")
    done = False
    #размер выборки из памяти для обучения
    batch_size = 32
    #метрикиe
    cumulativereward=0
    scoreavg=0
    timeavg=0
    EPISODES=1000000
    loss=0
    lenavg=0
    negative_amount=0
    
    for e in range(EPISODES):
        #генерируем матрицу ресурсов (размерность n_task*n_proc)
        comptime=wf.compgen(task_par,proc_par)
        inout=wf.out_gen(task_par)
        #генерируем дерево задач 
        chns=wf.treegen(task_par_min,task_par)
        random_task_amount=len(chns)
        lenavg+=random_task_amount
        #передаём их как параметры для окружения 
        wfl=wf.workflow(chns,comptime,task_par,inout)
        done=wfl.completed
        #получаем начальное сосотояние из окружения
        state = wfl.state
        #требуется для совместимости с tf
        state = np.reshape(state, [1, state_size])
        #5000 - с потолка
        for time in range(5000):
            #получаем маску для валидных задач
            mask=wfl.get_mask()
            #mask="none" #используется в случае, если мы не хотим использовать маску, а хотим чтобы сеть самостоятельно выучила правила (дольше)
            #тут передаётся маска и выбирается наиболее "предпочтительное" действие из политики с учётом маски
            action = agent.act(state,mask)
            #выбранное действие передаётся в окружение, чтобы  включить его в расписание
            total_time,_=wfl.act(action,"mask")
            #total_time,_=wfl.act(action,"random") # используется, если мы не включаем маску, например в рандоме (workflowRAND.py)
            #получаем новое состояние с учётом нового расписания
            next_state = wfl.state
            #проверяем, распределены ли все задачи
            done=wfl.completed
            #получаем награду
            reward=total_time
            if reward<0:
                negative_amount+=1
                #raise SystemExit(0)
            cumulativereward+=reward
            #требуется для совместимости с tf
            next_state = np.reshape(next_state, [1, state_size])
            #запоминаем цепочку действий
            agent.remember((state, action, reward, next_state, done))
            state = next_state
            if done:
                #выводим метрики, если расписание составлено
                scoreavg+=total_time
                timeavg+=time
                neps=e+1
                print("episode: {}/{},ntask: {},ntask_avg: {:.4}, score/ntask: {:.4}, s/t avg: {:.4}, neg: {:.4}".format(e, EPISODES,random_task_amount, lenavg/neps,total_time, scoreavg/neps, negative_amount/neps))
                learning61.append([scoreavg/neps])
                time61.append([timeavg/neps])
                break
        #обучаем нейронку
        if e%100==0:
            if len(agent.D) > batch_size:
                loss+=agent.replay(batch_size,e)
        if e%10000==0:
            agent.save("weights"+str(e))
    #строим кривую обучения
    import matplotlib.pyplot as plt 
    plt.figure(figsize=(10,5))
    plt.plot(learning61[100:], '-')
    #plt.axhline(y=776, color='b', linestyle='-')
    plt.ylabel('avg reward')
    plt.xlabel('episodes')
    plt.show()
