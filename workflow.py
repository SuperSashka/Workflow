#!/usr/bin/env python
from __future__ import print_function



import numpy as np
#среда для составления расписания с ограничениями и тп
import enviroment as wf
import tensorflow as tf
#агент с сетью, который составляет расписание
import actor 
    
if __name__ == "__main__":
    #число задач
    taskn=60
    #число процессоров
    procn=5
    #запускаем TF
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    #переменная для графика обучения
    learning61=[] 
    # при первом запуске с заданными задачами и процессорами python выдаёт ошибку, в которой содержится это число
    state_size = 2375
    #число задач*число процессоров
    action_size = 300
    #инициализируем агента и сеть
    agent = actor.DQNAgent(state_size, action_size)
    #переменная которая показывает, все ли задачи распределены
    done = False
    #размер "батча", набора рандомных действий, который мы берём тз памяти
    batch_size = 32
    #немного переменных для промежуточных статистик
    #в общем случае "награда" и "очки" - две разные величины, очки тут обозначают именно разность длины наихудшего 
    #и составленного расписания, а наргаду можно давать и за другие действия
    cumulativereward=0
    scoreavg=0
    #число эпизодов для обучения
    EPISODES=100000
    #dummy variable
    loss=0
    for e in range(EPISODES):
        #генерируем матрицу ресурсов
        comptime=wf.compgen(taskn,procn)
        #генерируем рандомное дерево        
        chns=wf.treegen(taskn)
        #инициализируем workflow с сгенерированными матрицами
        wfl=wf.workflow(chns,comptime)
        #переменная которая показывает, все ли задачи распределены
        done=wfl.completed
        #текущее состояние
        state = wfl.state
        #требуется в TF
        state = np.reshape(state, [1, state_size])
        #считается, что за 5к действий любое расписание будет создано
        for time in range(5000):
            #выбираем валидное действие с помощью сетки
            action = agent.act(state,wfl.val_mask)
            #отправляем его в wf
            total_time,_=wfl.act(action)
            #получаем состояние из wf
            next_state = wfl.state
            done=wfl.completed
            #и награду (0, если done не True)
            reward=total_time
            cumulativereward+=reward
            #требуется в TF
            next_state = np.reshape(next_state, [1, state_size])
            #запоминаем действие в память
            agent.remember((state, action, reward, next_state, done))
            state = next_state
            #если расписание готово, выводим промежуточные статистики
            if done:
                scoreavg+=total_time
                neps=e+1
                print("episode: {}/{}, score: {}, score avg: {:.4}, rew. avg, {:.4}"
                      .format(e, EPISODES, total_time, scoreavg/neps,cumulativereward/neps))
                learning61.append([scoreavg/neps])
                break
        #в конце каждого эпизода обучаем нейронку
        if len(agent.D) > batch_size:
            loss+=agent.replay(batch_size,e)
    #в конце эпохи выводим график обучения
    import matplotlib.pyplot as plt 
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(20,10))
    plt.plot(learning61[100:], '-')
    plt.ylabel('avg reward')
    plt.xlabel('episodes')
    plt.show()

