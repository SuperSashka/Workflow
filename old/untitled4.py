# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:31:32 2017

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt


#t=0

N = 3
P1=(12,16,4,7,19)
P2=(12,19,0,0,0)
P3=(2,13,18,0,0)
P=(P1,P2,P3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

G1=np.asarray(P)

G2=G1.transpose()

G2=tuple(map(tuple, G2))

plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 22})
p1 = plt.barh(ind, G2[0], width,color=['orange','r','y'])
p2 = plt.barh(ind, G2[1], width,
             left=G2[0],color=['c','g','b'])
p3 = plt.barh(ind, G2[2], width,
             left=tuple(map(sum, zip(G2[0], G2[1]))),color=['lightblue','white','m'])
p4 = plt.barh(ind, G2[3], width,
             left=tuple(map(sum, zip(G2[0], G2[1],G2[2]))),color=['black','white','white'])
p5 = plt.barh(ind, G2[4], width,
             left=tuple(map(sum, zip(G2[0], G2[1],G2[2],G2[3]))),color=['lightgreen','m','c','m'])

plt.xlabel('Time')
plt.title('Processor time (episode=0)')
plt.xticks(np.arange(0, 61, 10))
plt.yticks(ind, ('P1', 'P2', 'P3'))
plt.legend((p1[0],p1[1], p1[2],p2[0],p2[1],p2[2],p3[0],p3[2],p4[0],p5[0]), ('t1', 't0','t2','t8','t3','t5','t4','t6','t7','t9'))

plt.show()

#t=25000

N = 3
P1=(9,7,5,16,19)
P2=(9,17,0,0,0)
P3=(10,18,19,0,0)
P=(P1,P2,P3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

G1=np.asarray(P)

G2=G1.transpose()

G2=tuple(map(tuple, G2))

plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 22})
p1 = plt.barh(ind, G2[0], width,color=['r','green','orange'])
p2 = plt.barh(ind, G2[1], width,
             left=G2[0],color=['y','lightblue','m'])
p3 = plt.barh(ind, G2[2], width,
             left=tuple(map(sum, zip(G2[0], G2[1]))),color=['blue','white','black'])
p4 = plt.barh(ind, G2[3], width,
             left=tuple(map(sum, zip(G2[0], G2[1],G2[2]))),color=['c','white','white'])
p5 = plt.barh(ind, G2[4], width,
             left=tuple(map(sum, zip(G2[0], G2[1],G2[2],G2[3]))),color=['lightgreen','m','c','m'])

plt.xlabel('Time')
plt.title('Processor time (episode=25000)')
plt.xticks(np.arange(0, 61, 10))
plt.yticks(ind, ('P1', 'P2', 'P3'))
plt.legend((p1[0],p1[1], p1[2],p2[0],p2[1],p2[2],p3[0],p3[2],p4[0],p5[0]), ('t0', 't3','t1','t2','t4','t6','t5','t7','t8','t9'))

plt.show()

#t=50000

N = 3
P1=(9,16,7,19,0)
P2=(9,14,17,12,0)
P3=(10,2,0,0,0)
P=(P1,P2,P3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

G1=np.asarray(P)

G2=G1.transpose()

G2=tuple(map(tuple, G2))

plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 22})
p1 = plt.barh(ind, G2[0], width,color=['r','green','orange'])
p2 = plt.barh(ind, G2[1], width,
             left=G2[0],color=['c','blue','y'])
p3 = plt.barh(ind, G2[2], width,
             left=tuple(map(sum, zip(G2[0], G2[1]))),color=['black','lightblue','white'])
p4 = plt.barh(ind, G2[3], width,
             left=tuple(map(sum, zip(G2[0], G2[1],G2[2]))),color=['lightgreen','m','white'])
p5 = plt.barh(ind, G2[4], width,
             left=tuple(map(sum, zip(G2[0], G2[1],G2[2],G2[3]))),color=['lightgreen','m','c','m'])

plt.xlabel('Time')
plt.title('Processor time (episode=50000)')
plt.xticks(np.arange(0, 61, 10))
plt.yticks(ind, ('P1', 'P2', 'P3'))
plt.legend((p1[0],p1[1], p1[2],p2[0],p2[1],p2[2],p3[0],p3[1],p4[0],p4[1]), ('t0', 't3','t1','t8','t5','t2','t5','t7','t8','t9'))

plt.show()

#t=75000

N = 3
P1=(9,16,4,0,0)
P2=(11,14,12,11,0)
P3=(2,18,18,0,0)
P=(P1,P2,P3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

G1=np.asarray(P)

G2=G1.transpose()

G2=tuple(map(tuple, G2))

plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 22})
p1 = plt.barh(ind, G2[0], width,color=['r','orange','y'])
p2 = plt.barh(ind, G2[1], width,
             left=G2[0],color=['c','blue','green'])
p3 = plt.barh(ind, G2[2], width,
             left=tuple(map(sum, zip(G2[0], G2[1]))),color=['lightblue','m','lightgreen'])
p4 = plt.barh(ind, G2[3], width,
             left=tuple(map(sum, zip(G2[0], G2[1],G2[2]))),color=['white','black','white'])
p5 = plt.barh(ind, G2[4], width,
             left=tuple(map(sum, zip(G2[0], G2[1],G2[2],G2[3]))),color=['lightgreen','m','c','m'])

plt.xlabel('Time')
plt.title('Processor time (episode=75000)')
plt.xticks(np.arange(0, 61, 10))
plt.yticks(ind, ('P1', 'P2', 'P3'))
plt.legend((p1[0],p1[1], p1[2],p2[0],p2[1],p2[2],p3[0],p3[1],p3[2],p4[1]), ('t0', 't1','t2','t8','t5','t3','t4','t6','t9','t7'))

plt.show()

#t=100000

N = 3
P1=(9,4,7,19,0)
P2=(11,9,14,5,0)
P3=(2,18,0,0,0)
P=(P1,P2,P3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

G1=np.asarray(P)

G2=G1.transpose()

G2=tuple(map(tuple, G2))

plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 22})
p1 = plt.barh(ind, G2[0], width,color=['r','orange','y'])
p2 = plt.barh(ind, G2[1], width,
             left=G2[0],color=['lightblue','green','m'])
p3 = plt.barh(ind, G2[2], width,
             left=tuple(map(sum, zip(G2[0], G2[1]))),color=['black','blue','white'])
p4 = plt.barh(ind, G2[3], width,
             left=tuple(map(sum, zip(G2[0], G2[1],G2[2]))),color=['lightgreen','c','white'])
p5 = plt.barh(ind, G2[4], width,
             left=tuple(map(sum, zip(G2[0], G2[1],G2[2],G2[3]))),color=['lightgreen','m','c','m'])

plt.xlabel('Time')
plt.title('Processor time (episode=100000)')
plt.xticks(np.arange(0, 61, 10))
plt.yticks(ind, ('P1', 'P2', 'P3'))
plt.legend((p1[0],p1[1], p1[2],p2[0],p2[1],p2[2],p3[0],p3[1],p4[0],p4[1]), ('t0', 't1','t2','t4','t3','t6','t7','t5','t9','t8'))

plt.show()

def shdl_plot(Sconv):
    N = 3
    #t=100000
    #maximal task amount on one processor
    max_task=5
    #extract computation times from Sconv
    P1=[]
    for i in range(max_task):
        if i<len(Sconv[0]): 
            P1+=[Sconv[0][i][2]]
        else:
            P1+=[0]
    P1=tuple(P1)
    P2=[]
    for i in range(max_task):
        if i<len(Sconv[1]): 
            P2+=[Sconv[1][i][2]]
        else:
            P2+=[0]
    P2=tuple(P2)
    P3=[]
    for i in range(max_task):
        if i<len(Sconv[2]): 
            P3+=[Sconv[2][i][2]]
        else:
            P3+=[0]
    P3=tuple(P3)
    P=(P1,P2,P3)
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence
    #transpose comp_times
    G1=np.asarray(P)

    G2=G1.transpose()

    G2=tuple(map(tuple, G2))
    colorscheme=['r','orange','y','green','lightblue','blue','m','black','c','lightgreen']
    #extracting task numbers from Sconv
    P1t=[]
    for i in range(max_task):
        if i<len(Sconv[0]): 
            P1t+=[Sconv[0][i][0]]
        else:
            P1t+=[10]
    P1t=tuple(P1t)
    P2t=[]
    for i in range(max_task):
        if i<len(Sconv[1]): 
            P2t+=[Sconv[1][i][0]]
        else:
            P2t+=[10]
    P2t=tuple(P2t)
    P3t=[]
    for i in range(max_task):
        if i<len(Sconv[2]): 
            P3t+=[Sconv[2][i][0]]
        else:
            P3t+=[10]
    P3t=tuple(P3t)
    Pt=(P1t,P2t,P3t)
    #and converting them to the colors
    P1c=[]
    for i in range(max_task):
        if Pt[0][i]<10: 
            P1c+=[colorscheme[Pt[0][i]]]
        else:
            P1c+=['white']
    P2c=[]
    for i in range(max_task):
        if Pt[1][i]<10: 
            P2c+=[colorscheme[Pt[1][i]]]
        else:
            P2c+=['white']
    P3c=[]
    for i in range(max_task):
        if Pt[2][i]<10: 
            P3c+=[colorscheme[Pt[2][i]]]
        else:
            P3c+=['white']
    G1c=np.asarray([P1c,P2c,P3c])
    G2c=G1c.transpose()
    G2c=list(map(list, G2c))

    plt.figure(figsize=(20,10))
    plt.rcParams.update({'font.size': 22})
    p1 = plt.barh(ind, G2[0], width,color=G2c[0])
    p2 = plt.barh(ind, G2[1], width,
                 left=G2[0],color=G2c[1])
    p3 = plt.barh(ind, G2[2], width,
             left=tuple(map(sum, zip(G2[0], G2[1]))),color=G2c[2])
    p4 = plt.barh(ind, G2[3], width,
             left=tuple(map(sum, zip(G2[0], G2[1],G2[2]))),color=G2c[3])
    p5 = plt.barh(ind, G2[4], width,
             left=tuple(map(sum, zip(G2[0], G2[1],G2[2],G2[3]))),color=G2c[4])

    plt.xlabel('Time')
    plt.title('Processor time (episode=100000)')
    plt.xticks(np.arange(0, 61, 10))
    plt.yticks(ind, ('P1', 'P2', 'P3'))
    plt.legend((p1[0],p1[1], p1[2],p2[0],p2[1],p2[2],p3[0],p3[1],p4[0],p4[1]), ('t0', 't1','t2','t4','t3','t6','t7','t5','t9','t8'))
    plt.show()
    return;