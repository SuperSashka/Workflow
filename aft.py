# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 17:31:32 2017

@author: user
"""
import workflowenv as wf
import numpy as np



def order_violation(ast_chain,aft_chain):
    violation=False
    for i in range(len(ast_chain)):
        if len(ast_chain[i])>1:
            for j in range(len(ast_chain[i])-1):
                if ast_chain[i][j+1]<aft_chain[i][j]:
                    print('oder violation, chain {} element {} starts before {} ends'.format(i+1,j+2,j+1))
            violation=True
    return violation;

chns=wf.chgen(5)
comptime=np.random.randint(20, size=(3, 5))+1
wfl=wf.workflow(chns,comptime)

while not(wfl.completed):
    action = np.random.randint(15)
    reward,_=wfl.act(action)
    shdl=wfl.shdl   
    aft=wfl.AFT(shdl,comptime)
    ast=wfl.AST(shdl,comptime)
    aft_chain=wfl.AFT_chain(wfl.tsk_mem,aft)
    ast_chain=wfl.AST_chain(wfl.tsk_mem,ast)
    print(aft_chain)
    print(ast_chain)
    order_violation(ast_chain,aft_chain)

