# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:00:57 2017

@author: user
"""

import numpy as np
import workflowenv2 as wf

comptime=np.random.randint(20, size=(3, 5))+1
tree=wf.treegen(5)
print(tree)

wfl=wf.workflow(tree,comptime)


while not(wfl.completed):
    random_action=np.random.randint(16)
    print(wfl.actions[random_action])
    wfl.act(random_action)
    print(wfl.scheduled)

print(wfl.maxlength-wfl.schedule_length(wfl.shdl))
