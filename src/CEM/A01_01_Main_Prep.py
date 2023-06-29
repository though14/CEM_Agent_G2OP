# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:59:13 2023

@author: thoug
"""

import grid2op
import pandas as pd
import torch
import torch.nn as nn
import l2rpn_baselines
import gym
import numpy as np


import os
import re
import json

# torch.set_default_device('cuda')

env_name = "l2rpn_2019"


#%%
#Setting the parameter same as the paper


"""
Line : Thermal Limit(A)
0~6 : 1000
7:760
8:450
9:760
10~11:380
12~13:760
14:380
15:760
16~17:380
18~19:2000

Hard Constraint
1) system demand must be fully served
2) no generator may be disconnected
3) no electrical island be formed as result of topology control
4) AC power flow must converge

Soft Constraint
1) Transmission line 150% -> tripped for 10 time step
2) Line over loaded less than 150% -> agent has 2 time step to mitigate
    if not, tripped for 10 time step
3) Only one substation can be modifed per time step
4) After substation is used, 3 time step is required for another action


"""
from grid2op.Parameters import Parameters
from grid2op.Action import TopologyAction
from grid2op.Reward import L2RPNReward

#Creating Parameters
p = Parameters()

#Allowing one substation to be modified
p.MAX_SUB_CHANGED = 1

#Cooltime for substation
p.NB_TIMESTEP_COOLDOWN_SUB = 3

#Setting Hard Flow time as 150, threshold should became 1
#at 150% it will be disconnected
p.HARD_OVERFLOW_THRESHOLD = 1

#make it not connected even if goes over 100% of power line limit
p.NO_OVERFLOW_DISCONNECTION = True


thermal_limit = [1000,1000,1000,1000,1000,1000,1000,760,450,760,380,380,760,760,380,760,380,380,2000,2000]

th_lim = np.array(thermal_limit)

#after setting all the parameter, we make the environment
env_out = grid2op.make(env_name, 
                   param=p,
                   action_class = TopologyAction,
                   reward_class = L2RPNReward)

env_out.set_thermal_limit(th_lim)

env_out
#%%


