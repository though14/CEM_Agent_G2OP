# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 01:11:50 2023

@author: thoug
"""

import matplotlib.pyplot as plt
import numpy as np


import E_01_EP_STUDY_CLASS as E1

zero = np.linspace(0,10,100)
x = np.linspace(1, 1, 100)  # Sample data.

#buses
plt.figure(figsize=(5, 2.7), layout='constrained')
plt.plot(zero, x, label='bus1')  # Plot some data on the (implicit) axes.
plt.plot(zero, x*2, label='bus2')  # etc.


#try connect bus1 and bus2 at x=8
vertical = np.linspace(0,0,100)
plt.plot([8,8],[1,2])


#try make element connected to line

plt.plot([2,2],[-1,1])
plt.plot([3,3],[-1,2])



plt.axis([0,10,-2,3])
plt.xlabel('elements')
# plt.ylabel('y label')
plt.title("Bus-Bar Graph")
plt.legend()





#%%


