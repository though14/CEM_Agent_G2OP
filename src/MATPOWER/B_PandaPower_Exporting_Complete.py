# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:34:05 2023

@author: thoug
"""

import numpy as np
import pandapower as pp
import pandas as pd

import os
import openpyxl as oxl

from tqdm import tqdm
import time


# net = pp.from_json("grid.json")

path_folder = "C:\\Users\\thoug\\OneDrive\\SS2023\\Internship\\04_Code\\CEM_Agent_G2OP\\src\\MATPOWER\\File\\Grid"
file_name = "l2rpn_sandbox.json"

#%%
#General Reading

# def sheet_reading(SHEET_NAME):
    
#     sheet = pd.read_excel(io=os.path.join(path_folder,file_name), sheet_name = SHEET_NAME, index_col = 0)
    
#     return sheet


# def reading_col(sheet, COL_NAME):
    
#     info_col = sheet[COL_NAME]
    
#     return info_col

# #%%
# #Bus data

# #reading Bus sheet of G2OP

# bus_sheet = sheet_reading("bus")
# load_sheet = sheet_reading("load")

# #changing name of existing sheet to MATPOWER
# bus_mod = bus_sheet.rename(columns={"name":"BUS_I", "type":"BUS_TYPE", "zone":"BUS_AREA", "min_vm_pu":"VMIN", "max_vm_pu":"VMAX" })

# #try modified the data to make it fit to MATPOWER 
# bus_mod["PD"] = load_sheet



"""
Forget about Above, we find Better Way
"""


# import julia as ju
# import julia.PowerModels as jp
# import julia.Ipopt as ji
# import julia.PandaModels as jm



# # jp.solve_ac_opf(os.path.join(path_folder,"case3.m"), ji.Optimizer)


# # loading_julia = jp.parse_json(os.path.join(path_folder,'grid.json'))

net = pp.from_json(os.path.join(path_folder,file_name))

pp.runpp(net)

# net_pm = jm.call_pan
#%%


"""
Now I feel like an idiot
PandaPower support MPC format turning on its own
"""

pp.converter.to_mpc(net, "rte_case14.mat")