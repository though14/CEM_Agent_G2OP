# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:15:45 2023

@author: thoug
"""

"""
1. Decompress all .bz2 file
2. Move .csv file to .xlxs file
3. Change file format from .xlxs to .mat file

"""

import os
import pandas as pd
import csv

from tqdm import tqdm
import time


from numba import jit, cuda

#%%
#1

import bz2



# folder_path="C:\\Users\\thoug\\OneDrive\\SS2023\\Internship\\04_Code\\CEM_Agent_G2OP\\File\\MATPOWER\\Chronic_l2rpn_sandbox"
relative_path = "File/MATPOWER/Chronic_rte_case14"
folder_name = 000


# os.listdir(folder_path) : show the directory inside of the folder
folder_dir = os.listdir(relative_path)



#list of file_name
# file_name = ["_N_datetimes.csv.bz2", "_N_imaps.csv.bz2", "_N_loads_p.csv.bz2", "_N_loads_p_planned.csv.bz2", "_N_loads_q.csv.bz2", "_N_loads_q_planned.csv.bz2", "_N_prods_p.csv.bz2", "_N_prods_p_planned.csv.bz2", "_N_prods_v.csv.bz2", "_N_prods_v_planned.csv.bz2", "_N_simu_ids.csv.bz2", "hazards.csv.bz2", "maintenance.csv.bz2" ]


file_name = ['load_p.csv.bz2', 'load_p_forecasted.csv.bz2', 'load_q.csv.bz2', 'load_q_forecasted.csv.bz2', 'prod_p.csv.bz2', 'prod_p_forecasted.csv.bz2', 'prod_v.csv.bz2', 'prod_v_forecasted.csv.bz2']



file_name_all = os.listdir(os.path.join(relative_path, '197'))

# file_name = file_name_all.copy()

# del file_name[2]
# del file_name[2]




#%%
# while 0000 in folder_dir:
for folder_name in tqdm(folder_dir):
    for i in file_name:
        path_original = os.path.join(relative_path, folder_name)
        # path_decompress = os.path.join(folder_path, folder_name, file_name+'decomp')
        
        with bz2.open(os.path.join(path_original, i), 'rb') as f:
            uncompressed_content = f.read()
            
            content = uncompressed_content.decode('utf-8')
            
            f.close()

        with open(os.path.join(path_original, i+'.csv'), 'wt') as csvf:
            
            readed_data = csv.reader(content.splitlines(), delimiter=';')
            
            csvwriter = csv.writer(csvf)
            
            csvwriter.writerows(readed_data)
            
            csvf.close()
            # csvwriter.close()
            
        os.remove(os.path.join(path_original,i))  #try delete old .bz2 file so matlab can easily convert file into mat format

        
            

            
        # Everything is done by csv file for now
#%%
#We need to modify first row of the csv file

path_ex = "C:\\Users\\thoug\\OneDrive\\SS2023\\Internship\\02_G2OP\\File\\_N_loads_p.csv.bz2.csv"

"""
IDEA : Try read csv again with Pandas and change headers in Pandas.
Once it's mapped, it won't be that hard to do it
"""
 
header = ['2_C', '3_C', '4_C', '5_C', '6_C', '9_C', '10_C', '11_C', '12_C', '13_C', '14_C']
    
    
ex_pd = pd.read_csv(path_ex)

ex_pd.columns=header

ex_pd.to_csv(os.path.join(path_ex + 'ex_mod'))
                    
#Cool, it worked. Let's make for loop again to make it work


#%%
#3
# folder_path="C:/Users/thoug/OneDrive/SS2023/Internship/02_G2OP/File/rte_chronics"
rel_path = "File/MATPOWER/Chronic_rte_case14"
folder_name = 197


# os.listdir(folder_path) : show the directory inside of the folder
folder_dir = os.listdir(rel_path)


# file_name_csv = ["_N_loads_p.csv.bz2.csv", "_N_loads_p_planned.csv.bz2.csv", "_N_loads_q.csv.bz2.csv", "_N_loads_q_planned.csv.bz2.csv", "_N_prods_p.csv.bz2.csv", "_N_prods_p_planned.csv.bz2.csv", "_N_prods_v.csv.bz2.csv", "_N_prods_v_planned.csv.bz2.csv", "_N_simu_ids.csv.bz2.csv"]

# file_name_csv_load =["_N_loads_p.csv.bz2.csv", "_N_loads_p_planned.csv.bz2.csv", "_N_loads_q.csv.bz2.csv", "_N_loads_q_planned.csv.bz2.csv"]
file_name_csv_load = ['load_p.csv.bz2.csv', 'load_p_forecasted.csv.bz2.csv', 'load_q.csv.bz2.csv', 'load_q_forecasted.csv.bz2.csv']


# file_name_csv_gen = ["_N_prods_p.csv.bz2.csv", "_N_prods_p_planned.csv.bz2.csv", "_N_prods_v.csv.bz2.csv", "_N_prods_v_planned.csv.bz2.csv"]
file_name_csv_gen = ['prod_p.csv.bz2.csv', 'prod_p_forecasted.csv.bz2.csv', 'prod_v.csv.bz2.csv', 'prod_v_forecasted.csv.bz2.csv']

# file_name_csv_nochange = [  "_N_simu_ids.csv.bz2.csv", "hazards.csv.bz2.csv", "maintenance.csv.bz2.csv" ]

# file_name_csv_lines = ["_N_imaps.csv.bz2.csv"]

# file_name_csv_noneed = ["_N_datetimes.csv.bz2.csv"]

header_load = ['2_C', '3_C', '4_C', '5_C', '6_C', '9_C', '10_C', '11_C', '12_C', '13_C', '14_C']
header_gen = ['1_G', '3_G','6_G','2_G','8_G']
# header_gen = ['1_G','2_G','3_G','6_1_G', '6_2_G' ,'8_G'] #for l2rpn_sandbox
amps_col = ['1_2_1', "1_5_2", "2_3_3", "2_4_4", "2_5_5", "3_4_6", "4_5_7", "4_7_8", "4_9_9", "5_6_10", "6_11_11", "6_12_12", "6_13_13", "7_8_14", "7_9_15", "9_10_16", "9_14_17", "10_11_18", "12_13_19", "13_14_20"]
amps_col_cor = {'0':'1_2_1', '1': '1_5_2', '2':'2_3_3','3':'2_4_4','4':'2_5_5','5':'3_4_6','6':'4_5_7', '7':'4_7_8', '8':'4_9_9','9':'5_6_10','10':'6_11_11','11':'6_12_12','12':'6_13_13','13':'7_8_14','14':'7_9_15','15':'9_10_16','16':'9_14_17','17':'10_11_18','18':'12_13_19','19':'13_14_20'}

for folder_names in tqdm(folder_dir):
    for j in file_name_csv_load:
        path_csv = os.path.join(rel_path, folder_names)
        # path_decompress = os.path.join(folder_path, folder_name, file_name+'decomp')
        

        
        """
        Read files from Pandas
        if it's load/gen, we use header_load/gen
        """
        
        read_load = pd.read_csv(os.path.join(path_csv, j))
        # read_load.columns=header_load

        # read_load.insert(0,'1_C',[0]*len(read_load)) #adding 1_c
        # read_load.insert(6,'7_C',[0]*len(read_load))
        # read_load.insert(7,'8_C',[0]*len(read_load)) 
        read_load.insert(0,'load_1',[0]*len(read_load))
        read_load.insert(6,'load_6_',[0]*len(read_load)) 
        read_load.insert(7,'load_7_',[0]*len(read_load)) 
        read_load.to_csv(os.path.join(path_csv, j +'__mod.csv'),index=False)
        
        # os.remove(os.path.join(path_csv, '_N_imaps.csv.bz2.csv__mod.csv'))
        
        
    for k in file_name_csv_gen:
        path_csv = os.path.join(rel_path, folder_names)
        
        read_gen = pd.read_csv(os.path.join(path_csv, k))
        # gen_cols= ['1_G','2_G','3_G','6_G','8_G'] #l2rpn_2019
        # read_gen.columns=header_gen
        # read_gen.columns=gen_cols
        
        gen_cols= ['1_G','2_G','3_G','6_G','8_G'] #l2rpn_2019
        # gen_cols= ['1_G','2_G','3_G','6_1_G', '6_2_G' ,'8_G'] #l2rpn_sandbox
        read_gen.columns=header_gen
        
        read_gen=read_gen[gen_cols]
        
        read_gen.insert(3,'4_G',[0]*len(read_gen)) #adding 1_c
        read_gen.insert(4,'5_G',[0]*len(read_gen)) #adding 1_c
        read_gen.insert(6,'7_G',[0]*len(read_gen)) #adding 1_c
        read_gen.insert(8,'9_G',[0]*len(read_gen)) #adding 1_c
        read_gen.insert(9,'10_G',[0]*len(read_gen)) #adding 1_c
        read_gen.insert(10,'11_G',[0]*len(read_gen)) #adding 1_c
        read_gen.insert(11,'12_G',[0]*len(read_gen)) #adding 1_c
        read_gen.insert(12,'13_G',[0]*len(read_gen)) #adding 1_c
        read_gen.insert(13,'14_G',[0]*len(read_gen)) #adding 1_c
        
        #for l2rpn_sandbox
        # read_gen.insert(3,'4_G',[0]*len(read_gen)) #adding 1_c
        # read_gen.insert(4,'5_G',[0]*len(read_gen)) #adding 1_c
        # read_gen.insert(7,'7_G',[0]*len(read_gen)) #adding 1_c
        # read_gen.insert(9,'9_G',[0]*len(read_gen)) #adding 1_c
        # read_gen.insert(10,'10_G',[0]*len(read_gen)) #adding 1_c
        # read_gen.insert(11,'11_G',[0]*len(read_gen)) #adding 1_c
        # read_gen.insert(12,'12_G',[0]*len(read_gen)) #adding 1_c
        # read_gen.insert(13,'13_G',[0]*len(read_gen)) #adding 1_c
        # read_gen.insert(14,'14_G',[0]*len(read_gen)) #adding 1_c
        
        read_gen.to_csv(os.path.join(path_csv, k +'__mod.csv'),index=False)
        
    # for q in file_name_csv_nochange:
    #     path_csv = os.path.join(folder_path, folder_name)
        
    #     read_nochange = pd.read_csv(os.path.join(path_csv, q))
        
            


            
                        
    #     # read_nochange.to_csv(os.path.join(path_csv, q +'__mod.csv'),index=False)
        
    # for r in file_name_csv_noneed:
    #     path_csv = os.path.join(folder_path, folder_name)
        
    #     read_noneed = pd.read_csv(os.path.join(path_csv, r))
    #     # read_noneed.to_csv(os.path.join(path_csv, r +'__mod_nn.csv'),index=False)
        
        
    #     # os.remove(os.path.join(path_csv,'_N_datetimes.csv.bz2.csv__mod.csv'))
        
        
    # for s in file_name_csv_lines:
    #     path_csv = os.path.join(folder_path, folder_name)
        
        
    #     # os.remove(os.path.join(path_csv, '_N_imaps.csv.bz2.csv__mod.csv'))
    #     read_lines = pd.read_csv(os.path.join(path_csv,s))
        
    #     thermal_limit = [1000,1000,1000,1000,1000,1000,1000,760,450,760,380,380,760,760,380,760,380,380,2000,2000]   
    #     sr_thermal_limit = pd.Series(thermal_limit)
        
    #     df_thermal_limit = sr_thermal_limit.to_frame()
    #     df_thermal_limit_tf = df_thermal_limit.transpose()
    #     df_thermal_limit_cor = df_thermal_limit_tf.rename(columns={'0':'1_2_1', '1': '1_5_2', '2':'2_3_3','3':'2_4_4','4':'2_5_5','5':'3_4_6','6':'4_5_7', '7':'4_7_8', '8':'4_9_9','9':'5_6_10','10':'6_11_11','11':'6_12_12','12':'6_13_13','13':'7_8_14','14':'7_9_15','15':'9_10_16','16':'9_14_17','17':'10_11_18','18':'12_13_19','19':'13_14_20'})
        
    #     df_thermal_limit_cor.to_csv(os.path.join(path_csv, s+'__mod.csv'))

"""
Let's try reorder the missing value and that into one


Load missing : 1_C, 7_C 8_C

Gen missing : 4_G, 5_G, 7_G, 9_G, 10_G, 11_G, 12_G


So we make the colums based on the node number, which is 14
and if not assigned, we will just put the value as 0 or none




reorder_levels:https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reorder_levels.html
insert new col : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.insert.html

I think inserting new_col is faster to achieve


After sorting out, then re read it again, and update?

Load : 2 3 4 5 6 x x 9 10 11 12 13

Gen : 1 2 3 x x 6 x 8 x x x x x(13)    <-- Gen must sort the order as well


It is implemented above

"""


        
        
        
        
        
        
        

