# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 19:28:29 2023

@author: thoug
"""

import os
import grid2op
from grid2op.Episode import EpisodeData
import plotly.graph_objects as go
import plotly.io as pio

import numpy as np
import pandas as pd
import pickle


class TOPO_Study():
    def __init__(self, PATH = None, EP_NUM=0, ):
        self.path_str = PATH
        self.ep_num = EP_NUM
        
        self.result_path = "C:\\Users\\thoug\\OneDrive\\SS2023\\Internship\\02_G2OP\\File\\Agent\\June\\save_8"
        
        self.path_info = None
        self.all_obs =[]
        
    def path_read(self, FILE_NAME=None):
        
        path_info  = os.path.join(self.path_str, FILE_NAME)
        self.path_info = path_info
        
        return path_info
    
    
    def path_result_read(self):
        
        return self.result_path
    
    
    def ep_read(self):
        
        return self.ep_num
    
    
    def data_read(self, PATH_OBJ=None):
        
        with open(PATH_OBJ, 'rb') as filehandle:
            read_obj = pickle.load(filehandle)
            
        return read_obj
    
    
    def obs_read(self ):
        episode_studied_res = EpisodeData.list_episode(self.result_path)
        
        
        this_ep_res = EpisodeData.from_disk(*episode_studied_res[self.ep_num])
        ep = this_ep_res


        ep = this_ep_res
        eos = ep.observations
        
        all_obs = [el for el in eos]
        
        self.ll_obs = all_obs
        
        return all_obs
    
    
    def topo_table(self, all_obs=None):
        
        change_list= np.array([])
        c_list=np.array([])

        for sub_id in range(0,14):
            dict_ = all_obs[0].get_obj_connect_to(substation_id=sub_id)
            # print("There are {} elements connected to this substation (not counting shunt)".format(
            #           dict_["nb_elements"]))
            # print("The names of the loads connected to substation {} are: {}".format(
            #             sub_id, all_obs[0].name_load[dict_["loads_id"]]))
            # print("The names of the generators connected to substation {} are: {}".format(
            #             sub_id, all_obs[0].name_gen[dict_["generators_id"]]))
            # print("The powerline whose origin end is connected to substation {} are: {}".format(
            #             sub_id, all_obs[0].name_line[dict_["lines_or_id"]]))
            # print("The powerline whose extremity end is connected to substation {} are: {}".format(
            #             sub_id, all_obs[0].name_line[dict_["lines_ex_id"]]))
            # print("The storage units connected to substation {} are: {}".format(
            #             sub_id, all_obs[0].name_line[dict_["storages_id"]]))
            # print(f"_____________substation{sub_id} ends________________")
            
            sub_num = [sub_id]
            load_num = dict_['loads_id']
            gen_num = dict_['generators_id']
            line_or_num =dict_["lines_or_id"]
            line_ex_num = dict_["lines_ex_id"]
            
            sub_np = np.array([sub_id])
            
            
            change_list_loc = [line_ex_num, line_or_num, gen_num, load_num ]
            change_list_loc_new = np.concatenate([x for x in change_list_loc if len(x)>0])
            
            change_list_final = np.append(change_list, change_list_loc_new)
            change_list=change_list_final
            
            topo_vect_list = change_list  # we made the list of topology vector
            
            
            
            df_change_list_loc = pd.DataFrame(change_list_final, columns=['element_id'])
            df_sub_id = pd.DataFrame(np.array([0,0,0,1,1,1,1,1,1,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,7,7,8,8,8,8,8,9,9,9,10,10,10,11,11,11,12,12,12,12,13,13,13]), columns = ['sub_id'])
            
            d_obs_attr = {'obs_attr': ['LO', 'LO','G','LE','LO','LO','LO','G','L','LE','LO','G','L','LE','LE','LO','LO','LO','L','LE','LE','LE','LO','L','LE','LO','LO','LO','G','L','LE','LO','LO','LE','G','LE','LE','LO','LO','L','LE','LO','L','LE','LE','L','LE','LO','L','LE','LE','LO','L','LE','LE','L']}
            df_obs_attr = pd.DataFrame(d_obs_attr)
            
            
            df_change_list = pd.concat([df_sub_id, df_change_list_loc, df_obs_attr], axis=1)
            # topo_df_per_sub = pd.DataFrame(dict_)
            # topo_df = topo_df.append(dopo_df_per_sub)
            
        return df_change_list
    
    
    def change_loc(self,obs_before_action=None, obs_after_action=None):
        import numpy as np
        
        for i in range(len(obs_after_action)):
            if obs_before_action[i] != obs_after_action[i]:
                loc_change = np.where(obs_after_action != obs_before_action)[0]
                
        return loc_change
    
    
    
    def topo_change_check(self,location_change = None):

        
        change_list= np.array([])
        c_list=np.array([])
        
        for sub_id in range(0,14):
            dict_ = all_obs[0].get_obj_connect_to(substation_id=sub_id)
            # print("There are {} elements connected to this substation (not counting shunt)".format(
            #           dict_["nb_elements"]))
            # print("The names of the loads connected to substation {} are: {}".format(
            #            sub_id, all_obs[0].name_load[dict_["loads_id"]]))
            # print("The names of the generators connected to substation {} are: {}".format(
            #            sub_id, all_obs[0].name_gen[dict_["generators_id"]]))
            # print("The powerline whose origin end is connected to substation {} are: {}".format(
            #            sub_id, all_obs[0].name_line[dict_["lines_or_id"]]))
            # print("The powerline whose extremity end is connected to substation {} are: {}".format(
            #            sub_id, all_obs[0].name_line[dict_["lines_ex_id"]]))
            # print("The storage units connected to substation {} are: {}".format(
            #            sub_id, all_obs[0].name_line[dict_["storages_id"]]))
            # print(f"_____________substation{sub_id} ends________________")
            
            sub_num = [sub_id]
            load_num = dict_['loads_id']
            gen_num = dict_['generators_id']
            line_or_num =dict_["lines_or_id"]
            line_ex_num = dict_["lines_ex_id"]
            
            change_list_loc = [line_ex_num, line_or_num, gen_num, load_num ]
            change_list_loc_new = np.concatenate([x for x in change_list_loc if len(x)>0])
            
            change_list_final = np.append(change_list, change_list_loc_new)
            change_list=change_list_final
            
            topo_vect_list = change_list  # we made the list of topology vector
            
            df_change_list_loc = pd.DataFrame(change_list_final, columns=['element_id'])
            df_sub_id = pd.DataFrame(np.array([0,0,0,1,1,1,1,1,1,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,7,7,8,8,8,8,8,9,9,9,10,10,10,11,11,11,12,12,12,12,13,13,13]), columns = ['sub_id'])
            
            d_obs_attr = {'obs_attr': ['LO', 'LO','G','LE','LO','LO','LO','G','L','LE','LO','G','L','LE','LE','LO','LO','LO','L','LE','LE','LE','LO','L','LE','LO','LO','LO','G','L','LE','LO','LO','LE','G','LE','LE','LO','LO','L','LE','LO','L','LE','LE','L','LE','LO','L','LE','LE','LO','L','LE','LE','L']}
            df_obs_attr = pd.DataFrame(d_obs_attr)
            
            
            df_change_list = pd.concat([df_sub_id, df_change_list_loc, df_obs_attr], axis=1)
            
            df = df_change_list
            
        print(df.iloc[location_change])
            
        return df.iloc[location_change]
    
        
        
        
        
        
        
if __name__ == "__main__" :
    
    
    path_folder = "C:\\Users\\thoug\\OneDrive\\SS2023\\Internship\\02_G2OP\\File"
    
    file_name = 'obj_list.data'
    
    
    a=TOPO_Study(PATH=path_folder)
    # a.path_read(file_name)
    # a.ep_read()
    all_obs= a.obs_read()
    read_obj=a.data_read(a.path_read(file_name))
    
    # topo_tb = a.topo_table(all_obs)

    loc_change = a.change_loc(read_obj[0][2][0], read_obj[0][3][0])
    
    change_location = a.topo_change_check(loc_change)
        
        
        
        
        