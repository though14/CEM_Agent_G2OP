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
        
        self.episode_studied_res = None
        self.this_ep_res = None
        
    def path_read(self, FILE_NAME=None):
        
        path_info  = os.path.join(self.path_str, FILE_NAME)
        self.path_info = path_info
        
        return path_info
    
    
    def path_result_read(self):
        
        return self.result_path
    
    
    def ep_read(self, EP_NUM_LOC=0):
        
        ep_num = EP_NUM_LOC
        
        # self.ep_num = ep_num #should we update init value?????
        
        return ep_num
    
    
    def data_read(self, PATH_OBJ=None, OUTPUT_FORMAT = 'LIST'):
        
        with open(PATH_OBJ, 'rb') as filehandle:
            read_obj = pickle.load(filehandle)
            
            
        if OUTPUT_FORMAT == 'LIST':
            
            
            return read_obj
        
        elif OUTPUT_FORMAT == 'DF':
        
            df_ro = pd.DataFrame(read_obj, columns=['ep_num','TS','before','after'])
            
            return df_ro
        
            
    
    
    def obs_read(self ):
        episode_studied_res = EpisodeData.list_episode(self.result_path)
        self.episode_studied_res = episode_studied_res
        
        
        this_ep_res = EpisodeData.from_disk(*episode_studied_res[self.ep_num])
        self.this_ep_res = this_ep_res
        
        ep = this_ep_res


        ep = this_ep_res
        eos = ep.observations
        
        all_obs = [el for el in eos]
        
        self.all_obs = all_obs
        
        return all_obs
    
    
    def ep_result(self, EP_NUM=0):
        self.episode_studied_res
        
        self.this_ep_res
        
        return self.this_ep_res
        
    
    
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
    
    
    
    def topo_change_check(self,  location_change = None, PRINT=None):
        # all_obs = ALL_OBS
        
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
        
        if PRINT == True:
            
            print('---')
            # print(f'change of the episode number: {}')
            print(df.iloc[location_change])
            print('---')
            
        else:
            pass
        
        return df.iloc[location_change]
    
        
    
    def per_ep_change_check(self, DATAFRAME_Read_OBJ= None,  HOW_MANY_CHRONICS=1000 ):
        
        df_ro = DATAFRAME_Read_OBJ
        
        # df_init = pd.DataFrame()
        all_per_ep_change = pd.DataFrame()
        
        for EP_NUM in range(0,1000):     #k can go till 999 == range(0,1000)  #for now let's put k=8, since it has 2 values
            
            
            df_ro_local = df_ro[df_ro['ep_num']==EP_NUM] #currently, k = episode number
            df_ro_local=df_ro_local.reset_index(drop=True)

            df_ro_local
            
            all_per_ep_change_local=pd.DataFrame()
            
            for q in range(len(df_ro_local)):
                local_loc_change = self.change_loc(df_ro_local.loc[q]['before'][0], df_ro_local.loc[q]['after'][0])
                
                local_topo_change = self.topo_change_check(local_loc_change)
                local_topo_change = local_topo_change.reset_index(drop=True)
                
                # change_local = [local_topo_change]
                change_local = []
                change_local.append(local_topo_change)
                
                
                

                
                local_ep_num = df_ro_local['ep_num'][q]
                local_TS = df_ro_local['TS'][q]
                
                local_ep_num_df = pd.Series(local_ep_num)
                local_TS_df = pd.Series(local_TS)
                
                local_ep_ts = pd.concat([local_ep_num_df.rename('ep_num'),local_TS_df.rename('TS')], axis=1)
                local_change_df = pd.Series(change_local)
                # local_change_df = local_change_df.reset_index(drop=True)
                
                local_change_df_list=list(local_change_df)
                    
                per_ep_change = pd.concat([local_ep_ts, local_change_df.rename('changes')], axis=1)  #can access change info by: per_ep_change['changes'][0]
                
                all_per_ep_change_local = pd.concat([all_per_ep_change_local, per_ep_change])
            # per_ep_change = local_ep_ts.insert(2,'changes',change_local)
            
            all_per_ep_change = pd.concat([all_per_ep_change, all_per_ep_change_local])
            all_per_ep_change = all_per_ep_change.reset_index(drop=True)
        # all_per_ep_change = per_ep_change_init.append(per_ep_change)
            
        
                
        return all_per_ep_change
    
    
    # def anal_topo_change(self, CHANGED_DATA = None):
    #     data_tba = CHANGED_DATA  #tba stands for to_be_analyzed
    
    def find_pattern(self,df):
        # Combine all columns into a single string column
        combined_data = df.apply(lambda x: ' '.join(x.map(str)), axis=1)
    
        # Count the occurrences of each pattern
        pattern_counts = combined_data.value_counts()
    
        return pattern_counts
    
    
    def survival_length(self, EPISODE_NUMBER = None):
        if EPISODE_NUMBER == None:

            epr = self.ep_result(EP_NUM = self.ep_num)
            eos = epr.observations
            
            chronics_max_step = int(epr.meta['chronics_max_timestep'])
            this_agent_survived_step = eos._game_over
            
            if this_agent_survived_step >= chronics_max_step:
                print('fully survived')
            else:
                print(f'In this episode, Agent survived {this_agent_survived_step}/{chronics_max_step}')
                
        else:
            episode_studied_res = EpisodeData.list_episode(self.result_path)
            epr = EpisodeData.from_disk(*episode_studied_res[EPISODE_NUMBER])
            eos = epr.observations
            
            chronics_max_step = int(epr.meta['chronics_max_timestep'])
            this_agent_survived_step = eos._game_over
            
            # df = pd.DataFrame()
            data = {'EP_MAX_STEP':chronics_max_step, 'AGENT_MAX_STEP': this_agent_survived_step}
            df = pd.DataFrame(data, index=[0])
            
            
                
            
            if this_agent_survived_step >= chronics_max_step:
                print(f'EP{EPISODE_NUMBER}: fully survived {this_agent_survived_step}/{chronics_max_step}')
            else:
                print(f'In this episode{EPISODE_NUMBER}, Agent survived {this_agent_survived_step}/{chronics_max_step}')
                
                
        return df
    
    
    # def survival_matrix(self, df_FROM_survival_length = None):
        
    #     lc_df = df_FROM_survival_length
    #     lc_df_updated = pd.DataFrame()
    #     lc_df_updated.join(lc_df)
        
    #     return lc_df_updated
            
                
        
        
        
        
          
                    
                    

        
        
        
        
#%%        
        
if __name__ == "__main__" :
    
    
    path_folder = "C:\\Users\\thoug\\OneDrive\\SS2023\\Internship\\02_G2OP\\File"
    
    file_name = 'obj_list.data'
    
    
    a=TOPO_Study(PATH=path_folder)
    # a.path_read(file_name)
    # a.ep_read()
    all_obs= a.obs_read()
    read_obj=a.data_read(a.path_read(file_name), 'LIST')
    check_read_obj = a.data_read(a.path_read(file_name), 'DF')
    
    # topo_tb = a.topo_table(all_obs)

    loc_change = a.change_loc(read_obj[0][2][0], read_obj[0][3][0])
    
    change_location = a.topo_change_check(loc_change)
    
    per_ep_change = a.per_ep_change_check(DATAFRAME_Read_OBJ=check_read_obj)
    
    pattern = a.find_pattern(per_ep_change['changes'][0])
    
    # survived_step = a.survival_length()
    for_Survived_data = pd.DataFrame()
    for k in range(0,10):
        for_survived_step = a.survival_length(EPISODE_NUMBER=k)
       
        for_Survived_data.join(for_survived_step)
        
    
    
    
    
    # for i in range(0,100):
    #     ep_num_b = a.ep_read(i)
    #     loc_change_b = a.change_loc(read_obj[ep_num_b][2][0], read_obj[ep_num_b][3][0]) 
    #     change_location_b = a.topo_change_check(loc_change_b)
            
        
#%%
    # df_ro = pd.DataFrame(read_obj, columns=['ep_num','TS','before','after'])
    # for k in range(0,1000):     #k can go till 999 == range(0,1000)  #for now let's put k=8, since it has 2 values
        
    #     if k==8:   
    #         df_ro_local = df_ro[df_ro['ep_num']==k] #currently, k = episode number
    #         df_ro_local=df_ro_local.reset_index(drop=True)

    #         df_ro_local
            
    #         for q in range(len(df_ro_local)):
    #             local_loc_change = a.change_loc(df_ro_local.loc[q]['before'][0], df_ro_local.loc[q]['after'][0])
                
    #             local_topo_change = a.topo_change_check(local_loc_change)
    #             local_topo_change = local_topo_change.reset_index(drop=True)
                
    #             change_local = [local_topo_change]
    #             change_local.append(local_topo_change)
                
                
                

                
    #         local_ep_num = df_ro_local['ep_num']
    #         local_TS = df_ro_local['TS']
            
    #         local_ep_ts = pd.concat([local_ep_num,local_TS], axis=1)
    #         local_change_df = pd.Series(change_local)
            
    #         local_change_df_list=list(local_change_df)
                
    #         per_ep_change = pd.concat([local_ep_ts, local_change_df.rename('changes')], axis=1)  #can access change info by: per_ep_change['changes'][0]
            
    #         # per_ep_change = local_ep_ts.insert(2,'changes',change_local)
    #         # 

            
            
      
                
                
        
    #     else:
    #         pass
        