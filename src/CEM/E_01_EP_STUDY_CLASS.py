# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:22:22 2023

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
from tqdm import tqdm

from A01_01_Main_Prep import env_sandbox as env_out
from A01_01_Main_Prep import env_sandbox_test as env_test
from A01_01_Main_Prep import env_sandbox_test_no_p as env_sand_test


class TopoStudy():
    def __init__(self, 
                 PATH_FOLDER_INCLUDING_AGENT ="C:\\Users\\thoug\\OneDrive\\SS2023\\Internship\\04_Code\\CEM_Agent_G2OP\\File\\Agent\\July_Sandbox",
                 NAME_OF_FOLDER_WANT_TO_SEE = "save_5"
                 ):
        self.path_name = PATH_FOLDER_INCLUDING_AGENT
        self.path_to_load = os.path.join(self.path_name, NAME_OF_FOLDER_WANT_TO_SEE)
        self.episode_studied_res = EpisodeData.list_episode(self.path_to_load)
        self.ep_length = len(self.episode_studied_res)
        
        # self.ep = EpisodeData.from_disk(*episode_studied_res[0])
        # self.eos = self.ep.observations
        # self.eac = self.ep.actions
        
    
    def obs_read(self, NUM=0):
        # EpisodeData.reboot(self)
        ep = EpisodeData.from_disk(*self.episode_studied_res[NUM])
        
        eos = ep.observations
        
        all_obs = [el for el in eos] 
        
        return all_obs
    

        
    
    def list_obs(self):
        
        list_obs = []
        list_obs_loc = []
        
        for i in range(self.ep_length):
            ep = EpisodeData.from_disk(*self.episode_studied_res[i])
            eos = ep.observations
            eac = ep.actions
            
            
            
            list_act = []
            for act in eac:
                list_act.append(act.as_dict())
            all_obs = [el for el in eos]

            ep_name = i
            for j in range(len(all_obs)):
                if j == 0:
                    pass
                elif np.array_equal(all_obs[j-1].topo_vect, all_obs[j].topo_vect) == False:
                    ts_obs_change = j
                    obs_before = all_obs[j-1].topo_vect
                    obs_after = all_obs[j].topo_vect
                    list_obs_loc = [ep_name, ts_obs_change, [obs_before], [obs_after]]
                    list_obs.append(list_obs_loc)
                    
        return list_obs
                    
                    
    def save_list_obs(self, LIST_OBS, NAME_SAVE ='obj_list.data' ):
        
        import pickle
        
        with open(NAME_SAVE, 'wb') as filehandle:
            pickle.dump(LIST_OBS, filehandle)
            
            
            
    def load_list_obs(self, NAME_OF_LIST_OBS_SAVED ='obj_list.data'):
        
        import pickle
        
        with open('obj_list.data', 'rb') as filehandle:
            list_obs_from_disk = pickle.load(filehandle)
            
            
        return list_obs_from_disk
    
    
    
    def convert_list_obs_to_DF(self, LIST_OBS):
        
        df_ro = pd.DataFrame(LIST_OBS, columns=['ep_num','TS','before','after'])
        
        return df_ro
        
    
    def topo_table(self, ALL_OBS= None):
        
        ALL_OBS = self.obs_read(0)
        
        change_list= np.array([])
        c_list=np.array([])

        for sub_id in tqdm(range(0,14)):
            dict_ = ALL_OBS[0].get_obj_connect_to(substation_id=sub_id)

            
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
    
    
    def topo_change_check(self, TOPO_TABLE=None, location_change = None, PRINT=None):
        # all_obs = ALL_OBS
        
        TOPO_TABLE = self.topo_table()
        
        df = TOPO_TABLE
        
        
        if PRINT == True:
            
            print('---')
            # print(f'change of the episode number: {}')
            print(df.iloc[location_change])
            print('---')
            
        else:
            pass
        
        return df.iloc[location_change]
    
    
    def all_topology(self, ALL_OBS):
        
        all_obs = ALL_OBS
        len_all_obs = len(all_obs)
        
        all_topo=[]
        
        for i in range(len_all_obs):
            loc_topo=[]
            
            loc_topo = all_obs[i].topo_vect
            
            # loc_topo = np.where(loc_topo<0.5, 0,1)
            
            all_topo.append(loc_topo)
            
            
        all_topo_df = pd.DataFrame(all_topo)
            
        return all_topo_df
    
    
    
    def per_ep_change_check(self, DATAFRAME_Read_OBJ= None,  HOW_MANY_CHRONICS=1000 ):
        
        df_ro = DATAFRAME_Read_OBJ
        
        # df_init = pd.DataFrame()
        all_per_ep_change = pd.DataFrame()
        
        for EP_NUM in tqdm(range(self.ep_length)):     #k can go till 999 == range(0,1000)  #for now let's put k=8, since it has 2 values
            
            
            df_ro_local = df_ro[df_ro['ep_num']==EP_NUM] #currently, k = episode number
            df_ro_local=df_ro_local.reset_index(drop=True)

            df_ro_local
            
            all_per_ep_change_local=pd.DataFrame()
            
            for q in range(len(df_ro_local)):
                local_loc_change = self.change_loc(df_ro_local.loc[q]['before'][0], df_ro_local.loc[q]['after'][0])
                
                local_topo_change = self.topo_change_check(location_change=local_loc_change)
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
            epr = EpisodeData.from_disk(*self.episode_studied_res[EPISODE_NUMBER])
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
    
    
    def sub_bus_nr(self, ALL_OBS_LENGTH):
        
        all_obs = self.obs_read()
        sub_bus_nr = []
        for j in range(ALL_OBS_LENGTH):
            eos = all_obs[j]
            sub_bus_nr_loc=[]
            for i in range(0,14):
                sub_bus_nr_loc.append(eos.sub_topology(i))
            sub_bus_nr.append(sub_bus_nr_loc)
           
        return sub_bus_nr
    
    
    
    def analyze_pattern(self,pattern):
        unique_patterns = {}
        pattern_counts = {}
        order_of_changes = []
        base_pattern = None

        for row in pattern:
            pattern_str = str(row)
            if pattern_str not in unique_patterns:
                if base_pattern is None:
                    pattern_name = 'b'
                    base_pattern = pattern_str
                else:
                    pattern_name = 't' + str(len(unique_patterns))
                unique_patterns[pattern_str] = pattern_name
                pattern_counts[pattern_name] = 1
                order_of_changes.append(pattern_name)
            else:
                pattern_name = unique_patterns[pattern_str]
                if order_of_changes[-1] != pattern_name:
                    pattern_counts[pattern_name] += 1
                    order_of_changes.append(pattern_name)
                else :
                    pattern_counts[pattern_name] +=1

        # Count occurrences of the base pattern
        base_pattern_count = pattern_counts.get('b', 0)

        return unique_patterns, pattern_counts, order_of_changes, base_pattern_count
    
    
    
    def KPI(self, PER_EP_CHANGE):
        
        per_ep_change = PER_EP_CHANGE
        
        per_ep_topo_change_nr=per_ep_change['ep_num'].value_counts()
        
        
        # loc_depth = per_ep_change['TS'].diff()
        
        loc_depth = per_ep_change.groupby('ep_num')['TS'].diff()
        
        per_ep_change['local_depth']=loc_depth
        
        mean_topo_depth=[]  #it maybe actually mean sequence length
        
        for i in range(self.ep_length):
            sum_depth = per_ep_change.loc[per_ep_change['ep_num'] == i, 'local_depth'].sum()
            loc_mean_topo_depth=sum_depth/per_ep_topo_change_nr[i]
            mean_topo_depth.append(loc_mean_topo_depth)
            
        mean_topo_depth
        
        
                
        
        
        
        return per_ep_topo_change_nr, mean_topo_depth
    
        
        
        
    

    
    
if __name__ == "__main__" :
    
    # EpisodeData.reboot()
    env = grid2op.make('l2rpn_case14_sandbox_test')
    env_sand_test
    
    path_name = "C:\\Users\\thoug\\OneDrive\\SS2023\\Internship\\04_Code\\CEM_Agent_G2OP\\File\\Agent\\July_Sandbox"
    path_to_save = os.path.join(path_name, 'save_2')
    
    a = TopoStudy(path_name, path_to_save)
#%%    
    all_obs = a.obs_read(0)
    
    list_obs = a.list_obs()
    

    
    a.save_list_obs(list_obs)
    
    list_obs_read = a.load_list_obs()
    
    df_list_obs = a.convert_list_obs_to_DF(list_obs_read)
    
    topo_table = a.topo_table(all_obs)
    
    change_loc = a.change_loc(list_obs[0][2][0],list_obs[0][3][0])
    
    topo_change = a.topo_change_check(location_change=change_loc)
    
    per_ep_change = a.per_ep_change_check(DATAFRAME_Read_OBJ=df_list_obs )
    
    pattern = a.find_pattern(per_ep_change['changes'][0])
    
    for_Survived_data = pd.DataFrame()
    for k in range(0,2):
        for_survived_step = a.survival_length(EPISODE_NUMBER=k)
       
        for_Survived_data.join(for_survived_step)

    sub_bus_nr = a.sub_bus_nr(len(all_obs))
     
    
#%% 
    nr_topo_changes, mean_sequence_length = a.KPI(per_ep_change)
    
    all_topo_list = a.all_topology(all_obs)
    
    pattern_list = a.analyze_pattern(all_topo_list.to_numpy())
    
    
    """
    for i in range(0,14):
        print(f'sub_{i} Topology : {eos.sub_topology(i)}')
    
    #################################
    
    
    for i in range(0,14):
        print(f'sub_{i} Topology_Connection :')
        print(f'{eos.get_obj_substations(substation_id=i)}')
        
        
    #################################
        
    #to make list of substation 
    
    sub_bus_nr=[]
    for j in range(800,820):
        eos = all_obs[j]
        sub_bus_nr_loc=[]
        for i in range(0,14):
            
            print(f'sub_{i} Topology : {eos.sub_topology(i)}')
            sub_bus_nr_loc.append(eos.sub_topology(i))
        sub_bus_nr.append(sub_bus_nr_loc)
    print('-------------------------')
    
    #################################
    """
        