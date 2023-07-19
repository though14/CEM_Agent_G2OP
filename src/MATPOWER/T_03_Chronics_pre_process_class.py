# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 21:35:57 2023

@author: thoug
"""

import os
import sys
import numpy as np
from tqdm import tqdm

import pandas as pd
import csv



class preprocess():
    def __init__(self, FOLDER_PATH=None):
        self.folder_path = FOLDER_PATH
        # self.file_name = ['load_p.csv.bz2.csv__mod.csv', 'load_q.csv.bz2.csv__mod.csv', 'prod_p.csv.bz2.csv__mod.csv', 'prod_v.csv.bz2.csv__mod.csv' ]
        self.file_name = ['load_p.csv.bz2.csv__mod.csv',  'prod_p.csv.bz2.csv__mod.csv' ]
        
        
        
    def read_Chronic(self, CHRONIC_NUM=000):
        
        folder_name = str(CHRONIC_NUM).zfill(3)
 
        
        folder_dir = os.listdir(self.folder_path)
        file_name_all = os.listdir(os.path.join(self.folder_path, '000'))
        

                
        
        # for folder_name in folder_dir:
            #in case, for-loop of j needs to be indented
            
        for j in self.file_name:
            
            path_csv = os.path.join(self.folder_path, folder_name)
            
            
            if j == 'load_p.csv.bz2.csv__mod.csv':
            
                data_read_load = pd.read_csv(os.path.join(path_csv, j))
                
            else:
                data_read_gen = pd.read_csv(os.path.join(path_csv, j))
            
    
        return data_read_load, data_read_gen
    
    
    
    def process_Chronic(self, DATA_READ_LOAD=None, DATA_READ_GEN=None):
        
        data_load = []
        data_gen = []
        data_load_minus_gen = []
        
        read_data_load = DATA_READ_LOAD
        read_data_gen = DATA_READ_GEN
        
        SUM_Load = read_data_load.sum(axis=1, skipna=True, numeric_only=True)
        SUM_Gen = read_data_gen.sum(axis=1, skipna=True, numeric_only=True)
        
        SUM_Load_minus_Gen = SUM_Load - SUM_Gen
        
        
        
        return SUM_Load, SUM_Gen, SUM_Load_minus_Gen
    
    
    def max_min_one_Chronic(self, LOAD=None, GEN=None, Minus=None):
        

        tmp_load = LOAD
        tmp_gen = GEN
        tmp_minus = Minus
        
        # tmp_max_load_value=tmp_load[0][0].max()
        # tmp_max_gen_value=tmp_gen[0][0].max()
        # tmp_max_minus_value=tmp_minus[0][0].max()
        # tmp_min_minus_value=tmp_minus[0][0].min()
        
        # tmp_max_load_loc = tmp_load[0][0].idxmax()
        # tmp_max_gen_loc = tmp_gen[0][0].idxmax()
        # tmp_max_minus_loc = tmp_minus[0][0].idxmax()
        # tmp_min_minus_loc = tmp_minus[0][0].idxmin()
        
        tmp_max_load_value=tmp_load.max()
        tmp_max_gen_value=tmp_gen.max()
        tmp_max_minus_value=tmp_minus.max()
        tmp_min_minus_value=tmp_minus.min()
        
        tmp_max_load_loc = tmp_load.idxmax()
        tmp_max_gen_loc = tmp_gen.idxmax()
        tmp_max_minus_loc = tmp_minus.idxmax()
        tmp_min_minus_loc = tmp_minus.idxmin()
        
        return tmp_max_load_value, tmp_max_gen_value, tmp_max_minus_value, tmp_min_minus_value
        
        
        
        
                
                
                





if __name__ == "__main__":
    
    """
    to use it better, probably better to connect C01_CSV_to_Xlxs.py since then it will finish all pre-process.
    Maybe have to merge two files into one?? hmm.
    """
    
    
    
    FOLDER_LOCATION = "C:/Users/thoug/OneDrive/SS2023/Internship/02_G2OP/File/rte_chronics"
    
    a = preprocess(FOLDER_PATH= FOLDER_LOCATION)
    
    DATA_READ_LOAD, DATA_READ_GEN = a.read_Chronic()
    
    d_gen =[]
    d_load = []
    d_minus = []
    
    max_gen=[]
    max_load=[]
    max_minus=[]
    min_minus=[]
    
    for i in tqdm(range(0,1000)):
        DATA_READ_LOAD, DATA_READ_GEN = a.read_Chronic(i)
        
        w_load, w_gen, w_minus = a.process_Chronic(DATA_READ_LOAD, DATA_READ_GEN)
        
        d_gen.append(w_gen)
        d_load.append(w_load)
        d_minus.append(w_minus)   # this d_series are only to gather info of all w_series
        
        
        max_min = a.max_min_one_Chronic(w_load, w_gen, w_minus)
        
        max_load.append(max_min[0])
        max_gen.append(max_min[1])
        max_minus.append(max_min[2])
        min_minus.append(max_min[3])
        
        
    # max_load.max()
    # max_load.idxmax()
    
    # max_gen.max()
    # max_gen.idxmax()
    
    # max_minus.max()
    # max_minus.idxmax()
    
    # min_minus.min()
    # min_minus.idxmin()
    
    sr_max_load = pd.Series(max_load)
    sr_max_gen = pd.Series(max_gen)
    sr_max_minus=pd.Series(max_minus)
    sr_min_minus=pd.Series(min_minus)
    
    loc_max_load = sr_max_load.idxmax()
    val_max_load = sr_max_load.max()
    
    loc_max_gen = sr_max_gen.idxmax()
    val_max_gen = sr_max_gen.max()
    
    loc_max_minus = sr_max_minus.idxmax()
    val_max_minus = sr_max_minus.max()
    
    loc_min_minus = sr_min_minus.idxmin()
    val_min_minus = sr_min_minus.min()
    
    summary = pd.DataFrame()
    summary['gen_max_Chronics'] = [loc_max_gen]
    summary['load_max_Chronics'] = [loc_max_load]
    summary['Load_is_bigger_than_Gen_max'] = [loc_max_minus]
    summary['Gen_is_bigger_than_Load_max'] = [loc_min_minus]        
    
    """
    Please check summary variables, it has the number of Chronics that has each tab value
    """
        
        
        
    
    