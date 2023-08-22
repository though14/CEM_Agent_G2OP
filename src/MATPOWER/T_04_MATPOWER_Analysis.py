# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 23:37:41 2023

@author: thoug
"""
import numpy as np
import mat73
import pandas as pd
import scipy.io as sio
import re


data_dict = mat73.loadmat('20230808_134002_case14realistic_197_rd.mat')

shorten_dict = sio.loadmat('BR_STATUS_Relevant_r14_197.mat')


Topo = data_dict['BR_STATUS']
Topo = np.absolute(Topo)
Topo = np.transpose(Topo)
Topo = np.where(Topo<0.5, 0,1)


# w_TOPO = shorten_dict['BR_STATUS_Relavant_w_BBDC']
# w_TOPO = np.absolute(w_TOPO)
# w_TOPO = np.transpose(w_TOPO)
# w_TOPO = np.where(w_TOPO<0.5, 0,1)

w_TOPO = shorten_dict['BR_STATUS_Work']
w_TOPO = np.absolute(w_TOPO)
w_TOPO = np.transpose(w_TOPO)
# w_TOPO = np.where(w_TOPO<0.5, 0,1)
w_TOPO = np.where(np.isnan(w_TOPO), -1, np.where(w_TOPO < 0.5, 0, 1))


#in case -1 (NAN = not converged) happen
if np.where(w_TOPO == -1)[0] is not None:
    index = np.where(w_TOPO == -1)[0]
    if index.size > 0:
        w_TOPO = w_TOPO[:index[0] + 1]
    

w_TOPO





# wo_TOPO = shorten_dict['BR_STATUS_Relavant_wo_BBDC_right_load']
# wo_TOPO = np.absolute(wo_TOPO)
# wo_TOPO = np.transpose(wo_TOPO)
# wo_TOPO = np.where(wo_TOPO<0.5, 0,1)





# def count_pattern_occurrences(patterns):
#     pattern_counts = {}
#     pattern_order = []
#     alphabet_index = 0

#     for pattern in patterns:
#         pattern_str = chr(ord('a') + alphabet_index)

#         if pattern_str in pattern_counts:
#             pattern_counts[pattern_str] += 1
#         else:
#             pattern_counts[pattern_str] = 1

#         pattern_order.append(pattern_str)
#         alphabet_index += 1

#     return pattern_counts, pattern_order


# def analyze_pattern1(pattern):
#     unique_patterns = []
#     pattern_counts = {}
#     order_of_changes = []

#     for row in pattern:
#         if str(row) not in unique_patterns:
#             unique_patterns.append(str(row))
#             pattern_counts[str(row)] = 1
#             order_of_changes.append('b')
#         else:
#             pattern_counts[str(row)] += 1
#             order_of_changes.append('t' + str(unique_patterns.index(str(row)) + 1))

#     return unique_patterns, pattern_counts, order_of_changes



# def analyze_patterns2(pattern):
#     unique_patterns = []
#     pattern_counts = {}
#     order_of_changes = []

#     for row in pattern:
#         if str(row) not in unique_patterns:
#             unique_patterns.append(str(row))
#             pattern_counts[str(row)] = 1
#             order_of_changes.append('b')
#         else:
#             if order_of_changes[-1] != 't' + str(unique_patterns.index(str(row)) + 1):
#                 pattern_counts[str(row)] += 1
#                 order_of_changes.append('t' + str(unique_patterns.index(str(row)) + 1))

#     return unique_patterns, pattern_counts, order_of_changes




# def analyze_pattern(pattern):
#     unique_patterns = {}
#     pattern_counts = {}
#     order_of_changes = []

#     for row in pattern:
#         pattern_str = str(row)
#         if pattern_str not in unique_patterns:
#             pattern_name = 't' + str(len(unique_patterns) + 1)
#             unique_patterns[pattern_str] = pattern_name
#             pattern_counts[pattern_name] = 1
#             order_of_changes.append('b')
#         else:
#             pattern_name = unique_patterns[pattern_str]
#             if order_of_changes[-1] != pattern_name:
#                 pattern_counts[pattern_name] += 1
#                 order_of_changes.append(pattern_name)

#     return unique_patterns, pattern_counts, order_of_changes



def analyze_pattern(pattern):
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


Pattern = analyze_pattern(Topo)
pattern_to_topology = {value: key for key, value in Pattern[0].items()}
pattern_to_topology = pd.DataFrame.from_dict(pattern_to_topology, orient='index')
pattern_to_topology[0].str.replace('\W','')


Pattern_pre_processed = analyze_pattern(w_TOPO)
unique_topo = Pattern_pre_processed[0]
unique_topo = {value: key for key, value in unique_topo.items()}
for key in unique_topo:
    value = unique_topo[key]
    value = re.findall(r'\d+', value)  # Extract numeric values
    unique_topo[key] = list(map(int, value))  # Convert values to integers
    #it extract only numeric values, so at t4, where everything disconnected into -1, becomes just 1. 
    #this needs to be fixed

    

df_unique_topo = pd.DataFrame.from_dict(data=unique_topo, orient='index')

# wo_Pattern = analyze_pattern(wo_TOPO)


rep = Pattern[1]
rep_df = pd.DataFrame.from_dict(rep, orient='index')
print(f'the number of total Topology played: {rep_df.sum(0)[0]}')


# with open('unique_key_str.txt') as f:
#     unique_key_value = f.readlines()
    
# unique_key = pd.DataFrame(unique_key_value)
# unique_key[0] = unique_key[0].str.replace('\W','')



TOPO_TABLE_KEY = ['DISC_1_1','DISC_1_2','DISC_1_3','DISC_2_01','DISC_2_02','DISC_2_03','DISC_2_04','DISC_2_05','DISC_2_06','DISC_3_1','DISC_3_2','DISC_3_3','DISC_3_4','DISC_4_01','DISC_4_02','DISC_4_03','DISC_4_04','DISC_4_05','DISC_4_06','DISC_5_01',
'DISC_5_02','DISC_5_03','DISC_5_04','DISC_5_05','DISC_6_01','DISC_6_02','DISC_6_03','DISC_6_04','DISC_6_05','DISC_6_06','DISC_7_1',
'DISC_7_2','DISC_7_3','DISC_8_1','DISC_8_2','DISC_9_01','DISC_9_02','DISC_9_03','DISC_9_04','DISC_9_05','DISC_10_1','DISC_10_2',
'DISC_10_3','DISC_11_1','DISC_11_2','DISC_11_3','DISC_12_1','DISC_12_2','DISC_12_3','DISC_13_1','DISC_13_2','DISC_13_3',
'DISC_13_4','DISC_14_1','DISC_14_2','DISC_14_3']

df_w_TOPO = pd.DataFrame(w_TOPO)
df_w_TOPO.columns = TOPO_TABLE_KEY
df_unique_topo.columns = TOPO_TABLE_KEY

"""
to find the location of pattern_to_topology



for k in range(0,428):
    if k ==0:
        pass
    elif k==427:
        pass
    else:
        if k<75:
            if k%2==1:
                pt=pattern_to_topology.iloc[[1]][0][0][k]
                print(f'{pt}')
        elif 75<=k<150:
            if k%2==0:
                pt=pattern_to_topology.iloc[[1]][0][0][k]
                print(f'{pt}')
        elif 150<=k<225:
            if k%2==1:
                pt=pattern_to_topology.iloc[[1]][0][0][k]
                print(f'{pt}')
        elif 225<=k<300:
            if k%2==0:
                pt=pattern_to_topology.iloc[[1]][0][0][k]
                print(f'{pt}')
        elif 300<=k<375:
            if k%2==1:
               pt=pattern_to_topology.iloc[[1]][0][0][k]
               print(f'{pt}')
        elif 375<=k<427:
            if k%2 == 0:
                pt=pattern_to_topology.iloc[[1]][0][0][k]
                print(f'{pt}')


"""



