# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 23:37:41 2023

@author: thoug
"""
import numpy as np
import mat73
import pandas as pd
import scipy.io as sio


data_dict = mat73.loadmat('case14realistic_514_top.mat')

shorten_dict = sio.loadmat('BR_STATUS_Relevant.mat')


Topo = data_dict['BR_STATUS']
Topo = np.absolute(Topo)
Topo = np.transpose(Topo)
Topo = np.where(Topo<0.5, 0,1)


w_TOPO = shorten_dict['BR_STATUS_Relavant_w_BBDC']
w_TOPO = np.absolute(w_TOPO)
w_TOPO = np.transpose(w_TOPO)
w_TOPO = np.where(w_TOPO<0.5, 0,1)


wo_TOPO = shorten_dict['BR_STATUS_Relavant_wo_BBDC_right_load']
wo_TOPO = np.absolute(wo_TOPO)
wo_TOPO = np.transpose(wo_TOPO)
wo_TOPO = np.where(wo_TOPO<0.5, 0,1)


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


w_Pattern = analyze_pattern(w_TOPO)
wo_Pattern = analyze_pattern(wo_TOPO)


rep = Pattern[1]
rep_df = pd.DataFrame.from_dict(rep, orient='index')
print(f'the number of total Topology played: {rep_df.sum(0)[0]}')


with open('unique_key_str.txt') as f:
    unique_key_value = f.readlines()
    
unique_key = pd.DataFrame(unique_key_value)
unique_key[0] = unique_key[0].str.replace('\W','')

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



