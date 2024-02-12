# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:40:20 2023


"""

import os
import re
import numpy as np
import pandas as pd


def get_filename_list(path, data_name):
    files_list =[]
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name = file.split('.')[0]                                # 去除文件名后缀
            if data_name in file_name:
        
                patch_dist_rate = root.split('\\')[-1].split('=')[1]
                disp_among_rate = root.split('\\')[-2].split('-')[0].split('=')[1]
                disp_within_rate = root.split('\\')[-2].split('-')[1].split('=')[1]
                patch_num = root.split('\\')[-3].split('=')[1]
                reproduce_mode = root.split('\\')[-4]
                rep = root.split('\\')[-5].split('=')[1]
                scenario = root.split('\\')[-6]
                
                file_path = root+'\\'+file
                files_list.append((scenario, reproduce_mode, patch_num, disp_among_rate, disp_within_rate, patch_dist_rate, rep, file_path))
                
                #print(reproduce_mode, patch_num, is_heterogeneity, disp_among_rate, disp_within_rate, patch_dist_rate, rep, file_path)

    files_list.sort(key=(lambda x:int(x[6])))
    files_list.sort(key=(lambda x:x[5]))
    files_list.sort(key=(lambda x:x[4]))
    files_list.sort(key=(lambda x:x[3]))      
    files_list.sort(key=(lambda x:x[2]))
    files_list.sort(key=(lambda x:x[1]))
    files_list.sort(key=(lambda x:x[0]))

    return files_list
                
def inverse_Simpson_index(sp_num_array):
    N = sp_num_array.sum()
    inverse_simpson_index = 0
    for sp_num in sp_num_array:
        inverse_simpson_index += (sp_num*(sp_num-1))/(N*(N-1))
    return 1/inverse_simpson_index

def Shannon_diversity(sp_num_array):
    N = sp_num_array.sum()
    shannon_diversity = 0
    for sp_num in sp_num_array:
        shannon_diversity += sp_num/N * np.log(sp_num/N)
    return -1*shannon_diversity

def Hill_number(sp_num_array, a):
    N = sp_num_array.sum()
    if N==0: return 0
    if a==1: return np.exp(Shannon_diversity(sp_num_array))
    hill_num = 0
    for sp_num in sp_num_array:
        hill_num += np.power(sp_num/N, a)
    return np.power(hill_num, 1/(1-a))

def habitat_diversity(df, time_step, patch_num, hab_num, hill_index=2):
    patch_diversity_list = []
    for patch_id in ['patch%d' % (i+1) for i in range(patch_num)]:
        for habitat_id in ['h%d' % i for i in range(hab_num)]:
            species_num_array = df.loc[time_step][patch_id][habitat_id].value_counts().to_numpy()
            try:
                #habitat_diversity = inverse_Simpson_index(species_num_array)
                habitat_diversity = Hill_number(sp_num_array=species_num_array, a=hill_index)
            except:
                habitat_diversity = np.nan
            patch_diversity_list.append(habitat_diversity)
    return np.array(patch_diversity_list).reshape(patch_num*hab_num,1)

def patch_diversity(df, time_step, patch_num, hill_index=2):
    patch_diversity_list = []
    for patch_id in ['patch%d' % (i+1) for i in range(patch_num)]:
        species_num_array = df.loc[time_step][patch_id].value_counts().to_numpy()
        try:
            #patch_diversity = inverse_Simpson_index(species_num_array)
            patch_diversity = Hill_number(sp_num_array=species_num_array, a=hill_index)
        except:
            patch_diversity = np.nan
        patch_diversity_list.append(patch_diversity)
    return np.array(patch_diversity_list).reshape(patch_num,1)

def global_diversity(df, time_step, hill_index=2):
    species_num_array = df.loc[time_step].value_counts().to_numpy()
    try:
        global_diversity = Hill_number(sp_num_array=species_num_array, a=hill_index)
    except:
        global_diversity = np.nan
    return np.array([global_diversity]).reshape(-1,1)

def cumulative_species_richness_curve(df, time_step):
    cumulative_species = []
    new_species_area_index = []
    index = 1
    for sp_id in df.loc[time_step].to_numpy():
        if sp_id not in cumulative_species and not np.isnan(sp_id):

            cumulative_species.append(sp_id)
            new_species_area_index.append(index)
        else:
            pass
        index += 1
    return np.array(new_species_area_index).reshape(-1,1)

def hstack_fillup(array_1, array_2):
    
    if array_1.shape[0]>array_2.shape[0]:
        array_2 = np.vstack((array_2, np.nan*np.ones((array_1.shape[0]-array_2.shape[0],array_2.shape[1]))))
    elif array_1.shape[0]<array_2.shape[0]:
        array_1 = np.vstack((array_1, np.nan*np.ones((array_2.shape[0]-array_1.shape[0], array_1.shape[1]))))
        
    return np.hstack((array_1, array_2))



####################################################################################################################
files_list = get_filename_list(path='G:\\[data]_MetaIBM_v2.9.12_examples', data_name='meta_species_distribution_all_time')




header = [np.array([]),    # scenario
          np.array([]),    # reproduce_mode
          np.array([]),    # patch_num 
          np.array([]),    # disp_among_rate
          np.array([]),    # disp_within_rate
          np.array([]),    # patch_dist_rate
          np.array([])]    # rep

species_index = ['sp%d'%i for i in range(1,101)]
patch_index = ['patch%d'%i for i in range(1,101)]
patch_habitat_index=pd.MultiIndex.from_arrays([['patch%d'%i for i in range(1,101) for j in range(4)],['h0', 'h1', 'h2', 'h3']*100])
global_index = ['global']

all_PD_hill0, all_PD_hill1, all_PD_hill2 = np.empty((0,0)), np.empty((0,0)), np.empty((0,0))
all_HD_hill0, all_HD_hill1, all_HD_hill2 = np.empty((0,0)), np.empty((0,0)), np.empty((0,0))
all_GD_hill0, all_GD_hill1, all_GD_hill2 = np.empty((0,0)), np.empty((0,0)), np.empty((0,0))
all_CSR = np.empty((0,0))



counter = 0
for file in files_list:
    counter += 1
    print(counter, file[0], file[1], file[2], file[3], file[4], file[5], file[6])
    
    header = [np.append(header[0], file[0]),       # scenario
              np.append(header[1], file[1]),       # reproduce_mode
              np.append(header[2], file[2]),       # patch_num 
              np.append(header[3], file[3]),       # disp_among_rate
              np.append(header[4], file[4]),       # disp_within_rate
              np.append(header[5], file[5]),       # patch_dist_rate
              np.append(header[6], file[6])]       # rep  
    
    file_name = file[7]
    df = pd.read_csv(file_name, compression='gzip', header=[0,1,2], index_col=[0], skiprows=lambda x: x>=3 and x<=5002)
    
    habitat_diversity_res_hill0 = habitat_diversity(df, time_step='time_step4999', patch_num=int(file[2]), hab_num=4, hill_index=0)
    habitat_diversity_res_hill1 = habitat_diversity(df, time_step='time_step4999', patch_num=int(file[2]), hab_num=4, hill_index=1)
    habitat_diversity_res_hill2 = habitat_diversity(df, time_step='time_step4999', patch_num=int(file[2]), hab_num=4, hill_index=2)
    
    patch_diversity_res_hill0 = patch_diversity(df, time_step='time_step4999', patch_num=int(file[2]), hill_index=0)
    patch_diversity_res_hill1 = patch_diversity(df, time_step='time_step4999', patch_num=int(file[2]), hill_index=1)
    patch_diversity_res_hill2 = patch_diversity(df, time_step='time_step4999', patch_num=int(file[2]), hill_index=2)

    global_diversity_res_hill0 = global_diversity(df, time_step='time_step4999', hill_index=0)
    global_diversity_res_hill1 = global_diversity(df, time_step='time_step4999', hill_index=1)
    global_diversity_res_hill2 = global_diversity(df, time_step='time_step4999', hill_index=2)
    
    cumulative_species_richness_curve_res = cumulative_species_richness_curve(df, time_step='time_step4999')
    
    all_PD_hill0 = hstack_fillup(all_PD_hill0, patch_diversity_res_hill0)
    all_PD_hill1 = hstack_fillup(all_PD_hill1, patch_diversity_res_hill1)
    all_PD_hill2 = hstack_fillup(all_PD_hill2, patch_diversity_res_hill2)
    all_HD_hill0 = hstack_fillup(all_HD_hill0, habitat_diversity_res_hill0)
    all_HD_hill1 = hstack_fillup(all_HD_hill1, habitat_diversity_res_hill1)
    all_HD_hill2 = hstack_fillup(all_HD_hill2, habitat_diversity_res_hill2)
    all_GD_hill0 = hstack_fillup(all_GD_hill0, global_diversity_res_hill0)
    all_GD_hill1 = hstack_fillup(all_GD_hill1, global_diversity_res_hill1)
    all_GD_hill2 = hstack_fillup(all_GD_hill2, global_diversity_res_hill2)
    all_CSR = hstack_fillup(all_CSR, cumulative_species_richness_curve_res)
    
    #print(patch_diversity_res_hill0.shape, patch_diversity_res_hill1.shape, patch_diversity_res_hill2.shape)
    #print(habitat_diversity_res_hill0.shape, habitat_diversity_res_hill1.shape, habitat_diversity_res_hill2.shape)
    #print(cumulative_species_richness_curve_res.shape)
    
df_all_PD_hill0 = pd.DataFrame(all_PD_hill0, index=patch_index, columns=header)
df_all_PD_hill1 = pd.DataFrame(all_PD_hill1, index=patch_index, columns=header)
df_all_PD_hill2 = pd.DataFrame(all_PD_hill2, index=patch_index, columns=header)
df_all_HD_hill0 = pd.DataFrame(all_HD_hill0, index=patch_habitat_index, columns=header)
df_all_HD_hill1 = pd.DataFrame(all_HD_hill1, index=patch_habitat_index, columns=header)
df_all_HD_hill2 = pd.DataFrame(all_HD_hill2, index=patch_habitat_index, columns=header)
df_all_GD_hill0 = pd.DataFrame(all_GD_hill0, index=global_index, columns=header)
df_all_GD_hill1 = pd.DataFrame(all_GD_hill1, index=global_index, columns=header)
df_all_GD_hill2 = pd.DataFrame(all_GD_hill2, index=global_index, columns=header)
df_all_CSR = pd.DataFrame(all_CSR, index=species_index, columns=header)
    
df_all_PD_hill0.columns.names = ['scenario', 'reproduce_mode', 'patch_num', 'disp_among_rate', 'disp_within_rate', 'patch_dist_rate', 'rep']
df_all_PD_hill1.columns.names = ['scenario', 'reproduce_mode', 'patch_num', 'disp_among_rate', 'disp_within_rate', 'patch_dist_rate', 'rep']
df_all_PD_hill2.columns.names = ['scenario', 'reproduce_mode', 'patch_num', 'disp_among_rate', 'disp_within_rate', 'patch_dist_rate', 'rep']
df_all_HD_hill0.columns.names = ['scenario', 'reproduce_mode', 'patch_num', 'disp_among_rate', 'disp_within_rate', 'patch_dist_rate', 'rep']
df_all_HD_hill1.columns.names = ['scenario', 'reproduce_mode', 'patch_num', 'disp_among_rate', 'disp_within_rate', 'patch_dist_rate', 'rep']
df_all_HD_hill2.columns.names = ['scenario', 'reproduce_mode', 'patch_num', 'disp_among_rate', 'disp_within_rate', 'patch_dist_rate', 'rep']
df_all_GD_hill0.columns.names = ['scenario', 'reproduce_mode', 'patch_num', 'disp_among_rate', 'disp_within_rate', 'patch_dist_rate', 'rep']
df_all_GD_hill1.columns.names = ['scenario', 'reproduce_mode', 'patch_num', 'disp_among_rate', 'disp_within_rate', 'patch_dist_rate', 'rep']
df_all_GD_hill2.columns.names = ['scenario', 'reproduce_mode', 'patch_num', 'disp_among_rate', 'disp_within_rate', 'patch_dist_rate', 'rep']
df_all_CSR.columns.names = ['scenario', 'reproduce_mode', 'patch_num', 'disp_among_rate', 'disp_within_rate', 'patch_dist_rate', 'rep']

df_all_PD_hill0.to_csv('all_patch_diversity_hill0.csv')   
df_all_PD_hill1.to_csv('all_patch_diversity_hill1.csv')  
df_all_PD_hill2.to_csv('all_patch_diversity_hill2.csv')  
df_all_HD_hill0.to_csv('all_habitat_diversity_hill0.csv')  
df_all_HD_hill1.to_csv('all_habitat_diversity_hill1.csv')  
df_all_HD_hill2.to_csv('all_habitat_diversity_hill2.csv')  
df_all_GD_hill0.to_csv('all_global_diversity_hill0.csv')  
df_all_GD_hill1.to_csv('all_global_diversity_hill1.csv')  
df_all_GD_hill2.to_csv('all_global_diversity_hill2.csv')  
df_all_CSR.to_csv('all_culmulative_species_richness_curves.csv')  

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

