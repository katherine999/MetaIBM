# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 06:23:20 2023


"""

import os
import re
import math
import numpy as np
import pandas as pd
from distutils.util import strtobool

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

def survival_rate(d, phenotype_ls, env_val_ls, w = 0.5):
    #d is the baseline death rate responding to the disturbance strength.
    #phenotype_ls is a list of phenotype of each trait.
    #env_val_ls is a list of environment value responding to the environment type.
    #w is the width of the fitness function.
        
    survival_rate = (1-d)
    for index in range(len(phenotype_ls)):
        ei = phenotype_ls[index]               # individual phenotype of a trait 
        em = env_val_ls[index]                 # microsite environment value of a environment type
        survival_rate *= math.exp((-1)*math.pow(((ei-em)/w),2))
    return survival_rate

def get_theo_actual_dom_sp(df_actu_sp_dis, patch_id, habitat_id):
    #df_actu_sp_dis = pd.read_csv(file_name, compression='gzip', header=[0,1,2], index_col=[0], skiprows=lambda x: x>=4 and x<=5002)
    
    theo_habitat_sp = df_actu_sp_dis[patch_id][habitat_id].loc['optimun_sp_id_values']
    theo_dom_sp = int(theo_habitat_sp.mode()[0])
    
    try:
        actual_habitat_sp = df_actu_sp_dis[patch_id][habitat_id].loc['time_step4999']
        actu_dom_sp = int(actual_habitat_sp.mode()[0])
    except:
        actu_dom_sp = None
    
    #print(df_actu_sp_dis[patch_id][habitat_id].loc['optimun_sp_id_values'])
    #print(df_actu_sp_dis[patch_id][habitat_id].loc['time_step4999'])
    return theo_dom_sp, actu_dom_sp                  #numpy.float64

def hstack_fillup(array_1, array_2):
    if array_1.shape[0]>array_2.shape[0]:
        array_2 = np.vstack((array_2, np.nan*np.ones((array_1.shape[0]-array_2.shape[0],array_2.shape[1]))))
    elif array_1.shape[0]<array_2.shape[0]:
        array_1 = np.vstack((array_1, np.nan*np.ones((array_2.shape[0]-array_1.shape[0], array_1.shape[1]))))
    return np.hstack((array_1, array_2))

def patch_hab_id_2_patch_hab_id_tran(patch_id, habitat_id):
    ''' input 真实的id; output 转化的id '''
    patch_hab_id_ls = [('patch%d'%(i+1), 'h%d'%j) for i in range(4) for j in range(4)] #转化的id
    patch_hab_id_ls_for_sp_dis = [('patch%d'%(i+1), 'h%d'%j) for i in range(1) for j in range(16)] #真正的id
    
    index = patch_hab_id_ls_for_sp_dis.index((patch_id, habitat_id))
    patch_id_tran, habitat_id_tran = patch_hab_id_ls[index]
    return patch_id_tran, habitat_id_tran
    
##################################################################################################
files_list = get_filename_list(path='G:\\[data]_MetaIBM_v2.9.12_examples', data_name='meta_species_distribution_all_time')

header = [np.array([]),    # scenario
          np.array([]),    # reproduce_mode
          np.array([]),    # patch_num 
          np.array([]),    # disp_among_rate
          np.array([]),    # disp_within_rate
          np.array([]),    # patch_dist_rate
          np.array([])]    # rep

df_all_HD_hill2 = pd.read_csv('all_habitat_diversity_hill2.csv', index_col=[0,1], header=[0,1,2,3,4,5,6])
df_all_PD_hill2 = pd.read_csv('all_patch_diversity_hill2.csv', index_col=[0], header=[0,1,2,3,4,5,6])
df_all_GD_hill2 = pd.read_csv('all_global_diversity_hill2.csv', index_col=[0], header=[0,1,2,3,4,5,6])


counter = 0
for file in files_list:
    counter += 1
    assembly_mechanism = []
    
    scenario = file[0]
    reproduce_mode = file[1]
    patch_num = file[2]
    disp_among_rate = file[3]
    disp_within_rate = file[4]
    patch_dist_rate = file[5]
    rep = file[6]
    file_name = file[7]
    
    hab_num = 4
    
    print(counter, file[0], file[1], file[2], file[3], file[4], file[5], file[6])
    df_actu_sp_dis = pd.read_csv(file_name, compression='gzip', header=[0,1,2], index_col=[0], skiprows=lambda x: x>=4 and x<=5002)
    
    
    header = [np.append(header[0], file[0]),       # scenario
              np.append(header[1], file[1]),       # reproduce_mode
              np.append(header[2], file[2]),       # patch_num
              np.append(header[3], file[3]),       # disp_among_rate
              np.append(header[4], file[4]),       # disp_within_rate
              np.append(header[5], file[5]),       # patch_dist_rate
              np.append(header[6], file[6])]       # rep 
    
    
    global_diversity = df_all_GD_hill2[scenario][reproduce_mode][patch_num][disp_among_rate][disp_within_rate][patch_dist_rate][rep].loc['global']
    global_K = df_actu_sp_dis.loc['optimun_sp_id_values'].value_counts().sum() # global 群落个体数量
    global_N = df_actu_sp_dis.loc['time_step4999'].value_counts().sum()        # global 环境容纳量
    
    print(global_diversity, global_K, global_N)
    
    if 1<= global_diversity <= 1.25 and global_N/global_K >= 0.8:
        assembly_mechanism += ['GM' for i in range(int(patch_num)*int(hab_num))]
        print('global_diversity=', global_diversity, 'GM')
        
    elif global_N/global_K < 0.1:
        assembly_mechanism += ['GE' for i in range(int(patch_num)*int(hab_num))]
        print('global_diversity=', global_diversity, 'GE')
        
    else:
        for patch_id in ['patch%d'%(i+1) for i in range(int(patch_num))]:
            patch_diversity = df_all_PD_hill2[scenario][reproduce_mode][patch_num][disp_among_rate][disp_within_rate][patch_dist_rate][rep].loc[patch_id]
            patch_K = df_actu_sp_dis[patch_id].loc['optimun_sp_id_values'].value_counts().sum() # patch 群落个体数量
            patch_N = df_actu_sp_dis[patch_id].loc['time_step4999'].value_counts().sum()        # patch 环境容纳量
            
            patch_theo_dom_sp_ls = list(df_actu_sp_dis[patch_id].loc['optimun_sp_id_values'].value_counts().index)
            try: 
                patch_actu_dom_sp = df_actu_sp_dis[patch_id].loc['time_step4999'].mode()[0]
            except: 
                patch_actu_dom_sp = None
            print(patch_id, patch_diversity, patch_actu_dom_sp, patch_theo_dom_sp_ls)
            
            if 1 <= patch_diversity <= 1.25 and patch_N/patch_K >= 0.8 and (len(patch_theo_dom_sp_ls)>=2 or patch_actu_dom_sp not in patch_theo_dom_sp_ls) and patch_actu_dom_sp != None:
                assembly_mechanism += ['PM' for i in range(int(hab_num))]
                print(patch_id, 'patch_diversity=', patch_diversity, 'PM')
                
            elif patch_N/patch_K < 0.1:
                assembly_mechanism += ['PE' for i in range(int(hab_num))]
                print(patch_id, 'patch_diversity=', patch_diversity, 'PE')
                
            else:
                for habitat_id in ['h%d'%i for i in range(hab_num)]:
                    habitat_diversity = df_all_HD_hill2[scenario][reproduce_mode][patch_num][disp_among_rate][disp_within_rate][patch_dist_rate][rep].loc[patch_id][habitat_id]
                    habitat_K = df_actu_sp_dis[patch_id][habitat_id].loc['optimun_sp_id_values'].value_counts().sum() # habitat 群落个体数量
                    habitat_N = df_actu_sp_dis[patch_id][habitat_id].loc['time_step4999'].value_counts().sum()        # habitat 环境容纳量
                    
                    if 1 <= habitat_diversity <= 1.25 and habitat_N/habitat_K >= 0.5:
                        theo_dom_sp_id, actu_dom_sp_id = get_theo_actual_dom_sp(df_actu_sp_dis=df_actu_sp_dis, patch_id=patch_id, habitat_id=habitat_id)
                        print(patch_id, habitat_id, theo_dom_sp_id, actu_dom_sp_id)
                        
                        if theo_dom_sp_id == actu_dom_sp_id: 
                            assembly_mechanism += ['SS']
                            print(patch_id, habitat_id, 'habitat_diversity=', habitat_diversity, 'SS')
                        else: 
                            assembly_mechanism += ['HM']
                            print(patch_id, habitat_id, 'habitat_diversity=', habitat_diversity, 'HM')
                            
                    elif habitat_N/habitat_K < 0.5:
                        assembly_mechanism += ['HE']
                        print(patch_id, habitat_id, 'habitat_diversity=', habitat_diversity, 'HE')
                        
                    elif habitat_diversity > 1.25 and habitat_N/habitat_K >= 0.5:
                        assembly_mechanism += ['MA']
                        print(patch_id, habitat_id, 'habitat_diversity=', habitat_diversity, 'MA')
    assembly_mechanism = np.array(assembly_mechanism).reshape(-1,1)
    if counter == 1:
        all_assembly_mechanism = assembly_mechanism
    else:
        all_assembly_mechanism = hstack_fillup(all_assembly_mechanism, assembly_mechanism)
        
patch_habitat_index=pd.MultiIndex.from_arrays([['patch%d'%i for i in range(1,101) for j in range(4)],['h0', 'h1', 'h2', 'h3']*100])       
all_assembly_mechanism_df = pd.DataFrame(all_assembly_mechanism, index=patch_habitat_index, columns=header)
all_assembly_mechanism_df.to_csv('all_assembly_mechanism_SSorHM1.25MA_GM=PM0.8_HM0.5_GE0.1_PE0.1_HE0.5.csv')  



                        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    














