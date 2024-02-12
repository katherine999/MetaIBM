# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 17:34:11 2023

"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 

plt.style.use('ggplot')

def get_assembly_mechanism_count(Series, key):
    ''' '''
    try: value = Series[key]
    except: value = 0.0
    return value

def plot_assmembly_array(input_array):
    res_array = []
    for i in input_array:
        if i=='MA': res_array.append(0)
        if i=='SS': res_array.append(1)
        if i=='Eco': res_array.append(2)
        if i=='Evo': res_array.append(3)
        if i=='HM': res_array.append(4)
        if i=='PM': res_array.append(5)
        if i=='GM': res_array.append(6)
        if i=='EM': res_array.append(7)
    return np.array(res_array)

def count(input_arr):
    res_arr = np.bincount(input_arr)
    while len(res_arr) < 8:
        res_arr = np.append(res_arr, 0)
    return res_arr
        
#############################################################################
file_name = 'all_assembly_mechanism_SSorHM1.25MA_GM=PM0.8_HM0.5_GE0.1_PE0.1_HE0.5.csv'
all_assembly_mechanism_df = pd.read_csv(file_name, index_col=[0,1], header=[0,1,2,3,4,5,6], skipinitialspace=True)
all_assembly_counts = all_assembly_mechanism_df.apply(pd.value_counts, normalize=True).fillna(0)

headers = all_assembly_mechanism_df.columns


counter = 0
all_dom_mechanism = []
for header in headers:
    counter+=1
    scenario = header[0]
    reproduce_mode = header[1]
    patch_num  = header[2]
    disp_among = header[3]
    disp_within = header[4]
    dist_rate = header[5]
    rep = header[6]
    
    assembly_counts = all_assembly_counts[scenario][reproduce_mode][patch_num][disp_among][disp_within][dist_rate][rep]
    
    MA = get_assembly_mechanism_count(assembly_counts, 'MA')
    SS = get_assembly_mechanism_count(assembly_counts, 'SS')
    
    HM = get_assembly_mechanism_count(assembly_counts, 'HM')
    PM = get_assembly_mechanism_count(assembly_counts, 'PM')
    GM = get_assembly_mechanism_count(assembly_counts, 'GM')
    
    HE = get_assembly_mechanism_count(assembly_counts, 'HE')
    PE = get_assembly_mechanism_count(assembly_counts, 'PE')
    GE = get_assembly_mechanism_count(assembly_counts, 'GE')
    
    
    if (MA+SS) > 0.75:
        if MA > SS: 
            all_dom_mechanism.append('MA')
            continue
        if MA <= SS: 
            all_dom_mechanism.append('SS')
            continue
        
    elif (HM+PM+GM) > 0.75:
        if HM >= PM and HM >= GM: 
            all_dom_mechanism.append('HM')
            continue
        if PM >= HM and PM >= GM: 
            all_dom_mechanism.append('PM')
            continue
        if GM >= HM and GM >= PM: 
            all_dom_mechanism.append('GM')
            continue
        
    elif (HE+PE+GE) > 0.5:
        all_dom_mechanism.append('EM')
        
    else:
        if (SS+MA) >= (HM+PM+GM): 
            all_dom_mechanism.append('Eco')
            continue
        if (SS+MA) < (HM+PM+GM): 
            all_dom_mechanism.append('Evo')
            continue
        
all_dom_mechanism = np.array(all_dom_mechanism).reshape(1,-1)
all_dom_mechanism_df = pd.DataFrame(all_dom_mechanism, columns=headers, index=['dom_assembly'])  


i = 0
location = [3,4,6,7,8,10,11,12,13,14,15,16,17,18,19,20]
empty_location = [1,2,5,9]

fig = plt.figure(figsize=(40,30))

#plt.suptitle('Hill2 number at habitat, patch and global scale', x=0.17, y=0.9, fontsize=25, c='r')

disp_among_within_para = [('0.010000','0.010000'),('0.010000','0.100000'),
                          ('0.005000','0.001000'),('0.005000','0.010000'),('0.005000','0.100000'),
                          ('0.001000','0.001000'),('0.001000','0.010000'),('0.001000','0.100000'),
                          ('0.000100','0.000100'),('0.000100','0.001000'),('0.000100','0.010000'),('0.000100','0.100000'),
                          ('0.000010','0.000100'),('0.000010','0.001000'),('0.000010','0.010000'),('0.000010','0.100000')]

scenario_para = ['neutral','neutral+niche','neutral+niche+gradual_evolution','neutral+niche+rapid_evolution']
reproduce_mode_para = ['asexual', 'asexual', 'asexual', 'sexual']



'''
colors = ['red', 'lightcoral', 'palegreen', 'mediumseagreen', 'lightblue', 'steelblue', 'blue', 'black']
labels = ['ME', 'SS', 'Eco', 'Evo', 'HM', 'PM', 'GM', 'EM']
bounds = [0,1,2,3,4,5,6,7]
my_cmap = mpl.colors.ListedColormap(colors)


for (disp_among_rate, disp_within_rate) in disp_among_within_para:
    
    ax1 = plt.subplot(5,4,location[i])
    
    count_array = np.empty((0,8))
    for scenario, reproduce_mode in zip(scenario_para, reproduce_mode_para):
        dom_assembly_array = all_dom_mechanism_df.loc[:,(scenario, reproduce_mode, patch_num, disp_among_rate, disp_within_rate, dist_rate)].to_numpy().reshape(-1)
        dom_assembly_num_array = plot_assmembly_array(dom_assembly_array)
        dom_assembly_num_count_array  = count(dom_assembly_num_array)
        count_array = np.vstack((count_array, dom_assembly_num_count_array))
    
    i += 1
    
    x = np.arange(0,4) # the 4 scenarios
    scenario_array = np.array(['Neutral','Niche','Slow Evol.','Rapid Evol.']) # for labels
    
    left = np.zeros(4)
    for mechan_idx in range(0, count_array.shape[1]):
        y = count_array[:,mechan_idx]
        plt.barh(x,y,left=left,tick_label=scenario_array ,align='center',color=colors[mechan_idx], label=labels[mechan_idx])
        left += y
'''


colors = ['red', 'lightcoral', 'lightblue', 'steelblue', 'grey','black']
labels = ['MA', 'SS', 'HM', 'PM', 'HE', 'PE']
my_cmap = mpl.colors.ListedColormap(colors)

for (disp_among_rate, disp_within_rate) in disp_among_within_para:
    ax1 = plt.subplot(5,4,location[i])
    
    assembly_counts_s1 = all_assembly_counts.loc[:,(scenario_para[0], 'asexual', patch_num, disp_among_rate, disp_within_rate, dist_rate)].mean(axis=1)
    assembly_counts_s2 = all_assembly_counts.loc[:,(scenario_para[1], 'asexual', patch_num, disp_among_rate, disp_within_rate, dist_rate)].mean(axis=1)
    assembly_counts_s3 = all_assembly_counts.loc[:,(scenario_para[2], 'asexual', patch_num, disp_among_rate, disp_within_rate, dist_rate)].mean(axis=1)
    assembly_counts_s4 = all_assembly_counts.loc[:,(scenario_para[3], 'sexual', patch_num, disp_among_rate, disp_within_rate, dist_rate)].mean(axis=1)
      
    left = np.zeros(4)
    x = np.arange(0,4) # the 4 scenarios
    scenario_array = np.array(['Neutral','Niche','Slow Evol.','Rapid Evol.']) # for labels
    for mechan_idx in range(len(labels)):
        mechan = labels[mechan_idx]
        y = np.array([assembly_counts_s1[mechan], assembly_counts_s2[mechan], assembly_counts_s3[mechan], assembly_counts_s4[mechan]])
        plt.barh(x,y,left=left,tick_label=scenario_array ,align='center',color=colors[mechan_idx], label=labels[mechan_idx])
        left += y
    
    if location[i] in [1,5,9,13,17]:
        plt.yticks([0,1,2,3], ['Neutral','Niche','Slow Evol.','Rapid Evol.'], fontsize=24)
    else:
        plt.yticks([0,1,2,3], ['Neutral','Niche','Slow Evol.','Rapid Evol.'], fontsize=0)
    
    plt.xlim(0,1)
    plt.xticks([0,0.2,0.4,0.6,0.8,1.0], ['0', '20%', '40%', '60%', '80%', '100%'], fontsize=24)
    

    
    
    if location[i] in [1,2,3,4]:
        plt.title('disp_within=%.6f'%float(disp_within_rate), fontsize=26, color='r')
        
        
    if location[i] in [1,5,9,13,17]:
        ax3 = ax1.twinx()
        ax3.spines['left'].set_position(('outward', 140))
        ax3.yaxis.set_ticks_position('left')
        ax3.yaxis.set_label_position('left')
        plt.yticks([], color='w')
        ax3.set_ylabel('disp_among=%.5f'%float(disp_among_rate), fontsize=26, color='r')  
        
    i+=1
        
    
    

for loc in empty_location:
    ax1 = plt.subplot(5,4,loc)
    plt.xticks([0,0.5,1.5,2.5,3.5,4], fontsize=0)
    
    plt.ylim(0,4)
    if loc in [1,5,9,13,17]:
        plt.yticks([0.5,1.5,2.5,3.5], ['Neutral','Niche','Slow Evol.','Rapid Evol.'], fontsize=24)
    else:
        plt.yticks([0.5,1.5,2.5,3.5], ['Neutral','Niche','Slow Evol.','Rapid Evol.'], fontsize=0)
    
    plt.xlim(0,1)
    plt.xticks([0,0.2,0.4,0.6,0.8,1.0], ['0', '20%', '40%', '60%', '80%', '100%'], fontsize=24)
    
    if loc in [1]:
        plt.title('disp_within=%.4f'%float(0.0001), fontsize=26, color='r')
        
    if loc in [2]:
        plt.title('disp_within=%.3f'%float(0.001), fontsize=26, color='r')
    
    if loc in [1,5,9,13,17]:
        ax3 = ax1.twinx()
        ax3.spines['left'].set_position(('outward', 140))
        ax3.yaxis.set_ticks_position('left')
        ax3.yaxis.set_label_position('left')
        plt.yticks([], color='w')
        
        if loc == 1:
            ax3.set_ylabel('disp_among=%.2f'%float(0.01), fontsize=26, color='r')
        elif loc == 5:
            ax3.set_ylabel('disp_among=%.3f'%float(0.005), fontsize=26, color='r')
        elif loc == 9:
            ax3.set_ylabel('disp_among=%.3f'%float(0.001), fontsize=26, color='r')

plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.1)
plt.savefig('assembly_mechanism.jpg', bbox_inches='tight')
            
        








    
    
    
    
    
    
    
    
    
    
    
    
    
    
    