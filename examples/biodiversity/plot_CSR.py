# -*- coding: utf-8 -*-
"""
Created on Sun May  7 11:18:39 2023


"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def get_CSR_subplot_data(df, disp_among_rate, disp_within_rate):
    
    x_mean_dir, x_std_dir = {}, {}
    
    scenario_para = ['neutral','neutral+niche','neutral+niche+gradual_evolution','neutral+niche+rapid_evolution']
    reproduce_mode_para = ['asexual', 'asexual', 'asexual', 'sexual']
    patch_num = '100'
    dist_rate = '0.000010'
    
    y_array = np.arange(1,101,1)
    for scenario, reproduce_mode in zip(scenario_para, reproduce_mode_para):
        
        x_mean_array = np.nanmean(df.loc[:, (scenario, reproduce_mode, patch_num, disp_among_rate, disp_within_rate, dist_rate, slice(None))].to_numpy(), axis=1)
        x_std_array = np.nanstd(df.loc[:, (scenario, reproduce_mode, patch_num, disp_among_rate, disp_within_rate, dist_rate, slice(None))].to_numpy(), axis=1)
        
        x_mean_dir[scenario] = x_mean_array
        x_std_dir[scenario]  = x_std_array
         
    return y_array, x_mean_dir, x_std_dir


########################################################################################################################

df_all_CSR = pd.read_csv('all_culmulative_species_richness_curves.csv', header=[0,1,2,3,4,5,6], index_col=[0])

########################################################################################################################

i = 0
location = [3,4,6,7,8,10,11,12,13,14,15,16,17,18,19,20]
empty_location = [1,2,5,9]

fig = plt.figure(figsize=(26,30))
#plt.suptitle('Culmulative_Species_Richness', x=0.17, y=0.9, fontsize=25, c='r')

disp_among_within_para = [('0.010000','0.010000'),('0.010000','0.100000'),
                          ('0.005000','0.001000'),('0.005000','0.010000'),('0.005000','0.100000'),
                          ('0.001000','0.001000'),('0.001000','0.010000'),('0.001000','0.100000'),
                          ('0.000100','0.000100'),('0.000100','0.001000'),('0.000100','0.010000'),('0.000100','0.100000'),
                          ('0.000010','0.000100'),('0.000010','0.001000'),('0.000010','0.010000'),('0.000010','0.100000')]

for loc in empty_location:
    ax1 = plt.subplot(5,4,loc)
    #plt.xticks([0,0.5,1.5,2.5,3.5,4], fontsize=0)
    
    plt.ylim(0, 100)
    plt.xlim(0, 102400)
    plt.yticks([0,20,40,60,80,100], fontsize=20)
    plt.xticks([0,20000,40000,60000,80000,102400], fontsize=20, rotation=15)
    #plt.ylabel('Habitat or Patch Diversity', fontsize=20)
    
    #if loc in [1]:
        #plt.yticks([0,2.5,5,7.5,10,12.5,15,17.5,20], fontsize=16)
    
    #if loc not in [1,5,9,13,17]:
        #plt.yticks([0,2.5,5,7.5,10,12.5,15,17.5], fontsize=0)
        #plt.ylabel('Habitat or Patch Diversity', fontsize=0)
        
    if loc in [1]:
        plt.title('disp_within=%.4f'%float(0.0001), fontsize=24, color='r')
        
    if loc in [2]:
        plt.title('disp_within=%.3f'%float(0.001), fontsize=24, color='r')
    
    if loc in [1,5,9,13,17]:
        plt.ylabel('Number of Species', fontsize=24)
    
    
    
    if loc in [1,5,9,13,17]:
        ax3 = ax1.twinx()
        ax3.spines['left'].set_position(('outward', 70))
        ax3.yaxis.set_ticks_position('left')
        ax3.yaxis.set_label_position('left')
        plt.yticks([], color='w')
        
        if loc == 1:
            ax3.set_ylabel('disp_among=%.2f'%float(0.01), fontsize=24, color='r')
        elif loc == 5:
            ax3.set_ylabel('disp_among=%.3f'%float(0.005), fontsize=24, color='r')
        elif loc == 9:
            ax3.set_ylabel('disp_among=%.3f'%float(0.001), fontsize=24, color='r')
    
    
    
    
    
    
    
    #ax2 = ax1.twinx()
    #plt.yticks([])
    #plt.yticks(np.array([0,15,30,45,60,75,90,105,120]), fontsize=0)
    #plt.ylim(0, 100)
    #plt.xticks(index+bar_width, ('Scenario 1','Scenario 2','Scenario 3','Scenario 4'))
    
    #if loc in [1,5,9,13,17]:
        #ax3 = ax1.twinx()
        #ax3.spines['left'].set_position(('outward', 70))
        #ax3.yaxis.set_ticks_position('left')
        #ax3.yaxis.set_label_position('left')
        #plt.yticks([], color='w')
        
        #if loc == 1:
            #ax3.set_ylabel('disp_among_rate=%.6f'%float(0.01), fontsize=20, color='r')
        #elif loc == 5:
            #ax3.set_ylabel('disp_among_rate=%.6f'%float(0.005), fontsize=20, color='r')
        #elif loc == 9:
            #ax3.set_ylabel('disp_among_rate=%.6f'%float(0.001), fontsize=20, color='r')





for (disp_among_rate, disp_within_rate) in disp_among_within_para:
    
    ax1 = plt.subplot(5,4,location[i])
    plt.ylim(0, 100)
    plt.xlim(0,102400)

    y_array, x_mean_dir, x_std_dir = get_CSR_subplot_data(df_all_CSR.fillna(1024000), disp_among_rate, disp_within_rate)
    
    print(disp_among_rate, disp_within_rate)
    print(y_array)
    print(x_mean_dir)
    print(x_std_dir)
    
    plt.errorbar(x_mean_dir['neutral'], y_array, ms=4, capsize=2, elinewidth=2, fmt='or-', label='Neutral')
    plt.errorbar(x_mean_dir['neutral+niche'], y_array, ms=4, capsize=2, elinewidth=2, fmt='ob-', label='Niche')
    plt.errorbar(x_mean_dir['neutral+niche+gradual_evolution'], y_array, ms=4, capsize=2, elinewidth=2, fmt='og-', label='Slow Evol.')
    plt.errorbar(x_mean_dir['neutral+niche+rapid_evolution'], y_array, ms=4, capsize=2, elinewidth=2, fmt='oy-', label='Rapid Evol.')

    plt.yticks([0,20,40,60,80,100], fontsize=20)
    plt.xticks([0,20000,40000,60000,80000,102400], fontsize=20, rotation=15)

    

    if location[i] in [1,2,3,4]:
        plt.title('disp_within=%.6f'%float(disp_within_rate), fontsize=24, color='r')

    if location[i] in [1,5,9,13,17]:
        plt.ylabel('Number of Species', fontsize=24)

    

    if location[i] in [4]:
        plt.legend(loc='lower right', fontsize=22)


    if location[i] in [1,5,9,13,17]:
        ax3 = ax1.twinx()
        ax3.spines['left'].set_position(('outward', 70))
        ax3.yaxis.set_ticks_position('left')
        ax3.yaxis.set_label_position('left')
        plt.yticks([], color='w')
        ax3.set_ylabel('disp_among=%.6f'%float(disp_among_rate), fontsize=24, color='r') 
    i+=1
plt.savefig('Culmulative_Species_Richness.jpg', bbox_inches='tight')











































