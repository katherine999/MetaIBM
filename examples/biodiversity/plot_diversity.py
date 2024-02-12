# -*- coding: utf-8 -*-
"""
Created on Sun May  7 11:18:39 2023


"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def get_HD_subplot_data(df, disp_among_rate, disp_within_rate):
    x_ls, y_mean_ls, y_std_ls = [],[],[]
    
    scenario_para = ['neutral','neutral+niche','neutral+niche+gradual_evolution','neutral+niche+rapid_evolution']
    reproduce_mode_para = ['asexual', 'asexual', 'asexual', 'sexual']
    patch_num = '100'
    dist_rate = '0.000010'
    
    for scenario, reproduce_mode in zip(scenario_para, reproduce_mode_para):
    
            y_mean = np.nanmean(df.loc[:, (scenario, reproduce_mode, patch_num, disp_among_rate, disp_within_rate, dist_rate, slice(None))].to_numpy())
            y_std  = np.nanstd(df.loc[:, (scenario, reproduce_mode, patch_num, disp_among_rate, disp_within_rate, dist_rate, slice(None))].to_numpy())
            
            x_ls.append(scenario)
            y_mean_ls.append(y_mean)
            y_std_ls.append(y_std)

    return x_ls, y_mean_ls, y_std_ls


def get_PD_subplot_data(df, disp_among_rate, disp_within_rate):

    x_ls, y_mean_ls, y_std_ls = [],[],[]
    
    scenario_para = ['neutral','neutral+niche','neutral+niche+gradual_evolution','neutral+niche+rapid_evolution']
    reproduce_mode_para = ['asexual', 'asexual', 'asexual', 'sexual']
    patch_num = '100'
    dist_rate = '0.000010'
    
    for scenario, reproduce_mode in zip(scenario_para, reproduce_mode_para):
        
        y_mean = np.nanmean(df.loc[:, (scenario, reproduce_mode, patch_num, disp_among_rate, disp_within_rate, dist_rate, slice(None))].to_numpy())
        y_std  = np.nanstd(df.loc[:, (scenario, reproduce_mode, patch_num, disp_among_rate, disp_within_rate, dist_rate, slice(None))].to_numpy())
            
        x_ls.append(scenario)
        y_mean_ls.append(y_mean)
        y_std_ls.append(y_std)

    return x_ls, y_mean_ls, y_std_ls

def get_GD_subplot_data(df, disp_among_rate, disp_within_rate):
    
    x_ls, y_mean_ls, y_std_ls = [],[],[]
    
    scenario_para = ['neutral','neutral+niche','neutral+niche+gradual_evolution','neutral+niche+rapid_evolution']
    reproduce_mode_para = ['asexual', 'asexual', 'asexual', 'sexual']
    patch_num = '100'
    dist_rate = '0.000010'
    
    for scenario, reproduce_mode in zip(scenario_para, reproduce_mode_para):
        
        y_mean = np.nanmean(df.loc[:, (scenario, reproduce_mode, patch_num, disp_among_rate, disp_within_rate, dist_rate, slice(None))].to_numpy())
        y_std  = np.nanstd(df.loc[:, (scenario, reproduce_mode, patch_num, disp_among_rate, disp_within_rate, dist_rate, slice(None))].to_numpy())
            
        x_ls.append(scenario)
        y_mean_ls.append(y_mean)
        y_std_ls.append(y_std)

    return x_ls, y_mean_ls, y_std_ls

def get_CSR_subplot_data(df, reproduce_mode, is_heterogeneity, disp_among_rate, disp_within_rate, patch_dist_rate):
    
    patch_num_array = np.array(['001','004','016','064','256'])
    
    y_array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    
    x_mean_patch_num_1, x_std_patch_num_1 = [], []
    x_mean_patch_num_2, x_std_patch_num_2 = [], []
    x_mean_patch_num_3, x_std_patch_num_3 = [], []
    x_mean_patch_num_4, x_std_patch_num_4 = [], []
    x_mean_patch_num_5, x_std_patch_num_5 = [], []
    
    x_mean_dir = {'001':x_mean_patch_num_1, '004':x_mean_patch_num_2, '016':x_mean_patch_num_3, '064':x_mean_patch_num_4, '256':x_mean_patch_num_5}
    x_std_dir = {'001':x_std_patch_num_1, '004':x_std_patch_num_2, '016':x_std_patch_num_3, '064':x_std_patch_num_4, '256':x_std_patch_num_5}
    
    for patch_num in patch_num_array:
        x_mean_array = np.nanmean(df.loc[:, (reproduce_mode, patch_num, is_heterogeneity, disp_among_rate, disp_within_rate, patch_dist_rate, slice(None))].to_numpy(), axis=1)
        x_std_array = np.nanstd(df_all_CSR.loc[:, ('asexual', '001', 'True', '0.001000', '0.001000', '0.000010', slice(None))].to_numpy(), axis=1)
        
        x_mean_dir[patch_num] = x_mean_array
        x_std_dir[patch_num] = x_std_array
    
    return y_array, x_mean_dir, x_std_dir


########################################################################################################################

df_all_HD_hill0 = pd.read_csv('all_habitat_diversity_hill0.csv', header=[0,1,2,3,4,5,6], index_col=[0,1])
df_all_HD_hill1 = pd.read_csv('all_habitat_diversity_hill1.csv', header=[0,1,2,3,4,5,6], index_col=[0,1])
df_all_HD_hill2 = pd.read_csv('all_habitat_diversity_hill2.csv', header=[0,1,2,3,4,5,6], index_col=[0,1])

df_all_PD_hill0 = pd.read_csv('all_patch_diversity_hill0.csv', header=[0,1,2,3,4,5,6], index_col=[0])
df_all_PD_hill1 = pd.read_csv('all_patch_diversity_hill1.csv', header=[0,1,2,3,4,5,6], index_col=[0])
df_all_PD_hill2 = pd.read_csv('all_patch_diversity_hill2.csv', header=[0,1,2,3,4,5,6], index_col=[0])

df_all_GD_hill0 = pd.read_csv('all_global_diversity_hill0.csv', header=[0,1,2,3,4,5,6], index_col=[0])
df_all_GD_hill1 = pd.read_csv('all_global_diversity_hill1.csv', header=[0,1,2,3,4,5,6], index_col=[0])
df_all_GD_hill2 = pd.read_csv('all_global_diversity_hill2.csv', header=[0,1,2,3,4,5,6], index_col=[0])

df_all_CSR = pd.read_csv('all_culmulative_species_richness_curves.csv', header=[0,1,2,3,4,5,6], index_col=[0])

########################################################################################################################


#sns.pairplot(data=data, hue='species')



i = 0
location = [3,4,6,7,8,10,11,12,13,14,15,16,17,18,19,20]
empty_location = [1,2,5,9]

fig = plt.figure(figsize=(26,30))

#plt.suptitle('Hill2 number at habitat, patch and global scale', x=0.17, y=0.9, fontsize=25, c='r')

disp_among_within_para = [('0.010000','0.010000'),('0.010000','0.100000'),
                          ('0.005000','0.001000'),('0.005000','0.010000'),('0.005000','0.100000'),
                          ('0.001000','0.001000'),('0.001000','0.010000'),('0.001000','0.100000'),
                          ('0.000100','0.000100'),('0.000100','0.001000'),('0.000100','0.010000'),('0.000100','0.100000'),
                          ('0.000010','0.000100'),('0.000010','0.001000'),('0.000010','0.010000'),('0.000010','0.100000')]



bar_width = 0.2
index = np.arange(4)


for loc in empty_location:
    ax1 = plt.subplot(5,4,loc)
    plt.xticks([0,0.5,1.5,2.5,3.5,4], fontsize=0)
    
    plt.ylim(0, 20)
    plt.yticks([0,2.5,5,7.5,10,12.5,15,17.5], fontsize=20)
    plt.ylabel('Habitat or Patch Diversity', fontsize=24)
    
    if loc in [1]:
        plt.yticks([0,2.5,5,7.5,10,12.5,15,17.5,20], fontsize=20)
    
    if loc not in [1,5,9,13,17]:
        plt.yticks([0,2.5,5,7.5,10,12.5,15,17.5], fontsize=0)
        plt.ylabel('Habitat or Patch Diversity', fontsize=0)
        
    if loc in [1]:
        plt.title('disp_within=%.4f'%float(0.0001), fontsize=24, color='r')
        
    if loc in [2]:
        plt.title('disp_within=%.3f'%float(0.001), fontsize=24, color='r')
            
    ax2 = ax1.twinx()
    plt.yticks([])
    plt.yticks(np.array([0,15,30,45,60,75,90,105,120]), fontsize=0)
    #plt.ylim(0, 100)
    #plt.xticks(index+bar_width, ('Scenario 1','Scenario 2','Scenario 3','Scenario 4'))
    
    if loc in [1,5,9,13,17]:
        ax3 = ax1.twinx()
        ax3.spines['left'].set_position(('outward', 80))
        ax3.yaxis.set_ticks_position('left')
        ax3.yaxis.set_label_position('left')
        plt.yticks([], color='w')
        
        if loc == 1:
            ax3.set_ylabel('disp_among=%.2f'%float(0.01), fontsize=24, color='r')
        elif loc == 5:
            ax3.set_ylabel('disp_among=%.3f'%float(0.005), fontsize=24, color='r')
        elif loc == 9:
            ax3.set_ylabel('disp_among=%.3f'%float(0.001), fontsize=24, color='r')
    
    

for (disp_among_rate, disp_within_rate) in disp_among_within_para:
    
    ax1 = plt.subplot(5,4,location[i])
    plt.ylim(0, 20)
    #plt.yticks([0,2.5,5,7.5,10,12.5,15,17.5,20])
    
    
    if location[i] in [17,18,19,20]:
        plt.xticks(index+bar_width, ['Neutral','Niche','Slow Evol.','Rapid Evol.'], fontsize=24, rotation=20)
    else:
        plt.xticks(index+bar_width, ['Scen.1','Scen.2','Scen.3','Scen.4'], fontsize=0)
    
    if location[i] in [1,5,9,13,17]:
        plt.yticks([0,2.5,5,7.5,10,12.5,15,17.5], fontsize=20)
        plt.ylabel('Habitat or Patch Diversity', fontsize=24)
    if location[i] not in [1,5,9,13,17]:
        plt.yticks([0,2.5,5,7.5,10,12.5,15,17.5], fontsize=0)
    #plt.xlabel('Scenarios', loc='right')
    #plt.title('disp_among_rate=%.6f'%float(disp_among_rate)+'\n'+'disp_within_rate=%.6f'%float(disp_within_rate), fontsize=20)
    
    
    if location[i] in [1,2,3,4]:
        plt.title('disp_within=%.6f'%float(disp_within_rate), fontsize=24, color='r')

    x_ls, y_mean_ls, y_std_ls = get_HD_subplot_data(df_all_HD_hill2, disp_among_rate, disp_within_rate)
    bar1 = ax1.bar(index, y_mean_ls, bar_width, label='Habitat Diversity')
    ax1.errorbar(index, y_mean_ls, yerr=y_std_ls, capsize=3, elinewidth=2, fmt='.', ecolor='k', color='k')
    

    
    
    x_ls, y_mean_ls, y_std_ls = get_PD_subplot_data(df_all_PD_hill2, disp_among_rate, disp_within_rate)
    bar2 = ax1.bar(index+bar_width, y_mean_ls, bar_width, label='Patch Diversity')
    ax1.errorbar(index+bar_width, y_mean_ls, yerr=y_std_ls, capsize=3, elinewidth=2, fmt='.', ecolor='k', color='k')
    
    
    
    ax2 = ax1.twinx()
    x_ls, y_mean_ls, y_std_ls = get_GD_subplot_data(df_all_GD_hill2, disp_among_rate, disp_within_rate)
    bar3 = ax2.bar(index+2*bar_width, y_mean_ls, bar_width, color='g', label='Global Diversity')
    ax2.errorbar(index+2*bar_width, y_mean_ls, yerr=y_std_ls, capsize=3, elinewidth=2, fmt='.', ecolor='k', color='k')
    
    
    if location[i] in [4,8,12,16]:
        plt.yticks(np.array([15,30,45,60,75,90,105,120]), fontsize=20)
        plt.ylabel('Global Diversity', fontsize=24)
        
    if location[i] in [20]:
        plt.yticks(np.array([0,15,30,45,60,75,90,105,120]), fontsize=20)
    
    if location[i] not in [4,8,12,16,20]:
        plt.yticks(np.array([15,30,45,60,75,90,105,120]), fontsize=0)
    
    if location[i] in [4]:
        all_bar = [bar1[0]]+[bar2[0]]+[bar3[0]]
        labels = [bar1.get_label()] + [bar2.get_label()] + [bar3.get_label()]
        ax1.legend(all_bar, labels, loc='upper right', fontsize=20)
      
    if location[i] in [1,5,9,13,17]:
        ax3 = ax1.twinx()
        ax3.spines['left'].set_position(('outward', 80))
        ax3.yaxis.set_ticks_position('left')
        ax3.yaxis.set_label_position('left')
        plt.yticks([], color='w')
        ax3.set_ylabel('disp_among=%.6f'%float(disp_among_rate), fontsize=24, color='r')    

            
    #ax1.errorbar(x_221_1, y_221_1, yerr=y_221_1_err, capsize=3, elinewidth=2, fmt='or-', label='d=0.1')
    #ax1.errorbar(np.array(x_221_4)+width, y_221_4, yerr=y_221_4_err, capsize=3, elinewidth=2, fmt='og-', label='d=0.4')
    #plot_bar_text(np.array(x_221_1), y_221_1, h_text=0.01, fontsize=12)
    #plot_bar_text(np.array(x_221_4)+width, y_221_4, h_text=-0.05, fontsize=12)
    #plt.legend(loc='upper right')
    i+=1
plt.subplots_adjust(hspace=0.01)
plt.subplots_adjust(wspace=0.01)

plt.savefig('Hill2 number at each scale.jpg', bbox_inches='tight')








































