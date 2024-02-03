#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 00:25:14 2023

@author:  Unvieling it after the peers review of the article
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import metacommunity_IBM as metaIBM
from metacommunity_IBM import patch
from metacommunity_IBM import metacommunity

################################################# def simulation ###########################################################################################################
def new_round(_float, _len = 0):
    """
    Parameters
    ----------
    _float: float
    _len: int, 指定四舍五入需要保留的小数点后几位数为_len

    Returns
    -------
    type ==> float, 返回四舍五入后的值
    """
    if isinstance(_float, float):
        if str(_float)[::-1].find('.') <= _len:
            return (_float)
        if str(_float)[-1] == '5':
            return (round(float(str(_float)[:-1] + '6'), _len))
        else:
            return (round(_float, _len))
    else:
        return (round(_float, _len))

def generating_empty_metacommunity(meta_name, patch_num, patch_location_ls, hab_num, hab_length, hab_width, dormancy_pool_max_size, 
                                   x_axis_environment_values_ls, y_axis_environment_values_ls, environment_types_num, environment_types_name, 
                                   environment_variation_ls, is_heterogeneity=False):
    ''' '''
    log_info = 'generating empty metacommunity ... \n'
    meta_object = metacommunity(metacommunity_name=meta_name)
    patch_num_x_axis, patch_num_y_axis = int(np.sqrt(patch_num)), int(np.sqrt(patch_num))
    hab_num_x_axis, hab_num_y_axis = int(np.sqrt(hab_num)), int(np.sqrt(hab_num))
    
    x_axis_env_num, y_axis_env_num = len(x_axis_environment_values_ls), len(y_axis_environment_values_ls)
    
    for i in range(0, patch_num):
        patch_name = 'patch%d'%(i+1)
        patch_index = i
        location = patch_location_ls[i]
        p = patch(patch_name, patch_index, location)
        
        patch_x_loc, patch_y_loc = location[0], location[1]

        for j in range(hab_num):
            habitat_name = 'h%s'%str(j)
            hab_index = j
            
            hab_x_loc, hab_y_loc = patch_x_loc*hab_num_x_axis+j//hab_num_y_axis, patch_y_loc*hab_num_y_axis+j%hab_num_y_axis
            hab_location = (hab_x_loc, hab_y_loc)
            
            if is_heterogeneity==False:
                x_axis_environment_means_value = x_axis_environment_values_ls[int(hab_x_loc*x_axis_env_num/(patch_num_x_axis*hab_num_x_axis))%x_axis_env_num]
                y_axis_environment_means_value = y_axis_environment_values_ls[int(hab_y_loc*y_axis_env_num/(patch_num_y_axis*hab_num_y_axis))%y_axis_env_num]
            elif is_heterogeneity==True:
                x_axis_min, x_axis_max = x_axis_environment_values_ls[0], x_axis_environment_values_ls[1]
                y_axis_min, y_axis_max = y_axis_environment_values_ls[0], y_axis_environment_values_ls[1]
                
                x_step = (x_axis_max - x_axis_min)/(patch_num_x_axis*hab_num_x_axis)
                y_step = (y_axis_max - y_axis_min)/(patch_num_y_axis*hab_num_y_axis)
                
                x_axis_environment_means_value = x_axis_min+1/2*x_step + x_step*hab_x_loc
                y_axis_environment_means_value = y_axis_min+1/2*y_step + y_step*hab_y_loc
                
            p.add_habitat(hab_name=habitat_name, hab_index=hab_index, hab_location=hab_location, num_env_types=environment_types_num, env_types_name=environment_types_name, 
                          mean_env_ls=[x_axis_environment_means_value, y_axis_environment_means_value], var_env_ls=environment_variation_ls, length=hab_length, width=hab_width, dormancy_pool_max_size=dormancy_pool_max_size)
            
            info = '%s: %s, %s, %s, %s: x_axis_environment_means_value=%s, y_axis_environment_means_value=%s'%(meta_object.metacommunity_name, patch_name, str(location), habitat_name, str(hab_location), str(x_axis_environment_means_value), str(y_axis_environment_means_value))
            log_info = log_info + info + '\n'
        meta_object.add_patch(patch_name=patch_name, patch_object=p)
    #print(log_info)
    return meta_object, log_info

def generate_empty_mainland(meta_name, patch_num, patch_location_ls, hab_num, hab_length, hab_width, dormancy_pool_max_size, 
                            micro_environment_values_ls, macro_environment_values_ls, environment_types_num, environment_types_name, environment_variation_ls):
    log_info = 'generating empty mainland ... \n'
    meta_object = metacommunity(metacommunity_name=meta_name)
    for i in range(0, patch_num):
        patch_name = 'patch%d'%(i)
        patch_index = i
        location = patch_location_ls[i]
        patch_x_loc, patch_y_loc = location[0], location[1]
        p = patch(patch_name, patch_index, location)
        
        hab_num_length, hab_num_width = int(np.sqrt(hab_num)), int(np.sqrt(hab_num))
        
        for j in range(hab_num):
            habitat_name = 'h%s'%str(j)
            hab_index = j
            
            hab_x_loc, hab_y_loc = patch_x_loc*hab_num_length+j//hab_num_width, patch_y_loc*hab_num_width+j%hab_num_width
            hab_location = (hab_x_loc, hab_y_loc)
            
            micro_environment_mean_value = micro_environment_values_ls[j//hab_num_width]
            macro_environment_mean_value = macro_environment_values_ls[j%hab_num_width]
            
            p.add_habitat(hab_name=habitat_name, hab_index=hab_index, hab_location=hab_location, num_env_types=environment_types_num, env_types_name=environment_types_name, 
                          mean_env_ls=[micro_environment_mean_value, macro_environment_mean_value], var_env_ls=environment_variation_ls, length=hab_length, width=hab_width, dormancy_pool_max_size=dormancy_pool_max_size)
            
            info = '%s: %s, %s, %s, %s: micro_environment_mean_value=%s, macro_environment_mean_value=%s \n'%(meta_object.metacommunity_name, patch_name, str(location), habitat_name, str(hab_location), str(micro_environment_mean_value), str(macro_environment_mean_value))
            log_info = log_info + info
        meta_object.add_patch(patch_name=patch_name, patch_object=p)
    #print(log_info)
    return meta_object, log_info
########################################################################################################################################################
def mkdir_if_not_exist(reproduce_mode, patch_num, is_heterogeneity, disp_among_within_rate, patch_dist_rate):
    root_path = os.getcwd()
    
    reproduce_mode_dir = {'asexual':'/asexual', 'sexual':'/sexual'}
    patch_num_files_name = '/patch_num=%03d'%patch_num
    is_heterogeneity_files_name = '/is_heterogeneity=%s'%str(is_heterogeneity)
    disp_amomg_within_rate_files_name = '/disp_among=%f-disp_within=%f'%(disp_among_within_rate[0], disp_among_within_rate[1])
    patch_dist_rate_files_name = '/patch_dist_rate=%f'%patch_dist_rate
    
    goal_path = root_path + reproduce_mode_dir[reproduce_mode] + patch_num_files_name + is_heterogeneity_files_name + disp_amomg_within_rate_files_name + patch_dist_rate_files_name
    if os.path.exists(goal_path) == False:
        os.makedirs(goal_path)
    else:
        pass
    return goal_path
################################################## logging module ########################################################################################
def write_logger(log_info, is_logging=False, logger_file=None):
    if is_logging == True:
        print(log_info, file=logger_file)
    elif is_logging == False:
        print(log_info)
################################################## def main() ######################################################################################################
def main(rep, patch_num, reproduce_mode, total_disp_among_rate, disp_within_rate, patch_dist_rate, goal_path=None):
    if goal_path==None: goal_path = os.getcwd()
    
    ''' timer '''
    all_time_start = time.time()
    
    ''' replication index (not running) '''
    #rep = 0
    
    ''' time-step scales parameters '''
    all_time_step = 5000
    
    ''' map size parameters '''
    meta_length, meta_width = 320, 320
    #patch_num = 16  # 1, 4, 16, 64, 256, 1024
    patch_num_x_axis, patch_num_y_axis = int(np.sqrt(patch_num)), int(np.sqrt(patch_num))
    patch_location_ls = [(i,j) for i in range(patch_num_x_axis) for j in range(patch_num_y_axis)] # patch_row, hab_col
    if patch_num == 1: hab_num_in_patch = 16
    else: hab_num_in_patch = 4
    hab_num_x_axis, hab_num_y_axis = int(np.sqrt(hab_num_in_patch)), int(np.sqrt(hab_num_in_patch)) # hab_num_x_axis in a patch, hab_num_y_axis in a patch
    hab_length, hab_width = int(meta_length/patch_num_y_axis/hab_num_y_axis), int(meta_width/patch_num_x_axis/hab_num_x_axis) 

    ''' environmental parameters '''
    environment_types_num = 2
    environment_types_name=('x_axis_environment', 'y_axis_environment')
    environment_variation_ls = [0.025, 0.025]
    total_dormancy_pool_max_size = 0
    
    is_heterogeneity=False
    if is_heterogeneity==False: x_axis_environment_values_ls, y_axis_environment_values_ls = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] #离散值
    elif is_heterogeneity==True: x_axis_environment_values_ls, y_axis_environment_values_ls = [0.1, 0.9], [0.1, 0.9]                    #连续值
    
    ''' demography parameters '''
    base_dead_rate=0.1
    fitness_wid=0.5
    #reproduce_mode = 'sexual'
    asexual_birth_rate = 0.5
    sexual_birth_rate = 1
    mutation_rate = 0.0001
    
    ''' landscape parameters '''
    colonize_rate = 0.001
    #total_disp_among_rate = 0.001
    #disp_within_rate =0.1
    propagules_rain_num = 40000 * colonize_rate
    #patch_dist_rate = 0.00001
    
    ''' species parameters '''
    #species_num = 4
    traits_num = 2
    pheno_names_ls = ('x_axis_phenotype', 'y_axis_phenotype')
    pheno_var_ls=(0.025, 0.025)
    geno_len_ls=(20, 20)
    species_2_phenotype_ls = [[i/10, j/10] for i in range(0,10) for j in range(0,10)]
    #[[0.2,0.2], [0.4,0.4], [0.6,0.6], [0.8,0.8]] # (index+1) indicates species_id
    
    ''' logging or not logging '''
    is_logging = True
    
    ''' logging module '''
    if is_logging == True: 
        logger_file_name = goal_path+'/'+'rep=%d-logger.log'%(rep)
        logger_file = open(logger_file_name, "a")
    
    ''' initailization processes '''
    write_in_logger_info = ''
    
    mainland, log_info = generate_empty_mainland(meta_name='mainland', patch_num=1, patch_location_ls=[(0,0)], hab_num=100, hab_length=20, hab_width=20, dormancy_pool_max_size=0, 
                                                 micro_environment_values_ls=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], macro_environment_values_ls=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], 
                                                 environment_types_num=environment_types_num, environment_types_name=environment_types_name, environment_variation_ls=environment_variation_ls)
    write_in_logger_info += log_info
    
    meta_obj, log_info = generating_empty_metacommunity('metacommunity', patch_num, patch_location_ls, hab_num_in_patch, hab_length, hab_width, int(new_round(total_dormancy_pool_max_size/(patch_num*hab_num_in_patch))), 
                                                        x_axis_environment_values_ls, y_axis_environment_values_ls, environment_types_num, environment_types_name, environment_variation_ls, is_heterogeneity)
    write_in_logger_info += log_info
    
    write_in_logger_info += mainland.meta_initialize(traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls)
    
    write_logger(write_in_logger_info, is_logging, logger_file)
    
    ''' data saving and files controling '''
    columns_patch_id, columns_habitat_id, columns_mocrosite_id = meta_obj.columns_patch_habitat_microsites_id()
    columns = [columns_patch_id, columns_habitat_id, columns_mocrosite_id]
    #mode='w'
    #meta_sp_dis_all_time = meta_obj.get_meta_microsites_optimum_sp_id_val(base_dead_rate, fitness_wid, species_2_phenotype_ls)
    #meta_x_axis_phenotype_all_time = meta_obj.get_meta_microsite_environment_values(environment_name='x_axis_environment')
    #meta_y_axis_phenotype_all_time = meta_obj.get_meta_microsite_environment_values(environment_name='y_axis_environment')
    #mode='a'
    meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsites_optimum_sp_id_val(base_dead_rate, fitness_wid, species_2_phenotype_ls), 
                                                       file_name=goal_path+'/'+'rep=%d-meta_species_distribution_all_time.csv.gz'%(rep), 
                                                       index=['optimun_sp_id_values'], columns=columns, mode='w')
    meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsite_environment_values(environment_name='x_axis_environment'), 
                                                       file_name=goal_path+'/'+'rep=%d-meta_x_axis_phenotype_all_time.csv.gz'%(rep), 
                                                       index=['x_axis_environment_values'], columns=columns, mode='w')
    meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsite_environment_values(environment_name='y_axis_environment'), 
                                                       file_name=goal_path+'/'+'rep=%d-meta_y_axis_phenotype_all_time.csv.gz'%(rep), 
                                                       index=['y_axis_environment_values'], columns=columns, mode='w')
    for time_step in range(all_time_step): 
        write_in_logger_info = ''
        write_in_logger_info += 'time_step=%d \n'%time_step
        #print('time_step=%d'%time_step)
        if reproduce_mode == 'asexual':
            ''' dead selection process in mainland and metacommunity '''
            write_in_logger_info += mainland.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            write_in_logger_info += mainland.meta_mainland_asexual_birth_mutate_germinate(asexual_birth_rate, mutation_rate, pheno_var_ls)
            write_in_logger_info += meta_obj.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            ''' reproduction process '''
            write_in_logger_info += meta_obj.meta_asex_reproduce_calculation_into_offspring_marker_pool(asexual_birth_rate)
            #meta_obj.meta_asex_reproduce_mutate_into_offspring_pool(asexual_birth_rate, mutation_rate, pheno_var_ls)
            ''' dispersal processes '''
            write_in_logger_info += meta_obj.meta_colonize_from_propagules_rains(mainland, propagules_rain_num)
            write_in_logger_info += meta_obj.meta_dispersal_within_patch_from_offspring_marker_to_immigrant_marker_pool(disp_within_rate)
            #meta_obj.meta_dispersal_within_patch_from_offspring_to_immigrant_pool(disp_within_rate)
            write_in_logger_info += meta_obj.dispersal_aomng_patches_from_offspring_marker_pool_to_immigrant_marker_pool(total_disp_among_rate)
            #meta_obj.dispersal_aomng_patches_from_offspring_pool_to_immigrant_pool(total_disp_among_rate)
            ''' germination processes '''
            write_in_logger_info += meta_obj.meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool(mutation_rate, pheno_var_ls)
            #meta_obj.meta_local_germinate_from_offspring_and_immigrant_pool()
            #meta_obj.meta_local_germinate_from_offspring_immigrant_and_dormancy_pool()
            ''' dormancy process (not running) '''
            #meta_obj.meta_dormancy_process_from_offspring_pool_and_immigrant_pool()
            ''' disturbance process '''
            write_in_logger_info += meta_obj.meta_disturbance_process_in_patches(patch_dist_rate)
            ''' eliminating offspring and immigrant (marker) pool '''
            meta_obj.meta_clear_up_offspring_marker_and_immigrant_marker_pool()
            #meta_obj.meta_clear_up_offspring_and_immigrant_pool()
            
        elif reproduce_mode == 'sexual':
            ''' dead selection process in mainland and metacommunity '''
            write_in_logger_info += mainland.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            write_in_logger_info += mainland.meta_mainland_mixed_birth_mutate_germinate(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls)
            write_in_logger_info += meta_obj.meta_dead_selection(base_dead_rate, fitness_wid, method='niche_gaussian')
            ''' reproduction process '''
            write_in_logger_info += meta_obj.meta_mix_reproduce_calculation_with_offspring_marker_pool(asexual_birth_rate, sexual_birth_rate)
            #meta_obj.meta_mix_reproduce_mutate_into_offspring_pool(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls)
            ''' dispersal processes '''
            write_in_logger_info += meta_obj.pairwise_sexual_colonization_from_prpagules_rains(mainland, propagules_rain_num)
            write_in_logger_info += meta_obj.meta_dispersal_within_patch_from_offspring_marker_to_immigrant_marker_pool(disp_within_rate)
            #meta_obj.meta_dispersal_within_patch_from_offspring_to_immigrant_pool(disp_within_rate)
            write_in_logger_info += meta_obj.dispersal_aomng_patches_from_offspring_marker_pool_to_immigrant_marker_pool(total_disp_among_rate)
            #meta_obj.dispersal_aomng_patches_from_offspring_pool_to_immigrant_pool(total_disp_among_rate)
            ''' germination processes '''
            write_in_logger_info += meta_obj.meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool(mutation_rate, pheno_var_ls)
            #meta_obj.meta_local_germinate_from_offspring_and_immigrant_pool()
            #meta_obj.meta_local_germinate_from_offspring_immigrant_and_dormancy_pool()
            ''' dormancy process (not running) '''
            #meta_obj.meta_dormancy_process_from_offspring_pool_and_immigrant_pool()
            ''' disturbance process '''
            write_in_logger_info += meta_obj.meta_disturbance_process_in_patches(patch_dist_rate)
            ''' eliminating offspring and immigrant (marker) pool '''
            meta_obj.meta_clear_up_offspring_marker_and_immigrant_marker_pool()
            #meta_obj.meta_clear_up_offspring_and_immigrant_pool()
        
        ''' logging module '''
        write_logger(write_in_logger_info, is_logging, logger_file)
        
        ''' GUI interface updates at each time-step and saves as a jpg file (not running) '''
        #meta_obj.meta_show_species_distribution(sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, vmin=1, vmax=100, cmap=plt.get_cmap('tab20'), file_name=goal_path+'/'+'rep=%d-time_step=%d-metacommunity_sp_x_axis_phenotype_dis.jpg'%(rep, time_step))

        ''' data saving and files controling '''
        #mode='w
        #meta_sp_dis_all_time = np.vstack((meta_sp_dis_all_time, meta_obj.get_meta_microsites_individuals_sp_id_values()))
        #meta_x_axis_phenotype_all_time = np.vstack((meta_x_axis_phenotype_all_time, meta_obj.get_meta_microsites_individuals_phenotype_values(trait_name='x_axis_phenotype')))
        #meta_y_axis_phenotype_all_time = np.vstack((meta_y_axis_phenotype_all_time, meta_obj.get_meta_microsites_individuals_phenotype_values(trait_name='y_axis_phenotype')))
        #mode='a'
        meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsites_individuals_sp_id_values(), file_name=goal_path+'/'+'rep=%d-meta_species_distribution_all_time.csv.gz'%(rep), index=['time_step%d'%time_step], columns=columns, mode='a')
        meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsites_individuals_phenotype_values(trait_name='x_axis_phenotype'), file_name=goal_path+'/'+'rep=%d-meta_x_axis_phenotype_all_time.csv.gz'%(rep), index=['time_step%d'%time_step], columns=columns, mode='a')
        meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_obj.get_meta_microsites_individuals_phenotype_values(trait_name='y_axis_phenotype'), file_name=goal_path+'/'+'rep=%d-meta_y_axis_phenotype_all_time.csv.gz'%(rep), index=['time_step%d'%time_step], columns=columns, mode='a')
    
    ''' GUI interface updates at the end of a simulation saves as a jpg file '''
    #camp1,camp2,camp3 = plt.get_cmap('tab20'),plt.get_cmap('tab20b'),plt.get_cmap('tab20c')
    #new_camp = ListedColormap(camp1.colors+camp2.colors+camp3.colors)
    new_camp = plt.get_cmap('gist_rainbow')
    
    mainland.meta_show_species_distribution(sub_row=1, sub_col=1, hab_num_x_axis_in_patch=10, hab_num_y_axis_in_patch=10, hab_y_len=20, hab_x_len=20, vmin=1, vmax=100, cmap=new_camp, file_name=goal_path+'/'+'rep=%d-time_step=%d-mainland_sp_dis.jpg'%(rep, time_step))
    meta_obj.meta_show_species_distribution(sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, vmin=1, vmax=100, cmap=new_camp, file_name=goal_path+'/'+'rep=%d-time_step=%d-metacommunity_sp_dis.jpg'%(rep, time_step))
    meta_obj.meta_show_species_phenotype_distribution(trait_name='x_axis_phenotype', sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, cmap=plt.get_cmap('Blues'), file_name=goal_path+'/'+'rep=%d-time_step=%d-metacommunity_sp_x_axis_phenotype_dis.jpg'%(rep, time_step))
    meta_obj.meta_show_species_phenotype_distribution(trait_name='y_axis_phenotype', sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, cmap=plt.get_cmap('Greens'), file_name=goal_path+'/'+'rep=%d-time_step=%d-metacommunity_sp_y_axis_phenotype_dis.jpg'%(rep, time_step))
    #meta_obj.meta_show_environment_distribution(environment_name='x_axis_environment', sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, mask_loc='upper', cmap=plt.get_cmap('Blues'), file_name=goal_path+'/'+'rep=%d-metacommunity_x_axis_environment.jpg'%(rep))
    #meta_obj.meta_show_environment_distribution(environment_name='y_axis_environment', sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, mask_loc='lower', cmap=plt.get_cmap('Greens'), file_name=goal_path+'/'+'rep=%d-metacommunity_y_axis_environment.jpg'%(rep))
    meta_obj.meta_show_two_environment_distribution(environment1_name='x_axis_environment', environment2_name='y_axis_environment', sub_row=patch_num_y_axis, sub_col=patch_num_x_axis, hab_num_x_axis_in_patch=hab_num_x_axis, hab_num_y_axis_in_patch=hab_num_y_axis, hab_y_len=hab_length, hab_x_len=hab_width, mask_loc1='upper', mask_loc2='lower', cmap1=plt.get_cmap('Blues'), cmap2=plt.get_cmap('Greens'), file_name=goal_path+'/'+'rep=%d-metacommunity_environment.jpg'%(rep))

    #''' data saving as a csv.gz files, mode='w' '''
    #mode='w'
    #meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_sp_dis_all_time, file_name=goal_path+'/'+'rep=%d_meta_species_distribution_all_time.csv.gz'%(rep), index=['optimun_sp_id_values']+['time_step%d'%i for i in range(all_time_step)], columns=columns, mode='w')
    #meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_x_axis_phenotype_all_time, file_name=goal_path+'/'+'rep=%d_meta_x_axis_phenotype_all_time.csv.gz'%(rep), index=['x_axis_environment_values']+['time_step%d'%i for i in range(all_time_step)], columns=columns, mode='w')
    #meta_obj.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_y_axis_phenotype_all_time, file_name=goal_path+'/'+'rep=%d_meta_y_axis_phenotype_all_time.csv.gz'%(rep), index=['y_axis_environment_values']+['time_step%d'%i for i in range(all_time_step)], columns=columns, mode='w')
    
    ''' timer '''
    all_time_end = time.time()
    
    ''' logging module '''
    log_info = "总模拟运行时间：%.8s s \n" % (all_time_end-all_time_start)
    write_logger(log_info, is_logging, logger_file)
    if is_logging == True: logger_file.close()
    #print(log_info)
##############################################################################################################################################################################
if __name__ == '__main__':
    main(rep=0, patch_num=100, reproduce_mode='asexual', total_disp_among_rate=0.001, disp_within_rate=0.1, patch_dist_rate=0.00001)
