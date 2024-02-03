# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:32:30 2022

@title: metacommunuity individuals based model version 2.9.12

@author: Unvieling it after the peers review of the article
"""

import numpy as np
import random
#import networkx as nx
import matplotlib.pyplot as plt
import math
import copy
import re
import pandas as pd
import seaborn as sns

##################################################### class habitat #########################################################################################
class habitat():
    def __init__(self, hab_name, hab_index, hab_location, num_env_types, env_types_name, mean_env_ls, var_env_ls, length, width, dormancy_pool_max_size):
        '''
        int num_env_types is the number of environment types in the habitat.
        env_types_name is the list of names of env_types.
        list mean_env_ls is the tuple of mean environment values; the len(mean_env_ls)=num_env_types.
        list var_env_ls is the list of variation of enviroment distribution in the habitat.
        int length is the length of the habitat.
        int width is the width of the habitat.
        int size is the the number of microsites within a habitat.
        '''
        self.name = hab_name
        self.index = hab_index
        self.location = hab_location 
        self.num_env_types = num_env_types
        self.env_types_name = env_types_name
        self.mean_env_ls = mean_env_ls
        self.var_env_ls = var_env_ls
        self.length = length
        self.width = width
        self.size = length*width
        self.set = {}                     # self.data_set={} # to be improved
        self.indi_num = 0
        
        self.offspring_pool = []
        self.immigrant_pool = []
        self.dormancy_pool = []
        
        self.offspring_marker_pool = []   # used only without dormancy process, marker = (patch_id, h_id, reproduction_mode)
        self.immigrant_marker_pool = []   # used only without dormancy process, marker = (patch_id, h_id, reproduction_mode)
        
        self.species_category = {}
        self.occupied_site_pos_ls = []
        
        self.empty_site_pos_ls = [(i, j) for i in range(length) for j in range(width)]
        self.dormancy_pool_max_size = dormancy_pool_max_size
        
        self.reproduction_mode_threhold = 0.897
        self.asexual_parent_pos_ls = []                           # If an individual can fit its environment condition well, it goes through asexual reproduction. only for mix
        self.species_category_for_sexual_parents_pos = {}         # If an individual can not fit its environmet condition, it goes through sexual reproduction. only for mix

        for index in range(0, len(mean_env_ls)):
            mean_e_index = self.mean_env_ls[index]
            var_e_index = self.var_env_ls[index]
            name_e_index = self.env_types_name[index]
            microsite_e_values = np.random.normal(loc=0, scale=var_e_index, size=(self.length, self.width)) + mean_e_index
            self.set[name_e_index] = microsite_e_values

        microsite_individuals = [[None for i in range(self.length)] for i in range(self.width)]
        self.set['microsite_individuals'] = microsite_individuals
        
    def __str__(self):
        return str(self.set)
    
    def reset_environment_values(self, env_name, mean_env, var_env):
        pass

    def add_individual(self, indi_object, len_id, wid_id):
        if self.set['microsite_individuals'][len_id][wid_id] != None:
            print('the microsite in the habitat is occupied.')
        else:
            self.set['microsite_individuals'][len_id][wid_id] = indi_object
            self.empty_site_pos_ls.remove((len_id, wid_id))
            self.occupied_site_pos_ls.append((len_id, wid_id))
            self.indi_num +=1

            if indi_object.species_id in self.species_category.keys():
                if indi_object.gender in self.species_category[indi_object.species_id].keys():
                    self.species_category[indi_object.species_id][indi_object.gender].append((len_id, wid_id))
                else:
                    self.species_category[indi_object.species_id][indi_object.gender] = [(len_id, wid_id)]             
            else:
                self.species_category[indi_object.species_id] = {indi_object.gender:[(len_id, wid_id)]}
                
    def del_individual(self, len_id, wid_id):
        if self.set['microsite_individuals'][len_id][wid_id] == None:
            print('the microsite in the habitat is empty.')
        else:
            indi_object = self.set['microsite_individuals'][len_id][wid_id]
            self.set['microsite_individuals'][len_id][wid_id] = None
            self.empty_site_pos_ls.append((len_id, wid_id))
            self.occupied_site_pos_ls.remove((len_id, wid_id))
            self.indi_num -=1 
            self.species_category[indi_object.species_id][indi_object.gender].remove((len_id, wid_id))    
    
    def habitat_disturbance_process(self):
        microsite_individuals = [[None for i in range(self.length)] for i in range(self.width)]
        self.set['microsite_individuals'] = microsite_individuals
        self.empty_site_pos_ls = [(i, j) for i in range(self.length) for j in range(self.width)]
        self.occupied_site_pos_ls = []
        self.indi_num = 0
        self.species_category = {}
        self.asexual_parent_pos_ls = []
        self.species_category_for_sexual_parents_pos = {} 
        self.offspring_pool = []
        self.immigrant_pool = []
        
    def get_hab_pairwise_empty_site_pos_ls(self):
        ''' return as [((len_id, wid_id), (len_id, wid_id)) ...]'''
        hab_pairwise_empty_sites_pos_ls = []
        if len(self.empty_site_pos_ls) < 2:
            return hab_pairwise_empty_sites_pos_ls
        else:
            empty_sites_pos_ls = copy.deepcopy(self.empty_site_pos_ls) 
            random.shuffle(empty_sites_pos_ls)
            for i in range(0, len(empty_sites_pos_ls)-1, 2):
                empty_site_1_pos = empty_sites_pos_ls[i]
                empty_site_2_pos = empty_sites_pos_ls[i+1]
                
                hab_pairwise_empty_sites_pos_ls.append((empty_site_1_pos, empty_site_2_pos))
            return hab_pairwise_empty_sites_pos_ls    
    
    def get_hab_pairwise_occupied_site_pos_ls(self):
        ''' return as [((len_id, wid_id), (len_id, wid_id)) ...]'''
        hab_pairwise_occupied_sites_pos_ls = []
        if self.indi_num < 2:
            return hab_pairwise_occupied_sites_pos_ls
        else:
            species_category = copy.deepcopy(self.species_category)
            for sp_id, sp_id_val in species_category.items():
                try:
                    sp_id_female_ls = sp_id_val['female']
                    random.shuffle(sp_id_female_ls)
                except:
                    continue
                try:
                    sp_id_male_ls = sp_id_val['male']
                    random.shuffle(sp_id_male_ls)
                except:
                    continue
                try:
                    sp_id_pairwise_occupied_pos_ls = list(zip(sp_id_female_ls, sp_id_male_ls))
                    hab_pairwise_occupied_sites_pos_ls += sp_id_pairwise_occupied_pos_ls
                except:
                    continue
            return hab_pairwise_occupied_sites_pos_ls
    
    def hab_initialize(self, traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls):
        mean_pheno_val_ls = self.mean_env_ls
        species_id = 'sp%d'%(species_2_phenotype_ls.index(mean_pheno_val_ls)+1)
        
        for row in range(self.length):
            for col in range(self.width):
                if reproduce_mode == 'asexual': gender = 'female'
                if reproduce_mode == 'sexual': gender = random.sample(('male', 'female'), 1)[0]
                indi_object = individual(species_id=species_id, traits_num=traits_num, pheno_names_ls=pheno_names_ls, gender=gender)
                indi_object.random_init_indi(mean_pheno_val_ls, pheno_var_ls, geno_len_ls)
                self.add_individual(indi_object, row, col)
        return 0    
    
    def get_microsite_env_val_ls(self, len_id, wid_id):
        ''' return a list of environment value of all the environment type in the order of env_types_name '''
        env_val_ls = []
        for env_name in self.env_types_name:
            env_val = self.set[env_name][len_id][wid_id]
            env_val_ls.append(env_val)
        return env_val_ls
    
    def survival_rate(self, d, phenotype_ls, env_val_ls, w=0.5, method='niche_gaussian'):
        #d is the baseline death rate responding to the disturbance strength.
        #phenotype_ls is a list of phenotype of each trait.
        #env_val_ls is a list of environment value responding to the environment type.
        #w is the width of the fitness function.
        
        
        if method == 'niche_gaussian':
            survival_rate = (1-d)
            for index in range(len(phenotype_ls)):
                ei = phenotype_ls[index]               # individual phenotype of a trait 
                em = env_val_ls[index]                 # microsite environment value of a environment type
                survival_rate *= math.exp((-1)*math.pow(((ei-em)/w),2))
                
        elif method == 'neutral_uniform':
            survival_rate = 1-d

        return survival_rate

    def hab_dead_selection(self, base_dead_rate, fitness_wid, method):
        self.asexual_parent_pos_ls = []                           # If an individual can fit its environment condition well, it goes through asexual reproduction.
        self.species_category_for_sexual_parents_pos = {}         # If an individual can not fit its environmet condition, it goes through sexual reproduction.
        counter = 0
        for row in range(self.length):
            for col in range(self.width):
                env_val_ls = self.get_microsite_env_val_ls(row, col)
                
                if self.set['microsite_individuals'][row][col] != None:
                    individual_object = self.set['microsite_individuals'][row][col]
                    phenotype_ls = individual_object.get_indi_phenotype_ls()
                    survival_rate = self.survival_rate(d=base_dead_rate, phenotype_ls=phenotype_ls, env_val_ls=env_val_ls, w=fitness_wid, method=method)
                    
                    if survival_rate < np.random.uniform(0,1,1)[0]:
                        self.del_individual(len_id=row, wid_id=col)
                        counter += 1
                    else:
                        if survival_rate >= self.reproduction_mode_threhold: 
                            self.asexual_parent_pos_ls.append((row, col))     # the individual fits its local environment
                        else:
                            if individual_object.species_id in self.species_category_for_sexual_parents_pos.keys():
                                if individual_object.gender in self.species_category_for_sexual_parents_pos[individual_object.species_id].keys():
                                    self.species_category_for_sexual_parents_pos[individual_object.species_id][individual_object.gender].append((row, col))
                                else:
                                    self.species_category_for_sexual_parents_pos[individual_object.species_id][individual_object.gender] = [(row, col)]
                            else:
                                self.species_category_for_sexual_parents_pos[individual_object.species_id] = {individual_object.gender:[(row, col)]}         
                else:
                    continue
        return counter
#*************** for habitat for mainland only and do not contains dormant bank ******************************************
    def hab_asex_reproduce_mutate_with_num(self, mutation_rate, pheno_var_ls, num):
        ''' asexual reproduction for dispersal controlled by the parameter, num '''
        hab_disp_pool = []
        parent_pos_ls = random.sample(self.occupied_site_pos_ls, num)
        
        for parent_pos in parent_pos_ls:
            row = parent_pos[0]
            col = parent_pos[1]
            individual_object = self.set['microsite_individuals'][row][col]
            new_indivi_object = copy.deepcopy(individual_object)
            for i in range(new_indivi_object.traits_num):
                pheno_name = new_indivi_object.pheno_names_ls[i]
                var = pheno_var_ls[i] #### to be improved #### 
                genotype = new_indivi_object.genotype_set[pheno_name]
                phenotype = np.mean(genotype) + random.gauss(0, var)
                new_indivi_object.phenotype_set[pheno_name] = phenotype
            new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
            hab_disp_pool.append(new_indivi_object)
        return hab_disp_pool
    
    def hab_sexual_pairwise_parents_ls(self):
        pair_parents_ls = []
        for sp_id, sp_id_val in self.species_category.items():
            try:
                sp_id_female_ls = sp_id_val['female']
            except:
                continue
            try:
                sp_id_male_ls = sp_id_val['male']
            except:
                continue
            
            random.shuffle(sp_id_female_ls) #list of individuals location in habitat, i.e., (len_id, wid_id)
            random.shuffle(sp_id_male_ls)   #random sample of pairwise parents in sexual reproduction
            
            pair_parents_ls += list(zip(sp_id_female_ls, sp_id_male_ls))
        return pair_parents_ls
    
    def hab_sexual_pairwise_parents_num(self):
        return len(self.hab_sexual_pairwise_parents_ls())  
    
    def hab_sex_reproduce_mutate_with_num(self, mutation_rate, pheno_var_ls, num):
        ''' asexual reproduction for dispersal controlled by the parameter, num '''
        hab_disp_pool = []
        pairwise_parents_pos_ls = random.sample(self.hab_sexual_pairwise_parents_ls(), num)
        
        for female_pos, male_pos in pairwise_parents_pos_ls:
            female_row, female_col = female_pos[0], female_pos[1]
            male_row, male_col = male_pos[0], male_pos[1]
            female_indi_obj = self.set['microsite_individuals'][female_row][female_col]
            male_indi_obj = self.set['microsite_individuals'][male_row][male_col]
            
            new_indivi_object = copy.deepcopy(female_indi_obj)
            new_indivi_object.gender = random.sample(('male', 'female'), 1)[0]
            for i in range(new_indivi_object.traits_num):
                pheno_name = new_indivi_object.pheno_names_ls[i]
                var = pheno_var_ls[i] ##### to be improved  #####
                
                female_bi_genotype = female_indi_obj.genotype_set[pheno_name]
                genotype1 = random.sample(female_bi_genotype, 1)[0]
                
                male_bi_genotype = male_indi_obj.genotype_set[pheno_name]
                genotype2 = random.sample(male_bi_genotype, 1)[0]
                
                new_bi_genotype = [genotype1, genotype2]
                phenotype = np.mean(new_bi_genotype) + random.gauss(0, var)
                
                new_indivi_object.genotype_set[pheno_name] = new_bi_genotype
                new_indivi_object.phenotype_set[pheno_name] = phenotype
                
            new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
            hab_disp_pool.append(new_indivi_object)
        return hab_disp_pool
    
    def hab_mixed_sexual_pairwise_parents_ls(self):
        pair_parents_ls = []
        for sp_id, sp_id_val in self.species_category_for_sexual_parents_pos.items():
            try:
                sp_id_female_ls = sp_id_val['female']
            except:
                continue
            try:
                sp_id_male_ls = sp_id_val['male']
            except:
                continue
            
            random.shuffle(sp_id_female_ls) #list of individuals location in habitat, i.e., (len_id, wid_id)
            random.shuffle(sp_id_male_ls)   #random sample of pairwise parents in sexual reproduction
            
            pair_parents_ls += list(zip(sp_id_female_ls, sp_id_male_ls))
        return pair_parents_ls
    
    def hab_mixed_sexual_pairwse_parents_num(self):
        return len(self.hab_mixed_sexual_pairwise_parents_ls())
    
    def hab_mixed_asexual_parent_num(self):
        return len(self.asexual_parent_pos_ls)    
    
    def hab_mix_asex_reproduce_mutate_with_num(self, mutation_rate, pheno_var_ls, num):
        ''' mixed asexual reproduction for dispersal controlled by the parameter, num '''
        hab_disp_pool = []
        parent_pos_ls = random.sample(self.asexual_parent_pos_ls, num)
        
        for parent_pos in parent_pos_ls:
            row = parent_pos[0]
            col = parent_pos[1]
            individual_object = self.set['microsite_individuals'][row][col]
            new_indivi_object = copy.deepcopy(individual_object)
            for i in range(new_indivi_object.traits_num):
                pheno_name = new_indivi_object.pheno_names_ls[i]
                var = pheno_var_ls[i] #### to be improved #### 
                genotype = new_indivi_object.genotype_set[pheno_name]
                phenotype = np.mean(genotype) + random.gauss(0, var)
                new_indivi_object.phenotype_set[pheno_name] = phenotype
            new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
            hab_disp_pool.append(new_indivi_object)
        return hab_disp_pool
    
    def hab_mix_sex_reproduce_mutate_with_num(self, mutation_rate, pheno_var_ls, num):
        ''' mixed sexual reproduction for dispersal controlled by the parameter, num '''
        hab_disp_pool = []
        pairwise_parents_pos_ls = random.sample(self.hab_mixed_sexual_pairwise_parents_ls(), num)
        
        for female_pos, male_pos in pairwise_parents_pos_ls:
            female_row, female_col = female_pos[0], female_pos[1]
            male_row, male_col = male_pos[0], male_pos[1]
            female_indi_obj = self.set['microsite_individuals'][female_row][female_col]
            male_indi_obj = self.set['microsite_individuals'][male_row][male_col]
            
            new_indivi_object = copy.deepcopy(female_indi_obj)
            new_indivi_object.gender = random.sample(('male', 'female'), 1)[0]
            for i in range(new_indivi_object.traits_num):
                pheno_name = new_indivi_object.pheno_names_ls[i]
                var = pheno_var_ls[i] ##### to be improved  #####
                
                female_bi_genotype = female_indi_obj.genotype_set[pheno_name]
                genotype1 = random.sample(female_bi_genotype, 1)[0]
                
                male_bi_genotype = male_indi_obj.genotype_set[pheno_name]
                genotype2 = random.sample(male_bi_genotype, 1)[0]
                
                new_bi_genotype = [genotype1, genotype2]
                phenotype = np.mean(new_bi_genotype) + random.gauss(0, var)
                
                new_indivi_object.genotype_set[pheno_name] = new_bi_genotype
                new_indivi_object.phenotype_set[pheno_name] = phenotype
                
            new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
            hab_disp_pool.append(new_indivi_object)
        return hab_disp_pool
##******************************* birth and local germination *************************************#####    
    def hab_asexual_reprodece_germinate(self, asexual_birth_rate, mutation_rate, pheno_var_ls):
        ''' birth into empty site directly without considering the competition between local offspring and immigrant offspring '''
        counter = 0
        empty_sites_pos_ls = self.empty_site_pos_ls
        if len(empty_sites_pos_ls) < int(self.indi_num * asexual_birth_rate): 
            num = len(empty_sites_pos_ls)
        elif len(empty_sites_pos_ls) >= int(self.indi_num * asexual_birth_rate): 
            num = int(self.indi_num * asexual_birth_rate)  
        hab_offsprings_for_germinate = self.hab_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num)
        
        random.shuffle(empty_sites_pos_ls)
        random.shuffle(hab_offsprings_for_germinate)
        
        for pos, indi_object in list(zip(empty_sites_pos_ls, hab_offsprings_for_germinate)):
            len_id = pos[0]
            wid_id = pos[1]
            self.add_individual(indi_object, len_id, wid_id)
            counter += 1
        return counter
        
    def hab_sexual_reprodece_germinate(self, sexual_birth_rate, mutation_rate, pheno_var_ls):
        ''' birth into empty site directly without considering the competition between local offspring and immigrant offspring '''
        counter = 0
        empty_sites_pos_ls = self.empty_site_pos_ls
        if len(empty_sites_pos_ls) < int(self.hab_sexual_pairwise_parents_num() * sexual_birth_rate): 
            num = len(empty_sites_pos_ls)
        elif len(empty_sites_pos_ls) >= int(self.hab_sexual_pairwise_parents_num() * sexual_birth_rate): 
            num = int(self.hab_sexual_pairwise_parents_num() * sexual_birth_rate)  
        hab_offsprings_for_germinate = self.hab_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num)
    
        random.shuffle(empty_sites_pos_ls)
        random.shuffle(hab_offsprings_for_germinate)
        
        for pos, indi_object in list(zip(empty_sites_pos_ls, hab_offsprings_for_germinate)):
            len_id = pos[0]
            wid_id = pos[1]
            self.add_individual(indi_object, len_id, wid_id)
            counter += 1
        return counter
    
    def hab_mixed_reproduce_germinate(self, asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls):
        ''' birth into empty site directly without considering the competition between local offspring and immigrant offspring '''
        counter = 0
        empty_sites_pos_ls = self.empty_site_pos_ls
        empty_sites_num = len(empty_sites_pos_ls)
        
        asex_offs_expectation_num = int(np.around(self.hab_mixed_asexual_parent_num() * asexual_birth_rate))
        sex_offs_expectation_num = int(np.around(self.hab_mixed_sexual_pairwse_parents_num() * sexual_birth_rate))
        
        if empty_sites_num < asex_offs_expectation_num + sex_offs_expectation_num:
            asex_num = int(np.around(empty_sites_num * asex_offs_expectation_num/(asex_offs_expectation_num + sex_offs_expectation_num)))
            sex_num = int(np.around(empty_sites_num * sex_offs_expectation_num/(asex_offs_expectation_num + sex_offs_expectation_num)))
            
        elif empty_sites_num >= asex_offs_expectation_num + sex_offs_expectation_num:
            asex_num = asex_offs_expectation_num
            sex_num = sex_offs_expectation_num
        
        hab_offsprings_for_germinate = self.hab_mix_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, asex_num) + self.hab_mix_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, sex_num)
             
        for pos, indi_object in list(zip(empty_sites_pos_ls, hab_offsprings_for_germinate)):
            len_id = pos[0]
            wid_id = pos[1]
            self.add_individual(indi_object, len_id, wid_id)
            counter += 1
        return counter

#*****************  reproduction into offspring_marker_pool process *********************************
    def hab_asex_reproduce_calculation_into_offspring_marker_pool(self, patch_name, asexual_birth_rate):
        ''' 
        Calculating the offspring num and generating offspring markers 
        but do not reproduce actually until local germination process afterward after dispersal process 
        patch_id: the patch of the habitat belong with, (i.e., patch_object.patch_id)
        The function will be called by the the patch_object of the habitat belong with
        '''
        self.offspring_marker_pool = []
        offspring_num = self.indi_num * asexual_birth_rate
        offspring_num_int = int(offspring_num)                    #整数部分
        offspring_num_dem = offspring_num - offspring_num_int     #小数部分
        
        offspring_marker_ls = [(patch_name, self.name, 'asexual') for _ in range(offspring_num_int)]
        if offspring_num_dem > np.random.uniform(0,1,1)[0]:
            one_offspring_marker_ls = [(patch_name, self.name, 'asexual')]
        else:
            one_offspring_marker_ls = []
        self.offspring_marker_pool = offspring_marker_ls + one_offspring_marker_ls
        return len(self.offspring_marker_pool)

    def hab_sex_reproduce_calculation_into_offspring_marker_pool(self, patch_name, sexual_birth_rate):
        ''' 
        Calculating the offspring num and generating offspring markers 
        but do not reproduce actually until local germination process afterward after dispersal process 
        patch_id: the patch of the habitat belong with, (i.e., patch_object.patch_id)
        The function will be called by the the patch_object of the habitat belong with
        '''
        self.offspring_marker_pool = []
        offspring_num = self.indi_num * sexual_birth_rate
        offspring_num_int = int(offspring_num)                    #整数部分
        offspring_num_dem = offspring_num - offspring_num_int     #小数部分
        
        offspring_marker_ls = [(patch_name, self.name, 'sexual') for _ in range(offspring_num_int)]
        if offspring_num_dem > np.random.uniform(0,1,1)[0]:
            one_offspring_marker_ls = [(patch_name, self.name, 'sexual')]
        else:
            one_offspring_marker_ls = []
        self.offspring_marker_pool = offspring_marker_ls + one_offspring_marker_ls
        return len(self.offspring_marker_pool)
    
    def hab_mix_reproduce_calculation_into_offspring_marker_pool(self, patch_name, asexual_birth_rate, sexual_birth_rate):
        ''' 
        Calculating the offspring num and generating offspring markers 
        but do not reproduce actually until local germination process afterward after dispersal process 
        patch_id: the patch of the habitat belong with, (i.e., patch_object.patch_id)
        The function will be called by the the patch_object of the habitat belong with
        '''
        self.offspring_marker_pool = []
        asex_offspring_num = len(self.asexual_parent_pos_ls) * asexual_birth_rate
        asex_offspring_num_int = int(asex_offspring_num)
        asex_offspring_num_dem = asex_offspring_num - asex_offspring_num_int
        
        asex_offspring_marker_ls = [(patch_name, self.name, 'mix_asexual') for _ in range(asex_offspring_num_int)]
        if asex_offspring_num_dem > np.random.uniform(0,1,1)[0]:
            asex_one_offspring_marker_ls = [(patch_name, self.name, 'mix_asexual')]
        else:
            asex_one_offspring_marker_ls = []
            
        sex_offspring_num = self.hab_mixed_sexual_pairwse_parents_num() * sexual_birth_rate
        sex_offspring_num_int = int(sex_offspring_num)
        sex_offspring_num_dem = sex_offspring_num - sex_offspring_num_int
        
        sex_offspring_marker_ls = [(patch_name, self.name, 'mix_sexual') for _ in range(sex_offspring_num_int)]
        if sex_offspring_num_dem > np.random.uniform(0,1,1)[0]:
            sex_one_offspring_marker_ls = [(patch_name, self.name, 'mix_sexual')]
        else:
            sex_one_offspring_marker_ls = []
        self.offspring_marker_pool = asex_offspring_marker_ls + asex_one_offspring_marker_ls + sex_offspring_marker_ls + sex_one_offspring_marker_ls
            
        return len(self.offspring_marker_pool)
        
#*************************** reprodction into offsprings pool processes *************************************
    def hab_asex_reproduce_mutate_into_offspring_pool(self, asexual_birth_rate, mutation_rate, pheno_var_ls):
        self.offspring_pool = []
        offspring_num = self.indi_num * asexual_birth_rate
        offspring_num_int = int(offspring_num)                    #整数部分
        offspring_num_dem = offspring_num - offspring_num_int     #小数部分
        
        offspring_ls = self.hab_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=offspring_num_int) # a list of offspring individual object
        if offspring_num_dem > np.random.uniform(0,1,1)[0]:
            one_offspring_ls = self.hab_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)
        else:
            one_offspring_ls = []
        self.offspring_pool = offspring_ls + one_offspring_ls
        return len(self.offspring_pool)
    
    def hab_sex_reproduce_mutate_into_offspring_pool(self, sexual_birth_rate, mutation_rate, pheno_var_ls):
        self.offspring_pool = []
        offspring_num = self.hab_sexual_pairwise_parents_num() * sexual_birth_rate
        offspring_num_int = int(offspring_num)                    #整数部分
        offspring_num_dem = offspring_num - offspring_num_int     #小数部分
        
        offspring_ls = self.hab_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=offspring_num_int) # a list of offspring individual object
        if offspring_num_dem > np.random.uniform(0,1,1)[0]:
            one_offspring_ls = self.hab_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)
        else:
            one_offspring_ls = []
        self.offspring_pool = offspring_ls + one_offspring_ls
        return len(self.offspring_pool)
    
    def hab_mix_reproduce_mutate_into_offspring_pool(self, asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls):
        self.offspring_pool = []
        asex_offspring_num = len(self.asexual_parent_pos_ls) * asexual_birth_rate
        asex_offspring_num_int = int(asex_offspring_num)
        asex_offspring_num_dem = asex_offspring_num - asex_offspring_num_int
        
        asex_offspring_ls = self.hab_mix_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=asex_offspring_num_int)
        if asex_offspring_num_dem > np.random.uniform(0,1,1)[0]:
            asex_one_offspring_ls = self.hab_mix_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)
        else:
            asex_one_offspring_ls = []
            
        sex_offspring_num = self.hab_mixed_sexual_pairwse_parents_num() * sexual_birth_rate
        sex_offspring_num_int = int(sex_offspring_num)
        sex_offspring_num_dem = sex_offspring_num - sex_offspring_num_int
        
        sex_offspring_ls = self.hab_mix_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=sex_offspring_num_int)
        if sex_offspring_num_dem > np.random.uniform(0,1,1)[0]:
            sex_one_offspring_ls = self.hab_mix_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)
        else:
            sex_one_offspring_ls = []
        self.offspring_pool = asex_offspring_ls + asex_one_offspring_ls + sex_offspring_ls + sex_one_offspring_ls
        return len(self.offspring_pool)

#*********************************** local germination processes ************************************************
    def hab_local_germinate_from_offspring_and_dormancy_pool(self):
        hab_empty_pos_ls = self.empty_site_pos_ls
        hab_offspring_and_dormancy_pool = self.offspring_pool + self.dormancy_pool
        
        random.shuffle(hab_empty_pos_ls)
        random.shuffle(hab_offspring_and_dormancy_pool)
        
        counter = 0
        for (row_id, col_id), indi_object in list(zip(hab_empty_pos_ls, hab_offspring_and_dormancy_pool)):
            self.add_individual(indi_object=indi_object, len_id=row_id, wid_id=col_id)
            counter += 1
        return counter
    
    def hab_local_germinate_from_offspring_and_immigrant_pool(self):
        hab_empty_pos_ls = self.empty_site_pos_ls
        hab_offspring_and_immigrant_pool = self.offspring_pool + self.immigrant_pool
        
        random.shuffle(hab_empty_pos_ls)
        random.shuffle(hab_offspring_and_immigrant_pool)
        
        counter = 0
        for (row_id, col_id), indi_object in list(zip(hab_empty_pos_ls, hab_offspring_and_immigrant_pool)):
            self.add_individual(indi_object=indi_object, len_id=row_id, wid_id=col_id)
            counter += 1
        return counter
        
    def hab_local_germinate_from_offspring_immigrant_and_dormancy_pool(self):
        hab_empty_pos_ls = self.empty_site_pos_ls
        offspring_immigrant_and_dormancy_pool = self.offspring_pool + self.immigrant_pool + self.dormancy_pool
        
        random.shuffle(hab_empty_pos_ls)
        random.shuffle(offspring_immigrant_and_dormancy_pool)
        
        counter = 0
        for (row_id, col_id), indi_object in list(zip(hab_empty_pos_ls, offspring_immigrant_and_dormancy_pool)):
            self.add_individual(indi_object=indi_object, len_id=row_id, wid_id=col_id)
            counter += 1
        return counter
    
#************************** dormancy process in the habitat *********************************************************
    def hab_dormancy_process_from_offspring_pool_to_dormancy_pool(self):
        offspring_num = len(self.offspring_pool)
        dormancy_num = len(self.dormancy_pool)
        
        if self.dormancy_pool_max_size == 0:
            eliminate_from_dormancy_pool_num = 0
            survival_in_dormancy_pool_num = 0
            offspring_num = 0
            self.offspring_pool = []
            self.immigrant_pool = []
        
        elif offspring_num > self.dormancy_pool_max_size:
            eliminate_from_dormancy_pool_num = self.dormancy_pool_max_size
            survival_in_dormancy_pool_num = 0
            offspring_num = self.dormancy_pool_max_size
            self.dormancy_pool = random.sample(self.offspring_pool, self.dormancy_pool_max_size)
            self.offspring_pool = []
            self.immigrant_pool = []
        
        elif offspring_num + dormancy_num <= self.dormancy_pool_max_size:
            eliminate_from_dormancy_pool_num = 0
            survival_in_dormancy_pool_num = dormancy_num
            self.dormancy_pool += self.offspring_pool
            self.offspring_pool = []
            self.immigrant_pool = []
        
        else:
            eliminate_from_dormancy_pool_num = offspring_num + dormancy_num - self.dormancy_pool_max_size
            survival_in_dormancy_pool_num = self.dormancy_pool_max_size - offspring_num
            survival_dormancy_pool = random.sample(self.dormancy_pool, survival_in_dormancy_pool_num)
            self.dormancy_pool = survival_dormancy_pool + self.offspring_pool
            self.offspring_pool = []
            self.immigrant_pool = []
        return survival_in_dormancy_pool_num, eliminate_from_dormancy_pool_num, offspring_num, len(self.dormancy_pool)
    
    def hab_dormancy_process_from_offspring_pool_and_immigrant_pool(self):
        offspring_num = len(self.offspring_pool)
        immigrant_num = len(self.immigrant_pool)
        dormancy_num = len(self.dormancy_pool)
    
        if self.dormancy_pool_max_size < offspring_num:
            eliminate_from_dormancy_pool_num = self.dormancy_pool_max_size
            survival_in_dormancy_pool_num = 0
            new_dormancy_num = self.dormancy_pool_max_size
            self.dormancy_pool = random.sample(self.offspring_pool, self.dormancy_pool_max_size)
            self.offspring_pool = []
            self.immigrant_pool = []
            
        elif offspring_num <= self.dormancy_pool_max_size < (offspring_num + immigrant_num):
            eliminate_from_dormancy_pool_num = self.dormancy_pool_max_size
            survival_in_dormancy_pool_num = 0
            new_dormancy_num = self.dormancy_pool_max_size
            self.dormancy_pool = self.offspring_pool + random.sample(self.immigrant_pool, self.dormancy_pool_max_size-offspring_num)
            self.offspring_pool = []
            self.immigrant_pool = []
        
        elif (offspring_num + immigrant_num) <= self.dormancy_pool_max_size < (offspring_num + immigrant_num + dormancy_num):
            eliminate_from_dormancy_pool_num = offspring_num + immigrant_num
            survival_in_dormancy_pool_num = self.dormancy_pool_max_size - eliminate_from_dormancy_pool_num
            new_dormancy_num = offspring_num + immigrant_num
            self.dormancy_pool = self.offspring_pool + self.immigrant_pool + random.sample(self.dormancy_pool, survival_in_dormancy_pool_num)
            self.offspring_pool = []
            self.immigrant_pool = []
            
        else:
            eliminate_from_dormancy_pool_num = 0
            survival_in_dormancy_pool_num = dormancy_num
            new_dormancy_num = offspring_num + immigrant_num
            self.dormancy_pool += self.immigrant_pool + self.offspring_pool
            self.offspring_pool = []
            self.immigrant_pool = []
        return survival_in_dormancy_pool_num, eliminate_from_dormancy_pool_num, new_dormancy_num, len(self.dormancy_pool)
    
# ******************** clearing up offspring_and_immigrant_pool when dormancy pool do not run *******************************************************************#
    def hab_clear_up_offspring_and_immigrant_pool(self):
        self.offspring_pool = []
        self.immigrant_pool = []
    
    def hab_clear_up_offspring_marker_and_immigrant_marker_pool(self):
        self.offspring_marker_pool = []
        self.immigrant_marker_pool = []
    
# ******************************************************************************************************************            
############################################## class patch #################################################################
class patch():
    def __init__(self, patch_name, patch_index, location):
        self.name = patch_name
        self.index = patch_index
        self.set = {}            # self.data_set={} # to be improved
        self.hab_num = 0
        self.hab_id_ls = []
        self.location = location
        
    def get_data(self):
        output = {}
        for key, value in self.set.items():
            output[key]=value.set
        return output

    def add_habitat(self, hab_name, hab_index, hab_location, num_env_types, env_types_name, mean_env_ls, var_env_ls, length, width, dormancy_pool_max_size):
        h_object = habitat(hab_name, hab_index, hab_location, num_env_types, env_types_name, mean_env_ls, var_env_ls, length, width, dormancy_pool_max_size)
        self.set[hab_name] = h_object
        self.hab_id_ls.append(hab_name)
        self.hab_num += 1
        
    def patch_initialize(self, traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls):
        for h_id, h_object in self.set.items():
            h_object.hab_initialize(traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls)
        return 0    

    def patch_dead_selection(self, base_dead_rate, fitness_wid, method):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_dead_selection(base_dead_rate, fitness_wid, method)
        return counter

    def get_patch_individual_num(self):
        num = 0
        for key, value in self.set.items():
            num += value.indi_num
        return num

    def get_patch_empty_sites_ls(self):
        ''' return patch_empty_pos_ls as [(h_id, len_id, wid_id)] '''
        patch_empty_pos_ls = []
        for h_id, h_object in self.set.items():
            empty_site_pos_ls = h_object.empty_site_pos_ls
            for site_pos in empty_site_pos_ls:
                site_pos = (h_id, ) + site_pos
                patch_empty_pos_ls.append(site_pos)
        return patch_empty_pos_ls

    def patch_empty_sites_num(self):
        ''' return the number of empty microsite in the patches '''
        return len(self.get_patch_empty_sites_ls())

    def get_patch_pairwise_empty_sites_ls(self):
        ''' return patch_empty_pos_ls as [((h_id, len_id, wid_id), (h_id, len_id, wid_id))...] '''
        patch_pairwise_empty_pos_ls = []
        for h_id, h_object in self.set.items():
            pairwise_empty_sites_pos_ls = h_object.get_hab_pairwise_empty_site_pos_ls()
            for (empty_site_1_pos, empty_site_2_pos) in pairwise_empty_sites_pos_ls:
                empty_site_1_pos = (h_id, ) + empty_site_1_pos
                empty_site_2_pos = (h_id, ) + empty_site_2_pos
                patch_pairwise_empty_pos_ls.append((empty_site_1_pos, empty_site_2_pos))
        return patch_pairwise_empty_pos_ls
    
    def get_patch_pairwise_occupied_sites_ls(self):
        ''' return patch_empty_pos_ls as [((h_id, len_id, wid_id), (h_id, len_id, wid_id))...] '''
        patch_pairwise_occupied_pos_ls = []
        for h_id, h_object in self.set.items():
            pairwise_occupied_sites_pos_ls = h_object.get_hab_pairwise_occupied_site_pos_ls()
            for (occupied_site_1_pos, occupied_site_2_pos) in pairwise_occupied_sites_pos_ls:
                occupied_site_1_pos = (h_id, ) + occupied_site_1_pos
                occupied_site_2_pos = (h_id, ) + occupied_site_2_pos
                patch_pairwise_occupied_pos_ls.append((occupied_site_1_pos, occupied_site_2_pos))
        return patch_pairwise_occupied_pos_ls
    
    def get_patch_occupied_sites_ls(self):
        ''' return patch_occupied_pos_ls as [(h_id, len_id, wid_id)] '''
        patch_occupied_pos_ls = []
        for h_id, h_object in self.set.items():
            occupied_site_pos_ls = h_object.occupied_site_pos_ls
            for site_pos in occupied_site_pos_ls:
                site_pos = (h_id, ) + site_pos
                patch_occupied_pos_ls.append(site_pos)
        return patch_occupied_pos_ls
    
    def get_patch_offspring_marker_pool(self):
        ''' get the combination of all the offspring marker pool in all habitat in the patch 
        return as a list of offspring marker. '''
        patch_offsprings_marker_pool = []
        for h_id, h_object in self.set.items():
            patch_offsprings_marker_pool += h_object.offspring_marker_pool
        return patch_offsprings_marker_pool
    
    def get_patch_offspring_pool(self):
        ''' get the combination of all the offspring pool in all habitat in the patch 
        return as a list of offspring individual object. '''
        patch_offsprings_pool = []
        for h_id, h_object in self.set.items():
            patch_offsprings_pool += h_object.offspring_pool
        return patch_offsprings_pool
    
    def get_patch_dormancy_pool(self):
        ''' get the combination of all the dormancy pool in all habitat in the patch 
        return as a list of dormancy individual object. '''
        patch_dormancy_pool = []
        for h_id, h_object in self.set.items():
            patch_dormancy_pool += h_object.dormancy_pool
        return patch_dormancy_pool
    
    def get_patch_offspring_and_dormancy_pool(self):
        ''' get the combination of all the offspring_pool_and_dormancy_pool in all habitat in the patch 
        return as a list of offspring_and_dormancy individual object. '''
        patch_offspring_and_dormancy_pool = []
        for h_id, h_object in self.set.items():
            patch_offspring_and_dormancy_pool += h_object.offspring_pool
            patch_offspring_and_dormancy_pool += h_object.dormancy_pool
        return patch_offspring_and_dormancy_pool
    
    def get_patch_microsites_individals_sp_id_values(self):
        ''' get species_id, phenotypes distribution in the patch as values set '''
        values_set = np.array([], dtype=int)
        for h_id, h_object in self.set.items():
            hab_len = h_object.length
            hab_wid = h_object.width
            for row in range(hab_len):
                for col in range(hab_wid):
                    individual_object = h_object.set['microsite_individuals'][row][col]
                    if individual_object ==None:
                        values_set = np.append(values_set, np.nan)
                    else:
                        species_id = individual_object.species_id
                        species_id_value = int(re.findall(r"\d+",species_id)[0])
                        values_set = np.append(values_set, species_id_value)
        values_set = values_set.reshape(self.hab_num, h_object.size)
        return values_set
    
    def get_patch_microsites_individals_phenotype_values(self, trait_name):
        ''' get species_id, phenotypes distribution in the patch as values set '''
        values_set = []
        for h_id, h_object in self.set.items():
            hab_len = h_object.length
            hab_wid = h_object.width
            for row in range(hab_len):
                for col in range(hab_wid):
                    individual_object = h_object.set['microsite_individuals'][row][col]
                    if individual_object ==None:
                        values_set.append(np.nan)
                    else:
                        phenotype = individual_object.phenotype_set[trait_name]
                        values_set.append(phenotype)
        values_set = np.array(values_set).reshape(self.hab_num, h_object.size)
        return values_set
    
    def get_patch_microsites_environment_values(self, environment_name):
        ''' get microsite environment values distribution in the patch as values set '''
        values_array = np.array([])
        for h_id, h_object in self.set.items():
            hab_environment_values_array = h_object.set[environment_name] 
            values_array = np.append(values_array, hab_environment_values_array) # dimension of the return of np.append() is always in dim=1
        values_array = values_array.reshape(self.hab_num, h_object.size)
        return values_array
    
    
##********************* for patch in mainland only and do not contains dormant bank *******************************####
    def patch_asexual_birth_germinate(self, asexual_birth_rate, mutation_rate, pheno_var_ls):
        ''' birth into empty site directly without considering the competition between local offspring and immigrant offspring '''
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_asexual_reprodece_germinate(asexual_birth_rate, mutation_rate, pheno_var_ls)
        return counter
    
    def patch_sexual_birth_germinate(self, sexual_birth_rate, mutation_rate, pheno_var_ls):
        ''' birth into empty site directly without considering the competition between local offspring and immigrant offspring '''
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_sexual_reprodece_germinate(sexual_birth_rate, mutation_rate, pheno_var_ls)
        return counter
    
    def patch_mixed_birth_germinate(self, asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls):
        ''' birth into empty site directly without considering the competition between local offspring and immigrant offspring '''
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_mixed_reproduce_germinate(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls)
        return counter
    
#** calculating the offspring num and generating offspring markers but do not reproduce actually until local germination process afterward after dispersal process.*
    def patch_asex_reproduce_calculation_into_offspring_marker_pool(self, asexual_birth_rate):
        ''' used only when dormancy process do not occur '''
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_asex_reproduce_calculation_into_offspring_marker_pool(self.name, asexual_birth_rate)
        return counter
    
    def patch_sex_reproduce_calculation_into_offspring_marker_pool(self, sexual_birth_rate):
        ''' used only when dormancy process do not occur '''
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_sex_reproduce_calculation_into_offspring_marker_pool(self.name, sexual_birth_rate)
        return counter
    
    def patch_mix_reproduce_calculation_into_offspring_marker_pool(self, asexual_birth_rate, sexual_birth_rate):
        ''' used only when dormancy process do not occur '''
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_mix_reproduce_calculation_into_offspring_marker_pool(self.name, asexual_birth_rate, sexual_birth_rate)
        return counter
        
#********************************************************************************************************************#
#************************ all hab in patch reproduce into offspring pool ********************************************#
    def patch_asex_reproduce_mutate_into_offspring_pool(self, asexual_birth_rate, mutation_rate, pheno_var_ls):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_asex_reproduce_mutate_into_offspring_pool(asexual_birth_rate, mutation_rate, pheno_var_ls)
        return counter
    
    def patch_sex_reproduce_mutate_into_offspring_pool(self, sexual_birth_rate, mutation_rate, pheno_var_ls):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_sex_reproduce_mutate_into_offspring_pool(sexual_birth_rate, mutation_rate, pheno_var_ls)
        return counter
    
    def patch_mix_reproduce_mutate_into_offspring_pool(self, asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_mix_reproduce_mutate_into_offspring_pool(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls)
        return counter
#****************************************** dispersal among patch into habs offspring pool *******************************************************
    def split(self, ls, n):
        ''' split a ls into n pieces '''
        k, m = divmod(len(ls), n)
        divided_ls = list(ls[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
        random.shuffle(divided_ls)
        return divided_ls

    def migrants_to_patch_into_habs_immigrant_pool(self, migrants_indi_obj_to_patch_ls):
        divide_into_habs_indi_obj_ls = self.split(migrants_indi_obj_to_patch_ls, self.hab_num)
        for h_id, h_object in self.set.items():
            h_object.immigrant_pool += divide_into_habs_indi_obj_ls[h_object.index]
        return 0
    
    def migrants_marker_to_patch_into_habs_immigrant_marker_pool(self, migrants_to_j_offspring_marker_ls):
        divide_into_habs_offs_marker_ls = self.split(migrants_to_j_offspring_marker_ls, self.hab_num)
        for h_id, h_object in self.set.items():
            h_object.immigrant_marker_pool += divide_into_habs_offs_marker_ls[h_object.index]
        return 0
#****************************************** dispersal within patch ******************************************************
    def patch_matrix_around(self, matrix):   
        ''' 元素的小数部分，按照概率四舍五入 '''
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isnan(matrix[i, j]) == False:
                    integer = int(matrix[i,j])
                    demacial = matrix[i,j] - integer
                    if np.random.uniform(0,1,1)[0] <= demacial:
                        integer += 1
                    else:
                        pass
                else:
                    integer = 0
                matrix[i, j] = integer
        return matrix
    
    def get_patch_dispersal_within_rate_matrix(self, disp_within_rate):
        dispersal_within_rate_matrix = np.matrix(np.ones((self.hab_num, self.hab_num)))*disp_within_rate/(self.hab_num-1)
        dispersal_within_rate_matrix[np.diag_indices_from(dispersal_within_rate_matrix)] = 1-disp_within_rate  # elements in diagonal_line = 1-m
        return dispersal_within_rate_matrix

    def get_patch_habs_offspring_num_matrix(self):
        habs_offspring_num_matrix = np.matrix(np.zeros((self.hab_num, self.hab_num)))
        for h_id, h_object in self.set.items():
            offspring_num = len(h_object.offspring_pool)
            habs_offspring_num_matrix[h_object.index, h_object.index] = offspring_num
        return habs_offspring_num_matrix
    
    def get_patch_habs_offspring_marker_num_matrix(self):
        habs_offspring_marker_num_matrix = np.matrix(np.zeros((self.hab_num, self.hab_num)))
        for h_id, h_object in self.set.items():
            offspring_marker_num = len(h_object.offspring_marker_pool)
            habs_offspring_marker_num_matrix[h_object.index, h_object.index] = offspring_marker_num
        return habs_offspring_marker_num_matrix
    
    def get_patch_habs_dormancy_num_matrix(self):
        habs_dormancy_num_matrix = np.matrix(np.zeros((self.hab_num, self.hab_num)))
        for h_id, h_object in self.set.items():
            dormancy_num = len(h_object.dormancy_pool)
            habs_dormancy_num_matrix[h_object.index, h_object.index] = dormancy_num
        return habs_dormancy_num_matrix

    def get_patch_habs_empty_sites_num_matrix(self):
        habs_empty_sites_num_matrix = np.matrix(np.zeros((self.hab_num, self.hab_num)))
        for h_id, h_object in self.set.items():
            empty_site_num = len(h_object.empty_site_pos_ls)
            habs_empty_sites_num_matrix[h_object.index, h_object.index] = empty_site_num
        return habs_empty_sites_num_matrix
    
    def get_habs_emigrants_matrix(self, disp_within_rate):
        habs_offspring_num_matrix = self.get_patch_habs_offspring_num_matrix()
        habs_dormancy_num_matrix = self.get_patch_habs_dormancy_num_matrix()
        dispersal_within_rate_matrix = self.get_patch_dispersal_within_rate_matrix(disp_within_rate)
        return (habs_offspring_num_matrix + habs_dormancy_num_matrix) * dispersal_within_rate_matrix
    
    def get_habs_immigrants_matrix(self, disp_within_rate):
        habs_emigrants_matrix = self.get_habs_emigrants_matrix(disp_within_rate)
        habs_empty_sites_num_matrix = self.get_patch_habs_empty_sites_num_matrix()
        return (habs_emigrants_matrix/habs_emigrants_matrix.sum(axis=0))*habs_empty_sites_num_matrix
    
    def get_dispersal_within_num_matrix(self, disp_within_rate):
        habs_emigrants_matrix = self.get_habs_emigrants_matrix(disp_within_rate)
        habs_immigrants_matrix = self.get_habs_immigrants_matrix(disp_within_rate)
        return self.patch_matrix_around(np.minimum(habs_emigrants_matrix, habs_immigrants_matrix))

    def patch_dipersal_within_from_offspring_and_dormancy_pool(self, disp_within_rate):
        disp_within_num_matrix = self.get_dispersal_within_num_matrix(disp_within_rate)
        counter = 0
        for j in range(self.hab_num):
            h_j_id = self.hab_id_ls[j]
            h_j_object = self.set[h_j_id]
            h_j_empty_site_ls = h_j_object.empty_site_pos_ls
            migrants_to_j_indi_object_ls = []
            
            for i in range(self.hab_num):
                h_i_id = self.hab_id_ls[i]
                h_i_object = self.set[h_i_id]
                if i==j:
                    continue
                else:
                    migrants_i_j_num = int(disp_within_num_matrix[i, j])
                    hab_i_offspring_and_dormancy_pool = h_i_object.offspring_pool + h_i_object.dormancy_pool
                    migrants_to_j_indi_object_ls += random.sample(hab_i_offspring_and_dormancy_pool, migrants_i_j_num)
                    
            random.shuffle(h_j_empty_site_ls)
            random.shuffle(migrants_to_j_indi_object_ls)
            for (len_id, wid_id), migrants_object in list(zip(h_j_empty_site_ls, migrants_to_j_indi_object_ls)):
                self.set[h_j_id].add_individual(indi_object = migrants_object, len_id=len_id, wid_id=wid_id)
                counter += 1
        return counter
    
    def patch_dispersal_within_from_offspring_marker_pool_to_immigrant_marker_pool(self, disp_within_rate):
        offspring_marker_dispersal_matrix =  self.patch_matrix_around(self.get_patch_habs_offspring_marker_num_matrix()*self.get_patch_dispersal_within_rate_matrix(disp_within_rate))
        counter = 0
        for j in range(self.hab_num):
            h_j_id = self.hab_id_ls[j]
            h_j_object = self.set[h_j_id]
            migrants_to_j_offspring_marker_ls = []
            
            for i in range(self.hab_num):
                h_i_id = self.hab_id_ls[i]
                h_i_object = self.set[h_i_id]
                if i==j:
                    continue
                else:
                    migrants_i_j_num = int(offspring_marker_dispersal_matrix[i, j])
                    hab_i_offspring_marker_pool = h_i_object.offspring_marker_pool
                    migrants_to_j_offspring_marker_ls += random.sample(hab_i_offspring_marker_pool, migrants_i_j_num)
                    
            random.shuffle(migrants_to_j_offspring_marker_ls)
            h_j_object.immigrant_marker_pool += migrants_to_j_offspring_marker_ls
            counter += len(migrants_to_j_offspring_marker_ls)
        return counter
    
    def patch_dispersal_within_from_offspring_pool_to_immigrant_pool(self, disp_within_rate):
        offspring_dispersal_matrix = self.patch_matrix_around(self.get_patch_habs_offspring_num_matrix()*self.get_patch_dispersal_within_rate_matrix(disp_within_rate))
        counter = 0
        for j in range(self.hab_num):
            h_j_id = self.hab_id_ls[j]
            h_j_object = self.set[h_j_id]
            migrants_to_j_indi_object_ls = []
            
            for i in range(self.hab_num):
                h_i_id = self.hab_id_ls[i]
                h_i_object = self.set[h_i_id]
                if i==j: 
                    continue
                else:
                    migrants_i_j_num = int(offspring_dispersal_matrix[i, j])
                    hab_i_offspring_pool = h_i_object.offspring_pool
                    migrants_to_j_indi_object_ls += random.sample(hab_i_offspring_pool, migrants_i_j_num)
                    
            random.shuffle(migrants_to_j_indi_object_ls)
            h_j_object.immigrant_pool += migrants_to_j_indi_object_ls
            counter += len(migrants_to_j_indi_object_ls)
        return counter
    
    def patch_dispersal_within_from_offspring_pool_and_dormancy_pool_to_immigrant_pool(self, disp_within_rate):
        offspring_and_dormancy_dispersal_matrix =  self.patch_matrix_around(self.get_habs_emigrants_matrix(disp_within_rate))
        counter = 0
        for j in range(self.hab_num):
            h_j_id = self.hab_id_ls[j]
            h_j_object = self.set[h_j_id]
            migrants_to_j_indi_object_ls = []
            
            for i in range(self.hab_num):
                h_i_id = self.hab_id_ls[i]
                h_i_object = self.set[h_i_id]
                if i==j:
                    continue
                else:
                    migrants_i_j_num = int(offspring_and_dormancy_dispersal_matrix[i, j])
                    hab_i_offspring_and_dormancy_pool = h_i_object.offspring_pool + h_i_object.dormancy_pool
                    migrants_to_j_indi_object_ls += random.sample(hab_i_offspring_and_dormancy_pool, migrants_i_j_num)
                    
            random.shuffle(migrants_to_j_indi_object_ls)
            h_j_object.immigrant_pool += migrants_to_j_indi_object_ls
            counter += len(migrants_to_j_indi_object_ls)
        return counter        
        
#******************** local germination in all habitat in the patch *****************************#
    def patch_local_germinate_from_offspring_and_dormancy_pool(self):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_local_germinate_from_offspring_and_dormancy_pool()
        return counter
    
    def patch_local_germinate_from_offspring_and_immigrant_pool(self):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_local_germinate_from_offspring_and_immigrant_pool()
        return counter
            
    def patch_local_germinate_from_offspring_immigrant_and_dormancy_pool(self):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_local_germinate_from_offspring_immigrant_and_dormancy_pool()
        return counter

#********************************** dormancy process in the patch **********************************
    def patch_dormancy_process_from_offspring_pool_to_dormancy_pool(self):
        survival_counter = 0
        eliminate_counter = 0
        new_dormancy_counter = 0
        all_dormancy_num = 0
        for h_id, h_object in self.set.items():
            survival_num, eliminate_num, new_dormancy_num, dormancy_num = h_object.hab_dormancy_process_from_offspring_pool_to_dormancy_pool()
            survival_counter += survival_num
            eliminate_counter += eliminate_num
            new_dormancy_counter += new_dormancy_num
            all_dormancy_num += dormancy_num
        return survival_counter, eliminate_counter, new_dormancy_counter, all_dormancy_num
    
    def patch_dormancy_process_from_offspring_pool_and_immigrant_pool(self):
        survival_counter = 0
        eliminate_counter = 0
        new_dormancy_counter = 0
        all_dormancy_num = 0
        for h_id, h_object in self.set.items():
            survival_num, eliminate_num, new_dormancy_num, dormancy_num = h_object.hab_dormancy_process_from_offspring_pool_and_immigrant_pool()
            survival_counter += survival_num
            eliminate_counter += eliminate_num
            new_dormancy_counter += new_dormancy_num
            all_dormancy_num += dormancy_num
        return survival_counter, eliminate_counter, new_dormancy_counter, all_dormancy_num    
    
# ******************** clearing up offspring_and_immigrant_pool when dormancy pool do not run *******************************************************************#
    def patch_clear_up_offspring_and_immigrant_pool(self):
        for h_id, h_object in self.set.items():
            h_object.hab_clear_up_offspring_and_immigrant_pool()
    
    def patch_clear_up_offspring_marker_and_immigrant_marker_pool(self):
        for h_id, h_object in self.set.items():   
            h_object.hab_clear_up_offspring_marker_and_immigrant_marker_pool()
    
#********************************** disturbance process in the patch ************************************
    def patch_disturbance_process(self):
        for h_id, h_object in self.set.items():
            h_object.habitat_disturbance_process()
            
#********************************** data generating for saving ******************************************
    def get_patch_microsites_optimum_sp_id_value_array(self, d, w, species_2_phenotype_ls):
        ''''''
        values_array = np.array([], dtype=int)
        for h_id, h_object in self.set.items():
            hab_len = h_object.length
            hab_wid = h_object.width
            mean_env_tuple = h_object.mean_env_ls
            
            hab_opt_survival_rate = 0
            hab_opt_sp_id_val = np.nan
            for phenotype_ls in species_2_phenotype_ls:
                survival_rate = h_object.survival_rate(d=d, phenotype_ls=phenotype_ls, env_val_ls=mean_env_tuple, w=w)
                if survival_rate > hab_opt_survival_rate:
                    hab_opt_survival_rate = survival_rate
                    hab_opt_sp_id_val = species_2_phenotype_ls.index(phenotype_ls)+1

            hab_sp_id_val_array = np.ones(hab_len*hab_wid, dtype=int)*hab_opt_sp_id_val
            values_array = np.append(values_array, hab_sp_id_val_array)
        
        values_array = values_array.reshape(self.hab_num, h_object.size)
        return values_array

########################################## class metacommunity #############################################################
class metacommunity():
    def __init__(self, metacommunity_name):
        self.set = {}                       # self.data_set={} # to be improved
        self.patch_num = 0
        #self.meta_map = nx.Graph()
        self.metacommunity_name = metacommunity_name
        self.patch_id_ls = []
        self.patch_id_2_index_dir = {}
        self.disp_current_matrix = np.matrix([])

    def get_data(self):
        output = {}
        for key, value in self.set.items():
            output[key]=value.get_data()
        return output
    
    def __str__(self):
        return str(self.get_data())
        
    def add_patch(self, patch_name, patch_object):
        ''' add new patch to the metacommunity. '''
        self.set[patch_name] = patch_object
        self.patch_num += 1
        #self.meta_map.add_node(patch_name)
        self.patch_id_ls.append(patch_name)
        self.patch_id_2_index_dir[patch_name] = patch_object.index
        self.disp_current_matrix = np.matrix(np.zeros((self.patch_num, self.patch_num)))
        
    def reshape_habitat_data_in_patch(self, df, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, mask_loc=None):
        ''' reshape habitat_data in the a coorderation order for plotting'''
        reshape_data_col_row = np.empty((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 0))
        mask_data_col_row = np.empty((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 0))
        
        for x_loc in range(0, hab_num_x_axis_in_patch): #x坐标从原点开始
            reshape_data_col = np.empty((0, hab_x_len))
            mask_data_col = np.empty((0, hab_x_len))
            
            for y_loc in range(0, hab_num_y_axis_in_patch): #y坐标从原点开始
                h_index = x_loc*hab_num_y_axis_in_patch + y_loc
                hab_data = df.loc[h_index].to_numpy().reshape(hab_y_len, hab_x_len)
                mask_data = np.zeros((hab_y_len, hab_x_len))
                
                if mask_loc=='lower': 
                    mask_data[np.tril_indices_from(mask_data)] = True #下三角遮盖
                    
                elif mask_loc=='upper': 
                    mask_data[np.tril_indices_from(mask_data)] = True #上三角遮盖
                    mask_data = mask_data.T
                
                if y_loc == 0:
                    #hab_gap_for_vstack = np.ones((0, hab_x_len)) * np.nan
                    #hab_gap_for_vstack = np.empty((0, hab_x_len))
                    hab_gap_for_vstack = np.zeros((0, hab_x_len))
                    mask_gap_for_vstack = np.ones((0, hab_x_len))
                    
                else:
                    #hab_gap_for_vstack = np.ones((1, hab_x_len)) * np.nan
                    #hab_gap_for_vstack = np.empty((1, hab_x_len))
                    hab_gap_for_vstack = np.zeros((1, hab_x_len))
                    mask_gap_for_vstack = np.ones((1, hab_x_len))
                    
                reshape_data_col = np.vstack((hab_data, hab_gap_for_vstack, reshape_data_col))
                mask_data_col = np.vstack((mask_data, mask_gap_for_vstack, mask_data_col))
                
            if x_loc ==0:
                #hab_gap_for_hstack = np.ones((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 0)) * np.nan
                #hab_gap_for_hstack = np.empty((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 0)) 
                hab_gap_for_hstack = np.zeros((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 0)) 
                mask_gap_for_hstack = np.ones((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 0)) 
            else:
                #hab_gap_for_hstack = np.ones((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 1)) * np.nan
                #hab_gap_for_hstack = np.empty((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 1)) 
                hab_gap_for_hstack = np.zeros((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 1))
                mask_gap_for_hstack = np.ones((hab_y_len*hab_num_y_axis_in_patch+hab_num_y_axis_in_patch-1, 1)) 
                
            reshape_data_col_row = np.hstack((reshape_data_col_row, hab_gap_for_hstack, reshape_data_col))
            mask_data_col_row = np.hstack((mask_data_col_row, mask_gap_for_hstack, mask_data_col))
        return reshape_data_col_row, mask_data_col_row        
    
    def meta_show_species_distribution(self, sub_row, sub_col, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, vmin, vmax, cmap, file_name):
        fig = plt.figure(figsize=(40, 40))

        for patch_id, patch_object in self.set.items():
            #location = patch_object.index + 1
            location = sub_col*(sub_row-1)+1+(patch_object.index//sub_row)-(patch_object.index % sub_row)*sub_col
            ax = fig.add_subplot(sub_row, sub_col, location)
            ax.set_title(patch_id, fontsize = 40)
            plt.tight_layout()

            df = pd.DataFrame(patch_object.get_patch_microsites_individals_sp_id_values())
            
            reshape_numpy_data, mask = self.reshape_habitat_data_in_patch(df, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len)
            sns.heatmap(data=reshape_numpy_data, vmin=vmin, vmax=vmax, cbar=False, mask=mask, cmap=cmap, annot=True)
            
        plt.savefig(file_name)
        plt.clf()
        return 0
    
    def meta_show_species_phenotype_distribution(self, trait_name, sub_row, sub_col, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, cmap, file_name):
        fig = plt.figure(figsize=(40, 40))

        for patch_id, patch_object in self.set.items():
            #location = patch_object.index + 1
            location = sub_col*(sub_row-1)+1+(patch_object.index//sub_row)-(patch_object.index % sub_row)*sub_col
            ax = fig.add_subplot(sub_row, sub_col, location)
            ax.set_title(patch_id, fontsize = 40)
            plt.tight_layout()

            df = pd.DataFrame(patch_object.get_patch_microsites_individals_phenotype_values(trait_name))
            
            reshape_numpy_data, mask = self.reshape_habitat_data_in_patch(df, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len)
            sns.heatmap(data=reshape_numpy_data, vmin=0, vmax=0.8, cbar=False, mask=mask, cmap=cmap)
            
        plt.savefig(file_name)
        plt.clf()
        return 0
    
    def meta_show_environment_distribution(self, environment_name, sub_row, sub_col, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, mask_loc, cmap, file_name):
        fig = plt.figure(figsize=(40, 40))

        for patch_id, patch_object in self.set.items():
            #location = patch_object.index + 1
            location = sub_col*(sub_row-1)+1+(patch_object.index//sub_row)-(patch_object.index % sub_row)*sub_col
            ax = fig.add_subplot(sub_row, sub_col, location)
            ax.set_title(patch_id, fontsize = 80/((sub_row)/4))
            plt.tight_layout()

            df = pd.DataFrame(patch_object.get_patch_microsites_environment_values(environment_name))
            
            reshape_numpy_data, mask = self.reshape_habitat_data_in_patch(df, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, mask_loc)
            sns.heatmap(data=reshape_numpy_data, vmin=0, vmax=0.8, cbar=False, mask=mask, cmap=cmap)
            
        plt.savefig(file_name)
        plt.clf()
        return 0
    
    def meta_show_two_environment_distribution(self, environment1_name, environment2_name, sub_row, sub_col, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, mask_loc1, mask_loc2, cmap1, cmap2, file_name):
        fig = plt.figure(figsize=(40, 40))

        for patch_id, patch_object in self.set.items():
            #location = patch_object.index + 1
            location = sub_col*(sub_row-1)+1+(patch_object.index//sub_row)-(patch_object.index % sub_row)*sub_col
            ax = fig.add_subplot(sub_row, sub_col, location)
            ax.set_title(patch_id, fontsize = 80/((sub_row)/4))
            plt.tight_layout()

            df1 = pd.DataFrame(patch_object.get_patch_microsites_environment_values(environment1_name))
            reshape_numpy_data, mask = self.reshape_habitat_data_in_patch(df1, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, mask_loc1)
            sns.heatmap(data=reshape_numpy_data, vmin=0, vmax=0.8, cbar=False, mask=mask, cmap=cmap1)
            
            df2 = pd.DataFrame(patch_object.get_patch_microsites_environment_values(environment2_name))
            reshape_numpy_data, mask = self.reshape_habitat_data_in_patch(df2, hab_num_x_axis_in_patch, hab_num_y_axis_in_patch, hab_y_len, hab_x_len, mask_loc2)
            sns.heatmap(data=reshape_numpy_data, vmin=0, vmax=0.8, cbar=False, mask=mask, cmap=cmap2)
            
            
        plt.savefig(file_name)
        plt.clf()
        return 0
    
    def meta_initialize(self, traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls):
        for patch_id, patch_object in self.set.items():
            patch_object.patch_initialize(traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls)
        log_info = '%s initialization done! \n'%(self.metacommunity_name)
        #print(log_info)
        return log_info

    def get_meta_individual_num(self):
        num = 0
        for patch_id, patch_object in self.set.items():
            num += patch_object.get_patch_individual_num()
        return num
    
    def get_meta_empty_sites_ls(self):
        ''' return meta_empty_sites_ls as [(patch_id, h_id, len_id, wid_id)] '''   
        meta_empty_sites_ls = []
        for patch_id, patch_object in self.set.items():
            patch_empty_pos_ls = patch_object.get_patch_empty_sites_ls()
            for empty_pos in patch_empty_pos_ls:
                empty_pos = (patch_id, ) + empty_pos
                meta_empty_sites_ls.append(empty_pos)
        return meta_empty_sites_ls

    def show_meta_empty_sites_num(self):
        return len(self.get_meta_empty_sites_ls())
    
    def get_meta_pairwise_empty_sites_ls(self):
        ''' return meta_empty_sites_ls as [((patch_id, h_id, len_id, wid_id), (patch_id, h_id, len_id, wid_id))...],
        pairwise means the same habitat '''   
        meta_pairwise_empty_sites_ls = []
        for patch_id, patch_object in self.set.items():
            patch_pairwise_empty_pos_ls = patch_object.get_patch_pairwise_empty_sites_ls()
            for (empty_site_1_pos, empty_site_2_pos) in patch_pairwise_empty_pos_ls:
                empty_site_1_pos = (patch_id, ) + empty_site_1_pos
                empty_site_2_pos = (patch_id, ) + empty_site_2_pos
                meta_pairwise_empty_sites_ls.append((empty_site_1_pos, empty_site_2_pos))
        return meta_pairwise_empty_sites_ls
    
    def meta_get_occupied_location_ls(self):
        ''' return as [(patch_id, h_id, row_id, col_id)] '''
        meta_occupied_sites_ls = []
        for patch_id, patch_object in self.set.items():
            patch_empty_pos_ls = patch_object.get_patch_occupied_sites_ls()
            for site_pos in patch_empty_pos_ls:
                site_pos = (patch_id, ) + site_pos
                meta_occupied_sites_ls.append(site_pos)
        return meta_occupied_sites_ls
    
    def get_meta_pairwise_occupied_sites_ls(self):
        ''' return meta_occupied_sites_ls as [((patch_id, h_id, len_id, wid_id), (patch_id, h_id, len_id, wid_id))...],
        pairwise means the same species ((female_pos), (male_pos)) '''   
        meta_pairwise_occupied_sites_ls = []
        for patch_id, patch_object in self.set.items():
            patch_pairwise_occupied_pos_ls = patch_object.get_patch_pairwise_occupied_sites_ls()
            for (occupied_site_1_pos, occupied_site_2_pos) in patch_pairwise_occupied_pos_ls:
                occupied_site_1_pos = (patch_id, ) + occupied_site_1_pos
                occupied_site_2_pos = (patch_id, ) + occupied_site_2_pos
                meta_pairwise_occupied_sites_ls.append((occupied_site_1_pos, occupied_site_2_pos))
        return meta_pairwise_occupied_sites_ls  
    #****************************************************#
    def meta_offspring_pool_individual_num(self):
        counter = 0
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                counter += len(h_object.offspring_pool)
        return counter
    
    def meta_immigrant_pool_individual_num(self):
        counter = 0
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                counter += len(h_object.immigrant_pool)
        return counter
    
    def meta_dormancy_pool_individual_num(self):
        counter = 0
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                counter += len(h_object.dormancy_pool)
        return counter
    #****************************************************#
    def meta_offspring_marker_pool_marker_num(self):
        counter = 0
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                counter += len(h_object.offspring_marker_pool)
        return counter
    
    def meta_immigrant_marker_pool_marker_num(self):
        counter = 0
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                counter += len(h_object.immigrant_marker_pool)
        return counter 
    
# ********************************* dead selection in metacommunity ***********************************************
    def meta_dead_selection(self, base_dead_rate, fitness_wid, method):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_dead_selection(base_dead_rate, fitness_wid, method)
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Dead selection] in %s: there are %d individuals dead in selection; there are %d individuals in the %s; there are %d empty sites in the %s \n'%(self.metacommunity_name, counter, indi_num, self.metacommunity_name, empty_sites_num, self.metacommunity_name)
        #print(log_info)
        return log_info
    
#************************** for mainland only and do not contain dormant bank **************************************#
    def meta_mainland_asexual_birth_mutate_germinate(self, asexual_birth_rate, mutation_rate, pheno_var_ls):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_asexual_birth_germinate(asexual_birth_rate, mutation_rate, pheno_var_ls)
        
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Birth process] in %s: there are %d individuals germinating from local habitat; there are %d individuals in the %s; there are %d empty sites in the %s \n'%(self.metacommunity_name, counter, indi_num, self.metacommunity_name, empty_sites_num, self.metacommunity_name)
        #print(log_info)
        return log_info
    
    def meta_mainland_sexual_birth_mutate_germinate(self, sexual_birth_rate, mutation_rate, pheno_var_ls):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_sexual_birth_germinate(sexual_birth_rate, mutation_rate, pheno_var_ls)
            
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Birth process] in %s: there are %d individuals germinating from local habitat; there are %d individuals in the %s; there are %d empty sites in the %s \n'%(self.metacommunity_name, counter, indi_num, self.metacommunity_name, empty_sites_num, self.metacommunity_name)
        #print(log_info)
        return log_info
    
    def meta_mainland_mixed_birth_mutate_germinate(self, asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_mixed_birth_germinate(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls)
        
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Birth process] in %s: there are %d individuals germinating from local habitat; there are %d individuals in the %s; there are %d empty sites in the %s \n'%(self.metacommunity_name, counter, indi_num, self.metacommunity_name, empty_sites_num, self.metacommunity_name)
        #print(log_info)
        return log_info    
#*******************************************************************************************************************#  
#****************************************colonize_via_propagules_rains from mainland ********************************#  
    def meta_colonize_from_propagules_rains(self, mainland_obj, propagules_rain_num):
        mainland_occupied_sites_ls = mainland_obj.meta_get_occupied_location_ls()
        random.shuffle(mainland_occupied_sites_ls)
        
        int_propagules_rain_num, dem_propagules_rain_num = int(propagules_rain_num), propagules_rain_num - int(propagules_rain_num)
        if dem_propagules_rain_num >= np.random.uniform(0,1,1)[0]:
            int_propagules_rain_num += 1
            propagules_rain_num = int_propagules_rain_num
        else:
            propagules_rain_num = int_propagules_rain_num
        
        propagules_rain_pos_ls = random.sample(mainland_occupied_sites_ls, propagules_rain_num)
        meta_empty_pos_ls = self.get_meta_empty_sites_ls()
        random.shuffle(meta_empty_pos_ls)
        counter = 0
        for propagules_rain_pos, meta_empty_pos in list(zip(propagules_rain_pos_ls, meta_empty_pos_ls)):
            propagules_patch_id, propagules_h_id, propagules_row_id, propagules_col_id =  propagules_rain_pos[0], propagules_rain_pos[1], propagules_rain_pos[2], propagules_rain_pos[3]
            patch_id, h_id, len_id, wid_id = meta_empty_pos[0], meta_empty_pos[1], meta_empty_pos[2], meta_empty_pos[3]
            indi_object = mainland_obj.set[propagules_patch_id].set[propagules_h_id].set['microsite_individuals'][propagules_row_id][propagules_col_id]
            self.set[patch_id].set[h_id].add_individual(indi_object = indi_object, len_id=len_id, wid_id=wid_id)
            #self.set[patch_id].set[h_id].immigrant_pool.append(indi_object)
            counter += 1
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Colonization process]: there are %d individuals colonizing the metacommunity from mainland; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info            
        
    def pairwise_sexual_colonization_from_prpagules_rains(self, mainland_obj, propagules_rain_num):
        mainland_pairwise_occupied_sites_ls = mainland_obj.get_meta_pairwise_occupied_sites_ls()
        random.shuffle(mainland_pairwise_occupied_sites_ls)

        pairwise_propgules_num = propagules_rain_num/2
        int_pairwise_propgules_num, dem_pairwise_propgules_num = int(pairwise_propgules_num), pairwise_propgules_num-int(pairwise_propgules_num)
        if dem_pairwise_propgules_num >= np.random.uniform(0,1,1)[0]:
            int_pairwise_propgules_num += 1
            pairwise_propgules_num = int_pairwise_propgules_num
        else:
            pairwise_propgules_num = int_pairwise_propgules_num
        
        propagules_rain_pairwise_pos_ls = random.sample(mainland_pairwise_occupied_sites_ls, pairwise_propgules_num)
        meta_pairwise_empty_sites_ls = self.get_meta_pairwise_empty_sites_ls()
        random.shuffle(meta_pairwise_empty_sites_ls)
        counter = 0
        for (female_obj_pos, male_obj_pos), (empty_site_1_pos, empty_site_2_pos) in list(zip(propagules_rain_pairwise_pos_ls, meta_pairwise_empty_sites_ls)):
            
            female_propagules_patch_id, female_propagules_h_id, female_propagules_row_id, female_propagules_col_id =  female_obj_pos[0], female_obj_pos[1], female_obj_pos[2], female_obj_pos[3]
            female_obj = mainland_obj.set[female_propagules_patch_id].set[female_propagules_h_id].set['microsite_individuals'][female_propagules_row_id][female_propagules_col_id]
            site_1_patch_id, site_1_h_id, site_1_len_id, site_1_wid_id = empty_site_1_pos[0], empty_site_1_pos[1], empty_site_1_pos[2], empty_site_1_pos[3]
            self.set[site_1_patch_id].set[site_1_h_id].add_individual(indi_object = female_obj, len_id=site_1_len_id, wid_id=site_1_wid_id)
            #self.set[site_1_patch_id].set[site_1_h_id].immigrant_pool.append(female_obj)
            
            male_propagules_patch_id, male_propagules_h_id, male_propagules_row_id, male_propagules_col_id =  male_obj_pos[0], male_obj_pos[1], male_obj_pos[2], male_obj_pos[3]
            male_obj = mainland_obj.set[male_propagules_patch_id].set[male_propagules_h_id].set['microsite_individuals'][male_propagules_row_id][male_propagules_col_id]
            site_2_patch_id, site_2_h_id, site_2_len_id, site_2_wid_id = empty_site_2_pos[0], empty_site_2_pos[1], empty_site_2_pos[2], empty_site_2_pos[3]
            self.set[site_2_patch_id].set[site_2_h_id].add_individual(indi_object = male_obj, len_id=site_2_len_id, wid_id=site_2_wid_id)
            #self.set[site_2_patch_id].set[site_2_h_id].immigrant_pool.append(male_obj)
            
            counter += 2
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Colonization process] there are %d individuals colonizing the metacommunity from mainland; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
#******************** all hab in all patch in metacommunity, reproduce_mutate_process_into_offspring_pool ****************************************************#       
    def meta_asex_reproduce_mutate_into_offspring_pool(self, asexual_birth_rate, mutation_rate, pheno_var_ls):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_asex_reproduce_mutate_into_offspring_pool(asexual_birth_rate, mutation_rate, pheno_var_ls)
            
        log_info = '%s: there are %d individuals born into the offspring_pool; there are %d individuals in the offspring_pool \n'%(self.metacommunity_name, counter, self.meta_offspring_pool_individual_num())
        #print(log_info)
        return log_info
        
    def meta_sex_reproduce_mutate_into_offspring_pool(self, sexual_birth_rate, mutation_rate, pheno_var_ls):  
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_sex_reproduce_mutate_into_offspring_pool(sexual_birth_rate, mutation_rate, pheno_var_ls)
        
        log_info = '%s: there are %d individuals born into the offspring_pool; there are %d individuals in the offspring_pool \n'%(self.metacommunity_name, counter, self.meta_offspring_pool_individual_num())
        #print(log_info)
        return log_info
            
    def meta_mix_reproduce_mutate_into_offspring_pool(self, asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls):       
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_mix_reproduce_mutate_into_offspring_pool(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls)
        
        log_info = '[Reproduction into offspring_pool] in %s: there are %d individuals born into the offspring_pool; there are %d individuals in the offspring_pool \n'%(self.metacommunity_name, counter, self.meta_offspring_pool_individual_num())
        #print(log_info)
        return log_info
#******** calculating the offspring num but do not reproduce actually until local germination process afterward after dispersal process. *************
    def meta_asex_reproduce_calculation_into_offspring_marker_pool(self, asexual_birth_rate):
        ''' used only when dormancy process do not occur'''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_asex_reproduce_calculation_into_offspring_marker_pool(asexual_birth_rate)
            
        log_info = '[Reproduction into offspring_marker_pool] in %s: there are %d offspring_marker born into the offspring_marker_pool; there are %d offspring_marker in the offspring_marker_pool \n'%(self.metacommunity_name, counter, self.meta_offspring_marker_pool_marker_num())
        #print(log_info)
        return log_info
    
    def meta_sex_reproduce_calculation_with_offspring_marker_pool(self, sexual_birth_rate):
        ''' used only when dormancy process do not occur'''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_sex_reproduce_calculation_into_offspring_marker_pool(sexual_birth_rate)
            
        log_info = '[Reproduction into offspring_marker_pool] in %s: there are %d offspring_marker born into the offspring_marker_pool; there are %d offspring_marker in the offspring_marker_pool \n'%(self.metacommunity_name, counter, self.meta_offspring_marker_pool_marker_num())
        #print(log_info)
        return log_info
    
    def meta_mix_reproduce_calculation_with_offspring_marker_pool(self, asexual_birth_rate, sexual_birth_rate):
        ''' used only when dormancy process do not occur'''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_mix_reproduce_calculation_into_offspring_marker_pool(asexual_birth_rate, sexual_birth_rate)
            
        log_info = '[Reproduction into offspring_marker_pool] in %s: there are %d offspring_marker born into the offspring_marker_pool; there are %d offspring_marker in the offspring_marker_pool \n'%(self.metacommunity_name, counter, self.meta_offspring_marker_pool_marker_num())
        #print(log_info)
        return log_info
    
# ****************************** disperal among patches process *************************************************************************
    def matrix_around(self, matrix):   
        ''' 元素的小数部分，按照概率四舍五入 '''
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isnan(matrix[i, j]) == False:
                    integer = int(matrix[i,j])
                    demacial = matrix[i,j] - integer
                    if np.random.uniform(0,1,1)[0] <= demacial:
                        integer += 1
                    else:
                        pass
                else:
                    integer = 0
                matrix[i, j] = integer
        return matrix
        
    def get_disp_amomg_rate_matrix(self, total_disp_among_rate):
        ''' the element p_i_j is the propability that emigrants would disperse from patch_i to patch_j '''
        disp_amomg_matrix = np.matrix(np.ones((self.patch_num, self.patch_num))) * total_disp_among_rate/(self.patch_num-1)
        disp_amomg_matrix[np.diag_indices_from(disp_amomg_matrix)] = 1- total_disp_among_rate  # elements in diagonal_line = 1-m
        return disp_amomg_matrix
    
    def get_offspring_marker_pool_num_matrix(self):
        ''' the element o_i_i is the offspring_marker_nums in the offspring_marker_pool of patch_i '''
        offspring_marker_pool_num_matrix = np.matrix(np.zeros((self.patch_num, self.patch_num)))
        for patch_id, patch_object in self.set.items():
            patch_offspring_marker_num = 0
            for h_id, h_object in patch_object.set.items():
                patch_offspring_marker_num += len(h_object.offspring_marker_pool)
            offspring_marker_pool_num_matrix[patch_object.index, patch_object.index] = patch_offspring_marker_num
        return offspring_marker_pool_num_matrix

    def get_offspring_pool_num_matrix(self):
        ''' the element o_i_i is the offspring_muns in the doffspring_pool in patch_i '''
        offspring_pool_num_matrix = np.matrix(np.zeros((self.patch_num, self.patch_num)))
        for patch_id, patch_object in self.set.items():
            patch_offspring_num = 0
            for h_id, h_object in patch_object.set.items():
                patch_offspring_num += len(h_object.offspring_pool)
            offspring_pool_num_matrix[patch_object.index, patch_object.index] = patch_offspring_num
        return offspring_pool_num_matrix
            
    def get_dormance_pool_num_matrix(self):
        ''' the element d_i_i is the dormancy_num in the dormancy_pool in patch_i '''
        dormancy_pool_num_matrix = np.matrix(np.zeros((self.patch_num, self.patch_num)))
        for patch_id, patch_object in self.set.items():
            patch_dormancy_pool_num = 0
            for h_id, h_object in patch_object.set.items():
                patch_dormancy_pool_num += len(h_object.dormancy_pool)
            dormancy_pool_num_matrix[patch_object.index, patch_object.index] = patch_dormancy_pool_num
        return dormancy_pool_num_matrix
    
    def get_patch_empty_sites_num_matrix(self):
        patch_empty_sites_num_matrix = np.matrix(np.zeros((self.patch_num, self.patch_num)))
        for patch_id, patch_object in self.set.items():
            patch_empty_sites_num = patch_object.patch_empty_sites_num()
            patch_empty_sites_num_matrix[patch_object.index, patch_object.index] = patch_empty_sites_num
        return patch_empty_sites_num_matrix
    
    def get_emigrants_matrix(self, total_disp_among_rate):
        ''' the element em_i_j is the expectation of emigrants_num disperse from patch_i to patch_j '''
        return (self.get_offspring_pool_num_matrix() + self.get_dormance_pool_num_matrix()) * self.get_disp_amomg_rate_matrix(total_disp_among_rate)

    def get_immigrants_matrix(self, total_disp_among_rate):
        emigrants_matrix = self.get_emigrants_matrix(total_disp_among_rate)
        patch_empty_sites_num_matrix = self.get_patch_empty_sites_num_matrix()
        return (emigrants_matrix/emigrants_matrix.sum(axis=0))*patch_empty_sites_num_matrix
    
    def get_dispersal_among_num_matrix(self, total_disp_among_rate):
        immigrants_matrix = self.get_immigrants_matrix(total_disp_among_rate)
        emigrants_matrix = self.get_emigrants_matrix(total_disp_among_rate)
        return self.matrix_around(np.minimum(emigrants_matrix, immigrants_matrix))

    def dispersal_among_patches_from_offspring_pool_and_dormancy_pool(self, total_disp_among_rate):
        ''' dispersal from patch_i to patch_j '''
        if self.patch_num < 2:
            log_info = '[Dispersal among patches] in %s: patch_num < 2, there are 0 individuals disperse among patches \n'
            #print(log_info)
            return log_info
        dispersal_among_num_matrix = self.get_dispersal_among_num_matrix(total_disp_among_rate)
        counter = 0
        for j in range(self.patch_num):
            patch_j_id = self.patch_id_ls[j]
            patch_j_object = self.set[patch_j_id]
            patch_j_empty_site_ls = patch_j_object.get_patch_empty_sites_ls()
            migrants_to_j_indi_object_ls = []
            
            for i in range(self.patch_num):
                patch_i_id = self.patch_id_ls[i]
                patch_i_object = self.set[patch_i_id]
                if i==j:
                    continue
                else:
                    migrants_i_j_num = int(dispersal_among_num_matrix[i, j])
                    patch_i_offspring_and_dormancy_pool = patch_i_object.get_patch_offspring_and_dormancy_pool()
                    migrants_to_j_indi_object_ls += random.sample(patch_i_offspring_and_dormancy_pool, migrants_i_j_num)
                    
            random.shuffle(patch_j_empty_site_ls)
            random.shuffle(migrants_to_j_indi_object_ls)
            for (h_id, len_id, wid_id), migrants_object in list(zip(patch_j_empty_site_ls, migrants_to_j_indi_object_ls)):
                self.set[patch_j_id].set[h_id].add_individual(indi_object = migrants_object, len_id=len_id, wid_id=wid_id)
                counter += 1
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals disperse among patches; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
    def dispersal_aomng_patches_from_offspring_pool_to_immigrant_pool(self, total_disp_among_rate):
        ''' exchange between offspring pools among patches into immigrant pool'''
        if self.patch_num < 2:
            log_info = '[Dispersal among patches] in %s: patch_num < 2, there are 0 individuals disperse into habs_immigrant_pool among patches \n'
            #print(log_info)
            return log_info
        offspring_dispersal_matrix = self.matrix_around(self.get_offspring_pool_num_matrix()*self.get_disp_amomg_rate_matrix(total_disp_among_rate))
        counter = 0
        for j in range(self.patch_num):
            patch_j_id = self.patch_id_ls[j]
            patch_j_object = self.set[patch_j_id]
            migrants_to_j_indi_object_ls = []
            
            for i in range(self.patch_num):
                patch_i_id = self.patch_id_ls[i]
                patch_i_object = self.set[patch_i_id]
                if i==j:
                    continue
                else:
                    migrants_i_j_num = int(offspring_dispersal_matrix[i, j])
                    patch_i_offspring_pool = patch_i_object.get_patch_offspring_pool()
                    migrants_to_j_indi_object_ls += random.sample(patch_i_offspring_pool, migrants_i_j_num)
                    
            random.shuffle(migrants_to_j_indi_object_ls)
            patch_j_object.migrants_to_patch_into_habs_immigrant_pool(migrants_to_j_indi_object_ls)
            counter += len(migrants_to_j_indi_object_ls)
        log_info = '[Dispersal among patches] in %s: there are %d individuals disperse into habs_immigrant_pool among patches; there are %d individuals in the immigrant pools in the metacommunity \n'%(self.metacommunity_name, counter, self.meta_immigrant_pool_individual_num())    
        #print(log_info)
        return log_info
    
    def dispersal_aomng_patches_from_offspring_marker_pool_to_immigrant_marker_pool(self, total_disp_among_rate):
        ''' exchange between offspring marker pools among patches into immigrant marker pool '''
        if self.patch_num < 2:
            log_info = '[Dispersal among patches] in %s: patch_num < 2, there are 0 individuals disperse into habs_immigrant_marker_pool among patches \n'
            #print(log_info)
            return log_info
        
        offspring_marker_dispersal_matrix = self.matrix_around(self.get_offspring_marker_pool_num_matrix()*self.get_disp_amomg_rate_matrix(total_disp_among_rate))
        counter = 0
        for j in range(self.patch_num):
            patch_j_id = self.patch_id_ls[j]
            patch_j_object = self.set[patch_j_id]
            migrants_to_j_offspring_marker_ls = []
            
            for i in range(self.patch_num):
                patch_i_id = self.patch_id_ls[i]
                patch_i_object = self.set[patch_i_id]
                if i==j:
                    continue
                else:
                    migrants_i_j_num = int(offspring_marker_dispersal_matrix[i, j])
                    patch_i_offspring_marker_pool = patch_i_object.get_patch_offspring_marker_pool()
                    migrants_to_j_offspring_marker_ls += random.sample(patch_i_offspring_marker_pool, migrants_i_j_num)
                    
            random.shuffle(migrants_to_j_offspring_marker_ls)
            patch_j_object.migrants_marker_to_patch_into_habs_immigrant_marker_pool(migrants_to_j_offspring_marker_ls)
            counter += len(migrants_to_j_offspring_marker_ls)
        log_info = '[Dispersal among patches] in %s: there are %d individuals disperse into habs_immigrant_marker_pool among patches; there are %d individuals in the immigrant_marker_pools in the metacommunity \n'%(self.metacommunity_name, counter, self.meta_immigrant_marker_pool_marker_num())     
        #print(log_info)
        return log_info

    def dispersal_among_patches_from_offsrping_pool_and_dormancy_pool_to_immigrant_pool(self, total_disp_among_rate):
        ''' exchange between offspring pools and dormancy among patches into immigrant pool'''
        if self.patch_num < 2:
            log_info = '[Dispersal among patches] in %s: patch_num < 2, there are 0 individuals disperse into habs_immigrant_pool among patches \n'
            #print(log_info)
            return log_info
        offspring_dormancy_dispersal_matrix = self.matrix_around(self.get_emigrants_matrix(total_disp_among_rate))
        counter = 0
        for j in range(self.patch_num):
            patch_j_id = self.patch_id_ls[j]
            patch_j_object = self.set[patch_j_id]
            migrants_to_j_indi_object_ls = []
            
            for i in range(self.patch_num):
                patch_i_id = self.patch_id_ls[i]
                patch_i_object = self.set[patch_i_id]
                if i==j:
                    continue
                else:
                    migrants_i_j_num = int(offspring_dormancy_dispersal_matrix[i, j])
                    patch_i_offspring_and_dormancy_pool = patch_i_object.get_patch_offspring_and_dormancy_pool()
                    migrants_to_j_indi_object_ls += random.sample(patch_i_offspring_and_dormancy_pool, migrants_i_j_num)
                    
            random.shuffle(migrants_to_j_indi_object_ls)
            patch_j_object.migrants_to_patch_into_habs_immigrant_pool(migrants_to_j_indi_object_ls)
            counter += len(migrants_to_j_indi_object_ls)
        log_info = '%s: there are %d individuals disperse into habs_immigrant_pool among patches; there are %d individuals in the immigrant pools in the metacommunity \n'%(self.metacommunity_name, counter, self.meta_immigrant_pool_individual_num())
        #print(log_info)
        return log_info    
            
#******************************************* dispersal within patch process ************************************************
    def meta_dispersal_within_patch_from_offspring_marker_to_immigrant_marker_pool(self, disp_within_rate):
        ''' random dispersal within patch, offspring marker is the denotion of an dispering offspring to be born '''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_dispersal_within_from_offspring_marker_pool_to_immigrant_marker_pool(disp_within_rate)
        log_info = '[Dispersal within process] in %s: there are %d offspring marker disperse into habs_immigrant_marker_pool within patches; there are %d offspring marker in the immigrant marker pool in the metacommunity \n'%(self.metacommunity_name, counter, self.meta_immigrant_marker_pool_marker_num())
        #print(log_info)
        return log_info

    def meta_dispersal_within_patch_from_offspring_to_immigrant_pool(self, disp_within_rate):
        ''' random dispersal within patch '''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_dispersal_within_from_offspring_pool_to_immigrant_pool(disp_within_rate)
        log_info = '[Dispersal within process] %s: there are %d individuals disperse into habs_immigrant_pool within patches; there are %d individuals in the immigrant pools in the metacommunity \n'%(self.metacommunity_name, counter, self.meta_immigrant_pool_individual_num())
        #print(log_info)
        return log_info
    
    def meta_dispersal_within_patch_from_offspring_and_dormancy_to_immigrant_pool(self, disp_within_rate):
        ''' random dispersal within patch '''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_dispersal_within_from_offspring_pool_and_dormancy_pool_to_immigrant_pool(disp_within_rate)
        log_info = '[Dispersal within process] %s: there are %d individuals disperse into habs_immigrant_pool within patches; there are %d individuals in the immigrant pools in the metacommunity \n'%(self.metacommunity_name, counter, self.meta_immigrant_pool_individual_num())
        #print(log_info)
        return log_info        

    def meta_dispersal_within_patch_from_offspring_and_dormancy_pool(self, disp_within_rate):
        ''' random dispersal within patch to empty sites directly'''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_dipersal_within_from_offspring_and_dormancy_pool(disp_within_rate)
            
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '%s: there are %d individuals disperse within patches; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n '%(self.metacommunity_name, counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
#************************************************ local germination *******************************************************************
    def meta_local_germinate_from_offspring_and_dormancy_pool(self):
        ''' germination individual randomly chosen from local habitst offspring pool + dormancy_pool '''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_local_germinate_from_offspring_and_dormancy_pool()
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '%s: there are %d individuals germinating from local offspring_pool and dormancy_pool in the local habitat; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n'%(self.metacommunity_name, counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
    def meta_local_germinate_from_offspring_and_immigrant_pool(self):
        ''' germination individual randomly chosen from local habitst offspring pool + immigrant_pool '''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_local_germinate_from_offspring_and_immigrant_pool()
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Germination process] in %s: there are %d individuals germinating from local offspring_pool and immigrant_pool in the local habitat; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n'%(self.metacommunity_name, counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
    def meta_local_germinate_from_offspring_immigrant_and_dormancy_pool(self):
        ''' germination individual randomly chosen from local habitst offspring pool + immigrant_pool + dormancy_pool '''
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_local_germinate_from_offspring_immigrant_and_dormancy_pool()
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Germination process] %s: there are %d individuals germinating from local offspring_pool and dormancy_pool and immigrant_pool in the local habitat; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n'%(self.metacommunity_name, counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
    def meta_local_germinate_and_birth_from_offspring_marker_and_immigrant_marker_pool(self, mutation_rate, pheno_var_ls):
        ''' germination individual marker randomly chosen from local habitst offspring pool + immigrant_pool and then birth process according to the marker information '''
        counter = 0
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                hab_empty_pos_ls = h_object.empty_site_pos_ls
                hab_offspring_marker_and_immigrant_marker_pool = h_object.offspring_marker_pool + h_object.immigrant_marker_pool
                
                random.shuffle(hab_empty_pos_ls)
                random.shuffle(hab_offspring_marker_and_immigrant_marker_pool)
                
                for (row_id, col_id), indi_marker in list(zip(hab_empty_pos_ls, hab_offspring_marker_and_immigrant_marker_pool)):
                    birth_patch_id, birth_h_id, reproduce_mode = indi_marker[0], indi_marker[1], indi_marker[2]
                    birth_h_object = self.set[birth_patch_id].set[birth_h_id]  # birth place h_object
                    
                    if reproduce_mode == 'asexual':
                        indi_object = birth_h_object.hab_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)[0]
                    elif reproduce_mode == 'sexual':
                        indi_object = birth_h_object.hab_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)[0]
                    elif reproduce_mode == 'mix_asexual':
                        indi_object = birth_h_object.hab_mix_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)[0]
                    elif reproduce_mode == 'mix_sexual':
                        indi_object = birth_h_object.hab_mix_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)[0]
                    
                    h_object.add_individual(indi_object=indi_object, len_id=row_id, wid_id=col_id)
                    counter += 1
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = '[Germination & Birth process] %s: there are %d individuals germinating_and_birth from local offspring_marker_pool and immigrant_marker_pool in the local habitat; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity \n'%(self.metacommunity_name, counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
        
#************************************************* dormancy process in the metacommunity *********************************************************************
    def meta_dormancy_process_from_offspring_pool_to_dormancy_pool(self):
        survival_counter = 0
        eliminate_counter = 0
        new_dormancy_counter = 0
        all_dormancy_num = 0
        for patch_id, patch_object in self.set.items():
            survival_num, eliminate_num, new_dormancy_num, dormancy_num = patch_object.patch_dormancy_process_from_offspring_pool_to_dormancy_pool()
            survival_counter += survival_num
            eliminate_counter += eliminate_num
            new_dormancy_counter += new_dormancy_num
            all_dormancy_num += dormancy_num
            
        log_info_1 = '%s: there are %d dormancy survived in the dormancy pool; there are %d dormancy eliminated from the dormancy pool; there are %d new dormancy into the dormancy pool; there are totally %d dormancy in the dormancy pool across the metacommunity \n'%(self.metacommunity_name, survival_counter, eliminate_counter, new_dormancy_counter, all_dormancy_num)
        log_info_2 = '%s: there are %d individuals in the offspring_pool; there are %d individuals in the immigrant_pool; there are %d individuals in the dormancy_pool \n'%(self.metacommunity_name, self.meta_offspring_pool_individual_num(), self.meta_immigrant_pool_individual_num(), self.meta_dormancy_pool_individual_num())
        #print(log_info_1)
        #print(log_info_2)
        return log_info_1 + log_info_2
    
    def meta_dormancy_process_from_offspring_pool_and_immigrant_pool(self):
        survival_counter = 0
        eliminate_counter = 0
        new_dormancy_counter = 0
        all_dormancy_num = 0
        for patch_id, patch_object in self.set.items():
            survival_num, eliminate_num, new_dormancy_num, dormancy_num = patch_object.patch_dormancy_process_from_offspring_pool_and_immigrant_pool()
            survival_counter += survival_num
            eliminate_counter += eliminate_num
            new_dormancy_counter += new_dormancy_num
            all_dormancy_num += dormancy_num
            
        log_info_1 = '[Dormancy process] in %s: there are %d dormancy survived in the dormancy pool; there are %d dormancy eliminated from the dormancy pool; there are %d new dormancy into the dormancy pool; there are totally %d dormancy in the dormancy pool across the metacommunity \n'%(self.metacommunity_name, survival_counter, eliminate_counter, new_dormancy_counter, all_dormancy_num)
        log_info_2 = '%s: there are %d individuals in the offspring_pool; there are %d individuals in the immigrant_pool; there are %d individuals in the dormancy_pool \n'%(self.metacommunity_name, self.meta_offspring_pool_individual_num(), self.meta_immigrant_pool_individual_num(), self.meta_dormancy_pool_individual_num())
        #print(log_info_1)
        #print(log_info_2)
        return log_info_1 + log_info_2
    
# ******************** clearing up offspring_and_immigrant_pool when dormancy pool do not run *******************************************************************#
    def meta_clear_up_offspring_and_immigrant_pool(self):
        for patch_id, patch_object in self.set.items():
            patch_object.patch_clear_up_offspring_and_immigrant_pool()
    
    def meta_clear_up_offspring_marker_and_immigrant_marker_pool(self):
        for patch_id, patch_object in self.set.items():
            patch_object.patch_clear_up_offspring_marker_and_immigrant_marker_pool()
    
#************************************************** disturbance process in the metacommunity **************************************************************************
    def meta_disturbance_process_in_patches(self, patch_dist_rate):
        patch_dist_occur_ls = []
        for patch_id, patch_object in self.set.items():
            if np.random.uniform(0,1,1)[0] < patch_dist_rate:
                patch_object.patch_disturbance_process()
                patch_dist_occur_ls.append(patch_id)
            else:
                continue
        log_info = f'[Disturbance process] occurred in {patch_dist_occur_ls} \n'
        #print(log_info)
        return log_info
 
    def meta_disturbance_process_in_habitat(self, hab_dist_rate):
        habitat_dist_occur_ls = []
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                if np.random.uniform(0,1,1)[0] < hab_dist_rate:
                    h_object.habitat_disturbance_process()
                    habitat_dist_occur_ls.append(patch_id+'_'+h_id)
                else:
                    continue
        log_info = f'[Disturbance process] occurred in {habitat_dist_occur_ls} \n'
        #print(log_info)
        return log_info
                
#************************************************ data generating and saving module *********************************************************************
    def get_meta_microsites_optimum_sp_id_val(self, d, w, species_2_phenotype_ls):
        ''' '''
        meta_optimum_sp_id_val_dis = np.array([])
        for patch_id, patch_object in self.set.items():
            patch_optimum_sp_id_val_dis = patch_object.get_patch_microsites_optimum_sp_id_value_array(d, w, species_2_phenotype_ls)
            meta_optimum_sp_id_val_dis = np.append(meta_optimum_sp_id_val_dis, patch_optimum_sp_id_val_dis.reshape(-1))
        return meta_optimum_sp_id_val_dis.reshape(1,-1)
    
    def get_meta_microsite_environment_values(self, environment_name, digits=3):
        ''''''
        meta_environment_dis = np.array([])
        for patch_id, patch_object in self.set.items():
            patch_environment_dis = patch_object.get_patch_microsites_environment_values(environment_name)
            meta_environment_dis = np.append(meta_environment_dis, patch_environment_dis.reshape(-1))
            meta_environment_dis = np.around(meta_environment_dis, digits) #保留几位小数
        return meta_environment_dis.reshape(1,-1)

    def get_meta_microsites_individuals_sp_id_values(self):
        ''' '''
        meta_sp_dis = np.array([], dtype=int)
        for patch_id, patch_object in self.set.items():
            patch_sp_dis = patch_object.get_patch_microsites_individals_sp_id_values()
            meta_sp_dis = np.append(meta_sp_dis, patch_sp_dis.reshape(-1))
        return meta_sp_dis.reshape(1,-1)
    
    def get_meta_microsites_individuals_phenotype_values(self, trait_name, digits=3):
        ''''''
        meta_phenotype_dis = np.array([])
        for patch_id, patch_object in self.set.items():
            patch_phenotype_dis = patch_object.get_patch_microsites_individals_phenotype_values(trait_name)
            meta_phenotype_dis = np.append(meta_phenotype_dis, patch_phenotype_dis.reshape(-1))
        meta_phenotype_dis = np.around(meta_phenotype_dis, digits) # #保留几位小数 to save the storage
        return meta_phenotype_dis.reshape(1,-1)    

    def columns_patch_habitat_microsites_id(self):
        ''' return 3 lists of patch_id, h_id, microsite_id as the header of meta_sp_dis table '''
        columns_patch_id = np.array([])
        columns_habitat_id = np.array([])
        columns_mocrosite_id = np.array([])
        
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                h_len, h_wid = h_object.length, h_object.width
                columns_patch_id = np.append(columns_patch_id, np.array([patch_id for _ in range(h_len*h_wid)]))
                columns_habitat_id = np.append(columns_habitat_id, np.array([h_id for _ in range(h_len*h_wid)]))
                columns_mocrosite_id = np.append(columns_mocrosite_id, np.array(['r%d, c%d'%(i, j) for i in range(h_len) for j in range(h_wid)]))
        return columns_patch_id, columns_habitat_id, columns_mocrosite_id

    def meta_distribution_data_all_time_to_csv_gz(self, dis_data_all_time, file_name, index, columns, mode='w'):
        ''' '''
        df_species_distribution = pd.DataFrame(dis_data_all_time, index=index, columns=columns)
        if mode=='w': 
            df_species_distribution.to_csv(file_name, mode=mode, compression='gzip')
        elif mode=='a': 
            df_species_distribution.to_csv(file_name, mode=mode, compression='gzip', header=False)
        return df_species_distribution

################################################################ class individual #################################################################################################    
class individual():
    def __init__(self, species_id, traits_num, pheno_names_ls, gender='female', genotype_set=None, phenotype_set=None):
        self.species_id = species_id
        self.gender = gender
        self.traits_num = traits_num
        self.pheno_names_ls = pheno_names_ls
        self.genotype_set = genotype_set
        self.phenotype_set = phenotype_set
        self.age = 0
        
    def random_init_indi(self, mean_pheno_val_ls, pheno_var_ls, geno_len_ls):
        '''
        pheno_names is a tuple of the pheno_names (string) i.e., ('phenotye_1', 'phenotype_2',...,'phenotye_x') and the len(pheno_names) is equal to traits_num.
        mean_pheno_val (tuple) is the mean values (float) of the phenotypes of a species population which fit a gaussian distribution, i.e., (val1, val2,...,valx).
        pheno_var (tuple) is the variation (float) of the phenotypes of a species population which fit a gaussian distribution.
        geno_len (tuple) is the len of each genotype in the genotype_set, the genotype in which controls each phenotype of each trait.
        '''
        genotype_set = {}
        phenotype_set = {}
        
        for i in range(self.traits_num):
            name = self.pheno_names_ls[i]
            mean = mean_pheno_val_ls[i]
            var = pheno_var_ls[i]
            geno_len = geno_len_ls[i]
            
            #random_index = random.sample(range(0,geno_len*2),int(mean*geno_len*2))
            #genotype = np.array([1 if i in random_index else 0 for i in range(geno_len*2)])
            #bi_genotype = [genotype[0:geno_len], genotype[geno_len:geno_len*2]]
            
            random_index_1 = random.sample(range(0,geno_len),int(mean*geno_len))
            random_index_2 = random.sample(range(0,geno_len),int(mean*geno_len))
            genotype_1 = np.array([1 if i in random_index_1 else 0 for i in range(geno_len)])
            genotype_2 = np.array([1 if i in random_index_2 else 0 for i in range(geno_len)])
            
            bi_genotype = [genotype_1, genotype_2]
            phenotype = mean + random.gauss(0, var)
            
            genotype_set[name] = bi_genotype
            phenotype_set[name] = phenotype
        self.genotype_set = genotype_set
        self.phenotype_set = phenotype_set
        return 0
    
    def __str__(self):
        species_id_str = 'speceis_id=%s'%self.species_id
        gender_str = 'gender=%s'%self.gender
        traits_num_str = 'traits_num=%d'%self.traits_num
        genotype_set_str = 'genetype_set=%s'%str(self.genotype_set)
        phenotype_set_str = 'phenotype_set=%s'%str(self.phenotype_set)
        
        strings = species_id_str+'\n'+ gender_str+'\n'+traits_num_str+'\n'+genotype_set_str+'\n'+phenotype_set_str
        return strings
    
    def get_indi_phenotype_ls(self):
        indi_phenotype_ls = []
        for pheno_name in self.pheno_names_ls:
            phenotype = self.phenotype_set[pheno_name]
            indi_phenotype_ls.append(phenotype)
        return indi_phenotype_ls
    
    def mutation(self, rate, pheno_var_ls):
        for i in range(self.traits_num):
            mutation_counter = 0
            pheno_name = self.pheno_names_ls[i]
            var = pheno_var_ls[i]
            genotype1 = self.genotype_set[pheno_name][0]
            genotype2 = self.genotype_set[pheno_name][1]
            for index in range(len(genotype1)):
                if rate > np.random.uniform(0,1,1)[0]:
                    mutation_counter += 1
                    if genotype1[index] == 0: self.genotype_set[pheno_name][0][index]=1
                    elif genotype1[index] == 1: self.genotype_set[pheno_name][0][index]=0
                    
            for index in range(len(genotype2)):
                if rate > np.random.uniform(0,1,1)[0]:
                    mutation_counter += 1
                    if genotype2[index] == 0: self.genotype_set[pheno_name][1][index]=1
                    elif genotype2[index] == 1: self.genotype_set[pheno_name][1][index]=0
            if mutation_counter >=1: 
                phenotype = np.mean(self.genotype_set[pheno_name]) + random.gauss(0, var)
                self.phenotype_set[pheno_name] = phenotype
        return 0
    

    










