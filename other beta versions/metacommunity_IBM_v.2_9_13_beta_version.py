# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:09:31 2022

@author: JH_Lin
"""
import numpy as np
import random
import itertools
import networkx as nx
from networkx.algorithms import approximation
import matplotlib.pyplot as plt
import math
import copy
import re
import pandas as pd
import seaborn as sns
import time
from queue import Queue
import logging
###################################################################################################################################################
class habitat():
    def __init__(self, hab_name, num_env_types, env_types_name, mean_env_ls, var_env_ls, length, width):
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
        self.dormancy_pool = []
        self.species_category = {}
        self.occupied_site_pos_ls = []
        self.empty_site_pos_ls = [(i, j) for i in range(length) for j in range(width)]
        
        self.reproduction_mode_threhold = 0.897
        self.asexual_parent_pos_ls = []                           # If an individual can fit its environment condition well, it goes through asexual reproduction.
        self.species_category_for_sexual_parents_pos = {}         # If an individual can not fit its environmet condition, it goes through sexual reproduction.

        ####### to be improve #######
        self.dormancy_pool_max_size = 25
        #############################
        
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
    
    def update_one_environment_values(self, env_name, mean_env_value, var_env_value):
        ''' update a type of environment value to a new environment value, and the update each microsite value accroding to the mean and variation '''
        env_index = self.env_types_name.index(env_name)
        self.mean_env_ls[env_index] = mean_env_value
        self.var_env_ls[env_index] = var_env_value
        
        microsite_new_e_values = np.random.normal(loc=0, scale=var_env_value, size=(self.length, self.width)) + mean_env_value
        self.set[env_name] = microsite_new_e_values
        return 0
    
    def update_all_environment_values(self, env_types_name, mean_env_ls, var_env_ls):
        ''' update all types of environment value to a new environment value, and the update each microsite value accroding to the mean and variation '''
        for index in range(0, len(mean_env_ls)):
            mean_e_index = self.mean_env_ls[index]
            var_e_index = self.var_env_ls[index]
            name_e_index = self.env_types_name[index]
            microsite_e_values = np.random.normal(loc=0, scale=var_e_index, size=(self.length, self.width)) + mean_e_index
            self.set[name_e_index] = microsite_e_values
        return 0
    
    def get_microsite_env_val_ls(self, len_id, wid_id):
        ''' return a list of environment value of all the environment type in the order of env_types_name '''
        env_val_ls = []
        for env_name in self.env_types_name:
            env_val = self.set[env_name][len_id][wid_id]
            env_val_ls.append(env_val)
        return env_val_ls
    
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
                
######################################################################################################
    
    def survival_rate(self, d, phenotype_ls, env_val_ls, w = 0.5):
        # d is the baseline death rate responding to the disturbance strength.
        # phenotype_ls is a list of phenotype of each trait.
        # env_val_ls is a list of environment value responding to the environment type.
        # w is the width of the fitness function.
        survival_rate = (1-d)
        power = 0
        n = 0
        for index in range(len(phenotype_ls)):
            ei = phenotype_ls[index]               # individual phenotype of a trait 
            em = env_val_ls[index]                 #microsite environment value of a environment type
            power += math.pow(((ei-em)/w),2)
            n += 1
        survival_rate = (1-d) * math.exp((-1/n)*power)
        return survival_rate
    
######****************************************************************************#####################
    '''
    def survival_rate(self, d, phenotype_ls, env_val_ls, w = 0.5):
        
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
    '''
####################################################################################################
    def hab_dead_selection(self, base_dead_rate, fitness_wid):
        self.asexual_parent_pos_ls = []                           # If an individual can fit its environment condition well, it goes through asexual reproduction.
        self.species_category_for_sexual_parents_pos = {}         # If an individual can not fit its environmet condition, it goes through sexual reproduction.
        counter = 0
        for row in range(self.length):
            for col in range(self.width):
                env_val_ls = self.get_microsite_env_val_ls(row, col)
                
                if self.set['microsite_individuals'][row][col] != None:
                    individual_object = self.set['microsite_individuals'][row][col]
                    phenotype_ls = individual_object.get_indi_phenotype_ls()
                    survival_rate = self.survival_rate(d=base_dead_rate, phenotype_ls=phenotype_ls, env_val_ls=env_val_ls, w = fitness_wid)
                    
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

    def hab_asex_reproduce_mutate(self, asexual_birth_rate, mutation_rate, pheno_var_ls):
        self.offspring_pool = []
        nums = int(asexual_birth_rate)
        rate = asexual_birth_rate - nums
        for row in range(self.length):
            for col in range(self.width):
                if self.set['microsite_individuals'][row][col] == None:
                    continue
                else:
                    individual_object = self.set['microsite_individuals'][row][col]
                    for num in range(nums):
                        new_indivi_object = copy.deepcopy(individual_object)
                        for i in range(new_indivi_object.traits_num):
                            pheno_name = new_indivi_object.pheno_names_ls[i]
                            var = pheno_var_ls[i] #### to be improved ####
                            genotype = new_indivi_object.genotype_set[pheno_name]
                            phenotype = np.mean(genotype) + random.gauss(0, var)
                            new_indivi_object.phenotype_set[pheno_name] = phenotype
                        #print(individual_object, '\n'), print(new_indivi_object, '\n'), print('\n\n\n\n\n\n')
                        new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
                        self.offspring_pool.append(new_indivi_object)
                        
                    if rate > np.random.uniform(0,1,1)[0]:
                        new_indivi_object = copy.deepcopy(individual_object)
                        for i in range(new_indivi_object.traits_num):
                            pheno_name = new_indivi_object.pheno_names_ls[i]
                            var = pheno_var_ls[i] #### to be improved ####
                            genotype = new_indivi_object.genotype_set[pheno_name]
                            phenotype = np.mean(genotype) + random.gauss(0, var)
                            new_indivi_object.phenotype_set[pheno_name] = phenotype
                        #print(individual_object, '\n'), print(new_indivi_object, '\n'), print('\n\n\n\n\n\n')
                        new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
                        self.offspring_pool.append(new_indivi_object)
        return 0
    
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
    
    def hab_sex_reproduce_mutate(self, sexual_birth_rate, mutation_rate, pheno_var_ls):
        nums = int(sexual_birth_rate)
        rate = sexual_birth_rate - nums
        self.offspring_pool = []
        for sp_id, sp_id_val in self.species_category.items():
            try:
                sp_id_female_ls = sp_id_val['female']
            except:
                continue
            try:
                sp_id_male_ls = sp_id_val['male']
            except:
                continue
            
            random.shuffle(sp_id_female_ls) # list of individuals location in habitat, i.e., (len_id, wid_id)
            random.shuffle(sp_id_male_ls) # random sample of pairwise parents in sexual reproduction
            
            for female_pos, male_pos in list(zip(sp_id_female_ls, sp_id_male_ls)):
                female_indi_obj = self.set['microsite_individuals'][female_pos[0]][female_pos[1]]
                male_indi_obj = self.set['microsite_individuals'][male_pos[0]][male_pos[1]]
                for num in range(nums):
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
                    #print('female_indi_obj', female_indi_obj, '\n'), print('male_indi_obj', male_indi_obj, '\n'), print('new_indivi_object', new_indivi_object, '\n\n\n\n\n\n')    
                    new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
                    self.offspring_pool.append(new_indivi_object)
                
                if rate > np.random.uniform(0,1,1)[0]:
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
                    #print('female_indi_obj', female_indi_obj, '\n'), print('male_indi_obj', male_indi_obj, '\n'), print('new_indivi_object', new_indivi_object, '\n\n\n\n\n\n')    
                    new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
                    self.offspring_pool.append(new_indivi_object)
                else:
                    continue
        return 0                 
    
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

    def hab_germinate_from_offsprings_pool(self):
        ''' the offsprings in the habitat offsprings pool germinates in the empty microsite in the habitat'''
        counter = 0
        empty_sites_pos_ls = self.empty_site_pos_ls
        random.shuffle(empty_sites_pos_ls)
        
        hab_offsprings_pool = self.offspring_pool
        random.shuffle(hab_offsprings_pool)
        
        for pos, indi_object in list(zip(empty_sites_pos_ls, hab_offsprings_pool)):
            len_id = pos[0]
            wid_id = pos[1]
            self.add_individual(indi_object, len_id, wid_id)
            counter += 1
        return counter
    
    def hab_asexual_reprodece_germinate(self, asexual_birth_rate, mutation_rate, pheno_var_ls):
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
        
    def hab_dormancy_process(self):
        ''' offsprings pool into dormancy pool'''
        if len(self.offspring_pool) + len(self.dormancy_pool) <= self.dormancy_pool_max_size:
            self.dormancy_pool = self.dormancy_pool + self.offspring_pool
        else: 
            hab_dormancy_pool = random.sample(self.dormancy_pool,(self.dormancy_pool_max_size-len(self.offspring_pool)))
            self.dormancy_pool = hab_dormancy_pool + self.offspring_pool
            
        self.offspring_pool = []
        return 0
        
class patch():
    def __init__(self, patch_name, patch_index, location, asexual_birth_rate, sexual_birth_rate):
        self.name = patch_name
        self.index = patch_index
        self.set = {}            # self.data_set={} # to be improved
        self.hab_num = 0
        self.location = location
        self.asexual_birth_rate = asexual_birth_rate  # to be improved
        self.sexual_birth_rate = sexual_birth_rate    # to be improved
        
    def get_data(self):
        output = {}
        for key, value in self.set.items():
            output[key]=value.set
        return output

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
                    
    def __str__(self):
        return str(self.get_data())
    
    def get_patch_size(self):
        patch_size = 0
        for key, value in self.set.items():
            patch_size += value.size
        return patch_size
    
    def get_patch_individual_num(self):
        num = 0
        for key, value in self.set.items():
            num += value.indi_num
        return num
    
    def get_patch_sexual_pairwise_parents_num(self):
        num = 0
        for h_id, h_object in self.set.items():
            num += h_object.hab_sexual_pairwise_parents_num()
        return num
    
    def get_patch_mixed_sexual_pairwise_parents_num(self):
        num = 0
        for h_id, h_object in self.set.items():
            num += h_object.hab_mixed_sexual_pairwse_parents_num()
        return num
    
    def get_patch_mixed_asexual_parent_num(self):
        num = 0
        for h_id, h_object in self.set.items():
            num += h_object.hab_mixed_asexual_parent_num()
        return num

    def get_disp_within_offsprings_pool(self, target_hab_object):
        ''' return all the offspring (individual objects) in the patch, 
        in the exception of the target hab as a list for dispersal with patches'''
        disp_within_patch_offsprings_pool = []
        for h_id, h_object in self.set.items():
            if h_id != target_hab_object.name:
                disp_within_patch_offsprings_pool += h_object.offspring_pool
            else:
                continue
        return disp_within_patch_offsprings_pool
    
    def get_disp_within_asex_parent_pos_ls(self, target_hab_object):
        disp_within_patch_parent_pos_ls = []
        for h_id, h_object in self.set.items():
            if h_id != target_hab_object.name:
                occupied_site_pos_ls = h_object.occupied_site_pos_ls
                for site_pos in occupied_site_pos_ls:
                    site_pos = (h_id, ) + site_pos
                    disp_within_patch_parent_pos_ls.append(site_pos)
            else:
                continue
        return disp_within_patch_parent_pos_ls
    
    def get_disp_within_sex_pairwise_parents_pos_ls(self, target_hab_object):
        disp_within_patch_parent_pos_ls = []
        for h_id, h_object in self.set.items():
            if h_id != target_hab_object.name:
                pairwise_parents_pos_ls = h_object.hab_sexual_pairwise_parents_ls()
                for pairwise_parents_pos in pairwise_parents_pos_ls:
                    female_pos = pairwise_parents_pos[0]
                    male_pos = pairwise_parents_pos[1]
                    female_pos = (h_id, ) + female_pos
                    male_pos = (h_id, ) + male_pos
                    disp_within_patch_parent_pos_ls.append((female_pos, male_pos))
            else:
                continue
        return disp_within_patch_parent_pos_ls
    
    def get_disp_within_mixed_asex_parent_pos_ls(self, target_hab_object):
        disp_within_patch_parent_pos_ls = []
        for h_id, h_object in self.set.items():
            if h_id != target_hab_object.name:
                mixed_asexual_parent_pos_ls = h_object.asexual_parent_pos_ls
                for site_pos in mixed_asexual_parent_pos_ls:
                    site_pos = (h_id, ) + site_pos
                    disp_within_patch_parent_pos_ls.append(site_pos)
            else:
                continue
        return disp_within_patch_parent_pos_ls
    
    def get_disp_within_mixed_sex_pairwise_parents_pos_ls(self, target_hab_object):
        disp_within_patch_parent_pos_ls = []
        for h_id, h_object in self.set.items():
            if h_id != target_hab_object.name:
                mixed_pairwise_parents_pos_ls = h_object.hab_mixed_sexual_pairwise_parents_ls()
                for pairwise_parents_pos in mixed_pairwise_parents_pos_ls:
                    female_pos = pairwise_parents_pos[0]
                    male_pos = pairwise_parents_pos[1]
                    female_pos = (h_id, ) + female_pos
                    male_pos = (h_id, ) + male_pos
                    disp_within_patch_parent_pos_ls.append((female_pos, male_pos))
            else:
                continue
        return disp_within_patch_parent_pos_ls

    def get_patch_offsprings_pool(self):
        ''' return all the offspring (individual objects) in the patch as a list for dispersal among patches'''
        patch_offsprings_pool = []
        for h_id, h_object in self.set.items():
            patch_offsprings_pool += h_object.offspring_pool
        return patch_offsprings_pool
    
    def patch_offsprings_num(self):
        ''' return the number of offsprings in the patches '''
        return len(self.get_patch_offsprings_pool())
    
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
    
    def add_habitat(self, hab_name, num_env_types, env_types_name, mean_env_ls, var_env_ls, length, width):
        h_object = habitat(hab_name, num_env_types, env_types_name, mean_env_ls, var_env_ls, length, width)
        self.set[hab_name] = h_object
        self.hab_num += 1
        
    def patch_initialize(self, traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls):
        for h_id, h_object in self.set.items():
            h_object.hab_initialize(traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls)
        return 0
    
    def patch_dead_selection(self, base_dead_rate, fitness_wid):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_dead_selection(base_dead_rate, fitness_wid)
        return counter
    
    def patch_asex_reproduce_mutate(self, mutation_rate, pheno_var_ls):
        asexual_birth_rate = self.asexual_birth_rate
        for h_id, h_object in self.set.items():
            h_object.hab_asex_reproduce_mutate(asexual_birth_rate, mutation_rate, pheno_var_ls)
        return 0
    
    def patch_sex_reproduce_mutate(self, mutation_rate, pheno_var_ls):
        sexual_birth_rate = self.sexual_birth_rate
        for h_id, h_object in self.set.items():
            h_object.hab_sex_reproduce_mutate(sexual_birth_rate, mutation_rate, pheno_var_ls)
        return 0
    
    def asex_reproduce_mutate_for_dispersal_among_patches(self, mutation_rate, pheno_var_ls, patch_offs_num):
        patch_disp_among_pool = []
        patch_indi_num = self.get_patch_individual_num()
        
        if patch_offs_num == 0 or patch_indi_num == 0:
            return patch_disp_among_pool
        else:
            for h_id, h_object in self.set.items():
                hab_offs_raw_num = patch_offs_num * (h_object.indi_num/patch_indi_num)
                hab_offs_num = int(hab_offs_raw_num) # 整数部分表示后代个体数
                patch_disp_among_pool += h_object.hab_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, hab_offs_num)
                
                hab_offs_probability = hab_offs_raw_num - hab_offs_num # 小数部分表示概率
                if hab_offs_probability >= np.random.uniform(0,1,1)[0]:
                    patch_disp_among_pool += h_object.hab_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)
            #print('patch_offs_num=', patch_offs_num, 'len of patch_disp_among_pool=', len(patch_disp_among_pool))
            return patch_disp_among_pool
    
    def sex_reproduce_mutate_for_dispersal_among_patches(self, mutation_rate, pheno_var_ls, patch_offs_num):
        patch_disp_among_pool = []
        patch_pairwise_parents_num = self.get_patch_sexual_pairwise_parents_num()
        if patch_offs_num == 0 or patch_pairwise_parents_num == 0:
            return patch_disp_among_pool
        else:
            for h_id, h_object in self.set.items():
                hab_offs_raw_num = patch_offs_num * (h_object.hab_sexual_pairwise_parents_num()/patch_pairwise_parents_num)
                hab_offs_num = int(hab_offs_raw_num)  # 整数部分表示后代个体数
                patch_disp_among_pool += h_object.hab_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, hab_offs_num)
                
                hab_offs_probability = hab_offs_raw_num - hab_offs_num # 小数部分表示概率
                if hab_offs_probability >= np.random.uniform(0,1,1)[0]:
                    patch_disp_among_pool += h_object.hab_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)
            #print('patch_offs_num=', patch_offs_num, 'len of patch_disp_among_pool=', len(patch_disp_among_pool))
            return patch_disp_among_pool
    
    def mixed_reproduce_mutate_for_dispersal_among_patches(self, mutation_rate, pheno_var_ls, patch_offs_num):
        patch_disp_among_pool = []
        patch_asexual_mixed_parents_num = self.get_patch_mixed_asexual_parent_num()
        patch_sexual_mixed_pairwise_parents_num = self.get_patch_mixed_sexual_pairwise_parents_num()
        
        if patch_offs_num == 0:
            return patch_disp_among_pool
        elif (patch_asexual_mixed_parents_num + patch_sexual_mixed_pairwise_parents_num) == 0:
            return patch_disp_among_pool
        else:
            patch_asexual_mixed_offs_num = int(np.around(patch_offs_num * (patch_asexual_mixed_parents_num * self.asexual_birth_rate)/(patch_asexual_mixed_parents_num * self.asexual_birth_rate + patch_sexual_mixed_pairwise_parents_num * self.sexual_birth_rate)))
            patch_sexual_mixed_offs_num = int(np.around(patch_offs_num * (patch_sexual_mixed_pairwise_parents_num * self.sexual_birth_rate)/(patch_asexual_mixed_parents_num * self.asexual_birth_rate + patch_sexual_mixed_pairwise_parents_num * self.sexual_birth_rate)))

            for h_id, h_object in self.set.items():
                if patch_asexual_mixed_parents_num != 0: 
                    hab_asexual_mixed_offs_raw_num = patch_asexual_mixed_offs_num * (h_object.hab_mixed_asexual_parent_num()/patch_asexual_mixed_parents_num) # 期望值
                    hab_asexual_mixed_offs_num = int(hab_asexual_mixed_offs_raw_num) # 整数部分
                    hab_asexual_mixed_offs_probability = hab_asexual_mixed_offs_raw_num - hab_asexual_mixed_offs_num # 小数部分表示概率
                    patch_disp_among_pool += h_object.hab_mix_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, hab_asexual_mixed_offs_num)

                    if hab_asexual_mixed_offs_probability >= np.random.uniform(0,1,1)[0]:
                        patch_disp_among_pool += h_object.hab_mix_asex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)
                        hab_asexual_mixed_offs_num += 1
                    #print(h_id, 'hab_asexual_mixed_offs_raw_num=', hab_asexual_mixed_offs_raw_num, 'hab_asexual_mixed_offs_num=', hab_asexual_mixed_offs_num)
                    
                if patch_sexual_mixed_pairwise_parents_num != 0:
                    hab_sexual_mixed_offs_raw_num = patch_sexual_mixed_offs_num * (h_object.hab_mixed_sexual_pairwse_parents_num()/patch_sexual_mixed_pairwise_parents_num)
                    hab_sexual_mixed_offs_num = int(hab_sexual_mixed_offs_raw_num) # 整数部分
                    hab_sexual_mixed_offs_probability = hab_sexual_mixed_offs_raw_num - hab_sexual_mixed_offs_num # 小数部分表示概率
                    patch_disp_among_pool += h_object.hab_mix_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, hab_sexual_mixed_offs_num)

                    if hab_sexual_mixed_offs_probability >= np.random.uniform(0,1,1)[0]:
                        patch_disp_among_pool += h_object.hab_mix_sex_reproduce_mutate_with_num(mutation_rate, pheno_var_ls, num=1)
                        hab_sexual_mixed_offs_num += 1
                    #print(h_id, 'hab_sexual_mixed_offs_raw_num=', hab_sexual_mixed_offs_raw_num, 'hab_sexual_mixed_offs_num=', hab_sexual_mixed_offs_num)
  
            return patch_disp_among_pool
                
    def patch_disp_within_from_offsprings_pool(self, disp_within_rate, counter):
        ''''''
        for h_id, h_object in self.set.items():
            h_offspring_pool = h_object.offspring_pool # 本地生境后代个体对象列表
            disp_within_pool = random.sample(self.get_disp_within_offsprings_pool(h_object), int(len(self.get_disp_within_offsprings_pool(h_object))*disp_within_rate/(self.hab_num-1))) # 斑块内非本地生境后代迁入个体对象列表
            
            h_empty_site_ls = h_object.empty_site_pos_ls
            if (len(h_offspring_pool)+len(disp_within_pool)) != 0:
                disp_within_empty_site_num = int(np.around(len(h_empty_site_ls) * len(disp_within_pool)/(len(h_offspring_pool)+len(disp_within_pool))))
                disp_within_sites = random.sample(h_empty_site_ls, disp_within_empty_site_num)
            else:
                disp_within_sites = []
            
            if len(disp_within_pool) > len(disp_within_sites):
                disp_within_indi_ls = random.sample(disp_within_pool, len(disp_within_sites))
            else:
                disp_within_indi_ls = disp_within_pool
                random.shuffle(disp_within_indi_ls)
            
            for empty_site_pos, disp_indi_object in list(zip(disp_within_sites, disp_within_indi_ls)):
                len_id = empty_site_pos[0]
                wid_id = empty_site_pos[1]
                self.set[h_id].add_individual(indi_object=disp_indi_object, len_id=len_id, wid_id=wid_id)
                #print(counter, self.name, h_object.name, len_id, wid_id)
                counter += 1
        return counter
    
    def asex_reproduce_mutate_for_dispersal_within_patch(self, mutation_rate, pheno_var_ls, disp_within_rate, counter):
        ''''''
        asexual_birth_rate = self.asexual_birth_rate
        for h_id, h_object in self.set.items():
            
            h_asexual_parent_num = len(h_object.occupied_site_pos_ls) # 本地无性生殖母本个数
            h_asex_offs_expectation_num = h_asexual_parent_num * asexual_birth_rate # 本地无性繁殖子代数
            
            asex_parent_pos_ls = self.get_disp_within_asex_parent_pos_ls(h_object) # 斑块内外来生境的无性繁殖母本位置列表
            offsprings_expection_num = int(np.around(len(asex_parent_pos_ls) * asexual_birth_rate * disp_within_rate / (self.hab_num-1))) # 斑块内非本地生境无性生殖后代迁入至当前生境个体数期望值
            
            h_empty_site_ls = h_object.empty_site_pos_ls # 生境的空白斑块编号列表
            if (h_asex_offs_expectation_num + offsprings_expection_num) != 0:
                disp_within_empty_site_num = int(np.around(len(h_empty_site_ls) * offsprings_expection_num/(h_asex_offs_expectation_num + offsprings_expection_num)))
            else:
                disp_within_empty_site_num = 0
            
            if offsprings_expection_num > disp_within_empty_site_num:
                disp_within_sites = random.sample(h_empty_site_ls, disp_within_empty_site_num)
                disp_within_parent_pos_ls = random.sample(asex_parent_pos_ls, disp_within_empty_site_num)
            else:
                disp_within_sites = random.sample(h_empty_site_ls, offsprings_expection_num)
                disp_within_parent_pos_ls = random.sample(asex_parent_pos_ls, offsprings_expection_num)
                
            for empty_site_pos, parent_pos in list(zip(disp_within_sites, disp_within_parent_pos_ls)):
                empty_len_id, empty_wid_id = empty_site_pos[0], empty_site_pos[1]
                parent_h_id, parent_row, parent_col = parent_pos[0], parent_pos[1], parent_pos[2]
                
                parent_indi_object = self.set[parent_h_id].set['microsite_individuals'][parent_row][parent_col]
                new_indivi_object = copy.deepcopy(parent_indi_object)
                
                for i in range(new_indivi_object.traits_num):
                    pheno_name = new_indivi_object.pheno_names_ls[i]
                    var = pheno_var_ls[i] #### to be improved #### 
                    genotype = new_indivi_object.genotype_set[pheno_name]
                    phenotype = np.mean(genotype) + random.gauss(0, var)
                    new_indivi_object.phenotype_set[pheno_name] = phenotype
                    new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
                self.set[h_id].add_individual(indi_object=new_indivi_object, len_id=empty_len_id, wid_id=empty_wid_id)
                counter += 1
        return counter
            
    def sex_reproduce_mutate_for_dispersal_within_patch(self, mutation_rate, pheno_var_ls, disp_within_rate):
        ''''''
        sexual_birth_rate = self.sexual_birth_rate
        counter = 0
        for h_id, h_object in self.set.items():
            h_sexual_pairwise_parents_num = h_object.hab_sexual_pairwise_parents_num() # 本地有性生殖父母本对数
            h_sex_offs_expectation_num = h_sexual_pairwise_parents_num * sexual_birth_rate # 本地有性生殖子代数
            
            sex_pairwise_parents_pos_ls = self.get_disp_within_sex_pairwise_parents_pos_ls(h_object) # 斑块内外来生境的有性繁殖父母本位置对列表
            offsprings_expection_num = int(np.around(len(sex_pairwise_parents_pos_ls) * sexual_birth_rate * disp_within_rate / (self.hab_num-1))) # 斑块内非本地有性生殖后代迁入个体数期望值
            
            h_empty_site_ls = h_object.empty_site_pos_ls # 生境的空白斑块编号列表
            if (h_sex_offs_expectation_num + offsprings_expection_num) != 0:
                disp_within_empty_site_num = int(np.around(len(h_empty_site_ls) * offsprings_expection_num/(h_sex_offs_expectation_num + offsprings_expection_num)))
            else:
                disp_within_empty_site_num = 0
            
            if offsprings_expection_num > disp_within_empty_site_num:
                disp_within_sites = random.sample(h_empty_site_ls, disp_within_empty_site_num)
                disp_within_pairwise_parent_pos_ls = random.sample(sex_pairwise_parents_pos_ls, disp_within_empty_site_num)
            else:
                disp_within_sites = random.sample(h_empty_site_ls, offsprings_expection_num)
                disp_within_pairwise_parent_pos_ls = random.sample(sex_pairwise_parents_pos_ls, offsprings_expection_num)
            
            for empty_site_pos, pairwise_parents_pos in list(zip(disp_within_sites, disp_within_pairwise_parent_pos_ls)):
                empty_len_id, empty_wid_id = empty_site_pos[0], empty_site_pos[1]
                female_parent_h_id, female_parent_row, female_parent_col = pairwise_parents_pos[0][0], pairwise_parents_pos[0][1], pairwise_parents_pos[0][2]
                male_parent_h_id, male_parent_row, male_parent_col = pairwise_parents_pos[1][0], pairwise_parents_pos[1][1], pairwise_parents_pos[1][2]
                
                female_parent_indi_object = self.set[female_parent_h_id].set['microsite_individuals'][female_parent_row][female_parent_col]
                male_parent_indi_object = self.set[male_parent_h_id].set['microsite_individuals'][male_parent_row][male_parent_col]
                
                new_indivi_object = copy.deepcopy(female_parent_indi_object)
                new_indivi_object.gender = random.sample(('male', 'female'), 1)[0]
                for i in range(new_indivi_object.traits_num):
                    pheno_name = new_indivi_object.pheno_names_ls[i]
                    var = pheno_var_ls[i] # to be improved
                    
                    female_bi_genotype = female_parent_indi_object.genotype_set[pheno_name]
                    genotype1 = random.sample(female_bi_genotype, 1)[0]
                    
                    male_bi_genotype = male_parent_indi_object.genotype_set[pheno_name]
                    genotype2 = random.sample(male_bi_genotype, 1)[0]
                    
                    new_bi_genotype = [genotype1, genotype2]
                    phenotype = np.mean(new_bi_genotype) + random.gauss(0, var)
                    
                    new_indivi_object.genotype_set[pheno_name] = new_bi_genotype
                    new_indivi_object.phenotype_set[pheno_name] = phenotype
                new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
                self.set[h_id].add_individual(indi_object=new_indivi_object, len_id=empty_len_id, wid_id=empty_wid_id)
                counter += 1
        return counter
    
    def mixed_reproduce_mutate_for_dispersal_within_patch(self, mutation_rate, pheno_var_ls, disp_within_rate):
        ''''''
        asexual_birth_rate = self.asexual_birth_rate
        sexual_birth_rate = self.sexual_birth_rate
        counter = 0
        for h_id, h_object in self.set.items():
            
            h_asexual_parent_num = h_object.hab_mixed_asexual_parent_num() # 本地无性生殖母本个数
            h_sexual_pairwise_parents_num = h_object.hab_mixed_sexual_pairwse_parents_num() # 本地有性生殖父母本对数
            h_asex_offs_expectation_num = h_asexual_parent_num * asexual_birth_rate # 本地无性繁殖子代数
            h_sex_offs_expectation_num = h_sexual_pairwise_parents_num * sexual_birth_rate # 本地有性生殖子代数
            
            mixed_asex_parent_pos_ls = self.get_disp_within_mixed_asex_parent_pos_ls(h_object) # 斑块内外来生境的无性繁殖母本位置列表
            mixed_sex_pairwise_parents_pos_ls = self.get_disp_within_mixed_sex_pairwise_parents_pos_ls(h_object) # 斑块内外来生境的有性繁殖父母本位置对列表
            asex_offs_expectation_num = int(np.around(len(mixed_asex_parent_pos_ls) * asexual_birth_rate * disp_within_rate / (self.hab_num-1))) # 斑块内非本地生境无性生殖后代迁入个体数期望值
            sex_offs_expectation_num = int(np.around(len(mixed_sex_pairwise_parents_pos_ls) * sexual_birth_rate * disp_within_rate / (self.hab_num-1))) # 斑块内非本地有性生殖后代迁入个体数期望值
            
            h_empty_site_ls = h_object.empty_site_pos_ls # 生境的空白斑块编号列表
            if (h_asex_offs_expectation_num + h_sex_offs_expectation_num + asex_offs_expectation_num + sex_offs_expectation_num) != 0:
                disp_within_empty_site_num = int(np.around(len(h_empty_site_ls) * (asex_offs_expectation_num + sex_offs_expectation_num)/(h_asex_offs_expectation_num + h_sex_offs_expectation_num + asex_offs_expectation_num + sex_offs_expectation_num)))
                # 计算可以用于迁移拓殖的空白板块期望值：只有空白的微位点可以被拓殖，本地后代和斑块内被本地的迁移后代共同按比例竞争这些空白斑块
            else:
                disp_within_empty_site_num = 0
            
            if (asex_offs_expectation_num + sex_offs_expectation_num) > disp_within_empty_site_num: # 迁移者比空白斑块多
                asex_disp_num = int(np.around((disp_within_empty_site_num * asex_offs_expectation_num/(asex_offs_expectation_num + sex_offs_expectation_num))))
                sex_disp_num = int(np.around(disp_within_empty_site_num * sex_offs_expectation_num/(asex_offs_expectation_num + sex_offs_expectation_num)))
                
                disp_within_sites = random.sample(h_empty_site_ls, asex_disp_num + sex_disp_num)
                disp_within_asex_sites = disp_within_sites[:asex_disp_num]
                disp_within_sex_sites = disp_within_sites[asex_disp_num:asex_disp_num + sex_disp_num]
                
                disp_within_asexual_parent_pos_ls = random.sample(mixed_asex_parent_pos_ls, asex_disp_num)
                disp_within_pairwise_parent_pos_ls = random.sample(mixed_sex_pairwise_parents_pos_ls, sex_disp_num)
            else:                                                                                   # 迁移者比空白斑块少
                disp_within_sites = random.sample(h_empty_site_ls, asex_offs_expectation_num + sex_offs_expectation_num)
                disp_within_asex_sites = disp_within_sites[:asex_offs_expectation_num]
                disp_within_sex_sites = disp_within_sites[asex_offs_expectation_num:asex_offs_expectation_num + sex_offs_expectation_num]
                
                disp_within_asexual_parent_pos_ls = random.sample(mixed_asex_parent_pos_ls, asex_offs_expectation_num)
                disp_within_pairwise_parent_pos_ls = random.sample(mixed_sex_pairwise_parents_pos_ls, sex_offs_expectation_num)
                
            for empty_site_pos, parent_pos in list(zip(disp_within_asex_sites, disp_within_asexual_parent_pos_ls)):
                ''' asexual reproduction'''
                empty_len_id, empty_wid_id = empty_site_pos[0], empty_site_pos[1]
                parent_h_id, parent_row, parent_col = parent_pos[0], parent_pos[1], parent_pos[2]
                
                parent_indi_object = self.set[parent_h_id].set['microsite_individuals'][parent_row][parent_col]
                new_indivi_object = copy.deepcopy(parent_indi_object)
                
                for i in range(new_indivi_object.traits_num):
                    pheno_name = new_indivi_object.pheno_names_ls[i]
                    var = pheno_var_ls[i] #### to be improved #### 
                    genotype = new_indivi_object.genotype_set[pheno_name]
                    phenotype = np.mean(genotype) + random.gauss(0, var)
                    new_indivi_object.phenotype_set[pheno_name] = phenotype
                    new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
                self.set[h_id].add_individual(indi_object=new_indivi_object, len_id=empty_len_id, wid_id=empty_wid_id)
                counter += 1
                
            for empty_site_pos, pairwise_parents_pos in list(zip(disp_within_sex_sites, disp_within_pairwise_parent_pos_ls)):
                ''' sexual reproduction '''
                empty_len_id, empty_wid_id = empty_site_pos[0], empty_site_pos[1]
                female_parent_h_id, female_parent_row, female_parent_col = pairwise_parents_pos[0][0], pairwise_parents_pos[0][1], pairwise_parents_pos[0][2]
                male_parent_h_id, male_parent_row, male_parent_col = pairwise_parents_pos[1][0], pairwise_parents_pos[1][1], pairwise_parents_pos[1][2]
                
                female_parent_indi_object = self.set[female_parent_h_id].set['microsite_individuals'][female_parent_row][female_parent_col]
                male_parent_indi_object = self.set[male_parent_h_id].set['microsite_individuals'][male_parent_row][male_parent_col]
                
                new_indivi_object = copy.deepcopy(female_parent_indi_object)
                new_indivi_object.gender = random.sample(('male', 'female'), 1)[0]
                for i in range(new_indivi_object.traits_num):
                    pheno_name = new_indivi_object.pheno_names_ls[i]
                    var = pheno_var_ls[i] # to be improved
                    
                    female_bi_genotype = female_parent_indi_object.genotype_set[pheno_name]
                    genotype1 = random.sample(female_bi_genotype, 1)[0]
                    
                    male_bi_genotype = male_parent_indi_object.genotype_set[pheno_name]
                    genotype2 = random.sample(male_bi_genotype, 1)[0]
                    
                    new_bi_genotype = [genotype1, genotype2]
                    phenotype = np.mean(new_bi_genotype) + random.gauss(0, var)
                    
                    new_indivi_object.genotype_set[pheno_name] = new_bi_genotype
                    new_indivi_object.phenotype_set[pheno_name] = phenotype
                new_indivi_object.mutation(rate=mutation_rate, pheno_var_ls=pheno_var_ls)
                self.set[h_id].add_individual(indi_object=new_indivi_object, len_id=empty_len_id, wid_id=empty_wid_id)
                counter += 1
        return counter

    def patch_dormancy_processes(self):
        for h_id, h_object in self.set.items():
            h_object.hab_dormancy_process()
        return 0
    
    def patch_germinate_from_offsprings_pool(self):
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_germinate_from_offsprings_pool()
        return counter
    
    def patch_asexual_birth_germinate(self, mutation_rate, pheno_var_ls):
        asexual_birth_rate = self.asexual_birth_rate
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_asexual_reprodece_germinate(asexual_birth_rate, mutation_rate, pheno_var_ls)
        return counter
    
    def patch_sexual_birth_germinate(self, mutation_rate, pheno_var_ls):
        sexual_birth_rate = self.sexual_birth_rate
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_sexual_reprodece_germinate(sexual_birth_rate, mutation_rate, pheno_var_ls)
        return counter
    
    def patch_mixed_birth_germinate(self, mutation_rate, pheno_var_ls):
        asexual_birth_rate = self.asexual_birth_rate
        sexual_birth_rate = self.sexual_birth_rate
        counter = 0
        for h_id, h_object in self.set.items():
            counter += h_object.hab_mixed_reproduce_germinate(asexual_birth_rate, sexual_birth_rate, mutation_rate, pheno_var_ls)
        return counter
        
class metacommunity():
    def __init__(self, metacommunity_name):
        self.set = {}                       # self.data_set={} # to be improved
        self.patch_num = 0
        self.meta_map = nx.Graph()
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
        self.meta_map.add_node(patch_name)
        self.patch_id_ls.append(patch_name)
        self.patch_id_2_index_dir[patch_name] = patch_object.index
        self.disp_current_matrix = np.matrix(np.zeros((self.patch_num, self.patch_num)))
        
    def get_all_patches_location(self):
        output = {}
        for key, value in self.set.items():
            output[key]=value.location
        return output
    
    def get_meta_individual_num(self):
        num = 0
        for patch_id, patch_object in self.set.items():
            num += patch_object.get_patch_individual_num()
        return num
    
    def show_meta_individual_num(self):
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
    def show_meta_map(self, graph_object, title, pos=None):
        if pos == None:
            pos = self.get_all_patches_location()
        plt.figure(figsize=(10,8))
        plt.title(title, fontsize = 16)
        nx.draw_networkx(graph_object, pos=pos)
        nx.draw_networkx(graph_object, pos=pos, edge_color='b') 
        plt.savefig('meta_network.jpg')
        return 0
    
    def show_meta_species_distribution(self, cmap, file_name):
        pass
    
    def get_meta_microsites_individuals_sp_id_values(self):
        ''' '''
        meta_sp_dis = np.array([], dtype=int)
        for patch_id, patch_object in self.set.items():
            patch_sp_dis = patch_object.get_patch_microsites_individals_sp_id_values()
            meta_sp_dis = np.append(meta_sp_dis, patch_sp_dis.reshape(-1))
        return meta_sp_dis
    
    def get_meta_microsites_individuals_phenotype_values(self, trait_name):
        ''''''
        meta_phenotype_dis = np.array([])
        for patch_id, patch_object in self.set.items():
            patch_phenotype_dis = patch_object.get_patch_microsites_individals_phenotype_values(trait_name)
            meta_phenotype_dis = np.append(meta_phenotype_dis, patch_phenotype_dis.reshape(-1))
        meta_phenotype_dis = np.around(meta_phenotype_dis, 3) # saving the storage
        return meta_phenotype_dis
    
    def get_meta_microsite_environment_values(self, environment_name):
        ''''''
        meta_environment_dis = np.array([])
        for patch_id, patch_object in self.set.items():
            patch_environment_dis = patch_object.get_patch_microsites_environment_values(environment_name)
            meta_environment_dis = np.append(meta_environment_dis, patch_environment_dis.reshape(-1))
        return meta_environment_dis
        
    def get_meta_microsites_optimum_sp_id_val(self, d, w, species_2_phenotype_ls):
        ''' '''
        meta_optimum_sp_id_val_dis = np.array([])
        for patch_id, patch_object in self.set.items():
            patch_optimum_sp_id_val_dis = patch_object.get_patch_microsites_optimum_sp_id_value_array(d, w, species_2_phenotype_ls)
            meta_optimum_sp_id_val_dis = np.append(meta_optimum_sp_id_val_dis, patch_optimum_sp_id_val_dis.reshape(-1))
        return meta_optimum_sp_id_val_dis
    
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
                
    def meta_distribution_data_all_time_to_csv_gz(self, dis_data_all_time, first_row_index_name, all_time_step, file_name):
        ''' '''
        columns_patch_id, columns_habitat_id, columns_mocrosite_id = self.columns_patch_habitat_microsites_id()
        columns = [columns_patch_id, columns_habitat_id, columns_mocrosite_id]
        index = [first_row_index_name]+['time_step%d'%i for i in range(all_time_step)]
        
        df_species_distribution = pd.DataFrame(dis_data_all_time, index=index, columns=columns)
        df_species_distribution.to_csv(file_name, compression='gzip')
        return df_species_distribution
    
    def meta_disp_current_mat_to_csv(self, file_name):
        ''' '''
        disp_current_matrix = self.disp_current_matrix
        index, columns = self.patch_id_ls, self.patch_id_ls
        
        df_disp_current_matrix = pd.DataFrame(disp_current_matrix, index=index, columns=columns)
        df_disp_current_matrix.to_csv(file_name)
        return df_disp_current_matrix

    #### to be improved ####
    def customize_meta_map(self):
        pass
    ########################
    def int2bin(self, n, count):
        ''' returns the binary of integer n, using count number of digits'''
        return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])
    
    def empty_graph(self):
        ''' return an empty graph of the patches '''
        empty_graph = nx.Graph()
        pos = self.get_all_patches_location() # localtion of patch in dir
        for patch_id, patch_location in pos.items():
            empty_graph.add_node(patch_id)
        nx.set_node_attributes(empty_graph, pos, name='position')
        return empty_graph
    
    def full_con_map(self):
        ''' return full connected metacommunity network '''
        full_con_map = nx.Graph()
        pos = self.get_all_patches_location() # localtion of patch in dir
        for key1, value1 in pos.items():
            for key2, value2 in pos.items():
                if key1 != key2:
                    distance = math.sqrt(math.pow((value1[0]-value2[0]), 2)+math.pow((value1[1]-value2[1]), 2))
                    #print(key1, value1, key2, value2, distance)
                    full_con_map.add_edge(key1, key2, weight = distance)
                    # add all posible edges to form Graph with full connectance
        nx.set_node_attributes(full_con_map, pos, name='position')
        return full_con_map
                    
    def mini_span_tree(self):
        ''' return minimum connected metacommunity network '''
        full_con_map = self.full_con_map()
        mini_map = nx.minimum_spanning_tree(full_con_map)
        return mini_map
        
    def med_con_map(self, add_links_propotion):
        ''' return medimum connected metacommunity network with the degree of 'add_links_propotion'. '''
        full_con_map = self.full_con_map()
        mini_con_map = self.mini_span_tree()
        med_con_map = mini_con_map
        all_add_links = set(full_con_map.edges())-set(mini_con_map.edges())
        add_links = random.sample(all_add_links, int(len(all_add_links)*add_links_propotion))
        pos = self.get_all_patches_location()
        for edge in add_links:
            patch_id1, patch_id2 = edge[0], edge[1]
            location1, location2 = pos[patch_id1], pos[patch_id2]
            distance = math.sqrt(math.pow((location1[0]-location2[0]), 2)+math.pow((location1[1]-location2[1]), 2))
            med_con_map.add_edge(patch_id1, patch_id2, weight = distance)
        return med_con_map
    
    def paul_revere_network(self):
        ''' using dynamic programming to return a paul revere network with no more than 25 nodes (for 32G memory computer)'''
        full_con_map = self.full_con_map()
        index_to_nodes = list(full_con_map.nodes)
        pos = self.get_all_patches_location()                                  # {patch_id : (x, y)}; localtion of patch in dir
        
        nodes_num = len(full_con_map.nodes)
        state_num = 2**nodes_num
        
        ad_mat = nx.adjacency_matrix(full_con_map).todense()
        dp = np.matrix(np.ones((nodes_num, state_num)) * np.inf)
        prev = np.matrix(np.ones((nodes_num, state_num)) * np.inf)
        
        #dp[0, 1] = 0
        for i in range(nodes_num):
            dp[i, 2**i] = 0
        
        for j in range(1, state_num):                                          # state
            for i in range(1, nodes_num):                                      # next unvisited node
                if self.int2bin(j, nodes_num)[-(i+1)]=='0':                    # the next node is unvisted
                    for k in range(nodes_num):                                 # a visted intermediate node to the new unvisited next node
                        #print(j, i, k)
                        if dp[k, j] != np.inf:                                 # the intermediatenode is visited
                            if dp[k, j] + ad_mat[i, k] < dp[i, j+2**i]:
                                dp[i, j+2**i] = dp[k, j] + ad_mat[i, k]
                                prev[i, j+2**i] = k
        ans = float('inf')
        final_nodes_index = float('inf')
        
        for i in range(nodes_num):
            if dp[i, state_num-1] < ans: 
                ans = dp[i, state_num-1]
                final_nodes_index = i
        
        path_list = [final_nodes_index]
        current_node_index = final_nodes_index
        current_state = 2**nodes_num - 1
        
        for nodes in range(nodes_num-1):
            pre_node_index = prev[current_node_index, current_state]
            if pre_node_index == np.inf:
                break
            else:
                pre_state = current_state - 2**current_node_index
                path_list.insert(0, int(pre_node_index))
            
                current_node_index = int(pre_node_index)
                current_state = pre_state
            
        paul_revere_graph = nx.Graph()
        for node_id in list(full_con_map.nodes):
            paul_revere_graph.add_node(node_id)
        
        for i in range(len(path_list)-1):
            node1_index = path_list[i]
            node1_patch_id = index_to_nodes[node1_index]
            location1 = pos[node1_patch_id]
        
            node2_index = path_list[i+1]
            node2_patch_id = index_to_nodes[node2_index]
            location2 = pos[node2_patch_id]
            
            distance = math.sqrt(math.pow((location1[0]-location2[0]), 2)+math.pow((location1[1]-location2[1]), 2))
            paul_revere_graph.add_edge(node1_patch_id, node2_patch_id, weight = distance)
        nx.set_node_attributes(paul_revere_graph, pos, name='position')
        return paul_revere_graph
##################################################################################################################
    def dyn_pro_traveling_salesman_network(self):
        ''' TSP problem '''
        full_con_map = self.full_con_map()
        index_to_nodes = list(full_con_map.nodes)
        pos = self.get_all_patches_location()                                  # {patch_id : (x, y)}; localtion of patch in dir
        
        nodes_num = len(full_con_map.nodes)
        state_num = 2**nodes_num
        
        ad_mat = nx.adjacency_matrix(full_con_map).todense()
        dp = np.matrix(np.ones((nodes_num, state_num)) * np.inf)
        prev = np.matrix(np.ones((nodes_num, state_num)) * np.inf)
        
        #dp[0, 1] = 0
        for i in range(nodes_num):
            dp[i, 2**i] = 0
        
        for j in range(1, state_num):                                          # state
            for i in range(1, nodes_num):                                      # next unvisited node
                if self.int2bin(j, nodes_num)[-(i+1)]=='0':                    # the next node is unvisted
                    for k in range(nodes_num):                                 # a visted intermediate node to the new unvisited next node
                        #print(j, i, k)
                        if dp[k, j] != np.inf:                                 # the intermediatenode is visited
                            if dp[k, j] + ad_mat[i, k] < dp[i, j+2**i]:
                                dp[i, j+2**i] = dp[k, j] + ad_mat[i, k]
                                prev[i, j+2**i] = k
        ans = float('inf')
        final_nodes_index = float('inf')
        
        for i in range(nodes_num):
            if dp[i, state_num-1] < ans: 
                ans = dp[i, state_num-1]
                final_nodes_index = i
        
        path_list = [final_nodes_index]
        current_node_index = final_nodes_index
        current_state = 2**nodes_num - 1
        
        for nodes in range(nodes_num-1):
            pre_node_index = prev[current_node_index, current_state]
            if pre_node_index == np.inf:
                break
            else:
                pre_state = current_state - 2**current_node_index
                path_list.insert(0, int(pre_node_index))
            
                current_node_index = int(pre_node_index)
                current_state = pre_state
            
        paul_revere_graph = nx.Graph()
        for node_id in list(full_con_map.nodes):
            paul_revere_graph.add_node(node_id)
        
        for i in range(len(path_list)-1):
            node1_index = path_list[i]
            node1_patch_id = index_to_nodes[node1_index]
            location1 = pos[node1_patch_id]
        
            node2_index = path_list[i+1]
            node2_patch_id = index_to_nodes[node2_index]
            location2 = pos[node2_patch_id]
            
            distance = math.sqrt(math.pow((location1[0]-location2[0]), 2)+math.pow((location1[1]-location2[1]), 2))
            paul_revere_graph.add_edge(node1_patch_id, node2_patch_id, weight = distance)
        
        node1_index = path_list[0]
        node1_patch_id = index_to_nodes[node1_index]
        location1 = pos[node1_patch_id]
    
        node2_index = path_list[-1]
        node2_patch_id = index_to_nodes[node2_index]
        location2 = pos[node2_patch_id]
        
        distance = math.sqrt(math.pow((location1[0]-location2[0]), 2)+math.pow((location1[1]-location2[1]), 2))
        paul_revere_graph.add_edge(node1_patch_id, node2_patch_id, weight = distance)
        nx.set_node_attributes(paul_revere_graph, pos, name='position')
        return paul_revere_graph
    
    def traveling_salesman_network(self):
        ''' TSP problem '''
        full_con_map = self.full_con_map()
        pos = self.get_all_patches_location()                                  # {patch_id : (x, y)}; localtion of patch in dir
        
        tsp_graph = nx.Graph()
        #tsp_path = approximation.traveling_salesman_problem(G=full_con_map, weight='weight')
        init_cycle = approximation.greedy_tsp(G=full_con_map, weight='weight')
        tsp_path = approximation.simulated_annealing_tsp(G=full_con_map, init_cycle=init_cycle, weight='weight')
        
        for i in range(len(tsp_path)-1):
            node1_patch_id = tsp_path[i]
            location1 = pos[node1_patch_id]
        
            node2_patch_id = tsp_path[i+1]
            location2 = pos[node2_patch_id]
            
            distance = math.sqrt(math.pow((location1[0]-location2[0]), 2)+math.pow((location1[1]-location2[1]), 2))
            tsp_graph.add_edge(node1_patch_id, node2_patch_id, weight = distance)
        nx.set_node_attributes(tsp_graph, pos, name='position')
        return tsp_graph
####################################################################################################
    def star_graph(self, m):
        ''' star graph with top m centraility patches'''
        star_graph = nx.Graph()
        
        full_con_map = self.full_con_map()
        pos = self.get_all_patches_location() ## {patch_id : (x, y)}; localtion of patch in dir
        
        clossness_centrality_sorted_ls = sorted(nx.closeness_centrality(G=full_con_map, distance='weight').items(), key = lambda kv:(kv[1], kv[0]))
        clossness_centrality_sorted_ls.reverse() # from large to small, [(patch_id, centrality_degree),...,]
        top_m_nodes_id_ls = list(np.array(clossness_centrality_sorted_ls)[:, 0][:m]) # top m centrality degree patches for star graph
        
        internal_node_id = top_m_nodes_id_ls[0]
        internal_node_location = pos[internal_node_id]
        
        for i in range(1, len(top_m_nodes_id_ls)):
            leaf_node_id = top_m_nodes_id_ls[i]
            leaf_node_location = pos[leaf_node_id]
            
            distance = math.sqrt(math.pow((internal_node_location[0]-leaf_node_location[0]), 2)+math.pow((internal_node_location[1]-leaf_node_location[1]), 2))
            star_graph.add_edge(internal_node_id, leaf_node_id, weight = distance)
        return star_graph
        
    def barabasi_albert_graph(self, m):
        ''' BA network '''
        full_con_map = self.full_con_map()
        pos = self.get_all_patches_location()
        
        clossness_centrality_sorted_ls = sorted(nx.closeness_centrality(G=full_con_map, distance='weight').items(), key = lambda kv:(kv[1], kv[0]))
        clossness_centrality_sorted_ls.reverse() # from large to small, [(patch_id, centrality_degree),...,]
        sorted_nodes_id_ls = list(np.array(clossness_centrality_sorted_ls)[:, 0])
        
        G = self.star_graph(m+1)
        repeated_nodes = [n for n, d in G.degree() for _ in range(d)] 
        
        for new_node_id in sorted_nodes_id_ls:
            if new_node_id not in repeated_nodes:
                targets_nodes_id_ls = random.sample(repeated_nodes, m)
                for new_node_id, targets_nodes_id in zip([new_node_id] * m, targets_nodes_id_ls):
                    location1 = pos[new_node_id]
                    location2 = pos[targets_nodes_id]
                    distance = math.sqrt(math.pow((location1[0]-location2[0]), 2)+math.pow((location1[1]-location2[1]), 2))
                    
                    G.add_edge(new_node_id, targets_nodes_id, weight=distance)
                    repeated_nodes.append(new_node_id)
                    repeated_nodes.append(targets_nodes_id)
        nx.set_node_attributes(G, pos, name='position')
        return G
#####################################################################################################
    def one_center_network(self):
        ''' return hierachical network with one central node'''
        full_con_map = self.full_con_map()
        pos = self.get_all_patches_location()
        
        G = nx.Graph()
        for node_id in list(full_con_map.nodes):
            G.add_node(node_id)
        
        clossness_centrality_sorted_ls = sorted(nx.closeness_centrality(G=full_con_map, distance='weight').items(), key = lambda kv:(kv[1], kv[0]))
        clossness_centrality_sorted_ls.reverse() # from large to small, [(patch_id, centrality_degree),...,]
        sorted_nodes_id_ls = list(np.array(clossness_centrality_sorted_ls)[:, 0])
        
        center_node_id = sorted_nodes_id_ls[0]
        for new_node_id in sorted_nodes_id_ls:
            if new_node_id != center_node_id:
                location1 = pos[center_node_id]
                location2 = pos[new_node_id]
                distance = math.sqrt(math.pow((location1[0]-location2[0]), 2)+math.pow((location1[1]-location2[1]), 2))
                
                G.add_edge(center_node_id, new_node_id, weight=distance)
        nx.set_node_attributes(G, pos, name='position')
        return G
    
    def full_con_map_with_all_floating_points(self, x_range, y_range):
        G = nx.Graph()
        
        all_nodes_ls = list(itertools.product(x_range, y_range))
        
        full_con_map = self.full_con_map()
        pos = self.get_all_patches_location()
        pos_G = {}
        
        for node_id in full_con_map.nodes:
            G.add_node(node_id)
            
        for node_id in all_nodes_ls:
            if node_id not in pos.values():
                G.add_node(node_id)
        
        for node_id in G.nodes:
            if node_id in pos.keys():
                location = pos[node_id]
            else:
                location = node_id
                
            pos_G[node_id] = location
                
        for node1_id in G.nodes:
            for node2_id in G.nodes:
                if node1_id != node2_id:
                    location1 = pos_G[node1_id]
                    location2 = pos_G[node2_id]
                    distance = math.sqrt(math.pow((location1[0]-location2[0]), 2)+math.pow((location1[1]-location2[1]), 2))
                    G.add_edge(node1_id, node2_id, weight=distance)
                else:
                    continue
        return G, pos_G
    
    def sub_state(self, s):
        ''' return list of sub set of state s'''
        ls = []
        ss = s
        while ss > 0:
            ss -= 1
            ss &= s
            ls.append(ss)
        return ls
    
    def get_link_nodes(self, graph_object, node_id):
        ''' return a list of nodes linking the target node given '''
        linking_node_id_ls = []
        edges_ls = list(graph_object.edges)
        for edge in edges_ls:
            node1_id = edge[0]
            node2_id = edge[1]

            if node1_id == node_id:
                if node2_id not in linking_node_id_ls:
                    linking_node_id_ls.append(node2_id)
            elif node2_id == node_id:
                if node1_id not in linking_node_id_ls:
                    linking_node_id_ls.append(node1_id)
        return linking_node_id_ls
        
    def nodes_ls_index_ls(self, graph_object, terminal_nodes_ls):
        graph_node_ls = list(graph_object.nodes)
        for node_id in list(graph_object.nodes):
            if node_id in terminal_nodes_ls:
                graph_node_ls.remove(node_id)
                
        new_node_id_ls = terminal_nodes_ls + graph_node_ls
        return new_node_id_ls
    
    def ad_mat_exchange(self, graph_object, terminal_nodes_ls):
        
        old_nodes_ls = list(graph_object.nodes)
        old_ad_mat = nx.adjacency_matrix(graph_object).todense()
        new_nodes_ls = self.nodes_ls_index_ls(graph_object, terminal_nodes_ls)
        col_permutation = np.matrix(np.zeros((len(graph_object.nodes), len(graph_object.nodes))))
        
        for node_id in new_nodes_ls:
            old_index = old_nodes_ls.index(node_id)
            new_index = new_nodes_ls.index(node_id)
            col_permutation[old_index, new_index] = 1
            
        row_permutation = col_permutation.T
        new_ad_mat = row_permutation * old_ad_mat * col_permutation
        return new_ad_mat
            
    def dyn_pro_striner_tree_network(self, graph_object, terminal_nodes_ls):
        ''' using dynamic programming return a steiner tree network '''
        
        nodes_num = len(list(graph_object.nodes))
        k = len(terminal_nodes_ls)
        state_num = 2**k
        
        #ad_mat = nx.adjacency_matrix(graph_object).todense()
        #index2nodes = list(graph_object.nodes)
        
        nodes_and_index = self.nodes_ls_index_ls(graph_object, terminal_nodes_ls)
        new_ad_mat = self.ad_mat_exchange(graph_object, terminal_nodes_ls)
        
        dp = np.matrix(np.ones((nodes_num, state_num)) * np.inf)
        pre = [[np.inf for _ in range(state_num)] for _ in range(nodes_num)]

        vis = np.matrix(np.zeros((nodes_num, state_num)), dtype=int)
        q = Queue()
        
        for i in range(k):
            dp[i, 2**i] = 0
        
        for s in range(1, state_num):
            print(s)
            for ss in self.sub_state(s):
                for i in range(nodes_num):
                    if dp[i, s] > dp[i, ss] + dp[i, s^ss]:
                        dp[i, s] = dp[i, ss] + dp[i, s^ss]
                        pre[i][s] = (i, ss)
                    
            for i in range(nodes_num):
                if dp[i, s] != np.inf:
                    q.put(i)
                    vis[i, s] = 1
            while q.qsize() > 0:
                node_index = q.get()
                node_id = nodes_and_index[node_index]
                vis[node_index, s] = 0
                for link_node_id in self.get_link_nodes(graph_object, node_id):
                    link_node_index = nodes_and_index.index(link_node_id)
                    if dp[node_index, s] + new_ad_mat[node_index, link_node_index] < dp[link_node_index, s]:
                        dp[link_node_index, s] = dp[node_index, s] + new_ad_mat[node_index, link_node_index]
                        pre[link_node_index][s] = (node_index, s)
                        if vis[link_node_index, s] == 0 :
                            q.put(link_node_index)
                            vis[link_node_index, s] = 1
                        else:
                            continue
                    else:
                        continue
                    
        ans = np.inf
        for i in range(nodes_num):
            if dp[i, state_num-1] < ans:
                ans = dp[i, state_num-1]
                node_sta = (i, state_num-1) 
                  
        tran_q = Queue()
        tran_q.put(node_sta)
        path_ls = []
        while tran_q.qsize() > 0:
            node_sta_root = tran_q.get()
            node_sta_leaf = pre[node_sta_root[0]][node_sta_root[1]]
            
            if node_sta_leaf == np.inf:
                continue
            else:
                if node_sta_root[1] != node_sta_leaf[1]: # 状态不一样
                    s = node_sta_root[1]
                    ss = node_sta_leaf[1]
                    node_sta_leaf_comp = (node_sta_leaf[0], s^ss)
                    tran_q.put(node_sta_leaf)
                    tran_q.put(node_sta_leaf_comp)
                else:                                    # 状态一样
                    tran_q.put(node_sta_leaf)
                    path_ls.append((nodes_and_index[node_sta_root[0]], nodes_and_index[node_sta_leaf[0]]))
        
        return ans, path_ls
    
    def steiner_tree(self, x_range, y_range):
        full_con_map = self.full_con_map()
        G, pos_G = self.full_con_map_with_all_floating_points(x_range, y_range)
        ans, path_ls = self.dyn_pro_striner_tree_network(graph_object=G, terminal_nodes_ls=list(full_con_map.nodes))
        steiner_tree = nx.Graph()
        pos_steiner = {}
        
        for key_node_id in list(full_con_map.nodes):
            steiner_tree.add_node(key_node_id)
            
        for edge in path_ls:
            node1_id = edge[0]
            node2_id = edge[1]
            location1 = pos_G[node1_id]
            location2 = pos_G[node2_id]
            distance = math.sqrt(math.pow((location1[0]-location2[0]), 2)+math.pow((location1[1]-location2[1]), 2))
            steiner_tree.add_edge(node1_id, node2_id, weight=distance)
            
        for node_id in steiner_tree.nodes:
            pos_steiner[node_id] = pos_G[node_id]
            
        nx.set_node_attributes(steiner_tree, pos_steiner, name='position')
        return steiner_tree, pos_steiner
    
####################################################################################
    def central_hierachical_network(self):
        ''' return a hierachical network to the central node '''
        
        one = self.one_center_network()
        node_num = len(one.nodes())
        ad_mat_hier = np.matrix(np.ones((node_num, node_num)) * np.inf)
                               
        full_con_map = self.full_con_map()
        ad_mat = nx.adjacency_matrix(full_con_map).todense()
        index_to_nodes = list(full_con_map.nodes)
                                
        for i in range(len(one.nodes)):
            ad_mat_hier[i, i] = 0
            
        for edge in one.edges:
            node1_id = edge[0]
            node1_index = index_to_nodes.index(node1_id)
            node2_id = edge[1]
            node2_index = index_to_nodes.index(node2_id)
            ad_mat_hier[node1_index, node2_index] = ad_mat[node1_index, node2_index]
            ad_mat_hier[node2_index, node1_index] = ad_mat[node2_index, node1_index]
                       
        
        clossness_centrality_sorted_ls = sorted(nx.closeness_centrality(G=full_con_map, distance='weight').items(), key = lambda kv:(kv[1], kv[0]))
        clossness_centrality_sorted_ls.reverse() # from large to small, [(patch_id, centrality_degree),...,]
        sorted_nodes_id_ls = list(np.array(clossness_centrality_sorted_ls)[:, 0])
        
        central_node_id = sorted_nodes_id_ls[0]
        central_node_index = index_to_nodes.index(central_node_id)
    
        add_path = []
        for new_node_id in list(full_con_map.nodes):
            new_node_index = index_to_nodes.index(new_node_id)
            path = None
            dis = ad_mat[new_node_index, central_node_index]
            if new_node_id != central_node_id:
                
                for mid_node_id in list(full_con_map.nodes):
                    mid_node_index = index_to_nodes.index(mid_node_id)
                    if mid_node_id != central_node_id and mid_node_id != new_node_id:
                        if ad_mat[new_node_index, mid_node_index]  < dis:
                            dis = ad_mat[new_node_index, mid_node_index]
                            path = (mid_node_id, new_node_id)
                        else:
                            continue
                    else:
                        continue
                
                if path != None:
                    add_path.append(path)
                    one.add_edge(path[0], path[1], weight = dis)
                else:
                    continue
            else:
                continue
            
        for path in add_path:
            node1_id = path[0]
            node1_index = index_to_nodes.index(node1_id)
            node2_id = path[1]
            node2_index = index_to_nodes.index(node2_id)
            
            dis_1 = ad_mat[node1_index, central_node_index]
            dis_2 = ad_mat[node2_index, central_node_index]

            if dis_1 < dis_2:
                try:
                    one.remove_edge(central_node_id, node2_id)
                except:
                    continue
            else:
                try:
                    one.remove_edge(central_node_id, node1_id)
                except:
                    continue
                
        # 判断图中是否存在回路，并去除回路中最长的一条边
        cycles_ls_list = nx.cycle_basis(one)
        for cycle_ls in cycles_ls_list:
            cycle_ls.append(cycle_ls[0])
            longest_path_dis = 0
            for i in range(len(cycle_ls)-1):
                node1_id = cycle_ls[i]
                node1_index = index_to_nodes.index(node1_id)
                node2_id = cycle_ls[i+1]
                node2_index = index_to_nodes.index(node2_id)
                dis = ad_mat[node1_index, node2_index]
                if dis > longest_path_dis:
                    longest_path_dis = dis
                    longest_path = (node1_id, node2_id)
            print(longest_path[0], longest_path[1])
            one.remove_edge(longest_path[0], longest_path[1])
        return one
    
    def full_one_over_distance(self):
        full = nx.Graph()
        pos = self.get_all_patches_location() # localtion of patch in dir
        for key1, value1 in pos.items():
            for key2, value2 in pos.items():
                if key1 != key2:
                    distance = math.sqrt(math.pow((value1[0]-value2[0]), 2)+math.pow((value1[1]-value2[1]), 2))
                    #print(key1, value1, key2, value2, distance)
                    full.add_edge(key1, key2, weight = 1/distance)
                    # add all posible edges to form Graph with full connectance
        nx.set_node_attributes(full, pos, name='position')
        return full
    
    def k_factor_regular_random_network(self, k=4):
        ''' spanning a regular random network with k factors and k is the degree of each nodes '''
        full_one_over_distance = self.full_one_over_distance()
        regular_graph = self.empty_graph()
        pos = self.get_all_patches_location()       
        regular_graph_one_over_distance = nx.k_factor(G=full_one_over_distance, k=k, matching_weight='weight')
        for edge in regular_graph_one_over_distance.edges:
            node1_id = edge[0]
            node2_id = edge[1]
            location1 = pos[node1_id]
            location2 = pos[node2_id]
            distance = math.sqrt(math.pow((location1[0]-location2[0]), 2)+math.pow((location1[1]-location2[1]), 2))
            regular_graph.add_edge(node1_id, node2_id, weight=distance)
        return regular_graph
    
    def small_world_random_graph(self):
        regular_graph = self.k_factor_regular_random_network()
        pos = self.get_all_patches_location()     
        small_world = nx.lattice_reference(G=regular_graph, niter=5, D=None, connectivity=True, seed=None)
        nx.set_node_attributes(small_world, pos, name='position')
        
        for edge in small_world.edges:
            node1_id = edge[0]
            node2_id = edge[1]
            location1 = pos[node1_id]
            location2 = pos[node2_id]
            distance = math.sqrt(math.pow((location1[0]-location2[0]), 2)+math.pow((location1[1]-location2[1]), 2))
            small_world.add_edge(node1_id, node2_id, weight=distance)
        return small_world
    
##################### colonization from mainland species pool #####################
    def colonize_from_propagules_rains(self, species_pool_obj, reproduce_mode, propagules_rain_num):
        ''' colonizing the metacommunity from propagules rains of mainland species pool '''
        propagules_rains_ls = species_pool_obj.generate_propagules_rain_ls(num=propagules_rain_num, reproduce_mode=reproduce_mode)
        meta_empty_sites_ls = self.get_meta_empty_sites_ls()
        random.shuffle(propagules_rains_ls)
        random.shuffle(meta_empty_sites_ls)
        counter = 0
        for indi_object, empty_site_pos in list(zip(propagules_rains_ls, meta_empty_sites_ls)):
            patch_id = empty_site_pos[0]
            h_id = empty_site_pos[1]
            len_id = empty_site_pos[2]
            wid_id = empty_site_pos[3]
            self.set[patch_id].set[h_id].add_individual(indi_object = indi_object, len_id=len_id, wid_id=wid_id)
            counter += 1
            
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals colonizing the metacommunity from mainland; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
    def pairwise_sexual_colonization_from_prpagules_rains(self, species_pool_obj, propagules_rain_num):
        ''' pairwise sexual parents colonizing the metacommunity from propagules rains of mainland species pool '''
        pairwise_num = int(propagules_rain_num/2)
        pairwise_propagules_rain_ls = species_pool_obj.generate_pairwise_sexual_propagules_rain_ls(pairwise_num) #[(female_obj, male_obj), ..., (female_obj, male_obj)]
        meta_pairwise_empty_sites_ls = self.get_meta_pairwise_empty_sites_ls() # [(empty_site_1_pos, empty_site_2_pos) ... ] and the two sites are in the same habitat, and empty_site_1_pos = [(patch_id, h_id, len_id, wid_id)]
        random.shuffle(pairwise_propagules_rain_ls)
        random.shuffle(meta_pairwise_empty_sites_ls)
        counter = 0
        for (female_obj, male_obj), (empty_site_1_pos, empty_site_2_pos) in list(zip(pairwise_propagules_rain_ls, meta_pairwise_empty_sites_ls)):
            
            site_1_patch_id, site_1_h_id, site_1_len_id, site_1_wid_id = empty_site_1_pos[0], empty_site_1_pos[1], empty_site_1_pos[2], empty_site_1_pos[3]
            self.set[site_1_patch_id].set[site_1_h_id].add_individual(indi_object = female_obj, len_id=site_1_len_id, wid_id=site_1_wid_id)
            
            site_2_patch_id, site_2_h_id, site_2_len_id, site_2_wid_id = empty_site_2_pos[0], empty_site_2_pos[1], empty_site_2_pos[2], empty_site_2_pos[3]
            self.set[site_2_patch_id].set[site_2_h_id].add_individual(indi_object = male_obj, len_id=site_2_len_id, wid_id=site_2_wid_id)
            counter += 2
        
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals colonizing the metacommunity from mainland; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
################################ global warming ###########################################
    def global_warming(self, env_name, increment_value):
        ''' increasing the 'env_name' mean value with the increment_value in each habitat '''
        for patch_id, patch_object in self.set.items():
            for h_id, h_object in patch_object.set.items():
                env_index = h_object.env_types_name.index(env_name)
                mean_env_value = h_object.mean_env_ls[env_index] + increment_value
                var_env_value = h_object.var_env_ls[env_index]
                
                h_object.update_one_environment_values(env_name, mean_env_value, var_env_value)
        return 0
###################################################################################
    def dist2disp_function(self, k, x):
        ''' Exponential decay model for dispersal among patches.
        k is a scaling factor determining the strength of dispersal limitation.
        x is the distance between the two patches. '''
        return k * np.exp(-k*x)
    
    def mat_around(self, matrix):       
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
    
    def normalize_dispersal_among_patches_matrix(self, total_disp_among_rate, disp_kernal_matrix, axis):
        ''' normalize the elements (only for i≠j) in the disp_kernal_matrix 
        such that the total dispersal rate from all other patches to patchj
        equals total_disp_among_rate.
        normalization such that sum of column vector is 1. 
        column vector of disp_rate_matrix is the dispersal rate leaving the patch. '''
        
        np.fill_diagonal(disp_kernal_matrix, 0)
        disp_rate_matrix = total_disp_among_rate * disp_kernal_matrix/disp_kernal_matrix.sum(axis=0)
        np.fill_diagonal(disp_rate_matrix, 1-total_disp_among_rate)
        
        if axis == 0:
            return disp_rate_matrix
        if axis == 1:
            return disp_rate_matrix.T
        
    def emigrant_disp_rate_matrix(self, total_disp_among_rate, disp_kernal, graph_object):
        ''' return emigrant_dispersal_rate_matrix.
        the elements D_ij (row_i, col_j) in the matric means the probability that 
        the emigrants to patch j of the all emigrants (offspings) from patch i.
        the row vector of the matrix is idendity vector. '''
        #dis_matrix = nx.adjacency_matrix(graph_object).todense()
        short_path_dis_matrix = nx.floyd_warshall_numpy(graph_object)[:self.patch_num, :self.patch_num]
        disp_kernal_matrix = self.dist2disp_function(disp_kernal, short_path_dis_matrix)
        disp_rate_matrix = self.normalize_dispersal_among_patches_matrix(total_disp_among_rate, disp_kernal_matrix, axis=1)
        #disp_rate_matrix = disp_kernal_matrix/disp_kernal_matrix.sum(axis=1)
        return disp_rate_matrix
    
    def emigrant_matrix_from_offsprings_pool(self, total_disp_among_rate, disp_kernal, graph_object):
        ''' 
        patch_offs_num_matrix is a diagonal matrix, the nonzero elements in which represent the num of offspring in each patch
        emigrant_disp_rate_matrix is emigrant_dispersal_rate_matrix.
        emigrant_matrix = patch_offs_num_matrix * emigrant_disp_rate_matrix
        the element(i,j) in the result matrix means, of all the emigrants patch i can provided, the nums of emigrants offsprings from patch i to patch j
        sum of row vector (i) is the num of all the offsprings in patch i
        '''
        patch_offs_num_matrix = np.mat(np.zeros((self.patch_num, self.patch_num)))
        index = 0
        for patch_id in self.meta_map.nodes():
            patch_object = self.set[patch_id]
            patch_off_num = patch_object.patch_offsprings_num()
            patch_offs_num_matrix[index, index] = patch_off_num
            index+=1
        return self.mat_around(patch_offs_num_matrix * self.emigrant_disp_rate_matrix(total_disp_among_rate, disp_kernal, graph_object))
    
    def emigrant_matrix_expectation_asexual(self, total_disp_among_rate, disp_kernal, graph_object):
        ''''''
        patch_offs_num_matrix = np.mat(np.zeros((self.patch_num, self.patch_num)))
        index = 0
        for patch_id in self.meta_map.nodes():
            patch_object = self.set[patch_id]
            patch_off_num = patch_object.get_patch_individual_num() * patch_object.asexual_birth_rate
            patch_offs_num_matrix[index, index] = patch_off_num
            index+=1
        return self.mat_around(patch_offs_num_matrix * self.emigrant_disp_rate_matrix(total_disp_among_rate, disp_kernal, graph_object))
    
    def emigrant_matrix_expectation_sexual(self, total_disp_among_rate, disp_kernal, graph_object):
        ''''''
        patch_offs_num_matrix = np.mat(np.zeros((self.patch_num, self.patch_num)))
        index = 0
        for patch_id in self.meta_map.nodes():
            patch_object = self.set[patch_id]
            patch_off_num = patch_object.get_patch_sexual_pairwise_parents_num() * patch_object.sexual_birth_rate
            patch_offs_num_matrix[index, index] = patch_off_num
            index+=1
        return self.mat_around(patch_offs_num_matrix * self.emigrant_disp_rate_matrix(total_disp_among_rate, disp_kernal, graph_object))
    
    def emigrant_matrix_expectation_mixed(self, total_disp_among_rate, disp_kernal, graph_object):
        ''''''
        patch_offs_num_matrix = np.mat(np.zeros((self.patch_num, self.patch_num)))
        index = 0
        for patch_id in self.meta_map.nodes():
            patch_object = self.set[patch_id]
            patch_off_num = patch_object.get_patch_mixed_asexual_parent_num() * patch_object.asexual_birth_rate + patch_object.get_patch_mixed_sexual_pairwise_parents_num() * patch_object.sexual_birth_rate
            logging.info(patch_id + 'asexual_parent_num='+ str(patch_object.get_patch_mixed_asexual_parent_num()) + '; sexual_parents_num=' + str(patch_object.get_patch_mixed_sexual_pairwise_parents_num()))
            
            patch_offs_num_matrix[index, index] = patch_off_num
            index += 1
        #logging.info('patch_offs_num_matrix=')
        #logging.info(patch_offs_num_matrix)
        return self.mat_around(patch_offs_num_matrix * self.emigrant_disp_rate_matrix(total_disp_among_rate, disp_kernal, graph_object))
    
    def immigrant_matrix_to_patch_empty_sites(self, emigrants_matrix):
        '''
        patch_empty_sites_num_matrix is a diagonal matrix, the nonzero elements in which represent the num of empty sites in each patch
        imigrant_dispersal_rate_matrix is imigrant_dispersal_rate_matrix
        immigrant_matrix = imigrant_dispersal_rate_matrix * patch_empty_sites_num_matrix
        the element(i,j) in the result matrix means, of all the empty sites in patch j, the num of immigarnts from patch i
        sum of column vector (j) is the num of all the empty site in patch j 
        '''
        patch_empty_sites_num_matrix = np.mat(np.zeros((self.patch_num, self.patch_num)))
        index = 0
        for patch_id in self.meta_map.nodes():
            patch_object = self.set[patch_id]
            patch_empty_sites_num = patch_object.patch_empty_sites_num()
            patch_empty_sites_num_matrix[index, index] = patch_empty_sites_num
            index += 1
        #logging.info('EM/In,t = \n')
        #logging.info(emigrants_matrix/emigrants_matrix.sum(axis=0))
        #logging.info('patch_empty_sites_num_matrix=')
        #logging.info(patch_empty_sites_num_matrix)
        immigrant_matrix = self.mat_around(emigrants_matrix/emigrants_matrix.sum(axis=0) * patch_empty_sites_num_matrix)
        # may warning that 'RuntimeWarning: invalid value encountered in true_divide' if 0/0=np.nan occurs in the early time_step
        immigrant_matrix[np.isnan(immigrant_matrix)] = 0
        # we replaced np.nan into 0 to fixed the problem aboved.
        return immigrant_matrix
    
    def meta_initialize(self, traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls):
        for patch_id, patch_object in self.set.items():
            patch_object.patch_initialize(traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, reproduce_mode, species_2_phenotype_ls)
        return 0
    
    def get_meta_empty_sites_ls(self):
        ''' return meta_empty_sites_ls as [(patch_id, h_id, len_id, wid_id)] '''   
        meta_empty_sites_ls = []
        for patch_id, patch_object in self.set.items():
            patch_empty_pos_ls = patch_object.get_patch_empty_sites_ls()
            for empty_pos in patch_empty_pos_ls:
                empty_pos = (patch_id, ) + empty_pos
                meta_empty_sites_ls.append(empty_pos)
        return meta_empty_sites_ls
    
    def get_meta_pairwise_empty_sites_ls(self):
        ''' return meta_empty_sites_ls as [((patch_id, h_id, len_id, wid_id), (patch_id, h_id, len_id, wid_id))...] '''   
        meta_pairwise_empty_sites_ls = []
        for patch_id, patch_object in self.set.items():
            patch_pairwise_empty_pos_ls = patch_object.get_patch_pairwise_empty_sites_ls()
            for (empty_site_1_pos, empty_site_2_pos) in patch_pairwise_empty_pos_ls:
                empty_site_1_pos = (patch_id, ) + empty_site_1_pos
                empty_site_2_pos = (patch_id, ) + empty_site_2_pos
                meta_pairwise_empty_sites_ls.append((empty_site_1_pos, empty_site_2_pos))
        return meta_pairwise_empty_sites_ls
        
    def show_meta_empty_sites_num(self):
        return len(self.get_meta_empty_sites_ls())
    
    def meta_mixed_asex_and_sex_parents_num(self):
        ''''''
        asex_num = 0
        sex_num = 0
        for patch_id, patch_object in self.set.items():
            asex_num += patch_object.get_patch_mixed_asexual_parent_num()
            sex_num += patch_object.get_patch_mixed_sexual_pairwise_parents_num()*2
        log_info = 'there are %d asexual parents in the metacommunity; there are %d sexual parents in the metacommunity'%(asex_num, sex_num)
        #print(log_info)
        return log_info
    
    def meta_dead_selection(self, base_dead_rate, fitness_wid):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_dead_selection(base_dead_rate, fitness_wid)
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals dead in selection; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
    def meta_asex_reproduce_mutate(self, mutation_rate, pheno_var_ls):
        for patch_id, patch_object in self.set.items():
            patch_object.patch_asex_reproduce_mutate(mutation_rate, pheno_var_ls)
        return 0
    
    def meta_sex_reproduce_mutate(self, mutation_rate, pheno_var_ls):
        for patch_id, patch_object in self.set.items():
            patch_object.patch_sex_reproduce_mutate(mutation_rate, pheno_var_ls)
        return 0
    
    def meta_disp_among_patches_from_offsprings_pool(self, total_disp_among_rate, disp_kernal, graph_object):
        ''' dispersal from patch_i to patch j'''
        emigrants_matrix = self.emigrant_matrix_from_offsprings_pool(total_disp_among_rate, disp_kernal, graph_object)
        immigrants_matrix = self.immigrant_matrix_to_patch_empty_sites(emigrants_matrix)
        migrants_matrix = np.minimum(emigrants_matrix, immigrants_matrix)
        counter = 0
        # dispersal from patch i to patch j
        for j in range(len(self.meta_map.nodes())):
            patch_j_id = list(self.meta_map.nodes())[j]
            patch_j_object = self.set[patch_j_id]
            patch_j_empty_site_ls = patch_j_object.get_patch_empty_sites_ls()
            migrants_indi_object_ls = []
            
            for i in range(len(self.meta_map.nodes())):
                patch_i_id = list(self.meta_map.nodes())[i]
                patch_i_object = self.set[patch_i_id]
                patch_i_offspring_pool = patch_i_object.get_patch_offsprings_pool()
            
                if i==j: 
                    continue
                else:
                    migrants_num = int(migrants_matrix[i, j])
                    migrants_indi_object_ls += random.sample(patch_i_offspring_pool, migrants_num)
                    
            random.shuffle(patch_j_empty_site_ls)
            random.shuffle(migrants_indi_object_ls)
            
            for (h_id, len_id, wid_id), migrants_object in list(zip(patch_j_empty_site_ls, migrants_indi_object_ls)):
                self.set[patch_j_id].set[h_id].add_individual(indi_object = migrants_object, len_id=len_id, wid_id=wid_id)
                #print(counter, patch_j_id, h_id, len_id, wid_id)  
                counter += 1
                
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        print('there are %d individuals disperse among patches; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num))
        return 0
    
    def meta_disp_among_patches_from_dormancy_pool(self, disp_kernal, graph_object):
        pass
    
    def meta_disp_among_patches_from_offsprings_and_dormancy_pool(self, disp_kernal, graph_object):
        pass
    
    def meta_disp_within_patch_from_offsprings_pool(self, disp_within_rate):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter = patch_object.patch_disp_within_from_offsprings_pool(disp_within_rate, counter)
            
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        print('there are %d individuals disperse within patch; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num))
        return 0
    
    def meta_disp_within_patches_from_dormancy_pool(self, disp_kernal, graph_object):
        pass
    
    def meta_disp_within_patches_from_offsprings_and_dormancy_pool(self, disp_kernal, graph_object):
        pass
    
    def meta_asexual_reproduce_mutate_and_dispersal_among_patches(self, mutation_rate, pheno_var_ls, total_disp_among_rate, disp_kernal, graph_object):
        ''' dispersal from patch_i to patch j'''
        emigrants_matrix = self.emigrant_matrix_expectation_asexual(total_disp_among_rate, disp_kernal, graph_object)
        immigrants_matrix = self.immigrant_matrix_to_patch_empty_sites(emigrants_matrix)
        migrants_matrix = np.minimum(emigrants_matrix, immigrants_matrix)
        self.disp_current_matrix += migrants_matrix
        counter = 0
        for j in range(len(self.meta_map.nodes())):
            patch_j_id = list(self.meta_map.nodes())[j]
            patch_j_object = self.set[patch_j_id]
            patch_j_empty_site_ls = patch_j_object.get_patch_empty_sites_ls()
            migrants_indi_object_ls = []
            
            for i in range(len(self.meta_map.nodes())):
                patch_i_id = list(self.meta_map.nodes())[i]
                patch_i_object = self.set[patch_i_id]
                if i==j: 
                    continue
                else:
                    migrants_num = int(migrants_matrix[i, j])
                    patch_i_offspring_pool = patch_i_object.asex_reproduce_mutate_for_dispersal_among_patches(mutation_rate=mutation_rate, pheno_var_ls=pheno_var_ls, patch_offs_num=migrants_num)
                    migrants_indi_object_ls += patch_i_offspring_pool
                    #print('%d-%d=%d'%(migrants_num, len(patch_i_offspring_pool), migrants_num-len(patch_i_offspring_pool)))
            random.shuffle(patch_j_empty_site_ls)
            random.shuffle(migrants_indi_object_ls)
            
            for (h_id, len_id, wid_id), migrants_object in list(zip(patch_j_empty_site_ls, migrants_indi_object_ls)):
                self.set[patch_j_id].set[h_id].add_individual(indi_object = migrants_object, len_id=len_id, wid_id=wid_id)
                #print(counter, patch_j_id, h_id, len_id, wid_id)  
                counter += 1
                
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals disperse among patches; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num)
        return log_info
    
    def meta_sexual_reproduce_mutate_and_dispersal_among_patches(self, mutation_rate, pheno_var_ls, total_disp_among_rate, disp_kernal, graph_object):
        emigrants_matrix =  self.emigrant_matrix_expectation_sexual(total_disp_among_rate, disp_kernal, graph_object)
        immigrants_matrix = self.immigrant_matrix_to_patch_empty_sites(emigrants_matrix)
        migrants_matrix = np.minimum(emigrants_matrix, immigrants_matrix)
        self.disp_current_matrix += migrants_matrix
        counter = 0
        for j in range(len(self.meta_map.nodes())):
            patch_j_id = list(self.meta_map.nodes())[j]
            patch_j_object = self.set[patch_j_id]
            patch_j_empty_site_ls = patch_j_object.get_patch_empty_sites_ls()
            migrants_indi_object_ls = []
            
            for i in range(len(self.meta_map.nodes())):
                patch_i_id = list(self.meta_map.nodes())[i]
                patch_i_object = self.set[patch_i_id]
                if i==j: 
                    continue
                else:
                    migrants_num = int(migrants_matrix[i, j])
                    patch_i_offspring_pool = patch_i_object.sex_reproduce_mutate_for_dispersal_among_patches(mutation_rate=mutation_rate, pheno_var_ls=pheno_var_ls, patch_offs_num=migrants_num)
                    migrants_indi_object_ls += patch_i_offspring_pool
                    #print('%d-%d=%d'%(migrants_num, len(patch_i_offspring_pool), migrants_num-len(patch_i_offspring_pool)))
            random.shuffle(patch_j_empty_site_ls)
            random.shuffle(migrants_indi_object_ls)
            
            for (h_id, len_id, wid_id), migrants_object in list(zip(patch_j_empty_site_ls, migrants_indi_object_ls)):
                self.set[patch_j_id].set[h_id].add_individual(indi_object = migrants_object, len_id=len_id, wid_id=wid_id) 
                counter += 1
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals disperse among patches; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num)
        return log_info
    
    def meta_mixed_reproduce_mutate_and_dispersal_among_patches(self, mutation_rate, pheno_var_ls, total_disp_among_rate, disp_kernal, graph_object):
        emigrants_matrix = self.emigrant_matrix_expectation_mixed(total_disp_among_rate, disp_kernal, graph_object)
        immigrants_matrix = self.immigrant_matrix_to_patch_empty_sites(emigrants_matrix)
        migrants_matrix = np.minimum(emigrants_matrix, immigrants_matrix)
        self.disp_current_matrix += migrants_matrix
        
        counter = 0
        for j in range(len(self.meta_map.nodes())):
            patch_j_id = list(self.meta_map.nodes())[j]
            patch_j_object = self.set[patch_j_id]
            patch_j_empty_site_ls = patch_j_object.get_patch_empty_sites_ls()
            migrants_indi_object_ls = []
            
            for i in range(len(self.meta_map.nodes())):
                patch_i_id = list(self.meta_map.nodes())[i]
                patch_i_object = self.set[patch_i_id]
                if i==j: 
                    continue
                else:
                    migrants_num = int(migrants_matrix[i, j])
                    patch_i_offspring_pool = patch_i_object.mixed_reproduce_mutate_for_dispersal_among_patches(mutation_rate=mutation_rate, pheno_var_ls=pheno_var_ls, patch_offs_num=migrants_num)
                    migrants_indi_object_ls += patch_i_offspring_pool
                    #print('%d-%d=%d'%(migrants_num, len(patch_i_offspring_pool), migrants_num-len(patch_i_offspring_pool)))
            random.shuffle(patch_j_empty_site_ls)
            random.shuffle(migrants_indi_object_ls)

            for (h_id, len_id, wid_id), migrants_object in list(zip(patch_j_empty_site_ls, migrants_indi_object_ls)):
                self.set[patch_j_id].set[h_id].add_individual(indi_object = migrants_object, len_id=len_id, wid_id=wid_id) 
                counter += 1
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals disperse among patches; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info

    def meta_asexual_birth_disp_within_patches(self, mutation_rate, pheno_var_ls, disp_within_rate):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter = patch_object.asex_reproduce_mutate_for_dispersal_within_patch(mutation_rate, pheno_var_ls, disp_within_rate, counter)
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals disperse within patches; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
    def meta_sexual_birth_disp_within_patches(self, mutation_rate, pheno_var_ls, disp_within_rate):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.sex_reproduce_mutate_for_dispersal_within_patch(mutation_rate, pheno_var_ls, disp_within_rate)
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals disperse within patches; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
    def meta_mixed_birth_disp_within_patches(self, mutation_rate, pheno_var_ls, disp_within_rate):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.mixed_reproduce_mutate_for_dispersal_within_patch(mutation_rate, pheno_var_ls, disp_within_rate)
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals disperse within patches; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
            
    def meta_germinate_from_offsprings_pool(self):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_germinate_from_offsprings_pool()
            
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        print('there are %d individuals germinating from local offsprings pool; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num))
        return 0
    
    def meta_asexual_birth_mutate_germinate(self, mutation_rate, pheno_var_ls):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_asexual_birth_germinate(mutation_rate, pheno_var_ls)
        
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals germinating from local habitat; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num)
        return log_info
    
    def meta_sexual_birth_mutate_germinate(self, mutation_rate, pheno_var_ls):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_sexual_birth_germinate(mutation_rate, pheno_var_ls)
            
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals germinating from local habitat; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
    def meta_mixed_birth_mutate_germinate(self, mutation_rate, pheno_var_ls):
        counter = 0
        for patch_id, patch_object in self.set.items():
            counter += patch_object.patch_mixed_birth_germinate(mutation_rate, pheno_var_ls)
        
        indi_num = self.get_meta_individual_num()
        empty_sites_num = self.show_meta_empty_sites_num()
        log_info = 'there are %d individuals germinating from local habitat; there are %d individuals in the metacommunity; there are %d empty sites in the metacommunity'%(counter, indi_num, empty_sites_num)
        #print(log_info)
        return log_info
    
####################################################################################################################################################
class species_pool():
    def __init__(self, species_num, standar_species_ls):
        self.species_num = species_num
        self.standar_species_ls = standar_species_ls
    
    def generate_propagules_rain_ls(self, num, reproduce_mode):
        propagules_rain_ls = []
        for i in range(num):
            if reproduce_mode == 'asexual': gender = 'female'
            if reproduce_mode == 'sexual': gender = random.sample(('male', 'female'), 1)[0]
        
            standar_species_object = random.sample(self.standar_species_ls, 1)[0]
            individual_object = individual(species_id=standar_species_object.species_id, traits_num=standar_species_object.traits_num, pheno_names_ls=standar_species_object.pheno_names_ls, gender=gender)
            individual_object.random_init_indi(mean_pheno_val_ls=standar_species_object.mean_pheno_val_ls, pheno_var_ls=standar_species_object.pheno_var_ls, geno_len_ls=standar_species_object.geno_len_ls)

            propagules_rain_ls.append(individual_object)
        return propagules_rain_ls
    
    def generate_pairwise_sexual_propagules_rain_ls(self, pairwise_num):
        pairwise_propagules_rain_ls = []
        for i in range(pairwise_num):
            standar_species_object = random.sample(self.standar_species_ls, 1)[0]
            
            female_individual_obj = individual(species_id=standar_species_object.species_id, traits_num=standar_species_object.traits_num, pheno_names_ls=standar_species_object.pheno_names_ls, gender='female')
            female_individual_obj.random_init_indi(mean_pheno_val_ls=standar_species_object.mean_pheno_val_ls, pheno_var_ls=standar_species_object.pheno_var_ls, geno_len_ls=standar_species_object.geno_len_ls)
            
            male_individual_obj = individual(species_id=standar_species_object.species_id, traits_num=standar_species_object.traits_num, pheno_names_ls=standar_species_object.pheno_names_ls, gender='male')
            male_individual_obj.random_init_indi(mean_pheno_val_ls=standar_species_object.mean_pheno_val_ls, pheno_var_ls=standar_species_object.pheno_var_ls, geno_len_ls=standar_species_object.geno_len_ls)
            
            pairwise_individuals_object = (female_individual_obj, male_individual_obj)
            pairwise_propagules_rain_ls.append(pairwise_individuals_object)
        return pairwise_propagules_rain_ls
            
####################################################################################################################################################    
class species():
    def __init__(self, species_id, traits_num, pheno_names_ls, mean_pheno_val_ls, pheno_var_ls, geno_len_ls):

        self.species_id = species_id
        self.traits_num = traits_num
        self.pheno_names_ls = pheno_names_ls
        self.mean_pheno_val_ls = mean_pheno_val_ls
        self.pheno_var_ls = pheno_var_ls
        self.geno_len_ls = geno_len_ls
        
class individual():
    def __init__(self, species_id, traits_num, pheno_names_ls, gender='female', genotype_set=None, phenotype_set=None):
        self.species_id = species_id
        self.gender = gender
        self.traits_num = traits_num
        self.pheno_names_ls = pheno_names_ls
        self.genotype_set = genotype_set
        self.phenotype_set = phenotype_set
        
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
            
###################################################################################################
def generating_empty_metacommunity(meta_name, patch_num, patch_location_ls, asexual_birth_rate, sexual_birth_rate, hab_num, hab_length, hab_width, 
                                   micro_environment_values_ls, macro_environment_values_ls, environment_types_num, environment_types_name, environment_variation_ls):
    ''' '''
    meta_object = metacommunity(metacommunity_name=meta_name)
    log_info = ''
    for i in range(0, patch_num):
        patch_name = 'patch%d'%(i+1)
        patch_index = i
        location = patch_location_ls[i]
        p = patch(patch_name, patch_index, location, asexual_birth_rate, sexual_birth_rate)
        
        micro_environment_means_values_ls = micro_environment_values_ls
        macro_environment_means_value = macro_environment_values_ls[int(location[1])]
        
        for j in range(hab_num):
            habitat_name = 'h%s'%str(j+1)
            micro_environment_mean_value = micro_environment_means_values_ls[j]
            p.add_habitat(hab_name=habitat_name, num_env_types=environment_types_num, env_types_name=environment_types_name, 
                          mean_env_ls=[micro_environment_mean_value, macro_environment_means_value], var_env_ls=environment_variation_ls, length=hab_length, width=hab_width)
            log_info += '%s, %s, %s: micro_environment_mean_value=%s, macro_environment_means_value=%s \n'%(patch_name, str(location), habitat_name, str(micro_environment_mean_value), str(macro_environment_means_value))
            
        meta_object.add_patch(patch_name=patch_name, patch_object=p)
    return meta_object, log_info

def generating_mainland_species_pool(species_num, traits_num, pheno_names_ls, pheno_var_ls, geno_len_ls, species_2_phenotype_ls):
    standar_species_object_ls = [species(species_id='sp%d'%(i+1), traits_num=traits_num, pheno_names_ls=pheno_names_ls, mean_pheno_val_ls=(species_2_phenotype_ls[i]), pheno_var_ls=pheno_var_ls, geno_len_ls=geno_len_ls) for i in range(species_num)]
    mainland_object = species_pool(species_num=species_num, standar_species_ls=standar_species_object_ls)
    
    log_info = 'species_num=%d \n'%species_num
    for sp_object in standar_species_object_ls:
        log_info += '%s,  traits_num=%d,  %s=%s,  phenotypes_var=%s,  genotypes_len=%s \n'%(sp_object.species_id, sp_object.traits_num, str(sp_object.pheno_names_ls), str(sp_object.mean_pheno_val_ls), str(sp_object.pheno_var_ls), str(sp_object.geno_len_ls))
    
    return mainland_object, log_info

def calculate_graph_object(meta_object, graph_algorithm, metacommunity_x_range=None, metacommunity_y_range=None):
    ''' '''
    if graph_algorithm == 'isolation':
        empty = meta_object.empty_graph()
        nx.write_gpickle(empty, "isolation.gpickle")
        meta_object.show_meta_map(empty, title = 'isolated patches')
        return empty
    
    elif graph_algorithm == 'full_connection':
        full = meta_object.full_con_map()
        nx.write_gpickle(full, "full_connection.gpickle")
        meta_object.show_meta_map(full, title = 'full connection network')
        return full
    
    elif graph_algorithm == 'minimum_spanning_tree':
        mini = meta_object.mini_span_tree() 
        nx.write_gpickle(mini, "minimum_spanning_tree.gpickle")
        meta_object.show_meta_map(mini, title = 'minimum spanning tree')
        return mini
    
    elif graph_algorithm == 'travelling salesman problem':
        tsp = meta_object.dyn_pro_traveling_salesman_network()
        nx.write_gpickle(tsp, "travelling salesman problem.gpickle")
        meta_object.show_meta_map(tsp, title = 'travelling salesman network') 
        return tsp
    
    elif graph_algorithm == 'paul_revere':
        paul_revere = meta_object.paul_revere_network()
        nx.write_gpickle(paul_revere, "paul_revere.gpickle")
        meta_object.show_meta_map(paul_revere, title = 'paul revere network')
        return paul_revere
    
    elif graph_algorithm == 'one_center_network':
        one_center = meta_object.one_center_network()
        nx.write_gpickle(one_center, "one_center_network.gpickle")
        meta_object.show_meta_map(one_center, title = 'one center network')
        return one_center
    
    elif graph_algorithm == 'hierachical_network':
        hierarchy = meta_object.central_hierachical_network()
        nx.write_gpickle(hierarchy, "hierachical_network.gpickle")
        meta_object.show_meta_map(hierarchy, title = 'hierachical network')
        return hierarchy
    
    elif graph_algorithm == 'steiner_tree':
        steiner, pos_steiner = meta_object.steiner_tree(metacommunity_x_range, metacommunity_y_range)
        nx.write_gpickle(steiner, "steiner_tree.gpickle")
        meta_object.show_meta_map(steiner, title = 'steiner tree', pos = pos_steiner)
        return steiner
    
    elif graph_algorithm == 'regular_network':
        regular_network = meta_object.k_factor_regular_random_network()
        nx.write_gpickle(regular_network, "regular_network.gpickle")
        meta_object.show_meta_map(regular_network, title = 'regular_network')
        return regular_network
    
    elif graph_algorithm == 'small_world_network':
        small_world = meta_object.small_world_random_graph()
        nx.write_gpickle(small_world, "small_world_network.gpickle")
        meta_object.show_meta_map(small_world, title = 'small_world_network')
        return small_world
    
    else:
        raise ValueError('graph_algorithm inputed is no found.')

def read_graph_object_gpickle(meta_object, path):
    ''' '''
    graph_object = nx.read_gpickle(path)
    meta_object.show_meta_map(graph_object, title = str(path), pos = nx.get_node_attributes(graph_object, 'position'))
    return graph_object

def main():
    repeat_times =1
    all_time_step = 10000
    
    patch_num = 16
    metacommunity_x_range = range(0,10)
    metacommunity_y_range = range(0,10)
    patch_location_ls = [(0, 2), (2, 2), (2, 8), (4, 0), (8, 2), (7, 0), (0, 0), (1, 4), (6, 7), (4, 4), (5, 9), (1, 5), (8, 5), (6, 9), (9, 0), (3, 6)]
    
    hab_num_in_patch = 10
    hab_length, hab_width = 10, 10
    environment_types_num = 2
    environment_types_name=('micro_environment', 'macro_environment')
    environment_variation_ls = [0.025, 0.025]
    micro_environment_values_ls = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    macro_environment_values_ls = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    total_increment_value = 0.1 # global warming parameter
    per_time_step_increment_value = 0.0001 # global warming rate: 0.0001, 0.001 or 0.01
    warming_start_time_step_index = 5000
    
    base_dead_rate=0.1
    fitness_wid=0.5
    
    asexual_birth_rate = 0.5
    sexual_birth_rate = 1
    mutation_rate=0.00001
    
    propagules_rain_num = 10
    total_disp_among_rate = 0.01
    disp_kernal =2
    disp_within_rate =0.11
    
    species_num = 100
    traits_num = 2
    pheno_names_ls = ('micro_phenotype', 'macro_phenotype')
    pheno_var_ls=(0.025, 0.025)
    geno_len_ls=(20, 20)
    species_2_phenotype_ls = [(j/10, i/10) for i in range(10) for j in range(10)] # (index+1) indicates species_id
    
    logging.basicConfig(filename='model_logging.log', format='%(asctime)s %(message)s', level=logging.INFO)
    empty_metacommunity, log_info = generating_empty_metacommunity(meta_name='empty_metacommunity', patch_num=patch_num, patch_location_ls=patch_location_ls, asexual_birth_rate=asexual_birth_rate, sexual_birth_rate=sexual_birth_rate, 
                                                    hab_num=hab_num_in_patch, hab_length=hab_length, hab_width=hab_width, micro_environment_values_ls=micro_environment_values_ls, macro_environment_values_ls=macro_environment_values_ls, 
                                                    environment_types_num=environment_types_num, environment_types_name=environment_types_name, environment_variation_ls=environment_variation_ls)
    logging.info('The empty metacommunity has been generated! \n' + log_info)
    
    mainland, log_info = generating_mainland_species_pool(species_num=species_num, traits_num=traits_num, pheno_names_ls=pheno_names_ls, pheno_var_ls=pheno_var_ls, geno_len_ls=geno_len_ls, species_2_phenotype_ls=species_2_phenotype_ls)
    logging.info('The mainland species pool has been generated! \n' + log_info)
    
    small_world = calculate_graph_object(meta_object=empty_metacommunity, graph_algorithm='small_world_network')
    small_world_read = read_graph_object_gpickle(empty_metacommunity, 'small_world_network.gpickle')
    
    all_time_start = time.time()
    for rep in range(repeat_times):
        meta = copy.deepcopy(empty_metacommunity)
        
        #meta.meta_initialize(traits_num=2, pheno_names_ls=('micro_phenotype', 'macro_phenotype'), pheno_var_ls=(0.025, 0.025), geno_len_ls=(20, 20), reproduce_mode='asexual', species_2_phenotype_ls=species_2_phenotype_ls)
        #meta.meta_initialize(traits_num=2, pheno_names_ls=('micro_phenotype', 'macro_phenotype'), pheno_var_ls=(0.025, 0.025), geno_len_ls=(20, 20), reproduce_mode='sexual', species_2_phenotype_ls=species_2_phenotype_ls)
        
        meta_sp_dis_all_time = meta.get_meta_microsites_optimum_sp_id_val(base_dead_rate, fitness_wid, species_2_phenotype_ls)
        meta_micro_phenotype_all_time = meta.get_meta_microsite_environment_values(environment_name='micro_environment')
        meta_macro_phenotype_all_time = meta.get_meta_microsite_environment_values(environment_name='macro_environment')
        
        meta_optimum_sp_id_val_global_warming_time = meta.get_meta_microsites_optimum_sp_id_val(base_dead_rate, fitness_wid, species_2_phenotype_ls)
        meta_micro_environment_global_warming_time = meta.get_meta_microsite_environment_values(environment_name='micro_environment')
        meta_macro_environment_global_warming_time = meta.get_meta_microsite_environment_values(environment_name='macro_environment')
        
        starttime = time.time()
        for time_step in range(all_time_step):
            d1 = time.time()
            if time_step%100==0: print('rep=%d, time_step%d'%(rep, time_step))
            logging.info('rep=%d, time_step=%d'%(rep, time_step))
        
            if warming_start_time_step_index <= time_step <= (warming_start_time_step_index+int(total_increment_value/per_time_step_increment_value)-1):
                meta.global_warming(env_name='macro_environment', increment_value=per_time_step_increment_value)
                logging.info('global_waring_process: mean_value of macro_environment increase %f'%(per_time_step_increment_value))
                
                meta_optimum_sp_id_val_global_warming_time = np.vstack((meta_optimum_sp_id_val_global_warming_time, meta.get_meta_microsites_optimum_sp_id_val(base_dead_rate, fitness_wid, species_2_phenotype_ls)))
                meta_micro_environment_global_warming_time = np.vstack((meta_micro_environment_global_warming_time, meta.get_meta_microsite_environment_values(environment_name='micro_environment')))
                meta_macro_environment_global_warming_time = np.vstack((meta_macro_environment_global_warming_time, meta.get_meta_microsite_environment_values(environment_name='macro_environment')))
            '''    
            # only asexual reproduction
            log_info = meta.show_meta_individual_num()
            logging.info(log_info)
            log_info = meta.meta_dead_selection(base_dead_rate=base_dead_rate, fitness_wid=fitness_wid)
            logging.info('Dead selection process done! \n' + log_info)
            log_info = meta.colonize_from_propagules_rains(species_pool_obj=mainland, reproduce_mode='asexual', propagules_rain_num=propagules_rain_num)
            logging.info('Colonizing process under propagules rains done! \n' + log_info)
            log_info = meta.meta_asexual_reproduce_mutate_and_dispersal_among_patches(mutation_rate=mutation_rate, pheno_var_ls=pheno_var_ls, total_disp_among_rate=total_disp_among_rate, disp_kernal=disp_kernal, graph_object=small_world)
            logging.info('Dispersal among patches process done! \n' + log_info)
            log_info = meta.meta_asexual_birth_disp_within_patches(mutation_rate=mutation_rate, pheno_var_ls=pheno_var_ls, disp_within_rate=disp_within_rate)
            logging.info('Dispersal within patch process done! \n' + log_info)
            log_info = meta.meta_asexual_birth_mutate_germinate(mutation_rate=mutation_rate, pheno_var_ls=pheno_var_ls)
            logging.info('Local birth and germination process done! \n' + log_info)
            '''
        
            '''
            # only sexual reproduction
            log_info =  meta.show_meta_individual_num()
            logging.info(log_info)
            log_info = meta.meta_dead_selection(base_dead_rate=0.1, fitness_wid=0.5)
            logging.info('Dead selection process done! \n' + log_info)
            log_info = meta.colonize_from_propagules_rains(species_pool_obj=mainland, reproduce_mode='sexual', propagules_rain_num=100)
            logging.info('Colonizing process under propagules rains done! \n' + log_info)
            log_info = meta.meta_sexual_reproduce_mutate_and_dispersal_among_patches(mutation_rate=0.00001, pheno_var_ls=(0.025, 0.025), total_disp_among_rate=0.1, disp_kernal=1, graph_object=full)
            logging.info('Dispersal among patches process done! \n' + log_info)
            log_info = meta.meta_sexual_birth_disp_within_patches(mutation_rate=0.00001, pheno_var_ls=(0.025, 0.025), disp_within_rate=0.11)
            logging.info('Dispersal within patch process done! \n' + log_info)
            log_info = meta.meta_sexual_birth_mutate_germinate(mutation_rate=0.00001, pheno_var_ls=(0.025, 0.025))
            logging.info('Local birth and germination process done! \n' + log_info)
            '''
            
            
            # asexual and sexual reproduction are both allowed.
            log_info = meta.show_meta_individual_num()
            logging.info(log_info)
            log_info = meta.meta_dead_selection(base_dead_rate=base_dead_rate, fitness_wid=fitness_wid)
            logging.info('Dead selection process done! \n' + log_info)
            log_info = meta.pairwise_sexual_colonization_from_prpagules_rains(species_pool_obj=mainland, propagules_rain_num=propagules_rain_num)
            logging.info('Colonizing process under propagules rains done! \n' + log_info)
            logging.info(meta.meta_mixed_asex_and_sex_parents_num())
            log_info = meta.meta_mixed_reproduce_mutate_and_dispersal_among_patches(mutation_rate=mutation_rate, pheno_var_ls=pheno_var_ls, total_disp_among_rate=total_disp_among_rate, disp_kernal=disp_kernal, graph_object=small_world)
            logging.info('Dispersal among patches process done! \n' + log_info)
            log_info = meta.meta_mixed_birth_disp_within_patches(mutation_rate=mutation_rate, pheno_var_ls=pheno_var_ls, disp_within_rate=disp_within_rate)
            logging.info('Dispersal within patch process done! \n' + log_info)
            log_info = meta.meta_mixed_birth_mutate_germinate(mutation_rate=mutation_rate, pheno_var_ls=pheno_var_ls)
            logging.info('Local birth and germination process done! \n' + log_info)
            
            
            '''
            # from offsprings pool
            meta.show_meta_individual_num()
            meta.meta_dead_selection(base_dead_rate=0.1, fitness_wid=0.5)
            meta.meta_asex_reproduce_mutate(mutation_rate=0.00001, pheno_var_ls=(0.025, 0.025))
            #meta.meta_sex_reproduce_mutate(mutation_rate=0.00001, pheno_var_ls=(0.025, 0.025))
            meta.meta_disp_among_patches_from_offsprings_pool(total_disp_among_rate=0.1, disp_kernal=2, graph_object=mini)
            meta.meta_disp_within_patch_from_offsprings_pool(disp_within_rate=0.11)
            meta.meta_germinate_from_offsprings_pool()
            '''
            
            meta_sp_dis_all_time = np.vstack((meta_sp_dis_all_time, meta.get_meta_microsites_individuals_sp_id_values()))
            meta_micro_phenotype_all_time = np.vstack((meta_micro_phenotype_all_time, meta.get_meta_microsites_individuals_phenotype_values(trait_name='micro_phenotype')))
            meta_macro_phenotype_all_time = np.vstack((meta_macro_phenotype_all_time, meta.get_meta_microsites_individuals_phenotype_values(trait_name='macro_phenotype')))
            d2 = time.time()
            logging.info("时间步运行时间：%.8s s" % (d2-d1) + '\n') 
            
        endtime = time.time()
        dtime = endtime - starttime
        logging.info("一次模拟运行时间：%.8s s" % dtime) 
        
        meta.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_sp_dis_all_time, first_row_index_name='optimun_sp_id_values', all_time_step=all_time_step, file_name='rep=%d_meta_species_distribution_all_time.gz'%(rep))
        meta.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_micro_phenotype_all_time, first_row_index_name='micro_environment_values', all_time_step=all_time_step, file_name='rep=%d_meta_micro_phenotype_all_time.gz'%(rep))
        meta.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_macro_phenotype_all_time, first_row_index_name='macro_environment_values', all_time_step=all_time_step, file_name='rep=%d_meta_macro_phenotype_all_time.gz'%(rep))
        
        meta.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_optimum_sp_id_val_global_warming_time, first_row_index_name='before_global_warming_optimun_sp', all_time_step=int(total_increment_value/per_time_step_increment_value), file_name='rep=%d_meta_global_warming_time_optimum_species_distribution.gz'%(rep))
        meta.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_micro_environment_global_warming_time, first_row_index_name='before_global_warming_micro_environment_values', all_time_step=int(total_increment_value/per_time_step_increment_value), file_name='rep=%d_meta_global_warming_time_micro_environment.gz'%(rep))
        meta.meta_distribution_data_all_time_to_csv_gz(dis_data_all_time=meta_micro_environment_global_warming_time, first_row_index_name='before_global_warmingmacro_environment_values', all_time_step=int(total_increment_value/per_time_step_increment_value), file_name='rep=%d_meta_global_warming_time_macro_environment.gz'%(rep))

        meta.meta_disp_current_mat_to_csv(file_name='rep=%ddispersal_current_matrix.csv'%(rep))
    
    all_time_end = time.time()
    logging.info("总模拟运行时间：%.8s s" % (all_time_end-all_time_start)) 
    
if __name__ == '__main__':
    main()
    
    























