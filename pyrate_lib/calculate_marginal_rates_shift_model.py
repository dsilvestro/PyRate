#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate marginal rates from pyrate output
Currently only extinction rates

Created on Fri Aug 10 17:28:04 2018

@author: tobias
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import sys
import argparse
from collections import Counter


def get_args():
    parser = argparse.ArgumentParser(
        description="Calculate marginal rates from pyrate output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        required=True,
        default=None,
        help='Provide the path to the pyrate output dir (pyrate_mcmc_logs/)'
    )
    parser.add_argument(
        '--output',
        required=False,
        default=None,
        help='The output directory where marginal rates will be saved'
    )
    parser.add_argument(
        '--max_time',
        required=True,
        type=float,
        default=None,
        help='Set the maximum of the timescale for the dataset (=oldest occurrence), scaled in the same unit as the pyrate input data'
    )
    parser.add_argument(
        '--log_compression',
        type=int,
        default=1,
        help='Log-compression factor. E.g. if "--log_compression 10", only every 10th line of the individual log-files will be processed'
    )
    parser.add_argument(
        '--burnin',
        type=float,
        default=0.1,
        help='The burn-in as a fraction. E.g. "--burnin 0.1" will skip the first 10%% of each logfile as burn-in'
    )
    parser.add_argument(
        '--only_best_model',
        action='store_true',
        default=False,
        help='Use flag if you only want to calculate marginal rates from the best shift-model (evaluated across all replicates)'
    )
    parser.add_argument(
        '--set_shift_model',
        type=int,
        default=None,
        help='Manually set the shift-model that marginal rates should be calculated for, e.g. set to "1" if marginal rates should be calculated for the 1-shift model (overwrites "--only_best_model" flag)'
    )
    return parser.parse_args()


def calcHPD(data, level):
    assert (0 < level < 1)
    d = list(data)
    d.sort()
    nData = len(data)
    nIn = int(round(level * nData))
    if nIn < 2 :
        sys.exit('\n\nERROR!!!: Choose a different shift-model. The chosen model has too little data to calculate marginal rates.')
    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)):
         rk = d[k+nIn-1] - d[k]
         if rk < r :
             r = rk
             i = k
    assert 0 <= i <= i+nIn-1 < len(d)
    return (d[i], d[i+nIn-1])


def calculate_marginal_rates(pyrate_output_dir,outdir,max_time_scale_scaled,nstep,burnin,only_best_model,set_model):
    # get input files
    pattern = '%s/*ex_rates.log'%pyrate_output_dir
    files = glob.glob(pattern)
    labels = [re.split(r'(\d+)', file.split('/')[-1])[0] for file in files]
    
    # dict with label and all files
    file_dict = {}
    for label in labels:
        real_label = label.replace('_ts_te_dates','').replace('_past_future','').replace('_endemics','').lower().capitalize().replace('(',' (').replace('_raw',' (raw)').replace('_','')
        target_files = [file for file in files if file.split('/')[-1].startswith(label)]
        file_dict.setdefault(real_label,target_files)

    # get data
    if not outdir:
        outdir = os.path.join(pyrate_output_dir,'joined_log_files/')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for subset in file_dict.keys():
        print('\n%s'%subset)
        
        if only_best_model:
            shift_model_list = []
            print('Counting shifts in logfiles')
            for file in file_dict[subset]:
                shift_model_temp_list = []
                with open(file) as handle:
                    for lineno, line in enumerate(handle):
                        if lineno % nstep == 0:
                            #print(lineno)
                            mcmc_it = line.strip().split('\t')
                            data_mcmc_it = np.array(mcmc_it).astype(np.float)
                            count_of_shifts = int((len(data_mcmc_it)+1)/2)-1
                            shift_model_temp_list.append(count_of_shifts)
                    removed_burnin = shift_model_temp_list[int((lineno+1)/nstep*burnin):]
                    shift_model_list += removed_burnin
            counts = Counter(shift_model_list)
            models = [key for key, value in counts.items()]
            percentages = [np.round(value/sum([value for key, value in counts.items()]),3) for key, value in counts.items()]
            best_model = models[percentages.index(max(percentages))]

        counter = 0
        add_string = ''
        for file in file_dict[subset]:            
            #percent = int((counter+1)*100/len(file_dict[subset]))
            sys.stdout.write('\rProcessing log-files: %i/%i '%(int(counter+1),len(file_dict[subset])))
            with open(file) as handle:
                data_matrix = np.zeros([int((len(list(open(file)))-1)/nstep)+1,len(np.arange(0,max_time_scale_scaled+1))])
                for lineno, line in enumerate(handle):
                    if lineno % nstep == 0:
                        #print(lineno)
                        mcmc_it = line.strip().split('\t')
                        data_mcmc_it = np.array(mcmc_it).astype(np.float)
                        count_of_rates = int((len(data_mcmc_it)+1)/2)
                        count_of_shifts = count_of_rates-1
                        if only_best_model:
                            add_string = '_best_model'
                            if count_of_shifts == best_model:
                                rates = data_mcmc_it[:count_of_rates]
                                years_of_shift = np.array([max_time_scale_scaled+1] + list(data_mcmc_it[count_of_rates:]))
                                years_array = np.arange(0,max_time_scale_scaled+1)
                                ext_rates_array = np.zeros(len(years_array))
                                for i in np.arange(len(years_of_shift)):
                                    year = int(years_of_shift[i])
                                    ext_rates_array[0:year] = rates[i]
                                data_matrix[int(lineno/nstep),:] = ext_rates_array
                        elif set_model:
                            add_string = '_%i_shift'%set_model
                            if count_of_shifts == set_model:
                                rates = data_mcmc_it[:count_of_rates]
                                years_of_shift = np.array([max_time_scale_scaled+1] + list(data_mcmc_it[count_of_rates:]))
                                years_array = np.arange(0,max_time_scale_scaled+1)
                                ext_rates_array = np.zeros(len(years_array))
                                for i in np.arange(len(years_of_shift)):
                                    year = int(years_of_shift[i])
                                    ext_rates_array[0:year] = rates[i]
                                data_matrix[int(lineno/nstep),:] = ext_rates_array                            
                        else:
                            rates = data_mcmc_it[:count_of_rates]
                            years_of_shift = np.array([max_time_scale_scaled+1] + list(data_mcmc_it[count_of_rates:]))
                            years_array = np.arange(0,max_time_scale_scaled+1)
                            ext_rates_array = np.zeros(len(years_array))
                            for i in np.arange(len(years_of_shift)):
                                year = int(years_of_shift[i])
                                ext_rates_array[0:year] = rates[i]
                            data_matrix[int(lineno/nstep),:] = ext_rates_array
                # check if there is any data in the arrays, otherwise return error
            if counter == 0:
                final_array = data_matrix[np.max([5,int(burnin*len(data_matrix))]):]
            else:
                final_array = np.append(final_array,data_matrix[np.max([5,int(burnin*len(data_matrix))]):],axis=0)
            counter += 1
        try:
            years_array.shape
            print('\nCalculating marginal rates...')
        except:
            sys.exit('\n\nERROR!!!: Choose a different shift-model. The chosen model has no data.')
    
        if only_best_model or set_model:
            # remove all lines with only 0 values from array (those that didn't have the best model)
            final_array = final_array[~np.all(final_array == 0, axis=1)]
        years = years_array
        mean = np.mean(final_array,axis=0)
        confidence_interval = [calcHPD(final_array[:,i],0.95) for i in np.arange(len(years))]
        lower = [item[0] for item in confidence_interval]
        upper = [item[1] for item in confidence_interval]
        a_df = pd.DataFrame(index=np.arange(0,len(years)), columns=['time','mean','95m', '95M'])
        a_df['time'] = years
        a_df['mean'] = mean
        a_df['95m'] = lower
        a_df['95M'] = upper
        a_df.to_csv(os.path.join(outdir,'%s_extinction_rates_through_time_joined_all_%i_replicates_log_compression_factor_%i%s.txt' %(subset.lower().replace(' ','_').replace('(','').replace(')',''),len(file_dict[subset]),nstep,add_string)),index=False,sep='\t')
    return(outdir)

args = get_args()
pyrate_output_dir = args.input
outdir = args.output
max_time_scale_scaled = args.max_time
nstep = args.log_compression
burnin = args.burnin
only_best_model = args.only_best_model
if args.set_shift_model:
    only_best_model = False

outdir = calculate_marginal_rates(pyrate_output_dir,outdir,max_time_scale_scaled,nstep,burnin,only_best_model,args.set_shift_model)
print('\nResults printed to ./%s\n'%outdir)


                    
                    