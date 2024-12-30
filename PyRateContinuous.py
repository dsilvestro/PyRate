#!/usr/bin/env python 
import argparse, os,sys
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import random as rand
import platform, time
import multiprocessing, _thread
import multiprocessing.pool
import csv
from scipy.special import gdtr, gdtrix
from scipy.special import betainc
import scipy.stats
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
from multiprocessing import Pool, freeze_support
import _thread


from pyrate_lib.lib_updates_priors import *
from pyrate_lib.lib_DD_likelihood  import *
from pyrate_lib.lib_utilities import calcHPD as calcHPD
from pyrate_lib.lib_utilities import print_R_vec as print_R_vec
from pyrate_lib.lib_utilities import read_ts_te_table as read_ts_te_table
import pyrate_lib.lib_utilities as lib_utilities

self_path=os.getcwd()

#### ARGS
p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('-A', type=int, help='algorithm: "0" parameter estimation, "1" TI', default=0, metavar=0) # 0: par estimation, 1: TI
p.add_argument('-d', type=str, help='data set', default="", metavar="<file>")
p.add_argument('-c', type=str, help='covariate data set', default="", metavar="<file>")
p.add_argument('-j', type=int, help='replicate', default=0, metavar=0)
p.add_argument('-m', type=int, help='model: "-1" constant rate, "0" exponential, "1" linear, "2" ecological limit, "3" dynamic upper limit', default=0, metavar=0)
p.add_argument('-mSpEx', type=int, help='Speciation/Extinction models: "-1" constant rate, "0" exponential, "1" linear, "2" ecological limit, "3" dynamic upper limit', default=[-np.inf,-np.inf], metavar=-np.inf,nargs=2)
p.add_argument('-equal_G', type=int, help='model: "0" unconstrained G, "1" constrained G', default=0, metavar=0)
p.add_argument('-equal_R', type=int, help='model: "0" unconstrained R, "1" constrained R', default=0, metavar=0)
p.add_argument('-equal_Z', type=int, help='model: "0" unconstrained Z, "1" constrained Z', default=0, metavar=0)
p.add_argument('-n', type=int, help='mcmc generations',default=1050000, metavar=1050000)
p.add_argument('-s', type=int, help='sample freq.', default=1000, metavar=1000)
p.add_argument('-p', type=int, help='print freq.', default=1000, metavar=1000)
p.add_argument('-r', type=float, help='rescale values (0 to scale in [0,1], 0.1 to reduce range 10x, 1 to leave unchanged)', default=0, metavar=0)
p.add_argument('-clade', type=int, help='clade analyzed (set to -1 to analyze all species)', default=-1, metavar=-1)
p.add_argument('-b', type=float, help='burnin in *mcmc.log to generate input file', default=0.1, metavar=0.1)
p.add_argument('-w',  type=float, help='window sizes (bd rates, G)',  default=[1.4, .05], metavar=1.4, nargs=2)
p.add_argument('-ginput', type=str,help='generate input file from *mcmc.log', default="", metavar="<path_to_mcmc.log>")
p.add_argument('-tag', metavar='<*tag*.log>', type=str,help="Tag identifying files to be combined and plotted",default="")
p.add_argument('-mL',  type=str, help='calculate marginal likelihood',  default="", metavar="<path_to_log_files>")
p.add_argument('-stimes',  type=float, help='shift times',  default=[], metavar=0, nargs='+') 
p.add_argument('-slice',  type=float, help='ages of the time slice of interest (23 -> 23-0; 23 2 -> 23-2)',  default=[], metavar=0, nargs="+") 
p.add_argument("-est_start_time",  help='Estimate when the variable starts to have an effect (curve is flattened before that)', action='store_true', default=False)
p.add_argument("-ws_start_time",   type=float, help='Window size update start time', default=1, metavar=1)
p.add_argument('-extract_mcmc', type=int, help='Extract "cold" chain in separate log file', default=1, metavar=1)
p.add_argument("-DD",  help='Diversity Dependent Model', action='store_true', default=False)
p.add_argument('-plot', type=str, help='Log file', default="", metavar="")
p.add_argument("-rescale",   type=float, help='Rescale time axis (e.g. -rescale 1000: 1 -> 1000, time unit = 1Ky)', default=1, metavar=1)
p.add_argument('-use_hp', type=int, help='Use hyperpriors on rates and correlation parameters (0/1)', default=1, metavar=1)
p.add_argument("-pG",   type=float, help='St. dev. of normal prior on correlation parameters (only if -use_hp 0) | use negative values to set symmetric uniform prior', default=1, metavar=1)
p.add_argument("-pZ",   type=float, help='St. dev. of normal prior on power parameters (only if -use_hp 0) | use negative values to set symmetric uniform prior', default=1, metavar=1)
p.add_argument("-verbose",  help='Print curve trajectory', action='store_true', default=False)
p.add_argument("-lag",  help='estimate lagged effect', action='store_true', default=False)




args = p.parse_args()

mcmc_gen = args.n
sampling_freq = args.s
print_freq = args.p
dataset=args.d
cov_file=args.c
rescale_factor=args.r
focus_clade=args.clade
win_size=args.w
rep_j=np.maximum(args.j-1,0) # No np.max here because it ignores the 0 after the comma
est_start_time = args.est_start_time
w_size_start_time = args.ws_start_time
lagged_model = args.lag


s_times=np.sort(np.array(args.stimes))[::-1]
if len(args.slice)>0:
    index_slice_of_interest = 1
    s_times=np.sort(np.array(args.slice))[::-1]
    run_single_slice = 1
else: 
    run_single_slice = 0

equal_g = args.equal_G
equal_r = args.equal_R
equal_z = args.equal_Z

if args.ginput != "":
    lib_utilities.write_ts_te_table(args.ginput, tag=args.tag, clade=focus_clade,burnin=args.b)
    quit()

if args.mL != "":
    lib_utilities.calc_marginal_likelihood(infile=args.mL,burnin=int(args.b),extract_mcmc=args.extract_mcmc)
    quit()

useHP = args.use_hp
ts, te, _, _ = read_ts_te_table(dataset, rep_j, args.rescale, focus_clade=focus_clade)


output_wd = os.path.dirname(dataset)
name_file = os.path.splitext(os.path.basename(dataset))[0]

if np.max(args.mSpEx) > -np.inf:
    args_mSpEx = args.mSpEx
else:
    if args.m== -1: args_mSpEx = [-1,-1]
    if args.m==  0: args_mSpEx = [0,0]
    if args.m==  1: args_mSpEx = [1,1]
    if args.m==  2: args_mSpEx = [2,2]
    if args.m==  3: args_mSpEx = [3,3]
m2 = args_mSpEx[0] == 2 or args_mSpEx[1] == 2
m3 = args_mSpEx[0] == 3 or args_mSpEx[1] == 3
out_model = ["const","exp","lin","K", "DynLimit"]
if m2: args.DD = True
if m3: rescale_factor = 0

#print len(ts),len(te[te>0]),sum(ts-te)
pad_s_times = np.concatenate( (s_times + np.repeat(0.01, len(s_times)), s_times + np.repeat(-0.01, len(s_times))) )
pad_s_times = pad_s_times[pad_s_times >= 0]
if args.DD is True:
    head_cov_file = ["","DD"]
    ts_te_vec = np.sort( np.concatenate((ts,te,pad_s_times)) )[::-1]
    Dtraj = getDT(ts_te_vec,ts,te) + np.zeros(len(ts_te_vec))
    times_of_T_change =  ts_te_vec
    Temp_values_original = Dtraj
    #for i in range(len(Temp_values)):
    #    print "%s\t%s" % (times_of_T_change[i],Temp_values[i])
    
else:
    tempfile = np.loadtxt(cov_file,skiprows=1)
    head_cov_file = next(open(cov_file)).split()
    times_of_T_change = tempfile[:,0]*args.rescale # array of times of Temp change
    Temp_values=       tempfile[:,1] # array of Temp values at times_of_T_change
    trim_temp = times_of_T_change <= np.max(ts)
    times_of_T_change = times_of_T_change[trim_temp]
    Temp_values = Temp_values[trim_temp]
    times_of_T_change_indexes = np.concatenate((np.zeros(len(times_of_T_change)),np.zeros(len(pad_s_times))+1))
    times_incl_s_times = np.concatenate((times_of_T_change,pad_s_times))
    idx = np.argsort(times_incl_s_times)
    times_incl_s_times_ord = times_incl_s_times[idx]
    times_of_T_change_indexes_ord = times_of_T_change_indexes[idx]
    times_incl_s_times_ord = times_incl_s_times[idx]
    Temp_values = get_VarValue_at_time(times_incl_s_times_ord,Temp_values,times_of_T_change_indexes_ord,times_of_T_change,np.max(times_incl_s_times))
    times_of_T_change = times_incl_s_times_ord
    Temp_values_original = Temp_values[::-1]

def get_Temp_values(Temp_values):
    # Temp_values= (Temp_values-Temp_values[0]) # so l0 and m0 are rates at the present
    if rescale_factor > 0: Temp_values = Temp_values*rescale_factor
    else: 
        denom = (np.max(Temp_values)-np.min(Temp_values))
        if denom==0: denom=1.
        Temp_values = Temp_values/denom
        Temp_values = Temp_values-np.min(Temp_values) # curve rescaled between 0 and 1


    #for i in range(len(Temp_values)):
    #    print "%s\t%s" % (times_of_T_change[i],Temp_values[i])

    #print "BRL" , np.sum(ts-te)
    #print "range:", np.max(Temp_values)-np.min(Temp_values)

    # create matrix of all events sorted (1st row) with indexes 0: times_of_T_change, 1: ts, 2: te, 3: te=0
    z=np.zeros(len(te))+2
    z[te==0] = 3
    all_events_temp= np.array([np.concatenate((times_of_T_change,ts,te),axis=0),
    np.concatenate((np.zeros(len(times_of_T_change)),np.zeros(len(ts))+1,z),axis=0)])

    idx = np.argsort(all_events_temp[0])[::-1] # get indexes of sorted events
    all_events_temp2=all_events_temp[:,idx] # sort by time of event

    times_of_T_change_tste=all_events_temp2[0,:]
    times_of_T_change_indexes=all_events_temp2[1,:].astype(int)
    #times_of_T_change_tste=sort(np.concatenate((times_of_T_change,ts,te),axis=0))[::-1]
    #print shape(times_of_T_change_tste2),shape(times_of_T_change_tste),times_of_T_change_tste2[r],times_of_T_change_tste[r],times_of_T_change_indexes[r]
    all_events=times_of_T_change_tste # events are speciation/extinction that change the diversity trajectory
    #print len(all_events), len(all_events[times_of_T_change_indexes==1]),len(all_events[times_of_T_change_indexes==2])
    ind_s = np.arange(len(all_events))[times_of_T_change_indexes==1]
    ind_e = np.arange(len(all_events))[times_of_T_change_indexes==2]

    n_events=len(all_events)
    Dtraj=init_Dtraj(1,n_events)

    # make trajectory curves for each clade
    Dtraj[:,0]=getDT(all_events,ts,te)

    #print "TIME", np.max(times_of_T_change), np.max(ts),Temp_values[-1]

    Temp_at_events= get_VarValue_at_time(times_of_T_change_tste,Temp_values,times_of_T_change_indexes,times_of_T_change,np.max(ts))

    if args.DD is True:
        Temp_at_events = Dtraj[:,0] + 0.
        if rescale_factor > 0: Temp_at_events = Temp_at_events*rescale_factor
        else: 
            denom = (np.max(Temp_at_events)-np.min(Temp_at_events))
            if denom==0: denom=1.
            Temp_at_events = Temp_at_events/denom



    #print Temp_at_events[150:]
    #_print "HERE",len(ind_s),len(ind_e)

    ### Get indexes of all events based on times of shift
    max_times_of_T_change_tste = np.max(times_of_T_change_tste)

    shift_ind = np.zeros(len(times_of_T_change_tste)).astype(int)
    if len(s_times)>0:
        bins_h = sort([max_times_of_T_change_tste+1,-1] + list(s_times))
        # hist gives the number of events within each time bin (between shifts)
        hist=np.histogram(times_of_T_change_tste,bins=bins_h)[0][::-1]
        I=np.empty(0)
        for i in range(len(hist)): I=np.append(I,np.repeat(i,hist[i]))
        # shift_ind = [0,0,0,1,1,2,2,2,2,...N], where 0 is index of oldest bin, N of the most recent
        shift_ind =I.astype(int)

    ##
    if est_start_time:
        ### Get indexes of all events based on times of shift
        shift_ind_temp_CURVE = np.zeros(len(times_of_T_change_tste)).astype(int)    
        np.max_est_start_time = np.min(np.max(ts),np.max(times_of_T_change))
        print("np.max allowed start time:",np.max_est_start_time)
        effect_start_timeA = np.max_est_start_time*np.random.uniform(0.1,0.9)
        bins_h_temp = sort([max_times_of_T_change_tste+1,-1] + [effect_start_timeA])
        # hist gives the number of events within each time bin (between shifts)
        hist_temp=np.histogram(times_of_T_change_tste,bins=bins_h_temp)[0][::-1]
        Itemp=np.empty(0)
        for i in range(len(hist_temp)): Itemp=np.append(Itemp,np.repeat(i,hist_temp[i]))
        # shift_ind = [0,0,0,1,1,2,2,2,2,...N], where 0 is index of oldest bin, N of the most recent
        shift_ind_temp_CURVE =Itemp.astype(int)


    ##
    scaled_temp=np.zeros(len(Temp_at_events))
    for i in range(len(np.unique(shift_ind))):
        Temp_values= Temp_at_events[shift_ind==i]
        if args_mSpEx[0] < 2 and args_mSpEx[1] < 2:
            Temp_values= (Temp_values-np.median([np.min(Temp_values),np.max(Temp_values)])) # so l0 and m0 are rates at the mean temp value
        scaled_temp[shift_ind==i]= Temp_values

    Temp_at_events=scaled_temp
    #for i in range(len(all_events)): #range(600,650):
    #    print "%s\t%s" % (round(all_events[i],2),round(Temp_at_events[i],2))
    #quit()


    if run_single_slice == 1: # values rescaled between 0 and 1 within the slice
        #print np.max(Temp_values)-np.min(Temp_values)
        temp_values_slice= Temp_at_events[shift_ind==index_slice_of_interest] 
        if rescale_factor > 0:
            temp_values_slice = temp_values_slice * rescale_factor
        else:
            #temp_values_slice= (temp_values_slice-temp_values_slice[0]) / (np.max(temp_values_slice)-np.min(temp_values_slice))
            temp_values_slice = temp_values_slice / (np.max(temp_values_slice) - np.min(temp_values_slice))
            temp_values_slice = temp_values_slice - np.min(temp_values_slice)
        Temp_at_events[shift_ind==index_slice_of_interest] = temp_values_slice
        #print temp_values_slice, np.max(temp_values_slice)-np.min(temp_values_slice)
    
    vec = [shift_ind, max_times_of_T_change_tste, 
           Temp_at_events, Temp_values, all_events,
           ind_s, ind_e, n_events, Dtraj]
    return vec

# INIT ALL DATA
[
    shift_ind, max_times_of_T_change_tste, 
    Temp_at_events, Temp_values, all_events,
    ind_s, ind_e, n_events, Dtraj
] = get_Temp_values(Temp_values_original + 0)


def get_running_avg(eps=1):
    Temp_values = Temp_values_original[::-1] + 0
    running_avg = [Temp_values[0]] 
    for i in range(1, len(Temp_values)): 
        running_avg.append(
             running_avg[-1] * (1 - eps) + 
             Temp_values[i] * eps
        )
       
    Temp_values = np.array(running_avg)[::-1]
    return Temp_values
    
def get_Temp_values_running_avg(eps):
    tmp = get_running_avg(eps=eps)
    return get_Temp_values(tmp)


if args.verbose is True:
    print("total branch length:" , np.sum(ts-te))
    print("raw range: %s (%s-%s)"       % (np.max(tempfile[:,1])-np.min(tempfile[:,1]), np.max(tempfile[:,1]), np.min(tempfile[:,1])))
    print("rescaled range: %s (%s-%s)" % (np.max(Temp_values)-np.min(Temp_values), np.max(Temp_values), np.min(Temp_values)))
    print("np.max diversity:", np.max(Dtraj))
    print("rescaling factor:", rescale_factor)
    print("\ntime\tvar.value\tdiversity")
    # for i in range(len(all_events)):
    #    print("%s\t%s\t%s" %  (all_events[i],Temp_at_events[i], Dtraj[i,0]))
    for i in range(len(ind_s)):
        print("%s\t%s\t%s" %  (all_events[ind_s[i]],Temp_at_events[ind_s[i]], Dtraj[ind_s[i],0]))
    
    import matplotlib
    import matplotlib.pyplot as plt
    
    # plt.plot(Temp_values)
    # plt.show()
    
    """
    python PyRateContinuous.py -d example_files/Ursidae_SE.txt -c example_files/PhanerozoicTempSmooth.txt -verbose
    """
    if lagged_model:
        plt.plot(all_events,Temp_at_events)
        [
            shift_ind, max_times_of_T_change_tste, 
            Temp_at_events, Temp_values, all_events,
            ind_s, ind_e, n_events, Dtraj
        ] = get_Temp_values_running_avg(eps=0.2)
        plt.plot(all_events,Temp_at_events)

        [
            shift_ind, max_times_of_T_change_tste, 
            Temp_at_events, Temp_values, all_events,
            ind_s, ind_e, n_events, Dtraj
        ] = get_Temp_values_running_avg(eps=1e-15)
        plt.plot(all_events,Temp_at_events)
                
        plt.legend(
                labels=(
                    "Original predictor",
                    "Lag prm = 0.2",
                    "Lag prm = 0.02"),
                # loc="center left", # "lower left"
                facecolor='white'
            )
            
        plt.xlabel("Time")
        plt.ylabel("Rescaled values")
        
        

        # print(Temp_values_original[:10])
        # print(Temp_values[:10])
        # for i in range(len(ind_s)):
        #     print("%s\t%s\t%s" %  (all_events[ind_s[i]],Temp_at_events[ind_s[i]], Dtraj[ind_s[i],0]))
    
    
        plt.show()
    quit()

# TODO: add rolling average


### INIT PARAMS
n_time_bins=len(np.unique(shift_ind))
GarrayA=np.zeros((2,n_time_bins)) # correlation parameters with Temp of lambda (GarrayA[0]) and mu (GarrayA[1])
if m2 or m3:
    GarrayA = GarrayA + 5. * np.max(Temp_at_events) # Carrying capacity parameters for diversity dependence
ZarrayA = np.zeros((2, n_time_bins)) + 1e-5 # z for species-environment-relationship
l0A,m0A= init_BD(n_time_bins),init_BD(n_time_bins)
hypRA = 1.
lag_epsA = np.exp(-2) # range = [0,1]

if args.pG>0:
    hypGA= args.pG**2 # variance of normal prior
else:
    hypGA= args.pG # if negative use uniform prior
if args.pZ>0:
    hypZA= args.pZ**2 # variance of normal prior
else:
    hypZA= args.pZ # if negative use uniform prior

### PLOT RTT
def get_marginal_rates(model,l0,m0,Garray,Temp_at_events,shift_ind,root_age,all_events):
    if model==0: 
        l_at_events=trasfMultipleRateTemp(l0, Garray[0],Temp_at_events,shift_ind)
        m_at_events=trasfMultipleRateTemp(m0, Garray[1],Temp_at_events,shift_ind)
    if model==1: 
        l_at_events=trasfMultipleRateTempLinear(l0, Garray[0],Temp_at_events,shift_ind)
        m_at_events=trasfMultipleRateTempLinear(m0, Garray[1],Temp_at_events,shift_ind)
    age_vec, l_vec, m_vec = list(),list(),list()
    for i in range(len(Temp_at_events)):
        age = all_events[i]
        if run_single_slice==1:
            if age < np.max(s_times):
                if len(s_times)==2 and age >= np.min(s_times):
                    age_vec.append(np.round(age,8))
                    l_vec.append(np.round(l_at_events[i],8))
                    m_vec.append(np.round(m_at_events[i],8))
                else:
                    age_vec.append(np.round(age,8))
                    l_vec.append(np.round(l_at_events[i],8))
                    m_vec.append(np.round(m_at_events[i],8))
        elif age <= root_age: 
            age_vec.append(np.round(age,8))
            l_vec.append(np.round(l_at_events[i],8))
            m_vec.append(np.round(m_at_events[i],8))

    return(age_vec,l_vec,m_vec)
    

summary_file = args.plot
if summary_file != "":
    root_age = np.max(ts)
    print("\nParsing log file:", summary_file)
    t=np.loadtxt(summary_file, skiprows=np.max([1,int(args.b)]))
    head = next(open(summary_file)).split()
    
    L0_index = [head.index(i) for i in head if "l0" in i]
    M0_index = [head.index(i) for i in head if "m0" in i]    
    Gl_index = [head.index(i) for i in head if "Gl" in i]
    Gm_index = [head.index(i) for i in head if "Gm" in i]
    # this is to remove samples from TI with temp < 1
    TI_beta_index = head.index("beta") 
    t = t[ t[:,TI_beta_index]==1 ]
    n_rates = len(L0_index)

    print("\nCalculating marginal rates...")
    marginal_L= list()
    marginal_M= list()
    for j in range(shape(t)[0]):
        L0,Gl,M0,Gm = np.zeros(n_rates),np.zeros(n_rates),np.zeros(n_rates),np.zeros(n_rates)
        if len(Gl_index)==len(L0_index):
            for i in range(n_rates):
                L0[i] = t[j,L0_index[i]]
                Gl[i] = t[j,Gl_index[i]]
                M0[i] = t[j,M0_index[i]]
                Gm[i] = t[j,Gm_index[i]]
        else: # plot when model is equal_G
            for i in range(n_rates):
                L0[i] = t[j,L0_index[i]]
                Gl[i] = t[j,Gl_index[0]]
                M0[i] = t[j,M0_index[i]]
                Gm[i] = t[j,Gm_index[0]]
        
        Garray = np.array([Gl,Gm])
        age_vec,l_vec,m_vec = get_marginal_rates(args.m,L0,M0,Garray,Temp_at_events,shift_ind,root_age,all_events)
        marginal_L.append(l_vec)
        marginal_M.append(m_vec)
    
    marginal_L = np.array(marginal_L)
    marginal_M = np.array(marginal_M)
    
    l_vec= np.zeros(np.shape(marginal_L)[1])
    m_vec= np.zeros(np.shape(marginal_L)[1])
    hpd_array_L= np.zeros((2,np.shape(marginal_L)[1]))
    hpd_array_M= np.zeros((2,np.shape(marginal_L)[1]))
    for i in range(np.shape(marginal_L)[1]):
        l_vec[i] = np.median(marginal_L[:,i])
        m_vec[i] = np.median(marginal_M[:,i])
        hpd_array_L[:,i] = calcHPD(marginal_L[:,i])
        hpd_array_M[:,i] = calcHPD(marginal_M[:,i])
    print("done")    
    # write R file
    print("\ngenerating R file...", end=' ')
    out="%s/%s_%s_%s_%sSp%sEx_RTT.r" % (output_wd,name_file,head_cov_file[1],rep_j,out_model[1+args_mSpEx[0]],out_model[1+args_mSpEx[1]])
    newfile = open(out, "w")    
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        wd_forward = os.path.abspath(output_wd).replace('\\', '/')
        r_script= "\n\npdf(file='%s/%s_%s_%s_%sSp%sEx_RTT.pdf',width=0.6*20, height=0.6*20)\nlibrary(scales)\n" % (wd_forward,name_file,head_cov_file[1],rep_j,out_model[1+args_mSpEx[0]],out_model[1+args_mSpEx[1]])
    else: r_script= "\n\npdf(file='%s/%s_%s_%s_%sSp%sEx_RTT.pdf',width=0.6*20, height=0.6*20)\nlibrary(scales)\n" % (output_wd,name_file,head_cov_file[1],rep_j,out_model[1+args_mSpEx[0]],out_model[1+args_mSpEx[1]])

    r_script += print_R_vec("\n\nt",  age_vec)
    r_script += "\ntime = -t"
    r_script += print_R_vec("\nspeciation",l_vec)
    r_script += print_R_vec("\nextinction",m_vec)
    
    r_script += print_R_vec('\nL_hpd_m',hpd_array_L[0,:])
    r_script += print_R_vec('\nL_hpd_M',hpd_array_L[1,:])
    r_script += print_R_vec('\nM_hpd_m',hpd_array_M[0,:])
    r_script += print_R_vec('\nM_hpd_M',hpd_array_M[1,:])
    
    
    r_script += """
    par(mfrow=c(2,1))
    plot(speciation ~ time,type="l",col="#4c4cec", lwd=3,main="Speciation rates", ylim = c(0,max(c(L_hpd_M,M_hpd_M))),xlab="Time",ylab="speciation rate",xlim=c(min(time),0))
    polygon(c(time, rev(time)), c(L_hpd_M, rev(L_hpd_m)), col = alpha("#4c4cec",0.3), border = NA)    
    abline(v %s,lty=2,col="gray")

    plot(extinction ~ time,type="l",col="#e34a33",  lwd=3,main="Extinction rates", ylim = c(0,max(c(L_hpd_M,M_hpd_M))),xlab="Time",ylab="extinction",xlim=c(min(time),0))
    polygon(c(time, rev(time)), c(M_hpd_M, rev(M_hpd_m)), col = alpha("#e34a33",0.3), border = NA)
    abline(v %s,lty=2,col="gray")
    """ % (lib_utilities.print_R_vec("",-s_times),lib_utilities.print_R_vec("",-s_times))
    
    r_script+="n<-dev.off()"
    newfile.writelines(r_script)
    newfile.close()
    print("\nAn R script with the source for the RTT plot was saved as: %sRTT.r\n(in %s)" % (name_file, output_wd))
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        cmd="cd %s & Rscript %s_%s_%s_%sSp%sEx_RTT.r" % (output_wd,name_file,head_cov_file[1],rep_j,out_model[1+args_mSpEx[0]],out_model[1+args_mSpEx[1]])
    else: 
        cmd="cd %s; Rscript %s/%s_%s_%s_%sSp%sEx_RTT.r" % (output_wd,output_wd,name_file,head_cov_file[1],rep_j,out_model[1+args_mSpEx[0]],out_model[1+args_mSpEx[1]])
    os.system(cmd)
    print("done\n")
    
    sys.exit("\n")




if len(s_times)>0: s_times_str = "s_" + '_'.join(s_times.astype("str"))
else: s_times_str=""

if equal_g==1: add_equal_g="EG"
else: add_equal_g=""
if equal_r==1: add_equal_r="ER"
else: add_equal_r=""
add_equal_z=""
if equal_z==1 and m3: add_equal_z="EZ"
if est_start_time:
    add_equal_r+="_ST" # estimateeffect starting time
if useHP==1: add_use_hp ="_HP"
else: add_use_hp =""

if lagged_model:
    add_lag = "_lag"
else:    
    add_lag = ""
    
if output_wd=="":
    out_file_name="%s_%s_%s_%s%sSp_%sEx%s%s%s%s%s.log" % \
    (os.path.splitext(os.path.basename(dataset))[0],head_cov_file[1],rep_j, 
    s_times_str,out_model[1+args_mSpEx[0]],out_model[1+args_mSpEx[1]],
    add_equal_g,add_equal_r,add_equal_z,add_use_hp, add_lag)
else:
    out_file_name="%s/%s_%s_%s_%s%sSp_%sEx%s%s%s%s%s.log" % \
    (output_wd,os.path.splitext(os.path.basename(dataset))[0],head_cov_file[1],rep_j,  
    s_times_str,out_model[1+args_mSpEx[0]],out_model[1+args_mSpEx[1]],
    add_equal_g,add_equal_r,add_equal_z,add_use_hp, add_lag)



    
    
logfile = open(out_file_name , "w") 
wlog=csv.writer(logfile, delimiter='\t')

head="it\tposterior\tlikelihood\tprior" 
time_slices = sort([max_times_of_T_change_tste+1,0] + list(s_times))[::-1]
time_bin_label=[]
for i in range(1,len(time_slices)): time_bin_label.append("%s-%s" % (int(time_slices[i-1]),int(time_slices[i])))
    
for i in range(n_time_bins): head+="\tlik_L_%s" % (time_bin_label[i])
for i in range(n_time_bins): head+="\tlik_M_%s" % (time_bin_label[i])
for i in range(n_time_bins): head+="\tl0_t%s" % (time_bin_label[i])
for i in range(n_time_bins): head+="\tm0_t%s" % (time_bin_label[i])
if equal_g==0:
    for j in range(n_time_bins): 
        head+="\tGl_t%s" % (time_bin_label[j])
    for j in range(n_time_bins): 
        head+="\tGm_t%s" % (time_bin_label[j])
else:
    head+="\tGl\tGm"
if m3:
    if equal_z==0:
        for j in range(n_time_bins): 
            head+="\tZl_t%s" % (time_bin_label[j])
        for j in range(n_time_bins): 
            head+="\tZm_t%s" % (time_bin_label[j])
    else:
        head+="\tZl\tZm"
if est_start_time:
    head+="\tstart_time"
if lagged_model:
    head+="\teps_lag\tlog_eps_lag"
head+="\thp_rate"
head+="\thp_sig2"
if m3: head+="\thpz_sig2"
head+="\tbeta"
wlog.writerow(head.split('\t'))
logfile.flush()

if args.A==0:
    scal_fac_TI=np.ones(1)
elif args.A==1:
    # parameters for TI are currently hard-coded (K=10, alpha=0.3)
    scal_fac_TI=get_temp_TI()

d1 = win_size[0]
d2 = win_size[1] # starting win size for Gl, Gm
list_d2=sort(exp(scal_fac_TI))**3*d2+(exp(1-np.array(scal_fac_TI))-1)*d2

## prep for lik calculation
abs_diff = abs(np.diff(all_events))
#abs_diff = np.concatenate((abs_diff,np.zeros(10)))
V1,V2,V3,V4 = list(),list(),list(),list()
for i in range(n_time_bins):
    v1 = (shift_ind==i).nonzero()[0]
    if i == n_time_bins-1: v_1 = v1[0:-1]
    else: v_1 = v1
    v_2 = v_1+1
    V1.append(v_1)
    V2.append(v_2)
    V3.append(np.intersect1d(ind_s,v_1))
    V4.append(np.intersect1d(ind_e,v_1))


scal_fac_ind=0
lik_pA=np.zeros(n_time_bins)
freq_update_rate = 1./len(l0A)
lag_prms_A = None
M = 1000.
m = -1000.
if m2 or m3:
    M = 10.*np.max(Dtraj[:,0]*rescale_factor)
    if rescale_factor == 0: M = 10.*np.max(Temp_at_events)
    m = 1e-5
for iteration in range(mcmc_gen * len(scal_fac_TI)):    
    
    if (iteration+1) % (mcmc_gen+1) ==0: 
        print(iteration, mcmc_gen)  
        scal_fac_ind+=1
    
    hasting=0
    l0,m0=0+l0A,0+m0A
    Garray=0+GarrayA
    Zarray=0+ZarrayA
    if iteration==0:
        likA,priorA,postA=0,0,0
    lik,priorBD=0,0
    
    # update values
    sampling_freqs=[.50,.52] # 0: rates, 1: hyper-priors, (2: correlation params)
    rr=np.random.uniform(0,1,3)
    if args.m== -1: rr[0]=0 # never update Garray
    lag_eps = lag_epsA
    
    GIBBS = 0
    updated_lag_prm = 0
    if est_start_time: effect_start_time=effect_start_timeA+0
    if iteration>10:
        if rr[0]<sampling_freqs[0] or iteration<1000:
            
            if est_start_time: effect_start_time = update_parameter(effect_start_timeA,m=0.5,M=np.max_est_start_time-0.5,d=w_size_start_time)
            
            if equal_r==0:
                if rr[1]>.5: 
                    l0,U=update_multiplier_freq(l0A,d=d1,f=freq_update_rate)
                else:    
                    m0,U=update_multiplier_freq(m0A,d=d1,f=freq_update_rate)
            else:
                if rr[1]>.5:
                    temp_R = 0+l0A[1] 
                    l0_temp,U=update_multiplier_freq(np.array([temp_R]),d=d1,f=1)
                    l0 = l0A*0+l0_temp[0]
                else: 
                    temp_R = 0+m0A[1]    
                    m0_temp,U=update_multiplier_freq(np.array([temp_R]),d=d1,f=1)
                    m0 = m0A*0+m0_temp[0]
            hasting=U
            
            if random.random() > 0.9 and lagged_model:
                updated_lag_prm = 1
                # lag_eps = update_parameter(lag_epsA, m=0, M=1, d=0.05)
                lag_eps = np.exp(
                    - update_parameter(np.abs(np.log(lag_epsA)), m=0, M=100, d=2)
                )
                # lag_eps_tmp, U = update_multiplier_proposal_val(-np.log(lag_epsA))
                # lag_eps = np.exp( -lag_eps_tmp)
                # hasting += U
                
                
            
            
        elif rr[0]<sampling_freqs[1] and useHP ==1:# Gibbs sampling (only if set true)
            GIBBS = 1
            # Gibbs sampler - Exponential + Gamma
            G_hp_alpha,G_hp_beta=2.,2.
            g_shape=G_hp_alpha+len(l0A)+len(m0A)
            g_rate=G_hp_beta+np.sum(l0A)+np.sum(m0A)
            hypRA = np.random.gamma(shape= g_shape, scale= 1./g_rate)
            #__ # Gibbs sampler - Normal(loc=0, tau) + Gamma
            #__ G_hp_alpha,G_hp_beta=1.,1.
            #__ g_shape=G_hp_alpha + len(GarrayA.flatten())/2.
            #__ g_rate=G_hp_beta + np.sum((GarrayA.flatten()-0)**2)/2.
            #__ hypGA = np.random.gamma(shape= g_shape, scale= 1./g_rate)
            # Gibbs sampler - Normal(loc=0, sig2) + InvGamma
            G_hp_alpha,G_hp_beta=1.,.1
            g_shape=G_hp_alpha + len(GarrayA.flatten())/2.
            g_rate=G_hp_beta + np.sum((GarrayA.flatten()-0)**2)/2.
            hypGA = 1./np.random.gamma(shape= g_shape, scale= 1./g_rate)
            if m3:
                Z_hp_alpha,Z_hp_beta=1.,.1
                z_shape=Z_hp_alpha + len(ZarrayA.flatten())/2.
                z_rate=Z_hp_beta + np.sum((ZarrayA.flatten()-0)**2)/2.
                hypZA = 1./np.random.gamma(shape= z_shape, scale= 1./z_rate)
        else:
            if rr[2]>.5 and args_mSpEx[0]> -1:
                if equal_g==0:
                    Garray[0]=update_parameter_normal_2d_freq(Garray[0],list_d2[scal_fac_ind],f=.25,m=m,M=M) 
                else:
                    Garray[0,:]=update_parameter_normal(Garray[0,0],list_d2[scal_fac_ind])[0]
                if m3:
                    if equal_z==0:
                        Zarray[0] = update_parameter_normal_2d_freq(Zarray[0],list_d2[scal_fac_ind], f=.25, m = 0.00001, M = 100.)
                    else:
                        Zarray[0,:] = update_parameter_normal(Zarray[0,0],list_d2[scal_fac_ind])[0]
                    Zarray = abs(Zarray)
            elif args_mSpEx[1]> -1:
                if equal_g==0:
                    Garray[1]=update_parameter_normal_2d_freq(Garray[1],list_d2[scal_fac_ind],f=.25,m=m,M=M) 
                else:
                    Garray[1,:]=update_parameter_normal(Garray[1,0],list_d2[scal_fac_ind])[0]
                if m3:
                    if equal_z==0:
                        Zarray[1]=update_parameter_normal_2d_freq(Zarray[1],list_d2[scal_fac_ind], f=.25, m = 0.00001, M = 100.)
                    else:
                        Zarray[1,:]=update_parameter_normal(Zarray[1,0],list_d2[scal_fac_ind])[0]
                    Zarray = abs(Zarray)
    #####
    modified_Temp_at_events = 0+Temp_at_events
    
    if lagged_model:
        if updated_lag_prm or lag_prms_A is None:
            [
                shift_ind, max_times_of_T_change_tste,
                Temp_at_events, Temp_values, all_events,
                ind_s, ind_e, n_events, Dtraj
            ] = get_Temp_values_running_avg(eps=lag_eps)
        else:
            [
                shift_ind, max_times_of_T_change_tste,
                Temp_at_events, Temp_values, all_events,
                ind_s, ind_e, n_events, Dtraj
            ] = lag_prms_A

    
    
    if est_start_time:
        #print effect_start_time
        ### Get indexes of all events based on times of shift
        shift_ind_temp_CURVE = np.zeros(len(times_of_T_change_tste)).astype(int)
        bins_h_temp = sort([max_times_of_T_change_tste+1,-1] + [effect_start_time])
        # hist gives the number of events within each time bin (between shifts)
        hist_temp=np.histogram(times_of_T_change_tste,bins=bins_h_temp)[0][::-1]
        Itemp=np.empty(0)
        for i in range(len(hist_temp)): Itemp=np.append(Itemp,np.repeat(i,hist_temp[i]))
        # shift_ind = [0,0,0,1,1,2,2,2,2,...N], where 0 is index of oldest bin, N of the most recent
        shift_ind_temp_CURVE =Itemp.astype(int)
        modified_Temp_at_events[shift_ind_temp_CURVE==0] = modified_Temp_at_events[ (shift_ind_temp_CURVE==1).nonzero()[0][0]  ]
#    l0 = np.repeat(1.1, n_time_bins)
#    m0 = np.repeat(0.9, n_time_bins)
#    Garray[0] = np.repeat(0.2, n_time_bins)
#    Garray[1] = np.repeat(0.2, n_time_bins)
    
    if args_mSpEx[0]==0: 
        l_at_events=trasfMultipleRateTemp(l0, Garray[0],modified_Temp_at_events,shift_ind)
    if args_mSpEx[0]==1: 
        l_at_events=trasfMultipleRateTempLinear(l0, Garray[0],modified_Temp_at_events,shift_ind)
    if args_mSpEx[0]== -1: 
        l_at_events=trasfMultipleRateTempLinear(l0, Garray[0],modified_Temp_at_events,shift_ind) #np.repeat(l0,len(modified_Temp_at_events))
    if args_mSpEx[0]== 2:
        l_at_events = trasfMultipleRateK(l0, Garray[0],modified_Temp_at_events,shift_ind, "l")
    if args_mSpEx[0]== 3:
        l_at_events = trasfMultipleRateKsar(l0, Garray[0], modified_Temp_at_events,shift_ind, "l", Zarray[0], Dtraj[:,0])

    if args_mSpEx[1]==0: 
        m_at_events=trasfMultipleRateTemp(m0, Garray[1],modified_Temp_at_events,shift_ind)
    if args_mSpEx[1]==1: 
        m_at_events=trasfMultipleRateTempLinear(m0, Garray[1],modified_Temp_at_events,shift_ind)
    if args_mSpEx[1]== -1: 
        m_at_events=trasfMultipleRateTempLinear(m0, Garray[1],modified_Temp_at_events,shift_ind) #np.repeat(m0,len(modified_Temp_at_events))
    if args_mSpEx[1]== 2:
        m_at_events = trasfMultipleRateK(m0, Garray[1],modified_Temp_at_events,shift_ind, "m")
    if args_mSpEx[1]== 3:
        m_at_events = trasfMultipleRateKsar(m0, Garray[1], modified_Temp_at_events,shift_ind, "m", Zarray[1], Dtraj[:,0])

    #if iteration % 10000==0:
    #    print modified_Temp_at_events
    #    print m_at_events
    #quit() #m_at_events[shift_ind]
    
    
    # Global likelihood
    #__ l_s1a=l_at_events[ind_s]
    #__ m_e1a=m_at_events[ind_e]
        #__ 
    #__ lik =  np.sum(log(l_s1a))-np.sum( abs(np.diff(all_events))*l_at_events[0:len(l_at_events)-1]*(Dtraj[:,0][1:len(l_at_events)])) \
    #__       +sum(log(m_e1a))-np.sum( abs(np.diff(all_events))*m_at_events[0:len(m_at_events)-1]*(Dtraj[:,0][1:len(l_at_events)])) 

    # partial likelihoods
    if run_single_slice == 0:
        lik_p=np.zeros(n_time_bins*2)
        for i in range(n_time_bins):
            v_1 = V1[i]
            v_2 = V2[i]
            l_s1a=l_at_events[V3[i]]
            lik_p[i] = np.sum(log(l_s1a)) -np.sum( abs_diff[v_1] * l_at_events[v_1] * (Dtraj[v_2,0])) 

        for i in range(n_time_bins):
            v_1 = V1[i]
            v_2 = V2[i]
            m_e1a=m_at_events[V4[i]]
            lik_p[i+n_time_bins] = np.sum(log(m_e1a)) -np.sum( abs_diff[v_1] * m_at_events[v_1] * (Dtraj[v_2,0])) 
    else:
        lik_p=np.zeros(n_time_bins*2)           
        v_1 = V1[index_slice_of_interest]
        v_2 = V2[index_slice_of_interest]
        l_s1a=l_at_events[V3[index_slice_of_interest]]
        lik_p[index_slice_of_interest] = np.sum(log(l_s1a)) -np.sum( abs_diff[v_1] * l_at_events[v_1] * (Dtraj[v_2,0])) 
        m_e1a=m_at_events[V4[index_slice_of_interest]]
        lik_p[index_slice_of_interest+n_time_bins] = np.sum(log(m_e1a)) -np.sum( abs_diff[v_1] * m_at_events[v_1] * (Dtraj[v_2,0])) 
    
    # Check likelihoods  
    #__ if iteration % 100 ==0:
    #__    print round(lik - np.sum(lik_p), 8)
    lik=np.sum(lik_p)
    
    lik_alter = lik * scal_fac_TI[scal_fac_ind]
    
    # Add hyper-prior + Gibbs sampling 
    #print np.anp.max(abs(Garray)), -hypGA
    if hypGA>0: # use normal prior on G par
        prior = prior_normal(Garray,scale=sqrt(hypGA)) 
    else: # use uniform prior on G par
        if np.amax(np.abs(Garray)) > -hypGA:
            prior = -np.inf
        else: 
            prior = 0
    if m3:
        if hypZA>0: # use normal prior on Z par
            prior += prior_normal(Zarray,scale=sqrt(hypZA)) 
        else: # use uniform prior on Z par
            if np.amax(np.abs(Zarray)) > -hypZA:
                prior += -np.inf
            else: 
                prior += 0
    prior += prior_exponential(l0,rate=hypRA) + prior_exponential(m0,rate=hypRA)  # prior_normal_tau(Garray,precision=hypGA)
    
    if lagged_model:
        # prior += prior_beta(lag_eps, 1.01, 1) #
        prior += prior_exponential(-np.log(lag_eps),rate=0.1)
    
    if (lik_alter + prior + hasting) - postA >= log(rand.random()) or iteration==0 or GIBBS == 1:
        postA=lik_alter+prior
        likA=lik
        lik_pA=lik_p
        priorA=prior
        l0A=l0
        m0A=m0
        GarrayA=Garray
        ZarrayA=Zarray
        if est_start_time: 
            effect_start_timeA=effect_start_time
        if lagged_model:
            lag_epsA = lag_eps
            
        lag_prms_A = [
            shift_ind, max_times_of_T_change_tste, 
            Temp_at_events, Temp_values, all_events,
            ind_s, ind_e, n_events, Dtraj
        ]
        
        
    if iteration % print_freq ==0: 
        print(iteration, array([postA, likA,lik,prior]), hasting, scal_fac_TI[scal_fac_ind])
        print("l:",l0A, "\nm:", m0A, "\nG:", GarrayA.flatten())
        if m3: print("Z:", ZarrayA.flatten())
        if est_start_time: 
            print("start.time:", effect_start_timeA, max_times_of_T_change_tste,"\n")
        if lagged_model: 
            print("eps:", lag_epsA, np.log(lag_epsA))
    if iteration % sampling_freq ==0:
        if equal_g==0:
            g_vec_write = list(GarrayA.flatten())
        else: g_vec_write = [GarrayA[0,0],GarrayA[1,0]]
        z_vec_write = []
        hypZA_write = []
        if m3:
            if equal_z==0:
                z_vec_write = list(ZarrayA.flatten())
            else: z_vec_write = [ZarrayA[0,0],ZarrayA[1,0]]
            hypZA_write = list(np.array([hypZA]))
        if est_start_time: 
            log_state=[iteration,postA,likA,priorA] + list(lik_pA) + list(l0A) + list(m0A) +g_vec_write + z_vec_write +[effect_start_timeA]
        else: 
            log_state=[iteration,postA,likA,priorA] + list(lik_pA) + list(l0A) + list(m0A) +g_vec_write + z_vec_write
        if lagged_model:
            log_state = log_state + [lag_epsA, np.log(lag_epsA)]
        
        log_state = log_state + [hypRA,hypGA] + hypZA_write + [scal_fac_TI[scal_fac_ind]]
        wlog.writerow(log_state)
        logfile.flush()


quit()












