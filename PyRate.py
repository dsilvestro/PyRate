#!/usr/bin/env python
# Created by Daniele Silvestro on 02/03/2012 => pyrate.help@gmail.com
import argparse, os,sys, platform, time, csv, glob
import random as rand
import warnings, importlib
import importlib.util
import copy as copy_lib
import json

version= "PyRate"
build  = "v3.1.3 - 20230825"
if platform.system() == "Darwin": sys.stdout.write("\x1b]2;%s\x07" % version)

citation= """Silvestro, D., Antonelli, A., Salamin, N., & Meyer, X. (2019). 
Improved estimation of macroevolutionary rates from fossil data using a Bayesian framework. 
Paleobiology, doi: 10.1017/pab.2019.23.
"""

# check python version
V=list(sys.version_info[0:3])
if V[0]<3: sys.exit("""\nYou need Python v.3 to run this version of PyRate""")

# LOAD LIBRARIES
import argparse
try:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    from numpy import *
    import numpy as np
except(ImportError):
    sys.exit("\nError: numpy library not found.\nYou can install numpy using: 'pip install numpy'\n")

try:
    import scipy
    from scipy.special import gamma
    from scipy.special import beta as f_beta
    from scipy.special import gdtr, gdtrix
    from scipy.special import betainc
    import scipy.stats
    from scipy.optimize import fmin_powell as Fopt1
except(ImportError):
    sys.exit("\nError: scipy library not found.\nYou can install scipy using: 'pip install scipy'\n")

try:
    import pandas as pd
except(ImportError):
    print("\nWarning: pandas library not found.\nYou can install pandas using: 'pip install pandas'\n")


try:
    import multiprocessing, _thread
    import multiprocessing.pool
    class NoDaemonProcess(multiprocessing.Process):
        # make 'daemon' attribute always return False
        def _get_daemon(self): return False
        def _set_daemon(self, value): pass
        daemon = property(_get_daemon, _set_daemon)

    class mcmcMPI(multiprocessing.pool.Pool):
        # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
        # because the latter is only a wrapper function, not a proper class.
        Process = NoDaemonProcess
    use_seq_lik= 0
    if platform.system() == "Windows" or platform.system() == "Microsoft": use_seq_lik= 1
except(ImportError):
    print("\nWarning: library multiprocessing not found.\n")
    use_seq_lik= 1

if platform.system() == "Windows" or platform.system() == "Microsoft": use_seq_lik= 1

version_details="PyRate %s; OS: %s %s; Python version: %s; Numpy version: %s; Scipy version: %s" \
% (build, platform.system(), platform.release(), sys.version, np.version.version, scipy.version.version)

### numpy print options ###
np.set_printoptions(suppress= 1, precision=3) # prints floats, no scientific notation
numpy_major_version = int(np.__version__[0])
if numpy_major_version > 1:
    np.set_printoptions(legacy='1.25') # avoid printing data type ine.g. BDNN when using numpy >= 2.0

original_stderr = sys.stderr
NO_WARN = original_stderr #open('pyrate_warnings.log', 'w')
small_number= 1e-50

def get_self_path():
    self_path = -1
    path_list = [os.path.dirname(sys.argv[0]) , os.getcwd()]
    for path in path_list:
        try:
            self_path=path
            importlib.util.spec_from_file_location("test", "%s/pyrate_lib/lib_updates_priors.py" % (self_path))
            break
        except:
            self_path = -1
    if self_path== -1:
        print(os.getcwd(), os.path.dirname(sys.argv[0]))
        sys.exit("pyrate_lib not found.\n")
    return self_path


# Search for the module
hasFoundPyRateC = 0

try:
    if platform.system()=="Darwin": 
        os_spec_lib="macOS"
        try:
            from pyrate_lib.fastPyRateC.macOS._FastPyRateC import PyRateC_BD_partial_lik, PyRateC_HOMPP_lik, PyRateC_setFossils, \
                                PyRateC_getLogGammaPDF, PyRateC_initEpochs, PyRateC_HPP_vec_lik, \
                                                            PyRateC_NHPP_lik, PyRateC_FBD_T4
        except:
            from pyrate_lib.fastPyRateC.macOS_arm._FastPyRateC import PyRateC_BD_partial_lik, PyRateC_HOMPP_lik, PyRateC_setFossils, \
                                   PyRateC_getLogGammaPDF, PyRateC_initEpochs, PyRateC_HPP_vec_lik, \
                                                             PyRateC_NHPP_lik, PyRateC_FBD_T4
    elif platform.system() == "Windows" or platform.system() == "Microsoft": 
        os_spec_lib="Windows"
        py_version = sys.version_info.minor
        if py_version < 7:
            from pyrate_lib.fastPyRateC.Windows.py36._FastPyRateC import PyRateC_BD_partial_lik, PyRateC_HOMPP_lik, PyRateC_setFossils, \
                                   PyRateC_getLogGammaPDF, PyRateC_initEpochs, PyRateC_HPP_vec_lik, \
                                                             PyRateC_NHPP_lik, PyRateC_FBD_T4
        elif py_version == 7:
            from pyrate_lib.fastPyRateC.Windows.py37._FastPyRateC import PyRateC_BD_partial_lik, PyRateC_HOMPP_lik, PyRateC_setFossils, \
                                   PyRateC_getLogGammaPDF, PyRateC_initEpochs, PyRateC_HPP_vec_lik, \
                                                             PyRateC_NHPP_lik, PyRateC_FBD_T4
        elif py_version == 8:
            from pyrate_lib.fastPyRateC.Windows.py38._FastPyRateC import PyRateC_BD_partial_lik, PyRateC_HOMPP_lik, PyRateC_setFossils, \
                                   PyRateC_getLogGammaPDF, PyRateC_initEpochs, PyRateC_HPP_vec_lik, \
                                                             PyRateC_NHPP_lik, PyRateC_FBD_T4
        elif py_version == 9:
            from pyrate_lib.fastPyRateC.Windows.py39._FastPyRateC import PyRateC_BD_partial_lik, PyRateC_HOMPP_lik, PyRateC_setFossils, \
                                   PyRateC_getLogGammaPDF, PyRateC_initEpochs, PyRateC_HPP_vec_lik, \
                                                             PyRateC_NHPP_lik, PyRateC_FBD_T4
        elif py_version == 10:
            from pyrate_lib.fastPyRateC.Windows.py310._FastPyRateC import PyRateC_BD_partial_lik, PyRateC_HOMPP_lik, PyRateC_setFossils, \
                                   PyRateC_getLogGammaPDF, PyRateC_initEpochs, PyRateC_HPP_vec_lik, \
                                                             PyRateC_NHPP_lik, PyRateC_FBD_T4
        elif py_version == 11:
            from pyrate_lib.fastPyRateC.Windows.py311._FastPyRateC import PyRateC_BD_partial_lik, PyRateC_HOMPP_lik, PyRateC_setFossils, \
                                   PyRateC_getLogGammaPDF, PyRateC_initEpochs, PyRateC_HPP_vec_lik, \
                                                             PyRateC_NHPP_lik, PyRateC_FBD_T4
        elif py_version == 12:
            from pyrate_lib.fastPyRateC.Windows.py312._FastPyRateC import PyRateC_BD_partial_lik, PyRateC_HOMPP_lik, PyRateC_setFossils, \
                                   PyRateC_getLogGammaPDF, PyRateC_initEpochs, PyRateC_HPP_vec_lik, \
                                                             PyRateC_NHPP_lik, PyRateC_FBD_T4
        elif py_version == 13:
            from pyrate_lib.fastPyRateC.Windows.py313._FastPyRateC import PyRateC_BD_partial_lik, PyRateC_HOMPP_lik, PyRateC_setFossils, \
                                   PyRateC_getLogGammaPDF, PyRateC_initEpochs, PyRateC_HPP_vec_lik, \
                                                             PyRateC_NHPP_lik, PyRateC_FBD_T4
    else: 
        os_spec_lib = "Other"
        from pyrate_lib.fastPyRateC.Other._FastPyRateC import PyRateC_BD_partial_lik, PyRateC_HOMPP_lik, PyRateC_setFossils, \
                               PyRateC_getLogGammaPDF, PyRateC_initEpochs, PyRateC_HPP_vec_lik, \
                                                         PyRateC_NHPP_lik, PyRateC_FBD_T4

    #c_lib_path = "pyrate_lib/fastPyRateC/%s" % (os_spec_lib)
    #sys.path.append(os.path.join(self_path,c_lib_path))
    #print self_path, sys.path
    #import pyrate_lib.fastPyRateC.macOS._FastPyRateC

    hasFoundPyRateC = 1
    # print("Module FastPyRateC was loaded.")
    # Set that to true to enable sanity check (comparing python and c++ results)
    sanityCheckForPyRateC = 0
    sanityCheckThreshold = 1e-10
    if sanityCheckForPyRateC == 1:
        print("Sanity check for FastPyRateC is enabled.")
        print("Python and C results will be compared and any divergence greater than ", sanityCheckThreshold, " will be reported.")
except:
    # print("Module FastPyRateC was not found.")
    hasFoundPyRateC = 0
    sanityCheckForPyRateC = 0

########################## CALC PROBS ##############################
def calcHPD(data, level) :
    assert (0 < level < 1)
    d = list(data)
    d.sort()
    nData = len(data)
    nIn = int(round(level * nData))
    if nIn < 2 :
        raise RuntimeError("not enough data")
    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)) :
        rk = d[k+nIn-1] - d[k]
        if rk < r :
            r = rk
            i = k
    assert 0 <= i <= i+nIn-1 < len(d)
    return (d[i], d[i+nIn-1])

def check_burnin(b,I):
    #print b, I
    if b<1:burnin=int(b*I)
    else: burnin=int(b)
    if burnin>=(I-10) and b > 0:
        print("Warning: burnin too high! Excluding 10% instead.")
        burnin=int(0.1*I)
    return burnin

def calc_model_probabilities(f,burnin):
    print("parsing log file...\n")
    t=loadtxt(f, skiprows=1)
    num_it=shape(t)[0]
    if num_it<10: sys.exit("\nNot enough samples in the log file!\n")
    burnin=check_burnin(burnin, num_it)
    print(("First %s samples excluded as burnin.\n" % (burnin)))
    file1=open(f, 'r')
    L=file1.readlines()
    head= L[0].split()
    PAR1=["k_birth","k_death"]
    k_ind= [head.index(s) for s in head if s in PAR1]
    if len(k_ind)==0: k_ind =[head.index(s) for s in head if s in ["K_l","K_m"]]
    z1=t[burnin:,k_ind[0]]  # list of shifts (lambda)
    z2=t[burnin:,k_ind[1]]  # list of shifts (mu)
    y1= np.maximum(np.max(z1),np.max(z2))
    print("Model           Probability")
    print("          Speciation  Extinction")
    for i in range(1,int(y1)+1):
        k_l=float(len(z1[z1==i]))/len(z1)
        k_m=float(len(z2[z2==i]))/len(z2)
        print(("%s-rate    %s      %s" % (i,round(k_l,4),round(k_m,4))))
    print("\n")

    try:
        import collections,os
        d = collections.OrderedDict()

        def count_BD_config_freq(A):
            for a in A:
                t = tuple(a)
                if t in d: d[t] += 1
                else: d[t] = 1

            result = []
            for (key, value) in list(d.items()): result.append(list(key) + [value])
            return result

        BD_config = t[burnin:,np.array(k_ind)]
        B = np.asarray(count_BD_config_freq(BD_config))
        B[:,2]=B[:,2]/sum(B[:,2])
        B = B[B[:,2].argsort()[::-1]]
        cum_prob = np.cumsum(B[:,2])
        print("Best BD/ID configurations (rel.pr >= 0.05)")
        print("   B/I    D      Rel.pr")
        print(B[(np.round(B[:,2],3)>=0.05).nonzero()[0],:])

    except: pass
    quit()

def calc_ts_te(f, burnin):
    if f=="null": return FA,LO
    else:
        t_file=np.loadtxt(f, skiprows=1)
        head = np.array(next(open(f)).split())
        # if t_file.shape[0]==1: sys.exit("\nNot enough samples in the log file!\n")
        # if shape_f[1]<10: sys.exit("\nNot enough samples in the log file!\n")
        ind_start=np.where(head=="tot_length")[0][0]
        indexes= np.array([ind_start+1, ind_start+(t_file.shape[1]-ind_start)/2]).astype(int)
        # fixes the case of missing empty column at the end of each row
        if t_file[0,-1] != "False":
            indexes = indexes+np.array([0,1])
        ind_ts0 = indexes[0]
        ind_te0 = indexes[1]

        meanTS,meanTE=list(),list()
        burnin=check_burnin(burnin, t_file.shape[0])
        burnin+=1
        j=0
        for i in range(ind_ts0,ind_te0):
            meanTS.append(np.mean(t_file[burnin:,i]))
            meanTE.append(np.mean(t_file[burnin:,ind_te0+j]))
            j+=1
        return array(meanTS),array(meanTE)

def calc_BF(f1, f2):
    input_file_raw = [os.path.basename(f1),os.path.basename(f2)]
    def get_ML(FILE):
        file1=open(FILE, 'r')
        L=file1.readlines()
        for i in range(len(L)):
            if "Marginal likelihood" in L[i]:
                x=L[i].split("Marginal likelihood: ")
                ML= float(x[1])
                return ML
    BF= 2*(get_ML(f1)-get_ML(f2))
    if abs(BF)<2: support="negligible"
    elif abs(BF)<6: support="positive"
    elif abs(BF)<10: support="strong"
    else: support="very strong"
    if BF>0: best=0
    else: best=1
    print(("\nModel A: %s\nModelB: %s" % (input_file_raw[best],input_file_raw[abs(best-1)])))
    print(("\nModel A received %s support against Model B\nBayes Factor: %s\n\n" % (support, round(abs(BF), 4))))



def calc_BFlist(f1):
    #f1 = os.path.basename(f1)
    #print f1
    tbl = np.genfromtxt(f1,"str")
    file_list = tbl[:,tbl[0]=="file_name"]

    f_list, l_list = list(), list()

    for i in range(1, len(file_list)):
        fn = str(file_list[i][0])
        f_list.append(fn)
        l_list.append(float(tbl[i,tbl[0]=="likelihood"][0]))

    ml = l_list[l_list.index(max(l_list))]

    l_list = np.array(l_list)
    bf = 2*(ml - l_list)

    print("Found %s models:" % (len(bf)))
    for i in range(len(bf)): print("model %s: '%s'" % (i,f_list[i]))

    print("\nBest model:", f_list[(bf==0).nonzero()[0][0]], ml, "\n")


    for i in range(len(bf)):
        BF = bf[i]
        if abs(BF)==0: pass
        else:
            if abs(BF)<2: support="negligible"
            elif abs(BF)<6: support="positive"
            elif abs(BF)<10: support="strong"
            else: support="very strong"
            print("Support in favor of model %s: %s (%s)" % (i, BF, support))

#def get_DT(T,s,e): # returns the Diversity Trajectory of s,e at times T (x10 faster)
#    B=np.sort(np.append(T,T[0]+1))+.000001 # the + .0001 prevents problems with identical ages
#    ss1 = np.histogram(s,bins=B)[0]
#    ee2 = np.histogram(e,bins=B)[0]
#    DD=(ss1-ee2)[::-1]
#    #return np.insert(np.cumsum(DD),0,0)[0:len(T)]
#    return np.cumsum(DD)[0:len(T)]


def get_DT(T, s, e):
    change_e = np.full_like(e, -1, dtype=int)
    change_e[e == 0.0] = 0
    change = np.concatenate((np.ones_like(s, dtype=int), change_e))
    event_times = np.concatenate((s, e))
    order = np.argsort(-event_times) # Order by decreasing times
    event_times = event_times[order]
    change = change[order]
    diversity = np.cumsum(change)
    interp_func = scipy.interpolate.interp1d(-event_times, diversity, kind='previous', bounds_error=False, fill_value=0)
    B = np.sort(T)
    return interp_func(-B)[::-1]


########################## PLOT RTT ##############################
def plot_RTT(infile,burnin, file_stem="",one_file= 0, root_plot=0,plot_type=1):
    burnin = int(burnin)
    if burnin<=1:
        print("Burnin must be provided in terms of number of samples to be excluded.")
        print("E.g. '-b 100' will remove the first 100 samples.")
        print("Assuming burnin = 1.\n")
    def print_R_vec(name,v):
        new_v=[]
        for j in range(0,len(v)):
            value=v[j]
            if isnan(v[j]): value="NA"
            new_v.append(value)

        vec="%s=c(%s, " % (name,new_v[0])
        for j in range(1,len(v)-1): vec += "%s," % (new_v[j])
        vec += "%s)"  % (new_v[j+1])
        return vec


    path_dir = infile
    sys.path.append(infile)
    plot_title = file_stem.split('_')[0]
    print("FILE STEM:",file_stem, plot_title)
    if file_stem=="": direct="%s/*_marginal_rates.log" % infile
    else: direct="%s/*%s*marginal_rates.log" % (infile,file_stem)
    files=glob.glob(direct)
    files=sort(files)
    if one_file== 1: files=["%s/%smarginal_rates.log" % (infile,file_stem)]

    stem_file=files[0]
    name_file = os.path.splitext(os.path.basename(stem_file))[0]

    wd = "%s" % os.path.dirname(stem_file)
    #print(name_file, wd)
    print("found", len(files), "log files...\n")

    ########################################################
    ######           DETERMINE MIN ROOT AGE           ######
    ########################################################
    if root_plot==0:
        min_age=np.inf
        print("determining min age...", end=' ')
        for f in files:
            file_name =  os.path.splitext(os.path.basename(f))[0]
            sys.stdout.write(".")
            sys.stdout.flush()
            head = next(open(f)).split() # should be faster
            sp_ind = [head.index(s) for s in head if "l_" in s]
            min_age = np.minimum(min_age, len(sp_ind))

        print("Min root age:", min_age)
        max_ind=min_age-1
    else: max_ind = int(root_plot-1)

    print(max_ind, root_plot)
    ########################################################
    ######            COMBINE ALL LOG FILES           ######
    ########################################################
    print("\ncombining all files...", end=' ')
    file_n=0
    for f in files:
        file_name =  os.path.splitext(os.path.basename(f))[0]
        print(file_name)
        try:
            max_ind = int(max_ind)
            t=np.loadtxt(f, skiprows=np.maximum(1,burnin))
            sys.stdout.write(".")
            sys.stdout.flush()
            head = next(open(f)).split()
            l_ind= [head.index(s) for s in head if "l_" in s]
            m_ind= [head.index(s) for s in head if "m_" in s]
            r_ind= [head.index(s) for s in head if "r_" in s]
            l_ind=l_ind[0:max_ind]
            m_ind=m_ind[0:max_ind]
            r_ind=r_ind[0:max_ind]

            if file_n==0:
                L_tbl=t[:,l_ind]
                M_tbl=t[:,m_ind]
                R_tbl=t[:,r_ind]
                file_n=1
                #if np.min([np.max(L_tbl),np.max(M_tbl)])>0.1: no_decimals = 3
                #elif np.min([np.max(L_tbl),np.max(M_tbl)])>0.01: no_decimals = 5
                #else:
                no_decimals = 15
            else:
                L_tbl=np.concatenate((L_tbl,t[:,l_ind]),axis=0)
                M_tbl=np.concatenate((M_tbl,t[:,m_ind]),axis=0)
                R_tbl=np.concatenate((R_tbl,t[:,r_ind]),axis=0)
        except:
            print("skipping file:", f)

    ########################################################
    ######               CALCULATE HPDs               ######
    ########################################################
    print("\ncalculating HPDs...", end=' ')
    def get_HPD(threshold=.95):
        L_hpd_m,L_hpd_M=[],[]
        M_hpd_m,M_hpd_M=[],[]
        R_hpd_m,R_hpd_M=[],[]
        sys.stdout.write(".")
        sys.stdout.flush()
        for time_ind in range(shape(L_tbl)[1]):
            hpd1=np.around(calcHPD(L_tbl[:,time_ind],threshold),decimals=no_decimals)
            hpd2=np.around(calcHPD(M_tbl[:,time_ind],threshold),decimals=no_decimals)
            if len(r_ind)>0:
                hpd3=np.around(calcHPD(R_tbl[:,time_ind],threshold),decimals=no_decimals)
            else:
                hpd3 =  hpd1- hpd2
            L_hpd_m.append(hpd1[0])
            L_hpd_M.append(hpd1[1])
            M_hpd_m.append(hpd2[0])
            M_hpd_M.append(hpd2[1])
            R_hpd_m.append(hpd3[0])
            R_hpd_M.append(hpd3[1])

        return [L_hpd_m,L_hpd_M,M_hpd_m,M_hpd_M,R_hpd_m,R_hpd_M]

    def get_CI(threshold=.95):
        threshold = (1-threshold)/2.
        L_hpd_m,L_hpd_M=[],[]
        M_hpd_m,M_hpd_M=[],[]
        R_hpd_m,R_hpd_M=[],[]
        sys.stdout.write(".")
        sys.stdout.flush()
        for time_ind in range(shape(R_tbl)[1]):
            l=np.sort(L_tbl[:,time_ind])
            m=np.sort(M_tbl[:,time_ind])
            r=np.sort(R_tbl[:,time_ind])
            hpd1=np.around(np.array([l[int(threshold*len(l))] , l[int(len(l) - threshold*len(l))] ]),decimals=no_decimals)
            hpd2=np.around(np.array([m[int(threshold*len(m))] , m[int(len(m) - threshold*len(m))] ]),decimals=no_decimals)
            hpd3=np.around(np.array([r[int(threshold*len(r))] , r[int(len(r) - threshold*len(r))] ]),decimals=no_decimals)

            L_hpd_m.append(hpd1[0])
            L_hpd_M.append(hpd1[1])
            M_hpd_m.append(hpd2[0])
            M_hpd_M.append(hpd2[1])
            R_hpd_m.append(hpd3[0])
            R_hpd_M.append(hpd3[1])
        return [L_hpd_m,L_hpd_M,M_hpd_m,M_hpd_M,R_hpd_m,R_hpd_M]



    hpds95 =  np.array(get_HPD(threshold=.95))
    hpds50 =  np.array(get_CI(threshold=.50))
    #hpds10 =  get_CI(threshold=.10)

    L_tbl_mean=np.around(np.mean(L_tbl,axis=0),no_decimals)
    M_tbl_mean=np.around(np.mean(M_tbl,axis=0),no_decimals)
    if len(r_ind)>0: R_tbl_mean=np.around(np.mean(R_tbl,axis=0),no_decimals)
    else:
        R_tbl_mean= L_tbl_mean-M_tbl_mean
    mean_rates=np.array([L_tbl_mean,L_tbl_mean,M_tbl_mean,M_tbl_mean,R_tbl_mean,R_tbl_mean] )

    nonzero_rate = L_tbl_mean+ M_tbl_mean
    NA_ind = (nonzero_rate==0).nonzero()[0]

    hpds95[:,NA_ind] = np.nan
    #hpds50[:,NA_ind] = np.nan
    print(mean_rates)
    mean_rates[:,NA_ind] = np.nan
    print("HPD", hpds95)
    #print(np.shape(np.array(hpds50)    ), np.shape(L_tbl_mean))

    ########################################################
    ######                  PLOT RTTs                 ######
    ########################################################
    print("\ngenerating R file...", end=' ')
    out="%s/%s_RTT.r" % (wd,name_file)
    newfile = open(out, "w")
    Rfile="# %s files combined:\n" % (len(files))
    for f in files: Rfile+="# \t%s\n" % (f)
    Rfile+= """\n# 95% HPDs calculated using code from Biopy (https://www.cs.auckland.ac.nz/~yhel002/biopy/)"""

    if plot_type==1: n_plots=4
    else: n_plots=3

    if platform.system() == "Windows" or platform.system() == "Microsoft":
        wd_forward = os.path.abspath(wd).replace('\\', '/')
        Rfile+= "\n\npdf(file='%s/%s_RTT.pdf',width=10.8, height=8.4)\npar(mfrow=c(2,2))" % (wd_forward,name_file)
    else:
        Rfile+= "\n\npdf(file='%s/%s_RTT.pdf',width=10.8, height=8.4)\npar(mfrow=c(2,2))" % (wd,name_file)

    Rfile+= "\nlibrary(scales)"

    if plot_type==2: Rfile+= """\nplot_RTT <- function (age,hpd_M,hpd_m,mean_m,color){
    N=100
    beta=(1:(N-1))/N
    alpha_shape=0.25
    cat=1-(beta^(1./alpha_shape))
    for (i in 1:(N-1)){
        trans= 1/N + 2/N
        polygon(c(age, rev(age)), c(hpd_M-((hpd_M-mean_m)*cat[i]), rev(hpd_m+((mean_m-hpd_m)*cat[i]))), col = alpha(color,trans), border = NA)
    }
    lines(rev(age), rev(mean_m), col = color, lwd=3)\n}
    """

    def RTT_plot_in_R(args, alpha):
        count=0
        data=""

        name=['95','_mean'] # ,'50'
        for hpd_list in args:
            sys.stdout.write(".")
            sys.stdout.flush()
            [L_hpd_m,L_hpd_M,M_hpd_m,M_hpd_M,R_hpd_m,R_hpd_M]=hpd_list
            if name[count]=="_mean":
                data += print_R_vec('\nL_mean',L_hpd_m)
                data += print_R_vec('\nM_mean',M_hpd_m)
                data += print_R_vec('\nR_mean',R_hpd_m)
            else:
                data += print_R_vec('\nL_hpd_m%s',L_hpd_m) % name[count]
                data += print_R_vec('\nL_hpd_M%s',L_hpd_M) % name[count]
                data += print_R_vec('\nM_hpd_m%s',M_hpd_m) % name[count]
                data += print_R_vec('\nM_hpd_M%s',M_hpd_M) % name[count]
                data += print_R_vec('\nR_hpd_m%s',R_hpd_m) % name[count]
                data += print_R_vec('\nR_hpd_M%s',R_hpd_M) % name[count]
            if count==0:
                max_x_axis,min_x_axis = -len(L_hpd_m), 0 # root to the present
                max_x_axis,min_x_axis = -(len(L_hpd_m)+.05*len(L_hpd_m)), -(len(L_hpd_m)-len(L_hpd_m[np.isfinite(L_hpd_m)]))+.05*len(L_hpd_m)
                plot_L = "\ntrans=%s\nage=(0:(%s-1))* -1" % (alpha, len(L_hpd_m))
                plot_L += "\nplot(age,age,type = 'n', ylim = c(%s, %s), xlim = c(%s,%s), ylab = 'Speciation rate', xlab = 'Ma',main='%s' )" \
                    % (0,1.1*np.nanmax(L_hpd_M),max_x_axis,min_x_axis,plot_title)
                plot_M  = "\nplot(age,age,type = 'n', ylim = c(%s, %s), xlim = c(%s,%s), ylab = 'Extinction rate', xlab = 'Ma' )" \
                    % (0,1.1*np.nanmax(M_hpd_M),max_x_axis,min_x_axis)
                plot_R  = "\nplot(age,age,type = 'n', ylim = c(%s, %s), xlim = c(%s,%s), ylab = 'Net diversification rate', xlab = 'Ma' )" \
                    % (-abs(1.1*np.nanmin(R_hpd_m)),1.1*np.nanmax(R_hpd_M),max_x_axis,min_x_axis)
                plot_R += """\nabline(h=0,lty=2,col="darkred")""" # \nabline(v=-c(65,200,251,367,445),lty=2,col="darkred")

            if name[count]=="_mean":
                plot_L += """\nlines(rev(age), rev(L_mean), col = "#4c4cec", lwd=3)"""
                plot_M += """\nlines(rev(age), rev(M_mean), col = "#e34a33", lwd=3)"""
                plot_R += """\nlines(rev(age), rev(R_mean), col = "#504A4B", lwd=3)"""
            else:
                if plot_type==1:
                    plot_L += """\npolygon(c(age, rev(age)), c(L_hpd_M%s, rev(L_hpd_m%s)), col = alpha("#4c4cec",trans), border = NA)""" % (name[count],name[count])
                    plot_M += """\npolygon(c(age, rev(age)), c(M_hpd_M%s, rev(M_hpd_m%s)), col = alpha("#e34a33",trans), border = NA)""" % (name[count],name[count])
                    plot_R += """\npolygon(c(age, rev(age)), c(R_hpd_M%s, rev(R_hpd_m%s)), col = alpha("#504A4B",trans), border = NA)""" % (name[count],name[count])
                elif plot_type==2:
                    plot_L += """\nplot_RTT(age,L_hpd_M95,L_hpd_m95,L_mean,"#4c4cec")"""
                    plot_M += """\nplot_RTT(age,M_hpd_M95,M_hpd_m95,M_mean,"#e34a33")"""
                    plot_R += """\nplot_RTT(age,R_hpd_M95,R_hpd_m95,R_mean,"#504A4B")"""

            count+=1

        R_code=data+plot_L+plot_M+plot_R

        if plot_type==1:
            R_code += "\nplot(age,rev(1/M_mean),type = 'n', xlim = c(%s,%s), ylab = 'Longevity (Myr)', xlab = 'Ma' )" % (max_x_axis,min_x_axis)
            R_code += """\nlines(rev(age), rev(1/M_mean), col = "#504A4B", lwd=3)"""
            #R_code += """\npolygon(c(age, rev(age)), c((1/M_hpd_m95), rev(1/M_hpd_M95)), col = alpha("#504A4B",trans), border = NA)"""

        return R_code

    Rfile += RTT_plot_in_R([hpds95,mean_rates],.5) # ,hpds50

    Rfile += "\nn <- dev.off()"
    newfile.writelines(Rfile)
    newfile.close()
    print("\nAn R script with the source for the RTT plot was saved as: %s_RTT.r\n(in %s)" % (name_file, wd))
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        cmd="cd %s & Rscript %s_RTT.r" % (wd,name_file)
    else:
        cmd="cd %s; Rscript %s/%s_RTT.r" % (wd,wd,name_file)
    os.system(cmd)
    print("done\n")


def plot_ltt(tste_file,plot_type=1,rescale= 1,step_size=1.): # change rescale to change bin size
    # plot_type=1 : ltt + min/max range
    # plot_type=2 : log10 ltt + min/max range
    #step_size=int(step_size)
    # read data
    print("Processing data...")
    tbl = np.genfromtxt(tste_file, skip_header=1)
    j_max=int((np.shape(tbl)[1]-1)/2)
    j_range=np.arange(j_max)
    ts = tbl[:,2+2*j_range]*rescale
    te = tbl[:,3+2*j_range]*rescale
    time_vec = np.sort(np.linspace(np.min(te),np.max(ts),int((np.max(ts)-np.min(te))/float(step_size)) ))

    # create out file
    wd = "%s" % os.path.dirname(tste_file)
    out_file_name = os.path.splitext(os.path.basename(tste_file))[0]
    out_file="%s/%s" % (wd,out_file_name+"_ltt.txt")
    ltt_file = open(out_file , "w", newline="")
    ltt_log=csv.writer(ltt_file, delimiter='\t')


    # calc ltt
    print(time_vec)
    dtraj = []
    for rep in j_range:
        sys.stdout.write(".")
        sys.stdout.flush()
        dtraj.append(get_DT(time_vec,ts[:,rep],te[:,rep])[::-1])
    dtraj = np.array(dtraj)
    div_mean = np.mean(dtraj,axis=0)
    div_m    = np.min(dtraj,axis=0)
    div_M    = np.max(dtraj,axis=0)

    Ymin,Ymax,yaxis = 0,np.max(div_M)+1,""
    if np.min(div_m)>5: Ymin = np.min(div_m)-1

    if plot_type==2:
        div_mean = np.log10(div_mean)
        div_m    = np.log10(div_m   )
        div_M    = np.log10(div_M   )
        Ymin,Ymax,yaxis = np.min(div_m),np.max(div_M), " (Log10)"

    # write to file
    if plot_type==1 or plot_type==2:
        ltt_log.writerow(["time","diversity","m_div","M_div"])
        for i in range(len(time_vec)):
            ltt_log.writerow([time_vec[i]/rescale,div_mean[i],div_m[i],div_M[i]])
        ltt_file.close()
        plot2 = """polygon(c(time, rev(time)), c(tbl$M_div, rev(tbl$m_div)), col = alpha("#504A4B",0.5), border = NA)"""

    # write multiple LTTs to file
    if plot_type==3:
        header = ["time","diversity"]+["rep%s" % (i) for i in j_range]
        ltt_log.writerow(header)
        plot2=""
        for i in range(len(time_vec)):
            d = dtraj[:,i]
            ltt_log.writerow([time_vec[i]/rescale,div_mean[i]]+list(d))
            plot2 += """\nlines(time,tbl$rep%s, type="l",lwd = 1,col = alpha("#504A4B",0.5))""" % (i)
        ltt_file.close()

    ###### R SCRIPT
    R_file_name="%s/%s" % (wd,out_file_name+"_ltt.R")
    R_file=open(R_file_name, "w")
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        tmp_wd = os.path.abspath(wd).replace('\\', '/')
    else: tmp_wd = wd
    R_script = """
    setwd("%s")
    tbl = read.table(file = "%s_ltt.txt",header = T)
    pdf(file='%s_ltt.pdf',width=12, height=9)
    time = -tbl$time
    library(scales)
    plot(time,tbl$diversity, type="n",ylab= "Number of lineages%s", xlab="Time (Ma)", main="Range-through diversity through time", ylim=c(%s,%s),xlim=c(min(time),0))
    %s
    lines(time,tbl$diversity, type="l",lwd = 2)
    n<-dev.off()
    """ % (tmp_wd, out_file_name,out_file_name, yaxis, Ymin,Ymax,plot2)

    R_file.writelines(R_script)
    R_file.close()
    print("\nAn R script with the source for the stat plot was saved as: \n%s" % (R_file_name))
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        cmd="cd %s & Rscript %s" % (wd,out_file_name+"_ltt.R")
    else:
        cmd="cd %s; Rscript %s" % (wd,out_file_name+"_ltt.R")
    os.system(cmd)
    sys.exit("done\n")




########################## PLOT TS/TE STAT ##############################
def plot_tste_stats(tste_file, EXT_RATE, step_size,no_sim_ex_time,burnin,rescale,ltt_only=1):
    step_size=int(step_size)
    # read data
    print("Processing data...")
    tbl = np.loadtxt(tste_file,skiprows=1)
    j_max=(np.shape(tbl)[1]-1)/2
    j=np.arange(j_max)
    ts = tbl[:,2+2*j]*rescale
    te = tbl[:,3+2*j]*rescale
    root = int(np.max(ts)+1)

    if EXT_RATE==0:
        EXT_RATE = len(te[te>0])/np.sum(ts-te) # estimator for overall extinction rate
        print("estimated extinction rate:", EXT_RATE)

    wd = "%s" % os.path.dirname(tste_file)
    # create out file
    out_file_name = os.path.splitext(os.path.basename(tste_file))[0]
    out_file="%s/%s" % (wd,out_file_name+"_stats.txt")
    out_file=open(out_file, "w", newline="")

    out_file.writelines("time\tdiversity\tm_div\tM_div\tmedian_age\tm_age\tM_age\tturnover\tm_turnover\tM_turnover\tlife_exp\tm_life_exp\tM_life_exp\t")

    no_sim_ex_time = int(no_sim_ex_time)
    def draw_extinction_time(te,EXT_RATE):
        te_mod = np.zeros(np.shape(te))
        ind_extant = (te[:,0]==0).nonzero()[0]
        te_mod[ind_extant,:] = -np.random.exponential(1/EXT_RATE,(len(ind_extant),len(te[0]))) # sim future extinction
        te_mod += te
        return te_mod

    def calc_median(arg):
        if len(arg)>1: return np.median(arg)
        else: return np.nan

    extant_at_time_t_previous = [0]
    for i in range(0,root+1,step_size):
        time_t = root-i
        up = time_t+step_size
        lo = time_t
        extant_at_time_t = [np.intersect1d((ts[:,rep] >= lo).nonzero()[0], (te[:,rep] <= up).nonzero()[0]) for rep in j]
        extinct_in_time_t =[np.intersect1d((te[:,rep] >= lo).nonzero()[0], (te[:,rep] <= up).nonzero()[0]) for rep in j]
        diversity = [len(extant_at_time_t[rep]) for rep in j]
        try:
            #turnover = [1-len(np.intersect1d(extant_at_time_t_previous[rep],extant_at_time_t[rep]))/float(len(extant_at_time_t[rep])) for rep in j]
            turnover = [(len(extant_at_time_t[rep])-len(np.intersect1d(extant_at_time_t_previous[rep],extant_at_time_t[rep])))/float(len(extant_at_time_t[rep])) for rep in j]
        except:
            turnover = [np.nan for rep in j]

        if min(diversity)<=1:
            age_current_taxa = [np.nan for rep in j]
        else:
            ext_age = [calc_median(ts[extinct_in_time_t[rep],rep]-te[extinct_in_time_t[rep],rep]) for rep in j]
            age_current_taxa = [calc_median(ts[extant_at_time_t[rep],rep]-time_t) for rep in j]

        # EMPIRICAL/PREDICTED LIFE EXPECTANCY
        life_exp=list()
        try:
            ex_rate = [float(EXT_RATE)]
            r_ind = np.repeat(0,no_sim_ex_time)
        except(ValueError):
            t=np.loadtxt(EXT_RATE, skiprows=np.maximum(1,int(burnin)))
            head = next(open(EXT_RATE)).split()
            m_ind= [head.index(s) for s in head if "m_0" in s]
            ex_rate= [mean(t[:,m_ind])]
            r_ind = np.random.randint(0,len(ex_rate),no_sim_ex_time)

        if min(diversity)<=1:
            life_exp.append([np.nan for rep in j])
        else:
            for sim in range(no_sim_ex_time):
                #print ex_rate[r_ind[sim]]
                te_mod = draw_extinction_time(te,ex_rate[r_ind[sim]])
                te_t = [te_mod[extant_at_time_t[rep],:] for rep in j]
                life_exp.append([median(time_t-te_t[rep]) for rep in j])

        life_exp= np.array(life_exp)
        STR= "\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" \
        % (time_t, np.median(diversity),np.min(diversity),np.max(diversity),
        np.median(age_current_taxa),np.min(age_current_taxa),np.max(age_current_taxa),
        np.median(turnover),np.min(turnover),np.max(turnover),
        np.median(life_exp),np.min(life_exp),np.max(life_exp))
        extant_at_time_t_previous = extant_at_time_t
        STR = STR.replace("nan","NA")
        sys.stdout.write(".")
        sys.stdout.flush()
        out_file.writelines(STR)
    out_file.close()

    ###### R SCRIPT
    R_file_name="%s/%s" % (wd,out_file_name+"_stats.R")
    R_file=open(R_file_name, "w")
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        tmp_wd = os.path.abspath(wd).replace('\\', '/')
    else: tmp_wd = wd
    R_script = """
    setwd("%s")
    tbl = read.table(file = "%s_stats.txt",header = T)
    pdf(file='%s_stats.pdf',width=12, height=9)
    time = -tbl$time
    par(mfrow=c(2,2))
    library(scales)
    plot(time,tbl$diversity, type="l",lwd = 2, ylab= "Number of lineages", xlab="Time (Ma)", main="Diversity through time", ylim=c(0,max(tbl$M_div,na.rm =T)+1),xlim=c(min(time),0))
    polygon(c(time, rev(time)), c(tbl$M_div, rev(tbl$m_div)), col = alpha("#504A4B",0.5), border = NA)
    plot(time,tbl$median_age, type="l",lwd = 2, ylab = "Median age", xlab="Time (Ma)", main= "Taxon age", ylim=c(0,max(tbl$M_age,na.rm =T)+1),xlim=c(min(time),0))
    polygon(c(time, rev(time)), c(tbl$M_age, rev(tbl$m_age)), col = alpha("#504A4B",0.5), border = NA)
    plot(time,tbl$turnover, type="l",lwd = 2, ylab = "Fraction of new taxa", xlab="Time (Ma)", main= "Turnover", ylim=c(0,max(tbl$M_turnover,na.rm =T)+.1),xlim=c(min(time),0))
    polygon(c(time, rev(time)), c(tbl$M_turnover, rev(tbl$m_turnover)), col = alpha("#504A4B",0.5), border = NA)
    plot(time,tbl$life_exp, type="l",lwd = 2, ylab = "Median longevity", xlab="Time (Ma)", main= "Taxon (estimated) longevity", ylim=c(0,max(tbl$M_life_exp,na.rm =T)+1),xlim=c(min(time),0))
    polygon(c(time, rev(time)), c(tbl$M_life_exp, rev(tbl$m_life_exp)), col = alpha("#504A4B",0.5), border = NA)
    n<-dev.off()
    """ % (tmp_wd, out_file_name,out_file_name)
    R_file.writelines(R_script)
    R_file.close()
    print("\nAn R script with the source for the stat plot was saved as: \n%s" % (R_file_name))
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        cmd="cd %s & Rscript %s" % (wd,out_file_name+"_stats.R")
    else:
        cmd="cd %s; Rscript %s" % (wd,out_file_name+"_stats.R")
    os.system(cmd)
    print("done\n")



########################## COMBINE LOG FILES ##############################
def read_rates_log(f, burnin):
    f_temp = open(f, 'r')
    x_temp = [line for line in f_temp.readlines()]
    num_it = len(x_temp)
    b = check_burnin(burnin, num_it)
    f_temp = open(f, 'r')
    x_temp = [line for line in f_temp.readlines()]
    x_temp = x_temp[b:]
    x_temp = np.array(x_temp)
    return x_temp


def comb_rj_rates(infile, files, burnin, tag, resample, rate_type):
    j=0
    for f in files:
        x_temp = read_rates_log(f, burnin)
        if 2>1: #try:
            if resample>0:
                r_ind = np.sort(np.random.randint(0,len(x_temp),resample))
                x_temp = x_temp[r_ind]
            if j==0:
                comb = x_temp
            else:
                comb = np.concatenate((comb,x_temp))
            j+=1
        #except:
        #    print "Could not process file:",f

    outfile = "%s/combined_%s%s_%s.log" % (infile,len(files),tag,rate_type)
    with open(outfile, 'w') as f:
        for i in comb: f.write(i)

def comb_mcmc_files(infile, files,burnin,tag,resample,col_tag,file_type="", keep_q=False):
    j=0
    if file_type=="mcmc" and keep_q:
        n_q_shifts = np.zeros(len(files))
        for k in range(len(files)):
            f = files[k]
            if platform.system() == "Windows" or platform.system() == "Microsoft":
                f = f.replace("\\","/")
            head = np.array(next(open(f)).split())
            if len(col_tag) == 0:
                n_q_shifts[k] = len([i for i in range(len(head)) if head[i].startswith('q_')])
        max_q_shifts = int(np.max(n_q_shifts))
    
    for f in files:
        if platform.system() == "Windows" or platform.system() == "Microsoft":
            f = f.replace("\\","/")

        if 2>1: #try:
            file_name =  os.path.splitext(os.path.basename(f))[0]
            print(file_name, end=' ')
            num_it = np.loadtxt(f, skiprows=1).shape[0]
            b = check_burnin(burnin, num_it)
            t_file=np.loadtxt(f, skiprows=b + 1)
            shape_f=shape(t_file)
            print(shape_f)
            #t_file = t[burnin:shape_f[0],:]#).astype(str)
            # only sample from cold chain

            head = np.array(next(open(f)).split()) # should be faster\
            if j == 0:
                tbl_header = '\t'.join(head)
            if "temperature" in head or "beta" in head:
                try:
                    temp_index = np.where(head=="temperature")[0][0]
                except(IndexError):
                    temp_index = np.where(head=="beta")[0][0]

                temp_values = t_file[:,temp_index]
                t_file = t_file[temp_values==1,:]
                print("removed heated chains:",np.shape(t_file))


            if len(col_tag) == 0 and file_type == "mcmc":
                q_ind = np.array([i for i in range(len(head)) if head[i].startswith('q_')])
                if len(q_ind)>0 and not keep_q:
                    # exclude preservation rates under TPP model (they can mismatch)
                    mean_q = np.mean(t_file[:,q_ind],axis=1)
                    t_file = np.delete(t_file,q_ind,axis=1)
                    t_file = np.insert(t_file,q_ind[0],mean_q,axis=1)
                elif len(q_ind) < max_q_shifts:
                    missing_q = np.full((t_file.shape[0], max_q_shifts - len(q_ind)), np.nan)
                    idx = np.min(q_ind)
                    t_file = np.c_[t_file[:, :idx], missing_q, t_file[:, idx:]]

            shape_f=shape(t_file)

            if resample>0:
                r_ind= sort(np.random.randint(0,shape_f[0],resample))
                t_file = t_file[r_ind,:]



        #except: print "ERROR in",f
        if len(col_tag) == 0:
            if j==0:
                head_temp = np.array(next(open(f)).split())
                if file_type == "mcmc":
                    q_not_ind = np.array([i for i in range(len(head)) if not head[i].startswith('q_')])
                    q_ind = np.array([i for i in range(len(head)) if head[i].startswith('q_')])
                    if len(q_ind) > 0 and not keep_q:
                        head_temp = head_temp[q_not_ind]
                        head_temp = np.insert(head_temp,q_ind[0],"mean_q")
                    elif len(q_ind) > 0:
                        head_temp = head_temp[q_not_ind]
                        q_names = np.array(["q_" + str(i) for i in range(max_q_shifts)])
                        head_temp = np.insert(head_temp, q_ind[0], q_names)
                tbl_header=""
                for i in head_temp: tbl_header = tbl_header + i  + "\t"
                tbl_header+="\n"
                comb = t_file
            else:
                comb = np.concatenate((comb,t_file),axis=0)
        else:
            head_temp = next(open(f)).split() # should be faster
            sp_ind_list=[]
            for TAG in col_tag:
                if TAG in head_temp:
                    sp_ind_list+=[head_temp.index(s) for s in head_temp if s == TAG]

            try:
                col_tag_ind = np.array([int(tag_i) for tag_i in col_tag])
                sp_ind= np.array(col_tag_ind)
            except:
                sp_ind= np.array(sp_ind_list)

            #print "COLTAG",col_tag, sp_ind, head_temp
            #sys.exit()


            #print "INDEXES",sp_ind
            if j==0:
                head_temp= np.array(head_temp)
                head_t= ["%s\t" % (i) for i in head_temp[sp_ind]]
                tbl_header="it\t"
                for i in head_t: tbl_header+=i
                tbl_header+="\n"
                print("found", len(head_t), "columns")
                comb = t_file[:,sp_ind]
            else:
                comb = np.concatenate((comb,t_file[:,sp_ind]),axis=0)

        j+=1

    #print shape(comb)
    if len(col_tag) == 0:
        sampling_freq= comb[1,0]-comb[0,0]
        comb[:,0] = (np.arange(0,len(comb))+1)*sampling_freq
        fmt_list=['%i']
        for i in range(1,np.shape(comb)[1]): fmt_list.append('%4f')
    else:
        fmt_list=['%i']
        for i in range(1,np.shape(comb)[1]+1): fmt_list.append('%4f')
        comb = np.concatenate((np.zeros((len(comb[:,0]),1)),comb),axis=1)
    comb[:,0] = (np.arange(0,len(comb)))

    print(np.shape(comb), len(fmt_list))

    outfile = "%s/combined_%s%s_%s.log" % (infile,len(files),tag,file_type)

    with open(outfile, 'w') as f:
        f.write(tbl_header)
        if platform.system() == "Windows" or platform.system() == "Microsoft":
            np.savetxt(f, comb, delimiter="\t",fmt=fmt_list,newline="\r") #)
        else:
            np.savetxt(f, comb, delimiter="\t",fmt=fmt_list,newline="\n") #)

def comb_log_files_smart(path_to_files,burnin=0,tag="",resample=0,col_tag=[], keep_q=False):
    infile=path_to_files
    sys.path.append(infile)
    direct="%s/*%s*.log" % (infile,tag)
    files=glob.glob(direct)
    files=sort(files)
    print("found", len(files), "log files...\n")
    if len(files)==0: quit()
    j=0

    # RJ rates files
    files_temp = [f for f in files if "_sp_rates.log" in os.path.basename(f)]
    if len(files_temp)>1:
        print("processing %s *_sp_rates.log files" % (len(files_temp)))
        comb_rj_rates(infile, files_temp, burnin, tag, resample, rate_type="sp_rates")
    files_temp = [f for f in files if "_ex_rates.log" in os.path.basename(f)]
    if len(files_temp)>1:
        print("processing %s *_ex_rates.log files" % (len(files_temp)))
        comb_rj_rates(infile, files_temp, burnin, tag, resample, rate_type="ex_rates")

    # MCMC files
    files_temp = [f for f in files if "_mcmc.log" in os.path.basename(f)]
    if len(files_temp)>1:
        print("processing %s *_mcmc.log files" % (len(files_temp)))
        comb_mcmc_files(infile, files_temp,burnin,tag,resample,col_tag,file_type="mcmc", keep_q=keep_q)
    files_temp = [f for f in files if "_marginal_rates.log" in os.path.basename(f)]
    if len(files_temp)>1:
        print("processing %s *_marginal_rates.log files" % (len(files_temp)))
        comb_mcmc_files(infile, files_temp,burnin,tag,resample,col_tag,file_type="marginal_rates")

    # BDNN files
    files_temp = [f for f in files if "_per_species_rates.log" in os.path.basename(f)]
    if len(files_temp)>1:
        print("processing %s *_per_species_rates.log files" % (len(files_temp)))
        comb_mcmc_files(infile, files_temp,burnin,tag,resample,col_tag,file_type="per_species_rates")
    files_temp = [f for f in files if "_q_rates.log" in os.path.basename(f) and not "_species_q_rates.log" in os.path.basename(f)]
    if len(files_temp)>1:
        print("processing %s *_q_rates.log files" % (len(files_temp)))
        comb_rj_rates(infile, files_temp, burnin, tag, resample, rate_type="q_rates")
    files_temp = [f for f in files if "_species_q_rates.log" in os.path.basename(f)]
    if len(files_temp)>1:
        print("processing %s *_species_q_rates.log files" % (len(files_temp)))
        comb_mcmc_files(infile, files_temp,burnin,tag,resample,col_tag,file_type="species_q_rates")


def comb_log_files(path_to_files,burnin=0,tag="",resample=0,col_tag=[]):
    infile=path_to_files
    sys.path.append(infile)
    direct="%s/*%s*.log" % (infile,tag)
    files=glob.glob(direct)
    files=sort(files)
    print("found", len(files), "log files...\n")
    if len(files)==0: quit()
    j=0

    if "_sp_rates.log" in os.path.basename(files[0]) or "_ex_rates.log" in os.path.basename(files[0]):
        for f in files:
            x_temp = read_rates_log(f, burnin)
            try:
                if resample>0:
                    r_ind= np.sort(np.random.randint(0,len(x_temp),resample))
                    x_temp = x_temp[r_ind]
                if j==0:
                    comb = x_temp
                else:
                    comb = np.concatenate((comb,x_temp))
                j+=1
            except:
                print("Could not process file:",f)

        outfile = "%s/combined_%s%s.log" % (infile,len(files),tag)
        with open(outfile, 'w') as f:
            #
            for i in comb: f.write(i)
             #fmt_list=['%i']
             #if platform.system() == "Windows" or platform.system() == "Microsoft":
             #    np.savetxt(f, comb, delimiter="\t",fmt=fmt_list,newline="\r") #)
             #else:
             #    np.savetxt(f, comb, delimiter="\t",fmt=fmt_list,newline="\n") #)
               #
        sys.exit("done")


    for f in files:
        if platform.system() == "Windows" or platform.system() == "Microsoft":
            f = f.replace("\\","/")

        try:
            file_name =  os.path.splitext(os.path.basename(f))[0]
            print(file_name, end=' ')
            num_it = np.loadtxt(f, skiprows=1).shape[0]
            b = check_burnin(burnin, num_it)
            t_file=np.loadtxt(f, skiprows=b + 1)
            shape_f=shape(t_file)
            print(shape_f)
            #t_file = t[burnin:shape_f[0],:]#).astype(str)
            # only sample from cold chain

            head = np.array(next(open(f)).split()) # should be faster\
            #txt_tbl = np.genfromtxt(f, delimiter="\t")
            #print "TRY", txt_tbl[0:],np.shape(txt_tbl), head
            if j == 0:
                tbl_header = '\t'.join(head)
            if "temperature" in head or "beta" in head:
                try:
                    temp_index = np.where(head=="temperature")[0][0]
                except(IndexError):
                    temp_index = np.where(head=="beta")[0][0]

                temp_values = t_file[:,temp_index]
                t_file = t_file[temp_values==1,:]
                print("removed heated chains:",np.shape(t_file))
            shape_f=shape(t_file)

            if resample>0:
                r_ind= sort(np.random.randint(0,shape_f[0],resample))
                t_file = t_file[r_ind,:]



        except: print("ERROR in",f)
        if len(col_tag) == 0:
            if j==0:
                tbl_header = next(open(f))#.split()
                comb = t_file
            else:
                comb = np.concatenate((comb,t_file),axis=0)
        else:
            head_temp = next(open(f)).split() # should be faster
            sp_ind_list=[]
            for TAG in col_tag:
                if TAG in head_temp:
                    sp_ind_list+=[head_temp.index(s) for s in head_temp if s == TAG]

            try:
                col_tag_ind = np.array([int(tag_i) for tag_i in col_tag])
                sp_ind= np.array(col_tag_ind)
            except:
                sp_ind= np.array(sp_ind_list)

            #print "COLTAG",col_tag, sp_ind, head_temp
            #sys.exit()


            #print "INDEXES",sp_ind
            if j==0:
                head_temp= np.array(head_temp)
                head_t= ["%s\t" % (i) for i in head_temp[sp_ind]]
                tbl_header="it\t"
                for i in head_t: tbl_header+=i
                tbl_header+="\n"
                print("found", len(head_t), "columns")
                comb = t_file[:,sp_ind]
            else:
                comb = np.concatenate((comb,t_file[:,sp_ind]),axis=0)

        j+=1

    #print shape(comb)
    if len(col_tag) == 0:
        sampling_freq= comb[1,0]-comb[0,0]
        comb[:,0] = (np.arange(0,len(comb))+1)*sampling_freq
        fmt_list=['%i']
        for i in range(1,np.shape(comb)[1]): fmt_list.append('%4f')
    else:
        fmt_list=['%i']
        for i in range(1,np.shape(comb)[1]+1): fmt_list.append('%4f')
        comb = np.concatenate((np.zeros((len(comb[:,0]),1)),comb),axis=1)
    comb[:,0] = (np.arange(0,len(comb)))

    print(np.shape(comb), len(fmt_list))

    outfile = "%s/combined_%s%s.log" % (infile,len(files),tag)

    with open(outfile, 'w') as f:
        f.write(tbl_header)
        if platform.system() == "Windows" or platform.system() == "Microsoft":
            np.savetxt(f, comb, delimiter="\t",fmt=fmt_list,newline="\r") #)
        else:
            np.savetxt(f, comb, delimiter="\t",fmt=fmt_list,newline="\n") #)

########################## INITIALIZE MCMC ##############################
def get_gamma_rates(a):
    b=a
    m = gdtrix(b,a,YangGammaQuant) # user defined categories
    s=pp_gamma_ncat/sum(m) # multiplier to scale the so that the mean of the discrete distribution is one
    return array(m)*s # SCALED VALUES

# using get_gamma_rates() in bdnn_lib is not working because it cannot access the YangGammaQuant
def get_gamma_rates_qnn(a, pp_gamma_ncat, YangGammaQuant):
    b = a
    m = gdtrix(b, a, YangGammaQuant) # user defined categories
    s = pp_gamma_ncat / np.sum(m) # multiplier to scale the so that the mean of the discrete distribution is one
    return np.array(m)*s # SCALED VALUES

def init_ts_te(FA,LO):
    #ts=FA+np.random.exponential(.75, len(FA)) # exponential random starting point
    #tt=np.random.beta(2.5, 1, len(LO)) # beta random starting point
    #ts=FA+(.025*FA) #IMPROVE INIT
    #te=LO-(.025*LO) #IMPROVE INIT

    brl = FA-LO
    q = N/(brl+0.1)

    ts= FA+np.random.exponential(1./q,len(q))
    te= LO-np.random.exponential(1./q,len(q))

    if np.max(ts) > boundMax:
        ts[ts>boundMax] = np.random.uniform(FA[ts>boundMax],boundMax,len(ts[ts>boundMax])) # avoit init values outside bounds
    if np.min(te) < boundMin:
        te[te<boundMin] = np.random.uniform(boundMin,LO[te<boundMin],len(te[te<boundMin])) # avoit init values outside bounds
    if np.any(bound_te != bound_te[0]):
        te[te < bound_te] = np.random.uniform(bound_te[te < bound_te], LO[te < bound_te], len(te[te < bound_te]))
    #te=LO*tt
    if frac1==0: ts, te= FA,LO
    try:
        ts[SP_not_in_window] = boundMax
        te[EX_not_in_window] = boundMin
    except(NameError): pass
    return ts, te

def init_BD(n):
    #return np.repeat(0.5,max(n-1,1))
    return np.random.exponential(.2, np.maximum(n-1,1))+.1

def init_times(root_age,time_framesL,time_framesM,tip_age):
    timesL=np.linspace(root_age,tip_age,time_framesL+1)
    timesM=np.linspace(root_age,tip_age,time_framesM+1)
    timesM[1:time_framesM] +=1
    timesL[time_framesL] =0
    timesM[time_framesM] =0
    #print timesL,timesM
    #ts,te=sort(ts)[::-1],sort(te)[::-1]
    #indL = np.linspace(len(ts),0,time_framesL).astype(int)
    #indM = np.linspace(len(te[te>0]),0,time_framesM).astype(int)
    #
    #print indL[1:time_framesL], indM[1:time_framesM]
    #print ts[indL[1:time_framesL]],te[indM[1:time_framesM]]
    #b= [0]+ list(te[indM[1:time_framesM]])+[max(ts)]
    #print np.histogram(ts,bins=b)
    #quit()

    return timesL, timesM

def init_q_rates(): # p=1 for alpha1=alpha2
    return array([np.random.uniform(.5,1),np.random.uniform(0.25,1)]),np.zeros(3)

########################## UPDATES ######################################
def update_parameter(i, m, M, d, f):
    #d=fabs(np.random.normal(d,d/2.)) # variable tuning prm
    if i>0 and np.random.random()<=f:
        ii = i+(np.random.random()-.5)*d
        if ii<m: ii=(ii-m)+m
        if ii>M: ii=(M-(ii-M))
        if ii<m: ii=i
    else: ii=i
    return ii

def update_parameter_normal(i, m, M, d):
    if d>0: ii = np.random.normal(i,d)
    else: ii=i
    if ii<m: ii=(ii-m)+m
    if ii>M: ii=(M-(ii-M))
    if ii<m: ii=i
    return ii

def update_multiplier_proposal(i,d):
    S=shape(i)
    u = np.random.uniform(0,1,S)
    l = 2*log(d)
    m = exp(l*(u-.5))
    ii = i * m
    return ii, sum(log(m))

def update_rates_sliding_win(L,M,tot_L,mod_d3):
    Ln=zeros(len(L))
    Mn=zeros(len(M))
    #if random.random()>.5:
    for i in range(len(L)): Ln[i]=update_parameter(L[i],0, inf, mod_d3, f_rate)
    #else:
    for i in range(len(M)): Mn[i]=update_parameter(M[i],0, inf, mod_d3, f_rate)
    #Ln,Mn=Ln * scale_factor/tot_L , Mn * scale_factor/tot_L
    return Ln,Mn, 1

def update_parameter_normal_vec(oldL,d,f=.25, float_prec_f=np.float64):
    S = oldL.shape
    ii = np.random.normal(0,d,S)
    ff = np.random.binomial(1,np.min([f, 1]),S)
    # avoid no update being performed at all
    if np.sum(ff) == 0:
        up = np.random.randint(oldL.size, size=1)
        row, col = np.unravel_index(up, ff.shape)
        ff[row, col] = 1
    # print(np.sum(ff), S, f)
    s= oldL + float_prec_f(ii*ff)
    return s


def update_rates_multiplier(L,M,tot_L,mod_d3):
    if use_ADE_model == 0 and use_Death_model == 0:
        # UPDATE LAMBDA
        S=np.shape(L)
        #print L, S
        ff=np.random.binomial(1,f_rate,S)
        #print ff
        d=1.2
        u = np.random.uniform(0,1,S)
        l = 2*log(mod_d3)
        m = exp(l*(u-.5))
        m[ff==0] = 1.
        newL = L * m
        U=sum(log(m))
    else: U,newL = 0,L

    # UPDATE MU
    if use_Birth_model == 0:
        S=np.shape(M)
        ff=np.random.binomial(1,f_rate,S)
        d=1.2
        u = np.random.uniform(0,1,S) #*np.rint(np.random.uniform(0,f,S))
        l = 2*log(mod_d3)
        m = exp(l*(u-.5))
        m[ff==0] = 1.
        newM = M * m
        U+=sum(log(m))
    else: 
        newM = M
    return newL,newM,U

def update_q_multiplier(q,d=1.1,f=0.75):
    S=np.shape(q)
    ff=np.random.binomial(1,f,S)
    u = np.random.uniform(0,1,S)
    l = 2*log(d)
    m = exp(l*(u-.5))
    m[ff==0] = 1.
    new_q = q * m
    U=sum(log(m))
    return new_q,U

def update_times(times, max_time,min_time, mod_d4,a,b):
    rS= times+zeros(len(times))
    rS[0]=max_time
    if np.random.random()< 1.5:
        for i in range(a,b): rS[i]=update_parameter(times[i],min_time, max_time, mod_d4, 1)
    else:
        i = np.random.choice(list(range(a,b)))
        rS[i]=update_parameter(times[i],min_time, max_time, mod_d4*3, 1)
    y=sort(-rS)
    y=-1*y
    return y

def update_ts_te(ts, te, d1, bound_ts, bound_te, sample_extinction=1):
    tsn, ten= zeros(len(ts))+ts, zeros(len(te))+te
    f1=np.random.randint(1,frac1) #int(frac1*len(FA)) #-np.random.randint(0,frac1*len(FA)-1))
    ind=np.random.choice(SP_in_window,f1) # update only values in SP/EX_in_window
    tsn[ind] = ts[ind] + (np.random.uniform(0,1,len(ind))-.5)*d1
    M = bound_ts#np.inf #boundMax
    tsn[tsn > M] = M[tsn > M] - (tsn[tsn > M] - M[tsn > M])
    m = FA
    tsn[tsn<m]=(m[tsn<m]-tsn[tsn<m])+m[tsn<m]
    tsn[tsn>M] = ts[tsn>M]
    if sample_extinction:
        ind=np.random.choice(EX_in_window,f1)
        ten[ind] = te[ind] + (np.random.uniform(0,1,len(ind))-.5)*d1
        M = LO
        ten[ten>M]=M[ten>M]-(ten[ten>M]-M[ten>M])
        m = bound_te#0 #boundMin
        ten[ten < m] = (m[ten < m] - ten[ten < m]) + m[ten < m]
        ten[ten>M] = te[ten>M]
        ten[LO==0]=0                                     # indices of LO==0 (extant species)
    S= tsn-ten
    if np.min(S)<=0: print(S)
    tsn[SP_not_in_window] = np.maximum(boundMax, np.max(tsn[SP_in_window]))
    ten[EX_not_in_window] = np.minimum(boundMin, np.min(ten[EX_in_window]))
    return tsn,ten


def update_ts_te_indicator(ts, te, d1, sample_extinction=1):
    tsn, ten= zeros(len(ts))+ts, zeros(len(te))+te
    f1=np.random.randint(1,frac1) #int(frac1*len(FA)) #-np.random.randint(0,frac1*len(FA)-1))
    ind=np.random.choice(SP_in_window,f1) # update only values in SP/EX_in_window
    tsn[ind] = ts[ind] + (np.random.uniform(0,1,len(ind))-.5)*d1
    M = np.inf #boundMax
    tsn[tsn>M]=M-(tsn[tsn>M]-M)
    m = FA + 0
    tsn[tsn<m]=(m[tsn<m]-tsn[tsn<m])+m[tsn<m]
    tsn[tsn>M] = ts[tsn>M]
    if sample_extinction:
        # if np.random.random() < 0.5:
        ind=np.random.choice(EX_in_window,f1)
        ten[ind] = LO[ind] + LO[ind] * np.random.uniform(0, 0.1)
        M = LO
        ten[ten>M]=M[ten>M]-(ten[ten>M]-M[ten>M])
        m = 0 #boundMin
        ten[ten<m]=(m-ten[ten<m])+m
        ten[ten>M] = te[ten>M]
        ten[LO==0]=0                                     # indices of LO==0 (extant species)
        # else:
        # add rnd indicators
        p_vec = np.exp(LO) / np.sum(np.exp(LO))
        p_vec[LO == 0] = 0
        p_vec = (p_vec / np.sum(p_vec)) ** 0.1
        # print(p_vec / np.sum(p_vec))
        ind_swap = np.random.choice(range(len(ten)), p = p_vec / np.sum(p_vec), size=1, replace=False)
        ten[ind_swap] *=0
            # ten[ind_swap] = 0
    S= tsn-ten
    # if min(S)<=0: print(S)
    tsn[SP_not_in_window] = np.maximum(boundMax, np.max(tsn[SP_in_window]))
    ten[EX_not_in_window] = np.minimum(boundMin, np.min(ten[EX_in_window]))
    return tsn,ten


def update_ts_te_tune(ts, te, d1, d2, FA, LO, bound_ts, bound_te, sample_extinction=1):
    tsn, ten = np.zeros(len(ts)) + ts, np.zeros(len(te)) + te
    if sample_extinction == 0:
        ind = np.random.choice(SP_in_window, 1) # update only values in SP/EX_in_window
        tsn[ind] = ts[ind] + (np.random.uniform(0, 1, len(ind)) - .5) * d1[ind]
        M = bound_ts#np.inf #boundMax
        tsn[tsn > M] = M[tsn > M] - (tsn[tsn > M] - M[tsn > M])
        m = FA
        tsn[tsn < m] = (m[tsn < m] - tsn[tsn < m]) + m[tsn < m]
        tsn[tsn > M] = ts[tsn > M]
    else:
        ind = np.random.choice(EX_in_window[LO > 0.0], 1)
        ten[ind] = te[ind] + (np.random.uniform(0, 1, len(ind)) - .5) * d2[ind]
        M = LO
        ten[ten > M] = M[ten > M]-(ten[ten > M] - M[ten > M])
        m = bound_te #0 #boundMin
        ten[ten < m] = (m[ten < m] - ten[ten < m]) + m[ten < m]
        ten[ten > M] = te[ten > M]
        ten[LO == 0] = 0                                     # indices of LO==0 (extant species)
    S = tsn - ten
    if np.min(S) <= 0:
        print(S)
    tsn[SP_not_in_window] = np.maximum(boundMax, np.max(tsn[SP_in_window]))
    ten[EX_not_in_window] = np.minimum(boundMin, np.min(ten[EX_in_window]))
    return tsn,ten


#### GIBBS SAMPLER S/E
def draw_se_gibbs(fa,la,q_rates_L,q_rates_M,q_times, bound_ts, bound_te, tsA, teA):
    t = np.sort(np.array([fa, la] + list(q_times)))[::-1]
    # sample ts
    prior_to_fa = np.arange(len(q_times))[q_times>fa]
    tfa = (q_times[prior_to_fa]-fa)[::-1] # time since fa
    qfa = q_rates_L[prior_to_fa][::-1] # rates before fa
    
    i, attempt=0, 0
    # print( "QFA", qfa, tfa)
    while True:
        ts_temp=0
        # for i in range(len(qfa)):
        # print(attempt, i, fa, tfa, qfa)
        q = qfa[i]
        deltaT = np.random.exponential(1. / q)
        ts_temp = np.minimum(ts_temp+deltaT, tfa[i])
        if ts_temp < tfa[i] and (ts_temp + fa) < bound_ts:
            # print("new TS",(ts_temp + fa), bound_ts, "fa", fa, i, "ts_temp", ts_temp, tfa[i])
            break
        i+=1
        attempt += 1
        
        if i == len(qfa):
            i = 0 # try again
        # if attempt == 100:
        #     # print("timeout TE_gibbs", te_temp, tla[i], (la - te_temp), la, bound_te)
        #     te_temp = np.random.uniform(0, np.abs(bound_te - la)) # np.abs(la - te) #
        #     # quit()
        #
        #     break
        
        if attempt > 100:
            # print("timeout TS_gibbs", ts_temp, fa, bound_ts, ts_temp < tfa[i], (ts_temp + fa) < bound_ts)
            ts_temp = np.abs(tsA - fa) # reset to previous value
            break
            #   
    # break
        
            
        attempt += 1

    ts = ts_temp + fa
    # print("TS:", ts, fa, ts_temp, bound_ts)
    #print q_times
    #print la
    if ts > bound_ts:
        print("WTF", "new TS", (ts_temp + fa), bound_ts, "fa", fa, i, "ts_temp", ts_temp, "actual ts",  ts)
        quit()
        
    if la>0:
        # sample te
        after_la = np.arange(len(q_times))[q_times<la]
        tla = (la-q_times[after_la]) # time after la
        qla = q_rates_M[after_la-1] # rates after la
        # print( "QLA", qla, tla)
        te_temp=0
        i, attempt=0, 0
        while True:
            q = qla[i]
            deltaT = np.random.exponential(1. / q)
            te_temp = np.minimum(te_temp + deltaT, tla[i])
            #print attempt,i,te_temp,len(qla)
            if te_temp < tla[i] and (la - te_temp) > bound_te:
                break
            i+=1
            attempt += 1
            if i == len(qla):
                i = 0 # try again
            if attempt == 100:
                # print("timeout TE_gibbs", te_temp, tla[i], (la - te_temp), la, bound_te)
                # te_temp = np.random.uniform(0, np.abs(bound_te - la)) #
                te_temp = np.abs(la - teA) # reset to previous valu
                # quit()
                
                break

        te = la - te_temp
        #print "TE:", te
    else:
        te = 0
    return (ts,te)

def gibbs_update_ts_te(q_rates_L,q_rates_M,q_time_frames, bound_ts, bound_te, tsA, teA):
    #print q_rates,q_time_frames
    q_times= q_time_frames+0
    q_times[0] = np.inf
    new_ts = []
    new_te = []
    for sp_indx in range(len(FA)):
        #print "sp",sp_indx
        s,e = draw_se_gibbs(FA[sp_indx], LO[sp_indx],
                            q_rates_L, q_rates_M, q_times, 
                            bound_ts[sp_indx], bound_te[sp_indx],
                            tsA[sp_indx], teA[sp_indx])
        new_ts.append(s)
        new_te.append(e)
    return np.array(new_ts), np.array(new_te)


def gibbs_update_ts_te_bdnn(q_rates,sp_rates_L, sp_rates_M, q_time_frames):
    #print q_rates,q_time_frames
    q_times= q_time_frames + 0
    q_times[0] = np.inf
    new_ts = []
    new_te = []
    for sp_indx in range(0,len(FA)):
        #print "sp",sp_indx
        s,e = draw_se_gibbs(FA[sp_indx],
                            LO[sp_indx],
                            q_rates + sp_rates_L[sp_indx],
                            q_rates + sp_rates_M[sp_indx],
                            q_times)
        new_ts.append(s)
        new_te.append(e)
    return np.array(new_ts), np.array(new_te)


def make_tste_tune_obj(LO, bound_te, d1):
    n_taxa = len(LO)
    d1_ts = np.repeat(d1, n_taxa)
    d1_te = np.repeat(d1, n_taxa)
    b = 2 * (LO - bound_te)
    exceeds_LO = d1_te > b
    d1_te[exceeds_LO] = b[exceeds_LO]
    tste_tune_obj = np.zeros((n_taxa, 6)) # attempts, successes, and acceptance ratio for ts and te
    return d1_ts, d1_te, tste_tune_obj


def tune_tste_windows(d1_ts, d1_te,
                      LO, bound_te, tste_tune_obj,
                      it, tune_T_schedule,
                      updated, se_updated,
                      accepted=0, target=[0.2, 0.4]):

    se_updated *= 3
    n_taxa = len(LO)
    update_interval_frac = 0.5 * n_taxa * tune_T_schedule[1]

    # Keep track of acceptance ratio
    tste_tune_obj[updated, 0 + se_updated] += 1
    tste_tune_obj[updated, 1 + se_updated] += accepted
    tste_tune_obj[updated, 2 + se_updated] = tste_tune_obj[updated, 1 + se_updated] / tste_tune_obj[updated, 0 + se_updated]

    exceeds_tuning_interval = tste_tune_obj[:, 0 + se_updated] > tune_T_schedule[1]
    if se_updated == 3:
        exceeds_tuning_interval[LO == 0] = False
    any_exceeds_tuning_interval = np.any(exceeds_tuning_interval)

    # Tune window sizes
    if any_exceeds_tuning_interval and it < tune_T_schedule[0] and it > 0.1 * tune_T_schedule[0]:
        u = np.isin(np.arange(len(d1_ts)), updated)
        too_low = np.logical_and(tste_tune_obj[:, 2 + se_updated] < target[0], u)
        too_high = np.logical_and(tste_tune_obj[:, 2 + se_updated] > target[1], u)

        # decrease window size
        if se_updated == 0:
            d1_ts[too_low] = 0.9 * d1_ts[too_low]
        else:
            d1_te[too_low] = 0.9 * d1_te[too_low]

        # increase window size
        if se_updated == 0:
            d1_ts[too_high] = 1.1 * d1_ts[too_high]
        else:
            d1_te[too_high] = 1.1 * d1_te[too_high]

        b = 2 * (LO - bound_te)
        exceeds_LO = d1_te > b
        d1_te[exceeds_LO] = b[exceeds_LO]

    # Reset acceptance ratio calculation
    if any_exceeds_tuning_interval and it > 0.1 * tune_T_schedule[0]:
        tste_tune_obj[exceeds_tuning_interval, 0 + se_updated] = 0
        tste_tune_obj[exceeds_tuning_interval, 1 + se_updated] = 0
        tste_tune_obj[exceeds_tuning_interval, 2 + se_updated] = 0

    return d1_ts, d1_te, tste_tune_obj


def set_bound_se(b_ts, b_te, b_se, taxa):
    b_se = b_se[np.isin(b_se[:, 0], taxa), :]
    b_se = b_se[np.argsort(b_se[:, 0]), :]

    not_na = b_se[:, 1] != 'NA'
    max_ts = b_se[not_na, 1].astype(float)
    max_ts_taxa = b_se[not_na, 0]
    b_ts[np.isin(taxa, max_ts_taxa)] = max_ts

    not_na = b_se[:, 2] != 'NA'
    min_te = b_se[not_na, 2].astype(float)
    min_te_taxa = b_se[not_na, 0]
    b_te[np.isin(taxa, min_te_taxa)] = min_te

    return b_ts, b_te



def seed_missing(x,m,s): # assigns random normally distributed trait values to missing data
    return np.isnan(x)*np.random.normal(m,s)+np.nan_to_num(x)

def cond_alpha_proposal(hp_gamma_shape,hp_gamma_rate,current_alpha,k,n):
    z = [current_alpha + 1.0, float(n)]
    f = np.random.dirichlet(z,1)[0]
    eta = f[0]
    u = np.random.uniform(0,1,1)[0]
    x = (hp_gamma_shape + k - 1.0) / ((hp_gamma_rate - np.log(eta)) * n)
    if (u / (1.0-u)) < x: new_alpha = np.random.gamma( (hp_gamma_shape+k), (1./(hp_gamma_rate-np.log(eta))) )
    else: new_alpha = np.random.gamma( (hp_gamma_shape+k-1.), 1./(hp_gamma_rate-np.log(eta)) )
    return new_alpha

def get_post_sd(N,HP_shape=2,HP_rate=2,mean_Norm=0): # get sd of Normal from sample N and hyperprior G(a,b)
    n= len(N)
    G_shape = HP_shape + n*.5
    G_rate  = HP_rate  + sum((N-mean_Norm)**2)*.5
    tau = np.random.gamma(shape=G_shape,scale=1./G_rate)
    sd= sqrt(1./tau)
    return(sd)

########################## PRIORS #######################################
try:
    scipy.stats.gamma.logpdf(1, 1, scale=1./1,loc=0)
    def prior_gamma(L,a,b):
        if hasFoundPyRateC:
            return PyRateC_getLogGammaPDF(L, a, 1./b)#scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0)
        else:
            return scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0)
    def prior_normal(L,sd, mu=0):
        return scipy.stats.norm.logpdf(L,loc=mu,scale=sd)
    def prior_cauchy(x,s):
        return scipy.stats.cauchy.logpdf(x,scale=s,loc=0)
except(AttributeError): # for older versions of scipy
    def prior_gamma(L,a,b):
        return (a-1)*log(L)+(-b*L)-(log(b)*(-a)+ log(gamma(a)))
    def prior_normal(L,sd, mu=0):
        if mu != 0:
            sys.exit("Scipy required")
        else:             
            return -(L**2/(2*sd**2)) - log(sd*sqrt(2*np.pi))
    def prior_cauchy(x,s):
        return -log(np.pi*s * (1+ (x/s)**2))

def prior_times_frames(t, root, tip_age,a): # un-normalized Dirichlet (truncated)
    diff_t, min_t = abs(np.diff(t)), np.min(t)
    if np.min(diff_t)<=min_allowed_t: return -inf
    elif (min_t<=tip_age+min_allowed_t) and (min_t>0): return -inf
    else:
        t_rel=diff_t/root
        return (a-1)*log(t_rel)


def get_min_diffTime(times):
    diff_t = abs(np.diff(times))
    if fix_edgeShift==1: # min and max bounds
        diff_t = diff_t[1:-1]
    elif fix_edgeShift==2: # max bound
        diff_t = diff_t[1]
    elif fix_edgeShift==3: # min bound
        diff_t = diff_t[-1]
    return np.min(diff_t)


def prior_sym_beta(x,a):
    # return log(x)*(a-1)+log(1-x)*(a-1) # un-normalized beta
    return scipy.stats.beta.logpdf(x, a,a)

def prior_root_age(root, max_FA, l): # exponential (truncated)
    l=1./l
    if root>=max_FA: return log(l)-l*(root)
    else: return -inf

def prior_uniform(x,m,M):
    if x>m and x<M: return log(1./(M-m))
    else: return -inf

def G_density(x,a,b):
    #return (1./b)**a * 1./gamma(a) * x**(a-1) *exp(-(1./b)*x)
    return scipy.stats.gamma.pdf(x, a, scale=1./b,loc=0)

def logPERT4_density(M,m,a,b,x): # relative 'stretched' LOG-PERT density: PERT4 * (s-e)
    return log((M-x)**(b-1) * (-m+x)**(a-1)) - log ((M-m)**4 * f_beta(a,b))

def PERT4_density(M,m,a,b,x):  # relative 'stretched' PERT density: PERT4 * (s-e)
    return ((M-x)**(b-1) * (-m+x)**(a-1)) /((M-m)**4 * f_beta(a,b))

def logPERT4_density5(M,m,a,b,x): # relative LOG-PERT density: PERT4
    return log((M-x)**(b-1) * (-m+x)**(a-1)) - log ((M-m)**5 * f_beta(a,b))

########################## LIKELIHOODS ##################################
def HPBD1(timesL,timesM,L,M,T,s):
    return sum(prior_cauchy(L,s[0]))+sum(prior_cauchy(M,s[1]))

def HPBD2(timesL,timesM,L,M,T,s):
    return sum(prior_gamma(L,L_lam_r,s[0])) + sum(prior_gamma(M,M_lam_r,s[1]))

def HPBD3(timesL,timesM,L,M,T,s):
    def pNtvar(arg):
        T=arg[0]
        L=arg[1]
        M=arg[2]
        N=arg[3]
        Dt=-np.diff(T)
        r_t = (L - M)*Dt
        Beta=  sum(exp((L - M)*Dt))
        Alpha= sum(L*exp((L - M)*Dt))
        lnBeta=  log(sum(exp((L - M)*Dt)))
        lnAlpha= log(sum(L*exp((L - M)*Dt)))
        #P   = (Beta/(Alpha*(1+Alpha))) *    (Alpha/(1+Alpha))**N
        if N>0: lnP = (lnBeta-(lnAlpha+log(1+Alpha))) + (lnAlpha-log(1+Alpha))*N
        else:    lnP = log((1+Alpha-Beta)/(Alpha*(1+Alpha)))
        return lnP

    ### HYPER-PRIOR BD ###
    n0=1
    timesMtemp = np.zeros(len(timesM)) + timesM
    timesMtemp[1:len(timesM)-1] = timesMtemp[1:len(timesM)-1] +0.0001
    #if len(timesM)>2:
    all_t_frames=sort(np.append(timesL, timesMtemp[1:len(timesMtemp)-1] ))[::-1] # merge time frames

    #all_t_frames=sort(np.append(timesL, timesM[1:-1]+.001 ))[::-1] # merge time frames
    #else: all_t_frames=sort(np.append(timesL, timesM[1:-1] ))[::-1] # merge time frames
    sL=(np.in1d(all_t_frames,timesL[1:-1])+0).nonzero()[0] # indexes within 'all_t_frames' of shifts of L
    sM=(np.in1d(all_t_frames,timesMtemp[1:-1])+0).nonzero()[0] # indexes within 'all_t_frames' of shifts of M
    sL[(sL-1>len(M)-1).nonzero()]=len(M)
    sM[(sM-1>len(L)-1).nonzero()]=len(L)

    nL=zeros(len(all_t_frames)-1)
    nM=zeros(len(all_t_frames)-1)

    Ln=insert(L,sM,L[sM-1]) # l rates for all_t_frames
    Mn=insert(M,sL,M[sL-1]) # m rates for all_t_frames

    return pNtvar([all_t_frames,Ln,Mn,tot_extant])

# BIRTH-DEATH MODELS

#---- reconstructed BD
def p0(t,l,m,rho):
    return 1 - (rho*(l-m) / (rho*l + (l*(1-rho) -m) * exp(-(l-m)*t)))

def p1(t,l,m,rho):
    return  rho*(l-m)**2 * exp(-(l-m)*t)/(rho*l + (l*(1-rho) -m)*exp(-(l-m)*t))**2

def treeBDlikelihood(x,l,m,rho,root=1,survival=1):
    #_ lik = (root + 1) * log(p1(x[0], l, m, rho))
    #_ for i in range(1, len(x)) :
    #_     lik = lik + log(l * p1(x[i], l, m, rho))
    #_ if survival == 1:
    #_    lik = lik - (root + 1) * log(1 - p0(x[0], l, m, rho))
    #_ return lik
    lik1= (root + 1) * log(p1(x[0], l, m, rho))
    lik2= sum(log(l * p1(x[1:], l, m, rho)))
    lik3= - (root + 1) * log(1 - p0(x[0], l, m, rho))
    return lik1+lik2+lik3

#----

def BD_lik_discrete_trait(arg):
    [ts,te,L,M]=arg
    S = ts-te
    lik0 =  sum(log(L)*lengths_B_events )    #
    lik1 = -sum(L*sum(S))                   # assumes that speiation can arise from any trait state
    #lik1 = -sum([L[i]*sum(S[ind_trait_species==i]) for i in range(len(L))])
    lik2 =  sum(log(M)*lengths_D_events)                                        # Trait specific extinction
    lik3 = -sum([M[i]*sum(S[ind_trait_species==i]) for i in range(len(M))]) # only species with a trait state can go extinct
    return sum(lik0+lik1+lik2+lik3)

def BD_lik_discrete_trait_continuous(arg):
    [ts,te,L,M,cov_par]=arg
    S = ts-te
    # speciation
    sp_rate=exp(log(L)+cov_par[0]*(con_trait-parGAUS[0]))
    lik01 = sum(log(sp_rate)) + sum(-sp_rate*S) #, cov_par
    # extinction
    ex_rate = exp(log(M[ind_trait_species])+cov_par[0]*(con_trait-parGAUS[0]))
    lik2 =  sum(log(ex_rate[te>0]))   # only count extinct species
    lik3 = -sum([M[i]*sum(S[ind_trait_species==i]) for i in range(len(M))]) # only species with a trait state can go extinct
    return sum(lik01+lik2+lik3)


def BPD_lik_vec_times(arg):
    [ts,te,time_frames,L,M]=arg
    if fix_SE == 0 or fix_Shift == 0:
        BD_lik = 0
        B = sort(time_frames)+0.000001 # add small number to avoid counting extant species as extinct
        ss1 = np.histogram(ts,bins=B)[0][::-1]
        ee2 = np.histogram(te,bins=B)[0][::-1]

        for i in range(len(time_frames)-1):
            up, lo = time_frames[i], time_frames[i+1]
            len_sp_events=ss1[i]
            if i==0: len_sp_events = len_sp_events-no_starting_lineages
            len_ex_events=ee2[i]
            inTS = np.fmin(ts,up)
            inTE = np.fmax(te,lo)
            S    = inTS-inTE
            # speciation
            if use_poiD == 0:
                lik1 = log(L[i])*len_sp_events
                lik0 = -sum(L[i]*S[S>0]) # S < 0 when species outside up-lo range
            else:
                lik1 = log(L[i])*len_sp_events
                lik0 = -sum(L[i]*(up-lo)) # S < 0 when species outside up-lo range

            # extinction
            lik2 = log(M[i])*len_ex_events
            lik3 = -sum(M[i]*S[S>0]) # S < 0 when species outside up-lo range
            BD_lik += lik0+lik1+lik2+lik3
            #print "len",sum(S[S>0]),-sum(L[i]*S[S>0]), -L[i]*sum(S[S>0])
    else:
        lik0 =  log(L)*len_SS1
        lik1 = -(L* S_time_frame)
        lik2 =  log(M)*len_EE1
        lik3 = -(M* S_time_frame)
        BD_lik = lik0+lik1+lik2+lik3

    return BD_lik

def get_sp_indx_in_timeframe(ts, te, up, lo):
    return np.intersect1d((ts >= lo).nonzero()[0], (te <= up).nonzero()[0])

def get_sp_in_frame_br_length(ts,te,up,lo):
    # index species present in time frame
    n_all_inframe = np.intersect1d((ts >= lo).nonzero()[0], (te <= up).nonzero()[0])

    # tot br length within time frame
    n_t_ts,n_t_te=zeros(len(ts)),zeros(len(ts))

    n_t_ts[n_all_inframe]= ts[n_all_inframe]   # speciation events before time frame
    n_t_ts[(n_t_ts>up).nonzero()]=up           # for which length is accounted only from $up$ rather than from $ts$

    n_t_te[n_all_inframe]= te[n_all_inframe]   # extinction events in time frame
    n_t_te[np.intersect1d((n_t_te<lo).nonzero()[0], n_all_inframe)]=lo     # for which length is accounted only until $lo$ rather than to $te$

    # vector of br lengths within time frame  #(scaled by rho)
    n_S=((n_t_ts[n_all_inframe]-n_t_te[n_all_inframe])) #*rhos[n_all_inframe])
    return n_all_inframe, n_S


def BD_partial_lik(arg):
    [ts,te,up,lo,rate,par, cov_par,_]=arg
    # indexes of the species within time frame
    if par=="l": i_events=np.intersect1d((ts <= up).nonzero()[0], (ts > lo).nonzero()[0])
    else: i_events=np.intersect1d((te <= up).nonzero()[0], (te > lo).nonzero()[0])
    # index of extant/extinct species
    # extinct_sp=(te > 0).nonzero()[0]
    # present_sp=(te == 0).nonzero()[0]
    n_all_inframe, n_S = get_sp_in_frame_br_length(ts,te,up,lo)

    if cov_par !=0 and not cov_par is None: # covaring model: $r$ is vector of rates tranformed by trait
        r=exp(log(rate)+cov_par*(con_trait-parGAUS[0])) # exp(log(rate)+cov_par*(con_trait-mean(con_trait[all_inframe])))
        lik= sum(log(r[i_events])) + sum(-r[n_all_inframe]*n_S) #, cov_par
    else:           # constant rate model
        no_events = len(i_events)
        if par=="l" and up==max_age_fixed_ts:
            no_events = no_events-no_starting_lineages
        lik= log(rate)*no_events -rate*sum(n_S) #log(rate)*len(i_events) +sum(-rate*n_S)
    return lik

def BD_partial_lik_bounded(arg):
    [ts,te,up,lo,rate,par, cov_par,_]=arg
    # indexes of the species within time frame
    if par=="l":
        i_events=np.intersect1d((ts[SP_in_window] <= up).nonzero()[0], (ts[SP_in_window] > lo).nonzero()[0])
    else:
        i_events=np.intersect1d((te[EX_in_window] <= up).nonzero()[0], (te[EX_in_window] > lo).nonzero()[0])
    n_all_inframe, n_S = get_sp_in_frame_br_length(ts,te,up,lo)
    if cov_par !=0: # covaring model: $r$ is vector of rates tranformed by trait
        r=exp(log(rate)+cov_par*(con_trait-parGAUS[0])) # exp(log(rate)+cov_par*(con_trait-mean(con_trait[all_inframe])))
        lik= sum(log(r[i_events])) + sum(-r[n_all_inframe]*n_S) #, cov_par
    else:           # constant rate model
        #print par, len(i_events), len(te)
        lik= log(rate)*len(i_events) -rate*sum(n_S) #log(rate)*len(i_events) +sum(-rate*n_S)
    return lik

def BD_partial_lik_lithology(arg):
    [ts,te,up,lo,rate,par, cov_par,_]=arg
    n_all_inframe, n_S = get_sp_in_frame_br_length(ts,te,up,lo)
    # print(par, "n_all_inframe, n_S", len(n_all_inframe), np.sum(n_S))
    # indexes of the species within time frame
    ID_focal_lithology, ID_other_lithology, 
    
    if par=="l":
        i_events=np.intersect1d((ts[SP_in_window] <= up).nonzero()[0], (ts[SP_in_window] > lo).nonzero()[0])
        # br length only computed based on other lithologies
        n_all_inframe, n_S = get_sp_in_frame_br_length(ts[ID_other_lithology],te[ID_other_lithology],up,lo)
    else:
        i_events=np.intersect1d((te[EX_in_window] <= up).nonzero()[0], (te[EX_in_window] > lo).nonzero()[0])
        # br length only computed based on focal lithology
        n_all_inframe, n_S = get_sp_in_frame_br_length(ts[ID_focal_lithology],te[ID_focal_lithology],up,lo)
    
    # print(par, "n_all_inframe, n_S", len(n_all_inframe), np.sum(n_S))
    # print(len(ts[SP_not_in_window]), len(te[ID_focal_lithology]), len(i_events))

    if cov_par !=0: # covaring model: $r$ is vector of rates tranformed by trait
        r=exp(log(rate)+cov_par*(con_trait-parGAUS[0])) # exp(log(rate)+cov_par*(con_trait-mean(con_trait[all_inframe])))
        lik= sum(log(r[i_events])) + np.sum(-r[n_all_inframe]*n_S) #, cov_par
    else:           # constant rate model
        #print par, len(i_events), len(te)
        lik= log(rate)*len(i_events) -rate * np.sum(n_S) #log(rate)*len(i_events) +sum(-rate*n_S)
    return lik



# BDNN model
def relu_f(z):
    z[z < 0] = 0
    return z

def leaky_relu_f(z, prm=0.001):
    z[z < 0] = prm * z[z < 0]
    return z

def swish_f(z):
    # https://arxiv.org/abs/1710.05941
    z = z * (1 + np.exp(-z))**(-1)
    return z

def sigmoid_f(z):
    return 1 / (1 + np.exp(-z))
    
def sigmoid_rate(z):
    pr = 1 / (1 + np.exp(-z))
    rate = - np.log(1 - pr)
    return rate
    
def tanh_f(z):
    return np.tanh(z)

def tanh_f_approx(z):
#    print('z dtype\n', z.dtype)
#    print('z\n', z)
    tanh = 1.0 - ( 2.0 / ( np.exp(2.0 * z) + 1.0 ) )
    return tanh

def softPlus(z):
#    return np.log(np.exp(z) + 1)
    return np.logaddexp(0, z) # overflow safe softPlus

def mean_softPlus3D(z):
    # Is this needed when we have time-variable predictors e.g. sediment availability? 
    # Shouldn't we normalize the whole 3D array to a mean of 1 using mean_softPlus1D() or a sum of 1?
    z_prime = softPlus(z)
    return z_prime * (z_prime[0].size / np.sum(z_prime, axis=(-2, -1)))[:, np.newaxis, np.newaxis]

def mean_softPlus1D(z):
    z_prime = softPlus(z)
    return z_prime * (z_prime.size / np.sum(z_prime))

def expFun(z):
    return np.exp(z)


def init_NN_output(x, w, float_prec_f=np.float64):
    n_layers = len(w)
    x_shape = x.shape
    xs = [x_shape[0]]
    if len(x_shape) == 3:
        xs.append(x_shape[1])
    nn_out = []
    for i in range(n_layers):
        init_shape = tuple(xs + [w[i].shape[0]])
        nn_out.append(float_prec_f(np.zeros(init_shape)))
    return nn_out


def MatrixMultiplication(x1,x2):
    if x1.shape[1] == x2.shape[1]:
        #z1 = np.einsum('nj,ij->ni', x1, x2)
        z1 = np.dot(x1, x2.T)
    else:
        #z1 = np.einsum('nj,ij->ni', x1, x2[:, 1:]) # w/ bias node
        z1 = np.dot(x1, x2[:, 1:].T)
        z1 += x2[:, 0].T
    return z1

def MatrixMultiplication3D(x1, x2, bias_node_idx=[0]):
    if x1.shape[-1] == x2.shape[1]:
#        z1 = np.einsum('tnj,ij->tni', x1, x2, optimize=True)
#        z1 = np.tensordot(x1, x2.T, axes=([-1], [0]))
        z1 = np.tensordot(x1, x2, axes=([-1], [-1])) # Column-major order should be faster for large arrays
#        z1 = np.array([np.dot(x1[i], x2.T) for i in range(len(x1))])
    else:
#        z1 = np.einsum('tnj,ij->tni', x1, x2[:, 1:], optimize=True) # w/ bias node
#        z1 = np.tensordot(x1, x2[:, (bias_node_idx[-1] + 1):].T, axes=([-1], [0]))
        z1 = np.tensordot(x1, x2[:, (bias_node_idx[-1] + 1):], axes=([-1], [-1]))
#        z1 = np.array([np.dot(x1[i], x2[:, (bias_node_idx[-1] + 1):].T) for i in range(len(x1))])
        z1 += x2[:, bias_node_idx[-1]].T
    return z1

def get_rate_BDNN(t_reg, x, w, act_f, out_act_f, apply_reg):
    tmp = x+0
    for i in range(len(w)-1):
        tmp = act_f(MatrixMultiplication(tmp, w[i]))
    
    tmp = MatrixMultiplication(tmp, w[i+1])
    # output
    rates = out_act_f(tmp).flatten() + small_number
    reg_rates, denom = get_reg_rates(rates, t_reg, apply_reg)
    return reg_rates, denom

def get_reg_rates(rates, t_reg, apply_reg):
    r_tmp = rates ** t_reg
    denom = np.nanmean(r_tmp[apply_reg]) / np.nanmean(rates[apply_reg])
    reg_r = r_tmp / denom
    reg_r[~apply_reg] = rates[~apply_reg]
    return reg_r, denom


def update_NN(x, w, nnA, rnd_layer, act_f, bias_node_idx):
    nn = copy_lib.deepcopy(nnA)
    if rnd_layer != -1:
        n_layers = len(w) - 1
        if rnd_layer == 0:
            nn[0] = act_f(MatrixMultiplication3D(x, w[0]))
            for i in range(1, n_layers):
                nn[i] = act_f(MatrixMultiplication3D(nn[i - 1], w[i], bias_node_idx))
        elif rnd_layer < n_layers:
            for i in range(rnd_layer, len(w)-1):
                nn[i] = act_f(MatrixMultiplication3D(nn[i - 1], w[i], bias_node_idx))
        
        nn[-1] = MatrixMultiplication3D(nn[-2], w[-1], bias_node_idx)
    return nn


def get_unreg_rate_BDNN_3D(x, w, nnA, act_f, out_act_f, apply_reg=True, bias_node_idx=[0], fix_edgeShift=0, rnd_layer=0):
    if nnA is None:
        nn = nnA
        tmp = x + 0
        for i in range(len(w)-1):
            tmp = act_f(MatrixMultiplication3D(tmp, w[i], bias_node_idx))
        tmp = MatrixMultiplication3D(tmp, w[i+1], bias_node_idx)
        tmp = np.squeeze(tmp).T
    else:
        nn = update_NN(x, w, nnA, rnd_layer, act_f, bias_node_idx)
        tmp = np.squeeze(nn[-1]).T
    
    # add bias node values for the edge bins
    if fix_edgeShift > 0:
        w_add = np.tile(w[-1][:, bias_node_idx[0:-1]].reshape(-1), tmp.shape[0])
        tmp[~apply_reg] = w_add
    
    # output
    rates = out_act_f(tmp)
    rates += small_number
    
    return rates, nn


# def get_reg_rate_BDNN_3D(rates, t_reg, bin_ts_te=None):
#     normalize_factor = None
#     if rates.ndim == 2:
#         # set rate to NA before ts and after te
#         for i in range(rates.shape[0]):
#             rates[i, :bin_ts_te[i, 0]] = np.nan
#             rates[i, bin_ts_te[i, 1]:] = np.nan
#     reg_rates, denom = get_reg_rates(rates, t_reg)
#
#     return reg_rates, denom, normalize_factor


def get_rate_BDNN_3D(t_reg, x, w, nnA, act_f, out_act_f, apply_reg, bias_node_idx=[0], fix_edgeShift=0, rnd_layer=0):
    rates, nn = get_unreg_rate_BDNN_3D(x, w, nnA, act_f, out_act_f, apply_reg, bias_node_idx, fix_edgeShift, rnd_layer)
    reg_rates, denom = get_reg_rates(rates, t_reg, apply_reg)

    return reg_rates, denom, nn


def BDNN_likelihood(arg):
    [ts,te,trait_tbl,rate_l,rate_m,cov_par, apply_reg] = arg
    nn_prm_lam, nn_prm_mu = cov_par[0], cov_par[1]
    s = ts - te
    lam, _ = get_rate_BDNN(rate_l, trait_tbl[0], nn_prm_lam, hidden_act_f, out_act_f, apply_reg)
    mu, _ = get_rate_BDNN(rate_m, trait_tbl[1], nn_prm_mu, hidden_act_f, out_act_f, apply_reg)
    likL = np.sum(np.log(lam) - lam*s)
    likM = np.sum(np.log(mu[te>0])) - np.sum(mu*s)
    return likL + likM

def BDNN_fast_likelihood(arg):
    [ts,te,lam,mu] = arg
    s = ts - te
    likL = np.sum(np.log(lam) - lam*s)
    likM = np.sum(np.log(mu[te>0])) - np.sum(mu*s)
    return likL + likM

def BDNN_partial_lik(arg):
    [ts,te,up,lo,rate,par, nn_prm,indx, apply_reg]=arg
    # indexes of the species within time frame
    if par=="l": i_events=np.intersect1d((ts <= up).nonzero()[0], (ts > lo).nonzero()[0])
    else: i_events=np.intersect1d((te <= up).nonzero()[0], (te > lo).nonzero()[0])
    # index of extant/extinct species
    # extinct_sp=(te > 0).nonzero()[0]
    # present_sp=(te == 0).nonzero()[0]
    n_all_inframe, n_S = get_sp_in_frame_br_length(ts,te,up,lo)
    
    if np.isfinite(indx):
        if par=="l":
            r, _ = get_rate_BDNN(rate, trait_tbl_NN[0][indx], nn_prm, hidden_act_f, out_act_f, apply_reg)
        else:
            r, _ = get_rate_BDNN(rate, trait_tbl_NN[1][indx], nn_prm, hidden_act_f, out_act_f, apply_reg)
    else:
        if par=="l":
            r, _ = get_rate_BDNN(rate, trait_tbl_NN[0], nn_prm, hidden_act_f, out_act_f, apply_reg)
        else:
            r, _ = get_rate_BDNN(rate, trait_tbl_NN[1], nn_prm, hidden_act_f, out_act_f, apply_reg)
    
#    print(par, r)
    lik= np.sum(log(r[i_events])) + np.sum(-r[n_all_inframe]*n_S) 
    return lik


def BDNN_fast_partial_lik(arg):
    [i_events, n_S, r, include] = arg
    r_i_events = r * i_events
    r_i_events[np.isfinite(r_i_events)] = np.log(r_i_events[np.isfinite(r_i_events)])
    rns = np.nan_to_num(r_i_events) + -r * n_S
    lik = np.sum(rns[:, include[0, :]], axis=1) # Likelihood only within edges
#    lik = np.sum(rns, axis=1)
    return lik


def get_events_ns(ts, te, times, bin_size):
    num_taxa = len(ts)
    num_bins = len(times)-1
    i_events_sp = np.full((num_taxa, num_bins), np.nan)
    i_events_ex = np.full((num_taxa, num_bins), np.nan)
    ind = np.arange(num_taxa)
    ind_ts = np.digitize(ts, times[1:])
    ind_te = np.digitize(te, times[1:])
    i_events_sp[ind, ind_ts] = 1.0
    # extinction events only for extinct taxa
    is_extinct = te.nonzero()
    i_events_ex[ind[is_extinct], ind_te[is_extinct]] = 1.0

    n_S = bin_size + 0.0
    for i in range(num_taxa):
        n_S[i, :ind_ts[i]] = 0.0
        n_S[i, ind_te[i]:] = 0.0
    time_in_first_bin = ts - times[1:][ind_ts]
    time_in_last_bin = times[1:][ind_te - 1] - te
    n_S[ind, ind_ts] = time_in_first_bin
    n_S[ind, ind_te] = time_in_last_bin
    ts_te_same_bin = ind_ts == ind_te
    n_S[ts_te_same_bin, ind_ts[ts_te_same_bin]] = ts[ts_te_same_bin] - te[ts_te_same_bin]

    return i_events_sp, i_events_ex, n_S


def update_events_ns(ts, te, times, bin_size, events_sp, events_ex, n_S, ind_update):
    i_events_sp = events_sp + 0.0
    i_events_ex = events_ex + 0.0
    i_events_sp[ind_update, :] = np.nan
    i_events_ex[ind_update, :] = np.nan
    ts = ts[ind_update]
    te = te[ind_update]
    ind_ts = np.digitize(ts, times[1:])
    ind_te = np.digitize(te, times[1:])
    i_events_sp[ind_update, ind_ts] = 1.0
    is_extinct = te.nonzero()
    i_events_ex[ind_update[is_extinct], ind_te[is_extinct]] = 1.0

    n_S_up = bin_size[ind_update, :]
    num_taxa = len(ind_update)
    ind = np.arange(num_taxa)
    for i in range(num_taxa):
        n_S_up[i, :ind_ts[i]] = 0.0
        n_S_up[i, ind_te[i]:] = 0.0
    time_in_first_bin = ts - times[1:][ind_ts]
    time_in_last_bin = times[1:][ind_te - 1] - te
    n_S_up[ind, ind_ts] = time_in_first_bin
    n_S_up[ind, ind_te] = time_in_last_bin
    ts_te_same_bin = ind_ts == ind_te
    n_S_up[ts_te_same_bin, ind_ts[ts_te_same_bin]] = ts[ts_te_same_bin] - te[ts_te_same_bin]
    n_S = n_S + 0.0
    n_S[ind_update, :] = n_S_up

    return i_events_sp, i_events_ex, n_S


def get_act_f(i):
    return [np.abs, softPlus, expFun, relu_f, sigmoid_f, sigmoid_rate][i]

def get_hidden_act_f(i):
    return [tanh_f, relu_f, leaky_relu_f, swish_f, sigmoid_f, tanh_f_approx][i]


def get_float_prec_f(i):
    return [np.float64, np.float32][i]


def create_mask(w_layers, indx_input_list, nodes_per_feature_list):
    m_layers = []
    for w in w_layers:
        indx_features = indx_input_list[len(m_layers)]
        nodes_per_feature = nodes_per_feature_list[len(m_layers)]
        # print("\nw_layers", w)
        if len(indx_features) == 0:
            # fully connect
            m = np.ones(w.shape)
        else:
            m = np.zeros(w.shape)
            max_indx_rows = 0
            j = 0
            for i in range(len(indx_features)):
                # print(i, indx_features)
                if i > 0:
                    if indx_features[i] != indx_features[i - 1]:
                        j += 1
                        indx_rows = np.arange(nodes_per_feature[j]) + max_indx_rows
                else:
                    indx_rows = np.arange(nodes_per_feature[j])
                indx_cols = np.repeat(i, nodes_per_feature[j])
                m[indx_rows, indx_cols] = 1
                # indx_cols2 = np.repeat(indx_features[i], nodes_per_feature[j])
                # m[indx_rows, indx_cols2] = 1
                max_indx_rows = np.max(indx_rows) + 1

        m_layers.append(m)
    return m_layers

def init_weight_prm(n_nodes, n_features, size_output, float_prec_f=np.float64, init_std=0.1, bias_node=0):
    bn, bn2, bn3 = 0, 0, bias_node
    n_layers = len(n_nodes) + 1
    # 1st layer
    w_layers = [float_prec_f(np.random.normal(0, init_std, (n_nodes[0], n_features + bn)))]
    # add hidden layers
    for i in range(1, n_layers - 1):
        w_layers.append(float_prec_f(np.random.normal(0, init_std, (n_nodes[i], n_nodes[i - 1] + bn2))))
    # last layer
    w_layers.append(float_prec_f(np.random.normal(0, init_std, (size_output, n_nodes[-1] + bn3))))
    return w_layers


def make_trait_time_table(trait_tbl, time_var_tbl, num_fixed_times_of_shift, fixed_times_of_shift, n_taxa, dd):
    # create a list of trait tables, one for each time frame
    trait_tbl_list = []

    # Case where we have edge shifts but no time, dd, or timevar predictor
    use_time_as_trait = np.all(fixed_times_of_shift == 0.0) == False

    for i in range(1, num_fixed_times_of_shift):
        rescaled_time = np.mean([fixed_times_of_shift[i-1], fixed_times_of_shift[i]])
        if trait_tbl is not None:
            trait_tbl_tmp = trait_tbl + 0.0
        else:
            trait_tbl_tmp = np.zeros((n_taxa, 1))
        if time_var_tbl is not None:
            time_var_tbl_tmp = time_var_tbl[i - 1,:]
            time_var_tbl_tmp = np.tile(time_var_tbl_tmp, n_taxa).reshape((n_taxa, time_var_tbl.shape[1]))
            trait_tbl_tmp = np.hstack((trait_tbl_tmp, time_var_tbl_tmp))
        if dd:
            div = np.random.random(size=n_taxa).reshape((n_taxa, 1))
            trait_tbl_tmp = np.hstack((trait_tbl_tmp, div))
        if use_time_as_trait:
            trait_tbl_tmp = np.hstack((trait_tbl_tmp, rescaled_time * np.ones((n_taxa, 1))))
        if trait_tbl is None:
            trait_tbl_tmp = trait_tbl_tmp[:, 1:]
        trait_tbl_list.append(trait_tbl_tmp)
    return np.array(trait_tbl_list)


def init_trait_and_weights(trait_tbl, time_var_tbl_lambda, time_var_tbl_mu,
                           nodes, n_bias_node=0, fadlad=0.1,
                           verbose=False, fixed_times_of_shift=[],
                           use_time_as_trait=False, dd=False, n_taxa=None,
                           loaded_tbls="", fix_edgeShift=0, float_prec_f=np.float64):
    num_fixed_times_of_shift = len(fixed_times_of_shift)
    if (use_time_as_trait or time_var_tbl_lambda is not None or dd) and isinstance(loaded_tbls[0], np.ndarray) is False: # only availble with -fixShift option
        trait_tbl_lam = make_trait_time_table(trait_tbl, time_var_tbl_lambda, num_fixed_times_of_shift, fixed_times_of_shift, n_taxa, dd)
        trait_tbl_mu = make_trait_time_table(trait_tbl, time_var_tbl_mu, num_fixed_times_of_shift, fixed_times_of_shift, n_taxa, dd)
        w_lam = init_weight_prm(n_nodes=nodes, n_features=trait_tbl_lam[0].shape[1], size_output=1,
                                float_prec_f=float_prec_f, init_std=0.01, bias_node=n_bias_node)
        w_mu = init_weight_prm(n_nodes=nodes, n_features=trait_tbl_mu[0].shape[1], size_output=1,
                               float_prec_f=float_prec_f, init_std=0.01, bias_node=n_bias_node)

        for i in w_lam:
            print(i.shape)
    elif isinstance(loaded_tbls[0], np.ndarray):
#        if use_time_as_trait and num_fixed_times_of_shift - 1 > loaded_tbls[0].shape[0]:
#            sys.exit("Error: Number of taxon-time specific tables must be the same than age of the oldest fossil + 1 or -fixShifts ")
        if loaded_tbls[0].ndim == 3:
            trait_tbl_lam = loaded_tbls[0][::-1,:,:]
            trait_tbl_mu = loaded_tbls[1][::-1,:,:]
            n_features_sp = trait_tbl_lam[0].shape[1]
            n_features_ex = trait_tbl_mu[0].shape[1]
        else:
            trait_tbl_lam = loaded_tbls[0][::-1,:]
            trait_tbl_mu = loaded_tbls[1][::-1,:]
            n_features_sp = trait_tbl_lam.shape[1]
            n_features_ex = trait_tbl_mu.shape[1]
        if dd:
            n_taxa = trait_tbl_lam.shape[1]
            n_bins = trait_tbl_lam.shape[1]
#            add_zeros = np.zeros(n_taxa * n_bins).reshape((n_bins, n_taxa, 1))
            add_zeros = np.random.random(size=n_taxa * n_bins).reshape((n_bins, n_taxa, 1))
            trait_tbl_lam = np.c_[trait_tbl_lam, add_zeros]
            trait_tbl_mu = np.c_[trait_tbl_mu, add_zeros]
            n_features_sp += 1
            n_features_ex += 1
        if use_time_as_trait:
            rescaled_time = (fixed_times_of_shift[:-1] + fixed_times_of_shift[1:]) / 2
            # Append oldest trait table in case the earliest fossils exceeds the loaded trait tables
            # Oldest trait table is trait_tbl_lam[0, :, :] and the youngest trait_tbl_lam[-1, :, :]
            if num_fixed_times_of_shift - 1 > trait_tbl_lam.shape[0]:
                n_repeats = num_fixed_times_of_shift - 1 - trait_tbl_lam.shape[0]
                missing_trt_tbls = np.tile(trait_tbl_lam[0, :], (n_repeats, 1, 1)) # Oldest bin
                trait_tbl_lam = np.concatenate((missing_trt_tbls, trait_tbl_lam), axis=0) # Repeated oldest bin stacked on top of younger bins
            if num_fixed_times_of_shift - 1 > trait_tbl_mu.shape[0]:
                n_repeats = num_fixed_times_of_shift - 1 - trait_tbl_mu.shape[0]
                missing_trt_tbls = np.tile(trait_tbl_mu[0, :], (n_repeats, 1, 1))
                trait_tbl_mu = np.concatenate((missing_trt_tbls, trait_tbl_mu), axis=0)
            # Clip trait tables when they exceed the oldest fossil
            excl_bins = trait_tbl_lam.shape[0] - (num_fixed_times_of_shift - 1)
            trait_tbl_lam = trait_tbl_lam[excl_bins:, :]
            excl_bins = trait_tbl_mu.shape[0] - (num_fixed_times_of_shift - 1)
            trait_tbl_mu = trait_tbl_mu[excl_bins:, :]
            n_taxa = trait_tbl_lam.shape[1]
            rescaled_time = np.repeat(rescaled_time, n_taxa)
            rescaled_time = rescaled_time.reshape((num_fixed_times_of_shift - 1, n_taxa, 1))
            # Zero out any feature when we use edge shifts
            if fix_edgeShift in [1, 2]: # both or max boundary
                trait_tbl_lam[0][:] = 0
                trait_tbl_mu[0][:] = 0
            if fix_edgeShift in [1, 3]: # both or min boundary
                trait_tbl_lam[-1][:] = 0
                trait_tbl_mu[-1][:] = 0
            trait_tbl_lam = np.c_[trait_tbl_lam, rescaled_time]
            trait_tbl_mu = np.c_[trait_tbl_mu, rescaled_time]
            n_features_sp += 1
            n_features_ex += 1
        w_lam = init_weight_prm(n_nodes=nodes, n_features=n_features_sp, size_output=1,
                                float_prec_f=float_prec_f, init_std=0.01, bias_node=n_bias_node)
        w_mu = init_weight_prm(n_nodes=nodes, n_features=n_features_ex, size_output=1,
                               float_prec_f=float_prec_f, init_std=0.01, bias_node=n_bias_node)
    else:
        if fadlad:
            trait_tbl_lam = 0+np.hstack((trait_tbl, (fadlad*FA).reshape((trait_tbl.shape[0],1)))) 
            trait_tbl_mu =  0+np.hstack((trait_tbl, (fadlad*LO).reshape((trait_tbl.shape[0],1)))) 
            if verbose:
                print(FA[0:10])
                print(trait_tbl_lam[0:10])
                print(trait_tbl_mu[0:10])
        else:
            trait_tbl_lam = trait_tbl + 0 
            trait_tbl_mu = trait_tbl + 0 
        w_lam = init_weight_prm(n_nodes=nodes, n_features=trait_tbl_lam.shape[1], size_output=1,
                                float_prec_f=float_prec_f, init_std=0.01, bias_node=n_bias_node)
        w_mu = init_weight_prm(n_nodes=nodes, n_features=trait_tbl_mu.shape[1], size_output=1,
                               float_prec_f=float_prec_f, init_std=0.01, bias_node=n_bias_node)
        for i in w_lam:
            print(i.shape)
    
    return [float_prec_f(trait_tbl_lam), float_prec_f(trait_tbl_mu)], [w_lam,w_mu]


def init_sampling_trait_and_weights(trait_tbl, time_var_tbl, nodes, bias_node=False,
                                    n_taxa=None,
                                    replicates_tbls=None,
                                    loaded_tbls="",
                                    float_prec_f=np.float64):
    is_trait_dep = not trait_tbl is None
    is_env_dep = isinstance(time_var_tbl, np.ndarray)
    is_age_dep = not replicates_tbls is None
    n_env_bins = 1
    n_ads_bins = 1
    tbl_trt = np.empty((n_taxa, 0))
    tbl_env = np.empty((n_taxa, 0))
    tbl_ads = np.empty((n_taxa, 0))
    
    if is_env_dep:
        n_env_bins, n_env_var = time_var_tbl.shape
        tbl_env = np.repeat(time_var_tbl, repeats=n_taxa, axis=0).reshape((n_env_bins, n_taxa, n_env_var))
        if not is_age_dep:
            tbl_ads = np.tile(tbl_ads, (n_env_bins, 1, 0))
    
    if is_age_dep:
        n_ads_bins = len(replicates_tbls)
        tbl_ads = np.zeros((n_taxa * n_ads_bins, 1)).reshape((n_ads_bins, n_taxa, 1))
        if not is_env_dep:
            tbl_env = np.tile(tbl_env, (n_ads_bins, 1, 0))
    
    if is_trait_dep:
        tbl_trt = trait_tbl
    
    n_q_bins = np.maximum(n_env_bins, n_ads_bins)
    if n_q_bins > 1:
        tbl_trt = np.tile(tbl_trt, (n_q_bins, 1, 1))
    
    trait_tbl_q = np.c_[tbl_trt, tbl_env, tbl_ads]
    
    w_q = init_weight_prm(n_nodes=nodes, n_features=trait_tbl_q.shape[-1], size_output=1,
                          float_prec_f=float_prec_f, init_std=0.01, bias_node=bias_node)
    return float_prec_f(trait_tbl_q), w_q


#def get_q_rate_BDNN(q, t_reg, qnn_output, singleton_mask, qbin_ts_te):

#    qnn = qnn_output + 0.0
#    if qnn.ndim == 2:
#        # set rate to NA before ts and after te
#        for i in range(qnn.shape[0]):
#            qnn[i, :qbin_ts_te[i, 0]] = np.nan
#            qnn[i, qbin_ts_te[i, 1]:] = np.nan
#        qnn_not_nan = np.isnan(qnn) == False
#    else:
#        qnn_not_nan = np.repeat(True, qnn.shape[-1])
#    qnn[np.logical_and(singleton_mask, qnn_not_nan)] = np.nan # singletons, they should not impact the regularization

#    reg_qnn, denom = get_reg_rates(qnn, t_reg)

#    # convert to multipliers
#    normalize_factor = 1 / np.mean(reg_qnn[np.logical_and(~singleton_mask, qnn_not_nan)])
#    reg_qnn *= normalize_factor
#    reg_qnn[np.logical_and(singleton_mask, qnn_not_nan)] = 1.0

#    if len(q) > 2 or (len(q) == 2 and q[0] != 1.0): # 2nd case is when there is exactly one shift time
#        # Several q rates
#        if reg_qnn.ndim == 1:
#            # No time-varying trait table, no age-dependent sampling
#            reg_qnn = reg_qnn[:, np.newaxis] * q
#        else:
#            # Time-varying trait table or age-dependent sampling with shift in baseline
#            reg_qnn = reg_qnn * q[np.newaxis, :]
#    else:
#        # Constant baseline sampling
#        reg_qnn = reg_qnn * q[1] # Two q's because of NHPP, but 1st is always equal to 1
##    print('reg_qnn\n', reg_qnn)
#    return reg_qnn, denom, normalize_factor


def get_q_multipliers_NN(t_reg, qnn_output, singleton_mask, apply_reg, qbin_ts_te=None):
    qnn = qnn_output + 0.0
    if qnn.ndim == 2:
        # set rate to NA before ts and after te
        for i in range(qnn.shape[0]):
            qnn[i, :qbin_ts_te[i, 0]] = np.nan
            qnn[i, qbin_ts_te[i, 1]:] = np.nan
        qnn_not_nan = np.isnan(qnn) == False
    else:
        qnn_not_nan = np.repeat(True, qnn.shape[-1])
    qnn[np.logical_and(singleton_mask, qnn_not_nan)] = np.nan # singletons, they should not impact the regularization

    # perform regularization
    reg_qnn, denom = get_reg_rates(qnn, t_reg, apply_reg)

    # convert to multipliers
    normalize_factor = 1 / np.mean(reg_qnn[np.logical_and(~singleton_mask, qnn_not_nan)])
    reg_qnn *= normalize_factor
    reg_qnn[np.logical_and(singleton_mask, qnn_not_nan)] = 1.0

    return reg_qnn, denom, normalize_factor


def get_q_rates_NN(q, q_multipliers, const_q):
    q_rates_NN = q_multipliers + 0.0
    if const_q:
        # Constant baseline sampling
        q_rates_NN = q_rates_NN * q
    else:
        # Several q rates
        if q_rates_NN.ndim == 1:
            # No time-varying trait table, no age-dependent sampling
            q_rates_NN = q_rates_NN[:, np.newaxis] * q
        else:
            # Time-varying trait table or age-dependent sampling with shift in baseline
            q_rates_NN = q_rates_NN * q[np.newaxis, :]
    return q_rates_NN


def get_highres_repeats(args_qShift, new_bs, FA, qt=None):
    if (args_qShift != '' and new_bs > 0.0) or (not qt is None):
        # q_time_frames should be global
        # Shifts in sampling rate over time
        n_bins_lowres = len(times_q_shift) + 1
        if qt is None:
            qt = np.concatenate((FA, times_q_shift, np.zeros(1)), axis=None)[::-1]
        bin_size = np.diff(qt)
        n_bins_highres = np.floor(bin_size / new_bs).astype(int)
        n_bins_highres[n_bins_highres == 0] = 1
        highres_repeats = np.repeat(np.arange(n_bins_lowres), repeats=n_bins_highres)

        qt_highres = np.zeros(len(highres_repeats) - 1)
        for i in range(len(bin_size)):
            if n_bins_highres[i] == 1:
                 if i < len(bin_size):
                     add_shifts = qt[i + 1]
                 else:
                     add_shifts = qt[i]
                 until_idx = np.sum(n_bins_highres[:i])
                 if until_idx < len(qt_highres):
                    qt_highres[until_idx] = add_shifts
            else:
                 add_shifts = (qt[i] + new_bs) + np.linspace(0.0, bin_size[i] - new_bs, n_bins_highres[i])
                 idx = np.arange(np.sum(n_bins_highres[:i]), np.sum(n_bins_highres[:(i + 1)]), 1, dtype=int)
                 if i == (len(bin_size) - 1):
                     idx = idx[:-1]
                     add_shifts = add_shifts[:-1]
                 qt_highres[idx] = add_shifts
        qt_highres = qt_highres[::-1].tolist()
        highres_repeats = np.abs(highres_repeats[::-1] - np.max(highres_repeats))

    elif args_qShift != '' and new_bs == 0.0:
        highres_repeats = np.arange(len(times_q_shift) + 1)
        qt_highres = times_q_shift

    elif args_qShift == '':
        n_bins_highres = np.floor(FA)
        if new_bs > 0.0:
            n_bins_highres *= 1.0 / new_bs
            n_bins_highres = n_bins_highres.astype(int)
            highres_repeats = np.zeros(n_bins_highres, dtype=int)
            qt_highres = np.linspace(new_bs, FA - new_bs, num=n_bins_highres-1)
            qt_highres = qt_highres[::-1]
        else:
            qt_highres = np.arange(1, np.ceil(FA))[::-1]
            highres_repeats = np.zeros(len(qt_highres) + 1)

    return highres_repeats, qt_highres


def init_cov_par(trait_tbl, n_nodes_lam, n_nodes_mu):
    # lam
    trait_tbl.shape[0]


def rescale_trait_tbl(trait_tbl):
    # fixd rescaling for lat/lon, time 
    pass


def load_pkl(file_name):
    import pickle
    with open(file_name, "rb") as f:
        return pickle.load(f)


def get_names_variable(var_file):
    vn = np.loadtxt(var_file, max_rows = 1, dtype = str)[1:]
    vn = vn.tolist()
    vn = [vn[i].replace('"', '') for i in range(len(vn))]
    vn = [vn[i].replace("'", "") for i in range(len(vn))]
    return vn


def interpolate_constant(x, xp, yp):
    indices = np.searchsorted(xp, x, side='right')
    y = np.concatenate((yp[0], yp), axis = None)
    return y[indices]


def get_binned_time_variable(timebins, var_file, rescale, translate):
    names_var = get_names_variable(var_file)
    var = np.loadtxt(var_file, skiprows = 1)
    var = var[var[:, 0] >= -translate, :]
    times = var[:, 0] * rescale + translate
    values = var[:, 1:]
    
    # reorder var_file to ensure that the most recent is the first row
    reorder = np.argsort(times)
    times = times[reorder]
    values = values[reorder, :]
    
    nbins = len(timebins)
    nvars = values.shape[1]
    mean_var = np.zeros((nbins - 1, nvars))
    discr_var = np.zeros(nvars, dtype = bool)
    for i in range(nvars):
        va = np.unique(values[:, i])
        n_va = len(va)
        discr_var[i] = np.all(np.isin(va, np.arange(n_va)))
    # If there are no values for the recent, we add the most recent value (i.e. constant environment)
    if (times[0] > timebins[-1]):
        times = np.concatenate((np.zeros(1), times), axis=None)
        values = np.concatenate((values[0, :].reshape((1, -1)), values), axis=0)
    # Interpolation if temporal resolution of var_file is lower than time bins
    if nbins > values.shape[0]:
        times_comb = np.sort(np.concatenate((timebins, times), axis = None))
        values_comb = np.zeros((len(times_comb), nvars))
        for i in range(nvars):
            if discr_var[i]:
                values_comb[:, i] = interpolate_constant(times_comb, times, values[:, i])
            else:
                values_comb[:, i] = np.interp(times_comb, times, values[:, i])
        del(values)
        del(times)
        values = values_comb
        times = times_comb
    for i in range(1, nbins):
        t_max = timebins[i-1]
        t_min = timebins[i]
        in_range_M = (times <= t_max).nonzero()[0]
        in_range_m = (times >= t_min).nonzero()[0]
        values_bin = values[np.intersect1d(in_range_M, in_range_m),:]
        if values_bin.size == 0:
            # If there is no value for the bin (i.e. higher bin resolution than time series, take the younger value)
            values_bin = values[np.max(in_range_M), :]
            values_bin = values_bin.reshape((1, values_bin.size))
        if np.any(discr_var == False):
            mean_var[i - 1, discr_var == False] = np.mean(values_bin[:, discr_var == False], axis = 0)
        if np.any(discr_var):
            values_bin_disc = values_bin[:, discr_var]
            most_freq_state = np.zeros(values_bin_disc.shape[1])
            for j in range(values_bin_disc.shape[1]):
                va, counts = np.unique(values_bin_disc[:,j], return_counts = True)
                most_freq_state[j] = va[np.argmax(counts)]
            mean_var[i - 1, discr_var] = most_freq_state
    return mean_var, names_var


def write_pkl(obj, out_file):
    import pickle
    with open(out_file, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def get_taxon_rates_bdnn(arg):
    [tsA, teA, timesLA, timesMA, bdnn_lam_ratesA, bdnn_mu_ratesA] = arg
    digitized_ts = np.digitize(tsA, timesLA) - 1
    digitized_te = np.digitize(teA, timesMA) - 1
    digitized_ts[digitized_ts < 0] = 0 # fixed index of oldest ts in the dataset
    sp_rates_L = np.zeros(len(tsA))
    sp_rates_M = np.zeros(len(tsA))
    for i in range(len(tsA)):
        if bdnn_lam_ratesA.ndim == 2:
            sp_rates_L[i] = bdnn_lam_ratesA[i, digitized_ts[i]]
            sp_rates_M[i] = bdnn_mu_ratesA[i, digitized_te[i]]
        else:
            sp_rates_L[i] = bdnn_lam_ratesA[i]
            sp_rates_M[i] = bdnn_mu_ratesA[i]
    return sp_rates_L, sp_rates_M


def make_missing_bins(FA, bin_size=1):
    missing_bins = np.arange(1, 1000, bin_size)[::-1]
    max_FA = np.max(FA)
    missing_bins = missing_bins[missing_bins < max_FA]
    missing_bins_FA_0 = np.concatenate((max_FA, missing_bins, np.zeros(1)), axis=None)
    return missing_bins_FA_0, missing_bins


def get_occs_sp(fossil, q_bins):
    occs_sp_bin2 = list()
    for i in range(len(fossil)):
        occs_temp = fossil[i]
        h = np.histogram(occs_temp[occs_temp > 0], bins=q_bins[::-1])[0][::-1]
        occs_sp_bin2.append(h)
    occs_sp = np.array(occs_sp_bin2)
    return occs_sp


def get_fossil_features_q_shifts(fossil, q_bins, occs_sp, LO):
    duration_q_bins = np.abs(np.diff(q_bins))
    duration_q_bins = np.tile(duration_q_bins, len(fossil)).reshape((len(fossil), len(duration_q_bins)))
    log_factorial_occs = np.zeros(len(fossil))
    occs_single_bin = np.full(len(fossil), False)
    num_q_bins = len(q_bins)
    for i in range(len(fossil)):
        bin_with_fossils = np.where(occs_sp[i, :] > 0)[0]
        log_factorial_occs[i] = np.sum(np.log(np.arange(1, np.sum(occs_sp[i, :]) + 1)))
        cts = np.count_nonzero(occs_sp[i, :])
        single_bin_not_extant = cts == 1 and LO[i] > 0.0
        recent_bin_extant = np.min(bin_with_fossils) == num_q_bins and LO[i] == 0.0
        if single_bin_not_extant or recent_bin_extant:
            occs_single_bin[i] = True
    return log_factorial_occs, duration_q_bins, occs_single_bin


def precompute_fossil_features(arg_q_shift, arg_bdnn_timevar, bdnn_ads, bin_size=1):
    duration_q_bins = None
    occs_single_bin = None
    if arg_q_shift == "" and arg_bdnn_timevar == "" and bdnn_ads < 0.0:
        # Constant baseline q
        constant_q = True
        argsHPP = 1
        TPP_model = 0
        occs_sp = np.zeros(len(fossil))
        log_factorial_occs = np.zeros(len(fossil))
        for i in range(len(fossil)):
            occs_sp[i] = np.count_nonzero(fossil[i])
            log_factorial_occs[i] = np.sum(np.log(np.arange(1, occs_sp[i] + 1)))
        q_bins, _ = make_missing_bins(FA, bin_size)
        use_HPP_NN_lik = False
    else:
        # Shifts in baseline q, time-dependent features, and/or age-dependent sampling
        constant_q = False
        argsHPP = 0
        TPP_model = 1
        try:
            occs_sp_bin
        except NameError:
            has_q_shifts = False
        else:
            has_q_shifts = True
        if has_q_shifts or bdnn_ads > 0.0:
            q_bins = np.concatenate((np.max(FA), times_q_shift, np.zeros(1)), axis=None)
        else:
            q_bins, _ = make_missing_bins(FA, bin_size)
        occs_sp = get_occs_sp(fossil, q_bins)
        log_factorial_occs, duration_q_bins, occs_single_bin = get_fossil_features_q_shifts(fossil, q_bins, occs_sp, LO)
        use_HPP_NN_lik = True
        if (arg_q_shift == "" and bdnn_ads >= 0.0) or (arg_q_shift == "" and arg_bdnn_timevar != ""):
            argsHPP = 1
            TPP_model = 0
            constant_q = True
    return argsHPP, occs_sp, log_factorial_occs, duration_q_bins, occs_single_bin, q_bins, use_HPP_NN_lik, TPP_model, constant_q


# Preservation likelihood for neural network
def get_time_in_q_bins(ts, te, time_frames, duration_q_bins, single_bin):
    ind_ts = np.digitize(ts, time_frames[1:])
    ind_te = np.digitize(te, time_frames[1:])
    time_in_first_bin = ts - time_frames[1:][ind_ts]
    time_in_last_bin = time_frames[1:][ind_te - 1] - te
    d = duration_q_bins + 0.0

    # Not possible to precompute because ts/te might extent far beyond the bins containing fossils
    for i in range(d.shape[0]):
        d[i, :ind_ts[i]] = 0.0
        d[i, ind_te[i]:] = 0.0

    # Create a mask using broadcasting - slower than the loop
#    col_idx = np.arange(d.shape[1])
#    d[col_idx < ind_ts[:, None]] = 0.0
#    d[col_idx > ind_te[:, None]] = 0.0
    
    ind = np.arange(d.shape[0])
    d[ind, ind_ts] = time_in_first_bin
    d[ind, ind_te] = time_in_last_bin
    ts_te_same_bin = ind_ts == ind_te
    single_bin = np.logical_and(single_bin, ts_te_same_bin)
    d[single_bin, ind_ts[single_bin]] = ts[single_bin] - te[single_bin]
    return d


def make_singleton_mask(occs_sp, timevar_q="", ads=-1.0):
    s = occs_sp.shape
    if len(s) == 2:
        num_taxa, num_bins = s
        s_mask = np.sum(occs_sp, axis=1) == 1
        if timevar_q != "" or ads >= 0:
            s_mask = np.repeat(s_mask, num_bins).reshape((num_taxa, num_bins))
    else:
        s_mask = occs_sp == 1
    return s_mask


def HPP_NN_lik(arg):
    [ts, te, q, alpha, q_multipliers, const_q, k, log_factorial_occs, time_frames, duration_q_bins, single_bin, singletons, argsG, gamma_ncat, YangGammaQuant] = arg
    # calc time lived in each time frame
    d = get_time_in_q_bins(ts, te, time_frames, duration_q_bins, single_bin)
    if argsG == 0:
        if const_q:
            q = q[1]
        q_rates = get_q_rates_NN(q, q_multipliers, const_q)
        qtl = -q_rates * d
        lik = np.nansum(qtl + np.log(q_rates) * k, axis=1) - np.log(1 - np.exp(np.nansum(qtl, axis=1))) - log_factorial_occs
        weight_per_taxon = 1
    else:
        n_bins_q = len(q)
        n_taxa_q = len(ts)
        if const_q:
            # For a single constant q, alpha is the 1st element of the q array (but not for the TPP model where all q's are the rates)
            alpha = q[0]
            q = q[1]
            n_bins_q = q_multipliers.shape[-1]
        q_rates = np.zeros((gamma_ncat, n_taxa_q, n_bins_q))
        lik_vec = np.zeros((gamma_ncat, n_taxa_q))
        YangGamma = get_gamma_rates_qnn(alpha, gamma_ncat, YangGammaQuant)
        for i in range(gamma_ncat):
            q_rates[i, :, :] = get_q_rates_NN(YangGamma[i] * q, q_multipliers, const_q)
        q_rates[:, singletons, :] = q
        qtl = -q_rates * d
        lik_vec = np.nansum(qtl + np.log(q_rates) * k[np.newaxis, :, :], axis=2) - np.log(1 - np.exp(np.nansum(qtl, axis=2))) - log_factorial_occs
        lik_max = np.max(lik_vec, axis=0)
        lik_vec2 = lik_vec - lik_max
        lik = np.log(np.sum(np.exp(lik_vec2), axis=0) / gamma_ncat) + lik_max
        # Get relative weight of the gamma categories
        lik_vec_exp = np.exp(lik_vec2)
        weight_per_taxon = lik_vec_exp / np.sum(lik_vec_exp, axis=0)
        q_rates = np.sum(q_rates * weight_per_taxon[:, :, np.newaxis], axis=0)
    return lik, q_rates, weight_per_taxon


def HOMPP_NN_lik(arg):
    [ts, te, q, q_multipliers, const_q, k, log_factorial_occs, singletons, argsG, gamma_ncat, YangGammaQuant] = arg
    br_length = ts - te
    if argsG == 0:
        q_rates = get_q_rates_NN(q[1], q_multipliers, const_q)
        qtl = -q_rates * br_length
        lik = qtl + np.log(q_rates) * k - log_factorial_occs - np.log(1 - np.exp(qtl))
        weight_per_taxon = 1
    else:
        n_taxa_q = len(ts)
        q_rates = np.zeros((gamma_ncat, n_taxa_q))
        YangGamma = get_gamma_rates_qnn(q[0], gamma_ncat, YangGammaQuant)
        for i in range(gamma_ncat):
            q_rates[i, :] = get_q_rates_NN(YangGamma[i] * q[1], q_multipliers, const_q)
        q_rates[:, singletons] = q[1]
        qtl = -q_rates * br_length
        lik_vec = qtl + np.log(q_rates) * k - log_factorial_occs - np.log(1 - np.exp(qtl))
        lik_max = np.max(lik_vec, axis=0)
        lik_vec2 = lik_vec - lik_max
        lik = np.log(np.sum(np.exp(lik_vec2), axis=0) / gamma_ncat) + lik_max
        lik_vec_exp = np.exp(lik_vec)
        weight_per_taxon = lik_vec_exp / np.sum(lik_vec_exp, axis=0)
        q_rates = np.sum(q_rates * weight_per_taxon, axis=0)
    return lik, q_rates, weight_per_taxon


def harmonic_mean_q_through_time(ts, te, time_frames, q_rates):
    if q_rates.ndim == 2:
        w = np.ones(q_rates.shape)
    else:
        w = np.ones((q_rates.shape[0], len(time_frames) - 1))
    ind_ts = np.digitize(ts, time_frames[1:])
    ind_te = np.digitize(te, time_frames[1:]) + 1
    for i in range(w.shape[0]):
        w[i, :ind_ts[i]] = np.nan
        w[i, ind_te[i]:] = np.nan
    if q_rates.ndim == 2:
        q_sp_bin = q_rates * w
    else:
        q_sp_bin = q_rates[:, np.newaxis] * w
    qtt = np.full(q_sp_bin.shape[1], np.nan)
    not_all_nan = np.sum(np.isnan(q_sp_bin), axis=0) < q_sp_bin.shape[0]
    qtt[not_all_nan] = 1 / np.nanmean(1 / q_sp_bin[:, not_all_nan], axis=0)
    return qtt


def get_bin_ts_te(ts, te, time_frames):
    bin_ts_te = np.stack([np.digitize(ts, time_frames[1:]), np.digitize(te, time_frames[1:]) + 1], axis=1)
    return bin_ts_te


def add_taxon_age(ts, te, q_time_frames, trt_tbl, tsA=None, teA=None):
    if tsA is None:
        ts_te_changed = np.arange(len(ts))
    else:
#        ts_te_changed = np.unique(np.concatenate((np.where(ts != tsA)[0], np.where(te != teA)[0]), axis=None))
        ts_te_changed = np.where( (ts - te) != (tsA -teA) )[0]
    bins = q_time_frames[::-1]
    bins[-1] = np.inf
    n_bins = len(bins) - 2
    step_size = 0.01
    for i in ts_te_changed:
        trt_tbl[:, i, -1] = 0.0
        age = np.arange(te[i], ts[i], step=step_size)
        age_bins = np.digitize(age, bins) - 1
        u, c = np.unique(age_bins, return_counts=True)
        rel_age = np.array([0.5])
        if (ts[i] - te[i]) >= step_size:
            age_norm = (age - np.min(age)) / np.ptp(age)
            # This give 0.5 if a taxon occurs in just a single bin
            rel_age = np.bincount(age_bins - np.min(age_bins), weights=age_norm) / c
        trt_tbl[n_bins - u, i, -1] = rel_age[::-1]
#    trt_tbl[:, ts_te_changed, -1] -= 0.5 # center in 0
    return trt_tbl


def get_binned_div_traj(timebins, times, values):
    timebins2 = timebins + 0.0
    # Why are the timebins contain the inf outside the function when not doing a copy?
    timebins2[0] = np.inf
    bin_indices = np.digitize(times, timebins2) - 1
    num_bins = len(timebins) - 1
    binned_div = np.zeros(num_bins)
    for i in range(num_bins):
        mask = bin_indices == i
        if np.any(mask): # Avoid empty bins
            binned_div[i] = np.mean(values[mask])
    return binned_div


#def add_diversity(trt_tbl, ts, te, timesLA, time_vec, bdnn_rescale_div, n_taxa, div_idx_trt_tbl):
#    bdnn_time_div = np.arange(timesLA[0], 0.0, -0.001)
#    bdnn_div = get_DT(bdnn_time_div, ts, te)
#    bdnn_binned_div = get_binned_div_traj(time_vec, bdnn_time_div, bdnn_div).flatten()[:-1] / bdnn_rescale_div
#    bdnn_binned_div = np.repeat(bdnn_binned_div, n_taxa).reshape((len(bdnn_binned_div), n_taxa))
#    trt_tbl[0][ :, :, div_idx_trt_tbl] = bdnn_binned_div
#    trt_tbl[1][ :, :, div_idx_trt_tbl] = bdnn_binned_div
#    return trt_tbl


def get_diversity(ts, te, timesLA, time_vec, bdnn_rescale_div, n_taxa, step_size=0.01):
    bdnn_time_div = np.arange(timesLA[0], 0.0, -timesLA[0] / 10000.0)
    bdnn_div = get_DT(bdnn_time_div, ts, te)
    bdnn_binned_div = get_binned_div_traj(time_vec, bdnn_time_div, bdnn_div).reshape(-1) / bdnn_rescale_div
    bdnn_binned_div = np.repeat(bdnn_binned_div, n_taxa).reshape((len(bdnn_binned_div), n_taxa))
    return bdnn_binned_div


# ADE model
def cdf_WR(W_shape,W_scale,x):
    return (x/W_scale)**(W_shape)

def log_wr(t,W_shape,W_scale): # return log extinction rate at time t based on ADE model
    return log(W_shape/W_scale)+(W_shape-1)*log(t/W_scale)

def log_wei_pdf(x,W_shape,W_scale): # log pdf Weibull
    return log(W_shape/W_scale) + (W_shape-1)*log(x/W_scale) - (x/W_scale)**W_shape

def wei_pdf(x,W_shape,W_scale): # pdf Weibull
    return W_shape/W_scale * (x/W_scale)**(W_shape-1) *exp(-(x/W_scale)**W_shape)

def pdf_W_poi(W_shape,W_scale,q,x): # exp log Weibull + Q_function
    return exp(log_wei_pdf(x,W_shape,W_scale) + log(1-exp(-q*x)))

def pdf_W_poi_nolog(W_shape,W_scale,q,x): # Weibull + Q_function
    return wei_pdf(x,W_shape,W_scale) * (1-exp(-q*x))

def cdf_Weibull(x,W_shape,W_scale): # Weibull cdf
    return 1 - exp(-(x/W_scale)**W_shape)

def integrate_pdf(P,v,d,upper_lim):
    if upper_lim==0: return 0
    else: return sum(P[v<upper_lim])*d

# integration settings //--> Add to command list
nbins  = 1000
xLim   = 50
x_bins = np.linspace(0.0000001,xLim,nbins)
x_bin_size = x_bins[1]-x_bins[0]

def BD_bd_rates_ADE_lik(arg):
    [s,e,W_shape,W_scale]=arg
    # fit BD model
    birth_lik = len(s)*log(l)-l*sum(d) # replace with partial lik function
    d = s-e
    de = d[e>0] #takes only the extinct species times
    death_lik_de = sum(log_wr(de, W_shape, W_scale)) # log probability of death event
    death_lik_wte = sum(-cdf_WR(W_shape,W_scale, d[te==0]))
    # analytical integration
    death_lik_wte = sum(-m0*cdf_WR(W_shape,W_scale, d)) # log probability of waiting time until death event
    lik = birth_lik + death_lik_de + death_lik_wte
    return lik

def BD_age_partial_lik(arg):
    [ts,te,up,lo, rate,par,  cov_par,   W_shape,q]=arg
    W_scale = rate
    ind_ex_events=np.intersect1d((te <= up).nonzero()[0], (te >= lo).nonzero()[0])
    ts_time = ts[ind_ex_events]
    te_time = te[ind_ex_events]
    br = ts_time-te_time
    #br=ts-te
    lik1=(log_wei_pdf(br[te_time>0],W_shape,W_scale)) + (log(1-exp(-q*br[te_time>0])))
    v=x_bins
    # numerical integration + analytical for right tail
    P = pdf_W_poi(W_shape,W_scale,q,v)                 # partial integral (0 => xLim) via numerical integration
    d= x_bin_size
    const_int = (1- cdf_Weibull(xLim,W_shape,W_scale)) # partial integral (xLim => Inf) via CDF_weibull
    lik2 = log( sum(P)*d  + const_int )
    lik_extant = [log(sum(P[v>i])*d + const_int)-lik2 for i in ts_time[te_time==0] ] # P(x > ts | W_shape, W_scale, q)
    # this is equal to log(1- (sum(P[v<=i]) *(v[1]-v[0]) / exp(lik2)))
    lik_extinct = sum(lik1-lik2)
    lik = lik_extinct + sum(lik_extant)
    return lik

######## W-MEAN
def get_fraction_per_bin(ts,te,time_frames):
    len_time_intervals=len(time_frames)-1
    n=len(ts)
    tot_br = ts-te
    in_bin_br =np.zeros(n*len_time_intervals).reshape(n,len_time_intervals)
    in_bin_0_br =np.zeros(n)
    te_temp = np.fmax(te,np.ones(n)*time_frames[1])
    br_temp = ts-te_temp
    in_bin_0_br[br_temp>0] = br_temp[br_temp>0]
    in_bin_br[:,0]= in_bin_0_br
    for i in range(1,len_time_intervals):
        t_i, t_i1 = time_frames[i], time_frames[i+1]
        ts_temp = np.fmin(ts,np.ones(n)*t_i)
        te_temp = np.fmax(te,np.ones(n)*t_i1)
        br_temp = ts_temp-te_temp
        in_bin_i_br = br_temp
        in_bin_i_br[br_temp<0] = 0
        in_bin_br[:,i]= in_bin_i_br
    return in_bin_br.T/tot_br

def BD_age_lik_vec_times(arg):
    [ts,te,time_frames,W_shape,W_scales,q_rates,q_time_frames]=arg
    len_time_intervals=len(time_frames)-1
    if TPP_model == 0: # because in this case q_rates[0] is alpha par of gamma model (set to 1)
        q_rates = np.zeros(len_time_intervals)+q_rates[1]

    #Weigths = get_fraction_per_bin(ts,te,time_frames)
    #W_scale_species = np.sum(W_scales*Weigths.T,axis=1)
    W_scale_species = np.zeros(len(ts))+W_scales[0]
    #W_scale_species = np.zeros(len(ts))
    #W_scale_species = W_scales[np.round(Weigths).astype(int)][1]

    qWeigths = get_fraction_per_bin(ts,te,q_time_frames)
    q_rate_species  = np.sum(q_rates*qWeigths.T,axis=1)

    br=ts-te
    #print W_scales
    #print "O:",Weigths
    #print "T:", W_scale_species
    lik1=(log_wei_pdf(br[te>0],W_shape,W_scale_species[te>0])) + (log(1-exp(-q_rate_species[te>0]*br[te>0])))
    # the q density using the weighted mean is equal to (log(1-exp(-np.sum(q_rates*(br*Weigths).T,axis=1))))
    # numerical integration + analytical for right tail
    v=np.zeros((len(ts),len(x_bins)))+x_bins
    P = pdf_W_poi(W_shape,W_scale_species,q_rate_species,v.T)                 # partial integral (0 => xLim) via numerical integration
    d= x_bin_size
    const_int = (1- cdf_Weibull(xLim,W_shape,W_scale_species)) # partial integral (xLim => Inf) via CDF_weibull
    lik2 = log( np.sum(P,axis=0)*d  + const_int )
    ind_extant = (te==0).nonzero()[0]
    lik_extant =[log(sum(P[x_bins>br[i],i])*d + const_int[i])-lik2[i] for i in ind_extant] # P(x > ts | W_shape, W_scale, q)
    # this is equal to log(1- (sum(P[v<=i]) *(v[1]-v[0]) / exp(lik2)))
    lik_extinct = sum(lik1-lik2[te>0])
    lik = lik_extinct + sum(lik_extant)
    return lik

def pure_death_shift(arg):
    [ts,te,time_frames,L,M,Q]=arg
    BD_lik = 0
    B = sort(time_frames)+0.000001 # add small number to avoid counting extant species as extinct
    ss1 = np.histogram(ts,bins=B)[0][::-1]
    ee2 = np.histogram(te,bins=B)[0][::-1]

    for i in range(len(time_frames)-1):
        up, lo = time_frames[i], time_frames[i+1]
        len_sp_events=ss1[i]
        len_ex_events=ee2[i]
        inTS = np.fmin(ts,up)
        inTE = np.fmax(te,lo)
        S    = inTS-inTE
        # prob waiting times: qExp(S,mu) * samplingP(S,q) # CHECK for ANALYTICAL SOLUTION?
        # prob extinction events: dExp(S[inTE>lo],mu) * samplingD(S[inTE>lo],mu)
        # prob extant taxa: 1 - (dExp(S[inTE>lo],mu) * samplingD(S[inTE>lo],mu))/exp(P_waiting_time)


        BD_lik += lik0+lik1+lik2+lik3

def BDI_partial_lik(arg):
    [ts,te,up,lo,rate,par, cov_par,_]=arg
    ind_in_time = np.intersect1d((all_events_array[0] <= up).nonzero()[0], (all_events_array[0] > lo).nonzero()[0])
    traj_T=div_trajectory[ind_in_time]
    all_events_temp2_T=all_events_array[:,ind_in_time]

    L = np.zeros(len(traj_T))+rate * (1-model_BDI) # if model_BDI=0: BD, if model_BDI=1: ID
    M = np.zeros(len(traj_T))+rate
    I = np.zeros(len(traj_T))+rate * model_BDI
    k=traj_T
    event_at_state_k= all_events_temp2_T[1]-1 # events=0: speciation; =1: extinction
    Tk = dT_events[ind_in_time]
    Uk = 1-event_at_state_k
    Dk = event_at_state_k

    if par=="l":
        # calc likelihood only when diversity > 0
        lik = sum(log(L[k>0]*k[k>0]+I[k>0])*Uk[k>0] - (L[k>0]*k[k>0]+I[k>0])*Tk[k>0])
    else:
        # calc likelihood only when diversity > 0
        lik = sum(log(M[k>0]*k[k>0])*Dk[k>0] -(M[k>0]*k[k>0]*Tk[k>0]))
    return lik

def PoiD_partial_lik(arg):
    [ts,te,up,lo,rate,par, cov_par,_]=arg
    if par=="l":
        i_events=np.intersect1d((ts <= up).nonzero()[0], (ts > lo).nonzero()[0])
        n_i_events = len(i_events)
        lik = log(rate)*n_i_events - (rate * (up-lo))
    else:
        i_events=np.intersect1d((te <= up).nonzero()[0], (te > lo).nonzero()[0])
        n_i_events = len(i_events)
        n_all_inframe, n_S = get_sp_in_frame_br_length(ts,te,up,lo)
        lik= log(rate)*n_i_events  -rate*sum(n_S)
    return lik

# PRESERVATION
def HPP_vec_lik(arg, return_rate=False):
    [te,ts,time_frames,q_rates,i,alpha]=arg
    i=int(i) # species number
    k_vec = occs_sp_bin[i] # no. occurrences per time bin per species
    # e.g. k_vec = [0,0,1,12.,3,0]
    if ts in time_frames and ts != time_frames[0]: # if max(ts)<max(q_shift), ignore max(ts)
        time_frames = time_frames[time_frames != ts]
    h = np.histogram(np.array([ts,te]),bins=sort(time_frames))[0][::-1]
    ind_tste= (h).nonzero()[0]
    ind_min=np.min(ind_tste)
    ind_max=np.max(ind_tste)
    ind=np.arange(len(time_frames))
    ind = ind[ind_min:(ind_max+1)] # indexes of time frames where lineage is present
    # calc time lived in each time frame
    t = time_frames[time_frames<ts]
    t = t[t>te]
    t2 = np.array([ts]+list(t)+[te])
    d = np.abs(np.diff(t2))

    if argsG == 1 and sum(k_vec)>1: # for singletons no Gamma
        # loop over gamma categories
        YangGamma=get_gamma_rates(alpha)
        lik_vec = np.zeros(pp_gamma_ncat)
        for i in range(pp_gamma_ncat):
            qGamma= YangGamma[i]*q_rates
            lik_vec[i] = sum(-qGamma[ind]*d + log(qGamma[ind])*k_vec[ind]) - log(1-exp(sum(-qGamma[ind]*d))) -sum(log(np.arange(1,sum(k_vec)+1)))

        #print lik_vec
        lik2= lik_vec-np.max(lik_vec)
        lik = sum(log(sum(exp(lik2))/pp_gamma_ncat)+np.max(lik_vec))
    else:
        lik = sum(-q_rates[ind]*d + log(q_rates[ind])*k_vec[ind]) - log(1-exp(sum(-q_rates[ind]*d))) -sum(log(np.arange(1,sum(k_vec)+1)))
        if return_rate:
            YangGamma=get_gamma_rates(alpha)
            lik2 = np.ones(pp_gamma_ncat) / pp_gamma_ncat
    if return_rate:
        pr = np.exp(lik2) / np.sum(np.exp(lik2))
        weighted_mean = np.sum(YangGamma * pr)
        return weighted_mean
    else:
        return lik

def HOMPP_lik(arg):
    [m,M,shapeGamma,q_rate,i,cov_par, ex_rate]=arg
    i=int(i)
    x=fossil[i]
    lik=0
    k=len(x[x>0]) # no. fossils for species i
    br_length = M-m
    if useBounded_BD == 1:
        br_length = np.minimum(M,boundMax)-np.maximum(m, boundMin)
    if cov_par ==2: # transform preservation rate by trait value
        q=exp(log(q_rate)+cov_par*(con_trait[i]-parGAUS[0]))
    else: q=q_rate
    if argsG == 1:
        YangGamma=get_gamma_rates(shapeGamma)
        qGamma= YangGamma*q
        lik1= -qGamma*(br_length) + log(qGamma)*k - sum(log(np.arange(1,k+1)))  -log(1-exp(-qGamma*(br_length)))
        maxLik1 = np.max(lik1)
        lik2= lik1-maxLik1
        lik=log(sum(exp(lik2)*(1./pp_gamma_ncat)))+maxLik1
        return lik
    else:
        return -q*(br_length) + log(q)*k - sum(log(np.arange(1,k+1))) - log(1-exp(-q*(br_length)))

def NHPP_lik(arg):
    [m,M,shapeGamma,q_rate,i,cov_par, ex_rate]=arg
    i=int(i)
    x=fossil[i]
    lik=0
    k=len(x[x>0]) # no. fossils for species i
    x=sort(x)[::-1] # reverse
    xB1= -(x-M) # distance fossil-ts
    c=.5
    if cov_par !=0: # transform preservation rate by trait value
        q=exp(log(q_rate)+cov_par*(con_trait[i]-parGAUS[0]))
    else: q=q_rate
    if m==0 and use_DA == 1: # data augmentation
        l=1./ex_rate
        #quant=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
        quant=[.125,.375,.625,.875] # quantiles gamma distribution (predicted te)
        #quant=[.5]
        GM= -(-log(1-np.array(quant))/ex_rate) # get the x values from quantiles (exponential distribution)
        # GM=-array([gdtrix(ex_rate,1,jq) for jq in quant]) # get the x values from quantiles
        z=np.append(GM, [GM]*(k)).reshape(k+1,len(quant)).T
        xB=xB1/-(z-M) # rescaled fossil record from ts-te_DA to 0-1
        C=M-c*(M-GM)  # vector of modal values
        a = 1 + (4*(C-GM))/(M-GM) # shape parameters a,b=3,3
        b = 1 + (4*(-C+M))/(M-GM) # values will change in future implementations
        #print M, GM
        int_q = betainc(a,b,xB[:,k])* (M-GM)*q    # integral of beta(3,3) at time xB[k] (tranformed time 0)
        MM=np.zeros((len(quant),k))+M     # matrix speciation times of length quant x no. fossils
        aa=np.zeros((len(quant),k))+a[0]  # matrix shape parameters (3) of length quant x no. fossils
        bb=np.zeros((len(quant),k))+b[0]  # matrix shape parameters (3) of length quant x no. fossils
        X=np.append(x[x>0],[x[x>0]]*(len(quant)-1)).reshape(len(quant), k) # matrix of fossils of shape quant x no. fossils
        if len(quant)>1:
            den = sum(G_density(-GM,1,l)) + small_number
            #lik_temp= sum(exp(-(int_q) + np.sum((logPERT4_density(MM,z[:,0:k],aa,bb,X)+log(q)), axis=1) ) \
            #* (G_density(-GM,1,l)/den) / (1-exp(-int_q))) / len(GM)
            #if lik_temp>0: lik=log(lik_temp)
            #else: lik = -inf
            # LOG TRANSF
            log_lik_temp = (-(int_q) + np.sum((logPERT4_density(MM,z[:,0:k],aa,bb,X)+log(q)), axis=1) )  \
            + log(G_density(-GM,1,l)/den) - log(1-exp(-int_q))
            maxLogLikTemp = np.max(log_lik_temp)
            log_lik_temp_scaled = log_lik_temp-maxLogLikTemp
            lik = log(sum(exp(log_lik_temp_scaled))/ len(GM))+maxLogLikTemp
        else: lik= sum(-(int_q) + np.sum((logPERT4_density(MM,z[:,0:k],aa,bb,X)+log(q)), axis=1))
        lik += -sum(log(np.arange(1,k+1)))
    elif m==0: lik = HOMPP_lik(arg)
    else:
        C=M-c*(M-m)
        a = 1+ (4*(C-m))/(M-m)
        b = 1+ (4*(-C+M))/(M-m)
        lik = -q*(M-m) + sum(logPERT4_density(M,m,a,b,x)+log(q)) - log(1-exp(-q*(M-m)))
        lik += -sum(log(np.arange(1,k+1)))
    #if m==0: print i, lik, q, k, min(x),sum(exp(-(int_q)))
    return lik

def NHPPgamma(arg):
    [m,M,shapeGamma,q_rate,i,cov_par, ex_rate]=arg
    i=int(i)
    x=fossil[i]

    k=len(x[x>0])   # no. fossils for species i
    x=sort(x)[::-1] # reverse
    xB1= -(x-M)     # distance fossil-ts

    if cov_par ==2: # transform preservation rate by trait value
        q=exp(log(q_rate)+cov_par*(con_trait[i]-parGAUS[0]))
    else: q=q_rate

    YangGamma=get_gamma_rates(shapeGamma)
    qGamma=YangGamma*q
    c=.5
    if len(x)>1 and m>0:
        C=M-.5*(M-m)
        a = 1+ (4*(C-m))/(M-m)
        b = 1+ (4*(-C+M))/(M-m)
        W=PERT4_density(M,m,a,b,x)
        PERT4_den=np.append(W, [W]*(pp_gamma_ncat-1)).reshape(pp_gamma_ncat,len(W)).T
        #lik=log( sum( (exp(-qGamma*(M-m)) * np.prod((PERT4_den*qGamma), axis=0) / (1-exp(-qGamma*(M-m))))*(1./pp_gamma_ncat)) )
        tempL=exp(-qGamma*(M-m))
        if np.max(tempL)<1:
            L=log(1-tempL)
            if np.isfinite(sum(L)):
                lik1=-qGamma*(M-m) + np.sum(log(PERT4_den*qGamma), axis=0) - L
                maxLogLik1 = np.max(lik1)
                lik2=lik1-maxLogLik1
                lik=log(sum(exp(lik2)*(1./pp_gamma_ncat)))+maxLogLik1
            else: lik=-100000
        else: lik=-100000
    elif m==0 and use_DA == 0: lik = HOMPP_lik(arg)
    else: lik=NHPP_lik(arg)
    return lik



###### BEGIN FUNCTIONS for FBD Range ########
def init_ts_te_FBDrange(FA,LO):
    ts,te = init_ts_te(FA,LO)
    if FBDrange == 3:
        ts = FA+ np.random.gamma(1,10,len(FA)) #(FA-LO)*0.2
        te = LO
    min_dt = np.min(get_DT_FBDrange(ts,ts,te)[1:])
    print("""\n Using the FBD-range likelihood function \n(Warnock, Heath, and Stadler; Paleobiology, in press)\n""")
    while min_dt <= 1:
        ts = ts+0.01*ts
        #ts = np.random.exponential(5, len(FA)) + FA
        #te = LO - np.random.exponential(5, len(LO))
        #te[te<0] = np.random.uniform(LO[te<0],0)
        dt = get_DT_FBDrange(ts,ts,te)
        min_dt = min(dt[1:])

    return ts, te

def get_DT_FBDrange(T,s,e): # returns the Diversity Trajectory of s,e at times T (x10 faster)
    T_list = np.array(list(T) + [np.max(T)+1])
    B=np.sort(T_list)-.000001 # the - .0001 prevents problems with identical ages
    #print "B", B
    #print "T", T
    #quit()
    ss1 = np.histogram(s,bins=B)[0]
    ee2 = np.histogram(e,bins=B)[0]
    DD=(ss1-ee2)[::-1]
    return np.cumsum(DD)[0:len(T)].astype(float)


def get_L_FBDrange(T,s,e): # returns the Diversity Trajectory of s,e at times T (x10 faster)
    Lvec = [np.sum(get_sp_in_frame_br_length(s, e, T[i], T[i+1])[1]) for i in range(len(T)-1)]
    return np.array(Lvec)

def get_k(array_all_fossils, times):
    ff = np.histogram(array_all_fossils[array_all_fossils>0],bins=np.sort(times))[0]
    return ff[::-1]

def get_times_n_rates(timesQ, timesL, timesM, q_rates, Lt,Mt):
    Ltemp = 0+Lt
    Mtemp = 0+Mt
    merged = list(timesQ[1:])+ list(timesL[1:])+ list(timesM[1:])
    times = np.unique(merged)[::-1]
    psi = q_rates[np.digitize(times,timesQ[1:])]
    if len(Ltemp)>1: lam = Ltemp[np.digitize(times,timesL[1:])]
    else: lam = np.zeros(len(psi))+Ltemp[0]
    if len(Mtemp)>1: mu  = Mtemp[np.digitize(times,timesM[1:])]
    else: mu = np.zeros(len(psi))+Mtemp[0]
    times =np.insert(times,0, timesL[0])
    return times, psi, lam, mu

def calcAi(lam,mu,psi):
    Ai = abs(sqrt((lam-mu-psi)**2 + 4*lam*psi))
    return Ai

def calc_q(i, t, args):
    [intervalAs, lam, mu, psi, times, l, rho] = args
    intervalBs = np.zeros(l)
    intervalPs = np.zeros(l)

    def calc_p(i, t): # ti:
        if t ==0: return 1.
        ti = times[i+1]
        Ai = intervalAs[i]
        Bi = ((1 -2*(1-rho[i])* calc_p(i+1, ti)) * lam[i] +mu[i]+psi[i]) /Ai
        p = lam[i] + mu[i] + psi[i]
        p -= Ai * ( ((1+Bi) -(1-Bi) * exp(-Ai*(t-ti)) ) / ((1+Bi) +(1-Bi) * exp(-Ai*(t-ti) )) )
        p = p/(2. * lam[i])
        intervalBs[i] = Bi
        intervalPs[i] = p
        return p

    p = calc_p(i, t)
    Ai_t = intervalAs[i]*(t-times[i+1])
    qi_t = (log(4)-Ai_t) - (2* log( exp(-Ai_t) *(1-intervalBs[i]) + (1+intervalBs[i]) ) )
    return qi_t

def calc_qt(i, t, args):
    [intervalAs, lam, mu, psi, times, l, rho] = args
    qt = .5 * ( calc_q(i, t, args) - (lam[i]+mu[i]+psi[i])*(t-times[i+1]) )
    return qt

def likelihood_rangeFBD(times, psi, lam, mu, ts, te, k=[], intervalAs=[], int_indx=[], div_traj=[], rho=0, FALAmodel=0,alpha=1):
    l  = len(times)-1 # number of intervals (combination of qShift, Kl, Km)
    if rho: pass
    else: rho = np.zeros(l)

    # only recompute if updating lam, mu, or psi
    if len(intervalAs) > 0: pass
    else: intervalAs = calcAi(lam,mu,psi)

    # only recompute if updating times or ts/te
    if len(int_indx) > 0:
        bint = int_indx[0]
        oint = int_indx[1]
        dint = int_indx[2]
    else:
        bint = np.digitize(ts, times)-1 # which interval ts happens
        oint = np.digitize(FA, times)-1 # which interval FA happens
        dint = np.digitize(te, times)-1 # which interval te happens
        bint[bint<0] = 0
        int_indx = [bint, oint, dint]

    # only need to recompute div_traj when updating ts/te
    if len(div_traj) > 0: pass
    else: div_traj = get_DT_FBDrange(ts,ts,te)
    # print div_traj
    # print np.sort(ts)
    # print np.sort(te)

    # only need to update when changing times
    # if using FALA model: k == kappa' (Theorem 14 Stadler et al. 2018 JTB)
    if len(k) > 0: pass
    else: k = get_k(array_all_fossils, times)

    term0 = -log(lam[0])
    if rho[0]>0: term0 += log(rho[0])*(n-m)

    term1 = np.sum(k*log(psi))

    term2 = log(lam[bint])

    term3 = log(mu[dint])
    term3[te==0] = 0. # only counts for extinct lineages

    gamma_i = div_traj-1. # attachment points: diversity -1
    #gamma_i[gamma_i<=0] = 1
    gamma_i[0] = 1. # oldest range gets gamma= 1
    if np.min(gamma_i)<1: return [-np.inf, intervalAs, int_indx, div_traj, k]

    term4 = 0
    term4_c = 0
    
    if hasFoundPyRateC and alpha==1: # use the C version for term 4
        l_bint = bint.tolist()
        l_dint = dint.tolist()
        l_oint = oint.tolist()
        term4_c = PyRateC_FBD_T4(tot_number_of_species, l_bint, l_dint, l_oint, intervalAs, lam, mu, psi, rho, gamma_i, times, ts, te, FA)
    
    else:
        log_gamma_i = log(gamma_i)
        for i in range(tot_number_of_species):
            lik_sp_i = []
            for G_i in range(4):
                psi_g = np.array([get_gamma_rates(i)[G_i] for i in psi])                
                intervalAs = calcAi(lam,mu,psi_g)
                args = [intervalAs, lam, mu, psi_g, times, l, rho]

                term4_q  = calc_q(bint[i],ts[i],args)-calc_q(oint[i], FA[i],args)
                term4_qt = calc_qt(oint[i], FA[i],args)-calc_qt(dint[i], te[i],args)

                qj_1= 0
                for j in range(bint[i], oint[i]):
                    qj_1 += calc_q(j+1, times[j+1], args)

                qtj_1= 0
                for j in range(oint[i], dint[i]):
                    qtj_1 += calc_qt(j+1, times[j+1], args)

                term4_qj = qj_1 + qtj_1
                lik_sp_i.append(log_gamma_i[i] + term4_q + term4_qt + term4_qj)
            lik_sp_i = np.log(np.sum(np.exp(np.array(lik_sp_i))))
            term4_c += lik_sp_i 

        
    
    if not hasFoundPyRateC or sanityCheckForPyRateC: # We use the python version if PyRateC not found or if sanity check is asked
        log_gamma_i = log(gamma_i)
        args = [intervalAs, lam, mu, psi, times, l, rho]

        for i in range(tot_number_of_species):

            term4_q  = calc_q(bint[i],ts[i],args)-calc_q(oint[i], FA[i],args)
            term4_qt = calc_qt(oint[i], FA[i],args)-calc_qt(dint[i], te[i],args)

            qj_1= 0
            for j in range(bint[i], oint[i]):
                qj_1 += calc_q(j+1, times[j+1], args)

            qtj_1= 0
            for j in range(oint[i], dint[i]):
                qtj_1 += calc_qt(j+1, times[j+1], args)

            term4_qj = qj_1 + qtj_1
            term4 += log_gamma_i[i] + term4_q + term4_qt + term4_qj # + term3[i] + term4[i]

        if hasFoundPyRateC and sanityCheckForPyRateC: # Sanity check only done if needed
            absDivergence = abs(term4 - term4_c)
            if absDivergence > sanityCheckThreshold:
                print("[WARNING] PyRateC_FBD_T4 diverged for more than ", sanityCheckThreshold, " (", absDivergence, ")")

    if hasFoundPyRateC:
        term4 = term4_c
    
    if FALAmodel == 2:
        term5 = np.sum(psi * get_L_FBDrange(times,FA,LO))
    else: 
        term5 = 0
    
    
    likelihood = np.sum([term1, term0+sum(term2),sum(term3), term4, term5])
    res = [likelihood, intervalAs, int_indx, div_traj, k]

    return res

######  END FUNCTIONS for FBD Range  ########



###### BEGIN FUNCTIONS for BDMCMC ########

def born_prm(times, R, ind, tse):
    #B=max(1./times[0], np.random.beta(1,(len(R)+1))) # avoid time frames < 1 My
    #B=np.random.beta(1,(len(R)+1))
    alpha=zeros(len(R)+1)+lam_s
    B=np.maximum(1./times[0], np.random.dirichlet(alpha,1)[0][0])
    Q=np.diff(times*(1-B))
    ADD=-B*times[0]
    Q1=insert(Q, ind,ADD)
    Q2=times[0]+cumsum(Q1)
    Q3=insert(Q2,0,times[0])
    Q3[len(Q3)-1]=0
    #n_R= insert(R, ind, R[max(ind-1,0)]) # R[np.random.randint(0,len(R))]
    #print "old", R, n_R,

    # better proposals for birth events
    #if len(R)>1: R_init=mean(R)
    #else: R_init=R
    if np.random.random()>.5:
        n_R= insert(R, ind, init_BD(1))
    else:
        R_init = R[np.maximum(ind-1,0)]
        n_R= insert(R, ind,update_parameter(R_init,0,5,.05,1))


    #print "new", n_R
    n_times= sort(Q3)[::-1]
    go= 1
    for j in range(len(n_times)-1):
        up,lo=n_times[j],n_times[j+1]
        if len(np.intersect1d((tse <= up).nonzero()[0], (tse > lo).nonzero()[0]))<=1:
            go= 0
            #print len(np.intersect1d((tse <= up).nonzero()[0], (tse > lo).nonzero()[0])), up , lo

    if min(abs(np.diff(n_times)))<=1: return times, R
    elif go == 0: return times, R
    else: return n_times, n_R

    #R= insert(R, ind, init_BD(1))
    #n_times= sort(Q3)[::-1]
    #return n_times, R

def kill_prm(times, R, ind):
    P=np.diff(times)         # time intervals
    Pi= abs(P[ind]/times[0])
    P2= P/(1-Pi)
    P3=np.delete(P2, ind)    # remove interval
    Q2=times[0]+cumsum(P3)   # re-adjust length remaining time frames
    Q3=insert(Q2,0,times[0]) # add root
    Q3[len(Q3)-1]=0
    R=np.delete(R,ind)       # remove corresponding rate
    n_times= sort(Q3)[::-1]  # reverse array
    #ind_rm= max(1,ind)
    #n_times= np.delete(times,ind_rm)
    return n_times, R

def estimate_delta(likBDtemp, R,par,times, ts, te, cov_par, ind,deathRate,n_likBD,len_L):
    # ESTIMATE DELTAS (individual death rates)
    #print "now",R,times
    for temp in range(0,len(R)):
        if par=="l":
            temp_l=temp
            cov_par_one=cov_par[0]
        else:
            temp_l=temp+ind
            cov_par_one=cov_par[1]
        n_times, n_rates=kill_prm(times, R, temp)
        #print temp,n_rates,n_times

        tempL=0
        for temp1 in range(len(n_times)-1):
            up, lo = n_times[temp1], n_times[temp1+1]
            l = n_rates[temp1]
            #print up,lo,l,n_rates
            args=[ts, te, up, lo, l, par, cov_par_one,len_L]
            tempL+=BPD_partial_lik(args)
        #print "LIK",    tempL, sum(likBDtemp[ind:ind+len(R)])
        D=np.minimum(tempL - np.sum(likBDtemp[ind:ind + len(R)]), 100) # to avoid overflows
        deathRate[temp_l]=exp(D)
    return deathRate #, n_likBD

def Alg_3_1(arg):
    [it,likBDtemp, ts, te, L,M, timesL, timesM, cov_par,len_L]=arg
    cont_time=0
    #T=max(ts)
    #priorBD= get_hyper_priorBD(timesL,timesM,L,M,T)
    while cont_time<len_cont_time:
        #print cont_time, sum(likBDtemp), len(L),len(M) #, timesL, L
        deathRate=zeros(len(likBDtemp))
        n_likBD=zeros(len(likBDtemp))

        # ESTIMATE DELTAS (individual death rates)
        if len(L)>1 :  # SPECIATION RATES
            deathRate=estimate_delta(likBDtemp, L,"l",timesL, ts, te, cov_par, 0,deathRate,n_likBD,len(L))
        if len(M)>1:  # EXTINCTION RATES
            deathRate=estimate_delta(likBDtemp, M,"m",timesM, ts, te, cov_par, len(L),deathRate,n_likBD,len(L))
        deltaRate=sum(deathRate)
        #print it, "DELTA:", round(deltaRate,3), "\t", deathRate, len(L), len(M),round(sum(likBDtemp),3)
        cont_time += np.random.exponential(1. / np.minimum((deltaRate+birthRate), 100000))
        if cont_time>len_cont_time: break

        else: # UPDATE MODEL
            Pr_birth= birthRate/(birthRate+deltaRate)
            Pr_death= 1-Pr_birth
            #IND=-1
            #print sum(likBDtemp), Pr_birth, Pr_death, deathRate
            if np.random.random()<Pr_birth or len(L)+len(M)==2: # ADD PARAMETER
                LL=len(L)+len(M)
                if np.random.random()>.5 and use_Death_model == 0:
                    ind=np.random.randint(0,len(L))
                    timesL, L = born_prm(timesL, L, ind, ts[SP_in_window])
                    IND=ind
                else:
                    ind=np.random.randint(0,len(M))
                    timesM, M = born_prm(timesM, M, ind, te[EX_in_window])
                    IND=ind+len(timesL)-1
                if LL == len(L)+len(M): IND=-1
            else: # REMOVE PARAMETER
                probDeath=np.cumsum(deathRate/deltaRate) # cumulative prob (used to randomly sample one
                r=np.random.random()                          # parameter based on its deathRate)
                probDeath=sort(append(probDeath, r))
                ind=np.where(probDeath==r)[0][0] # just in case r==1
                if ind < len(L): timesL, L = kill_prm(timesL, L, ind)
                else: timesM, M = kill_prm(timesM, M, ind-len(L))
            # UPDATE LIKELIHOODS
            tempL=zeros(len(L)+len(M))
            tempP=zeros(len(L)+len(M))
            for temp_l in range(len(timesL)-1):
                up, lo = timesL[temp_l], timesL[temp_l+1]
                l = L[temp_l]
                args=[ts, te, up, lo, l, 'l', cov_par[0],len(L)]
                tempL[temp_l]=BPD_partial_lik(args)
            for temp_m in range(len(timesM)-1):
                up, lo = timesM[temp_m], timesM[temp_m+1]
                m = M[temp_m]
                args=[ts, te, up, lo, m, 'm', cov_par[1],len(L)]
                tempL[len(timesL)-1+temp_m]=BPD_partial_lik(args)
            likBDtemp=tempL

            #priorBDnew= get_hyper_priorBD(timesL,timesM,L,M,T)-priorBD
            #print IND, timesL, timesM
            #if IND > -1: likBDtemp[IND] += priorBDnew


    #if priorBDnew-priorBD >= log(np.random.random()):
    #    return likBDtemp, L,M, timesL, timesM, cov_par
    #else:
    #    return arg[1],arg[4], arg[5],arg[6], arg[7],cov_par
    return likBDtemp, L,M, timesL, timesM, cov_par

######    END FUNCTIONS for BDMCMC ######

####### BEGIN FUNCTIONS for RJMCMC #######
def random_choice(vector):
    ind = np.random.choice(list(range(len(vector))))
    return [vector[ind], ind]

def add_DoubleShift_RJ_rand_gamma(rates,times):
    r_time, r_time_ind = random_choice(np.diff(times))
    delta_t_prime      = np.random.uniform(0,r_time,2)
    t_prime            = times[r_time_ind] + delta_t_prime
    times_prime        = np.sort(np.array(list(times)+list(t_prime)))[::-1]
    a,b                = shape_gamma_RJ,rate_gamma_RJ
    rate_prime         = np.random.gamma(a,scale=1./b,size=2)
    log_q_prob         = -sum(prior_gamma(rate_prime,a,b)) +log(abs(r_time)) # prob latent parameters: Gamma pdf, - (Uniform pdf )
    #print "PROB Q", prior_gamma(rate_prime,a,b), -log(1/abs(r_time))
    rates_prime        = np.insert(rates,r_time_ind+1,rate_prime)
    Jacobian           = 0 # log(1)
    return rates_prime,times_prime,log_q_prob+Jacobian

def remove_DoubleShift_RJ_rand_gamma(rates,times):
    rm_shift_ind  = np.random.choice(list(range(2,len(times)-1)))
    rm_shift_ind  = np.array([rm_shift_ind-1,rm_shift_ind])
    rm_shift_time = times[rm_shift_ind]
    dT            = abs(times[rm_shift_ind[1]+1]-times[rm_shift_ind[0]-1]) # if rm t_i: U[t_i-1, t_i+1]
    times_prime   = np.setdiff1d(times, rm_shift_time)[::-1]
    rm_rate       = rates[rm_shift_ind]
    a,b           = shape_gamma_RJ,rate_gamma_RJ
    log_q_prob    = sum(prior_gamma(rm_rate,a,b)) -log(dT) # log_q_prob_rm = 1/(log_q_prob_add)
    rates_prime   = np.delete(rates,rm_shift_ind)
    Jacobian      = 0 # log(1)
    return rates_prime,times_prime,log_q_prob+Jacobian



def add_shift_RJ_rand_gamma(rates,times):
    if fix_edgeShift==1: # min and max bounds
        random_indx = np.random.choice(list(range(1,len(times)-2)))
        r_time, r_time_ind = np.diff(times)[random_indx],random_indx
    elif fix_edgeShift==2: # max bound
        random_indx = np.random.choice(list(range(1,len(times)-1)))
        r_time, r_time_ind = np.diff(times)[random_indx],random_indx
    elif fix_edgeShift==3: # min bound
        random_indx = np.random.choice(list(range(0,len(times)-2)))
        r_time, r_time_ind = np.diff(times)[random_indx],random_indx
    else:
        r_time, r_time_ind = random_choice(np.diff(times))
    delta_t_prime      = np.random.uniform(0,r_time)
    t_prime            = times[r_time_ind] + delta_t_prime
    times_prime        = np.sort(np.array(list(times)+[t_prime]))[::-1]
    a,b                = shape_gamma_RJ,rate_gamma_RJ
    rate_prime         = np.random.gamma(a,scale=1./b)
    log_q_prob         = -prior_gamma(rate_prime,a,b) +log(abs(r_time)) # prob latent parameters: Gamma pdf, - (Uniform pdf )
    #print "PROB Q", prior_gamma(rate_prime,a,b), -log(1/abs(r_time))
    rates_prime        = np.insert(rates,r_time_ind+1,rate_prime)
    Jacobian           = 0 # log(1)
    return rates_prime,times_prime,log_q_prob+Jacobian

def remove_shift_RJ_rand_gamma(rates,times):
    if fix_edgeShift==1:  # min and max bounds
        random_indx = np.random.choice(list(range(2,len(times)-2)))
    elif fix_edgeShift==2: # max bound
        random_indx = np.random.choice(list(range(2,len(times)-1)))
    elif fix_edgeShift==3: # min bound
        random_indx = np.random.choice(list(range(1,len(times)-2)))
    else:
        random_indx = np.random.choice(list(range(1,len(times)-1)))
    rm_shift_ind  = random_indx
    rm_shift_time = times[rm_shift_ind]
    dT            = abs(times[rm_shift_ind+1]-times[rm_shift_ind-1]) # if rm t_i: U[t_i-1, t_i+1]
    times_prime   = times[times != rm_shift_time]
    rm_rate       = rates[rm_shift_ind] ## CHECK THIS: could also be rates[rm_shift_ind-1] ???
    a,b           = shape_gamma_RJ,rate_gamma_RJ
    log_q_prob    = prior_gamma(rm_rate,a,b) -log(dT) # log_q_prob_rm = 1/(log_q_prob_add)
    rates_prime   = rates[rates != rm_rate]
    Jacobian      = 0 # log(1)
    return rates_prime,times_prime,log_q_prob+Jacobian

def add_shift_RJ_weighted_mean(rates,times ):
    if fix_edgeShift==1: # min and max bounds
        random_indx = np.random.choice(list(range(1,len(times)-2)))
        r_time, r_time_ind = np.diff(times)[random_indx],random_indx
    elif fix_edgeShift==2: # max bound
        random_indx = np.random.choice(list(range(1,len(times)-1)))
        r_time, r_time_ind = np.diff(times)[random_indx],random_indx
    elif fix_edgeShift==3: # min bound
        random_indx = np.random.choice(list(range(0,len(times)-2)))
        r_time, r_time_ind = np.diff(times)[random_indx],random_indx
    else:
        r_time, r_time_ind = random_choice(np.diff(times))
    delta_t_prime           = np.random.uniform(0,r_time)
    t_prime                 = times[r_time_ind] + delta_t_prime
    times_prime             = np.sort(np.array(list(times)+[t_prime]))[::-1]
    time_i1                 = times[r_time_ind]
    time_i2                 = times[r_time_ind+1]
    try:
        p1 = (time_i1-t_prime)/(time_i1-time_i2)
        p2 = (t_prime-time_i2)/(time_i1-time_i2)
        u = np.random.beta(shape_beta_RJ,shape_beta_RJ)  #np.random.random()
        rate_i                  = rates[r_time_ind]
        rates_prime1            = exp( log(rate_i)-p2*log((1-u)/u) )
        rates_prime2            = exp( log(rate_i)+p1*log((1-u)/u) )
        rates_prime             = np.insert(rates,r_time_ind+1,rates_prime2)
        #print p1+p2
        #print u,rates_prime1, rate_i,rates_prime2
        #print time_i1,times_prime,time_i2
        rates_prime[r_time_ind] = rates_prime1
        if np.isfinite(r_time) and np.isfinite(u) and u > 0:
            log_q_prob              = log(abs(r_time))-prior_sym_beta(u,shape_beta_RJ) # prob latent parameters: Gamma pdf
            Jacobian                = 2*log(rates_prime1+rates_prime2)-log(rate_i)
        else:
            print(delta_t_prime, r_time, u)
            print(times)
            print(time_i1, time_i2)
    except:
        print(delta_t_prime, r_time)
        print(times)
        print(time_i1, time_i2)
    return rates_prime,times_prime,log_q_prob+Jacobian

def remove_shift_RJ_weighted_mean(rates,times):
    if fix_edgeShift==1:  # min and max bounds
        random_indx = np.random.choice(list(range(2,len(times)-2)))
    elif fix_edgeShift==2: # max bound
        random_indx = np.random.choice(list(range(2,len(times)-1)))
    elif fix_edgeShift==3: # min bound
        random_indx = np.random.choice(list(range(1,len(times)-2)))
    else:
        random_indx = np.random.choice(list(range(1,len(times)-1)))
    rm_shift_ind  = random_indx
    t_prime       = times[rm_shift_ind]
    time_i1       = times[rm_shift_ind-1]
    time_i2       = times[rm_shift_ind+1]
    dT            = abs(times[rm_shift_ind+1]-times[rm_shift_ind-1]) # if rm t_i: U[t_i-1, t_i+1]
    times_prime   = times[times != t_prime]
    p1 = (time_i1-t_prime)/(time_i1-time_i2)
    p2 = (t_prime-time_i2)/(time_i1-time_i2)
    rate_i1       = rates[rm_shift_ind-1]
    rate_i2       = rates[rm_shift_ind]
    rate_prime    = exp(p1 *log(rate_i1) + p2 *log(rate_i2))
    #print p1,p2
    #print rate_i1, rate_i2,rate_prime
    #print t_prime, times_prime
    rm_rate       = rates[rm_shift_ind]
    rates_prime   = rates[rates != rm_rate]
    rates_prime[rm_shift_ind-1] = rate_prime
    #print rates
    #print rates_prime
    u             = 1./(1+rate_i2/rate_i1) # == rate_i1/(rate_i1+rate_i2)
    log_q_prob    = -log(dT)+prior_sym_beta(u,shape_beta_RJ) # log_q_prob_rm = 1/(log_q_prob_add)
    Jacobian      = log(rate_prime)-(2*log(rate_i1+rate_i2))
    return rates_prime,times_prime,log_q_prob+Jacobian

def RJMCMC(arg):
    [L,M, timesL, timesM,maxFA,minLA]=arg
    r=np.random.random(3)
    newL,log_q_probL = L,0
    newM,log_q_probM = M,0
    
    timesL_init = timesL + 0
    timesM_init = timesM + 0
    timesL = timesL + 0
    timesM = timesM + 0
    timesL[0] = maxFA
    timesM[0] = maxFA
    timesL[len(timesL)-1] = minLA
    timesM[len(timesM)-1] = minLA    
    newtimesL = timesL
    newtimesM = timesM
    
    

    if r[0]>sample_shift_mu:
        # ADD/REMOVE SHIFT LAMBDA
        if r[1]>0.5:
            if r[2]>0.5 or allow_double_move==0:
                newL,newtimesL,log_q_probL = add_shift_RJ(L,timesL)
            else:
                newL,newtimesL,log_q_probL = add_DoubleShift_RJ_rand_gamma(L,timesL)
        # if 1-rate model this won't do anything, keeping the frequency of add/remove equal
        elif len(L)> min_allowed_n_rates: # defined for the edgeShift model
            if r[2]>0.5 or allow_double_move==0:
                newL,newtimesL,log_q_probL = remove_shift_RJ(L,timesL)
            elif len(L)>2:
                newL,newtimesL,log_q_probL = remove_DoubleShift_RJ_rand_gamma(L,timesL)
    else:
        # ADD/REMOVE SHIFT MU
        if r[1]>0.5:
            if r[2]>0.5 or allow_double_move==0:
                newM,newtimesM,log_q_probM = add_shift_RJ(M,timesM)
            else:
                newM,newtimesM,log_q_probM = add_DoubleShift_RJ_rand_gamma(M,timesM)
        # if 1-rate model this won't do anything, keeping the frequency of add/remove equal
        elif len(M)> min_allowed_n_rates: # defined for the edgeShift model
            if r[2]>0.5 or allow_double_move==0:
                newM,newtimesM,log_q_probM = remove_shift_RJ(M,timesM)
            elif len(M)>2:
                newM,newtimesM,log_q_probM = remove_DoubleShift_RJ_rand_gamma(M,timesM)
    
    newtimesL[0] = timesL_init[0]
    newtimesM[0] = timesM_init[0]
    newtimesL[len(newtimesL)-1] = timesL_init[len(timesL_init)-1]
    newtimesM[len(newtimesM)-1] = timesM_init[len(timesM_init)-1]
    #print(newtimesL, timesL_init, maxFA,minLA)
    return newL,newtimesL,newM,newtimesM,log_q_probL+log_q_probM

def get_post_rj_HP(xl,xm):
    G_shape_rjHP = 2. # 1.1
    G_rate_rjHP  = 1. # 0.1 # mode at 1
    n = 2 # sp, ex
    a = G_shape_rjHP + xl + xm
    b = G_rate_rjHP + n
    Poi_lambda_rjHP = np.random.gamma(a,1./b)
    #print "Mean Poi_lambda:", a/b
    return Poi_lambda_rjHP

def Poisson_prior(k,rate):
    return k*log(rate) - rate - sum(log(np.arange(1,k+1)))

####### BEGIN FUNCTIONS for DIRICHLET PROCESS PRIOR #######

def random_choice_P(vector):
    probDeath=np.cumsum(vector/sum(vector)) # cumulative prob (used to randomly sample one
    r=np.random.random()                          # parameter based on its deathRate)
    probDeath=sort(append(probDeath, r))
    ind=np.where(probDeath==r)[0][0] # just in case r==1
    return [vector[ind], ind]

def calc_rel_prob(log_lik):
    rel_prob=np.exp(log_lik-np.max(log_lik))
    return rel_prob/np.sum(rel_prob)

def G0(alpha=1.5,beta=5,n=1):
    #return np.array([np.random.random()])
    return np.random.gamma(shape=alpha,scale=1./beta,size=n)
    #return init_BD(n)


def DDP_gibbs_sampler(arg): # rate_type = "l" or "m" (for speciation/extinction respectively)
    [ts,te,parA,ind,time_frames,alpha_par_Dir,rate_type]=arg
    # par: parameters for each category
    n_data=len(ind)
    # GIBBS SAMPLER for NUMBER OF CATEGORIES - Algorithm 4. (Neal 2000)
    par=parA # parameters for each category
    eta = np.array([len(ind[ind==j]) for j in range(len(par))]) # number of elements in each category
    u1 = np.random.uniform(0,1,n_data) # init random numbers
    new_lik_vec=np.zeros(n_data) # store new sampled likelihoods
    new_alpha_par_Dir = 0 + cond_alpha_proposal(hp_gamma_shape,hp_gamma_rate,alpha_par_Dir,len(par),n_data)
    for i in range(0,n_data):
        up=time_frames[i]
        lo=time_frames[i+1]
        k1 = len(par)

        if len(ind[ind==ind[i]])==1: # is singleton
            k1 = k1 - 1
            par_k1 = par
            if u1[i]<= k1/(k1+1.): pass
            else: ind[i] = k1 + 1 # this way n_ic for singleton is not 0
        else: # is not singleton
            par_k1 = np.concatenate((par,G0()), axis=0)

        # construct prob vector FAST!
        lik_vec=BPD_partial_lik([ts,te,up,lo,par_k1,rate_type,0,1])
        rel_lik = calc_rel_prob(lik_vec)
        if len(par_k1)>len(eta): # par_k1 add one element only when i is not singleton
            eta[ind[i]] -= 1
            eta_temp=np.append(eta,new_alpha_par_Dir/(k1+1.))
        else: eta_temp = eta
        P=eta_temp*rel_lik

        # randomly sample a new value for indicator ind[i]
        IND = random_choice_P(P)[1]  # numpy.random.choice(a, size=None, replace= 1, p=None)
        ind[i] = IND # update ind vector
        if IND==(len(par_k1)-1): par = par_k1 # add category


        # Change the state to contain only those par are now associated with an observation
        # create vector of number of elements per category
        eta = np.array([len(ind[ind==j]) for j in range(len(par))])
        # remove parameters for which there are no elements
        par = par[eta>0]
        # rescale indexes
        ind_rm = (eta==0).nonzero()[0] # which category has no elements
        if len(ind_rm)>0: ind[ind>=ind_rm] = ind[ind>=ind_rm]-1
        # update eta
        eta = np.delete(eta,ind_rm)

        # Update lik vec
        new_lik_vec[i]=lik_vec[IND]


    likA = sum(new_lik_vec)
    parA = par
    return likA,parA, ind,new_alpha_par_Dir


def get_rate_HP(n,target_k,hp_gamma_shape):
    def estK(alpha,N):
        return sum([alpha/(alpha+i-1) for i in range(1,int(N+1))])

    def opt_gamma_rate(a):
        a= abs(a[0])
        ea =estK(a,n)
        return exp(abs( ea-target_k ))
    # from scipy.optimize import fmin_powell as Fopt1
    opt = Fopt1(opt_gamma_rate, [np.array(0.001)], full_output=1, disp=0)
    expected_cp=abs(opt[0])
    hp_gamma_rate = expected_cp/hp_gamma_shape
    return hp_gamma_rate


####### END FUNCTIONS for DIRICHLET PROCESS PRIOR #######

def get_init_values(mcmc_log_file, taxa_names, float_prec_f):
    tbl = np.loadtxt(mcmc_log_file,skiprows=1)
    last_row = np.shape(tbl)[0]-1
    head = next(open(mcmc_log_file)).split()
    ts_index_temp = [head.index(i) for i in head if "_TS" in i]
    te_index_temp = [head.index(i) for i in head if "_TE" in i]
    ts_index,te_index = [],[]
    for ts_i in ts_index_temp:
        sp=head[ts_i].split("_TS")[0]
        if sp in taxa_names: ts_index.append(ts_i)
    for te_i in te_index_temp:
        sp=head[te_i].split("_TE")[0]
        if sp in taxa_names: te_index.append(te_i)
    if len(ts_index) != len(ts_index_temp):
        print("Excluded", len(ts_index) - len(ts_index_temp), "taxa")

    alpha_pp=1
    cov_par = np.zeros(3)
    try:
        q_rates_index = np.array([head.index("alpha"), head.index("q_rate")])
        q_rates = tbl[last_row,q_rates_index]
    except:
        q_rates_index = [head.index(i) for i in head if i.startswith('q_')]
        q_rates = tbl[last_row,q_rates_index]
        try:
            alpha_pp = tbl[last_row,head.index("alpha")]
        except: pass
    ts = tbl[last_row,ts_index]
    te = tbl[last_row,te_index]
    if len(fixed_times_of_shift)>0: # fixShift
        try:
            hyp_index = [head.index("hypL"), head.index("hypM")]
            l_index = [head.index(i) for i in head if "lambda_" in i]
            m_index = [head.index(i) for i in head if "mu_" in i]
            lam = tbl[last_row,l_index]
            mu  = tbl[last_row,m_index]
            hyp = tbl[last_row,hyp_index]
        except:
            lam = np.array([float(len(ts))/sum(ts-te)      ])    # const rate ML estimator
            mu  = np.array([float(len(te[te>0]))/sum(ts-te)])    # const rate ML estimator
            hyp = np.ones(2)
    else:
        lam = np.array([float(len(ts))/sum(ts-te)      ])    # const rate ML estimator
        mu  = np.array([float(len(te[te>0]))/sum(ts-te)])    # const rate ML estimator
        hyp = np.ones(2)

    if BDNNmodel:
        from pyrate_lib.bdnn_lib import bdnn_reshape_w
        pkl_file = mcmc_log_file.replace("_mcmc.log", "") + ".pkl"
        bdnn_obj = load_pkl(pkl_file)
        cov_par = [0] * 6
        if BDNNmodel in [1, 3]:
            w_lam_index = [head.index(i) for i in head if "w_lam_" in i]
            w_mu_index = [head.index(i) for i in head if "w_mu_" in i]
            w_lam = tbl[last_row, w_lam_index].reshape((1, len(w_lam_index)))
            w_mu = tbl[last_row, w_mu_index].reshape((1, len(w_mu_index)))
            w_lam = float_prec_f(w_lam)
            w_mu = float_prec_f(w_mu)
            w_lam = bdnn_reshape_w(w_lam, bdnn_obj, rate_type="diversification")[0]
            w_mu = bdnn_reshape_w(w_mu, bdnn_obj, rate_type="diversification")[0]
            cov_par[0] = w_lam
            cov_par[1] = w_mu
            cov_par[3] = tbl[last_row, head.index("t_reg_lam")]
            cov_par[4] = tbl[last_row, head.index("t_reg_mu")]
        if BDNNmodel in [2, 3]:
            w_q_index = [head.index(i) for i in head if "w_q_" in i]
            w_q = tbl[last_row, w_q_index].reshape((1, len(w_q_index)))
            w_q = bdnn_reshape_w(w_q, bdnn_obj, rate_type="sampling")[0]
            cov_par[2] = w_q
            cov_par[5] = tbl[last_row, head.index("t_reg_q")]

    return [ts,te,q_rates,lam,mu,hyp,alpha_pp, cov_par]

########################## MCMC #########################################

def MCMC(all_arg):
    [it,n_proc, I,sample_freq, print_freq, temperatures, burnin, marginal_frames, arg]=all_arg
    if it==0: # initialize chain
        print("initializing chain...")
        if fix_SE == 1:
            maxFA,minLA = np.max(fixed_ts), np.min(fixed_te)
        else: 
            maxFA,minLA = np.max(FA), np.min(LO)
        
        
        if fix_SE == 1: tsA, teA = fixed_ts, fixed_te
        elif FBDrange==0: tsA, teA = init_ts_te(FA,LO)
        else:
            tsA, teA = init_ts_te_FBDrange(FA,LO)
            if FBDrange == 3:
                teA = LO
            res_FBD_A = []
        if restore_chain == 1 and fix_SE != 1:
            tsA_temp, teA_temp = init_ts_te(FA,LO)
            tsA, teA = restore_init_values[0], restore_init_values[1]
            # avoid incompatibilities due to age randomizations
            # print len(tsA[tsA<FA]),len(tsA)
            tsA[tsA<FA]=FA[tsA<FA]+1
            teA[teA>LO]=LO[teA>LO]-1
            teA[teA<0]=teA_temp[teA<0]
        maxTSA = np.max(tsA)
        
        timesLA, timesMA = init_times(maxTSA,time_framesL,time_framesM, np.min(teA))
        if len(fixed_times_of_shift) > 0:
            timesLA[1:-1], timesMA[1:-1] = fixed_times_of_shift, fixed_times_of_shift
        if fix_edgeShift > 0 and not BDNNmodel in [1, 3]:
            print(edgeShifts,fix_edgeShift)
            if fix_edgeShift == 1:
                timesLA, timesMA = init_times(edgeShifts[0],time_framesL,time_framesM, edgeShifts[1]) # starting shift tims within allowed window
                timesLA[0],timesMA[0]= maxTSA,maxTSA
                timesLA[1],timesMA[1]= edgeShifts[0],edgeShifts[0]
                timesLA[-2],timesMA[-2]= edgeShifts[1],edgeShifts[1]
            elif fix_edgeShift == 2: # max age edge shift
                timesLA, timesMA = init_times(edgeShifts[0],time_framesL,time_framesM, 0) # starting shift tims within allowed window
                timesLA[1],timesMA[1]= edgeShifts[0],edgeShifts[0]
            elif fix_edgeShift == 3: # min age edge shift
                timesLA, timesMA = init_times(maxTSA,time_framesL,time_framesM, edgeShifts[0]) # starting shift tims within allowed window
                timesLA[-2],timesMA[-2]= edgeShifts[0],edgeShifts[0]
            # if len(edgeShifts)==1:
            #    if args.edgeShifts[0] < np.inf:
            #        timesLA, timesMA = init_times(edgeShifts[0],time_framesL,time_framesM, 0) # starting shift tims within allowed window
            #        timesLA[1],timesMA[1]= edgeShifts[0],edgeShifts[0]
            #    else:
            #        timesLA, timesMA = init_times(maxTSA,time_framesL,time_framesM, edgeShifts[0]) # starting shift tims within allowed window
            #        timesLA[1],timesMA[1]= edgeShifts[0],edgeShifts[0]
            # if len(edgeShifts)>1:
            #    timesLA, timesMA = init_times(edgeShifts[0],time_framesL,time_framesM, edgeShifts[1]) # starting shift tims within allowed window
            #    timesLA[0],timesMA[0]= maxTSA,maxTSA
            #    timesLA[1],timesMA[1]= edgeShifts[0],edgeShifts[0]
            #    timesLA[-2],timesMA[-2]= edgeShifts[1],edgeShifts[1]
            print("times", timesLA)
            print("times", timesMA)
                #quit()
        if TDI<3:
            LA = init_BD(len(timesLA))
            MA = init_BD(len(timesMA))
            if restore_chain == 1:
                LAt = restore_init_values[3]
                MAt = restore_init_values[4]
                if len(LAt) == len(LA): LA = LAt # if restored mcmc has different number of rates ignore them
                if len(MAt) == len(MA): MA = MAt
            if use_ADE_model >= 1: MA = np.random.uniform(3,5,len(timesMA)-1)
            if use_Death_model == 1: LA = np.ones(1)
            if use_Birth_model == 1: MA = np.zeros(1)+init_M_rate
            if useDiscreteTraitModel == 1:
                LA = init_BD(len(lengths_B_events)+1)
                MA = init_BD(len(lengths_D_events)+1)

        elif TDI==3 : ### DPP
            LA = init_BD(1) # init 1 rate
            MA = init_BD(1) # init 1 rate
            indDPP_L = np.zeros(len(timesLA)-1).astype(int) # init category indexes
            indDPP_M = np.zeros(len(timesLA)-1).astype(int) # init category indexes
            alpha_par_Dir_L = np.random.uniform(0,1) # init concentration parameters
            alpha_par_Dir_M = np.random.uniform(0,1) # init concentration parameters
        else:
            LA = init_BD(len(timesLA))
            MA = init_BD(len(timesMA))
            if use_Death_model == 1: LA = np.ones(1)
            if use_Birth_model == 1: MA = np.zeros(1)+init_M_rate
            rj_cat_HP= 1
        
        q_ratesA,cov_parA = init_q_rates() # use 1 for symmetric PERT
        
        if BDNNmodel in [1, 3]:
            if use_time_as_trait or bdnn_timevar[0] or bdnn_dd or bdnn_loaded_tbls_timevar:
                if fix_edgeShift in [0, 3]: # no or min boundary
                    timesLA, timesMA = init_times(maxTSA, time_framesL_bdnn, time_framesM_bdnn, np.min(teA))
                else:
                    # maxTSA could be after the enforced edge shift; use whatever age
                    timesLA, timesMA = init_times(2.0 * fixed_times_of_shift_bdnn[0], time_framesL_bdnn, time_framesM_bdnn, 0.0)
                timesLA[1:-1], timesMA[1:-1] = fixed_times_of_shift_bdnn, fixed_times_of_shift_bdnn

            if bdnn_dd:
                binned_div = get_diversity(tsA, teA, timesLA, time_vec, bdnn_rescale_div, n_taxa)
                trait_tbl_NN[0][ :, :, div_idx_trt_tbl] = binned_div
                trait_tbl_NN[1][ :, :, div_idx_trt_tbl] = binned_div

            cov_parA = cov_par_init_NN
            nn_lamA, nn_muA = None, None
            if not nn_activate_all:
                nn_lamA = init_NN_output(trait_tbl_NN[0], cov_parA[0], float_prec_f)
                nn_muA = init_NN_output(trait_tbl_NN[1], cov_parA[1], float_prec_f)
            if restore_chain:
                cov_parA = restore_init_values[7]

            bdnn_prior_cov_parA = np.zeros(4)
            cov_par_update_f = np.array([0.1, 0.55, -1.0])
            if independ_reg:
                cov_par_update_f = np.array([0.05, 0.5, 0.55])

            # rates not updated and replaced by bias node
            if bdnn_const_baseline:
                LA = np.ones(len(timesLA) - 1)
                MA = np.ones(len(timesMA) - 1)

            bdnn_lam_ratesA, denom_lamA, nn_lamA = get_rate_BDNN_3D(cov_parA[3], trait_tbl_NN[0], cov_parA[0], nn_lamA,
                                                                    hidden_act_f, out_act_f,
                                                                    apply_reg, bias_node_idx, fix_edgeShift)
            bdnn_mu_ratesA, denom_muA, nn_muA = get_rate_BDNN_3D(cov_parA[4], trait_tbl_NN[1], cov_parA[1], nn_muA,
                                                                 hidden_act_f, out_act_f,
                                                                 apply_reg, bias_node_idx, fix_edgeShift)
            if use_time_as_trait or bdnn_timevar[0] or bdnn_dd or bdnn_loaded_tbls_timevar:
                bin_size_lam_mu = np.tile(np.abs(np.diff(timesLA)), n_taxa).reshape((n_taxa, len(timesLA) - 1))
                i_events_spA, i_events_exA, n_SA = get_events_ns(tsA, teA, timesLA, bin_size_lam_mu)
                likBDtempA = np.zeros((2, n_taxa))
                likBDtempA[0, :] = BDNN_fast_partial_lik([i_events_spA, n_SA, bdnn_lam_ratesA, apply_reg])
                likBDtempA[1, :] = BDNN_fast_partial_lik([i_events_exA, n_SA, bdnn_mu_ratesA, apply_reg])
            bdnn_prior_cov_parA[0] = np.sum([np.sum(prior_normal(cov_parA[0][i],prior_bdnn_w_sd[i])) for i in range(len(cov_parA[0]))])
            bdnn_prior_cov_parA[1] = np.sum([np.sum(prior_normal(cov_parA[1][i],prior_bdnn_w_sd[i])) for i in range(len(cov_parA[1]))])
            if prior_lam_t_reg[0] > 0:
                bdnn_prior_cov_parA[2] = np.log(prior_lam_t_reg[0]) - prior_lam_t_reg[0] * cov_parA[3]
            if prior_lam_t_reg[1] > 0 and independ_reg:
                bdnn_prior_cov_parA[3] = np.log(prior_lam_t_reg[1]) - prior_lam_t_reg[1] * cov_parA[4]
        
        alpha_pp_gammaA = 1.
        if TPP_model == 1: # init multiple q rates
            q_ratesA = np.zeros(time_framesQ)+q_ratesA[1]
        if restore_chain == 1 and fix_SE != 1:
            q_ratesA = restore_init_values[2]
            if TPP_model == 1:
                if len(q_ratesA) != time_framesQ:
                    q_ratesA=np.zeros(time_framesQ)+mean(q_ratesA)
        timevar_qnn = False
        if BDNNmodel in [2, 3]:
            cov_parA = cov_par_init_NN
            nn_qA = None
            if not nn_activate_all:
                nn_qA = init_NN_output(trait_tbl_NN[2], cov_parA[2], float_prec_f)
            if restore_chain:
                cov_parA = restore_init_values[7]

            q_rates_tmp = q_ratesA
            if bdnn_ads >= 0.0:
                trait_tbl_NN[2] = add_taxon_age(tsA, teA, q_time_frames_bdnn, trait_tbl_NN[2])
                if not highres_q_repeats is None: #bdnn_ads > 0.0 and argsHPP == 0:
                    q_rates_tmp = q_ratesA[highres_q_repeats]
            qbin_ts_te = None
            bdnn_q_ratesA = np.zeros(n_taxa)
            if occs_sp.ndim == 2:
                qbin_ts_te = get_bin_ts_te(tsA, teA, q_time_frames_bdnn)
                timevar_qnn = True
                bdnn_q_ratesA = np.zeros((n_taxa, len(q_time_frames_bdnn) - 1))
            # singleton index to calculate preservation likelihood
            singleton_lik = copy_lib.deepcopy(singleton_mask)
            if singleton_lik.ndim == 2:
                singleton_lik = singleton_lik[:, 0].reshape(-1)
            rnd_layer_q = -1
            qnn_output_unregA, nn_qA = get_unreg_rate_BDNN_3D(trait_tbl_NN[2], cov_parA[2], nn_qA, hidden_act_f, out_act_f_q)
            q_multiA, denom_qA, norm_facA = get_q_multipliers_NN(cov_parA[5], qnn_output_unregA, singleton_mask, apply_reg_q, qbin_ts_te)
            bdnn_prior_qA = np.sum([np.sum(prior_normal(cov_parA[2][i], prior_bdnn_w_q_sd[i])) for i in range(len(cov_parA[2]))])
            if prior_lam_t_reg[-1] > 0:
                bdnn_prior_qA += np.log(prior_lam_t_reg[-1]) - prior_lam_t_reg[-1] * cov_parA[5]

        if est_COVAR_prior == 1:
            covar_prior = 1.
            cov_parA = np.random.random(3)*f_cov_par # f_cov_par is 0 or >0 depending on COVAR model
        else:
            covar_prior = covar_prior_fixed

        #if fix_hyperP == 0:    hyperPA=np.ones(2)
        hyperPA = hypP_par
        if restore_chain == 1:
            hyperPA = restore_init_values[5]

        if argsG == 0 and TPP_model == 0:
            q_ratesA[0] = 1
        if argsG == 1 and TPP_model == 1 and restore_chain == 1 and fix_SE != 1:
            alpha_pp_gammaA = restore_init_values[6]
        SA=np.sum(tsA-teA)
        W_shapeA=1.

        if analyze_tree >=1:
            MA = LA*np.random.random()
            r_treeA = np.random.random()
            m_treeA = np.random.random()
            if analyze_tree==4:
                r_treeA = np.random.random(len(phylo_times_of_shift))+2.
                m_treeA = np.random.random(len(phylo_times_of_shift))
                if args_bdc:
                    r_treeA = np.ones(len(phylo_times_of_shift))*0.8

    else: # restore values
        [itt, n_proc_,PostA, likA, priorA,tsA,teA,timesLA,timesMA,LA,MA,q_ratesA, cov_parA, lik_fossilA,likBDtempA]=arg
        SA=np.sum(tsA-teA)



    if fix_SE == 0:
        d1_ts, d1_te, tste_tune_obj = make_tste_tune_obj(LO, bound_te, d1)

    # start threads
    if num_processes>0: pool_lik = multiprocessing.Pool(num_processes) # likelihood
    if frac1>=0 and num_processes_ts>0: pool_ts = multiprocessing.Pool(num_processes_ts) # update ts, te
    tmp, marginal_lik, lik_tmp=0, zeros(len(temperatures)), 0

    while it<I:
        I_effective=I-burnin
        #if it==0: print (I_effective/len(temperatures))
        if it>0 and (it-burnin) % (I_effective/len(temperatures)) == 0 and it>burnin or it==I-1: # and it<I:
            if TDI==1:
                marginal_lik[tmp]=lik_tmp/((I_effective/len(temperatures))/sample_freq) # mean lik: Baele et al. 2012
                if it<I-1:
                    tmp += 1
                    lik_tmp=0
        temperature=temperatures[tmp]

        # update parameters
        ts,te=tsA, teA
        timesL,timesM=timesLA,timesMA
        try: # to make it work with MC3
            hyperP=hyperPA
            W_shape=W_shapeA
        except:
            hyperPA,W_shapeA=[1,1],[1]
            hyperP,W_shape=hyperPA,W_shapeA


        # GLOBALLY CHANGE TRAIT VALUE
        if model_cov >0 and not BDNNmodel:
            global con_trait
            con_trait=seed_missing(trait_values,meanGAUS,sdGAUS)

        if fix_SE == 1:
            rr=random.uniform(f_update_q,1)
            stop_update=0
            tsA, teA= fixed_ts, fixed_te
            lik_fossilA=np.zeros(1)
        elif it < fast_burnin:
            rr=random.uniform(f_update_se*0.95,1) # change update freq
            stop_update=I+1
        else:
            rr=random.uniform(0,1) #random.uniform(.8501, 1)
            stop_update=I+1

        if np.random.random() < 1./freq_Alg_3_1 and it>start_Alg_3_1 and TDI in [2,4]:
            stop_update=inf
            rr=1.5 # no updates

        if np.random.random() < 1./freq_dpp and TDI==3 and it > 1000: ### DPP
            stop_update=inf
            rr=1.5 # no updates

        if it>0 and (it-burnin) % (I_effective/len(temperatures)) == 0 and it>burnin or it==I-1: rr=1.5 # no updates when changing temp

        q_rates=np.zeros(len(q_ratesA))
        alpha_pp_gamma=alpha_pp_gammaA
        cov_par=np.zeros(3)
        if BDNNmodel in [2, 3]:
            cov_par = copy_lib.deepcopy(cov_parA)
            bdnn_prior_q = bdnn_prior_qA + 0.0
            q_multi = q_multiA + 0.0
            qnn_output_unreg = qnn_output_unregA + 0.0
            bdnn_q_rates = bdnn_q_ratesA + 0.0
            denom_q = denom_qA + 0.0
            norm_fac = norm_facA + 0.0
            nn_q = nn_qA
        if BDNNmodel in [1, 3]:
            cov_par = copy_lib.deepcopy(cov_parA)
            bdnn_prior_cov_par = bdnn_prior_cov_parA + 0.0
            bdnn_lam_rates = bdnn_lam_ratesA
            bdnn_mu_rates = bdnn_mu_ratesA
            denom_lam = denom_lamA + 0.0
            denom_mu = denom_muA + 0.0
            nn_lam = nn_lamA
            nn_mu = nn_muA
            if use_time_as_trait or bdnn_timevar[0] or bdnn_dd or bdnn_loaded_tbls_timevar:
                likBDtemp = likBDtempA + 0.0
                i_events_sp = i_events_spA
                i_events_ex = i_events_exA
                n_S = n_SA
        L,M = np.zeros(len(LA)),np.zeros(len(MA))
        tot_L = np.sum(tsA-teA)
        hasting=0
        
        # autotuning
        if TDI != 1: tmp=0
        mod_d1 = d1           # window size ts, te
        mod_d3= list_d3[tmp] # window size rates
        mod_d4= list_d4[tmp] # window size shift times

        move_type = 0 # move types: 1) ts/te; 2) q rates; 3) timesL/M; 4) L/M rates;

        updated_lam_mu = False
        ts_te_updated = 0
        cov_lam_updated = 0
        cov_mu_updated = 0
        cov_q_updated = 0

        if rr<f_update_se: # ts/te
            ts_te_updated = 1
            move_type = 1
            if FBDrange == 3:
                ts,te=update_ts_te(tsA, teA, mod_d1, bound_ts, bound_te, sample_extinction=0)
            else:
                if edge_indicator and it > 10000:
                    ts,te = update_ts_te_indicator(tsA,teA,mod_d1)
                elif tune_T_schedule[0] > 0:
                    ts_or_te_updated = np.random.randint(low=0, high=2, size=1)[0]
                    ts, te = update_ts_te_tune(tsA, teA, d1_ts, d1_te, FA, LO, bound_ts, bound_te,
                                               sample_extinction=ts_or_te_updated)
                else:
                    ts, te = update_ts_te(tsA, teA, mod_d1, bound_ts, bound_te)
                
            if use_gibbs_se_sampling or it < fast_burnin:
                if BDNNmodel in [1, 3]:
                    arg_taxon_rates = [tsA, teA, timesLA, timesMA, bdnn_lam_ratesA, bdnn_mu_ratesA]
                    sp_rates_L, sp_rates_M = get_taxon_rates_bdnn(arg_taxon_rates)
                    ts, te = gibbs_update_ts_te_bdnn(q_ratesA, sp_rates_L, sp_rates_M, np.sort(np.array([np.inf,0]+times_q_shift))[::-1],
                        bound_ts=bound_ts, bound_te=bound_te, tsA=tsA, teA=teA)
                
                elif sum(timesL[1:-1])==np.sum(times_q_shift):
                    ts,te = gibbs_update_ts_te(q_ratesA+LA+MA, q_ratesA+LA+MA, 
                        np.sort(np.array([np.inf,0]+times_q_shift))[::-1],
                        bound_ts=bound_ts, bound_te=bound_te, tsA=tsA, teA=teA)
                else:
                    times_q_temp = np.sort(np.array([np.inf,0]+times_q_shift))[::-1]
                    q_temp_time = np.sort(np.unique(list(times_q_shift)+list(timesLA[1:])+list(timesMA[1:])))[::-1]
                    q_rates_temp =  q_ratesA[np.digitize(q_temp_time,times_q_temp[1:])]
                    if len(LA)==1:
                        q_rates_temp_L = q_rates_temp + LA[0] + MA[0]
                    else:
                        q_rates_temp_L = q_rates_temp + LA[np.digitize(q_temp_time,timesLA[1:])] + MA[np.digitize(q_temp_time,timesMA[1:])] 
                    if len(MA)==1:
                        q_rates_temp_M = q_rates_temp + MA[0] + LA[0]
                    else:
                        q_rates_temp_M = q_rates_temp + MA[np.digitize(q_temp_time,timesMA[1:])] + LA[np.digitize(q_temp_time,timesLA[1:])]
                    ts,te = gibbs_update_ts_te(q_rates_temp_L,q_rates_temp_M,times_q_temp, 
                        bound_ts=bound_ts, bound_te=bound_te, tsA=tsA, teA=teA)

            if BDNNmodel:
                if bdnn_dd:
                    bdnn_divA = trait_tbl_NN[0][ :, :, div_idx_trt_tbl] + 0.0
                    binned_div = get_diversity(ts, te, timesLA, time_vec, bdnn_rescale_div, n_taxa)
                    trait_tbl_NN[0][ :, :, div_idx_trt_tbl] = binned_div
                    trait_tbl_NN[1][ :, :, div_idx_trt_tbl] = binned_div
                    # Recalculate bdnn rates
                    bdnn_lam_rates, denom_lam, nn_lam = get_rate_BDNN_3D(cov_parA[3], trait_tbl_NN[0], cov_parA[0], nn_lamA,
                                                                            hidden_act_f, out_act_f,
                                                                            apply_reg, bias_node_idx, fix_edgeShift)
                    bdnn_mu_rates, denom_mu, nn_mu = get_rate_BDNN_3D(cov_parA[4], trait_tbl_NN[1], cov_parA[1], nn_muA,
                                                                        hidden_act_f, out_act_f,
                                                                        apply_reg, bias_node_idx, fix_edgeShift)
                if BDNNmodel in [2, 3]:
                    if bdnn_ads >= 0.0:
                        trait_tbl_NN[2] = add_taxon_age(ts, te, q_time_frames_bdnn, trait_tbl_NN[2], tsA, teA)
                        rnd_layer_q = 0

            tot_L=np.sum(ts-te)
        
        elif rr<f_update_q: # q/alpha
            move_type = 2
            q_rates=np.zeros(len(q_ratesA))+q_ratesA
            if TPP_model == 1:
                q_rates, hasting = update_q_multiplier(q_ratesA,d=d2[1],f=f_qrate_update)
                if np.random.random()> 1./len(q_rates) and argsG == 1:
                    alpha_pp_gamma, hasting2 = update_multiplier_proposal(alpha_pp_gammaA,d2[0]) # shape prm Gamma
                    hasting += hasting2
            elif np.random.random()>.5 and argsG == 1:
                q_rates[0], hasting=update_multiplier_proposal(q_ratesA[0],d2[0]) # shape prm Gamma
            else:
                q_rates[1], hasting=update_multiplier_proposal(q_ratesA[1],d2[1]) #  preservation rate (q)

        elif rr < f_update_lm: # l/m
            updated_lam_mu = True
            if np.random.random()<f_shift and len(LA)+len(MA)>2:
                move_type = 3
                if fix_edgeShift > 0 and not BDNNmodel in [1, 3]:
                    if fix_edgeShift == 1:
                        timesL=update_times(timesLA, edgeShifts[0],edgeShifts[1],mod_d4,2,len(timesL)-2)
                        timesM=update_times(timesMA, edgeShifts[0],edgeShifts[1],mod_d4,2,len(timesM)-2)
                    elif fix_edgeShift == 2: # max age edge shift
                        timesL=update_times(timesLA, edgeShifts[0],min(te),mod_d4,2,len(timesL)-1)
                        timesM=update_times(timesMA, edgeShifts[0],min(te),mod_d4,2,len(timesM)-1)
                    elif fix_edgeShift == 3: # min age edge shift
                        timesL=update_times(timesLA,np.max(ts),edgeShifts[0],mod_d4,1,len(timesL)-2)
                        timesM=update_times(timesMA,np.max(ts),edgeShifts[0],mod_d4,1,len(timesM)-2)

                else:
                    timesL=update_times(timesLA, maxFA,minLA,mod_d4,1,len(timesL))
                    timesM=update_times(timesMA, maxFA,minLA,mod_d4,1,len(timesM))
            else:
                move_type = 4
                if TDI<2: #
                    if np.random.random()<.95 or est_hyperP == 0 or fix_hyperP == 1:
                        L,M,hasting=update_rates(LA,MA,3,mod_d3)
                        update_W_shape =1
                        if use_ADE_model == 1 and update_W_shape:
                            W_shape, hasting2 = update_multiplier_proposal(W_shapeA,1.1)
                            hasting+=hasting2
                    else:
                        hyperP,hasting = update_multiplier_proposal(hyperPA,d_hyperprior)
                else: # DPP or BDMCMC
                        L,M,hasting=update_rates(LA,MA,3,mod_d3)

        elif rr<f_update_cov: # cov
            # Do not update weights for lam/mu and q at the same time
            rr_bdnn = (np.random.random() - 0.5) * BDNNmodel in [3]
            if BDNNmodel in [1, 3] and rr_bdnn <= 0.0:
#                cov_lam_updated = 1
#                cov_mu_updated = 1
#                rnd_layer = np.random.randint(0, len(cov_parA[0]))
#                # update layers B rate
#                cov_par[0][rnd_layer] = update_parameter_normal_vec(cov_parA[0][rnd_layer], d=0.05, f= bdnn_update_f[rnd_layer], float_prec_f=float_prec_f)
#                bdnn_prior_cov_par[0] = np.sum([np.sum(prior_normal(cov_par[0][i],prior_bdnn_w_sd[i])) for i in range(len(cov_par[0]))])
#                # update layers D rate
#                cov_par[1][rnd_layer] = update_parameter_normal_vec(cov_parA[1][rnd_layer], d=0.05, f= bdnn_update_f[rnd_layer], float_prec_f=float_prec_f)
#                bdnn_prior_cov_par[1] = np.sum([np.sum(prior_normal(cov_par[1][i],prior_bdnn_w_sd[i])) for i in range(len(cov_par[1]))])
#                if BDNN_MASK_lam:
#                    for i_layer in range(len(cov_parA[0])):
#                        cov_par[0][i_layer] *= BDNN_MASK_lam[i_layer]
#                if BDNN_MASK_mu:
#                    for i_layer in range(len(cov_parA[1])):
#                        cov_par[1][i_layer] *= BDNN_MASK_mu[i_layer]
#                if prior_lam_t_reg[0] > 0:
#                    cov_par[3] = update_parameter(cov_parA[3], 0, 1, d=0.05, f=1)
#                    bdnn_prior_cov_par[2] = np.log(prior_lam_t_reg[0]) - prior_lam_t_reg[0] * cov_par[3]
#                    if not independ_reg:
#                        cov_par[4] = cov_par[3]
#                if prior_lam_t_reg[1] > 0 and independ_reg:
#                    cov_par[4] = update_parameter(cov_parA[4], 0, 1, d=0.05, f=1)
#                    bdnn_prior_cov_par[3] = np.log(prior_lam_t_reg[1]) - prior_lam_t_reg[1] * cov_par[4]
#                bdnn_lam_rates, denom_lam, nn_lam = get_rate_BDNN_3D(cov_par[3], trait_tbl_NN[0], cov_par[0], nn_lamA,
#                                                                     hidden_act_f, out_act_f,
#                                                                     apply_reg, bias_node_idx, fix_edgeShift,
#                                                                     rnd_layer)
#                bdnn_mu_rates, denom_mu, nn_mu = get_rate_BDNN_3D(cov_par[4], trait_tbl_NN[1], cov_par[1], nn_muA,
#                                                                  hidden_act_f, out_act_f,
#                                                                  apply_reg, bias_node_idx, fix_edgeShift,
#                                                                  rnd_layer)
#                # do not update all at once
                rr_cov_lam_mu = np.random.random()
                if rr_cov_lam_mu < cov_par_update_f[0] and prior_lam_t_reg[0] > 0:
                    # update treg lam
                    cov_lam_updated = 1
                    rnd_layer_lam = -1
                    cov_par[3] = update_parameter(cov_parA[3], 0, 1, d=0.05, f=1)
                    bdnn_prior_cov_par[2] = np.log(prior_lam_t_reg[0]) - prior_lam_t_reg[0] * cov_par[3]
                    if not independ_reg:
                        cov_par[4] = cov_par[3]
                elif rr_cov_lam_mu < cov_par_update_f[1]:
                    # update layers B rate
                    cov_lam_updated = 1
                    rnd_layer_lam = np.random.randint(0, len(cov_parA[0]))
                    cov_par[0][rnd_layer_lam] = update_parameter_normal_vec(cov_parA[0][rnd_layer_lam], d=0.05, f=bdnn_update_f[rnd_layer_lam], float_prec_f=float_prec_f)
                    bdnn_prior_cov_par[0] = np.sum([np.sum(prior_normal(cov_par[0][i],prior_bdnn_w_sd[i])) for i in range(len(cov_par[0]))])
                    if BDNN_MASK_lam:
                        for i_layer in range(len(cov_parA[0])):
                            cov_par[0][i_layer] *= BDNN_MASK_lam[i_layer]
                elif rr_cov_lam_mu < cov_par_update_f[2] and prior_lam_t_reg[1] > 0:
                    # update treg mu
                    cov_mu_updated = 1
                    rnd_layer_mu = -1
                    cov_par[4] = update_parameter(cov_parA[4], 0, 1, d=0.05, f=1)
                    bdnn_prior_cov_par[3] = np.log(prior_lam_t_reg[1]) - prior_lam_t_reg[1] * cov_par[4]
                else:
                    # update layers D rate
                    cov_mu_updated = 1
                    rnd_layer_mu = np.random.randint(0, len(cov_parA[1]))
                    cov_par[1][rnd_layer_mu] = update_parameter_normal_vec(cov_parA[1][rnd_layer_mu], d=0.05, f=bdnn_update_f[rnd_layer_mu], float_prec_f=float_prec_f)
                    bdnn_prior_cov_par[1] = np.sum([np.sum(prior_normal(cov_par[1][i],prior_bdnn_w_sd[i])) for i in range(len(cov_par[1]))])
                    if BDNN_MASK_mu:
                        for i_layer in range(len(cov_parA[1])):
                            cov_par[1][i_layer] *= BDNN_MASK_mu[i_layer]
                # Recalculate bdnn rates when we updated cov_par
                if cov_lam_updated:
                    bdnn_lam_rates, denom_lam, nn_lam = get_rate_BDNN_3D(cov_par[3], trait_tbl_NN[0], cov_par[0], nn_lamA,
                                                                         hidden_act_f, out_act_f,
                                                                         apply_reg, bias_node_idx, fix_edgeShift,
                                                                         rnd_layer_lam)
                if cov_mu_updated:
                    bdnn_mu_rates, denom_mu, nn_mu = get_rate_BDNN_3D(cov_par[4], trait_tbl_NN[1], cov_par[1], nn_muA,
                                                                      hidden_act_f, out_act_f,
                                                                      apply_reg, bias_node_idx, fix_edgeShift,
                                                                      rnd_layer_mu)

            if BDNNmodel in [2, 3] and rr_bdnn >= 0.0:
                cov_q_updated = 1
                rnd_layer_q = -1
                if prior_lam_t_reg[-1] > 0 and np.random.random() < 0.1:
                    cov_par[5] = update_parameter(cov_parA[5], 0, 1, d=0.05, f=1)
                else:
                    rnd_layer_q = np.random.randint(0, len(cov_parA[2]))
                    # update layers q rate
                    cov_par[2][rnd_layer_q] = update_parameter_normal_vec(cov_parA[2][rnd_layer_q], d=0.05, f=bdnn_update_f[rnd_layer_q], float_prec_f=float_prec_f)


            if not BDNNmodel:
                rcov=np.random.random()
                if est_COVAR_prior == 1 and rcov<0.05:
                    covar_prior = get_post_sd(cov_parA[cov_parA>0]) # est hyperprior only based on non-zero rates
                    stop_update=inf
                elif rcov < f_cov_par[0]: # cov lambda
                    cov_par[0]=update_parameter_normal(cov_parA[0],-1000,1000,d5[0])
                elif rcov < f_cov_par[1]: # cov mu
                    cov_par[1]=update_parameter_normal(cov_parA[1],-1000,1000,d5[1])
                else:
                    cov_par[2]=update_parameter_normal(cov_parA[2],-1000,1000,d5[2])

        if constrain_time_frames == 1: timesM=timesL
        q_rates[(q_rates==0).nonzero()]=q_ratesA[(q_rates==0).nonzero()]
        L[(L==0).nonzero()]=LA[(L==0).nonzero()]
        M[(M==0).nonzero()]=MA[(M==0).nonzero()]
        if not BDNNmodel:
            cov_par[(cov_par==0).nonzero()]=cov_parA[(cov_par==0).nonzero()]
        
        max_ts = np.max(ts)
        if not (BDNNmodel == 1 and fix_edgeShift in [1, 2]):
            timesL[0]=max_ts
            timesM[0]=max_ts

        if fix_SE == 0:
            if TPP_model == 1:
                q_time_frames = np.sort(np.array([max_ts,0]+times_q_shift))[::-1]
            else:
                q_time_frames = np.array([max_ts,0])

        # NHPP Lik: multi-thread computation (ts, te)
        # generate args lik (ts, te)
        if fix_SE == 0 and FBDrange==0:
            ind1=list(range(0,len(fossil)))
            ind2=[]
            if it>0 and rr<f_update_se and not timevar_qnn: # recalculate likelihood only for ts, te that were updated
                ind1=((ts-te != tsA-teA).nonzero()[0]).tolist()
                ind2=(ts-te == tsA-teA).nonzero()[0]
            lik_fossil=zeros(len(fossil))

            if len(ind1)>0 and it<stop_update and fix_SE == 0:
                # generate args lik (ts, te)
                if not BDNNmodel in [2, 3]:
                    z=zeros(len(fossil)*7).reshape(len(fossil),7)
                    z[:,0]=te
                    z[:,1]=ts
                    z[:,2]=q_rates[0]   # shape prm Gamma
                    z[:,3]=q_rates[1]   # baseline foss rate (q)
                    z[:,4]=list(range(len(fossil)))
                    z[:,5]=cov_par[2]  # covariance baseline foss rate
                    z[:,6]=M[len(M)-1] # ex rate
                    if useDiscreteTraitModel == 1: z[:,6] = mean(M)
                    args=list(z[ind1])

                if hasFoundPyRateC:
                    if TPP_model == 1:
                        # This uses the median for gamma rates
                        YangGamma = [1]
                        if argsG :
                            YangGamma=get_gamma_rates(alpha_pp_gamma)

                        lik_fossil = np.array(PyRateC_HPP_vec_lik(ind1, ts, te, q_time_frames, q_rates, YangGamma))
                        # This uses the mean for gamma rates
                        #lik_fossil2 = np.array(PyRateC_HPP_vec_lik(ind1, ts, te, q_time_frames, q_rates, pp_gamma_ncat, alpha_pp_gamma))

                        # Check correctness of results by comparing with python version
                        if sanityCheckForPyRateC == 1:
                            lik_fossil2 = zeros(len(fossil))
                            for j in range(len(ind1)):
                                i=ind1[j] # which species' lik
                                lik_fossil2[i] = HPP_vec_lik([te[i],ts[i],q_time_frames,q_rates,i,alpha_pp_gamma])

                            absDivergence = abs(np.sum(lik_fossil2) - np.sum(lik_fossil))
                            if absDivergence > sanityCheckThreshold:
                                print("[WARNING] HPP_vec_lik diverged for more than ", sanityCheckThreshold, " (", absDivergence, ")")

                    elif argsHPP == 1:
                        YangGamma = [1]
                        if argsG :
                            YangGamma=get_gamma_rates(q_rates[0])

                        lik_fossil = np.array(PyRateC_HOMPP_lik(ind1, ts, te, q_rates[1], YangGamma, cov_par[2], M[len(M)-1]))
                        
                        # Check correctness of results by comparing with python version
                        if sanityCheckForPyRateC == 1:
                            lik_fossil2 = zeros(len(fossil))
                            for j in range(len(ind1)):
                                i=ind1[j] # which species' lik
                                lik_fossil2[i] = HOMPP_lik(args[j])

                            absDivergence = abs(np.sum(lik_fossil2) - np.sum(lik_fossil))
                            if absDivergence > sanityCheckThreshold:
                                print("[WARNING] PyRateC_HOMPP_lik diverged for more than ", sanityCheckThreshold, " (", absDivergence, ")")

                    else:
                        YangGamma = [1]
                        if argsG :
                            YangGamma=get_gamma_rates(q_rates[0])
                        lik_fossil = np.array(PyRateC_NHPP_lik(use_DA==1, ind1, ts, te, q_rates[1], YangGamma, cov_par[2], M[len(M)-1]))

                        # Check correctness of results by comparing with python version
                        if sanityCheckForPyRateC == 1:
                            lik_fossil2 = zeros(len(fossil))
                            for j in range(len(ind1)):
                                i=ind1[j] # which species' lik
                                if argsG == 1: lik_fossil2[i] = NHPPgamma(args[j])
                                else: lik_fossil2[i] = NHPP_lik(args[j])

                            absDivergence = abs(np.sum(lik_fossil2) - np.sum(lik_fossil))
                            if absDivergence > sanityCheckThreshold:
                                print("[WARNING] PyRateC_NHPP_lik diverged for more than ", sanityCheckThreshold, " (", absDivergence, ")")

                else:
                    if num_processes_ts==0:
                        if BDNNmodel in [2, 3]:
                            if it > 0 and (rr == 1.5 or updated_lam_mu):
                                # RJMCMC move for lam/mu or last mcmc iteration
                                lik_fossil = lik_fossilA + 0.0
                            else:
                                q_rates_tmp = q_rates
                                if not highres_q_repeats is None: #bdnn_ads > 0.0 and argsHPP == 0:
                                    q_rates_tmp = q_rates[highres_q_repeats]
                                if timevar_qnn and ts_te_updated:
                                    qbin_ts_te = get_bin_ts_te(ts, te, q_time_frames_bdnn)
                                if cov_q_updated or (ts_te_updated and bdnn_ads > 0.0):
                                    qnn_output_unreg, nn_q = get_unreg_rate_BDNN_3D(trait_tbl_NN[2], cov_par[2], nn_qA,
                                                                                    hidden_act_f, out_act_f_q, rnd_layer=rnd_layer_q)
                                q_multi, denom_q, norm_fac = get_q_multipliers_NN(cov_par[5], qnn_output_unreg, singleton_mask, apply_reg_q, qbin_ts_te)
                                bdnn_prior_q = np.sum([np.sum(prior_normal(cov_par[2][i], prior_bdnn_w_q_sd[i])) for i in range(len(cov_par[2]))])
                                if cov_q_updated and prior_lam_t_reg[-1] > 0:
                                    bdnn_prior_q += np.log(prior_lam_t_reg[-1]) - prior_lam_t_reg[-1] * cov_par[5]

                            if use_HPP_NN_lik:
                                lik_fossil[ind1], bdnn_q_rates[ind1, :], _ = HPP_NN_lik([ts[ind1], te[ind1],
                                                                                            q_rates_tmp, alpha_pp_gamma,
                                                                                            q_multi[ind1], const_q,
                                                                                            occs_sp[ind1, :], log_factorial_occs[ind1],
                                                                                            q_time_frames_bdnn, duration_q_bins[ind1, :],
                                                                                            occs_single_bin[ind1], singleton_lik[ind1],
                                                                                            argsG, pp_gamma_ncat, YangGammaQuant])
                            else:
                                lik_fossil[ind1], bdnn_q_rates[ind1], _ = HOMPP_NN_lik([ts[ind1], te[ind1],
                                                                                        q_rates_tmp,
                                                                                        q_multi[ind1], const_q,
                                                                                        occs_sp[ind1], log_factorial_occs[ind1],
                                                                                        singleton_lik[ind1],
                                                                                        argsG, pp_gamma_ncat, YangGammaQuant])
                        else:
                            for j in range(len(ind1)):
                                i=ind1[j] # which species' lik
                                if TPP_model == 1:
                                    lik_fossil[i] = HPP_vec_lik([te[i],ts[i],q_time_frames,q_rates,i,alpha_pp_gamma])
                                else:
                                    if argsHPP == 1 or  frac1==0: 
                                        lik_fossil[i] = HOMPP_lik(args[j])
                                    elif argsG == 1:
                                        lik_fossil[i] = NHPPgamma(args[j])
                                    else:
                                        lik_fossil[i] = NHPP_lik(args[j])
                    else:
                        if TPP_model == 1: sys.exit("TPP_model model can only run on a single processor")
                        if argsHPP == 1 or  frac1==0: lik_fossil[ind1] = array(pool_ts.map(HOMPP_lik, args))
                        elif argsG == 1: lik_fossil[ind1] = array(pool_ts.map(NHPPgamma, args))
                        else: lik_fossil[ind1] = array(pool_ts.map(NHPP_lik, args))

            if it>0:
                lik_fossil[ind2] = lik_fossilA[ind2]

        # FBD range likelihood
        elif FBDrange:
            move_type = 0
            stop_update = 0
            if np.random.random()<0.01 and TDI==4:
                rj_cat_HP = get_post_rj_HP(len(LA),len(MA)) # est Poisson hyperprior on number of rates (RJMCMC)
                stop_update=inf
            elif np.random.random()< 0.3 and TDI==4:
                stop_update=0
                L,timesL,M,timesM,hasting2 = RJMCMC([LA,MA, timesLA, timesMA, maxFA,minLA])
                hasting += hasting2
                move_type=0

            if it==0: stop_update=1
            # move types: 1) ts/te; 2) q rates; 3) timesL/M; 4) L/M rates;
            if it==0 or len(res_FBD_A)==0: move_type=0

            # only need to update if changed timesL/timesM or psi, lam, mu
            if move_type in [0,2,3,4] or len(res_FBD_A)==0 or np.max(ts) != maxTSA:
                times_fbd_temp, psi_fbd_temp, lam_fbd_temp, mu_fbd_temp = get_times_n_rates(q_time_frames, timesL, timesM, q_rates, L, M)
            else:
                [times_fbd_temp, psi_fbd_temp, lam_fbd_temp, mu_fbd_temp] = FBD_temp_A

            # times, psi, lam, mu, ts, te, k=0, intervalAs=0, int_indx=0, div_traj=0, rho=0
            if move_type == 1: # updated ts/te
                res_FBD = likelihood_rangeFBD(times_fbd_temp, psi_fbd_temp, lam_fbd_temp, mu_fbd_temp, ts, te, 
                        intervalAs=res_FBD_A[1], k=res_FBD_A[4], FALAmodel = FBDrange, alpha = alpha_pp_gamma)
            elif move_type in [2,4]: # updated q/s/e rates
                res_FBD = likelihood_rangeFBD(times_fbd_temp, psi_fbd_temp, lam_fbd_temp, mu_fbd_temp, ts, te,  
                        int_indx=res_FBD_A[2], div_traj=res_FBD_A[3], k=res_FBD_A[4], FALAmodel = FBDrange, alpha = alpha_pp_gamma)
            elif move_type in [3]: # updated times
                res_FBD = likelihood_rangeFBD(times_fbd_temp, psi_fbd_temp, lam_fbd_temp, mu_fbd_temp, ts, te,  
                        intervalAs=res_FBD_A[1], div_traj=res_FBD_A[3], FALAmodel = FBDrange, alpha = alpha_pp_gamma)
            else:
                res_FBD = likelihood_rangeFBD(times_fbd_temp, psi_fbd_temp, lam_fbd_temp, mu_fbd_temp, ts, te, FALAmodel = FBDrange,
                        alpha = alpha_pp_gamma)
            # res = [likelihood, intervalAs, int_indx, div_traj, k]
            # if it % 1000==0:
            #    print times_fbd_temp
            #    print psi_fbd_temp
            #    print lam_fbd_temp
            #    print mu_fbd_temp
            #    print res_FBD[3]
            #    print res_FBD[4]

            lik_fossil = res_FBD[0]
            #if it>1: print lik_fossil,lik_fossilA


        else: lik_fossil=np.zeros(1)

        if FBDrange==0:
            if it>=stop_update or stop_update==inf: lik_fossil = lik_fossilA

        # pert_prior defines gamma prior on q_rates[1] - fossilization rate
        if TPP_model == 1:
            if pert_prior[1]>0:
                prior = np.sum(prior_gamma(q_rates,pert_prior[0],pert_prior[1]))+ prior_uniform(alpha_pp_gamma,0,20)
            else: # use hyperprior on Gamma rate on q
                hpGammaQ_shape = 1.01 # hyperprior is essentially flat
                hpGammaQ_rate =  0.1
                post_rate_prm_Gq = np.random.gamma( shape=hpGammaQ_shape+pert_prior[0]*len(q_rates), scale=1./(hpGammaQ_rate+np.sum(q_rates)) )
                prior = np.sum(prior_gamma(q_rates,pert_prior[0],post_rate_prm_Gq)) + prior_uniform(alpha_pp_gamma,0,20)
        else:
            prior = prior_gamma(q_rates[1],pert_prior[0],pert_prior[1]) + prior_uniform(q_rates[0],0,20)
        if est_hyperP == 1:
            prior += ( prior_uniform(hyperP[0],0,20)+prior_uniform(hyperP[1],0,20) ) # hyperprior on BD rates
        if BDNNmodel in [2, 3]:
            prior += bdnn_prior_q
        
        ### DPP begin
        if TDI==3:
            likBDtemp=0
            if stop_update != inf: # standard MCMC
                likBDtemp = BPD_lik_vec_times([ts,te,timesL,L[indDPP_L],M[indDPP_M]])
                #n_data=len(indDPP_L)
                #for time_frame_i in range(n_data):
                #    up=timesL[time_frame_i]
                #    lo=timesL[time_frame_i+1]
                #    likBDtemp+= BPD_partial_lik_vec([ts,te,up,lo,L[indDPP_L[time_frame_i]], "l"])
                #    likBDtemp+= BPD_partial_lik_vec([ts,te,up,lo,M[indDPP_M[time_frame_i]], "m"])
            else: ### RUN DPP GIBBS SAMPLER
                lik1, L, indDPP_L, alpha_par_Dir_L = DDP_gibbs_sampler([ts,te,L,indDPP_L,timesL,alpha_par_Dir_L,"l"])
                lik2, M, indDPP_M, alpha_par_Dir_M = DDP_gibbs_sampler([ts,te,M,indDPP_M,timesL,alpha_par_Dir_M,"m"])
                likBDtemp = lik1+lik2
        ### DPP end
        
        elif BDNNmodel in [1, 3] and TDI == 0 and fix_Shift == 0:
            # arg = [ts,te,trait_tbl_NN,L,M,cov_par]
            # likBDtemp = BDNN_likelihood(arg)
            arg = [ts, te, bdnn_lam_rates, bdnn_mu_rates]
            likBDtemp = BDNN_fast_likelihood(arg)

        elif FBDrange == 0:
            if TDI==4 and np.random.random()<0.01:
                rj_cat_HP = get_post_rj_HP(len(LA),len(MA)) # est Poisson hyperprior on number of rates (RJMCMC)
                stop_update=inf

            # Birth-Death Lik: construct 2D array (args partial likelihood)
            # parameters of each partial likelihood and prior (l)
            if stop_update != inf:
                if useDiscreteTraitModel == 1:
                    if twotraitBD == 1:
                        likBDtemp = BD_lik_discrete_trait_continuous([ts,te,L,M,cov_par])
                    else:
                        likBDtemp = BD_lik_discrete_trait([ts,te,L,M])

                elif use_ADE_model >= 1 and TPP_model == 1:
                    likBDtemp = BD_age_lik_vec_times([ts,te,timesL,W_shape,M,q_rates,q_time_frames])
                elif fix_Shift == 1 and not BDNNmodel in [1, 3]:
                    if use_ADE_model == 0: 
                        likBDtemp = BPD_lik_vec_times([ts,te,timesL,L,M])
                    else: 
                        likBDtemp = BD_age_lik_vec_times([ts,te,timesL,W_shape,M,q_rates])
                else:
                    args=list()
                    if use_ADE_model == 0: # speciation rate is not used under ADE model
                        if BDNNmodel in [1, 3]:
                            ind_bdnn_lik = np.arange(n_taxa)
                            if ts_te_updated:
                                if not bdnn_dd:
                                    ind_bdnn_lik = (ts-te != tsA-teA).nonzero()[0]
                                i_events_sp, i_events_ex, n_S = update_events_ns(ts, te, timesL, bin_size_lam_mu, i_events_sp, i_events_ex, n_S, ind_bdnn_lik)
                            args.append([i_events_sp[ind_bdnn_lik, :], n_S[ind_bdnn_lik, :], bdnn_lam_rates[ind_bdnn_lik, :], apply_reg[ind_bdnn_lik, :]])
                        else:
                            for temp_l in range(len(timesL)-1):
                                up, lo = timesL[temp_l], timesL[temp_l+1]
                                l = L[temp_l]
                                args.append([ts, te, up, lo, l, 'l', cov_par[0],np.nan])
                    # print(args[0][4], args[0][6], args[0][-1])
                    # parameters of each partial likelihood and prior (m)
                    if BDNNmodel in [1, 3]:
#                        print('i_events_ex\n', i_events_ex)
#                        print('n_S\n', n_S)
                        args.append([i_events_ex[ind_bdnn_lik, :], n_S[ind_bdnn_lik, :], bdnn_mu_rates[ind_bdnn_lik, :], apply_reg[ind_bdnn_lik, :]])
                    else:
                        for temp_m in range(len(timesM)-1):
                            up, lo = timesM[temp_m], timesM[temp_m+1]
                            m = M[temp_m]
                            if use_ADE_model == 0:
                                args.append([ts, te, up, lo, m, 'm', cov_par[1],np.nan])
                            elif use_ADE_model >= 1:
                                args.append([ts, te, up, lo, m, 'm', cov_par[1],W_shape,q_rates[1]])
                    
                    if hasFoundPyRateC and model_cov==0 and use_ADE_model == 0 and not BDNNmodel in [1, 3]:
                        likBDtemp = PyRateC_BD_partial_lik(ts, te, timesL, timesM, L, M)

                        # Check correctness of results by comparing with python version
                        if sanityCheckForPyRateC == 1:
                            likBDtemp2=np.zeros(len(args))
                            i=0
                            for i in range(len(args)):
                                likBDtemp2[i]=BPD_partial_lik(args[i])
                                i+=1

                            absDivergence = abs(np.sum(likBDtemp) - np.sum(likBDtemp2))
                            if absDivergence > sanityCheckThreshold:
                                print("[WARNING] BPD_partial_lik diverged for more than ", sanityCheckThreshold, " (", absDivergence, ")")

                    else:
                        if num_processes==0:
                            if TDI == 4 and BDNNmodel in [2]:
                                args = []
                                args=list()
                                for temp_l in range(len(timesL)-1):
                                    up, lo = timesL[temp_l], timesL[temp_l+1]
                                    l = L[temp_l]
                                    args.append([ts, te, up, lo, l, 'l', cov_par[0],1])
                                for temp_m in range(len(timesM)-1):
                                    up, lo = timesM[temp_m], timesM[temp_m+1]
                                    m = M[temp_m]
                                    args.append([ts, te, up, lo, m, 'm', cov_par[1],1])
#                                # Non-constant baseline bdnn. Not working any longer with t_reg
#                                rj_ind = 0  
#                                for temp_l in range(len(fixed_times_of_shift_bdnn)-1):
#                                    if fixed_times_of_shift_bdnn[temp_l + 1] < timesL[rj_ind + 1]:
#                                        rj_ind += 1
#                                    up, lo = fixed_times_of_shift_bdnn[temp_l], fixed_times_of_shift_bdnn[temp_l+1]
#                                    l = L[rj_ind]
#                                    # print(l, up, lo, fixed_times_of_shift_bdnn[temp_l+1], timesL[rj_ind])
#                                    args.append([ts, te, up, lo, l, 'l', cov_par[0],1])
#                                rj_ind = 0 
#                                # print("timesM", timesM, M)
#                                for temp_m in range(len(fixed_times_of_shift_bdnn)-1):
#                                    if fixed_times_of_shift_bdnn[temp_m + 1] < timesM[rj_ind + 1]:
#                                        rj_ind += 1
#                                    up, lo = fixed_times_of_shift_bdnn[temp_m], fixed_times_of_shift_bdnn[temp_m+1]
#                                    m = M[rj_ind]
#                                    # print(m, up, lo, fixed_times_of_shift_bdnn[temp_m+1], timesM[rj_ind])
#                                    args.append([ts, te, up, lo, m, 'm', cov_par[1],1])
                                
                            if BDNNmodel in [1, 3]:
                                if ts_te_updated or cov_lam_updated:
                                    likBDtemp[0, ind_bdnn_lik] = BDNN_fast_partial_lik(args[0])
                                if ts_te_updated or cov_mu_updated:
                                    likBDtemp[1, ind_bdnn_lik] = BDNN_fast_partial_lik(args[1])
                            else:
                                likBDtemp=np.zeros(len(args))
                                for i in range(len(args)):
                                    likBDtemp[i] = BPD_partial_lik(args[i])
                        # multi-thread computation of lik and prior (rates)
                        else:
                            likBDtemp = np.array(pool_lik.map(BPD_partial_lik, args))
                    

            else:
                if TDI==2: # run BD algorithm (Alg. 3.1)
                    sys.stderr = NO_WARN
                    args=[it, likBDtempA,tsA, teA, LA,MA, timesLA, timesMA, cov_parA,len(LA)]
                    likBDtemp, L,M, timesL, timesM, cov_par = Alg_3_1(args)
                    sys.stderr = original_stderr

                elif (TDI==4 and FBDrange==0) or (TDI==4 and BDNNmodel in [2]): # run RJMCMC
                    stop_update = 0
                    L,timesL,M,timesM,hasting = RJMCMC([LA,MA, timesLA, timesMA,maxFA,minLA])
                    # print(L, timesL, M, timesM) #,hasting
                    args=list()
                    for temp_l in range(len(timesL)-1):
                        up, lo = timesL[temp_l], timesL[temp_l+1]
                        l = L[temp_l]
                        args.append([ts, te, up, lo, l, 'l', cov_par[0],1])
                    for temp_m in range(len(timesM)-1):
                        up, lo = timesM[temp_m], timesM[temp_m+1]
                        m = M[temp_m]
                        args.append([ts, te, up, lo, m, 'm', cov_par[1],1])

                    if hasFoundPyRateC and model_cov==0:
                        likBDtemp = PyRateC_BD_partial_lik(ts, te, timesL, timesM, L, M)

                        # Check correctness of results by comparing with python version
                        if sanityCheckForPyRateC == 1:
                            likBDtemp2=np.zeros(len(args))
                            i=0
                            for i in range(len(args)):
                                likBDtemp2[i]=BPD_partial_lik(args[i])
                                i+=1

                            absDivergence = abs(np.sum(likBDtemp) - np.sum(likBDtemp2))
                            if absDivergence > sanityCheckThreshold:
                                print("[WARNING] BPD_partial_lik diverged for more than ", sanityCheckThreshold, " (", absDivergence, ")")

                    else:
                        if num_processes==0:
                            likBDtemp=np.zeros(len(args))
                            i=0
                            for i in range(len(args)):
                                likBDtemp[i]=BPD_partial_lik(args[i])
                                i+=1
                        # multi-thread computation of lik and prior (rates)
                        else:likBDtemp = array(pool_lik.map(BPD_partial_lik, args))
                        #print sum(likBDtemp)-sum(likBDtempA),hasting,get_hyper_priorBD(timesL,timesM,L,M,T,hyperP)+(-log(max(ts)\
                        # -min(te))*(len(L)-1+len(M)-1))-(get_hyper_priorBD(timesLA,timesMA,LA,MA,T,hyperP)+(-log(max(tsA)\
                        # -min(teA))*(len(LA)-1+len(MA)-1))), len(L),len(M)
                elif TDI == 4 and BDNNmodel in [1, 3]:
                    stop_update = 0
                    L,timesL,M,timesM,hasting = RJMCMC([LA,MA, timesLA, timesMA,maxFA,minLA])
                    args = []
                    rj_ind = 0  
                    for temp_l in range(len(fixed_times_of_shift_bdnn)-1):
                        if fixed_times_of_shift_bdnn[temp_l + 1] < timesL[rj_ind + 1]:
                            rj_ind += 1
                        up, lo = fixed_times_of_shift_bdnn[temp_l], fixed_times_of_shift_bdnn[temp_l+1]
                        l = L[rj_ind]
                        # print(l, up, lo, fixed_times_of_shift_bdnn[temp_l+1], timesL[rj_ind])
                        args.append([ts, te, up, lo, l, 'l', cov_par[0],1])
                    rj_ind = 0 
                    for temp_m in range(len(fixed_times_of_shift_bdnn)-1):
                        if fixed_times_of_shift_bdnn[temp_m + 1] < timesM[rj_ind + 1]:
                            rj_ind += 1
                        up, lo = fixed_times_of_shift_bdnn[temp_m], fixed_times_of_shift_bdnn[temp_m+1]
                        m = M[rj_ind]
                        # print(m, up, lo, fixed_times_of_shift_bdnn[temp_m+1], timesM[rj_ind])
                        args.append([ts, te, up, lo, m, 'm', cov_par[1],1])
                    
                    likBDtemp=np.zeros(len(args))
                    i=0
                    for i in range(len(args)):
                        likBDtemp[i] = BDNN_partial_lik(args[i])
                    # print(LA)
                    # print(likBDtemp)
                    # sys.exit("RJMCMC not allowed with BD-NN model!")

                # NHPP Lik: needs to be recalculated after Alg 3.1 or RJ (but only if NHPP+DA)
                if fix_SE == 0 and TPP_model == 0 and argsHPP == 0 and use_DA == 1:
                    # NHPP calculated only if not -fixSE
                    # generate args lik (ts, te)
                    ind1=list(range(0,len(fossil)))
                    lik_fossil=zeros(len(fossil))
                    # generate args lik (ts, te)
                    z=zeros(len(fossil)*7).reshape(len(fossil),7)
                    z[:,0]=te
                    z[:,1]=ts
                    z[:,2]=q_rates[0]   # shape prm Gamma
                    z[:,3]=q_rates[1]   # baseline foss rate (q)
                    z[:,4]=list(range(len(fossil)))
                    if not BDNNmodel in [2, 3]:
                        z[:,5]=cov_par[2]  # covariance baseline foss rate
                    z[:,6]=M[len(M)-1] # ex rate
                    args=list(z[ind1])
                    if num_processes_ts==0:
                        for j in range(len(ind1)):
                            i=ind1[j] # which species' lik
                            if argsG == 1: lik_fossil[i] = NHPPgamma(args[j])
                            else: lik_fossil[i] = NHPP_lik(args[j])
                    else:
                        if argsG == 1: lik_fossil[ind1] = array(pool_ts.map(NHPPgamma, args))
                        else: lik_fossil[ind1] = array(pool_ts.map(NHPP_lik, args))

        elif FBDrange:
            likBDtemp = 0 # alrady included in lik_fossil

        preburnin = 0
        if it < preburnin:
            bd_tempering = 3
            # prior_bdnn_w_sd = prior_bdnn_w_sd
        elif it == preburnin:
            bd_tempering = 1
            accept_it = 1
            if preburnin:
                print("mu_empirical_prior BEFORE", 0)
                print("prior_bdnn_w_sd BEFORE", prior_bdnn_w_sd) 
                mu_empirical_prior = cov_parA
                for i in range(len(prior_bdnn_w_sd)):
                    prior_bdnn_w_sd[i] = 1
                
                # u = [i*0+0.1 for i in prior_bdnn_w_sd]
                # prior_bdnn_w_sd = u
                print("mu_empirical_prior", mu_empirical_prior)
                print("prior_bdnn_w_sd", prior_bdnn_w_sd) 
        elif it > preburnin:
            accept_it = 0

        lik= np.sum(lik_fossil) + np.sum(likBDtemp) * bd_tempering + PoiD_const

        maxTs= np.max(ts)
        minTe= np.min(te)
        if TDI < 3:
            prior += np.sum(prior_times_frames(timesL, maxTs, minTe, lam_s))
            prior += np.sum(prior_times_frames(timesM, maxTs, minTe, lam_s))
        if TDI ==4:
            #prior_old = -log(max(ts)-max(te))*len(L-1)  #sum(prior_times_frames(timesL, max(ts),min(te), 1))
            #prior_old += -log(max(ts)-max(te))*len(M-1)  #sum(prior_times_frames(timesM, max(ts),min(te), 1))
            prior += -log(maxTs-minTe)*(len(L)-1+len(M)-1)
            prior += Poisson_prior(len(L),rj_cat_HP)+Poisson_prior(len(M),rj_cat_HP)
            #if it % 100 ==0: print len(L),len(M), prior_old, -log(max(ts)-min(te))*(len(L)-1+len(M)-1), hasting

            if get_min_diffTime(timesL)<=min_allowed_t or get_min_diffTime(timesM)<=min_allowed_t: prior = -np.inf

        priorBD= get_hyper_priorBD(timesL,timesM,L,M,maxTs,hyperP)
        if use_ADE_model >= 1:
            # M in this case is the vector of Weibull scales
            priorBD = np.sum(prior_normal(log(W_shape),2)) # Normal prior on log(W_shape): highest prior pr at W_shape=1

        prior += priorBD
        ###
        if model_cov >0: prior+=np.sum(prior_normal(cov_par,covar_prior))
        
        if BDNNmodel:
            if preburnin and it >= preburnin:
                if BDNNmodel in [1, 3]:
                    prior +=  np.sum([np.sum(prior_normal(cov_par[0][i],prior_bdnn_w_sd[i], mu=mu_empirical_prior[0][i])) for i in range(len(cov_par[0]))])
                    prior +=  np.sum([np.sum(prior_normal(cov_par[1][i],prior_bdnn_w_sd[i], mu=mu_empirical_prior[1][i])) for i in range(len(cov_par[1]))])
            else:
                if BDNNmodel in [1, 3]:
                    prior += np.sum(bdnn_prior_cov_par)

        # exponential prior on root age
        maxFA = np.max(FA)
        prior += prior_root_age(maxTs,maxFA,maxFA)

        # add tree likelihood
        if analyze_tree ==1: # independent rates model
            r_tree, h1 = update_multiplier_proposal(r_treeA,1.1) # net diversification
            m_tree, h2 = update_multiplier_proposal(m_treeA,1.1) # extinction rate
            l_tree = m_tree+r_tree
            tree_lik = treeBDlikelihood(tree_node_ages,l_tree,m_tree,rho=tree_sampling_frac)
            hasting = hasting+h1+h2
        elif analyze_tree ==2: # compatible model (BDC)
            r_tree = update_parameter(r_treeA, m=0, M=1, d=0.1, f=1)
            l_tree = (M[0]*r_tree) + (L[0]-M[0])
            m_tree = M[0]*r_tree
            tree_lik = treeBDlikelihood(tree_node_ages,l_tree,m_tree,rho=tree_sampling_frac)
        elif analyze_tree ==3: # equal rate model
            r_tree = 0
            l_tree = L[0]
            m_tree = M[0]
            tree_lik = treeBDlikelihood(tree_node_ages,l_tree,m_tree,rho=tree_sampling_frac)
        elif analyze_tree ==4: # skyline independent model
            m_tree,r_tree,h1,h2 = m_treeA+0., r_treeA+0.,0,0
            if args_bdc: # BDC model
                ind = np.random.choice(list(range(len(m_tree))))
                r_tree[ind] = update_parameter(r_treeA[ind], 0, 1, 0.1, 1)
                l_tree = L[::-1] - M[::-1] + M[::-1]*r_tree
                m_tree = M[::-1]*r_tree #  so mu > 0
            else:
                m_tree, h2 = update_q_multiplier(m_treeA,d=1.1,f=0.5) # extinction rate
                r_tree, h1 = update_q_multiplier(r_treeA,d=1.1,f=0.5) # speciation rate
                l_tree = r_tree*m_tree # this allows extinction > speciation
                # args = (x,t,l,mu,sampling,posdiv=0,survival=1,groups=0)

            if np.min(l_tree)<=0:
                tree_lik = -np.inf
            else:
                tree_lik = treeBDlikelihoodSkyLine(tree_node_ages,phylo_times_of_shift,l_tree,m_tree,tree_sampling_frac)
                hasting = hasting+h1+h2
                prior += np.sum(prior_gamma(l_tree,1.1,1)) + np.sum(prior_gamma(m_tree,1.1,1))
        else:
            tree_lik = 0

        if temperature==1:
            tempMC3=1./(1+n_proc*temp_pr)
            lik_alter=lik
        else:
            tempMC3=1
            lik_alter=(np.sum(lik_fossil)+ PoiD_const) + (np.sum(likBDtemp)+ PoiD_const)*temperature
        
        Post=lik_alter+prior+tree_lik
        accept_it = 0
        if it==0:
            accept_it = 1
            PostA = Post
        if it>0 and (it-burnin) % (I_effective/len(temperatures)) == 0 and it>burnin or it==I-1:
            accept_it = 1 # when temperature changes always accept first iteration
            PostA = Post

        if rr<f_update_se and use_gibbs_se_sampling==1:
            accept_it = 1

        if rr<f_update_se and it < fast_burnin:
            accept_it = 1

        #print Post, PostA, q_ratesA, sum(lik_fossil), sum(likBDtemp),  prior
        #print sum(lik_fossil), sum(likBDtemp), PoiD_const
        if Post>-inf and Post<inf:
            r_acc = log(np.random.random())
            is_accepted = Post*tempMC3-PostA*tempMC3 + hasting >= r_acc or stop_update==inf and TDI in [2,3,4] or accept_it==1
            if is_accepted:
                likBDtempA=likBDtemp
                PostA=Post
                priorA=prior
                likA=lik
                timesLA=timesL
                timesMA=timesM
                LA,MA=L,M
                hyperPA=hyperP
                tsA,teA=ts,te
                SA=np.sum(tsA-teA)
                q_ratesA=q_rates
                alpha_pp_gammaA=alpha_pp_gamma
                lik_fossilA=lik_fossil
                cov_parA=cov_par
                W_shapeA=W_shape
                tree_likA = tree_lik
                if analyze_tree >=1:
                    r_treeA = r_tree
                    m_treeA = m_tree
                if FBDrange:
                    res_FBD_A = res_FBD
                    FBD_temp_A = [times_fbd_temp, psi_fbd_temp, lam_fbd_temp, mu_fbd_temp]
                if BDNNmodel in [1, 3]:
                    bdnn_lam_ratesA = bdnn_lam_rates
                    bdnn_mu_ratesA = bdnn_mu_rates
                    denom_lamA = denom_lam
                    denom_muA = denom_mu
                    nn_lamA = nn_lam
                    nn_muA = nn_mu
                    if use_time_as_trait or bdnn_timevar[0] or bdnn_dd or bdnn_loaded_tbls_timevar:
                        bdnn_prior_cov_parA = bdnn_prior_cov_par + 0.0
                        i_events_spA = i_events_sp
                        i_events_exA = i_events_ex
                        n_SA = n_S
                if BDNNmodel in [2, 3] and rr < 1.5:
                    q_multiA = q_multi + 0.0
                    qnn_output_unregA = qnn_output_unreg + 0.0
                    bdnn_q_ratesA = bdnn_q_rates
                    denom_qA = denom_q
                    norm_facA = norm_fac
                    bdnn_prior_qA = bdnn_prior_q
                    nn_qA = nn_q
                if ts_te_updated and tune_T_schedule[0] > 0:
                    d1_ts, d1_te, tste_tune_obj = tune_tste_windows(d1_ts, d1_te, LO, bound_te, tste_tune_obj, it,
                                                                    tune_T_schedule, ind1, ts_or_te_updated,
                                                                    accepted=1)
            elif BDNNmodel:
                if BDNNmodel in [1, 3]:
                    bdnn_lam_rates = bdnn_lam_ratesA
                    bdnn_mu_rates = bdnn_mu_ratesA
                    denom_lam = denom_lamA
                    denom_mu = denom_muA
                    nn_lam = nn_lamA
                    nn_mu = nn_muA
                    if use_time_as_trait or bdnn_timevar[0] or bdnn_dd or bdnn_loaded_tbls_timevar:
                        bdnn_prior_cov_par = bdnn_prior_cov_parA + 0.0
                        i_events_sp = i_events_spA
                        i_events_ex = i_events_exA
                        n_S = n_SA
                        if bdnn_dd and ts_te_updated:
                            trait_tbl_NN[0][ :, :, div_idx_trt_tbl] = bdnn_divA
                            trait_tbl_NN[1][ :, :, div_idx_trt_tbl] = bdnn_divA
                if BDNNmodel in [2, 3]:
                    if bdnn_ads >= 0.0 and ts_te_updated:
                        trait_tbl_NN[2] = add_taxon_age(tsA, teA, q_time_frames_bdnn, trait_tbl_NN[2], ts, te)
                    nn_q = nn_qA
#                    if trait_tbl_NN[2].ndim == 3 and ts_te_updated == 1:
#                        qbin_ts_te = get_bin_ts_te(tsA, teA, q_time_frames_bdnn)
            if not is_accepted and ts_te_updated and tune_T_schedule[0] > 0:
                d1_ts, d1_te, tste_tune_obj = tune_tste_windows(d1_ts, d1_te, LO, bound_te, tste_tune_obj, it,
                                                                tune_T_schedule, ind1, ts_or_te_updated,
                                                                accepted=0)

        if it % print_freq ==0 or it==burnin:
            try: l=[round(y, 2) for y in [PostA, likA, priorA, SA]]
            except:
                print("An error occurred.")
                print(PostA,Post,lik, prior, prior_root_age(np.max(ts),np.max(FA),np.max(FA)), priorBD, np.max(ts),np.max(FA))
                print(prior_gamma(q_rates[1],pert_prior[0],pert_prior[1]) + prior_uniform(q_rates[0],0,20))
                quit()
            if it>burnin and n_proc==0:
                print_out= "\n%s\tpost: %s lik: %s (%s, %s) prior: %s tot.l: %s" \
                % (it, l[0], l[1], round(sum(lik_fossilA), 2), round(sum(likBDtempA)+ PoiD_const, 2),l[2], l[3])
                if TDI==1: print_out+=" beta: %s" % (round(temperature,4))
                if TDI in [2,3,4]: print_out+=" k: %s" % (len(LA)+len(MA))
                print(print_out)
                if edge_indicator:
                    print("\t Extant: %s of which added: %s; observed: %s" % (len(teA[teA == 0]), len(teA[teA == 0]) - len(LO[LO == 0]), len(LO[LO == 0])))
                if analyze_tree >=1:
                    print("\ttree lik:", np.round(tree_likA,2)) 
                #if TDI==1: print "\tpower posteriors:", marginal_lik[0:10], "..."
                if TDI==3:
                    print("\tind L", indDPP_L)
                    print("\tind M", indDPP_M)
                else:
                    if bdnn_const_baseline == 0:
                        print("\tt.frames:", timesLA, "(sp.)")
                        print("\tt.frames:", timesMA, "(ex.)")
                if use_ADE_model >= 1:
                    print("\tWeibull.shape:", round(W_shapeA,3))
                    print("\tWeibull.scale:", MA, 1./MA)
                else:
                    if bdnn_const_baseline == 0:
                        print("\tsp.rates:", LA)
                        print("\tex.rates:", MA)
                #   print(np.array([tree_likA, r_treeA+m_treeA, m_treeA]))
                # if analyze_tree==2:
                #     ltreetemp,mtreetemp = (M[0]*r_tree) + (L[0]-M[0]), M[0]*r_tree
                #     print(np.array([tree_likA,ltreetemp,mtreetemp,ltreetemp-mtreetemp,LA[0]-MA[0]]))
                # if analyze_tree==4:
                #     ltreetemp,mtreetemp = list(r_treeA[::-1]*m_treeA[::-1]), list(m_treeA[::-1])
                #     print(np.array([tree_likA] + ltreetemp + mtreetemp))

                if est_hyperP == 1: print("\thyper.prior.par", hyperPA)


                if model_cov>=1:
                    print("\tcov. (sp/ex/q):", cov_parA)
                    if est_COVAR_prior == 1: print("\tHP_covar:",round(covar_prior,3))
                if fix_SE == 0:
                    if TPP_model == 1:
                        print("\tq.rates:", q_ratesA, "\n\tGamma.prm:", round(alpha_pp_gammaA,3))
                    else:
                        print("\tq.rate:", round(q_ratesA[1], 3), "\tGamma.prm:", round(q_ratesA[0], 3))
                    print("\tts:", tsA[0:5], "...")
                    print("\tte:", teA[0:5], "...")
                if BDNNmodel in [1, 3]:
                    print("\tbdnn-lam:",[np.round(np.mean(cov_parA[0][i]), 2) for i in range(len(cov_par_init_NN[0]))])
                    print("\tbdnn-mu: ",[np.round(np.mean(cov_parA[1][i]), 2) for i in range(len(cov_par_init_NN[1]))])
            if it<=burnin and n_proc==0: print(("\n%s*\tpost: %s lik: %s prior: %s tot length %s" \
            % (it, l[0], l[1], l[2], l[3])))

        if n_proc != 0: pass
        elif it % sample_freq ==0 and it>=burnin or it==0 and it>=burnin:
            s_max=np.max(tsA)
            if fix_SE == 0:
                if TPP_model == 0: log_state = [it,PostA, priorA, sum(lik_fossilA), (likA-sum(lik_fossilA))/bd_tempering, q_ratesA[1], q_ratesA[0]]
                else:
                    log_state= [it,PostA, priorA, sum(lik_fossilA), (likA-sum(lik_fossilA))/bd_tempering]
                    log_state += list(q_ratesA)
                    log_state += [alpha_pp_gammaA]
                    if pert_prior[1]==0:
                        log_state += [post_rate_prm_Gq]
            else:
                log_state= [it,PostA, priorA, (likA-sum(lik_fossilA))/bd_tempering]

            if model_cov>=1:
                log_state += cov_parA[0], cov_parA[1],cov_parA[2]
                if est_COVAR_prior == 1: log_state += [covar_prior]
            if edge_indicator:
                log_state.append(len(teA[teA == 0]) - len(LO[LO == 0]))

            if TDI<2: # normal MCMC or MCMC-TI
                log_state += s_max,np.min(teA)
                if TDI==1: log_state += [temperature]
                if est_hyperP == 1: log_state += list(hyperPA)
                if BDNNmodel in [1, 3] and bdnn_const_baseline:
                    pass
                #     log_state.append(LA[0])
                elif use_ADE_model == 0:
                    log_state += list(LA)
                elif use_ADE_model == 1:
                    log_state+= [W_shapeA]
                elif use_ADE_model == 2:
                    # this correction is for the present (recent sp events are unlikely to show up)
                    xtemp = np.linspace(0,5,101)
                    pdf_q_sampling = np.round(1-exp(-q_ratesA[1]*xtemp),2)
                    #try:
                    #    q95 = np.min([xtemp[pdf_q_sampling==0.75][0],0.25*s_max]) # don't remove more than 25% of the time window
                    #except: q95 = 0.25*s_max
                    q95 = np.min(tsA[tsA>0])
                    # estimate sp rate based on ex rate and ratio between observed sp and ex events
                    corrSPrate = float(len(tsA[tsA>q95]))/np.maximum(1,len(teA[teA>q95])) * 1./MA
                    log_state+= list(corrSPrate)

                if BDNNmodel in [1, 3] and bdnn_const_baseline:
                    pass # log_state.append(MA[0])
                elif use_ADE_model <= 1:
                    log_state += list(MA) # This is W_scale in the case of ADE models
                if use_ADE_model == 2:
                    log_state += list(1./MA) # when using model 2 shape = 1, and 1/scale = extinction rate

                if use_ADE_model >= 1:
                    log_state+= list(MA * gamma(1 + 1./W_shapeA))
                if fix_Shift== 0:
                    log_state += list(timesLA[1:-1])
                    log_state += list(timesMA[1:-1])
                if analyze_tree ==1:
                    log_state += [tree_likA, r_treeA+m_treeA, m_treeA]
                if analyze_tree ==2:
                    log_state += [tree_likA, (MA[0]*r_treeA) + (LA[0]-MA[0]), MA[0]*r_treeA]
                if analyze_tree ==3:
                    log_state += [tree_likA, LA[0], MA[0]]
                if analyze_tree ==4:
                    if args_bdc: # BDC model
                        ltreetemp = (MA[::-1]*r_treeA) + (LA[::-1]-MA[::-1])
                        mtreetemp =  MA[::-1]*r_treeA
                        ltreetemp = list(ltreetemp[::-1])
                        mtreetemp = list(mtreetemp[::-1])
                    else:
                        ltreetemp,mtreetemp = list(r_treeA[::-1]*m_treeA[::-1]), list(m_treeA[::-1])
                    log_tree_lik_temp = [tree_likA] + ltreetemp + mtreetemp
                    log_state += log_tree_lik_temp

            elif TDI == 2: # BD-MCMC
                log_state+= [len(LA), len(MA), s_max,min(teA)]
            elif TDI == 3: # DPP
                log_state+= [len(LA), len(MA), alpha_par_Dir_L,alpha_par_Dir_M, s_max,min(teA)]
            elif TDI ==4: # RJMCMC
                log_state+= [len(LA), len(MA),rj_cat_HP, s_max,min(teA)]

            if useDiscreteTraitModel == 1:
                for i in range(len(lengths_B_events)): log_state += [sum(tsA[ind_trait_species==i]-teA[ind_trait_species==i])]
            log_state += [SA]
            
            if BDNNmodel:
                if BDNNmodel in [1, 3]:
                    # weights lam
                    for i in range(len(cov_par_init_NN[0])):
                        log_state += list(cov_parA[0][i].flatten())
                    # weights mu
                    for i in range(len(cov_par_init_NN[1])):
                        log_state += list(cov_parA[1][i].flatten())
                if BDNNmodel in [2, 3]:
                    # weights q
                    for i in range(len(cov_par_init_NN[2])):
                        log_state += list(cov_parA[2][i].flatten())
                if BDNNmodel in [1, 3]:
                    log_state += [cov_parA[3]]
                    if independ_reg:
                        log_state += [cov_parA[4]]
                    else:
                        log_state += [cov_parA[3]]
                    log_state += [denom_lamA]
                    log_state += [denom_muA]
                if BDNNmodel in [2, 3]:
                    log_state += [cov_parA[5], denom_qA, norm_facA]
            
            if sp_specific_q_rates:
                sp_q_rates = []
                for i in range(len(tsA)):
                    w_rates = HPP_vec_lik([teA[i],tsA[i],q_time_frames,q_ratesA,i,alpha_pp_gamma], return_rate=True)
                    sp_q_rates.append(w_rates)
                
                sp_q_marg.writerow([it, alpha_pp_gammaA] + sp_q_rates)
                sp_q_marg_rate_file.flush()
                os.fsync(sp_q_marg_rate_file)
            
            if fix_SE == 0:
                log_state += list(tsA)
                log_state += list(teA)

            if tune_T_schedule[0] > 0:
                log_state += list(np.mean(tste_tune_obj[:, 2]).flatten())
                if np.any(LO > 0):
                    log_state += list(np.mean(tste_tune_obj[LO > 0, 5]).flatten())
                log_state += list(np.mean(d1_ts).flatten())
                if np.any(LO > 0):
                    log_state += list(np.mean(d1_te[LO > 0]).flatten())

            wlog.writerow(log_state)
            logfile.flush()
            os.fsync(logfile)

            lik_tmp += sum(likBDtempA)

            if log_marginal_rates_to_file==1:
                if TDI in [0,2,4] and n_proc==0 and use_ADE_model == 0 and useDiscreteTraitModel == 0:
                    margL=np.zeros(len(marginal_frames))
                    margM=np.zeros(len(marginal_frames))
                    if useBounded_BD == 1: min_marginal_frame = boundMin
                    else: min_marginal_frame = min(LO)

                    for i in range(len(timesLA)-1): # indexes of the 1My bins within each timeframe
                        ind=np.intersect1d(marginal_frames[marginal_frames<=timesLA[i]],marginal_frames[marginal_frames>=np.maximum(min_marginal_frame,timesLA[i+1])])
                        j=array(ind)
                        margL[j]=LA[i]
                    for i in range(len(timesMA)-1): # indexes of the 1My bins within each timeframe
                        ind=np.intersect1d(marginal_frames[marginal_frames<=timesMA[i]],marginal_frames[marginal_frames>=np.maximum(min_marginal_frame,timesMA[i+1])])
                        j=array(ind)
                        margM[j]=MA[i]
                    marginal_rates(it, margL, margM, marginal_file, n_proc)
                if n_proc==0 and TDI==3: # marg rates DPP | times of shift are fixed and equal for L and M
                    margL=zeros(len(marginal_frames))
                    margM=zeros(len(marginal_frames))
                    for i in range(len(timesLA)-1): # indexes of the 1My bins within each timeframe
                        ind=np.intersect1d(marginal_frames[marginal_frames<=timesLA[i]],marginal_frames[marginal_frames>=timesLA[i+1]])
                        j=array(ind)
                        margL[j]=LA[indDPP_L[i]]
                        margM[j]=MA[indDPP_M[i]]
                    marginal_rates(it, margL, margM, marginal_file, n_proc)
            elif TDI in [0,2,4] and log_marginal_rates_to_file==0 and not BDNNmodel:
                w_marg_sp.writerow(list(LA) + list(timesLA[1:len(timesLA)-1]))
                marginal_sp_rate_file.flush()
                os.fsync(marginal_sp_rate_file)
                w_marg_ex.writerow(list(MA) + list(timesMA[1:len(timesMA)-1]))
                marginal_ex_rate_file.flush()
                os.fsync(marginal_ex_rate_file)
            
            if BDNNmodel:
                # log harmonic mean of rates through time
                if BDNNmodel in [1, 3]:
                    if use_time_as_trait or bdnn_timevar[0] or bdnn_dd or bdnn_loaded_tbls_timevar:
                        sp_lam = np.zeros(len(timesLA) - 1)
                        sp_mu = np.zeros(len(timesLA) - 1)
                        for temp_l in range(len(timesLA)-1):
                            sp_lam_tmp = bdnn_lam_ratesA[:, temp_l]
                            sp_mu_tmp = bdnn_mu_ratesA[:, temp_l]
                            indx = get_sp_indx_in_timeframe(tsA, teA, up = timesLA[temp_l], lo = timesLA[temp_l + 1])
                            sp_lam[temp_l] = 1 / np.mean(1 / sp_lam_tmp[indx])
                            sp_mu[temp_l] = 1 / np.mean(1 / sp_mu_tmp[indx])
                    else:
                        sp_lam = np.zeros(len(times_rtt) - 1)
                        sp_mu = np.zeros(len(times_rtt) - 1)
                        for temp_l in range(len(times_rtt)-1):
                            indx = get_sp_indx_in_timeframe(tsA, teA, up = times_rtt[temp_l], lo = times_rtt[temp_l + 1])
                            sp_lam[temp_l] = 1 / np.mean(1 / bdnn_lam_ratesA[indx])
                            sp_mu[temp_l] = 1 / np.mean(1 / bdnn_mu_ratesA[indx])
                    w_marg_sp.writerow(list(sp_lam) + list(fixed_times_of_shift_bdnn_logger))
                    marginal_sp_rate_file.flush()
                    os.fsync(marginal_sp_rate_file)
                    w_marg_ex.writerow(list(sp_mu) + list(fixed_times_of_shift_bdnn_logger))
                    marginal_ex_rate_file.flush()
                    os.fsync(marginal_ex_rate_file)
                if BDNNmodel in [2, 3]:
                    # get marginal q rate through time
                    if bdnn_ads > 0.0 and bdnn_time_res > bdnn_ads:
                        qtt = harmonic_mean_q_through_time(tsA, teA, q_time_frames_bdnn, bdnn_q_ratesA)
                    elif use_HPP_NN_lik and 'highres_q_repeats_rtt' in globals():
                        qtt = harmonic_mean_q_through_time(tsA, teA, times_q_shift_rtt, bdnn_q_ratesA[..., highres_q_repeats_rtt])
                    else:
                        qtt = harmonic_mean_q_through_time(tsA, teA, times_q_shift_rtt, bdnn_q_ratesA)
                    qtt = list(qtt) + list(times_q_shift_rtt[1:-1])
                    w_marg_q.writerow(qtt)
                    marginal_q_rate_file.flush()
                    os.fsync(marginal_q_rate_file)
                    if not samplingNN_TDI0:
                        w_marg_sp.writerow(list(LA) + list(timesLA[1:len(timesLA)-1]))
                        marginal_sp_rate_file.flush()
                        os.fsync(marginal_sp_rate_file)
                        w_marg_ex.writerow(list(MA) + list(timesMA[1:len(timesMA)-1]))
                        marginal_ex_rate_file.flush()
                        os.fsync(marginal_ex_rate_file)
                
                if log_per_species_rates and BDNNmodel in [1, 3]:
                    # get time-trait dependent rate at ts (speciation) and te (extinction) | (only works with bdnn_const_baseline)
                    arg_taxon_rates = [tsA, teA, timesLA, timesMA, bdnn_lam_ratesA, bdnn_mu_ratesA]
                    sp_lam_vec, sp_mu_vec = get_taxon_rates_bdnn(arg_taxon_rates)
                    species_rate_writer.writerow([it] + list(sp_lam_vec) + list(sp_mu_vec))
                    species_rate_file.flush()
                    os.fsync(species_rate_file)
        
        it += 1
    if TDI==1 and n_proc==0: marginal_likelihood(marginal_file, marginal_lik, temperatures)
    if use_seq_lik == 0 or num_processes > 0:
        pool_lik.close()
        pool_lik.join()
        if frac1>=0 and num_processes_ts > 0:
            pool_ts.close()
            pool_ts.join()
    return [it, n_proc,PostA, likA, priorA,tsA,teA,timesLA,timesMA,LA,MA,q_ratesA, cov_parA,lik_fossilA,likBDtempA]

def marginal_rates(it, margL,margM, marginal_file, run):
    log_state= [it]
    log_state += list(margL)
    log_state += list(margM)
    log_state += list(margL-margM)
    wmarg.writerow(log_state)
    marginal_file.flush()
    os.fsync(marginal_file)

def marginal_likelihood(marginal_file, l, t):
    mL=0
    for i in range(len(l)-1): mL+=((l[i]+l[i+1])/2.)*(t[i]-t[i+1]) # Beerli and Palczewski 2010
    print("\n Marginal likelihood:", mL)
    o= "\n Marginal likelihood: %s\n\nlogL: %s\nbeta: %s" % (mL,l,t)
    marginal_file.writelines(o)
    marginal_file.close()
    
    
# bdnn class
class bdnn():
    def __init__(self,
                 bdnn_settings=None,
                 weights=None,
                 trait_tbls=None,
                 sp_fad_lad=None,
                 occ_data=None):
        self.bdnn_settings = bdnn_settings
        self.v = version + build
        self.weights = weights
        self.trait_tbls = trait_tbls
        self.sp_fad_lad = sp_fad_lad
        self.occ_data = occ_data  
    
    def func(self, x="f"):
        print(x)
    
    def reset(self):
        self.f = func

########################## PARSE ARGUMENTS #######################################
if __name__ == '__main__': 
    
    print(("""
                     %s - %s

              Bayesian estimation of origination,
               extinction and preservation rates
                  from fossil occurrence data

                     pyrate.help@gmail.com

    \n""" % (version, build)))
    
    if hasFoundPyRateC:
        print("Module FastPyRateC was loaded.")
    else:
        print("Module FastPyRateC was not found.")
    
    
    
    p = argparse.ArgumentParser() #description='<input file>')

    p.add_argument('-v',         action='version', version=version_details)
    p.add_argument('-seed',      type=int, help='random seed', default=-1,metavar=-1)
    p.add_argument('-useCPPlib', type=int, help='Use C++ library if available (boolean)', default=1,metavar=1)
    p.add_argument('-cite',      help='print PyRate citation', action='store_true', default=False)
    p.add_argument('input_data', metavar='<input file>', type=str,help='Input python file - see template',default=[],nargs='*')
    p.add_argument('-j',         type=int, help='number of data set in input file', default=1, metavar=1)
    p.add_argument('-trait',     type=int, help='number of trait for Cov model', default=1, metavar=1)
    p.add_argument('-logT',      type=int, help='Transform trait (or rates for -plotRJ): 0) False, 1) Ln(x), 2) Log10(x)', default=0, metavar=0)
    p.add_argument("-N",         type=int, help='number of exant species', default=-1)
    p.add_argument("-wd",        type=str, help='path to working directory', default="")
    p.add_argument("-out",       type=str, help='output tag', default="")
    p.add_argument('-singleton', type=float, help='Remove singletons (min no. occurrences)', default=0, metavar=0)
    p.add_argument('-frac_sampled_singleton', type=float, help='Random fraction of singletons not removed', default=0, metavar=0)
    p.add_argument("-rescale",   type=float, help='Rescale data (e.g. -rescale 1000: 1 -> 1000, time unit = 1Ky)', default=1, metavar=1)
    p.add_argument("-translate", type=float, help='Shift data (e.g. -translate 10: 1My -> 10My)', default=0, metavar=0)
    p.add_argument('-d',         type=str,help="Load SE table",metavar='<input file>',default="")
    p.add_argument('-clade',     type=int, help='clade analyzed (set to -1 to analyze all species)', default=-1, metavar=-1)
    p.add_argument('-trait_file',type=str,help="Load trait table",metavar='<input file>',default="")
    p.add_argument('-restore_mcmc',type=str,help="Load mcmc.log file",metavar='<input file>',default="")
    p.add_argument('-filter',     type=float,help="Filter lineages with all occurrences within time range ",default=[inf,0], metavar=inf, nargs=2)
    p.add_argument('-filter_taxa',type=str,help="Filter lineages within list (drop all others) ",default="", metavar="taxa_file")
    p.add_argument('-initDiv',    type=int, help='Number of initial lineages (option only available with -d SE_table or -fixSE)', default=0, metavar=0)
    p.add_argument('-PPmodeltest',help='Likelihood testing among preservation models', action='store_true', default=False)
    p.add_argument('-log_marginal_rates',type=int,help='0) save summary file, default for -A 4; 1) save marginal rate file, default for -A 0,2 ', default=-1,metavar=-1)
    p.add_argument('-log_sp_q_rates', help='Save species-specific relative preservation rates', action='store_true', default=False)
    p.add_argument("-drop_zero",  type=float, help='remove 0s from occs data (set to 1 to drop all)', default=0, metavar=0)
    p.add_argument("-drop_internal",  action='store_true', default=False)

    # phylo test
    p.add_argument('-tree',       type=str,help="Tree file (NEXUS format)",default="", metavar="")
    p.add_argument('-sampling',   type=float,help="Taxon sampling (phylogeny)",default=1., metavar=1.)
    p.add_argument('-bdc',      help='Run BDC:Compatible model', action='store_true', default=False)
    p.add_argument('-eqr',      help='Run BDC:Equal rate model', action='store_true', default=False)

    # PLOTS AND OUTPUT
    p.add_argument('-plot',       metavar='<input file>', type=str,help="RTT plot (type 1): provide path to 'marginal_rates.log' files or 'marginal_rates' file",default="")
    p.add_argument('-plot2',      metavar='<input file>', type=str,help="RTT plot (type 2): provide path to 'marginal_rates.log' files or 'marginal_rates' file",default="")
    p.add_argument('-plot3',      metavar='<input file>', type=str,help="RTT plot for fixed number of shifts: provide 'mcmc.log' file",default="")
    p.add_argument('-plotRJ',     metavar='<input file>', type=str,help="RTT plot for runs with '-log_marginal_rates 0': provide path to 'mcmc.log' files",default="")
    p.add_argument('-plotBDNN',   metavar='<input file>', type=str,help="RTT plot for BDNN runs: provide path to the 'mcmc.log' file",default="")
    p.add_argument('-plotBDNN_groups', metavar='<input file>', type=str, help="Path to tab-separated text file giving the taxa for which the RTT plot should be created (see BDNN tutorial)", default = "")
    p.add_argument('-plotBDNN_effects',   metavar='<input file>', type=str, help="Effect plot for BDNN runs: provide path and base name for 'mcmc.log' file (e.g. .../pyrate_mcmc_logs/example_BDS_BDNN_16_8Tc_mcmc.log)", default = "")
    p.add_argument('-plotBDNN_transf_features', metavar='<input file>', type=str,
                   help="Optional back transformation of z-standardized BDNN features (text file with name of the feature as header, its mean, and standard deviation before z-standardization", default = "")
    p.add_argument('-BDNN_groups', metavar='<dict>', type=json.loads,
                    help="""dictionary with features to plot together (e.g. on-hot encoded discrete features). E.g.: '{"Trait1": ["T1_state1", "T1_state2", "T1_state3"], "Trait2": ["T2_state1", "T2_state2", "T2_state3", "T2_state4"]}'""", default = '{}')
    p.add_argument('-BDNN_interaction',   metavar='<input file>', type=str, help="""Create text files with PDP rates for k-way interactions for BDNN runs: provide path for 'mcmc.log' file (e.g. .../pyrate_mcmc_logs/example_BDS_BDNN_16_8Tc_mcmc.log); use -BDNN_groups to specify features. E.g. '{"Trait1": ["Trait1"], "Trait2": ["Trait2"], "Trait3": ["T3_state1", "T3_state2", "T3_state3"]}'""", default = "")
    p.add_argument("-BDNN_interaction_fix", help='Fix predictors specified for "-BDNN_interaction" to the ones in the trait file', action='store_true', default=False)
    p.add_argument('-BDNN_PDRTT',   metavar='<input file>', type=str,help="PD rates through time for BDNN: provide path to the 'mcmc.log' file and features via -BDNN_groups", default="")
    p.add_argument('-n_prior',     type=int,help="n. samples from the prior to compute Bayes factors",default=100000)
    p.add_argument('-plotQ',      metavar='<input file>', type=str,help="Plot preservation rates through time: provide 'mcmc.log' file and '-qShift' argument ",default="")
    p.add_argument('-grid_plot',  type=float, help='Plot resolution in Myr (only for plot3 and plotRJ commands). If set to 0: 100 equal time bins', default=0, metavar=0)
    p.add_argument('-root_plot',  type=float, help='User-defined root age for RTT plots', default=0, metavar=0)
    p.add_argument('-min_age_plot',type=float, help='User-defined minimum age for RTT plots (only with plotRJ option)', default=0, metavar=0)
    p.add_argument('-tag',        metavar='<*tag*.log>', type=str,help="Tag identifying files to be combined and plotted (-plot and -plot2) or summarized in SE table (-ginput)",default="")
    p.add_argument('-ltt',        type=int,help='1) Plot lineages-through-time; 2) plot Log10(LTT)', default=0, metavar=0)
    p.add_argument('-mProb',      type=str,help="Input 'mcmc.log' file",default="")
    p.add_argument('-BF',         type=str,help="Input 'marginal_likelihood.txt' files",metavar='<2 input files>',nargs='+',default=[])
    p.add_argument("-data_info",  help='Summary information about an input data', action='store_true', default=False)
    p.add_argument('-SE_stats',   type=str,help="Calculate and plot stats from SE table:",metavar='<extinction rate at the present, bin_size, #_simulations>',nargs='+',default=[])
    p.add_argument('-ginput',     type=str,help='generate SE table from *mcmc.log files', default="", metavar="<path_to_mcmc.log>")
    p.add_argument('-combLog',    type=str,help='Combine (and resample) log files', default="", metavar="<path_to_log_files>")
    p.add_argument('-combLogRJ',  type=str,help='Combine (and resample) all log files form RJMCMC', default="", metavar="<path_to_log_files>")
    p.add_argument('-combBDNN',  type=str,help='Combine (and resample) all log files form BDNN', default="", metavar="<path_to_log_files>")
    p.add_argument('-resample',   type=int,help='Number of samples for each log file (-combLog). Use 0 to keep all samples. Number of ts/te estimates (-ginput). Number of BDNN samples (-plotBDNN and -plotBDNN_effects)', default=0, metavar=0)
    p.add_argument('-col_tag',    type=str,help='Columns to be combined using combLog', default=[], metavar="column names",nargs='+')
    p.add_argument('-check_names',type=str,help='Automatic check for typos in taxa names (provide SpeciesList file)', default="", metavar="<*_SpeciesList.txt file>")
    p.add_argument('-reduceLog',  type=str,help='Reduce file size (mcmc.log) to quickly assess convergence', default="", metavar="<*_mcmc.log file>")

    # MCMC SETTINGS
    p.add_argument('-n',      type=int, help='mcmc generations',default=10000000, metavar=10000000)
    p.add_argument('-s',      type=int, help='sample freq.', default=1000, metavar=1000)
    p.add_argument('-p',      type=int, help='print freq.',  default=1000, metavar=1000)
    p.add_argument('-b',      type=float, help='burnin', default=0, metavar=0)
    p.add_argument('-fast_burnin',      type=float, help='n. fast-burnin generations', default=0, metavar=0)
    p.add_argument('-thread', type=int, help='no. threads used for BD and NHPP likelihood respectively (set to 0 to bypass multi-threading)', default=[0,0], metavar=4, nargs=2)

    # MCMC ALGORITHMS
    p.add_argument('-A',        type=int, help='0) parameter estimation, 1) marginal likelihood, 2) BDMCMC, 3) DPP, 4) RJMCMC', default=4, metavar=4)
    p.add_argument("-use_DA",   help='Use data augmentation for NHPP likelihood opf extant taxa', action='store_true', default=False)
    p.add_argument('-r',        type=int,   help='MC3 - no. MCMC chains', default=1, metavar=1)
    p.add_argument('-t',        type=float, help='MC3 - temperature', default=.03, metavar=.03)
    p.add_argument('-sw',       type=float, help='MC3 - swap frequency', default=100, metavar=100)
    p.add_argument('-M',        type=int,   help='BDMCMC/RJMCMC - frequency of model update', default=10, metavar=10)
    p.add_argument('-B',        type=int,   help='BDMCMC - birth rate', default=1, metavar=1)
    p.add_argument('-T',        type=float, help='BDMCMC - time of model update', default=1.0, metavar=1.0)
    p.add_argument('-S',        type=int,   help='BDMCMC - start model update', default=1000, metavar=1000)
    p.add_argument('-k',        type=int,   help='TI - no. scaling factors', default=10, metavar=10)
    p.add_argument('-a',        type=float, help='TI - shape beta distribution', default=.3, metavar=.3)
    p.add_argument('-dpp_f',    type=float, help='DPP - frequency ', default=500, metavar=500)
    p.add_argument('-dpp_hp',   type=float, help='DPP - shape of gamma HP on concentration parameter', default=2., metavar=2.)
    p.add_argument('-dpp_eK',   type=float, help='DPP - expected number of rate categories', default=2., metavar=2.)
    p.add_argument('-dpp_grid', type=float, help='DPP - size of time bins',default=1.5, metavar=1.5)
    p.add_argument('-dpp_nB',   type=float, help='DPP - number of time bins',default=0, metavar=0)
    p.add_argument('-rj_pr',       type=float, help='RJ - proposal (0: Gamma, 1: Weighted mean) ', default=1, metavar=1)
    p.add_argument('-rj_Ga',       type=float, help='RJ - shape of gamma proposal (if rj_pr 0)', default=1.5, metavar=1.5)
    p.add_argument('-rj_Gb',       type=float, help='RJ - rate of gamma proposal (if rj_pr 0)',  default=3., metavar=3.)
    p.add_argument('-rj_beta',     type=float, help='RJ - shape of beta multiplier (if rj_pr 1)',default=10, metavar=10)
    p.add_argument('-rj_dm',       type=float, help='RJ - allow double moves (0: no, 1: yes)',default=0, metavar=0)
    p.add_argument('-rj_bd_shift', type=float, help='RJ - 0: only sample shifts in speciation; 1: only sample shifts in extinction',default=0.5, metavar=0.5)
    p.add_argument('-se_gibbs',    help='Use aprroximate S/E Gibbs sampler', action='store_true', default=False)

    # PRIORS
    p.add_argument('-pL',      type=float, help='Prior - speciation rate (Gamma <shape, rate>) | (if shape=n,rate=0 -> rate estimated)', default=[1.1, 1.1], metavar=1.1, nargs=2)
    p.add_argument('-pM',      type=float, help='Prior - extinction rate (Gamma <shape, rate>) | (if shape=n,rate=0 -> rate estimated)', default=[1.1, 1.1], metavar=1.1, nargs=2)
    p.add_argument('-pP',      type=float, help='Prior - preservation rate (Gamma <shape, rate>) | (if shape=n,rate=0 -> rate estimated)', default=[1.5, 1.1], metavar=1.5, nargs=2)
    p.add_argument('-pS',      type=float, help='Prior - time frames (Dirichlet <shape>)', default=2.5, metavar=2.5)
    p.add_argument('-pC',      type=float, help='Prior - Covar parameters (Normal <standard deviation>) | (if pC=0 -> sd estimated)', default=1, metavar=1)
    p.add_argument("-cauchy",  type=float, help='Prior - use hyper priors on sp/ex rates (if 0 -> estimated)', default=[-1, -1], metavar=-1, nargs=2)
    p.add_argument("-min_dt",  type=float, help='Prior - minimum allowed distance between rate shifts', default=1., metavar=1)

    # MODEL
    p.add_argument("-mHPP",    help='Model - Homogeneous Poisson process of preservation', action='store_true', default=False)
    #p.add_argument("-TPP_model",help='Model - Poisson process of preservation with shifts', action='store_true', default=False)
    p.add_argument('-mL',      type=int, help='Model - no. (starting) time frames (speciation)', default=1, metavar=1)
    p.add_argument('-mM',      type=int, help='Model - no. (starting) time frames (extinction)', default=1, metavar=1)
    p.add_argument('-mC',      help='Model - constrain time frames (l,m)', action='store_true', default=False)
    p.add_argument('-mCov',    type=int, help='COVAR model: 1) speciation, 2) extinction, 3) speciation & extinction, 4) preservation, 5) speciation & extinction & preservation', default=0, metavar=0)
    p.add_argument("-mG",      help='Model - Gamma heterogeneity of preservation rate', action='store_true', default=False)
    p.add_argument('-mPoiD',   help='Poisson-death diversification model', action='store_true', default=False)
    p.add_argument('-mBirth',  type=float, help='Birth model with fix extinction rate', default= -1, metavar= -1)
    p.add_argument('-mDeath',  help='Pure-death model', action='store_true', default=False)
    p.add_argument("-mBDI",    type=int, help='BDI sub-model - 0) birth-death, 1) immigration-death', default=-1, metavar=-1)
    p.add_argument("-ncat",    type=int, help='Model - Number of categories for Gamma heterogeneity', default=4, metavar=4)
    p.add_argument('-fixShift',metavar='<input file>', type=str,help="Input tab-delimited file",default="")
    p.add_argument('-qShift',  metavar='<input file>', type=str,help="Poisson process of preservation with shifts (Input tab-delimited file)",default="")
    p.add_argument('-fixSE',   metavar='<input file>', type=str,help="Input mcmc.log file",default="")
    p.add_argument('-ADE',     type=int, help='ADE model: 0) no age dependence 1) estimated age dep', default=0, metavar=0)
    p.add_argument('-discrete',help='Discrete-trait-dependent BD model (requires -trait_file)', action='store_true', default=False)
    p.add_argument('-twotrait',help='Discrete-trait-dependent extinction + Covar', action='store_true', default=False)
    p.add_argument('-bound',   type=float, help='Bounded BD model', default=[np.inf, 0], metavar=0, nargs=2)
    p.add_argument('-bound_se', type=str, help="Path to a text file with three columns: taxon, minimum origination time, maximum extinction time", metavar='<input file>',default="")
    p.add_argument('-partialBD', help='Partial BD model (with -d)', action='store_true', default=False)
    p.add_argument('-edgeShift',type=float, help='Fixed times of shifts at the edges (when -mL/-mM > 3)', default=[np.inf, 0], metavar=0, nargs=2)
    p.add_argument('-qFilter', type=int, help='if set to zero all shifts in preservation rates are kept, even if outside observed timerange', default=1, metavar=1)
    p.add_argument('-FBDrange', type=int, help='use FBDrange likelihood (experimental)', default=0, metavar=0)
    p.add_argument('-BDNNmodel', type=int, help='use neural network model for 1) speciation & extinction, 2) sampling, 3) speciation & extinction & sampling', default=0, metavar=0)
    p.add_argument('-BDNNnodes', type=int, help='number of BD-NN nodes', nargs='+',default=[16, 8])
    p.add_argument('-BDNNfadlad', type=float, help='if > 0 include FAD LAD as traits (rescaled i.e. FAD * BDNNfadlad)', default=0, metavar=0)
    p.add_argument('-BDNNtimetrait', type=float, help='if > 0 use (rescaled) time as a trait (only with -fixShift option). if = -1 auto-rescaled', default= -1, metavar= -1)
    p.add_argument('-BDNNtimeres', type=float, help='Time resolution for the BDNN model. Is overridden by -fixShift argument.', default=1, metavar=1)
    p.add_argument('-BDNNconstbaseline', type=int, help='constant baseline rates (only with -fixShift option AND time as a trait)', default=1, metavar=1)
    p.add_argument('-BDNNoutputfun', type=int, help='Activation function output layer: 0) abs, 1) softPlus, 2) exp, 3) relu 4) sigmoid 5) sigmoid_rate', default=1, metavar=1)
    p.add_argument('-BDNNactfun', type=int, help='Activation function hidden layer(s): 0) tanh, 1) relu, 2) leaky_relu, 3) swish, 4) sigmoid, 5) fast approximation tanh', default=5, metavar=0)
    p.add_argument('-BDNNprecision', type=int, help='Floating point precision for network nodes: 0) double 64, 1) single 32', default=1, metavar=1)
    p.add_argument('-BDNNprior', type=float, help='sd normal prior', default=1, metavar=1)
    p.add_argument('-BDNNreg', type=float, help='regularization prior (-1.0 to turn off regularization, provide two values for independent regularization of lam and mu)', default=[1.0], metavar=[1.0], nargs='+')
    p.add_argument('-BDNNblockmodel',help='Block NN model', action='store_true', default=False)
    p.add_argument('-BDNNactivate_all',help='Activate all hidden layers even if weights were not updated; increases run time', action='store_true', default=False)
    p.add_argument('-BDNNtimevar', type=str, help='Time variable file for birth-death process (e.g. PhanerozoicTempSmooth.txt), several variable in different columns possible. If paths to two different time variable files are provided, the first one is used for lambda and the second for mu', default=[""], metavar="", nargs='+')
    p.add_argument('-BDNNtimevar_q', type=str, help='Time variable file for sampling process, several variable in different columns possible', default="", metavar="")
    p.add_argument('-BDNNads', type=float, help='(Relative)age-dependent sampling (-1.0: off; 0.0: use qShifts; >0.0 resample qShifts to this value)', default=-1.0, metavar=1)
    p.add_argument('-BDNNpath_taxon_time_tables', type=str, help='Path to director(y|ies) with table(s) of taxon-time specific predictors. One path for identical speciation/extinction predictors, two paths if they differ.', default=["", ""], nargs='+')
    p.add_argument('-BDNNexport_taxon_time_tables', help='Export BDNN predictors. Creates a new directory with one text file per time bin (from most recent to earliest).', action='store_true', default=False)
    p.add_argument('-BDNNupdate_se_f', type=float, help='fraction of updated times of origination and extinction', default=[0.6], metavar=[0.6], nargs=1)
    p.add_argument('-BDNNupdate_f', type=float, help='fraction of updated weights', default=[0.1], metavar=[0.1], nargs='+')
    p.add_argument('-BDNNdd', help='Diversity-dependent BDNN', action='store_true', default=False)
    p.add_argument('-BDNNpklfile', type=str, help='Load BDNN pickle file', default="", metavar="")
    p.add_argument('-BDNN_pred_importance', metavar='<input file>', type = str, help = "Predictor importance in BDNN: provide path for 'mcmc.log' file (e.g. .../pyrate_mcmc_logs/example_BDS_BDNN_16_8Tc_mcmc.log)", default = "")
    p.add_argument('-BDNN_nsim_expected_cv', type=int, help='Number of simulations to get expected coefficient of rate variation', default=100, metavar=100)
    p.add_argument('-BDNN_pred_importance_interaction', help='Obtain importance for two-way interactions in addition to the main effects', action='store_false', default=True)
    p.add_argument('-BDNN_pred_importance_window_size', type=float, help='Resample to time windows of a given size. Same value for birth-death and sampling or two different values.', default=[-1.0], metavar=[-1.0], nargs='+')
    p.add_argument('-BDNN_pred_importance_nperm', type=int, help='Number of permutation for BDNN predictor importance', default=100, metavar=100)
    p.add_argument('-BDNN_mean_shap_per_group', help='Calculating shap value for BDNN_groups using the mean importance across all grouped predictors', action='store_true', default=False)
    
    p.add_argument("-edge_indicator",      help='Model - Gamma heterogeneity of preservation rate', action='store_true', default=False)
    

    # TUNING
    p.add_argument('-tT',     type=float, help='Tuning - window size (ts, te)', default=1., metavar=1.)
    p.add_argument('-nT',     type=int,   help='Tuning - max number updated values (ts, te)', default=5, metavar=5)
    p.add_argument('-tuneT',  type=float, help='Autotuning window sizes tT. Maximum iteration and tuning interval', default=[0, 1000], nargs=2)
    p.add_argument('-tQ',     type=float, help='Tuning - window sizes (q/alpha: 1.2 1.2)', default=[1.2,1.2], nargs=2)
    p.add_argument('-tR',     type=float, help='Tuning - window size (rates)', default=1.2, metavar=1.2)
    p.add_argument('-tS',     type=float, help='Tuning - window size (time of shift)', default=1., metavar=1.)
    p.add_argument('-fR',     type=float, help='Tuning - fraction of updated values (rates)', default=.5, metavar=.5)
    p.add_argument('-fS',     type=float, help='Tuning - fraction of updated values (shifts)', default=.7, metavar=.7)
    p.add_argument('-fQ',     type=float, help='Tuning - fraction of updated values (q rates, TPP)', default=.5, metavar=.5)
    p.add_argument('-tC',     type=float, help='Tuning - window sizes cov parameters (l,m,q)', default=[.2, .2, .15], nargs=3)
    p.add_argument('-fU',     type=float, help='Tuning - update freq. (q: .02, l/m: .18, cov: .08)', default=[.02, .18, .08], nargs=3)
    p.add_argument('-multiR', type=int,   help='Tuning - Proposals for l/m: 0) sliding win 1) muliplier ', default=1, metavar=1)
    p.add_argument('-tHP',    type=float, help='Tuning - window sizes hyperpriors on l and m', default=[1.2, 1.2], nargs=2)

    args = p.parse_args()
    t1=time.time()

    if args.seed==-1:
        rseed=np.random.randint(0,9999)
    else: rseed=args.seed
    rand.seed(rseed)  # set as argument/ use get seed function to get it and save it to sum.txt file
    random.seed(rseed)
    np.random.seed(rseed)

    FBDrange = args.FBDrange
    # 1: OCCS model 2: FALA model


    if args.useCPPlib==1 and hasFoundPyRateC == 1:
        #print("Loaded module FastPyRateC")
        CPPlib="\nUsing module FastPyRateC"
    else:
        hasFoundPyRateC= 0
        CPPlib=""

    if args.cite:
        sys.exit(citation)
    ############################ MODEL SETTINGS ############################
    # PRIORS
    L_lam_r,L_lam_m = args.pL # shape and rate parameters of Gamma prior on sp rates
    M_lam_r,M_lam_m = args.pM # shape and rate parameters of Gamma prior on ex rates
    lam_s = args.pS                              # shape parameter dirichlet prior on time frames
    pert_prior = [args.pP[0],args.pP[1]] # gamma prior on foss. rate; beta on mode PERT distribution
    covar_prior_fixed=args.pC # std of normal prior on th covariance parameters

    # MODEL
    time_framesL=args.mL          # no. (starting) time frames (lambda)
    time_framesM=args.mM          # no. (starting) time frames (mu)
    constrain_time_frames=args.mC # True/False
    pp_gamma_ncat=args.ncat              # args.ncat
    YangGammaQuant = None
    if args.mG:             # number of gamma categories
        argsG = 1
        YangGammaQuant=(np.linspace(0,1,pp_gamma_ncat+1)-np.linspace(0,1,pp_gamma_ncat+1)[1]/2)[1:]
    else: argsG = 0
    model_cov=args.mCov           # boolean 0: no covariance 1: covariance (speciation,extinction) 2: covariance (speciation,extinction,preservation)

    sp_specific_q_rates = args.log_sp_q_rates
    if sp_specific_q_rates:
        if argsG == 0 or args.qShift == "":
            sys.exit("option only available with TPP + Gamma model")
    
    edge_indicator = args.edge_indicator

    if args.mHPP: argsHPP=1
    else: argsHPP=0
    ############################ MCMC SETTINGS ############################
    # GENERAL SETTINGS
    TDI=args.A                  # 0: parameter estimation, 1: thermodynamic integration, 2: BD-MCMC
    BDNNmodel = args.BDNNmodel
    if BDNNmodel in [1, 3]:
        TDI = 0
    if constrain_time_frames == 1 or args.fixShift != "":
        if TDI in [2,4]:
            # print("\nConstrained shift times (-mC,-fixShift) cannot be used with BD/RJ MCMC alorithms. Using standard MCMC instead.\n")
            TDI = 0
    if args.ADE>=1 and TDI>1:
        # print("\nADE models (-ADE 1) cannot be used with BD/RJ MCMC alorithms. Using standard MCMC instead.\n")
        TDI = 0
    mcmc_gen=args.n             # no. total mcmc generations
    sample_freq=args.s
    print_freq=args.p
    burnin=args.b
    num_processes = args.thread[0]    # BDlik
    num_processes_ts = args.thread[1] # NHPPlik
    if BDNNmodel and num_processes > 0:
        print(sys.exit("Neural-network model model can only run on a single processor"))
    if num_processes+num_processes_ts==0: use_seq_lik = 1
    if use_seq_lik == 1: num_processes,num_processes_ts=0,0
    min_allowed_t=args.min_dt

    # RJ arguments
    addrm_proposal_RJ = args.rj_pr      # 0: random Gamma; 1: weighted mean
    shape_gamma_RJ    = args.rj_Ga
    rate_gamma_RJ     = args.rj_Gb
    shape_beta_RJ     = args.rj_beta
    if addrm_proposal_RJ == 0:
        add_shift_RJ    = add_shift_RJ_rand_gamma
        remove_shift_RJ = remove_shift_RJ_rand_gamma
    elif addrm_proposal_RJ == 1:
        add_shift_RJ    = add_shift_RJ_weighted_mean
        remove_shift_RJ = remove_shift_RJ_weighted_mean
    allow_double_move = args.rj_dm



    # TUNING
    d1=args.tT                     # win-size (ts, te)
    frac1= args.nT                 # max number updated values (ts, te)
    tune_T_schedule = args.tuneT   # schedule tuning win-size (ts, te)
    if args.tuneT[0] > 0 and args.tuneT[0] < 1:
        tune_T_schedule[0] = int(args.tuneT[0] * mcmc_gen)        
    
    d2=args.tQ                     # win-sizes (q,alpha)
    d3=args.tR                     # win-size (rates)
    f_rate=args.fR                 # fraction of updated values (rates)
    d4=args.tS                     # win-size (time of shift)
    f_shift=args.fS                # update frequency (time of shift) || will turn into 0 when no rate shifts
    f_qrate_update =args.fQ        # update frequency (preservation rates under TPP model)
    freq_list=args.fU              # generate update frequencies by parm category
    d5=args.tC                     # win-size (cov)
    d_hyperprior=np.array(args.tHP)          # win-size hyper-priors onf l/m (or W_scale)
    if model_cov==0: freq_list[2]=0
    f_update_se=1-sum(freq_list)
    if frac1==0: f_update_se=0
    [f_update_q,f_update_lm,f_update_cov]=f_update_se+np.cumsum(array(freq_list))
    # print("f_update_se", f_update_se)

    if args.se_gibbs: 
        use_gibbs_se_sampling = 1
    else: use_gibbs_se_sampling = 0

    fast_burnin =args.fast_burnin


    multiR = args.multiR
    if multiR==0:
        update_rates =  update_rates_sliding_win
    else:
        update_rates = update_rates_multiplier
        d3 = np.maximum(args.tR,1.01) # avoid win size < 1


    if args.ginput != "" or args.check_names != "" or args.reduceLog != "":
        try:
            import pyrate_lib.lib_DD_likelihood
            import pyrate_lib.lib_utilities
            import pyrate_lib.check_species_names
        except:
            sys.exit("""\nWarning: library pyrate_lib not found.\nMake sure PyRate.py and pyrate_lib are in the same directory.
            You can download pyrate_lib here: <https://github.com/dsilvestro/PyRate> \n""")

        if args.ginput != "":
            n_samples = np.max([args.resample,1])
            pyrate_lib.lib_utilities.write_ts_te_table(args.ginput, tag=args.tag, clade=-1,burnin=int(burnin)+1, n_samples=n_samples)
        elif args.check_names != "":
            SpeciesList_file = args.check_names
            pyrate_lib.check_species_names.run_name_check(SpeciesList_file)
        elif args.reduceLog != "":
            pyrate_lib.lib_utilities.reduce_log_file(args.reduceLog,np.maximum(1,int(args.b)))
        quit()


    if args.use_DA: use_DA = 1
    else: use_DA = 0
    useBounded_BD = 0

    # freq update CovPar
    if model_cov==0: f_cov_par= [0  ,0  ,0 ]
    if model_cov==1: f_cov_par= [1  ,0  ,0 ]
    if model_cov==2: f_cov_par= [0  ,1  ,0 ]
    if model_cov==3: f_cov_par= [.5 ,1  ,0 ]
    if model_cov==4: f_cov_par= [0  ,0  ,1 ]
    if model_cov==5: f_cov_par= [.33,.66,1 ]

    if covar_prior_fixed==0: est_COVAR_prior = 1
    else: est_COVAR_prior = 0

    if args.fixShift != "" or TDI==3:     # fix times of rate shift or DPP
        try:
            try: 
                fixed_times_of_shift = np.sort(np.loadtxt(args.fixShift))[::-1]
            except: 
                fixed_times_of_shift = np.array([np.loadtxt(args.fixShift)])
            fixed_times_of_shift = fixed_times_of_shift[fixed_times_of_shift > 0]
            fixed_times_of_shift += args.translate
            f_shift=0
            time_framesL=len(fixed_times_of_shift)+1
            time_framesM=len(fixed_times_of_shift)+1
            min_allowed_t=0
            fix_Shift = 1
        except:
            if TDI==3:
                fixed_times_of_shift=np.arange(0,10000,args.dpp_grid)[::-1] # run fixed_times_of_shift[fixed_times_of_shift<max(FA)] below
                fixed_times_of_shift=fixed_times_of_shift[:-1]              # after loading input file
                f_shift=0
                time_framesL=len(fixed_times_of_shift)+1
                time_framesM=len(fixed_times_of_shift)+1
                min_allowed_t=0
                fix_Shift = 1
            else:
                msg = "\nError in the input file %s.\n" % (args.fixShift)
                sys.exit(msg)
    else:
        fixed_times_of_shift=[]
        fix_Shift = 0

    if args.edgeShift[0] != np.inf or args.edgeShift[1] != 0:
        edgeShifts = []
        if args.edgeShift[0] != np.inf: # max boundary
            edgeShifts.append(args.edgeShift[0])
            fix_edgeShift = 2
            min_allowed_n_rates = 2
        if args.edgeShift[1] != 0: # min boundary
            edgeShifts.append(args.edgeShift[1])
            fix_edgeShift = 3
            min_allowed_n_rates = 2
        if len(edgeShifts)==2: # min and max boundaries
            fix_edgeShift = 1
            min_allowed_n_rates = 3
        time_framesL = np.maximum(min_allowed_n_rates,args.mL) # change number of starting rates based on edgeShifts
        time_framesM = np.maximum(min_allowed_n_rates,args.mM) # change number of starting rates based on edgeShifts
        edgeShifts = np.array(edgeShifts)*args.rescale+args.translate
    else:
        fix_edgeShift = 0
        min_allowed_n_rates = 1

    use_time_as_trait = args.BDNNtimetrait != 0
    bdnn_time_res = args.BDNNtimeres
    fixed_times_of_shift_bdnn = []
    bdnn_loaded_tbls = args.BDNNpath_taxon_time_tables
    bdnn_loaded_tbls_timevar = False
    bdnn_loaded_timevar_pred = False
    if bdnn_loaded_tbls[0] != "":
        import pyrate_lib.bdnn_lib as bdnn_lib
        bdnn_loaded_tbls, bdnn_loaded_names_traits, bdnn_loaded_timevar_pred = bdnn_lib.load_trait_tbl(bdnn_loaded_tbls)
        if bdnn_loaded_tbls[0].ndim == 3:
            bdnn_loaded_tbls_timevar = True

    needs_bdnn_time = args.BDNNtimetrait != 0 or args.BDNNtimevar[0] or args.BDNNdd or bdnn_loaded_tbls_timevar
    if args.BDNNmodel in [1, 3]:
        if fix_edgeShift > 0 and not needs_bdnn_time:
            # No time-varying BDNN but we want edgeShifts
            use_time_as_trait = True
            fix_Shift = 1
            fixed_times_of_shift_bdnn = []
            if fix_edgeShift in [1, 2]: # both or max boundary
                fixed_times_of_shift_bdnn.append(edgeShifts[0])
            if fix_edgeShift in [1, 3]: # both or min boundary
                fixed_times_of_shift_bdnn.append(edgeShifts[-1])
            fixed_times_of_shift_bdnn = np.array(fixed_times_of_shift_bdnn)

        elif needs_bdnn_time:
            # if args.A == 4:
                # fixed_times_of_shift_bdnn = np.arange(1, 1000)[::-1]
                # time_framesL_bdnn=len(fixed_times_of_shift_bdnn)+1
                # time_framesM_bdnn=len(fixed_times_of_shift_bdnn)+1
                # TDI = 4
            TDI = 0
            if fix_Shift == 1:
                fixed_times_of_shift_bdnn = fixed_times_of_shift
            else:
                f_shift=0
                fixed_times_of_shift_bdnn = np.arange(bdnn_time_res, 1000, bdnn_time_res)[::-1]
                min_allowed_t=0
                fix_Shift = 1
            if fix_edgeShift > 0:
                # Trim all bins older and/or younger than the edgeShifts
                if fix_edgeShift in [1, 2]: # both or max boundary
                    fixed_times_of_shift_bdnn = fixed_times_of_shift_bdnn[fixed_times_of_shift_bdnn < edgeShifts[0]]
                if fix_edgeShift in [1, 3]: # both or min boundary
                    fixed_times_of_shift_bdnn = fixed_times_of_shift_bdnn[fixed_times_of_shift_bdnn > edgeShifts[-1]]
                fixed_times_of_shift_bdnn = np.sort(np.concatenate((fixed_times_of_shift_bdnn, edgeShifts), axis=None))[::-1]



    # BDMCMC & MCMC SETTINGS
    runs=args.r              # no. parallel MCMCs (MC3)
    if runs>1 and TDI>0:
        print("\nWarning: MC3 algorithm is not available for TI and BDMCMC. Using a single chain instead.\n")
        runs,TDI=1,0
    num_proc = runs          # processors MC3
    temp_pr=args.t           # temperature MC3
    IT=args.sw
    freq_Alg_3_1=args.M      # frequency of model update
    birthRate=args.B         # birthRate (=Poisson prior)
    len_cont_time=args.T     # length continuous time of model update
    start_Alg_3_1=args.S     # start sampling model after


    if runs==1 or use_seq_lik == 1:
        IT=mcmc_gen

    if TDI==1:                # Xie et al. 2011; Baele et al. 2012
        K=args.k-1.        # K+1 categories
        k=array(list(range(int(K+1))))
        beta=k/K
        alpha=args.a            # categories are beta distributed
        temperatures=list(beta**(1./alpha))
        temperatures[0]+= small_number # avoid exactly 0 temp
        temperatures.reverse()
        if multiR==0: # tune win sizes only if sliding win proposals
            list_d3=sort(exp(temperatures))**2.5*d3+(exp(1-array(temperatures))-1)*d3
        else:
            list_d3=np.repeat(d3,len(temperatures))
        list_d4=sort(exp(temperatures))**1.5*d4+exp(1-array(temperatures))-1
    else:
        temperatures=[1]
        list_d3=[d3]
        list_d4=[d4]

    # ARGS DPP
    freq_dpp       = args.dpp_f
    hp_gamma_shape = args.dpp_hp
    target_k       = args.dpp_eK

    ############### PLOT RTT
    path_dir_log_files=""
    if args.plot != "":
        path_dir_log_files=args.plot
        plot_type=1
    elif args.plot2 != "":
        path_dir_log_files=args.plot2
        plot_type=2
    elif args.plot3 != "":
        path_dir_log_files=args.plot3
        plot_type=3
    elif args.plotRJ != "":
        path_dir_log_files=args.plotRJ
        plot_type=4
    elif args.plotQ != "":
        path_dir_log_files=args.plotQ
        plot_type=5
    elif args.plotBDNN != "":
        path_dir_log_files = args.plotBDNN
        plot_type = 6
    elif args.BDNN_PDRTT != "":
        path_dir_log_files = args.BDNN_PDRTT
        plot_type = 7
    elif args.plotBDNN_effects != "":
        path_dir_log_files = args.plotBDNN_effects
        plot_type = 8

    #print(args.plotQ)

    list_files_BF=sort(args.BF)
    file_stem=args.tag
    root_plot=args.root_plot
    grid_plot = args.grid_plot
    if path_dir_log_files != "":
        self_path = get_self_path()
        if plot_type>=3 and plot_type != 8:
            import pyrate_lib.lib_DD_likelihood as lib_DD_likelihood
            import pyrate_lib.lib_utilities as lib_utilities
            import pyrate_lib.rtt_plot_bds as rtt_plot_bds
            if plot_type==3:
                if grid_plot==0: grid_plot=1
                rtt_plot_bds.RTTplot_high_res(path_dir_log_files,grid_plot,int(burnin),root_plot)
            elif plot_type==4:
                rtt_plot_bds = rtt_plot_bds.plot_marginal_rates(path_dir_log_files,name_tag=file_stem,bin_size=grid_plot,
                        burnin=burnin,min_age=args.min_age_plot,max_age=root_plot,logT=args.logT,n_reps=args.n_prior,min_allowed_t=min_allowed_t)
            elif plot_type== 5:
                rtt_plot_bds = rtt_plot_bds.RTTplot_Q(path_dir_log_files,args.qShift,burnin=burnin,max_age=root_plot)
            elif plot_type== 6:
                import pyrate_lib.bdnn_lib as bdnn_lib
                bdnn_lib.plot_rtt(path_dir_log_files, burn=burnin, translate=args.translate, min_age=args.min_age_plot, max_age=root_plot)
                if args.plotBDNN_groups != "":
                    bdnn_lib.plot_bdnn_rtt_groups(path_dir_log_files, args.plotBDNN_groups, burn=burnin,
                                                  translate=args.translate, min_age=args.min_age_plot, max_age=root_plot, bdnn_precision=args.BDNNprecision)
            elif plot_type== 7:
                import pyrate_lib.bdnn_lib as bdnn_lib
                bdnn_lib.get_PDRTT(path_dir_log_files, args.BDNN_groups,
                                   burn=burnin, thin=args.resample,
                                   groups_path=args.plotBDNN_groups,
                                   translate=args.translate, min_age=args.min_age_plot, max_age=root_plot,
                                   bdnn_precision=args.BDNNprecision, num_processes=args.thread[0], show_progressbar=True)

        elif plot_type == 8:
            import pyrate_lib.bdnn_lib as bdnn_lib
            mcmc_file = path_dir_log_files
            path_dir_log_files = path_dir_log_files.replace("_mcmc.log", "")
            pkl_file = path_dir_log_files + ".pkl"
            obj_effect_plot = bdnn_lib.get_effect_objects(mcmc_file, pkl_file,
                                                          burnin,
                                                          thin=args.resample,
                                                          combine_discr_features=args.BDNN_groups,
                                                          file_transf_features=args.plotBDNN_transf_features,
                                                          bdnn_precision=args.BDNNprecision,
                                                          min_age=args.min_age_plot,
                                                          max_age=root_plot,
                                                          translate=args.translate,
                                                          num_processes=args.thread[0],
                                                          show_progressbar=True)
            bdnn_obj, cond_trait_tbl_sp, cond_trait_tbl_ex, cond_trait_tbl_q, names_features_sp, names_features_ex, names_features_q, sp_rate_cond, ex_rate_cond, q_rate_cond, mean_tste, backscale_par = obj_effect_plot
            bdnn_lib.plot_effects(path_dir_log_files,
                                  cond_trait_tbl_sp,
                                  cond_trait_tbl_ex,
                                  cond_trait_tbl_q,
                                  sp_rate_cond,
                                  ex_rate_cond,
                                  q_rate_cond,
                                  bdnn_obj,
                                  mean_tste,
                                  backscale_par,
                                  names_features_sp,
                                  names_features_ex,
                                  names_features_q,
                                  suffix_pdf="PDP",
                                  translate=args.translate)

        else:
            #path_dir_log_files=sort(path_dir_log_files)
            # plot each file separately
            print(root_plot)
            if file_stem == "":
                path_dir_log_files = os.path.abspath(path_dir_log_files)
                direct="%s/*marginal_rates.log" % path_dir_log_files
                files=glob.glob(direct)
                files=sort(files)
                if len(files)==0:
                    if 2>1: #try:
                        name_file = os.path.splitext(os.path.basename(str(path_dir_log_files)))[0]
                        path_dir_log_files = os.path.dirname(str(path_dir_log_files))
                        name_file = name_file.split("marginal_rates")[0]
                        one_file= 1
                        plot_RTT(path_dir_log_files, burnin, name_file,one_file,root_plot,plot_type)
                    #except: sys.exit("\nFile or directory not recognized.\n")
                else:
                    for f in files:
                        name_file = os.path.splitext(os.path.basename(f))[0]
                        name_file = name_file.split("marginal_rates")[0]
                        one_file = 0
                        plot_RTT(path_dir_log_files, burnin, name_file,one_file,root_plot,plot_type)
            else:
                one_file = 0
                plot_RTT(path_dir_log_files, burnin, file_stem,one_file,root_plot,plot_type)
        quit()
    elif args.BDNN_pred_importance != "":
        import pyrate_lib.bdnn_lib as bdnn_lib
        path_dir_log_files = args.BDNN_pred_importance.replace("_mcmc.log", "")
        pkl_file = path_dir_log_files + ".pkl"
        mcmc_file = path_dir_log_files + "_mcmc.log"
        do_inter_imp = args.BDNN_pred_importance_interaction is False
        BDNNmodel = bdnn_lib.get_bdnn_model(pkl_file)
        sp_taxa_shap, ex_taxa_shap, q_taxa_shap, use_taxa_sp, use_taxa_ex = None, None, None, None, None
        sp_main_consrank, ex_main_consrank, q_main_consrank, sp_feat_missing, ex_feat_missing = None, None, None, None, None
        if BDNNmodel in [1, 3] and args.BDNN_nsim_expected_cv > 0:
            print("Getting expected coefficient of rate variation")
            bdnn_lib.get_coefficient_rate_variation(path_dir_log_files, burnin,
                                                    combine_discr_features=args.BDNN_groups,
                                                    num_sim=args.BDNN_nsim_expected_cv,
                                                    min_age=args.min_age_plot,
                                                    max_age=root_plot,
                                                    translate=args.translate,
                                                    num_processes=args.thread[0],
                                                    show_progressbar=True)
        if BDNNmodel in [2, 3] and args.BDNN_nsim_expected_cv > 0:
            print("Getting expected coefficient of sampling variation")
            bdnn_lib.get_coefficient_sampling_variation(path_dir_log_files, burnin,
                                                        combine_discr_features=args.BDNN_groups,
                                                        num_sim=args.BDNN_nsim_expected_cv,
                                                        num_processes=args.thread[0],
                                                        show_progressbar=True)
        if BDNNmodel in [1, 3]:
            print("Getting permutation importance birth-death")
            sp_featperm, ex_featperm = bdnn_lib.feature_permutation(mcmc_file, pkl_file,
                                                                    burnin,
                                                                    thin=args.resample,
                                                                    min_bs=args.BDNN_pred_importance_window_size[0],
                                                                    n_perm=args.BDNN_pred_importance_nperm,
                                                                    combine_discr_features=args.BDNN_groups,
                                                                    do_inter_imp=do_inter_imp,
                                                                    bdnn_precision=args.BDNNprecision,
                                                                    min_age=args.min_age_plot,
                                                                    max_age=root_plot,
                                                                    translate=args.translate,
                                                                    num_processes=args.thread[0], show_progressbar=True)
        if BDNNmodel in [2, 3]:
            print("Getting permutation importance sampling")
            q_featperm = bdnn_lib.feature_permutation_sampling(mcmc_file, pkl_file,
                                                               burnin,
                                                               thin=args.resample,
                                                               min_bs=args.BDNN_pred_importance_window_size[-1],
                                                               n_perm=args.BDNN_pred_importance_nperm,
                                                               combine_discr_features= args.BDNN_groups,
                                                               do_inter_imp=do_inter_imp,
                                                               bdnn_precision=args.BDNNprecision,
                                                               num_processes=args.thread[0],
                                                               show_progressbar=True)
        if BDNNmodel in [1, 3]:
            print("Getting SHAP values birth-death")
            sp_shap, ex_shap, sp_taxa_shap, ex_taxa_shap, use_taxa_sp, use_taxa_ex = bdnn_lib.k_add_kernel_shap(mcmc_file,
                                                                                                                pkl_file,
                                                                                                                burnin,
                                                                                                                thin=args.resample,
                                                                                                                combine_discr_features=args.BDNN_groups,
                                                                                                                do_inter_imp=do_inter_imp,
                                                                                                                use_mean=args.BDNN_mean_shap_per_group,
                                                                                                                bdnn_precision=args.BDNNprecision,
                                                                                                                min_age=args.min_age_plot,
                                                                                                                max_age=root_plot,
                                                                                                                translate=args.translate,
                                                                                                                num_processes=args.thread[0],
                                                                                                                show_progressbar=True)
        if BDNNmodel in [2, 3]:
            print("Getting SHAP values sampling")
            q_shap, q_taxa_shap = bdnn_lib.k_add_kernel_shap_sampling(mcmc_file, pkl_file,
                                                                      burnin,
                                                                      thin=args.resample,
                                                                      combine_discr_features=args.BDNN_groups,
                                                                      do_inter_imp=do_inter_imp,
                                                                      bdnn_precision=args.BDNNprecision,
                                                                      num_processes=args.thread[0],
                                                                      show_progressbar=True)
        obj_effect = bdnn_lib.get_effect_objects(mcmc_file, pkl_file,
                                                 burnin,
                                                 thin=args.resample,
                                                 combine_discr_features=args.BDNN_groups,
                                                 file_transf_features=args.plotBDNN_transf_features,
                                                 do_inter_imp=do_inter_imp,
                                                 bdnn_precision=args.BDNNprecision,
                                                 min_age=args.min_age_plot,
                                                 max_age=root_plot,
                                                 translate=args.translate,
                                                 num_processes=args.thread[0],
                                                 show_progressbar=True)
        bdnn_obj, cond_trait_tbl_sp, cond_trait_tbl_ex, cond_trait_tbl_q, names_features_sp, names_features_ex, names_features_q, sp_rate_part, ex_rate_part, q_rate_part, sp_fad_lad, backscale_par = obj_effect
        if BDNNmodel in [1, 3]:
            print("Getting marginal probabilities birth-death")
            sp_pv = bdnn_lib.get_prob_effects(cond_trait_tbl_sp, sp_rate_part, bdnn_obj, names_features_sp, rate_type='speciation')
            ex_pv = bdnn_lib.get_prob_effects(cond_trait_tbl_ex, ex_rate_part, bdnn_obj, names_features_ex, rate_type='extinction')
        if BDNNmodel in [2, 3]:
            print("Getting marginal probabilities sampling")
            q_pv = bdnn_lib.get_prob_effects(cond_trait_tbl_q, q_rate_part, bdnn_obj, names_features_q, rate_type='sampling')
        if BDNNmodel in [1, 3]:
            # consensus among 3 feature importance methods
            print("Getting consensus ranking birth-death")
            sp_feat_importance, sp_main_consrank, sp_feat_missing = bdnn_lib.get_consensus_ranking(sp_pv, sp_shap, sp_featperm)
            ex_feat_importance, ex_main_consrank, ex_feat_missing = bdnn_lib.get_consensus_ranking(ex_pv, ex_shap, ex_featperm)
            output_wd = os.path.dirname(os.path.realpath(path_dir_log_files))
            name_file = os.path.basename(path_dir_log_files)
            ex_feat_merged_file = os.path.join(output_wd, name_file + '_ex_predictor_influence.csv')
            ex_feat_importance.to_csv(ex_feat_merged_file, na_rep='NA', index=False)
            sp_feat_merged_file = os.path.join(output_wd, name_file + '_sp_predictor_influence.csv')
            sp_feat_importance.to_csv(sp_feat_merged_file, na_rep='NA', index=False)
            sp_taxa_shap = bdnn_lib.remove_feature_from_taxa_shaps(sp_taxa_shap, sp_feat_missing)
            sp_taxa_shap_file = os.path.join(output_wd, name_file + '_sp_shap_per_species.csv')
            sp_taxa_shap.to_csv(sp_taxa_shap_file, na_rep='NA', index=False)
            ex_taxa_shap = bdnn_lib.remove_feature_from_taxa_shaps(ex_taxa_shap, ex_feat_missing)
            ex_taxa_shap_file = os.path.join(output_wd, name_file + '_ex_shap_per_species.csv')
            ex_taxa_shap.to_csv(ex_taxa_shap_file, na_rep='NA', index=False)
        if BDNNmodel in [2, 3]:
            print("Getting consensus ranking sampling")
            q_feat_importance, q_main_consrank, _ = bdnn_lib.get_consensus_ranking(q_pv, q_shap, q_featperm)
            output_wd = os.path.dirname(os.path.realpath(path_dir_log_files))
            name_file = os.path.basename(path_dir_log_files)
            q_feat_merged_file = os.path.join(output_wd, name_file + '_q_predictor_influence.csv')
            q_feat_importance.to_csv(q_feat_merged_file, na_rep='NA', index=False)
            q_taxa_shap_file = os.path.join(output_wd, name_file + '_q_shap_per_species.csv')
            q_taxa_shap.to_csv(q_taxa_shap_file, na_rep='NA', index=False)
        # Plot contribution to species-specific rates
        bdnn_lib.dotplot_species_shap(mcmc_file, pkl_file, burnin, args.resample, output_wd, name_file,
                                      sp_taxa_shap, ex_taxa_shap, q_taxa_shap,
                                      sp_main_consrank, ex_main_consrank, q_main_consrank,
                                      combine_discr_features=args.BDNN_groups,
                                      file_transf_features=args.plotBDNN_transf_features,
                                      translate=args.translate,
                                      use_taxa_sp=use_taxa_sp, use_taxa_ex=use_taxa_ex,
                                      sp_feat_missing=sp_feat_missing, ex_feat_missing=ex_feat_missing)
        quit()
    elif args.BDNN_interaction != "":
        import pyrate_lib.bdnn_lib as bdnn_lib
        path_dir_log_files = args.BDNN_interaction.replace("_mcmc.log", "")
        pkl_file = path_dir_log_files + ".pkl"
        mcmc_file = path_dir_log_files + "_mcmc.log"
        bdnn_obj, w_sp, w_ex, _, sp_fad_lad, ts_post, te_post, t_reg_lam, t_reg_mu, _, reg_denom_lam, reg_denom_mu, _, _, _ = bdnn_lib.bdnn_parse_results(mcmc_file, pkl_file, burnin, args.resample)
        backscale_par = bdnn_lib.read_backscale_file(args.plotBDNN_transf_features)
        sp_inter, sp_trt_tbl, names_features = bdnn_lib.get_pdp_rate_free_combination(bdnn_obj, sp_fad_lad, ts_post, te_post,
                                                                                      w_sp, t_reg_lam, reg_denom_lam,
                                                                                      args.BDNN_groups,
                                                                                      backscale_par,
                                                                                      len_cont=100, rate_type="speciation",
                                                                                      fix_observed=args.BDNN_interaction_fix,
                                                                                      min_age=args.min_age_plot,
                                                                                      max_age=root_plot,
                                                                                      translate=args.translate,
                                                                                      bdnn_precision=args.BDNNprecision,
                                                                                      num_processes=args.thread[0],
                                                                                      show_progressbar=True)
        ex_inter, ex_trt_tbl, names_features = bdnn_lib.get_pdp_rate_free_combination(bdnn_obj, sp_fad_lad, ts_post, te_post,
                                                                                      w_ex, t_reg_mu, reg_denom_mu,
                                                                                      args.BDNN_groups,
                                                                                      backscale_par,
                                                                                      len_cont=100, rate_type="extinction",
                                                                                      fix_observed=args.BDNN_interaction_fix,
                                                                                      min_age=args.min_age_plot,
                                                                                      max_age=root_plot,
                                                                                      translate=args.translate,
                                                                                      bdnn_precision=args.BDNNprecision,
                                                                                      num_processes=args.thread[0],
                                                                                      show_progressbar=True)
        output_wd = os.path.dirname(os.path.realpath(path_dir_log_files))
        name_file = '_'.join(names_features)
        sp_trt_tbl_file = os.path.join(output_wd, name_file + '_at_speciation.csv')
        sp_trt_tbl.to_csv(sp_trt_tbl_file, na_rep = 'NA', index = True)
        sp_inter_file = os.path.join(output_wd, name_file + '_speciation_pdp.csv')
        sp_inter.to_csv(sp_inter_file, na_rep = 'NA', index = False)
        ex_trt_tbl_file = os.path.join(output_wd, name_file + '_at_extinction.csv')
        ex_trt_tbl.to_csv(ex_trt_tbl_file, na_rep = 'NA', index = True)
        ex_inter_file = os.path.join(output_wd, name_file + '_extinction_pdp.csv')
        ex_inter.to_csv(ex_inter_file, na_rep = 'NA', index = False)
        quit()
    elif args.mProb != "": calc_model_probabilities(args.mProb,burnin)
    elif len(list_files_BF):
        print(list_files_BF[0])
        if len(list_files_BF)==1: calc_BFlist(list_files_BF[0])
        else: calc_BF(list_files_BF[0],list_files_BF[1])
            #
        #    sys.exit("\n2 '*marginal_likelihood.txt' files required.\n")
        quit()
    elif args.combLog != "": # COMBINE LOG FILES
        comb_log_files(args.combLog,burnin,args.tag,resample=args.resample,col_tag=args.col_tag)
        sys.exit("\n")
    elif args.combLogRJ != "": # COMBINE LOG FILES
        comb_log_files_smart(args.combLogRJ,burnin,args.tag,resample=args.resample,col_tag=args.col_tag)
        sys.exit("\n")
    elif args.combBDNN != "":
        from pyrate_lib.bdnn_lib import combine_pkl
        tag = args.tag
        combine_pkl(args.combBDNN, tag, burnin, args.resample)
        comb_log_files_smart(args.combBDNN, burnin, tag, resample=args.resample, col_tag=args.col_tag, keep_q=True)
        sys.exit("\n")
    elif len(args.input_data)==0 and args.d == "": sys.exit("\nInput file required. Use '-h' for command list.\n")

    use_se_tbl = 0
    if args.d != "":
        use_se_tbl = 1
        se_tbl_file  = args.d

    if len(args.SE_stats)>0:
        if use_se_tbl == 0: sys.exit("\nProvide an SE table using command -d\n")
        #if len(args.SE_stats)<1: sys.exit("\nExtinction rate at the present\n")
        #else:
        try: EXT_RATE  = float(args.SE_stats[0])
        except: EXT_RATE = 0
        if EXT_RATE==0: print("\nExtinction rate set to 0: using estimator instead.\n")
        if len(args.SE_stats)>1: step_size = args.SE_stats[1]
        else: step_size = 1
        if len(args.SE_stats)>2: no_sim_ex_time = args.SE_stats[2]
        else: no_sim_ex_time = 100
        plot_tste_stats(se_tbl_file, EXT_RATE, step_size,no_sim_ex_time,burnin,args.rescale)
        quit()

    if args.ltt>0:
        grid_plot = args.grid_plot
        if grid_plot==0: grid_plot=0.1
        plot_ltt(se_tbl_file,plot_type=args.ltt,rescale=args.rescale, step_size=grid_plot)

    twotraitBD = 0
    if args.twotrait == 1:
        twotraitBD = 1

    ############################ LOAD INPUT DATA ############################
    match_taxa_trait = 0
    use_partial_BD = args.partialBD
    if use_se_tbl==0:
        input_file_raw = os.path.basename(args.input_data[0])
        input_file = os.path.splitext(input_file_raw)[0]  # file name without extension

        if args.wd=="":
            output_wd = os.path.dirname(args.input_data[0])
            if output_wd=="": output_wd= get_self_path()
        else: output_wd=args.wd

        print("\n",input_file, args.input_data, "\n")
        try: 
            test_spec = importlib.util.spec_from_file_location(input_file,args.input_data[0])
            input_data_module = importlib.util.module_from_spec(test_spec)
            test_spec.loader.exec_module(input_data_module)
        except(IOError): sys.exit("\nInput file required. Use '-h' for command list.\n")

        j=np.maximum(args.j-1,0)
        try: fossil_complete=input_data_module.get_data(j)
        except(IndexError):
            fossil_complete=input_data_module.get_data(0)
            print(("Warning: data set number %s not found. Using the first data set instead." % (args.j)))
            j=0

        if args.filter_taxa != "":
            list_included_taxa = [line.rstrip() for line in open(args.filter_taxa)]
            taxa_names_temp = input_data_module.get_taxa_names()
            print("Included taxa:")
        
        if args.drop_zero > 0:
            rns_nms = np.random.random(len(fossil_complete))
            count = 0
            for i in range(len(fossil_complete)):
                if rns_nms[i] < args.drop_zero:
                    tmp = fossil_complete[i] + 0
                    fossil_complete[i] = tmp[tmp > 0]
                    count += 1    
            print("Removed extant taxa:", count)
        fossil=list()
        have_record=list()
        singletons_excluded = list()
        taxa_included = list()
        
        if args.drop_internal:
            foss = [np.array([np.min(f), np.max(f)]) for f in fossil_complete if len(f) > 1]
            fossil_complete = foss
        
        for i in range(len(fossil_complete)):
            if len(fossil_complete[i])==1 and fossil_complete[i][0]==0: pass # exclude taxa with no fossils

            elif max(fossil_complete[i]) > max(args.filter) or min(fossil_complete[i]) < min(args.filter):
                print("excluded taxon with age range:",round(max(fossil_complete[i]),3), round(min(fossil_complete[i]),3))

            elif args.singleton == -1: # exclude extant taxa (if twotraitBD == 1: extant (re)moved later)
                if min(fossil_complete[i])==0 and twotraitBD == 0: singletons_excluded.append(i)
                else:
                    have_record.append(i)
                    fossil.append(fossil_complete[i]*args.rescale+args.translate)
                    taxa_included.append(i)

            elif args.translate < 0:
                # exclude recent taxa after 'translating' records towards zero
                if max(fossil_complete[i]*args.rescale+args.translate)<=0:
                    singletons_excluded.append(i)
                else:
                    if len(fossil_complete[i]) <= args.singleton and np.random.random() >= args.frac_sampled_singleton:
                        singletons_excluded.append(i)
                    else:
                        if args.filter_taxa != "": # keep only taxa within list
                            if taxa_names_temp[i] in list_included_taxa:
                                have_record.append(i)
                                fossil_occ_temp = fossil_complete[i]*args.rescale+args.translate
                                fossil_occ_temp[fossil_occ_temp<0] = 0.0
                                fossil.append(np.unique(fossil_occ_temp[fossil_occ_temp>=0]))
                                taxa_included.append(i)
                                print(taxa_names_temp[i])
                            else:
                                singletons_excluded.append(i)
                        else:
                            have_record.append(i)
                            fossil_occ_temp = fossil_complete[i]*args.rescale+args.translate
                            fossil_occ_temp[fossil_occ_temp<0] = 0.0
                            fossil.append(np.unique(fossil_occ_temp[fossil_occ_temp>=0]))
                            taxa_included.append(i)


            elif args.singleton > 0: # min number of occurrences
                if len(fossil_complete[i]) <= args.singleton and np.random.random() >= args.frac_sampled_singleton:
                    singletons_excluded.append(i)
                else:
                    have_record.append(i) # some (extant) species may have trait value but no fossil record
                    fossil.append(fossil_complete[i]*args.rescale+args.translate)
                    taxa_included.append(i)
            elif args.filter_taxa != "": # keep only taxa within list
                if taxa_names_temp[i] in list_included_taxa:
                    have_record.append(i) # some (extant) species may have trait value but no fossil record
                    fossil.append(fossil_complete[i]*args.rescale+args.translate)
                    taxa_included.append(i)
                    print(taxa_names_temp[i])
                else: singletons_excluded.append(i)
            else:
                have_record.append(i) # some (extant) species may have trait value but no fossil record
                fossil.append(fossil_complete[i]*args.rescale+args.translate)
                taxa_included.append(i)
        if len(singletons_excluded)>0 and args.data_info == 0: print("The analysis includes %s species (%s were excluded)" % (len(fossil),len(singletons_excluded)))
        else: print("\nThe analysis includes %s species (%s were excluded)" % (len(fossil),len(fossil_complete)-len(fossil)))
        out_name=input_data_module.get_out_name(j) +args.out

        try:
            taxa_names=input_data_module.get_taxa_names()
            match_taxa_trait = 1
        except(AttributeError):
            taxa_names=list()
            for i in range(len(fossil)): taxa_names.append("taxon_%s" % (i))

#        print('singletons_excluded\n', singletons_excluded)
        taxa_included = np.array(taxa_included)
        taxa_names = np.array(taxa_names)
        taxa_names = taxa_names[taxa_included]

        FA,LO,N=np.zeros(len(fossil)),np.zeros(len(fossil)),np.zeros(len(fossil))
        array_all_fossils = []
        for i in range(len(fossil)):
            FA[i]=max(fossil[i])
            LO[i]=min(fossil[i])
            N[i]=len(fossil[i])
            array_all_fossils = array_all_fossils + list(fossil[i])
        array_all_fossils = np.array(array_all_fossils)

        # constrain origination and extinction to a given age
        bound_ts = np.full(len(taxa_names), np.inf)
        bound_te = np.zeros(len(taxa_names))
        if args.bound_se != '':
            bound_se = np.genfromtxt(args.bound_se, dtype=str, skip_header=1)
            bound_ts, bound_te = set_bound_se(bound_ts, bound_te, bound_se, taxa_names)


    # """
    # use_se_trait_id_tbl
    # """
    elif use_partial_BD:
        # use_se_trait_id_tbl_file = "/Users/dsilvestro/Software/PyRate/example_files/lithology_example.txt"
        t_file=np.loadtxt(se_tbl_file, skiprows=1)
        FA=t_file[:,1]*args.rescale+args.translate
        LO=t_file[:,2]*args.rescale+args.translate
        fix_SE= 1
        fixed_ts, fixed_te=FA, LO
        output_wd = os.path.dirname(se_tbl_file)
        if output_wd=="": output_wd= get_self_path()
        out_name="%s_%s"  % (os.path.splitext(os.path.basename(se_tbl_file))[0],args.out)
        # speciations excluding other lithologies and beginning and end of cores
        # lith in focus: t_file[:,5] = 1
        # t_file[:,3] == 0 when start of the core
        # t_file[:,4] == 0 when end of the core
        SP_in_window = (t_file[:,5] * t_file[:,3]==1).nonzero()[0]     
        EX_in_window = (t_file[:,5] * t_file[:,4]==1).nonzero()[0] 
        # extinctions excluding other lithologies and beginning and end of cores
        SP_not_in_window = (t_file[:,5] * t_file[:,3]==0).nonzero()[0]
        EX_not_in_window = (t_file[:,5] * t_file[:,4]==0).nonzero()[0]
        ID_focal_lithology = np.where(t_file[:,5] > 0)[0]
        ID_other_lithology = np.where(t_file[:,5] == 0)[0]
        BPD_partial_lik = BD_partial_lik_lithology
        PoiD_const = 0
        use_se_tbl = 1
        hasFoundPyRateC= 0
        print(SP_not_in_window, t_file.shape)
        print(EX_not_in_window)
        
    else:
        print(se_tbl_file)
        # allow spelled out species names instead of just species index
        from pyrate_lib.lib_utilities import read_ts_te_table as read_ts_te_table
        j = np.maximum(args.j-1,0)
        focus_clade = args.clade
        FA, LO, _, taxa_names, _ = read_ts_te_table(se_tbl_file, j, args.rescale, args.translate, focus_clade)
        
        print(j, len(FA), "species")
        fix_SE= 1
        fixed_ts, fixed_te=FA, LO
        output_wd = os.path.dirname(se_tbl_file)
        if output_wd=="": output_wd= get_self_path()
        out_name="%s_%s_%s"  % (os.path.splitext(os.path.basename(se_tbl_file))[0],j,args.out)
        if focus_clade>=0: out_name+= "_c%s" % (focus_clade)

    float_prec_f = get_float_prec_f(args.BDNNprecision)
    if args.restore_mcmc != "":
        restore_init_values = get_init_values(args.restore_mcmc, taxa_names, float_prec_f)
        restore_chain = 1
    else:
        restore_chain = 0


    if args.se_gibbs: 
        if argsHPP == 1:
            times_q_shift = np.array([np.max(FA), 0])


    ###### SET UP BD MODEL WITH STARTING NUMBER OF LINEAGES > 1
    no_starting_lineages = args.initDiv
    max_age_fixed_ts = np.max(FA)

    if no_starting_lineages>0:
        if use_se_tbl==0:
            sys.exit("Starting lineages > 1 only allowed with -d option")
        if model_cov>=1:
            sys.exit("Starting lineages > 1 only allowed with -d option")
        #print sort(fixed_ts)
        fixed_ts_ordered = np.sort(fixed_ts+0.)[::-1]
        fixed_ts_ordered_not_speciation = fixed_ts_ordered[0:no_starting_lineages]
        ind = np.array([i for i in range(len(fixed_ts)) if fixed_ts[i] in fixed_ts_ordered_not_speciation])
        fixed_ts[ind] = max_age_fixed_ts
        #print sort(fixed_ts)

    ################

    if argsG == 1: out_name += "_G"
    if args.se_gibbs: out_name += "_seGibbs"
    
    add_to_bdnnblock_mask = 1
    bdnn_const_baseline = args.BDNNconstbaseline
    out_act_f = get_act_f(args.BDNNoutputfun)
    out_act_f_q = get_act_f(args.BDNNoutputfun)
    hidden_act_f = get_hidden_act_f(args.BDNNactfun)
    block_nn_model = args.BDNNblockmodel
    nn_activate_all = args.BDNNactivate_all
    bdnn_timevar = args.BDNNtimevar
    bdnn_timevar_q = args.BDNNtimevar_q
    bdnn_dd = args.BDNNdd
    div_idx_trt_tbl = -1
    if bdnn_dd and use_time_as_trait:
        div_idx_trt_tbl = -2
    if bdnn_dd:
        add_to_bdnnblock_mask += 1
    bias_node_idx = [0]
    num_bias_node = 1

    ############################ SET BIRTH-DEATH MODEL ############################

    # Number of extant taxa (user specified)
    if args.N > -1: tot_extant=args.N
    else: tot_extant = -1

    n_taxa = len(FA)
    apply_reg = np.full(n_taxa, True)
    if len(fixed_times_of_shift_bdnn) > 0:
        if not fix_edgeShift in [1, 2]: # both or max boundary
            # Respect user definition of edges. Do not exclude shifts even if there are no occurrences in the earliest bin.
            fixed_times_of_shift_bdnn = fixed_times_of_shift_bdnn[fixed_times_of_shift_bdnn < np.max(FA)]
        time_framesL_bdnn = len(fixed_times_of_shift_bdnn) + 1
        time_framesM_bdnn = len(fixed_times_of_shift_bdnn) + 1
        apply_reg = np.full((n_taxa, time_framesL_bdnn), True)
        if fix_edgeShift in [1, 2]: # both or max boundary
            apply_reg[:, 0] = False
            bias_node_idx.append(bias_node_idx[-1] + 1)
            num_bias_node += 1
        if fix_edgeShift in [1, 3]: # both or min boundary
            apply_reg[:, -1] = False
            bias_node_idx.append(bias_node_idx[-1] + 1)
            num_bias_node += 1

    if len(fixed_times_of_shift)>0:
        fixed_times_of_shift=fixed_times_of_shift[fixed_times_of_shift<np.max(FA)]
        # fixed number of dpp bins
        if args.dpp_nB>0:
            t_bin_set = np.linspace(0,np.max(FA),args.dpp_nB+1)[::-1]
            fixed_times_of_shift = t_bin_set[1:len(t_bin_set)-1]
        time_framesL=len(fixed_times_of_shift)+1
        time_framesM=len(fixed_times_of_shift)+1
        # estimate DPP hyperprior
        hp_gamma_rate  = get_rate_HP(time_framesL,target_k,hp_gamma_shape)



    if args.fixSE != "" or use_se_tbl==1:          # fix TS, TE
        if use_se_tbl==1: pass
        else:
            fix_SE=1
            fixed_ts, fixed_te= calc_ts_te(args.fixSE, burnin=args.b)
    else: fix_SE=0

    if args.discrete == 1: useDiscreteTraitModel = 1
    else: useDiscreteTraitModel = 0

    if args.bound[0] != np.inf or args.bound[1] != 0:
        useBounded_BD = 1
    boundMax = max(args.bound) # if not specified it is set to Inf
    boundMin = min(args.bound) # if not specified it is set to 0

    # Get trait values (COVAR and DISCRETE models)
    if model_cov>=1 or useDiscreteTraitModel == 1 or useBounded_BD == 1:
        if 2>1: #try:
            if args.trait_file != "": # Use trait file
                traitfile=open(args.trait_file, 'r')

                L=traitfile.readlines()
                head= L[0].split()

                if useBounded_BD == 1: # columns: taxon_name, SP, EX (SP==1 if speciate in window)
                    trait_val=[l.split()[1:3] for l in L][1:]
                else:
                    if len(head)==2: col=1
                    elif len(head)==3: col=2 ####  CHECK HERE: WITH BOUNDED-BD 3 columns but two traits!
                    else: sys.exit("\nNo trait data found!")
                    trait_val=[l.split()[col] for l in L][1:]

                if useBounded_BD == 1:
                    trait_values = np.array(trait_val)
                    trait_values = trait_values.astype(float)
                else:
                    trait_values = np.zeros(len(trait_val))
                    trait_count = 0
                    for i in range(len(trait_val)):
                        try:
                            trait_values[i] = float(trait_val[i])
                            trait_count+=1
                        except:
                            trait_values[i] = np.nan
                    if trait_count==0: sys.exit("\nNo trait data found.")

                if match_taxa_trait == 1:
                    trait_taxa=np.array([l.split()[0] for l in L][1:])
                    #print taxa_names
                    #print sort(trait_taxa)
                    matched_trait_values = []
                    for i in range(len(taxa_names)):
                        taxa_name=taxa_names[i]
                        matched_val = trait_values[trait_taxa==taxa_name]
                        if len(matched_val)>0:
                            matched_trait_values.append(matched_val[0])
                            print("matched taxon: %s\t%s\t%s" % (taxa_name, matched_val[0], max(fossil[i])-min(fossil[i])))
                        else:
                            if useBounded_BD == 1: # taxa not specified originate/go_extinct in window
                                matched_trait_values.append([1,1])
                            else:
                                matched_trait_values.append(np.nan)
                            #print taxa_name, "did not have data"
                    trait_values= np.array(matched_trait_values)
                    #print trait_values

            else:             # Trait data from .py file
                trait_values=input_data_module.get_continuous(np.maximum(args.trait-1,0))
            #
            if twotraitBD == 1:
                trait_values=input_data_module.get_continuous(0)
                discrete_trait=input_data_module.get_continuous(1)
                discrete_trait=discrete_trait[taxa_included]

                #print discrete_trait
                #print len(np.isfinite(discrete_trait).nonzero()[0])
                if args.singleton == -1:
                    ind_extant_sp =(LO==0).nonzero()[0]
                    print("Treating %s extant taxa as missing discrete data" % (len(ind_extant_sp)))
                    #temp_trait = discrete_trait+0 #np.zeros(len(discrete_trait))*discrete_trait
                    discrete_trait[ind_extant_sp]=np.nan
                    #discrete_trait = temp_trait
                    #print len(np.isfinite(discrete_trait).nonzero()[0])
                    #print discrete_trait

            if model_cov>=1:
                if args.logT==0: pass
                elif args.logT==1: trait_values = log(trait_values)
                else: trait_values = np.log10(trait_values)
        #except: sys.exit("\nTrait data not found! Check input file.\n")

        if model_cov>=1:
            # Mid point age of each lineage
            if use_se_tbl==1: MidPoints = (fixed_ts+fixed_te)/2.
            else:
                MidPoints=np.zeros(len(fossil_complete))
                for i in range(len(fossil_complete)):
                    MidPoints[i]=np.mean([max(fossil_complete[i]),min(fossil_complete[i])])

            try: 
                MidPoints = MidPoints[taxa_included]
            except: 
                print("\nAssuming taxa in the same order in the tste file and in the trait file!\n")
                taxa_names = ["taxon_%s" % i for i in range(len(trait_values))]
                have_record = np.ones(len(trait_values))
            # fit linear regression (for species with trait value - even if without fossil data)
            print(len(trait_values), len(np.isfinite(trait_values)), len(taxa_names), len(MidPoints), len(have_record))
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(MidPoints[np.isfinite(trait_values)],trait_values[np.isfinite(trait_values)])

            #
            ind_nan_trait= (np.isfinite(trait_values)== 0).nonzero()
            meanGAUScomplete=np.zeros(len(MidPoints))
            meanGAUScomplete[ind_nan_trait] = slope*MidPoints[ind_nan_trait] + intercept

            if use_se_tbl==1 or args.trait_file != "": meanGAUS= meanGAUScomplete
            else:
                trait_values= trait_values[np.array(have_record)]
                meanGAUS= meanGAUScomplete[np.array(have_record)]

            sdGAUS = std_err
            regression_trait= "\n\nEstimated linear trend trait-value: \nslope=%s; sd. error= %s (intercept= %s; R2= %s; P-value= %s\nTrait data for %s of %s taxa)" \
            % (round(slope,2), round(std_err,2), round(intercept,2), round(r_value,2), round(p_value,2), len(trait_values)-len(ind_nan_trait[0]), len(trait_values))
            print(regression_trait)

            #print trait_values
            parGAUS=scipy.stats.norm.fit(trait_values[np.isfinite(trait_values)]) # fit normal distribution
            #global con_trait
            con_trait=seed_missing(trait_values,meanGAUS,sdGAUS) # fill the gaps (missing data)
            #print con_trait
            if est_COVAR_prior == 1: out_name += "_COVhp"
            else: out_name += "_COV"

        if useDiscreteTraitModel == 1:
            if twotraitBD == 0:
                discrete_trait = trait_values
            ind_nan_trait= (np.isfinite(discrete_trait)== 0).nonzero()
            regression_trait= "\n\nDiscrete trait data for %s of %s taxa" \
            % (len(trait_values)-len(ind_nan_trait[0]), len(trait_values))
            print(regression_trait)
        else: print("\n")


    if useDiscreteTraitModel == 1:
        ind_trait_species = discrete_trait
        print(ind_trait_species)
        ind_trait_species[np.isnan(ind_trait_species)]=np.nanmax(ind_trait_species)+1
        print(ind_trait_species)
        ind_trait_species = ind_trait_species.astype(int)
        len_trait_values = len(np.unique(discrete_trait))
        lengths_B_events=[]
        lengths_D_events=[]
        for i in range(len_trait_values):
            lengths_B_events.append(len(discrete_trait[discrete_trait==i]))
            lo_temp = LO[discrete_trait==i]
            lengths_D_events.append(len(lo_temp[lo_temp>0]))
        lengths_B_events = np.array([sum(lengths_B_events)]) # ASSUME CONST BIRTH RATE
        #lengths_B_events = np.array(lengths_D_events)
        lengths_D_events = np.array(lengths_D_events)
        #ind_trait_species = ind_trait_species-ind_trait_species
        print(lengths_B_events, lengths_D_events)
        obs_S = [sum(FA[ind_trait_species==i]-LO[ind_trait_species==i]) for i in range(len(lengths_B_events))]
        print(obs_S)
        TDI = 0

    use_poiD=args.mPoiD
    PoiD_const = 0
    if use_poiD == 1:
        BPD_partial_lik = PoiD_partial_lik
        PoiD_const = - (sum(log(np.arange(1,len(FA)+1))))
    elif useBounded_BD == 1:
        BPD_partial_lik = BD_partial_lik_bounded
        SP_in_window = (trait_values[:,0]==1).nonzero()[0]
        EX_in_window = (trait_values[:,1]==1).nonzero()[0]
        SP_not_in_window = (trait_values[:,0]==0).nonzero()[0]
        EX_not_in_window = (trait_values[:,1]==0).nonzero()[0]
        #print SP_in_window, EX_in_window
        ###### NEXT: change function update_ts_te() so that only SP/EX_in_win are updated
        # make sure the starting points are set to win boundaries for the other species and
        # within boundaries for SP/EX_in_win
        argsHPP = 1 # only HPP can be used with bounded BD
    elif use_partial_BD:
        pass
    else:
        BPD_partial_lik = BD_partial_lik
        SP_in_window = np.arange(len(FA)) # all ts/te can be updated
        EX_in_window = np.arange(len(LO))
        SP_not_in_window = []
        EX_not_in_window = []

    if args.mDeath == 1: use_Death_model = 1
    else: use_Death_model= 0

    if args.mBirth >= 0:
        use_Birth_model = 1
        init_M_rate = args.mBirth
    else:
        use_Birth_model = 0


    # USE BDI subMODELS
    model_BDI=args.mBDI
    if model_BDI >=0:
        try: ts,te = fixed_ts, fixed_te
        except: sys.exit("\nYou must use options -fixSE or -d to run BDI submodels.")
        z=np.zeros(len(te))+2
        z[te==0] = 3
        te_orig = te+0.
        te= te[te>0]  # ignore extant
        z = z[z==2]   # ignore extant
        all_events_temp= np.array([np.concatenate((ts,te),axis=0),np.concatenate((np.zeros(len(ts))+1,z),axis=0)])
        idx = np.argsort(all_events_temp[0])[::-1] # get indexes of sorted events
        all_events_array=all_events_temp[:,idx] # sort by time of event
        #print all_events_array
        all_events = all_events_array[0,:]
        dT_events= -(np.diff(np.append(all_events,0)))

        #div_trajectory =get_DT(np.append(all_events,0),ts,te_orig)
        #div_trajectory =div_trajectory[1:]
        div_traj = np.zeros(len(ts)+len(te))
        current_div,j = 0,0
        for i in all_events_array[1]:
            if i == 1: current_div+=1
            if i == 2: current_div-=1
            div_traj[j] = current_div
            j+=1

        #j=0
        #for i in all_events_array[0]:
        #    print i, "\t", div_trajectory[j],  "\t", div_traj[j], "\t",dT_events[j]
        #    j+=1
        div_trajectory=div_traj
        #print div_traj
        BPD_partial_lik = BDI_partial_lik
        if model_BDI==0: out_name += "BD"
        if model_BDI==1: out_name += "ID"
        if TDI<2: out_name = "%s%s%s" % (out_name,time_framesL,time_framesM)


    # SET UO AGE DEP. EXTINCTION MODEL
    use_ADE_model = 0
    if args.ADE == 1:
        use_ADE_model = 1
        BPD_partial_lik = BD_age_partial_lik
        out_name += "_ADE"
        argsHPP = 1

    if args.ADE == 2:
        use_ADE_model = 2
        BPD_partial_lik = BD_age_partial_lik
        out_name += "_CorrBD"
        argsHPP = 1

    est_hyperP = 0
    use_cauchy = 0
    fix_hyperP = 0
    if sum(args.cauchy) >= 0:
        hypP_par = np.ones(2)
        use_cauchy = 1
        est_hyperP = 1
        if sum(args.cauchy) > 0:
            fix_hyperP = 1
            hypP_par = np.array(args.cauchy) # scale of Cauchy distribution
    else:
        hypP_par = np.array([L_lam_m,M_lam_m]) # rate of Gamma distribution
        if min([L_lam_m,M_lam_m])==0:
            est_hyperP = 1
            hypP_par = np.ones(2)

    if use_ADE_model >= 1:
        hypP_par[1]=0.1
        tot_extant = -1
        d_hyperprior[0]=1 # first hyper-prior on sp.rates is not used under ADE, thus not updated (multiplier update =1)

    qFilter=args.qFilter # if set to zero all times of shifts (and preservation rates) are kept, even if they don't have occurrences
    if args.qShift != "":
        try: times_q_shift=np.sort(np.loadtxt(args.qShift))[::-1]*args.rescale + args.translate
        except: times_q_shift=np.array([np.loadtxt(args.qShift)])*args.rescale + args.translate
        # filter qShift times based on observed time frame
        if qFilter == 1:
            times_q_shift=times_q_shift[times_q_shift<np.max(FA)]
            times_q_shift=list(times_q_shift[times_q_shift>min(LO)])
        else: # q outside observed range (sampled from the prior)
            times_q_shift = list(times_q_shift)
        time_framesQ=len(times_q_shift)+1
        occs_sp_bin =list()
        temp_times_q_shift = np.array(list(times_q_shift)+[np.max(FA)+1]+[0])
        for i in range(len(fossil)):
            occs_temp = fossil[i]
            h = np.histogram(occs_temp[occs_temp>0],bins=sort( temp_times_q_shift ))[0][::-1]
            occs_sp_bin.append(h)
        argsHPP = 1
        TPP_model = 1
        print(times_q_shift, np.max(FA), np.min(LO))
    else:
        TPP_model = 0

    bdnn_ads = args.BDNNads
    if BDNNmodel in [2, 3]:
        args_qShift = args.qShift
        highres_q_repeats = None
        if bdnn_ads >= 0.0 or (args_qShift != '' and bdnn_timevar_q != ''):
            min_bin_size = bdnn_time_res
            if bdnn_ads >= 0.0:
                min_bin_size = np.minimum(bdnn_ads, bdnn_time_res)
            highres_q_repeats, times_q_shift = get_highres_repeats(args_qShift, min_bin_size, np.max(FA))
        argsHPP, occs_sp, log_factorial_occs, duration_q_bins, occs_single_bin, q_time_frames_bdnn, use_HPP_NN_lik, TPP_model, const_q = precompute_fossil_features(args_qShift, bdnn_timevar_q, bdnn_ads, bdnn_time_res)
        singleton_mask = make_singleton_mask(occs_sp, bdnn_timevar_q, bdnn_ads)
        apply_reg_q = np.full_like(singleton_mask, True)
        if (((use_HPP_NN_lik and bdnn_ads <= 0.0) or (not use_HPP_NN_lik and bdnn_time_res < 1.0) or (bdnn_timevar_q != '' and bdnn_time_res == 1)) and not (bdnn_timevar_q != '' and TPP_model == 0)):
            highres_q_repeats_rtt, times_q_shift_rtt = get_highres_repeats(args_qShift, bdnn_time_res, np.max(FA))
            times_q_shift_rtt = np.concatenate((np.inf, times_q_shift_rtt, np.zeros(1)), axis=None)
        elif (bdnn_ads > 0.0 and bdnn_time_res < bdnn_ads):
            highres_q_repeats_rtt, times_q_shift_rtt = get_highres_repeats(args_qShift, bdnn_time_res, np.max(FA), q_time_frames_bdnn[::-1])
            times_q_shift_rtt = np.concatenate((np.inf, times_q_shift_rtt, np.zeros(1)), axis=None)
        else:
            times_q_shift_rtt = q_time_frames_bdnn + 0.0

    if fix_Shift == 1 and use_ADE_model == 0: 
        est_hyperP = 1
    if (args.BDNNtimetrait != 0 or bdnn_timevar[0] or bdnn_dd or bdnn_loaded_tbls_timevar) and args.BDNNmodel in [1, 2, 3] and bdnn_const_baseline:
        est_hyperP = 0
    # define hyper-prior function for BD rates
    if tot_extant==-1 or TDI ==3 or use_poiD == 1:
        if use_ADE_model == 0 and fix_Shift == 1 and TDI < 3 and use_cauchy == 1:
            if est_hyperP == 0 or fix_hyperP == 1:
                prior_setting= "Using Cauchy priors on the birth-death rates (C_l[0,%s],C_l[0,%s]).\n" % (hypP_par[0],hypP_par[1])
            else:
                    prior_setting= "Using Cauchy priors on the birth-death rates (C_l[0,est],C_l[0,est]).\n"
            get_hyper_priorBD = HPBD1 # cauchy with hyper-priors
        else:
            if est_hyperP == 0:
                if BDNNmodel in [1, 3]:
                    prior_setting= ""
                else:
                    prior_setting= "Using Gamma priors on the birth-death rates (G_l[%s,%s], G_m[%s,%s]).\n" % (L_lam_r,hypP_par[0],M_lam_r,hypP_par[1])
            else:
                prior_setting= "Using Gamma priors on the birth-death rates (G_l[%s,est], G_m[%s,est]).\n" % (L_lam_r,M_lam_r)
            get_hyper_priorBD = HPBD2 # gamma
    else:
        prior_setting= "Priors on the birth-death rates based on extant diversity (N = %s).\n" % (tot_extant)
        get_hyper_priorBD = HPBD3 # based on no. extant
    print(prior_setting)

    if use_poiD == 1:
        if model_cov>=1:
            print("PoiD not available with trait correlation. Using BD instead.")
            BPD_partial_lik = BD_partial_lik
            PoiD_const = 0
        if fix_SE== 0:
            print("PoiD not available with SE estimation. Using BD instead.")
            BPD_partial_lik = BD_partial_lik
            PoiD_const = 0
        if hasFoundPyRateC:
            print("PoiD not available using FastPyRateC library. Using Python version instead.")
            hasFoundPyRateC = 0
    
    if BDNNmodel:
        # model_cov = 6
        f_cov_par = [0.4, 0.5,0.9,1]
        f_update_se = args.BDNNupdate_se_f[0]
        if len(args.BDNNupdate_f) == 1:
            bdnn_update_f = args.BDNNupdate_f * (len(args.BDNNnodes) + 1)
            print("bdnn_update_f", bdnn_update_f)
        elif len(args.BDNNupdate_f) != (len(args.BDNNnodes) + 1):
            sys.exit("BDNNupdate_f flag should be called with 1 or %s values for a model with %s layers " % (len(args.BDNNnodes) + 1, len(args.BDNNnodes)))
        else:
            bdnn_update_f = args.BDNNupdate_f
    
        if bdnn_const_baseline:
            [f_update_q,f_update_lm,f_update_cov]=f_update_se+ np.array([0.1, 0,1-f_update_se])
            if BDNNmodel in [2]:
                [f_update_q, f_update_lm, f_update_cov] = f_update_se + np.array([0.1, 0.2, 1 - f_update_se])
        else:
            # print([f_update_q,f_update_lm,f_update_cov])
            [f_update_q,f_update_lm,f_update_cov]=f_update_se+ np.array([0.1, 0.15,1-f_update_se])

        n_BDNN_nodes = args.BDNNnodes
        prior_lam_t_reg = args.BDNNreg
        independ_reg = True
        if len(prior_lam_t_reg) == 1:
            independ_reg = False
            if BDNNmodel == 1:
                prior_lam_t_reg = 2 * prior_lam_t_reg
            elif BDNNmodel == 3:
                prior_lam_t_reg = 3 * prior_lam_t_reg
    
        # load trait data
        names_traits = []
        if args.trait_file != "":
            traitfile=open(args.trait_file, 'r')

            L=traitfile.readlines()
            head= L[0].split()
            names_traits = head[1:]
            names_traits = [names_traits[i].replace('"', '') for i in range(len(names_traits))]
            names_traits = [names_traits[i].replace("'", "") for i in range(len(names_traits))]

            trait_val=[l.split() for l in L][1:]

            trait_values = np.array(trait_val)

            trait_taxa=np.array([l.split()[0] for l in L][1:])
            trait_taxa = [trait_taxa[i].replace('"', '') for i in range(len(trait_taxa))]
            trait_taxa = [trait_taxa[i].replace("'", "") for i in range(len(trait_taxa))]
            trait_taxa = np.array(trait_taxa)
            matched_trait_values = []
            for i in range(len(taxa_names)):
                taxa_name=taxa_names[i]
                matched_val = trait_values[trait_taxa==taxa_name]
                if len(matched_val)>0:
                    matched_trait_values.append(matched_val[0, 1:].astype(float))
                else:
                    matched_trait_values.append(np.nan)
                    sys.exit( "Species %s did not have data" % taxa_name)
            trait_values= np.array(matched_trait_values)    
        else:
            trait_values = None
        n_taxa = len(FA)
    
        if len(fixed_times_of_shift_bdnn) > 0:
            LO_min = np.min(LO)
            FA_LO = np.max(FA)
            if LO_min > 0.0:
                LO_min = np.zeros(1)
            FA_LO = np.concatenate((FA_LO, LO_min), axis=None)
            time_vec = np.sort(np.concatenate((FA_LO, fixed_times_of_shift_bdnn)))[::-1]
        else:
            time_vec = np.sort(np.array([np.max(FA), np.min(LO)] + list(fixed_times_of_shift)))[::-1]

        rescaled_time = []
        BDNNtimetrait_rescaler = 1
        if needs_bdnn_time or fix_edgeShift > 0:
            if args.BDNNtimetrait == -1:
                BDNNtimetrait_rescaler = 1 / np.max(time_vec)
            else:
                BDNNtimetrait_rescaler = args.BDNNtimetrait
            rescaled_time = time_vec * BDNNtimetrait_rescaler
            # print("rescaled times", rescaled_time, len(rescaled_time))

        if args.BDNNpklfile:
            print("loading BDNN pickle")
            bdnn_pkl = load_pkl(args.BDNNpklfile)
            trait_tbl_NN = [bdnn_pkl.trait_tbls[0], bdnn_pkl.trait_tbls[1]]
            cov_par_init_NN = bdnn_pkl.weights
            prior_bdnn_w_sd = [np.ones(cov_par_init_NN[0][i].shape) * args.BDNNprior for i in range(len(cov_par_init_NN[0]))]
            # settings
            block_nn_model = bdnn_pkl.bdnn_settings['block_nn_model']
            BDNN_MASK_lam = bdnn_pkl.bdnn_settings['mask_lam']
            BDNN_MASK_mu = bdnn_pkl.bdnn_settings['mask_mu']
        else:
            cov_par_init_NN = [0, 0, 0, 1.0, 1.0, 1.0]
            trait_tbl_NN = [None] * 3
            if BDNNmodel in [1, 3]:
                time_var_lambda = None
                time_var_mu = None
                names_time_var = []
                if bdnn_timevar[0]:
                    # For lambda
                     time_var_lambda, names_time_var = get_binned_time_variable(time_vec, bdnn_timevar[0], args.rescale, args.translate)
                     # For mu
                     time_var_mu, _ = get_binned_time_variable(time_vec, bdnn_timevar[-1], args.rescale, args.translate)
                     add_to_bdnnblock_mask += len(names_time_var)
                trait_tbl_lm, cov_par_init_lm = init_trait_and_weights(trait_values,
                                                                       time_var_lambda, time_var_mu,
                                                                       n_BDNN_nodes,
                                                                       n_bias_node=num_bias_node,
                                                                       fadlad=args.BDNNfadlad,
                                                                       use_time_as_trait=use_time_as_trait,
                                                                       dd=bdnn_dd,
                                                                       fixed_times_of_shift=rescaled_time,
                                                                       n_taxa=n_taxa,
                                                                       loaded_tbls=bdnn_loaded_tbls,
                                                                       float_prec_f=float_prec_f)
                trait_tbl_NN[0] = trait_tbl_lm[0]
                trait_tbl_NN[1] = trait_tbl_lm[1]
                cov_par_init_NN[0] = cov_par_init_lm[0]
                cov_par_init_NN[1] = cov_par_init_lm[1]
                prior_bdnn_w_sd = [np.ones(cov_par_init_NN[0][i].shape) * args.BDNNprior for i in range(len(cov_par_init_NN[0]))]
                from pyrate_lib.bdnn_lib import get_idx_feature_without_variance
                invariant_bdnn_pred = [get_idx_feature_without_variance(trait_tbl_lm[0]), get_idx_feature_without_variance(trait_tbl_lm[1])]
                has_invariant_bdnn_pred = np.sum(np.concatenate(invariant_bdnn_pred)) > 0
                if prior_lam_t_reg[0] > 0:
                    cov_par_init_NN[3] = 0.5
                if prior_lam_t_reg[1] > 0:
                    cov_par_init_NN[4] = 0.5
            if BDNNmodel in [2, 3]:
                hasFoundPyRateC = 0
                CPPlib = ""
                time_var_q = None
                names_time_var_q = []
                if bdnn_timevar_q:
                    time_var_q, names_time_var_q = get_binned_time_variable(q_time_frames_bdnn, bdnn_timevar_q, args.rescale, args.translate)
                trait_tbl_NN[2], cov_par_init_NN_q = init_sampling_trait_and_weights(trait_values,
                                                                                     time_var_q,
                                                                                     n_BDNN_nodes,
                                                                                     bias_node=False,
                                                                                     n_taxa=n_taxa,
                                                                                     replicates_tbls=highres_q_repeats,
                                                                                     float_prec_f=float_prec_f)
                cov_par_init_NN[2] = cov_par_init_NN_q
                prior_bdnn_w_q_sd = [np.ones(cov_par_init_NN[2][i].shape) * args.BDNNprior for i in range(len(cov_par_init_NN[2]))]
                if prior_lam_t_reg[-1] > 0:
                    cov_par_init_NN[5] = 0.5
            
            
            if BDNNmodel in [1, 3] and (block_nn_model or has_invariant_bdnn_pred):
                num_zeros = trait_tbl_NN[0].shape[-1]
                indx_input_list_1 = np.zeros(num_zeros) # includes +1 for time, diversity dependence and temperature
                indx_input_list_1[-1] = 1 # different block for time
                # mask block - 2nd layer (equal split trait, time)
                nodes_traits = int(n_BDNN_nodes[0] / 2)
                nodes_time = n_BDNN_nodes[0] - nodes_traits
                indx_input_list_2 = np.concatenate((np.zeros(nodes_traits),np.ones(nodes_time))).astype(int)
                nodes_per_feature_list = [list(np.unique(indx_input_list_2, return_counts=True)[1])]
                if len(n_BDNN_nodes) == 2:
                    nodes_traits = int(n_BDNN_nodes[1] / 2)
                    nodes_time = n_BDNN_nodes[1] - nodes_traits
                    nodes_per_feature_list.append([nodes_traits, nodes_time])
                nodes_per_feature_list.append([])
        
                BDNN_MASK_lam = create_mask(cov_par_init_NN[0],
                                            indx_input_list=[indx_input_list_1, indx_input_list_2, []],
                                            # nodes_per_feature_list=[[1, 1], [1, 1], []])
                                            nodes_per_feature_list=nodes_per_feature_list) # [[4, 4], [1, 1], []]
                BDNN_MASK_mu = create_mask(cov_par_init_NN[1],
                                           indx_input_list=[indx_input_list_1, indx_input_list_2, []],
                                           # nodes_per_feature_list=[[1, 1], [1, 1], []])
                                           nodes_per_feature_list=nodes_per_feature_list) # [[4, 4], [1, 1], []]
                if has_invariant_bdnn_pred:
                    # Block input of invariant predictors into neural network
                    if block_nn_model is False:
                        for i in range(len(BDNN_MASK_lam)):
                            BDNN_MASK_lam[i][:] = 1.0
                            BDNN_MASK_mu[i][:] = 1.0
                    BDNN_MASK_lam[0][:, invariant_bdnn_pred[0]] = 0.0
                    BDNN_MASK_mu[0][:, invariant_bdnn_pred[1]] = 0.0
                else:
                    [print("\n", i) for i in BDNN_MASK_lam]
        
            else:
                BDNN_MASK_lam = None
                BDNN_MASK_mu = None
        #---
            if False:
                print(rescaled_time)
                for i in trait_tbl_NN[0]:
                    print(i[0,:] )

        log_per_species_rates = True

        n_prm = 0
        n_free_prm = 0
        bdnn_settings = ""
        if BDNNmodel in [1, 3]:
            for i_layer in range(len(cov_par_init_NN[0])):
                if BDNN_MASK_lam:
                    cov_par_init_NN[0][i_layer] *= BDNN_MASK_lam[i_layer]
                if BDNN_MASK_mu:
                    cov_par_init_NN[1][i_layer] *= BDNN_MASK_mu[i_layer]
                n_prm += cov_par_init_NN[0][i_layer].size
                n_free_prm += np.sum(cov_par_init_NN[0][i_layer] != 0)
                bdnn_settings = bdnn_settings + "\n %s" % str(cov_par_init_NN[0][i_layer].shape)
        if BDNNmodel in [2, 3]:
            for i_layer in range(len(cov_par_init_NN[2])):
                n_prm += cov_par_init_NN[2][i_layer].size
                n_free_prm += np.sum(cov_par_init_NN[2][i_layer] != 0)
                bdnn_settings = bdnn_settings + "\n %s" % str(cov_par_init_NN[2][i_layer].shape)

        bdnn_settings = "\n\nUsing BDNN model\nN. free parameters: %s \nN. parameters: %s\n%s\n" % (n_prm, n_free_prm, bdnn_settings)
        print(bdnn_settings)




    ##### SETFREQ OF PROPOSING B/D shifts (RJMCMC)
    sample_shift_mu = args.rj_bd_shift # 0: updates only lambda; 1: only mu; default: 0.5



    #### ANALYZE PHYLOGENY
    analyze_tree = 0
    if args.tree != "":
        try:
            import dendropy
        except:
            sys.exit("Library 'dendropy' not found!\n")
        try:
            tree_list = dendropy.TreeList.get_from_path(args.tree, schema="nexus", preserve_underscores= 1)
            tree=tree_list[0]
            tree.resolve_polytomies(update_bipartitions= 1)
            #tree_node_ages = sort(tree.calc_node_ages(ultrametricity_precision=0.001,is_return_internal_node_ages_only= 1))[::-1]
            tree.calc_node_ages(ultrametricity_precision=0.001) #
            nd= tree.ageorder_node_iter(include_leaves=False, filter_fn=None, descending= 1)
            ages=list()
            for n, node in enumerate(nd): ages.append(node.age)
            tree_node_ages = sort(np.array(ages))[::-1]
        except:
            sys.exit("Tree format not recognized (NEXUS file required). \n")
        tree_sampling_frac = args.sampling
        analyze_tree = 1
        if args.bdc: analyze_tree = 2
        if args.eqr: analyze_tree = 3

        if fix_Shift == 1:
            import pyrate_lib.phylo_bds_likelihood as phylo_bds_likelihood
            analyze_tree = 4
            treeBDlikelihoodSkyLine = phylo_bds_likelihood.TreePar_LikShifts
            # args = (x,t,l,mu,sampling,posdiv=0,survival=1,groups=0)
            tree_node_ages = np.sort(tree_node_ages)
            phylo_times_of_shift = np.sort(np.array(list(fixed_times_of_shift) + [0]))
            tree_sampling_frac = np.array([tree_sampling_frac] + list(np.ones(len(fixed_times_of_shift))))
            # print(phylo_times_of_shift)
            # print(tree_node_ages)
            if args.bdc: 
                print("Using BDC Skyline model")
                args_bdc = 1
            else: 
                print("Using Skyline independent model")
                args_bdc = 0
            # print tree_sampling_frac
            #quit()


        TDI = 0




    # GET DATA SUMMARY INFO
    if args.data_info == 1:
        print("\nDATA SUMMARY\n")
        if len(singletons_excluded)>0: print("%s taxa excluded" % (len(singletons_excluded)))
        print("%s taxa included in the analysis" % (len(fossil)))
        one_occ_sp,all_occ,extant_sp,n_occs_list  = 0,0,0,list()
        for i in fossil:
            if len(i[i>0])==1: one_occ_sp+=1
            all_occ += len(i[i>0])
            n_occs_list.append(len(i[i>0]))
            if min(i)==0: extant_sp+=1
        print("%s taxa have a single occurrence, %s taxa are extant" % (one_occ_sp,extant_sp))
        j=0
        m_ages,M_ages=[],[]
        while True:
            try: fossil_complete=input_data_module.get_data(j)
            except(IndexError): break
            min_age, max_age = np.inf, 0
            sp_indtemp = 0
            for i in fossil_complete:
                if sp_indtemp in taxa_included:
                    a,b = min(i[i>0]), max(i)
                    if a < min_age: min_age=a
                    if b > max_age: max_age=b
                sp_indtemp+=1
            m_ages.append(min_age)
            M_ages.append(max_age)
            j+=1
        print("%s fossil occurrences (%s replicates), ranging from %s (+/- %s) to %s (+/- %s) Ma\n" % \
        (all_occ, j, round(mean(M_ages),3), round(std(M_ages),3),round(mean(m_ages),3), round(std(m_ages),3)))
        # print species FA,LO
        print("Taxon\tFA\tLA")
        for i in range(len(fossil)):
            foss_temp = fossil[i]
            if min(foss_temp)==0: status_temp="extant"
            else: status_temp=""
            print("%s\t%s\t%s\t%s" % (taxa_names[i], round(max(foss_temp),3),round(min(foss_temp[foss_temp>0]),3),status_temp))

        # print histogram
        n_occs_list = np.array(n_occs_list)
        hist = np.histogram(n_occs_list,bins = np.arange(np.max(n_occs_list)+1)+1)[0]
        hist2 = hist.astype(float)/np.max(hist) * 50
        #print "occs.\ttaxa\thistogram"
        #for i in range(len(hist)): print "%s\t%s\t%s" % (i+1,int(hist[i]),"*"*int(hist2[i]))
        sys.exit("\n")

    # RUN PP-MODEL TEST
    if args.PPmodeltest:
        self_path = get_self_path()
        pyrate_lib_path = "pyrate_lib"
        sys.path.append(os.path.join(self_path,pyrate_lib_path))
        import PPmodeltest
        if TPP_model== 0: times_q_shift = 0
        PPmodeltest.run_model_testing(fossil,q_shift=times_q_shift,min_n_fossils=2,verbose=1)
        # PPmodeltest.run_model_testing_n_shifts(fossil,q_shift=times_q_shift,min_n_fossils=2,verbose=1)
        sys.exit("\ndone.")


    # CREATE C++ OBJECTS
    if hasFoundPyRateC:
        if use_se_tbl==1:
            pass
        else:
            fossilForPyRateC = [ f.tolist() for f in fossil ]
            PyRateC_setFossils(fossilForPyRateC) # saving all fossil data as C vector

        
            fossilForPyRateC = [ f.tolist() for f in fossil ]
            PyRateC_setFossils(fossilForPyRateC) # saving all fossil data as C vector

        if args.qShift != "":  # q_shift times
            tmpEpochs = np.sort(np.array(list(times_q_shift)+[np.max(FA)+1]+[0]))[::-1]
            PyRateC_initEpochs(tmpEpochs)

    ############################ MCMC OUTPUT ############################
    try: os.mkdir(output_wd)
    except(OSError): pass
    if output_wd !="":
        path_dir = os.path.join(output_wd, "pyrate_mcmc_logs") 
    else: path_dir = "pyrate_mcmc_logs"

    folder_name="pyrate_mcmc_logs"
    try: os.mkdir(path_dir)
    except(OSError): pass

    suff_out=out_name
    if args.eqr: suff_out+= "_EQR"
    elif args.bdc: suff_out+= "_BDC"
    else:
        if TDI<=1: 
            if use_ADE_model >= 1:
                suff_out+= "_ADE"
            elif len(fixed_times_of_shift) > 0:
                suff_out+= "_BDS"
            else:
                suff_out+= "_BD%s-%s" % (args.mL,args.mM)
        if TDI==1: suff_out+= "_TI"
        if TDI==3: suff_out+= "dpp"
        if TDI==4: suff_out+= "rj"

    if BDNNmodel:
        suff_out+= "_BDNN_"
        suff_out+="_".join(map(str, n_BDNN_nodes))
        if BDNNmodel in [1, 3]:
            not_edgeshift_but_no_time = np.all(rescaled_time == 0.0) == False
            if use_time_as_trait and not_edgeshift_but_no_time:
                suff_out+= "T"
            if bdnn_timevar[0] or bdnn_loaded_timevar_pred:
                suff_out+= "V"
            if bdnn_dd:
                suff_out+= "DD"
            if bdnn_const_baseline:
                suff_out+= "c"
            if block_nn_model:
                suff_out+= "b"
        if BDNNmodel in [2, 3]:
            suff_out+= "q"

        # save BDNN object
        if isinstance(bdnn_loaded_tbls[0], np.ndarray):
            names_traits = bdnn_loaded_names_traits
        

        bdnn_dict = {
            'hidden_act_f': hidden_act_f,
            'prior_t_reg': prior_lam_t_reg,
            'independent_t_reg': independ_reg,
            'prior_cov': args.BDNNprior
        }
        # store fad/lad
        sp_fad_lad = []
        if use_se_tbl == 0:
            for i in range(len(fossil)):
                foss_temp = fossil[i]
                sp_fad_lad.append([taxa_names[i], np.max(foss_temp), np.min(foss_temp)])
            occ_data = fossil
        else:
            for i in range(len(taxa_names)):
                sp_fad_lad.append([taxa_names[i], FA[i], LO[i]])
            occ_data = None
        sp_fad_lad = pd.DataFrame(sp_fad_lad)
        sp_fad_lad.columns = ["Taxon", "FAD", "LAD"]
        
        if BDNNmodel in [1, 3]:
            layer_shapes_bd = [cov_par_init_NN[0][i_layer].shape for i_layer in range(len(cov_par_init_NN[0]))]
            layer_sizes_bd = [cov_par_init_NN[0][i_layer].size for i_layer in range(len(cov_par_init_NN[0]))]
            names_features_bd = []
            names_features_bd += names_traits
            names_features_bd += names_time_var
            if bdnn_dd:
                names_features_bd += ['diversity']
            if use_time_as_trait and not_edgeshift_but_no_time:
                names_features_bd += ['time']
            bdnn_rescale_div = 0.0
            if bdnn_dd:
                bdnn_obs_div = get_DT(time_vec, sp_fad_lad["FAD"], sp_fad_lad["LAD"])
                bdnn_rescale_div = np.max(bdnn_obs_div)
            bdnn_dict.update({
                'layers_shapes': layer_shapes_bd,
                'layers_sizes': layer_sizes_bd,
                'out_act_f': out_act_f,
                'float_prec_f': float_prec_f,
                'mask_lam': BDNN_MASK_lam,
                'mask_mu': BDNN_MASK_mu,
                'fixed_times_of_shift_bdnn': fixed_times_of_shift_bdnn,
                'use_time_as_trait': use_time_as_trait,
                'time_rescaler': BDNNtimetrait_rescaler,
                'bdnn_const_baseline': bdnn_const_baseline,
                'block_nn_model': block_nn_model,
                'names_features': names_features_bd,
                'div_rescaler': bdnn_rescale_div,
            })
            if fix_edgeShift > 0:
                bdnn_dict.update({
                    'apply_reg': apply_reg,
                    'bias_node_idx': bias_node_idx,
                    'fix_edgeShift': fix_edgeShift,
                    'edgeShifts': edgeShifts
                })
        
        if BDNNmodel in [2, 3]:
            layer_shapes_q = [cov_par_init_NN[2][i_layer].shape for i_layer in range(len(cov_par_init_NN[2]))]
            layer_sizes_q = [cov_par_init_NN[2][i_layer].size for i_layer in range(len(cov_par_init_NN[2]))]
            names_features_q = []
            names_features_q += names_traits
            names_features_q += names_time_var_q
            if bdnn_ads >= 0.0:
                names_features_q += ['taxon_age']
            bdnn_dict.update({
                'layers_shapes_q': layer_shapes_q,
                'layers_sizes_q': layer_sizes_q,
                'out_act_f_q': out_act_f_q,
                'names_features_q': names_features_q,
                'log_factorial_occs': log_factorial_occs,
                'singleton_mask': singleton_mask,
                'occs_sp': occs_sp,
                'pert_prior': pert_prior,
                'pp_gamma_ncat': pp_gamma_ncat
            })
            if use_HPP_NN_lik == 1:
                bdnn_dict.update({
                    'q_time_frames': q_time_frames_bdnn,
                    'duration_q_bins': duration_q_bins,
                    'occs_single_bin': occs_single_bin
                })
            if not highres_q_repeats is None: # bdnn_ads >= 0.0:
                bdnn_dict.update({'highres_q_repeats': highres_q_repeats})
        
        obj = bdnn(bdnn_settings=bdnn_dict,
                   weights=cov_par_init_NN,
                   trait_tbls=trait_tbl_NN,
                   sp_fad_lad=sp_fad_lad,
                   occ_data=occ_data)

        # print("obj.bdnn_settings", obj.bdnn_settings, cov_par_init_NN)
        bdnn_obj_out_file = "%s/%s.pkl" % (path_dir,suff_out)
        write_pkl(obj, bdnn_obj_out_file)
        print("\n\nBDNN object saved as:", bdnn_obj_out_file, "\n")
        # sys.exit()
    
        if args.BDNNexport_taxon_time_tables:
            import pyrate_lib.bdnn_lib as bdnn_lib
            path_predictors = bdnn_lib.export_trait_tbl(trait_tbl_NN, names_features_bd, output_wd)
            sys.exit("BDNN predictors export into %s" % path_predictors)
    
    if mcmc_gen > 0:
        # OUTPUT 0 SUMMARY AND SETTINGS
        o0 = "\n%s build %s\n" % (version, build)
        o1 = "\ninput: %s output: %s/%s" % (args.input_data, path_dir, out_name)
        o2 = "\n\nPyRate was called as follows:\n%s\n" % (args)
        if model_cov>=1 or useDiscreteTraitModel == 1: o2 += regression_trait
        if TDI==3: o2 += "\n\nHyper-prior on concentration parameter (Gamma shape, rate): %s, %s\n" % (hp_gamma_shape, hp_gamma_rate)
        if len(fixed_times_of_shift)>0:
            o2 += "\nUsing birth-death model with fixed times of rate shift: "
            for i in fixed_times_of_shift: o2 += "%s " % (i)
        o2+= "\n"+prior_setting

        if use_se_tbl != 1:
            if argsHPP == 1:
                if TPP_model == 0:
                    o2+="Using Homogeneous Poisson Process of preservation (HPP)."
                else:
                    o2 += "\nUsing Time-variable Poisson Process of preservation (TPP) at: "
                    for i in times_q_shift: o2 += "%s " % (i)

            else: o2+="Using Non-Homogeneous Poisson Process of preservation (NHPP)."
        
        if BDNNmodel:
            o2 += bdnn_settings

        version_notes="""\n
        Please cite: \n%s\n
        Feedback and support: pyrate.help@gmail.com
        OS: %s %s
        Python version: %s\n
        Numpy version: %s
        Scipy version: %s\n
        Random seed: %s %s
        """ % (citation,platform.system(), platform.release(), sys.version, np.version.version, scipy.version.version,rseed,CPPlib)



        o=''.join([o0,o1,o2,version_notes])
        out_sum = "%s/%s_sum.txt" % (path_dir,suff_out)
        sumfile = open(out_sum , "w", newline="")
        sumfile.writelines(o)
        sumfile.close()

        # OUTPUT 1 LOG MCMC
        out_log = "%s/%s_mcmc.log" % (path_dir, suff_out) #(path_dir, output_file, out_run)
        logfile = open(out_log , "w", newline="")
        if fix_SE == 0:
            if TPP_model == 0:
                head="it\tposterior\tprior\tPP_lik\tBD_lik\tq_rate\talpha\t"
            else:
                head="it\tposterior\tprior\tPP_lik\tBD_lik\t"
                for i in range(time_framesQ): head += "q_%s\t" % (i)
                head += "alpha\t"
                if pert_prior[1]==0: head +="hypQ\t"
        else:
            head="it\tposterior\tprior\tBD_lik\t"

        if model_cov>=1:
            head += "cov_sp\tcov_ex\tcov_q\t"
            if est_COVAR_prior == 1: head+="cov_hp\t"
        if edge_indicator:
            head += "pI\t"
        if TDI<2:
            head += "root_age\tdeath_age\t"
            if TDI==1: head += "beta\t"
            if est_hyperP == 1:
                head += "hypL\thypM\t"

            if BDNNmodel in [1, 3] and bdnn_const_baseline:
                pass #head += "lambda_avg\tmu_avg\t"
            elif use_ADE_model == 0 and useDiscreteTraitModel == 0 or BDNNmodel in [2]:
                for i in range(time_framesL): head += "lambda_%s\t" % (i)
                for i in range(time_framesM): head += "mu_%s\t" % (i)
            elif use_ADE_model >= 1:
                if use_ADE_model == 1:
                    head+="w_shape\t"
                    for i in range(time_framesM): head += "w_scale_%s\t" % (i)
                elif use_ADE_model == 2:
                    head+="corr_lambda\t"
                    for i in range(time_framesM): head += "corr_mu_%s\t" % (i)
                for i in range(time_framesM): head += "mean_longevity_%s\t" % (i)
            elif useDiscreteTraitModel == 1:
                for i in range(len(lengths_B_events)): head += "lambda_%s\t" % (i)
                for i in range(len(lengths_D_events)): head += "mu_%s\t" % (i)

            if fix_Shift== 0:
                for i in range(1,time_framesL): head += "shift_sp_%s\t" % (i)
                for i in range(1,time_framesM): head += "shift_ex_%s\t" % (i)

            if analyze_tree >=1:
                if analyze_tree==4:
                    head += "tree_lik\t"
                    for i in range(time_framesL): head += "tree_sp_%s\t" % (i)
                    for i in range(time_framesL): head += "tree_ex_%s\t" % (i)
                else:
                    head += "tree_lik\ttree_sp\ttree_ex\t"

        elif TDI == 2: head+="k_birth\tk_death\troot_age\tdeath_age\t"
        elif TDI == 3: head+="k_birth\tk_death\tDPP_alpha_L\tDPP_alpha_M\troot_age\tdeath_age\t"
        elif TDI == 4: head+="k_birth\tk_death\tRJ_hp\troot_age\tdeath_age\t"


        if useDiscreteTraitModel == 1:
            for i in range(len(lengths_B_events)): head += "S_%s\t" % (i)
        head += "tot_length"
        head=head.split('\t')
        if use_se_tbl == 0: tot_number_of_species = len(taxa_names)

        if BDNNmodel:
            if BDNNmodel in [1, 3]:
                # weights lam
                for i in range(len(cov_par_init_NN[0])):
                    head += ["w_lam_%s_%s" % (i, j) for j in range(cov_par_init_NN[0][i].size)]
                # weights mu
                for i in range(len(cov_par_init_NN[1])):
                    head += ["w_mu_%s_%s" % (i, j) for j in range(cov_par_init_NN[1][i].size)]
            if BDNNmodel in [2, 3]:
                # weights q
                for i in range(len(cov_par_init_NN[2])):
                    head += ["w_q_%s_%s" % (i, j) for j in range(cov_par_init_NN[2][i].size)]
            if BDNNmodel in [1, 3]:
                head += ["t_reg_lam", "t_reg_mu"]
                head += ["reg_denom_lam", "reg_denom_mu"]
            if BDNNmodel in [2, 3]:
                head += ["t_reg_q", "reg_denom_q", "normalize_q"]

        if fix_SE == 0:
            for i in taxa_names: head.append("%s_TS" % (i))
            for i in taxa_names: head.append("%s_TE" % (i))

        if tune_T_schedule[0] > 0:
            head += ["accept_ratio_ts"]
            if np.any(LO > 0):
                head += ["accept_ratio_te"]
            head += ["tT_ts"]
            if np.any(LO > 0):
                head += ["tT_te"]

        wlog=csv.writer(logfile, delimiter='\t')
        wlog.writerow(head)
        
        #logfile.writelines(head)
        logfile.flush()
        os.fsync(logfile)

        # OUTPUT 2 MARGINAL RATES
        if args.log_marginal_rates == -1: # default values
            if TDI==4 or use_ADE_model != 0 or BDNNmodel: log_marginal_rates_to_file = 0
            else: log_marginal_rates_to_file = 1
        else:
            log_marginal_rates_to_file = args.log_marginal_rates

        # save regular marginal rate file
        if TDI!=1 and use_ADE_model == 0 and useDiscreteTraitModel == 0 and log_marginal_rates_to_file==1: # (path_dir, output_file, out_run)
            if useBounded_BD == 1: max_marginal_frame = boundMax+1
            else: max_marginal_frame = np.max(FA)
            marginal_frames= array([int(fabs(i-int(max_marginal_frame))) for i in range(int(max_marginal_frame)+1)])
            if log_marginal_rates_to_file==1:
                out_log_marginal = "%s/%s_marginal_rates.log" % (path_dir, suff_out)
                marginal_file = open(out_log_marginal , "w", newline="")
                head="it\t"
                for i in range(int(max_marginal_frame)+1): head += "l_%s\t" % i #int(fabs(int(max(FA))))
                for i in range(int(max_marginal_frame)+1): head += "m_%s\t" % i #int(fabs(int(max(FA))))
                for i in range(int(max_marginal_frame)+1): head += "r_%s\t" % i #int(fabs(int(max(FA))))
                head=head.split('\t')
                wmarg=csv.writer(marginal_file, delimiter='\t')
                wmarg.writerow(head)
                marginal_file.flush()
                os.fsync(marginal_file)

        # save files with sp/ex rates and times of shift
        elif log_marginal_rates_to_file == 0:
            samplingNN_TDI0 = BDNNmodel == 2 and TDI == 0
            if TDI==4 or use_ADE_model != 0 or (BDNNmodel and not samplingNN_TDI0):
                marginal_sp_rate_file_name = "%s/%s_sp_rates.log" % (path_dir, suff_out)
                marginal_sp_rate_file = open(marginal_sp_rate_file_name , "w", newline="")
                w_marg_sp=csv.writer(marginal_sp_rate_file, delimiter='\t')
                marginal_sp_rate_file.flush()
                os.fsync(marginal_sp_rate_file)
                marginal_ex_rate_file_name = "%s/%s_ex_rates.log" % (path_dir, suff_out)
                marginal_ex_rate_file = open(marginal_ex_rate_file_name , "w", newline="")
                w_marg_ex=csv.writer(marginal_ex_rate_file, delimiter='\t')
                marginal_ex_rate_file.flush()
                os.fsync(marginal_ex_rate_file)
            if BDNNmodel in [2, 3]:
                marginal_q_rate_file_name = "%s/%s_q_rates.log" % (path_dir, suff_out)
                marginal_q_rate_file = open(marginal_q_rate_file_name , "w", newline="")
                w_marg_q = csv.writer(marginal_q_rate_file, delimiter='\t')
                marginal_q_rate_file.flush()
                os.fsync(marginal_q_rate_file)
            marginal_frames=0
            if BDNNmodel:
                fixed_times_of_shift_bdnn_logger = fixed_times_of_shift_bdnn
                if not (use_time_as_trait or bdnn_timevar[0] or bdnn_dd or bdnn_loaded_tbls_timevar): #  or (TPP_model != 1 and BDNNmodel in [2])
                    times_rtt, fixed_times_of_shift_bdnn_logger = make_missing_bins(FA)

        # OUTPUT 3 MARGINAL LIKELIHOOD
        elif TDI==1:
            out_log_marginal_lik = "%s/%s_marginal_likelihood.txt" % (path_dir, suff_out)
            marginal_file = open(out_log_marginal_lik , "w", newline="")
            marginal_file.writelines(o)
            marginal_frames=0
        else: marginal_frames=0

        if fix_SE == 1 and fix_Shift == 1:
            time_frames  = np.sort(np.array(list(fixed_times_of_shift) + [0,np.max(fixed_ts)]))
            B = sort(time_frames)+0.000001 # add small number to avoid counting extant species as extinct
            ss1 = np.histogram(fixed_ts,bins=B)[0][::-1]
            ss1[0] = ss1[0]-no_starting_lineages
            ee2 = np.histogram(fixed_te,bins=B)[0][::-1]
            len_SS1,len_EE1 = list(),list()
            S_time_frame =list()
            time_frames = time_frames[::-1]
            for i in range(len(time_frames)-1):
                up, lo = time_frames[i], time_frames[i+1]
                len_SS1.append(ss1[i])
                len_EE1.append(ee2[i])
                inTS = np.fmin(fixed_ts,up)
                inTE = np.fmax(fixed_te,lo)
                temp_S = inTS-inTE
                S_time_frame.append(sum(temp_S[temp_S>0]))

            len_SS1 = np.array(len_SS1)
            len_EE1 = np.array(len_EE1)
            S_time_frame = np.array(S_time_frame)

        # OUTPUT 4 PER-SPECIES RATES
        if BDNNmodel in [1, 3] and log_per_species_rates:
            species_rate_file_name = "%s/%s_per_species_rates.log" % (path_dir, suff_out)
            head = ["iteration"]
            for i in taxa_names: head.append("%s_lam" % (i))
            for i in taxa_names: head.append("%s_mu" % (i))
            species_rate_file = open(species_rate_file_name , "w", newline="")
            species_rate_writer=csv.writer(species_rate_file, delimiter='\t')
            species_rate_writer.writerow(head)
            species_rate_file.flush()

        # OUTPUT 5 species-specific (relative) preservation rate
        if sp_specific_q_rates: #  or BDNNmodel in [2, 3]
            sp_q_marg_rate_file_name = "%s/%s_per_species_q_rates.log" % (path_dir, suff_out)
            head = ["iteration", "alpha"]
            for i in taxa_names: head.append("%s_rel_q" % (i))
            sp_q_marg_rate_file = open(sp_q_marg_rate_file_name , "w", newline="")
            sp_q_marg=csv.writer(sp_q_marg_rate_file, delimiter='\t')
            sp_q_marg.writerow(head)
            sp_q_marg_rate_file.flush()
    

        ########################## START MCMC ####################################
        if burnin<1 and burnin>0:
            burnin = int(burnin*mcmc_gen)    

        # print("TDI", TDI)

        def start_MCMC(run):
            # marginal_file is either for rates or for lik
            return MCMC([0,run, IT, sample_freq, print_freq, temperatures, burnin, marginal_frames, list()])

        # Metropolis-coupled MCMC (Altekar, et al. 2004)
        if use_seq_lik == 0 and runs>1:
            marginal_frames= array([int(fabs(i-int(np.max(FA)))) for i in range(int(np.max(FA))+1)])
            pool = mcmcMPI(num_proc)
            res = pool.map(start_MCMC, list(range(runs)))
            current_it=0
            swap_rate, attempts=0, 0
            while current_it<mcmc_gen:
                n1=np.random.randint(0,runs-1,2)
                temp=1./(1+n1*temp_pr)
                [j,k]=n1
                #print "try: ", current_it, j,k #, res[j] #[2],res[0][2], res[1][2],res[2][2]
                r=(res[k][2]*temp[0]+res[j][2]*temp[1]) - (res[j][2]*temp[0]+res[k][2]*temp[1])
                if r>=log(random.random()) and j != k:
                    args=list()
                    best,ind=res[1][2],0
                    #print current_it, "swap %s with chain %s [%s / %s] temp. [%s / %s]" % (j, k, res[j][2],res[k][2], temp[0],temp[1])
                    swap_rate+=1
                    res_temp1=res[j]
                    res_temp2=res[k]
                    res[j]=res_temp2
                    res[k]=res_temp1
                current_it=res[0][0]
                res[0][0]=0
                args=list()
                for i in range(runs):
                    seed=0
                    args.append([current_it,i, current_it+IT, sample_freq, print_freq, temperatures, burnin, marginal_frames, res[i]])
                res = pool.map(MCMC, args)
                #except: print current_it,i, current_it+IT
                attempts+=1.
                #if attempts % 100 ==0:
                #    print "swap freq.", swap_rate/attempts
                #    for i in range(runs): print "chain", i, "post:", res[i][2], sum(res[i][5]-res[i][6])

        else:
            if runs>1: print("\nWarning: MC3 algorithm requires multi-threading.\nUsing standard (BD)MCMC algorithm instead.\n")
            res=start_MCMC(0)
        print("\nfinished at:", time.ctime(), "\nelapsed time:", round(time.time()-t1,2), "\n")
        logfile.close()
