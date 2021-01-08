#!/usr/bin/env python 
import argparse, os,sys
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import scipy.special
import random as rand
import platform, time
import multiprocessing, _thread
import multiprocessing.pool
import os, csv, glob
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
from pyrate_lib.lib_utilities import get_mode as get_mode
import pyrate_lib.lib_utilities as lib_utilities


#### DATA ###

p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('-d',     type=str,     help='data set (s/e table)', default=0, metavar=0)
p.add_argument('-m',     type=int,     help='model (0: exponential (default), 1: linear)', default=0, metavar=0)
p.add_argument('-n',     type=int,     help='MCMC iterations', default=10000000, metavar=10000000)
p.add_argument('-s',     type=int,     help='sampling freq.', default=5000, metavar=5000)
p.add_argument('-p',     type=int,     help='print freq.', default=5000, metavar=5000)
p.add_argument('-j',     type=int,     help='replicate', default=0, metavar=0)
p.add_argument('-jvar',     type=int, help='replicate variable', default=-1, metavar=-1)
p.add_argument('-b',     type=int,     help='burnin (number of generations)', default=1, metavar=1)
p.add_argument('-r',     type=float,   help='Data scaling (default option recommended)', default=0, metavar=0)
p.add_argument('-plot',  type=str,     help='Plot rates-through-time (Log file)', default="", metavar="")
p.add_argument('-var',   type=str,     help='Directory to continuous variables (takes all files)', default="", metavar="")
p.add_argument('-maxT',     type=float,   help='Max age (truncate data)', default=-1, metavar=-1)
p.add_argument('-minT',     type=float,   help='Min age (truncate data)', default=-1, metavar=-1)
p.add_argument('-out',   type=str,     help='tag added to output file', default="", metavar="")
p.add_argument('-bound', type=float,   help='absolute boundaries to local shrinkage (0 +/- bound)', default=np.inf, metavar=np.inf)
p.add_argument('-rmDD',  type=int,     help='model (0: analysis includes self-diversity-dependence, 1: analysis excludes selfdiversity-dependence', default=0, metavar=0)
p.add_argument('-hsp',  type=int,     help='1: use Horshoe prior; 0: fixed prior (gamma conjugate prior on precision)', default=1, metavar=1)
p.add_argument('-birth_model',  type=int,     help='1: use birth-only process', default=0, metavar=0)
p.add_argument('-death_model',  type=int,     help='1: use death-only process', default=0, metavar=0)
p.add_argument('-ignore_clade_col',  type=int,     help='1: ignore clade assignement (clades currently not supported)', default=1, metavar=1)






args = p.parse_args()


dataset=args.d
n_iterations=args.n
sampling_freq=args.s
print_freq = args.p
useHSP = args.hsp
birth_model = args.birth_model
death_model = args.death_model






#t_file=np.genfromtxt(dataset, names=True, delimiter='\t', dtype=float)
t_file=np.loadtxt(dataset, skiprows=1)

name_file = os.path.splitext(os.path.basename(dataset))[0]
wd = "%s" % os.path.dirname(dataset)

ts=t_file[:,2+2*args.j]
te=t_file[:,3+2*args.j]
if args.ignore_clade_col:
    clade_ID = np.zeros(len(ts)).astype(int)
else:
    clade_ID=t_file[:,0]
    clade_ID=clade_ID.astype(int)

if args.plot != "":
    j = np.arange((np.shape(t_file)[1]-2)/2)
    ts_all=t_file[:,np.array(2+2*j).astype(int)]
    te_all=t_file[:,np.array(3+2*j).astype(int)]
    ts=np.mean(ts_all,axis=1)
    te=np.mean(te_all,axis=1)
    

corr_model=args.m
if corr_model ==0: model_name = "exp"
else: model_name = "lin"
if args.out != "": out_tag = args.out + "_"
else:  out_tag = ""
root_age=max(ts)

single_focal_clade = True
fixed_focal_clade = 0

burnin = args.b

remove_selfDD = args.rmDD

#print len(ts),len(te)
#### ADD DATA FROM FILES
dir_to_files = args.var
all_files="%s/*" % (dir_to_files)
list_files_temp=list(sort(glob.glob(all_files)))

if args.jvar != -1:
    file_tag = "_%s_" % args.jvar
    list_files_temp = [ list_files_temp[i] for i in range(len(list_files_temp)) if file_tag in list_files_temp[i] ]

list_files = [""]+list_files_temp

variable_names=[]
for i in range(len(list_files)):
    name_var_file = os.path.splitext(os.path.basename(list_files[i]))[0]
    if i==0: name_var_file = "Diversity dependence"
    variable_names.append(name_var_file)
    if remove_selfDD and i==0: pass
    else: print(i, name_var_file)

# first item is empty because it's were the Dtraj goes
print("Processing files...")
for i in range(1,len(list_files)): # add data from curves
    try: temp_tbl = np.loadtxt(list_files[i],skiprows=1)
    except: sys.exit("Could not read file: %s" % (list_files[i]))
    time_var = temp_tbl[:,0] # time
    #var_val  = temp_tbl[:,1] # var value
    index_curve = np.ones(len(time_var))*i
    index_curve=index_curve.astype(int)
    clade_ID = np.concatenate((clade_ID,index_curve),axis=0)
    ts=np.concatenate((ts,time_var),axis=0)
    te=np.concatenate((te,np.zeros(len(index_curve))),axis=0)
    
# print clade_ID, len(ts),len(te)


all_events=sort(np.concatenate((ts,te),axis=0))[::-1] # events are speciation/extinction that change the diversity trajectory
n_clades,n_events=max(clade_ID)+1,len(all_events)
Dtraj=init_Dtraj(n_clades,n_events)

##### RTT PLOTS
plot_RTT = 0

# NEW FUNCTION 
if args.plot != "":
    plot_RTT = 1
    np.summary_file = args.plot
    name_file = os.path.splitext(os.path.basename(np.summary_file))[0]
    print("Parsing log file:", np.summary_file)
    fixed_focal_clade,baseline_L,baseline_M,Gl_focal_clade,Gm_focal_clade,est_kl,est_km = lib_utilities.parse_hsp_logfile(np.summary_file,burnin)
    fixed_focal_clade,baseline_L_list,baseline_M_list,Gl_focal_clade_list,Gm_focal_clade_list,est_kl,est_km = lib_utilities.parse_hsp_logfile_HPD(np.summary_file,burnin)
    #else: sys.exit("Unable to parse file.")


##### get indexes
s_list = []
e_list = []
s_or_e_list=[]
clade_inx_list=[]
unsorted_events = []
print("Indexing events...")
for i in range(n_clades):
    "used for Dtraj"
    s_list.append(ts[clade_ID==i])
    e_list.append(te[clade_ID==i])
    "used for lik calculation"
    if i==0: # diversity traj
        s_or_e_list += list(np.repeat(1,len(ts[clade_ID==i]))) # index 1 for s events
        s_or_e_list += list(np.repeat(2,len(te[clade_ID==i]))) # index 2 for e events
    else: # additional curves
        s_or_e_list += list(np.repeat(0,len(ts[clade_ID==i]))) # index 0 for events of continuous variable change
        s_or_e_list += list(np.repeat(0,len(te[clade_ID==i]))) # index 0 for events of continuous variable change
        
    clade_inx_list += list(np.repeat(i,2*len(te[clade_ID==i])))
    unsorted_events += list(ts[clade_ID==i])
    unsorted_events += list(te[clade_ID==i])

s_or_e_array= np.array(s_or_e_list)
unsorted_events= np.array(unsorted_events)
s_or_e_array[unsorted_events==0] = 3

""" so now: s_or_e_array = 0 (cont variable change), s_or_e_array = 1 (s events), s_or_e_array = 2 (e events), s_or_e_array = 3 (e=0 events)"""


""" concatenate everything:
                          1st row: all events  2nd row index s,e     3rd row clade index """
all_events_temp= np.array([unsorted_events,    s_or_e_array,         np.array(clade_inx_list)])
# sort by time
idx = np.argsort(all_events_temp[0])[::-1] # get indexes of sorted events
all_events_temp2=all_events_temp[:,idx] # sort by time of event
#print all_events_temp2
#print shape(all_events_temp2),len(all_events)
all_time_eve=all_events_temp2[0]


idx_s = []
idx_e = []
for i in range(n_clades): # make trajectory curves for each clade
    if remove_selfDD and i==0: pass
    else: print("\tparsing variable", i)
    if i==0:
        dd_focus_clade=getDT(all_events_temp2[0],s_list[i],e_list[i]) + np.zeros(len(all_events_temp2[0]))
        # dd_focus_clade: raw diversity trajectory (not rescaled 0 to 1) is used in the likelihood calculation
        Dtraj[:,i] = dd_focus_clade/np.max(dd_focus_clade)
        ind_clade_i = np.arange(len(all_events_temp2[0]))[all_events_temp2[2]==i]
        ind_sp = np.arange(len(all_events_temp2[0]))[all_events_temp2[1]==1]
        ind_ex = np.arange(len(all_events_temp2[0]))[all_events_temp2[1]==2]
        idx_s.append(np.intersect1d(ind_clade_i,ind_sp))
        idx_e.append(np.intersect1d(ind_clade_i,ind_ex))
    else:
        # print list_files[i]
        temp_tbl = np.loadtxt(list_files[i],skiprows=1)
        time_var = temp_tbl[:,0] # time
        var_val  = temp_tbl[:,1] # var value
        
        # REQUIRED ARGS:
        all_Times = all_events_temp2[0]
        Var_values = var_val
        times_of_T_change_indexes = all_events_temp2[1] # all curves; times_of_T_change_indexes==0 curve change
        times_of_T_change = time_var
        root_age = max(ts)
        clade_indexes = all_events_temp2[2]
        curve_index = i
        # get curve values
        Dtraj[:,i] = get_VarValue_at_timeMCDD(all_Times,Var_values,times_of_T_change_indexes,times_of_T_change,root_age,clade_indexes,curve_index )
        v = get_VarValue_at_timeMCDD(all_Times,Var_values,times_of_T_change_indexes,times_of_T_change,root_age,clade_indexes,curve_index )
        
        # CHECK
        #print "\n\n\n\n\n\n"
        #for j in range(len(all_Times)):
        #    print "%s\t%s" % (all_Times[j],v[j])#,dd[j])
        #
        #print "\n\n\n\n\n\n"
        #sys.exit()
        ##print "\n\n\n\n\nso far..."

# print Dtraj


##### HORSESHOE PRIOR FUNCTIONS
def pdf_gamma(L,a,b): 
    return scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0)

def pdf_normal(L,sd): 
    return scipy.stats.norm.logpdf(L,loc=0,scale=sd)

def pdf_cauchy(x,s=1):
    return scipy.stats.cauchy.logpdf(x,scale=s,loc=0)
 

def sample_lam_mod(lam,beta,tau):
    eta=1./(lam**2)
    mu =beta/tau
    u =np.random.uniform(0, 1./(1+eta), len(eta))
    truncate = (1-u)/u
    # 2/(mu**2) = scale parameter
    new_eta = np.random.exponential( 2/(mu**2), len(mu)  )
    new_lam = np.zeros(len(lam))+lam
    new_lam[new_eta<truncate]= sqrt(1./new_eta[new_eta<truncate])
    return new_lam

def sample_tau_mod(lam,beta,tau):
    eta=1./(tau**2)
    u =np.random.uniform(0, 1./(1+eta))
    truncate = (1-u)/u
    theta = (beta/lam)
    a = (len(lam.flatten())+1)/2.
    b = np.sum((theta**2)/2)
    # 1./b = scale parameter = 2/np.sum(theta**2) || cf. 2/(mu**2) = scale parameter above
    new_eta = np.random.gamma( a, 1./b, len(tau)  )
    new_tau = np.zeros(len(tau))+tau
    new_tau[new_eta<truncate]= sqrt(1./new_eta[new_eta<truncate])
    return new_tau


# Scott 2010 arXiv:1010.5265v1
#eta = 1/(Tau^2)
#u = runif(1,0,1/(eta + 1))
#ub = (1-u)/u
#a = (p+1)/2
#b = np.sum(Theta^2)/2
#ub2 = pgamma(ub,a,rate=b)
#u2 = runif(1,0,ub2)
#eta = qgamma(u2,a,rate=b)
#Tau = 1/sqrt(eta)

#####
scaling =args.r
maxG = args.bound
if scaling==0: # All trajectories are scaled to range between 0 and 1
    Dtraj= np.add(Dtraj, -np.min(Dtraj, axis=0))
    scale_factor = 1.
    scale_factor = 1./(np.max(Dtraj, axis=0)-np.min(Dtraj, axis=0))
    if corr_model==1: trasfRate_general = trasfMultiRateND 
    elif corr_model==0: trasfRate_general = trasfMultiRateND_exp
elif scaling == 1: # all curves scaled to the max of the highest curve (useful if they are in the same unit, e.g. species)
    scale_factor = 1./np.max(Dtraj)
    if maxG ==0: maxG = 0.30/scale_factor # as in Silvestro et al. 2015 PNAS
    trasfRate_general = trasfMultiRate
elif scaling ==2:
    scale_factor = 1./np.max(Dtraj, axis=0)
    trasfRate_general = trasfMultiRateCladeScaling

Dtraj = Dtraj*scale_factor
print("scale_factor",scale_factor, np.max(Dtraj), np.max(Dtraj, axis=0))
print(maxG, scale_factor)

if remove_selfDD==1:
    Dtraj = Dtraj[:,1:] # remove the diversity column
    n_clades = n_clades -1 # remove diversity from the number of co-variates
    scale_factor = scale_factor[1:] # remove the scale factor for diversity
    variable_names = variable_names[1:]

print(np.shape(Dtraj))

GarrayA=init_Garray(n_clades) # 3d array so:
                              # Garray[i,:,:] is the 2d G for one clade
                        # Garray[0,0,:] is G_lambda, Garray[0,1,:] is G_mu for clade 0
                    
                    
if birth_model: GarrayA[fixed_focal_clade,1,:] = 0
elif death_model: GarrayA[fixed_focal_clade,0,:] = 0
                    
if plot_RTT: 
    # G estimates are given per species but Dtraj are rescaled when:  scaling > 0 (default: scaling = 1)
    GarrayA[fixed_focal_clade,0,:] += Gl_focal_clade/scale_factor 
    GarrayA[fixed_focal_clade,1,:] += Gm_focal_clade/scale_factor 
else:
    GarrayA[fixed_focal_clade,:,:] += np.random.normal(0,0.001,np.shape(GarrayA[fixed_focal_clade,:,:]))
    print(dataset,args.j,model_name,out_tag)
    dataset_name = dataset.replace(".txt", "")
    out_file_name="%s_%s_%s_%sMBD.log" % (dataset_name,args.j,model_name,out_tag)
    logfile = open(out_file_name , "w") 
    wlog=csv.writer(logfile, delimiter='\t')

    lik_head=""
    head="it\tposterior\tlikelihood\tprior"
    head+="\tl%s" % (fixed_focal_clade)
    head+="\tm%s" % (fixed_focal_clade)
    for j in range(n_clades): 
        head+="\tGl%s_%s" % (fixed_focal_clade,j)
    for j in range(n_clades): 
        head+="\tGm%s_%s" % (fixed_focal_clade,j)
    for j in range(n_clades): 
        head+="\tWl%s_%s" % (fixed_focal_clade,j)
    for j in range(n_clades): 
        head+="\tWm%s_%s" % (fixed_focal_clade,j)

    head+="\tLAM_mu"        
    head+="\tLAM_sd"        
    head+="\tTau"        
    head+="\thypR"
    wlog.writerow(head.split('\t'))
    logfile.flush()


LAM=init_Garray(n_clades)
LAM[fixed_focal_clade,:,:] = 1.
l0A,m0A=init_BD(n_clades),init_BD(n_clades)

TauA=np.array([.5]) # np.ones(1) # P(G==0)
hypRA=np.ones(1)
Tau=TauA

max_T = args.maxT
min_T = args.minT

if max_T != -1 or min_T != -1: 
    if max_T == -1:
        max_T = np.max(all_Times)
    if min_T == -1:
        min_T = np.min(all_Times)
        
        
    index_temp = np.arange(0,len(all_Times))
    M_index_events_included = index_temp[all_Times <= max_T]
    
    sp_times = all_Times[idx_s[fixed_focal_clade]]
    ex_times = all_Times[idx_e[fixed_focal_clade]]
    
    M_index_temp = np.arange(0,len(sp_times))
    M_index_included_sp_times = M_index_temp[sp_times <= max_T]

    M_index_temp = np.arange(0,len(ex_times))
    M_index_included_ex_times = M_index_temp[ex_times <= max_T]

    index_temp = np.arange(0,len(all_Times))
    m_index_events_included = index_temp[all_Times >=min_T]
       
    m_index_temp = np.arange(0,len(sp_times))
    m_index_included_sp_times = m_index_temp[sp_times >= min_T]

    m_index_temp = np.arange(0,len(ex_times))
    m_index_included_ex_times = m_index_temp[ex_times >= min_T]
    
    # combined
    index_temp = np.intersect1d(M_index_temp, m_index_temp)
    index_events_included = np.intersect1d(M_index_events_included, m_index_events_included)
    
    index_included_sp_times = np.intersect1d(m_index_included_sp_times, M_index_included_sp_times)
    index_included_ex_times = np.intersect1d(m_index_included_ex_times, M_index_included_ex_times)
    
    




########################## PLOT RTT ##############################
if plot_RTT: # NEW FUNCTION 2
    out="%s/%s_RTT.r" % (wd,name_file)
    newfile = open(out, "w") 
    if model_name == "exp": model_type = "Exponential"
    else: model_type = "Linear"
        
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        wd_forward = os.path.abspath(wd).replace('\\', '/')
        r_script= "\n\npdf(file='%s/%s_RTT.pdf',width=0.6*20, height=0.6*10)\nlibrary(scales)\n" % (wd_forward,name_file)
    else: 
        r_script= "\n\npdf(file='%s/%s_RTT.pdf',width=0.6*20, height=0.6*10)\nlibrary(scales)\n" % (wd,name_file)
    
    for i in range(n_clades):
        r_script+=lib_utilities.print_R_vec("\nclade_%s", Dtraj[:,i]) % (i+1)
    
    newfile.writelines(r_script)
    newfile.flush()
    # get marginal rates
    print("Getting marginal rates...")
    
    
    for i in range(-1, n_clades):
        marginal_L = list()
        marginal_M = list()
        Gl_temp,Gm_temp=[], []
        for j in range(len(baseline_L_list)): # loop over MCMC samples
            baseline_L = baseline_L_list[j]
            baseline_M = baseline_M_list[j]
            Gl_focal_clade = Gl_focal_clade_list[j,:]
            Gm_focal_clade = Gm_focal_clade_list[j,:]
            # G estimates are given per species but Dtraj are rescaled when:  scaling > 0 (default: scaling = 1)
            GarrayA=init_Garray(n_clades)
            GarrayA[fixed_focal_clade,0,:] += Gl_focal_clade/scale_factor 
            GarrayA[fixed_focal_clade,1,:] += Gm_focal_clade/scale_factor 
    
            if i==-1:
                G_temp = GarrayA+0
                #if j==0: print GarrayA[fixed_focal_clade,0,:] 
            else:
                G_temp = init_Garray(n_clades)
                G_temp[fixed_focal_clade,:,i] += GarrayA[fixed_focal_clade,:,i]
                Gl_temp.append(G_temp[fixed_focal_clade,0,i])
                Gm_temp.append(G_temp[fixed_focal_clade,1,i])
                #if j==0: print G_temp[fixed_focal_clade,0,:] 
                    
    
            marginal_L.append(trasfRate_general(baseline_L, G_temp[fixed_focal_clade,0,:],Dtraj))
            marginal_M.append(trasfRate_general(baseline_M, G_temp[fixed_focal_clade,1,:],Dtraj))
            if j % 100 ==0: 
                sys.stdout.write(".")
                sys.stdout.flush()

        if i== -1: print("\nCalculating mean rates and HPDs...")            
        else: print("\nProcessing variable:", variable_names[i])
        
        marginal_L = np.array(marginal_L)
        marginal_M = np.array(marginal_M)
        #print np.shape(marginal_L)

        l_vec= np.zeros(np.shape(marginal_L)[1])
        m_vec= np.zeros(np.shape(marginal_L)[1])
        hpd_array_L= np.zeros((2,np.shape(marginal_L)[1]))
        hpd_array_M= np.zeros((2,np.shape(marginal_L)[1]))
        hpd_array_L50= np.zeros((2,np.shape(marginal_L)[1]))
        hpd_array_M50= np.zeros((2,np.shape(marginal_L)[1]))
        
        if i>=0:
            l_vec = np.mean(marginal_L, axis=0) # get_mode
            m_vec = np.mean(marginal_M, axis=0) # get_mode
        else:        
            for ii in range(np.shape(marginal_L)[1]): # loop over marginal rates
                l_vec[ii] = np.mean(marginal_L[:,ii]) # get_mode
                m_vec[ii] = np.mean(marginal_M[:,ii]) # get_mode
                hpd_array_L[:,ii] = calcHPD(marginal_L[:,ii])
                hpd_array_M[:,ii] = calcHPD(marginal_M[:,ii])
                hpd_array_L50[:,ii] = calcHPD(marginal_L[:,ii],0.75)
                hpd_array_M50[:,ii] = calcHPD(marginal_M[:,ii],0.75)

        r_script  = lib_utilities.print_R_vec("\n\nt",all_events)
        r_script += "\ntime = -t"
        r_script += lib_utilities.print_R_vec("\nspeciation",l_vec)
        if i==-1:
            r_script += lib_utilities.print_R_vec("\nsp_hdp_m",hpd_array_L[0])
            r_script += lib_utilities.print_R_vec("\nsp_hdp_M",hpd_array_L[1])
            r_script += lib_utilities.print_R_vec("\nsp_hdp_m50",hpd_array_L50[0])
            r_script += lib_utilities.print_R_vec("\nsp_hdp_M50",hpd_array_L50[1])
        r_script += lib_utilities.print_R_vec("\nextinction",m_vec)
        if i==-1:
            r_script += lib_utilities.print_R_vec("\nex_hdp_m",hpd_array_M[0])
            r_script += lib_utilities.print_R_vec("\nex_hdp_M",hpd_array_M[1])
            r_script += lib_utilities.print_R_vec("\nex_hdp_m50",hpd_array_M50[0])
            r_script += lib_utilities.print_R_vec("\nex_hdp_M50",hpd_array_M50[1])
        

        if i==-1:
            if max_T == -1:
                r_script += "\nXLIM = c(min(time[clade_1>0]),0)"
            else:
                r_script += "\nXLIM = c(%s, %s)\nclade_1[t>%s] = 0\nclade_1[t<%s] = 0 " % (-max_T, -min_T, max_T, min_T)
            
            
            r_script += """
par(mfrow=c(1,2))
YLIM = c(0,max(c(sp_hdp_M[clade_1>0],ex_hdp_M[clade_1>0])))
YLIMsmall = c(0,max(c(sp_hdp_M50[clade_1>0],ex_hdp_M50[clade_1>0])))
plot(speciation[clade_1>0] ~ time[clade_1>0],type="l",col="#4c4cec", lwd=3,main="Speciation rates - Combined effects", ylim = YLIM,xlab="Time (Ma)",ylab="Speciation rates",xlim=XLIM)
mtext("%s correlations")
polygon(c(time[clade_1>0], rev(time[clade_1>0])), c(sp_hdp_M[clade_1>0], rev(sp_hdp_m[clade_1>0])), col = alpha("#4c4cec",0.1), border = NA)    
polygon(c(time[clade_1>0], rev(time[clade_1>0])), c(sp_hdp_M50[clade_1>0], rev(sp_hdp_m50[clade_1>0])), col = alpha("#4c4cec",0.3), border = NA)    
abline(v=-c(65,200,251,367,445),lty=2,col="gray")
plot(extinction[clade_1>0] ~ time[clade_1>0],type="l",col="#e34a33",  lwd=3,main="Extinction rates - Combined effects", ylim = YLIM,xlab="Time (Ma)",ylab="Extinction rates",xlim=XLIM)
mtext("%s correlations")
polygon(c(time[clade_1>0], rev(time[clade_1>0])), c(ex_hdp_M[clade_1>0], rev(ex_hdp_m[clade_1>0])), col = alpha("#e34a33",0.1), border = NA)    
polygon(c(time[clade_1>0], rev(time[clade_1>0])), c(ex_hdp_M50[clade_1>0], rev(ex_hdp_m50[clade_1>0])), col = alpha("#e34a33",0.3), border = NA)    
abline(v=-c(65,200,251,367,445),lty=2,col="gray")
""" % (model_type,model_type)  #(fixed_focal_clade+1,fixed_focal_clade+1,fixed_focal_clade+1,fixed_focal_clade+1)
        else:
            r_script += """
par(mfrow=c(1,2))
plot(speciation[clade_1>0] ~ time[clade_1>0],type="l",col="#4c4cec", lwd=3,main="Effect of: %s", ylim =  c(0,max(c(speciation,extinction))+0.05*max(c(speciation,extinction))),xlab="Time (Ma)",ylab="Speciation and extinction rates",xlim=XLIM)
mtext("Wl = %s, Wm = %s, Gl = %s, Gm = %s")
lines(extinction[clade_1>0] ~ time[clade_1>0], col="#e34a33", lwd=3)
abline(v=-c(65,200,251,367,445),lty=2,col="gray")
plot(clade_%s[clade_1>0] ~ time[clade_1>0],type="l", main = "Trajectory of variable: %s",xlab="Time (Ma)",ylab="Rescaled value",xlim=XLIM)
abline(v=-c(65,200,251,367,445),lty=2,col="gray")
""" % (variable_names[i],round(est_kl[i],2),round(est_km[i],2),round(np.mean(np.array(Gl_temp)*scale_factor[i]),2),round(np.mean(np.array(Gm_temp)*scale_factor[i]),2),i+1,variable_names[i])
                   
        newfile.writelines(r_script)
        newfile.flush()
            

    r_script = "n<-dev.off()"
    newfile.writelines(r_script)
    newfile.close()
    print("\nAn R script with the source for the RTT plot was saved as: %sRTT.r\n(in %s)" % (name_file, wd))
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        cmd="cd %s & Rscript %s_RTT.r" % (wd,name_file)
    else: 
        cmd="cd %s; Rscript %s/%s_RTT.r" % (wd,wd,name_file)
    os.system(cmd)
    print("done\n")    
    sys.exit("\n")

##############################################################


t1=time.time()
iteration=0
while True:
    hasting=0
    gibbs_sampling=0
    if iteration==0:
        actualGarray=GarrayA*scale_factor
        likA,priorA,postA=np.zeros(n_clades),0,0
        
    l0,m0=l0A,m0A
    Garray=GarrayA
    Tau=TauA
    lik,priorBD=np.zeros(n_clades),0
    
    lik_test=np.zeros(n_clades)    
    
    if iteration==0:
        uniq_eve=np.unique(all_events,return_index=True)[1]  # indexes of unique values
        Garray_temp=Garray
        prior_r=0
        i = fixed_focal_clade
        l_at_events=trasfRate_general(l0[i],Garray_temp[i,0,:],Dtraj)
        m_at_events=trasfRate_general(m0[i],Garray_temp[i,1,:],Dtraj)
        l_s1a=l_at_events[idx_s[i]]
        m_e1a=m_at_events[idx_e[i]]
        lik[i] = (np.sum(log(l_s1a))-np.sum(abs(np.diff(all_events))*l_at_events[0:len(l_at_events)-1]*(dd_focus_clade[1:len(l_at_events)])) \
                 +np.sum(log(m_e1a))-np.sum(abs(np.diff(all_events))*m_at_events[0:len(m_at_events)-1]*(dd_focus_clade[1:len(l_at_events)])) )
        likA=lik

    else:    
        sampling_freqs=[.10,.40]        
        if iteration<1000: rr = np.random.uniform(0,sampling_freqs[1])
        else: rr = np.random.random()

        focal_clade=fixed_focal_clade
        
        if rr<sampling_freqs[0]:
            rr2 = np.random.random()
            if rr2<.5 or death_model==1: 
                l0=np.zeros(n_clades)+l0A
                l0[focal_clade],hasting=update_multiplier_proposal(l0A[focal_clade],1.2)
            else:    
                m0=np.zeros(n_clades)+m0A
                m0[focal_clade],hasting=update_multiplier_proposal(m0A[focal_clade],1.2)

        elif rr<sampling_freqs[1]: # update hypZ and hypR
            gibbs_sampling=1
            if useHSP == 1:
                if  np.random.random() > 0.15:
                    # Gibbs sampler (slice-sampling, Scott 2011)
                    LAM[focal_clade,0,:] = sample_lam_mod(LAM[focal_clade,0,:],GarrayA[focal_clade,0,:],Tau)
                    LAM[focal_clade,1,:] = sample_lam_mod(LAM[focal_clade,1,:],GarrayA[focal_clade,1,:],Tau)
                else:
                    Tau = sample_tau_mod(LAM[focal_clade,:,:],GarrayA[focal_clade,:,:],TauA)
            else:
                precision = 1./(Tau**2) # Tau is std here
                T_hp_alpha,T_hp_beta=10.,1.
                
                if birth_model:      new_precision = np.random.gamma( T_hp_alpha+(n_clades)/2, scale =1./(T_hp_beta + np.sum(GarrayA[focal_clade,0,:]**2))/2   , size=1)
                elif death_model:    new_precision = np.random.gamma( T_hp_alpha+(n_clades)/2, scale =1./(T_hp_beta + np.sum(GarrayA[focal_clade,1,:]**2))/2   , size=1)
                else:                new_precision = np.random.gamma( T_hp_alpha+(n_clades*2)/2, scale =1./(T_hp_beta + np.sum(GarrayA[focal_clade,:,:]**2))/2   , size=1)
                Tau = np.sqrt(1./new_precision)
                    
            # Gibbs sampler (Exponential + Gamma[2,2])
            G_hp_alpha,G_hp_beta=1.,.01
            #_  g_shape=G_hp_alpha+len(l0A)+len(m0A)
            #_  rate=G_hp_beta+np.sum(l0A)+np.sum(m0A)
            #_  hypRA = np.random.gamma(shape= g_shape, scale= 1./rate, size=1)
            fixed_shape = 2.
            post_rate_prm_Gamma_prior = np.random.gamma( shape=G_hp_alpha+fixed_shape*(len(l0A)+len(m0A)), scale=1./(G_hp_beta+np.sum(l0A)+np.sum(m0A)), size=1)
            hypRA = post_rate_prm_Gamma_prior
            
        else: # update Garray (effect size) 
            Garray_temp= update_parameter_normal_2d_freq((GarrayA[focal_clade,:,:]),d=.5,f=.1,m=-maxG,M=maxG)
            Garray=np.zeros((n_clades,2,n_clades))+GarrayA
            Garray[focal_clade,:,:]=Garray_temp
            if birth_model: Garray[fixed_focal_clade,1,:] *= 0
            elif death_model: Garray[fixed_focal_clade,0,:] *= 0
            
            #print GarrayA[focal_clade,:,:]-Garray[focal_clade,:,:]

        
        Garray_temp=Garray
        i=focal_clade 
        l_at_events=trasfRate_general(l0[i], Garray_temp[i,0,:],Dtraj)
        m_at_events=trasfRate_general(m0[i], Garray_temp[i,1,:],Dtraj)
        ### calc likelihood - clade i ###
        l_s1a=l_at_events[idx_s[i]]
        m_e1a=m_at_events[idx_e[i]]
        
        if max_T == -1 and min_T == -1:
            lik_clade = [np.sum(log(l_s1a))-np.sum(abs(np.diff(all_events))*l_at_events[0:len(l_at_events)-1]*(dd_focus_clade[1:len(l_at_events)])), \
                         np.sum(log(m_e1a))-np.sum(abs(np.diff(all_events))*m_at_events[0:len(m_at_events)-1]*(dd_focus_clade[1:len(l_at_events)])) ]
        else:
            lik_clade = [np.sum(log(l_s1a)[index_included_sp_times])-np.sum((abs(np.diff(all_events))*l_at_events[0:len(l_at_events)-1]*(dd_focus_clade[1:len(l_at_events)]))[index_events_included-1] ), \
                         np.sum(log(m_e1a)[index_included_ex_times])-np.sum((abs(np.diff(all_events))*m_at_events[0:len(m_at_events)-1]*(dd_focus_clade[1:len(l_at_events)]))[index_events_included-1] ) ]

        if birth_model: lik_clade = lik_clade[0]
        elif death_model: lik_clade = lik_clade[1]
        else: lik_clade = np.sum(lik_clade)
            
            
        ind_focal=np.ones(n_clades)
        ind_focal[focal_clade]=0
        lik = likA*ind_focal
        lik[focal_clade] = lik_clade
        ###### END FOCAL

    """ len(Rtemp[Rtemp==0]), where Rtemp=R[i,:,:]
    should be equal to n_clades*2 - np.sum(R[i,:,:]) and len(Rtemp[Rtemp==0]) = np.sum(R[i,:,:]
    BTW, it is n_clades*2 because the same prior is used for both l0 and m0
    
    THUS:
    
    np.sum_R_per_clade = np.sum(RA,axis=(1,2))
    log(TauA) * (1-np.sum_R_per_clade) + log(1-TauA)*(np.sum_R_per_clade))
    
    """
    #if iteration % print_freq ==0: 
    #    print pdf_normal( np.array([0,2,10,50]) ,sd=LAM[fixed_focal_clade,:,:]*Tau )
    #    print np.max(Garray[fixed_focal_clade,:,:]), np.min(Garray[fixed_focal_clade,:,:])
    #    #quit()
    prior = np.sum(pdf_normal(Garray[fixed_focal_clade,:,:],sd=LAM[fixed_focal_clade,:,:]*Tau ))
    if useHSP==1:
        prior +=np.sum(pdf_cauchy(LAM[fixed_focal_clade,:,:]))
        prior +=np.sum(pdf_cauchy(Tau))    
    else:
        T_hp_alpha,T_hp_beta=10.,1.
        prior += prior_gamma(1./(Tau**2),T_hp_alpha,T_hp_beta)
    #_ prior += prior_exponential(l0,hypRA)+prior_exponential(m0,hypRA)
    fixed_shape = 2.
    prior += prior_gamma(l0,fixed_shape,hypRA)+prior_gamma(m0,fixed_shape,hypRA)
    
    if (np.sum(lik) + prior) - postA + hasting >= log(np.random.random()) or iteration==0 or gibbs_sampling==1:
        postA=np.sum(lik)+prior
        likA=lik
        priorA=prior
        l0A=l0
        m0A=m0
        GarrayA=Garray
        actualGarray=GarrayA[fixed_focal_clade,:,:]*scale_factor
        TauA=Tau
        #hypRA=hypR
    
    if iteration % print_freq ==0: 
        k= 1./(1+TauA**2 * LAM[fixed_focal_clade,:,:]**2) # Carvalho 2010 Biometrika, p. 471
        loc_shrinkage = (1-k) # so if loc_shrinkage > 0 is signal, otherwise it's noise (cf. Carvalho 2010 Biometrika, p. 474)
        print(iteration, array([postA]), TauA, mean(LAM[fixed_focal_clade,:,:]), len(loc_shrinkage[loc_shrinkage>0.5])) #, np.sum(likA),np.sum(lik),prior, hasting
        #print likA
        #print "l:",l0A
        #print "m:", m0A
        #print "G:", actualGarray.flatten()
        #print "R:", RA.flatten()
        #print "Gr:", GarrayA.flatten()
        #print "Hmu:", TauA, 1./hypRA[0] #,1./hypRA[1],hypRA[2]
    if iteration % sampling_freq ==0:
        k= 1./(1+TauA**2 * LAM[fixed_focal_clade,:,:]**2) # Carvalho 2010 Biometrika, p. 471
        loc_shrinkage = (1-k) # so if loc_shrinkage > 0 is signal, otherwise it's noise (cf. Carvalho 2010 Biometrika, p. 474)
        #loc_shrinkage =LAM[fixed_focal_clade,:,:]**2
        log_state=[iteration,postA,np.sum(likA)]+[priorA]+[l0A[fixed_focal_clade]]+[m0A[fixed_focal_clade]]+list(actualGarray.flatten())+list(loc_shrinkage.flatten())+[mean(LAM[fixed_focal_clade,:,:]),std(LAM[fixed_focal_clade,:,:])] +list(TauA) +[hypRA[0]]
        wlog.writerow(log_state)
        logfile.flush()

    iteration+=1
    if iteration ==n_iterations: break    

print(time.time()-t1)
quit()










