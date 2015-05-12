#!/usr/bin/env python 
import argparse, os,sys
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import random as rand
import platform, time
import multiprocessing, thread
import multiprocessing.pool
import csv
from scipy.special import gdtr, gdtrix
from scipy.special import betainc
import scipy.stats
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
from multiprocessing import Pool, freeze_support
import thread
import imp
lib_updates_priors = imp.load_source("lib_updates_priors", "pyrate_lib/lib_updates_priors.py")
lib_DD_likelihood = imp.load_source("lib_DD_likelihood", "pyrate_lib/lib_DD_likelihood.py")
lib_utilities = imp.load_source("lib_utilities", "pyrate_lib/lib_utilities.py")
from lib_updates_priors import *
from lib_DD_likelihood  import *

#### ARGS
p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('-A', type=int, help='algorithm: "0" parameter estimation, "1" TI', default=0, metavar=0) # 0: par estimation, 1: TI
p.add_argument('-d', type=str, help='data set', default="", metavar="<file>")
p.add_argument('-c', type=str, help='covariate data set', default="", metavar="<file>")
p.add_argument('-j', type=int, help='replicate', default=0, metavar=0)
p.add_argument('-m', type=int, help='model: "-1" constant rate, "0" exponential, "1" linear', default=0, metavar=0)
p.add_argument('-n', type=int, help='mcmc generations',default=1050000, metavar=1050000)
p.add_argument('-s', type=int, help='sample freq.', default=1000, metavar=1000)
p.add_argument('-p', type=int, help='print freq.', default=1000, metavar=1000)
p.add_argument('-r', type=float, help='rescale values (0 to scale in [0,1], 0.1 to reduce range 10x, 1 to leave unchanged)', default=0, metavar=0)
p.add_argument('-clade', type=int, help='clade analyzed', default=0, metavar=0)
p.add_argument('-ginput', type=str,help='generate input file from *mcmc.log', default="", metavar="<path_to_mcmc.log>")
p.add_argument('-b', type=float, help='burnin in *mcmc.log to generate input file', default=0.1, metavar=0.1)
p.add_argument('-w',  type=float, help='window sizes (bd rates, G)',  default=[1.4, .05], metavar=1.4, nargs=2)

args = p.parse_args()

mcmc_gen = args.n
sampling_freq = args.s
print_freq = args.p
dataset=args.d
cov_file=args.c
rescale_factor=args.r
focus_clade=args.clade
win_size=args.w

if args.ginput != "":
	lib_utilities.write_ts_te_table(args.ginput, clade=focus_clade,burnin=args.b)
	quit()


#t_file=np.genfromtxt(dataset, names=True, delimiter='\t', dtype=float)
t_file=np.loadtxt(dataset, skiprows=1)

ts=t_file[:,2+2*args.j]
te=t_file[:,3+2*args.j]

# assign short branch length to singletons (ts=te)
ind_singletons=(ts==te).nonzero()[0]
z=np.zeros(len(ts))
z[ind_singletons] =0.1
ts =ts+z
# if more than one clade only one is analyzed (flag -clade)
clade_ID=t_file[:,0].astype(int)
ts,te=ts[clade_ID==focus_clade],te[clade_ID==focus_clade]	


###____ TMP
#	r_times= np.load("rand_times.npy")
#	add_rand_ts = r_times[:,0+2*args.j]
#	add_rand_te = r_times[:,1+2*args.j]
#	ind_singletons=(ts==te).nonzero()[0]
#	for i in ind_singletons:
#		M = max(add_rand_ts[i],add_rand_te[i])
#		m = min(add_rand_ts[i],add_rand_te[i])
#		add_rand_ts[i] = M
#	        add_rand_te[i] = m
#	
#	add_rand_ts=add_rand_ts[0:len(ts)]
#	add_rand_te=add_rand_te[0:len(te)]
#	
#	ts += add_rand_ts
#	te[te>0] += add_rand_te[te>0]
###____ TMP




print len(ts),len(te[te>0]),sum(ts-te)

tempfile=loadtxt(cov_file,skiprows=1)
head_cov_file = next(open(cov_file)).split()
times_of_T_change= tempfile[:,0] # array of times of Temp change
Temp_values=       tempfile[:,1] # array of Temp values at times_of_T_change

Temp_values= (Temp_values-Temp_values[0]) # so l0 and m0 are rates at the present
if rescale_factor > 0: Temp_values = Temp_values*rescale_factor
else: Temp_values = Temp_values/(max(Temp_values)-min(Temp_values))

#print "BRL" , sum(ts-te)
#print "range:", max(Temp_values)-min(Temp_values)

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

#print "TIME", max(times_of_T_change), max(ts),Temp_values[-1]

def get_Temp_at_time(all_Times,Temp_values):
	temperatures,ind=list(),list()
	for t in all_Times:
		if t in times_of_T_change: # t is a time of change i.e. not ts,te
			ind=np.where(times_of_T_change-t==0)[0]
			temperatures.append(mean(Temp_values[ind]))
		elif max(ts) > max(times_of_T_change): temperatures.append(Temp_values[-1]) # if root older than oldest temp take first temp
		else: temperatures.append(mean(Temp_values[ind]))
	return array(temperatures)


Temp_at_events= get_Temp_at_time(times_of_T_change_tste,Temp_values)
#_print ind_s[3566:]
#_print Temp_at_events[1000:1010]
#_print "HERE",len(ind_s),len(ind_e)


#for i in range(len(all_events)):
#	print all_events[i],Temp_at_events[i]

GarrayA=np.zeros(2) # correlation parameters with Temp of lambda and mu, respectively

l0A,m0A= init_BD(1),init_BD(1)

if args.m== -1: out_file_name="%s/%s_%s_%s_const.log"  % (os.path.dirname(dataset),os.path.splitext(os.path.basename(dataset))[0],head_cov_file[1],args.j)
if args.m==  0: out_file_name="%s/%s_%s_%s_exp.log"    % (os.path.dirname(dataset),os.path.splitext(os.path.basename(dataset))[0],head_cov_file[1],args.j)
if args.m==  1: out_file_name="%s/%s_%s_%s_linear.log" % (os.path.dirname(dataset),os.path.splitext(os.path.basename(dataset))[0],head_cov_file[1],args.j)



logfile = open(out_file_name , "wb") 
wlog=csv.writer(logfile, delimiter='\t')

head="it\tposterior\tlikelihood\tprior" 
head+="\tl0" 
head+="\tm0" 
head+="\tGl"
head+="\tGm"
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

scal_fac_ind=0
for iteration in range(mcmc_gen * len(scal_fac_TI)):	
	
	if (iteration+1) % (mcmc_gen+1) ==0: 
		print iteration, mcmc_gen  
		scal_fac_ind+=1
	
	hasting=0
	l0,m0=l0A,m0A
	Garray=GarrayA
	if iteration==0:
		likA,priorA,postA=0,0,0
	lik,priorBD=0,0
	
	# update values
	rr=np.random.uniform(0,1,3)
	if args.m== -1: rr[0]=0 # never update Garray
	
	if iteration>10:
		if rr[0]<.25 or iteration<1000:
			if rr[1]>.5: 
				l0=np.zeros(1)+l0A
				l0,U=update_multiplier_proposal(l0A,d1)
			else: 	
				m0=np.zeros(1)+m0A
				m0,U=update_multiplier_proposal(m0A,d1)
			hasting=U
		else:
			if rr[2]>.5:
				Garray=GarrayA+np.zeros(2)
				Garray[0]=update_parameter_normal(Garray[0],list_d2[scal_fac_ind]) 
			else:
				Garray=GarrayA+np.zeros(2)
				Garray[1]=update_parameter_normal(Garray[1],list_d2[scal_fac_ind]) 
			
	if args.m==0: 
		l_at_events=trasfRateTemp(l0, Garray[0],Temp_at_events)
		m_at_events=trasfRateTemp(m0, Garray[1],Temp_at_events)
	if args.m==1: 
		l_at_events=trasfRateTempLinear(l0, Garray[0],Temp_at_events)
		m_at_events=trasfRateTempLinear(m0, Garray[1],Temp_at_events)
	if args.m== -1: 
		l_at_events=np.repeat(l0,len(Temp_at_events))
		m_at_events=np.repeat(m0,len(Temp_at_events))
		
	l_s1a=l_at_events[ind_s]
	m_e1a=m_at_events[ind_e]
	
	lik =  sum(log(l_s1a))-sum( abs(np.diff(all_events))*l_at_events[0:len(l_at_events)-1]*(Dtraj[:,0][1:len(l_at_events)])) \
	      +sum(log(m_e1a))-sum( abs(np.diff(all_events))*m_at_events[0:len(m_at_events)-1]*(Dtraj[:,0][1:len(l_at_events)])) 
	
	lik_alter = lik * scal_fac_TI[scal_fac_ind]
	
	prior= prior_normal(Garray,scale=2) +prior_gamma(l0,1.1,.5)+prior_gamma(m0,1.1,.5)  
	
	if (lik_alter + prior + hasting) - postA >= log(rand.random()) or iteration==0:
		postA=lik_alter+prior
		likA=lik
		priorA=prior
		l0A=l0
                m0A=m0
		GarrayA=Garray
	if iteration % print_freq ==0: 
		print iteration, array([postA, likA,lik,prior]), hasting, scal_fac_TI[scal_fac_ind]
		print "l:",l0A, "m:", m0A, "G:", GarrayA
	if iteration % sampling_freq ==0:
		log_state=[iteration,postA,likA,priorA, l0A[0], m0A[0]] +list(GarrayA) + [scal_fac_TI[scal_fac_ind]]
		wlog.writerow(log_state)
		logfile.flush()


quit()












