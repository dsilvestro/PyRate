#!/usr/bin/env python 
import argparse, os,sys
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import random as rand
import sys, platform, time
import multiprocessing, thread
import multiprocessing.pool
import os, csv
from scipy.special import gdtr, gdtrix
from scipy.special import betainc
import scipy.stats
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
from multiprocessing import Pool, freeze_support
import thread
from lib_updates_priors import *
from lib_DD_likelihood import *

#### DATA ###
def return_float(v):
	new_v=list()
	for i in v:
		try: new_v.append(float(i))
		except: pass
	return np.array(new_v)

p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('-d', type=str, help='data set', default=0, metavar=0)
p.add_argument('-c', type=int, help='clade analyzed', default=0, metavar=0)
p.add_argument('-j', type=int, help='replicate', default=0, metavar=0)
args = p.parse_args()


dataset=args.d
focus_clade=args.c
#t_file=np.genfromtxt(dataset, names=True, delimiter='\t', dtype=float)
t_file=np.loadtxt(dataset, skiprows=1)

clade_ID=t_file[:,0]
clade_ID=clade_ID.astype(int)
ts=t_file[:,2+2*args.j]
te=t_file[:,3+2*args.j]

tempfile=loadtxt("tempPyRate.txt")
times_of_T_change= tempfile[:,0] # array of times of Temp change
Temp_values=       tempfile[:,1] # array of Temp values at times_of_T_change


times_of_T_change_tste=sort(np.concatenate((times_of_T_change,ts[clade_ID==focus_clade],te[clade_ID==focus_clade]),axis=0))[::-1]

all_events=times_of_T_change_tste # events are speciation/extinction that change the diversity trajectory
n_clades,n_events=max(clade_ID)+1,len(all_events)
Dtraj=init_Dtraj(n_clades,n_events)

for i in range(n_clades): # make trajectory curves for each clade
	Dtraj[:,i]=getDT(all_events,ts[clade_ID==i],te[clade_ID==i])

def get_Temp_at_time(all_Times,Temp_values):
	temperatures=list()
	for t in all_Times:
		if t in times_of_T_change: # t is a time of change i.e. not ts,te
			ind=np.where(times_of_T_change-t==0)[0]
			temperatures.append(mean(Temp_values[ind]))
		else: temperatures.append(mean(Temp_values[ind]))
	return array(temperatures)


Temp_at_events= get_Temp_at_time(times_of_T_change_tste,Temp_values)

GarrayA=np.zeros(2) # correlation parameters with Temp of lambda and mu, respectively

l0A,m0A=init_BD(1),init_BD(1)

out_file_name="%s_clade%s_%s_Temp.log" % (dataset,args.c,args.j)
logfile = open(out_file_name , "wb") 
wlog=csv.writer(logfile, delimiter='\t')

head="it\tposterior\tlikelihood\tprior" 
head+="\tl%s" % (focus_clade)
head+="\tm%s" % (focus_clade)
head+="\tGl"
head+="\tGm"
wlog.writerow(head.split('\t'))
logfile.flush()


for iteration in range(1250000):	
	hasting=0
	l0,m0=l0A,m0A
	Garray=GarrayA
	if iteration==0:
		likA,priorA,postA=np.zeros(n_clades),0,0
	lik,priorBD=0,0
		
	# update values
	if rand.random()<.5 or iteration<1000:
		l0=update_positive_rate_vec(l0A,.5) #np.array([ .15,.075,.15]))
		m0=update_positive_rate_vec(m0A,.5) #np.array([.075,.075,.15]))
	else:
		Garray=update_parameter_normal_2d_freq(GarrayA,.5,m=-10,M=10) 

	uniq_eve=np.unique(all_events,return_index=True)[1]  # indexes of unique values
	l_at_events=trasfRateTemp(l0, Garray[0],Temp_at_events)
	m_at_events=trasfRateTemp(m0, Garray[1],Temp_at_events)
 	s1,e1=ts[clade_ID==focus_clade],te[clade_ID==focus_clade]	
	ind_s1=np.nonzero(np.in1d(all_events,s1))[0]         # indexes of s1 in all_events
	ind_s =np.intersect1d(ind_s1,uniq_eve)               
	ind_e1=np.nonzero(np.in1d(all_events,e1))[0]
	ind_e =np.intersect1d(ind_e1,uniq_eve)
	l_s1a=l_at_events[ind_s]
	m_e1a=m_at_events[ind_e]
	
	lik =  sum(log(l_s1a))-sum( abs(np.diff(all_events))*l_at_events[0:len(l_at_events)-1]*(Dtraj[:,focus_clade][1:len(l_at_events)])) \
	      +sum(log(m_e1a))-sum( abs(np.diff(all_events))*m_at_events[0:len(m_at_events)-1]*(Dtraj[:,focus_clade][1:len(l_at_events)])) 
        	
	prior= prior_normal(Garray,scale=2) +prior_gamma(l0,1.1,.5)+prior_gamma(m0,1.1,.5)  
	
	if (lik + prior + hasting) - postA >= log(rand.random()) or iteration==0:
		postA=lik+prior+hasting
		likA=lik
		priorA=prior
		l0A=l0
                m0A=m0
		GarrayA=Garray
	if iteration % 1000 ==0: 
		print iteration, array([postA, likA,lik,prior]), hasting
		print "l:",l0A
		print "m:", m0A
		print "G:", GarrayA
	if iteration % 1000 ==0:
		log_state=[iteration,postA,likA,priorA, l0A[0], m0A[0]] +list(GarrayA)
		wlog.writerow(log_state)
		logfile.flush()


quit()












