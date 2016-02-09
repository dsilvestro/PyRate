#!/usr/bin/env python 
import argparse, os,sys
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import platform, time
import csv
from scipy.special import gdtr, gdtrix
from scipy.special import betainc
import scipy.stats
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
self_path=os.getcwd()


# make up data
s = np.random.uniform(1, 25, 10) # random speciation times
e = np.random.uniform(0, s,  10) # random extinction times



# BD likelihood (constant rate)
def BDlik (l, m):
	sp_events = np.ones(len(s))  # define speciation events
	ex_events = np.zeros(len(e)) # define extinction events
	ex_events[e>0] = 1           # ex_events = 0 for extant taxa, ex_events=1 for extinct taxa
	birth_lik = log(l)*sp_events - l*(s-e) # vector likelihoods for each species
	death_lik = log(m)*ex_events - m*(s-e)
	species_lik = birth_lik + death_lik
	lik = sum(species_lik)
	return lik


# prior
def prior_gamma(L,a=2,b=2): return sum(scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0))


# function to update parameters
def update_multiplier_proposal(i,d=1.2,f=1):
	S=shape(i)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
 	ii = i * m
	U=sum(log(m))
	return ii, U



# create log file
logfile = open("mcmc.log" , "wb") 
wlog=csv.writer(logfile, delimiter='\t')

head = ["it","post","lik","prior","l","m"]
wlog.writerow(head)
logfile.flush()









iteration =0
sampling_freq =10
# init parameters
lA = 0.5
mA = 0.1

while True:
	# update parameters
	if np.random.random() >0.5:
		l, hastings = update_multiplier_proposal(lA)
		m = mA
	else:
		m, hastings = update_multiplier_proposal(mA)
		l = lA
	
	
	# calc lik
	lik = BDlik(l, m)
	
	# calc priors
	prior = prior_gamma(l) + prior_gamma(m)
		
	if iteration ==0:  
		likA = lik
		priorA = prior
	
	
	posterior_ratio = (prior+lik) - (priorA+likA)
	
	if posterior_ratio + hastings > log(np.random.random()):
		# accept state
		likA = lik
		priorA = prior
		lA = l
		mA = m
	
	if iteration % 100 ==0:
		print likA, priorA, lA, mA
		
		
	if iteration % sampling_freq ==0:
		log_state=[iteration,likA+priorA,likA,priorA,lA,mA]
		wlog.writerow(log_state)
		logfile.flush()
		
		
	
	iteration +=1
	if iteration==10000: break
	
	
	








	