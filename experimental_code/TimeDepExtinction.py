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


#OH# leaving the function below as a reference but changed the calls below to the "new" BDwelik ('w' stands for weibull and 'e' extinction)
# BD likelihood (constant rate)
def BDlik (l, m):
	sp_events = np.ones(len(s))  # define speciation events
	ex_events = np.zeros(len(e)) # define extinction events
	ex_events[e>0] = 1           # ex_events = 0 for extant taxa, ex_events=1 for extinct taxa
	birth_lik = log(l)*sp_events - l*(s-e) # vector likelihoods for each species
	death_lik = log(m)*ex_events - m*(s-e) #OH# why do you still have the second part for extant taxa here?
	species_lik = birth_lik + death_lik
	lik = sum(species_lik)
	return lik

#OH# Weibull log PDF and CDF functions
def log_wei_pdf(x,scale,shape):
	# Log of Weibull pdf
	log_wei_pdf = log(shape/scale) + (k-1)*log(x/scale) - (x/scale)**shape
	return log_wei_pdf

def wei_cdf(x,scale,shape):
	# Weibull cdf
	log_wei_cdf = 1 - exp(-(x/scale)**shape)
	return wei_cdf
	

#OH# BDwe likelihood (constant speciation rate and age dependent weibull extinction)
def BDwelik (l, m, shape, scale):
	sp_events = np.ones(len(s))  # define speciation events
	ex_events = np.zeros(len(e)) # define extinction events
	ex_events[e>0] = 1           # ex_events = 0 for extant taxa, ex_events=1 for extinct taxa
	birth_lik = log(l)*sp_events - l*(s-e) # vector likelihoods for each species
	#OH# now following the log of PDF for Weibull, when x>=0, which is our case...
	death_lik = log(m)*ex_events + log_wei_pdf(e,scale,shape) - m*wei_cdf(s-e,scale,shape)
	species_lik = birth_lik + death_lik
	lik = sum(species_lik)
	return lik

# prior
#OH# should this also be changed?
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
head = ["it","post","lik","prior","l","m","shape","scale"]
wlog.writerow(head)
logfile.flush()


iteration =0
sampling_freq =10
# init parameters
lA = 0.5
mA = 0.1
shapeA = 0.1 #OH# proposition of parameter, here starting with strong age-dependency, mainly high likelyhood for younger species
scaleA = 0.1 #OH# proposition of parameter

while True:
	# update parameters
	if np.random.random() >0.5:
		l, hastings = update_multiplier_proposal(lA)
		m = mA
		shape = shapeA
		scale = scaleA
	else:
		m, hastings = update_multiplier_proposal(mA)
		shape, hastings = update_multiplier_proposal(shapeA)
		scale, hastings = update_multiplier_proposal(scaleA)
		l = lA
	
	
	# calc lik
	lik = BDwelik(l, m, shape, scale)
	
	# calc priors
	prior = prior_gamma(l) + prior_gamma(m) + prior_gamma(shape) + prior_gamma(scale)
		
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
		shapeA = shape
		scaleA = scale
	
	if iteration % 100 ==0:
		print likA, priorA, lA, mA, shapeA, scaleA
		
	if iteration % sampling_freq ==0:
		log_state=[iteration,likA+priorA,likA,priorA,lA,mA,shapeA,scaleA]
		wlog.writerow(log_state)
		logfile.flush()
		
		
	
	iteration +=1
	if iteration==10000: break
	
