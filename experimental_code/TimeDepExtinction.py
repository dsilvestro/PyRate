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



#OH# Weibull log PDF and CDF functions
def log_wei_pdf(x,W_scale,W_shape):
	# Log of Weibull pdf
	log_wei_pdf = log(W_shape/W_scale) + (k-1)*log(x/W_scale) - (x/W_scale)**W_shape
	return log_wei_pdf

def wei_cdf(x,W_scale,W_shape):
	# Weibull cdf
	log_wei_cdf = 1 - exp(-(x/W_scale)**W_shape)
	return wei_cdf
	
#OH# BDwe likelihood (constant speciation rate and age dependent weibull extinction)
def BDwelik (l, m0, W_shape, W_scale):
	d = s-e
	#sp_events = np.ones(len(s))  # define speciation events
	#ex_events = np.zeros(len(e)) # define extinction events
	ex_events[e>0] = 1           # ex_events = 0 for extant taxa, ex_events=1 for extinct taxa
	birth_lik = len(s)*log(l)-sum(l*(d)) # vector likelihoods for each species
	death_lik_de = sum(log(m0)+log_pdf_Weibull(e[e>0], W_shape, W_scale))
	dead_lik_wte = -sum(m0*wei_cdf(d,W_scale,W_shape))
	lik = birth_lik + death_lik_de + death_lik_wte
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
W_shapeA = 0.1 #OH# proposition of parameter, here starting with strong age-dependency, mainly high likelyhood for younger species
W_scaleA = 0.1 #OH# proposition of parameter

while True:
	# update parameters
	if np.random.random() >0.5:
		l, hastings = update_multiplier_proposal(lA)
		m = mA
		W_shape = W_shapeA
		W_scale = W_scaleA
	else:
		m, hastings = update_multiplier_proposal(mA)
		W_shape, hastings = update_multiplier_proposal(W_shapeA)
		W_scale, hastings = update_multiplier_proposal(W_scaleA)
		l = lA
	
	
	# calc lik
	lik = BDwelik(l, m, W_shape, W_scale)
	
	# calc priors
	prior = prior_gamma(l) + prior_gamma(m) + prior_gamma(W_shape) + prior_gamma(W_scale)
		
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
		W_shapeA = W_shape
		W_scaleA = W_scale
	
	if iteration % 100 ==0:
		print likA, priorA, lA, mA, W_shapeA, W_scaleA
		
	if iteration % sampling_freq ==0:
		log_state=[iteration,likA+priorA,likA,priorA,lA,mA,W_shapeA,W_scaleA]
		wlog.writerow(log_state)
		logfile.flush()
		
		
	
	iteration +=1
	if iteration==10000: break
	
