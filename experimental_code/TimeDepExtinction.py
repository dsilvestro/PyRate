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


#OH# leaving the function below as a ref. but changed the calls below to the "new" BDwelik ('w' stands for weibull and 'e' extinction)
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

# BDwe likelihood (constant speciation rate and age dependent weibull extinction)
def BDwelik (l, shape, scale):
	sp_events = np.ones(len(s))  # define speciation events
	ex_events = np.zeros(len(e)) # define extinction events
	ex_events[e>0] = 1           # ex_events = 0 for extant taxa, ex_events=1 for extinct taxa
	birth_lik = log(l)*sp_events - l*(s-e) # vector likelihoods for each species
	#OH# now following the log of PDF for Weibull, when x>=0, which is our case...
	death_lik = log((shape/scale)*(((s-e)/scale)**(shape-1)))*ex_events - (((s-e)/scale)**shape)
	#OH# Daniele, I am not sure if I am doing this right... I am just guessing from what I studied from your code.... maybe there is
	#also a way to do this without basic arithmetic operators? I am doing weibull first so that you can see if you think I did It
	#correcly before I start changing much (p.s. I preserved the BDlik above)
	#also I am not use of multipying the first part of the operation by the ex_events... I see the point of excluding this first part 
	#for the extant taxa on the exponential case.. but here?
	#sorry to bombard with comments, you can reply by e-mail. I am not sure if I am confusing PDF with CDF here too
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

#OH# head = ["it","post","lik","prior","l","m"]
head = ["it","post","lik","prior","l","shape", "scale"]
wlog.writerow(head)
logfile.flush()









iteration =0
sampling_freq =10
# init parameters
lA = 0.5
#OH# mA = 0.1
shapeA = 0.1 #OH# proposition of parameter, here starting with strong age-dependency, mainly high likelyhood for younger species
scaleA = 0.1 #OH# proposition of parameter

while True:
	# update parameters
	if np.random.random() >0.5:
		l, hastings = update_multiplier_proposal(lA)
		#OH# m = mA
		shape = shapeA
		scale = scaleA
	else:
		#OH# m, hastings = update_multiplier_proposal(mA)
		shape, hastings = update_multiplier_proposal(shapeA)
		scale, hastings = update_multiplier_proposal(scaleA)
		l = lA
	
	
	# calc lik
	#OH# lik = BDlik(l, m)
	lik = BDwelik(l, shape, scale)
	
	# calc priors
	#OH# prior = prior_gamma(l) + prior_gamma(m)
	prior = prior_gamma(l) + prior_gamma(shape) + prior_gamma(scale)
		
	if iteration ==0:  
		likA = lik
		priorA = prior
	
	
	posterior_ratio = (prior+lik) - (priorA+likA)
	
	if posterior_ratio + hastings > log(np.random.random()):
		# accept state
		likA = lik
		priorA = prior
		lA = l
		#OH# mA = m
		shapeA = shape
		scaleA = scale
	
	if iteration % 100 ==0:
		#OH# print likA, priorA, lA, mA
		print likA, priorA, lA, shapeA, scaleA
		
	if iteration % sampling_freq ==0:
		#OH# log_state=[iteration,likA+priorA,likA,priorA,lA,mA]
		log_state=[iteration,likA+priorA,likA,priorA,lA,shapeA,scaleA]
		wlog.writerow(log_state)
		logfile.flush()
		
		
	
	iteration +=1
	if iteration==10000: break
	
	
	








	
