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


#import simulated TAB separated data
relative_dropbox_folder=r"C:\Users\oskar\Documents\Dropbox" #Oskar  #this is mine, create yours and comment mine while testing...
# relative_dropbox_folder='HERE YOUR REL PATH'
tbl = np.loadtxt(fname=relative_dropbox_folder+r"\PyRate_Age-Dependency_and_Beyond\Toy_Datasets_TreeSimGM\Output_ToyTrees2\age10rexp1rweibull0.5_1.txt", skiprows=1)
s = tbl[:,2]
e = tbl[:,3]


# Aged dependet rate leading to a Weibull waiting time
def wr(t,W_scale,W_shape):
	# rate
	wr=(W_shape/W_scale)*(t/W_scale)**(W_shape-1)
	return wr


# Log of aged dependet rate leading to a Weibull waiting time
def log_wr(t,W_scale,W_shape):
	# rate
	log_wr=log(W_shape/W_scale)+(W_shape-1)*log(t/W_scale)
	return log_wr
	
# Integral of  wr	
def wr_int(startingx, endingx, numberofRectangles):
	width = (float(endingx)-float(startingx))/numberofRectangles
	runningSum = 0
	for i in range(numberofRectangles):
		height = wr(startingx + i*width)
		area = height * width
		runningSum += area
	return runningSum
	
	
#OH# BDwe likelihood (constant speciation rate and age dependent with weibull waiting time until extinction)
def BDwelik (l, m0, W_shape, W_scale):
	d = s-e
	birth_lik = len(s)*log(l)-l*sum(d) # log probability of speciation
	death_lik_de = sum(log(m0)+log_pdf_Weibull(e[e>0], W_shape, W_scale)) # log probability of death event
	death_lik_wte = -sum(m0*cdf_Weibull(d,W_scale,W_shape)) # log probability of waiting time until death event
	lik = birth_lik + death_lik_de + death_lik_wte
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
head = ["it","post","lik","prior","l","m","shape","scale"]
wlog.writerow(head)
logfile.flush()


iteration =0
sampling_freq =10
n_iterations = 1000
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
	if iteration==n_iterations: break
