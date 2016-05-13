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

self_path=os.getcwd()
p = argparse.ArgumentParser() #description='<input file>') 

p.add_argument('-v',         action='version', version='%(prog)s')
p.add_argument('-d',         type=str,help="Load SE table",metavar='<1 input file>',default="")

args = p.parse_args()

output_wd = os.path.dirname(args.d)
if output_wd=="": output_wd= self_path

#import simulated TAB separated data
#relative_dropbox_folder=r"C:\Users\oskar\Documents\Dropbox" #Oskar  #this is mine, create yours and comment mine while testing...
# relative_dropbox_folder='HERE YOUR REL PATH'
tbl = np.loadtxt(fname=args.d, skiprows=1)
s = tbl[:,2]
e = tbl[:,3]


# Aged dependet rate leading to a Weibull waiting time
def wr(t,W_shape,W_scale):
	# rate
	wr=(W_shape/W_scale)*(t/W_scale)**(W_shape-1)
	return wr


# Log of aged dependet rate leading to a Weibull waiting time
def log_wr(t,W_shape,W_scale):
	# rate
	log_wr=log(W_shape/W_scale)+(W_shape-1)*log(t/W_scale)
	return log_wr
	
# Integral of  wr	
def wr_int(startingx, endingx, W_shape, W_scale, numberofRectangles=1000):
	print startingx,endingx,W_shape,W_scale
	quit()
	width = (endingx-startingx)/float(numberofRectangles)
	runningSum = 0
	for i in range(numberofRectangles):
		height = wr(startingx + i*width, W_shape, W_scale)
		area = height * width
		runningSum += area
	return runningSum

# Integrating wr without for loop	
def pdf_WR(arg,x):
	W_shape = arg[0]
	W_scale = arg[1]
	return (W_shape/W_scale)*(x/W_scale)**(W_shape-1)

def int_function(function, arg_function, starting_x, ending_x, n_bins=1000.):
	INT = np.zeros(len(ending_x))
	for i in range(len(ending_x)):
		v= np.linspace(starting_x,ending_x[i],n_bins)
		INT[i] = sum(pdf_WR(arg_function,v))*(v[1]-v[0])
	#V = np.repeat(v,len(ending_x)).reshape(len(v),len(ending_x))
	return INT

### Integration using Trapezoidal rule | incorporate in the code!
def int_trapez_function(func, arg_function, starting_x, ending_x, n_bins=10):
	v= np.linspace(starting_x,ending_x,n_bins)
	y= func(arg_function,v)
	d= v[1]-v[0]
	return sum((np.diff(y)*d)/2. + (y[0:-1]*d) )

	
# BDwwte likelihood (constant speciation rate and age dependent with weibull waiting time until extinction)
def BDwwte (l, m0, W_shape, W_scale):
	d = s-e
	de = d[e>0] #takes only the extinct species times
	birth_lik = len(s)*log(l)-l*sum(d) # log probability of speciation
	death_lik_de = sum(log(m0)+log_wr(de, W_shape, W_scale)) # log probability of death event
	#death_lik_wte = sum(-m0*wr_int(0,d,W_shape,W_scale)) # log probability of waiting time until death event
	#print len(s)*log(l)-l*sum(d), log(m0)+log_wr(de, W_shape, W_scale), -m0*wr_int(0,d,W_shape,W_scale)
	death_lik_wte = sum(-m0*int_function(pdf_WR, [W_shape,W_scale], 0.001, d)) # log probability of waiting time until death event
	lik = birth_lik + death_lik_de + death_lik_wte
	return lik

# prior
def prior_gamma(L,a=2,b=2): return sum(scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0))


# function to update parameters
def update_multiplier_proposal(i,d=1.1,f=1):
	S=shape(i)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
 	ii = i * m
	U=sum(log(m))
	return ii, U



# create log file
input_file_raw = os.path.basename(args.d)
input_file = os.path.splitext(input_file_raw)[0]  # file name without extension
log_file_name = "%s/%s_mcmc.log" % (output_wd,input_file)
logfile = open(log_file_name , "wb") 
wlog=csv.writer(logfile, delimiter='\t')
head = ["it","post","lik","prior","l","m","shape","scale"]
wlog.writerow(head)
logfile.flush()


iteration =0
sampling_freq =10
n_iterations = 10000
# init parameters
lA = 0.5
mA = 1
W_shapeA = 0.5 #OH# proposition of parameter, here starting with strong age-dependency, mainly high likelyhood for younger species
W_scaleA = 1.5 #OH# proposition of parameter

while True:
	# update parameters
	if np.random.random() >0.5:
		l, hastings = update_multiplier_proposal(lA)
		m = mA
		W_shape = W_shapeA
		W_scale = W_scaleA
	else:
		#m, hastings = update_multiplier_proposal(mA)
		m = mA
		W_shape, hastings = update_multiplier_proposal(W_shapeA)
		W_scale, hastings = update_multiplier_proposal(W_scaleA)
		l = lA
	
	
	# calc lik
	lik = BDwwte(l, m, W_shape, W_scale)
	
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




