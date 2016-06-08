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
try: 
	self_path= os.path.dirname(sys.argv[0])
	lib_updates_priors = imp.load_source("lib_updates_priors", "%s/pyrate_lib/lib_updates_priors.py" % (self_path))
	lib_DD_likelihood = imp.load_source("lib_DD_likelihood", "%s/pyrate_lib/lib_DD_likelihood.py" % (self_path))
except:
	self_path=os.getcwd()
	lib_updates_priors = imp.load_source("lib_updates_priors", "%s/pyrate_lib/lib_updates_priors.py" % (self_path))
	lib_DD_likelihood = imp.load_source("lib_DD_likelihood", "%s/pyrate_lib/lib_DD_likelihood.py" % (self_path))

from lib_updates_priors import *
from lib_DD_likelihood  import *


#### DATA ###

p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('-d', type=str,   help='data set', default=0, metavar=0)
p.add_argument('-m', type=int,   help='model', default=0, metavar=0)
p.add_argument('-n', type=int,   help='MCMC iterations', default=1000000, metavar=1000000)
p.add_argument('-s', type=int,   help='sampling freq.', default=1000, metavar=1000)
p.add_argument('-p', type=int,   help='print freq.', default=5000, metavar=5000)
p.add_argument('-j', type=int,   help='replicate', default=0, metavar=0)
p.add_argument('-c', type=int, help='clade (if -1 joint analysis of all clades; first clade is number 0)', default=-1, metavar=-1)
p.add_argument('-beta', type=float, help='shape parameter (beta) of Be hyper=prior pn indicators', default=1, metavar=1)


args = p.parse_args()


dataset=args.d
n_iterations=args.n
sampling_freq=args.s
print_freq = args.p
#t_file=np.genfromtxt(dataset, names=True, delimiter='\t', dtype=float)
t_file=np.loadtxt(dataset, skiprows=1)

clade_ID=t_file[:,0]
clade_ID=clade_ID.astype(int)
ts=t_file[:,2+2*args.j]
te=t_file[:,3+2*args.j]

constr=args.m

if args.c==-1:
	single_focal_clade = False
	clade_name = ""
else:
	single_focal_clade = True
	fixed_focal_clade = args.c
	clade_name = "_c%s" % (fixed_focal_clade)

Be_shape_beta = args.beta
if Be_shape_beta>1: beta_value = "_B%s" % (args.beta)	
else: beta_value = ""

all_events=sort(np.concatenate((ts,te),axis=0))[::-1] # events are speciation/extinction that change the diversity trajectory
n_clades,n_events=max(clade_ID)+1,len(all_events)
Dtraj=init_Dtraj(n_clades,n_events)


##### get indexes
s_list = []
e_list = []
s_or_e_list=[]
clade_inx_list=[]
unsorted_events = []
for i in range(n_clades):
	"used for Dtraj"
	s_list.append(ts[clade_ID==i])
	e_list.append(te[clade_ID==i])
	"used for lik calculation"
	s_or_e_list += list(np.repeat(1,len(ts[clade_ID==i]))) # index 1 for s events
	s_or_e_list += list(np.repeat(2,len(te[clade_ID==i]))) # index 2 for e events
	clade_inx_list += list(np.repeat(i,2*len(te[clade_ID==i])))
	unsorted_events += list(ts[clade_ID==i])
	unsorted_events += list(te[clade_ID==i])

s_or_e_array= np.array(s_or_e_list)
unsorted_events= np.array(unsorted_events)
s_or_e_array[unsorted_events==0] = 3
""" so now: s_or_e_array = 1 (s events), s_or_e_array = 2 (e events), s_or_e_array = 3 (e=0 events)"""


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
	Dtraj[:,i]=getDT(all_events_temp2[0],s_list[i],e_list[i])
	ind_clade_i = np.arange(len(all_events_temp2[0]))[all_events_temp2[2]==i]
	ind_sp = np.arange(len(all_events_temp2[0]))[all_events_temp2[1]==1]
	ind_ex = np.arange(len(all_events_temp2[0]))[all_events_temp2[1]==2]
	idx_s.append(np.intersect1d(ind_clade_i,ind_sp))
	idx_e.append(np.intersect1d(ind_clade_i,ind_ex))
#####
scale_factor = 1./np.max(Dtraj)
MAX_G = 0.30/scale_factor

GarrayA=init_Garray(n_clades)+1. # 3d array so:
                                 # Garray[i,:,:] is the 2d G for one clade
			         # Garray[0,0,:] is G_lambda, Garray[0,1,:] is G_mu for clade 0
RA=init_Garray(n_clades)
Constr_matrix=make_constraint_matrix(n_clades, constr)

l0A,m0A=init_BD(n_clades),init_BD(n_clades)

out_file_name="%s_%s_m%s_MCDD%s%s.log" % (dataset,args.j,constr,clade_name,beta_value)
logfile = open(out_file_name , "wb") 
wlog=csv.writer(logfile, delimiter='\t')

lik_head=""
for i in range(n_clades): lik_head+="\tlik_%s" % (i)
head="it\tposterior\tlikelihood%s\tprior" % (lik_head)
for i in range(n_clades): head+="\tl%s" % (i)
for i in range(n_clades): head+="\tm%s" % (i)
for i in range(n_clades): 
	for j in range(n_clades): 
		head+="\tGl%s_%s" % (i,j)
	for j in range(n_clades): 
		head+="\tGm%s_%s" % (i,j)
for j in range(n_clades): 
	head+="\thypZ%s" % (j)
		
head+="\thypR"
wlog.writerow(head.split('\t'))
logfile.flush()

hypZeroA=np.ones(n_clades)-.05 # P(G==0)
hypRA=np.ones(1)
hypZero=hypZeroA

t1=time.time()
for iteration in range(n_iterations):	
	hasting=0
	gibbs_sampling=0
	if iteration==0:
		actualGarray=GarrayA*scale_factor
		likA,priorA,postA=np.zeros(n_clades),0,0
		
	l0,m0=l0A,m0A
	Garray=GarrayA
	R=RA+init_Garray(n_clades)
	lik,priorBD=np.zeros(n_clades),0
	
	lik_test=np.zeros(n_clades)	
	
	if iteration==0:
		uniq_eve=np.unique(all_events,return_index=True)[1]  # indexes of unique values
		Garray_temp=Garray*R
		prior_r=0
		for i in range(n_clades):
			l_at_events=trasfMultiRate(l0[i],-Garray_temp[i,0,:],Dtraj)
			m_at_events=trasfMultiRate(m0[i],Garray_temp[i,1,:],Dtraj)
			l_s1a=l_at_events[idx_s[i]]
			m_e1a=m_at_events[idx_e[i]]
			lik[i] = (sum(log(l_s1a))-sum(abs(np.diff(all_events))*l_at_events[0:len(l_at_events)-1]*(Dtraj[:,i][1:len(l_at_events)])) \
			         +sum(log(m_e1a))-sum(abs(np.diff(all_events))*m_at_events[0:len(m_at_events)-1]*(Dtraj[:,i][1:len(l_at_events)])) )
		likA=lik

	else:	
		##### START FOCAL CLADE ONLY
		sampling_freqs=[.20,.205,.60]		
		if iteration<25000: rr = np.random.uniform(0,sampling_freqs[1])
		else: rr = np.random.random()
		
		if single_focal_clade is True and rr > sampling_freqs[1]: focal_clade=fixed_focal_clade
		else: focal_clade= np.random.random_integers(0,(n_clades-1),1)[0]
		
		if rr<sampling_freqs[0]:
			if rand.random()>.5: 
				l0=np.zeros(n_clades)+l0A
				l0[focal_clade],U=update_multiplier_proposal(l0A[focal_clade],1.2)
			else: 	
				m0=np.zeros(n_clades)+m0A
				m0[focal_clade],U=update_multiplier_proposal(m0A[focal_clade],1.2)
			hasting=U
		elif rr<sampling_freqs[1]: # update hypZ and hypR
			# Gibbs sampler (Bernoulli distribution + Beta[1,1])
			gibbs_sampling=1
			B_hp_alpha,B_hp_beta=Be_shape_beta,1. # uniform hyper-prior in [0,1]
			sum_R_per_clade = np.sum(RA,axis=(1,2))
			number_of_draws_per_clade = n_clades*2.
			alpha = B_hp_alpha+ (number_of_draws_per_clade - sum_R_per_clade) # no. zeros  
			beta  = B_hp_beta + sum_R_per_clade # no. ones
			hypZeroA=np.random.beta(alpha,beta)			
			#print RA
			#print alpha, beta, hypZeroA
			# Gibbs sampler (Exponential + Gamma[2,2])
			G_hp_alpha,G_hp_beta=1.,.01
			g_shape=G_hp_alpha+len(l0A)+len(m0A)
			rate=G_hp_beta+sum(l0A)+sum(m0A)
			hypRA = np.random.gamma(shape= g_shape, scale= 1./rate, size=1)		
		elif rr<sampling_freqs[2]: # update Garray (effect size) 
			Garray_temp= update_parameter_normal_2d_freq(GarrayA[focal_clade,:,:],.35,m=-MAX_G,M=MAX_G) 			
			Garray=np.zeros(n_clades*n_clades*2).reshape(n_clades,2,n_clades)+GarrayA
			Garray[focal_clade,:,:]=Garray_temp
		else:
			rrr=np.random.uniform(0,1,2) # update R (indicators)
			r_clade =np.random.randint(0,n_clades,2)
			r1= np.array([focal_clade, r_clade[0]])
			r2= np.array([focal_clade, r_clade[1]])
			# Gl
			if rrr[0]>.5: R[r1[0],0,r1[1]]=0
			else: R[r1[0],0,r1[1]]=1.
			# Gm
			if rrr[1]>.5: R[r2[0],1,r2[1]]=0
			else: R[r2[0],1,r2[1]]=1.
		
		Garray_temp=Garray*R
		i=focal_clade 
		l_at_events=trasfMultiRate(l0[i],-Garray_temp[i,0,:],Dtraj)
		m_at_events=trasfMultiRate(m0[i], Garray_temp[i,1,:],Dtraj)
		### calc likelihood - clade i ###
		l_s1a=l_at_events[idx_s[i]]
		m_e1a=m_at_events[idx_e[i]]
		lik_clade = (sum(log(l_s1a))-sum(abs(np.diff(all_events))*l_at_events[0:len(l_at_events)-1]*(Dtraj[:,i][1:len(l_at_events)])) \
		         +sum(log(m_e1a))-sum(abs(np.diff(all_events))*m_at_events[0:len(m_at_events)-1]*(Dtraj[:,i][1:len(l_at_events)])) )
		ind_focal=np.ones(n_clades)
		ind_focal[focal_clade]=0
		lik = likA*ind_focal
		lik[focal_clade] = lik_clade
		###### END FOCAL

	""" len(Rtemp[Rtemp==0]), where Rtemp=R[i,:,:]
	should be equal to n_clades*2 - sum(R[i,:,:]) and len(Rtemp[Rtemp==0]) = sum(R[i,:,:]
	BTW, it is n_clades*2 because the same prior is used for both l0 and m0
	
	THUS:
	
	sum_R_per_clade = np.sum(RA,axis=(1,2))
	log(hypZeroA) * (1-sum_R_per_clade) + log(1-hypZeroA)*(sum_R_per_clade))
	
	"""	
	sum_R_per_clade = np.sum(RA,axis=(1,2))
	prior_r = sum(log(hypZeroA) * ((n_clades*2)-sum_R_per_clade) + log(1-hypZeroA)*(sum_R_per_clade))	
	prior = prior_exponential(l0,hypRA)+prior_exponential(m0,hypRA)+prior_r
	
	if (sum(lik) + prior) - postA + hasting >= log(rand.random()) or iteration==0 or gibbs_sampling==1:
		postA=sum(lik)+prior
		likA=lik
		priorA=prior
		l0A=l0
                m0A=m0
		GarrayA=Garray
		RA=R
		actualGarray=GarrayA*RA*scale_factor
		#hypZeroA=hypZero
		#hypRA=hypR
	
	if iteration % print_freq ==0: 
		print iteration, array([postA]), sum(likA),sum(lik),prior, hasting
		#print likA
		#print "l:",l0A
		#print "m:", m0A
		#print "G:", actualGarray.flatten()
		#print "R:", RA.flatten()
		#print "Gr:", GarrayA.flatten()
		#print "Hmu:", hypZeroA, 1./hypRA[0] #,1./hypRA[1],hypRA[2]
	if iteration % sampling_freq ==0:
		log_state=[iteration,postA,sum(likA)]+list(likA)+[priorA]+list(l0A)+list(m0A)+list(actualGarray.flatten())+list(hypZeroA) +[hypRA[0]]
		wlog.writerow(log_state)
		logfile.flush()

print time.time()-t1
quit()










