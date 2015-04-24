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
p.add_argument('-m', type=int, help='model', default=0, metavar=0)
p.add_argument('-j', type=int, help='replicate', default=0, metavar=0)
args = p.parse_args()


dataset=args.d
#t_file=np.genfromtxt(dataset, names=True, delimiter='\t', dtype=float)
t_file=np.loadtxt(dataset, skiprows=1)

clade_ID=t_file[:,0]
clade_ID=clade_ID.astype(int)
ts=t_file[:,2+2*args.j]
te=t_file[:,3+2*args.j]

constr=args.m

all_events=sort(np.concatenate((ts,te),axis=0))[::-1] # events are speciation/extinction that change the diversity trajectory
n_clades,n_events=max(clade_ID)+1,len(all_events)
Dtraj=init_Dtraj(n_clades,n_events)

for i in range(n_clades): # make trajectory curves for each clade
	Dtraj[:,i]=getDT(all_events,ts[clade_ID==i],te[clade_ID==i])

scale_factor = 1./np.max(Dtraj)

GarrayA=init_Garray(n_clades)+1. # 3d array so:
                                 # Garray[i,:,:] is the 2d G for one clade
			         # Garray[0,0,:] is G_lambda, Garray[0,1,:] is G_mu for clade 0
RA=init_Garray(n_clades)
Constr_matrix=make_constraint_matrix(n_clades, constr)

l0A,m0A=init_BD(n_clades),init_BD(n_clades)

out_file_name="%s_%s_m%s_JH2.log" % (dataset,args.j,constr)
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
		
head+="\thypR\thypGa\thypGb"
wlog.writerow(head.split('\t'))
logfile.flush()

hypZeroA=np.ones(n_clades)-.05 # P(G==0)
hypRA=np.ones(3)

for iteration in range(10000000):	
	hasting=0
	if iteration==0:
		actualGarray=GarrayA*scale_factor
		likA,priorA,postA=np.zeros(n_clades),0,0
	l0,m0=l0A,m0A
	Garray=GarrayA
	hypZero=hypZeroA
	hypR=hypRA
	R=RA+init_Garray(n_clades)
	lik,priorBD=np.zeros(n_clades),0
		
	# update values
	rr=rand.random()
	if rr<.1 or iteration<1000:
		l0=update_positive_rate_vec(l0A,.075)
		m0=update_positive_rate_vec(m0A,.075)
	elif rr<.175: 
		hypZero=update_rate_vec(hypZeroA, .05, m=0,M=1)
		hypR=update_positive_rate_vec(hypRA,np.array([.05,0,0]))+np.zeros(3)
		if hypR[1]>1: hypR[1]=abs(1-(hypR[1]-1))
	elif rr<.5:
		Garray=abs(update_parameter_normal_2d_freq(GarrayA,.35,m=-10,M=10) )
	else:
		rrr=np.random.uniform(0,1,2)
		r1,r2=np.random.randint(0,n_clades,2),np.random.randint(0,n_clades,2)
		# Gl
		if rrr[0]>.5: R[r1[0],0,r2[0]]=0
		else: R[r1[0],0,r2[0]]=1.
		# Gm
		if rrr[1]>.5: R[r1[1],1,r2[1]]=0
		else: R[r1[1],1,r2[1]]=1.
	
	uniq_eve=np.unique(all_events,return_index=True)[1]  # indexes of unique values
	Garray_temp=Garray*R
	prior_r=0
	for i in range(n_clades):
		l_at_events=trasfMultiRate(l0[i],-Garray_temp[i,0,:],Dtraj)
		m_at_events=trasfMultiRate(m0[i],Garray_temp[i,1,:],Dtraj)
	 	s1,e1=ts[clade_ID==i],te[clade_ID==i]
        	
        	### calc likelihood - clade 0 ###
		ind_s1=np.nonzero(np.in1d(all_events,s1))[0]         # indexes of s1 in all_events
		ind_s =np.intersect1d(ind_s1,uniq_eve)               
		ind_e1=np.nonzero(np.in1d(all_events,e1))[0]
		ind_e =np.intersect1d(ind_e1,uniq_eve)
		l_s1a=l_at_events[ind_s]
		m_e1a=m_at_events[ind_e]
		
		lik[i] =  sum(log(l_s1a))-sum(abs(np.diff(all_events))*l_at_events[0:len(l_at_events)-1]*(Dtraj[:,i][1:len(l_at_events)])) \
		         +sum(log(m_e1a))-sum(abs(np.diff(all_events))*m_at_events[0:len(m_at_events)-1]*(Dtraj[:,i][1:len(l_at_events)])) 

		Rtemp=R[i,:,:]
        	prior_r += sum(log( hypZero[i]**len(Rtemp[Rtemp==0])* (1-hypZero[i])**len(Rtemp[Rtemp==1])))
	
	def pdf_exp(x,r): 
		rate=1./r
		return sum(log(rate)-rate*x)
	prior = pdf_exp(l0,hypR[0])+pdf_exp(m0,hypR[0])+prior_r
	prior += pdf_exp(hypR,10) 
	
	if (sum(lik) + prior + hasting) - postA >= log(rand.random()) or iteration==0:
		postA=sum(lik)+prior+hasting
		likA=lik
		priorA=prior
		l0A=l0
                m0A=m0
		GarrayA=Garray
		RA=R
		actualGarray=GarrayA*RA*scale_factor
		hypZeroA=hypZero
		hypRA=hypR
	
	if iteration % 1000 ==0: 
		print iteration, array([postA]), sum(likA),sum(lik),prior, hasting
		print "l:",l0A
		print "m:", m0A
		print "G:", actualGarray.flatten()
		print "R:", RA.flatten()
		print "Gr:", GarrayA.flatten()
		print "Hmu:", hypZeroA, 1./hypRA[0],1./hypRA[1],hypRA[2]
	if iteration % 1000 ==0:
		log_state=[iteration,postA,sum(likA)]+list(likA)+[priorA]+list(l0A)+list(m0A)+list(actualGarray.flatten())+list(hypZeroA) +[1./hypRA[0],1./hypRA[1],hypRA[2]]
		wlog.writerow(log_state)
		logfile.flush()


quit()










