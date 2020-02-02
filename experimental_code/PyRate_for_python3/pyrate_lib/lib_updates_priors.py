#!/usr/bin/env python 
import argparse, os,sys
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import random as rand
import sys, platform, time
import os, csv
from scipy.special import gdtr, gdtrix
from scipy.special import betainc
import scipy.stats
np.set_printoptions(suppress=True) 
np.set_printoptions(precision=3)   



########################## INITIALIZE #######################################
def init_BD(n): return np.random.exponential(.5, n)

def init_Dtraj(n_clades,n_events): return np.zeros(n_clades*n_events).reshape(n_events,n_clades)

def init_Garray(n): return np.zeros(n*n*2).reshape(n,2,n)
########################## UPDATES #######################################
def update_parameter(i, m, M, d): 
	ii = i+(rand.random()-.5)*d
	if ii<m: ii=(ii-m)+m
	if ii>M: ii=(M-(ii-M))
	if ii<m: ii=i
	return ii

def update_positive_rate_vec(i, d): 
	ii = fabs(i+(np.random.uniform(0,1,len(i))-.5)*d)
	return ii

def update_rate_vec(i, d, m=-2,M=2): 
	s = i+(np.random.uniform(0,1,len(i))-.5)*d
	#print s
	s[s>M]=M-(s[s>M]-M)
	s[s<m]=(m-s[s<m])+m
	return s


def update_parameter_uni_2d_freq(oldL,d,f=.65,m=-2,M=2):
	S=shape(oldL)
	ii=(np.random.uniform(0,1,S)-.5)*d
	ff=np.rint(np.random.uniform(0,f,S))
	s= oldL + ii*ff	
	s[s>M]=M-(s[s>M]-M)
	s[s<m]=(m-s[s<m])+m
	return s


def update_parameter_normal(i, d):
	ii = np.random.normal(i,d,1)
	return ii


def update_multiplier_proposal(i,d,f=.65):
	S=shape(i)
	u = np.random.uniform(0,1,S) #*np.rint(np.random.uniform(0,f,S))
	l = 2*log(d)
	m = exp(l*(u-.5))
	#print "\n",u,m,"\n"
	ii = i * m
	U=sum(log(m))
	return ii, U

def update_multiplier_freq(q,d=1.1,f=0.25):
	S=np.shape(q)
	ff=np.random.binomial(1,f,S)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
	m[ff==0] = 1.
	new_q = q * m
	U=sum(log(m))
	return new_q,U


def update_parameter_normal_2d(L, d):
	ii = np.random.normal(L.flatten(),d,size(L)).reshape(shape(L))
	return ii

def update_parameter_normal_2d_freq(oldL,d,f=.25,m=-2,M=2):
	S=shape(oldL)
	dV = np.random.uniform(0.5*d,2*d,S)
	ii=np.random.normal(0,dV,S)
	ff = np.random.binomial(1,f,S)
	s= oldL + ii*ff	
	s[s>M]=M-(s[s>M]-M)
	s[s<m]=(m-s[s<m])+m
	return s
	
def multiplier_normal_proposal_pos_neg_vec(oldL,d1 = 0.3,d2 = 1.2,f=.25):
	S=shape(oldL)
	ff = np.random.binomial(1,f,S)
	if np.random.random()<.5:
		ii=np.random.normal(0,d1,S)
		s= oldL + ii*ff
		U = 0.
		return s, U
	else:
		u = np.random.uniform(0,1,S) #*np.rint(np.random.uniform(0,f,S))
		l = 2*log(d2)
		m = exp(l*(u-.5))
		m[ff==0] = 1.
		ii = oldL * m
		U=sum(log(m))
		return ii, U

def multiplier_proposal_pos_neg_vec(i,d):
	S=shape(i)
	ff=np.rint(np.random.uniform(0,.65,S))
	ii = np.zeros(S)
	
	if np.random.random() < .05:
		ii += i
		ii[ff>0] = -ii[ff>0] # change sign
		return ii, 0 
	else: # multiplier proposal
		u = np.random.uniform(0,1,S)
		l = 2*log(d)
		m = exp(l*(u-.5))
		m = m * ff
		ii = i * m
		ii[m==0] = i[m==0]
		return ii, sum(log(m[m>0]))


def make_constraint_matrix(n, constraint):
	M=np.zeros(n*n*2).reshape(n,2,n)
	if constraint==-1: return M
	if constraint==0 or constraint==3 or constraint==4: return M+1.
	### MODEL I (diversity dependent)
	if constraint==1:
		for i in range(n):
				M[i,0,i]=1 
				M[i,1,i]=1 
		return M	
	### MODEL II (biotic interaction)
	if constraint==2:
		for i in range(n):
			for j in range(n):
				if i != j:
					M[i,0,j]=1
					M[i,1,j]=1
		return M	
		
	
########################## PRIORS #######################################
def prior_exponential(L,rate): return sum(scipy.stats.expon.logpdf(L, scale=1./rate))

def prior_gamma(L,a,b): return sum(scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0))

def prior_normal(L,loc=0,scale=1): return sum(scipy.stats.norm.logpdf(L,loc,scale))

def prior_normal_tau(L,loc=0,precision=1): 
	# precision: tau = 1/sig2
	# scale: sqrt(sig2) = sqrt(1/tau)
	return sum(scipy.stats.norm.logpdf(L,loc,scale=sqrt(1./precision)))

	
def prior_times_frames(t, root,a): # un-normalized Dirichlet (truncated)
	t_rel=abs(np.diff(t))/root
	if min(abs(np.diff(t)))<=1: return -inf
	else: return (a-1)*log(t_rel)

def prior_beta(x,a): return scipy.stats.beta.logpdf(x, a,a)
	
def prior_root_age(root, max_FA, l): # exponential (truncated)
	l=1./l
	if root>=max_FA: return log(l)-l*(root)
	else: return -inf

def prior_uniform(x,m,M): 
	x=x.flatten()
	if min(x)>m and max(x)<M: return 0 #sum(log(1./(M-m)))
	else: return -inf

def G_density(x,a,b): return scipy.stats.gamma.pdf(x, a, scale=1./b,loc=0)

def logPERT4_density(M,m,a,b,x): # relative 'stretched' LOG-PERT density: PERT4 * (s-e)
	return log((M-x)**(b-1) * (-m+x)**(a-1)) - log ((M-m)**4 * f_beta(a,b))

def PERT4_density(M,m,a,b,x):  # relative 'stretched' PERT density: PERT4 * (s-e) 
	return ((M-x)**(b-1) * (-m+x)**(a-1)) /((M-m)**4 * f_beta(a,b))

def logPERT4_density5(M,m,a,b,x): # relative LOG-PERT density: PERT4
	return log((M-x)**(b-1) * (-m+x)**(a-1)) - log ((M-m)**5 * f_beta(a,b))

#def pdf_exp(x,r): 
#	rate=1./r
#	return sum(log(rate)-rate*x)


