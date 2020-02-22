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
np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit
from multiprocessing import Pool, freeze_support
import thread
small_number= 1e-10

def pNtvar(arg):
	T=arg[0]
	L=arg[1]
	M=arg[2]
	N=arg[3]
	Dt=-np.diff(T)
        r_t = (L - M)*Dt
	#Beta=  sum(exp((L - M)*Dt))
	Alpha= sum(L*exp((L - M)*Dt))
	lnBeta=  log(sum(exp((L - M)*Dt)))
	lnAlpha= log(sum(L*exp((L - M)*Dt)))
	#P   = (Beta/(Alpha*(1+Alpha))) *    (Alpha/(1+Alpha))**N
	lnP = (lnBeta-(lnAlpha+log(1+Alpha))) + (lnAlpha-log(1+Alpha))*N
	return lnP


#### Calculate the likelihood of s1,e1 given s2,e2 ###
def getDT(T,s,e): # returns the Diversity Trajectory of s,e at times T
	return array([len(s[s>t])-len(s[e>t]) for t in T])

def get_DT(T,s,e): # returns the Diversity Trajectory of s,e at times T (x10 faster)
	B=np.sort(np.append(T,T[0]+1))+small_number # the + small_number prevents problems with identical ages
	ss1 = np.histogram(s,bins=B)[0]
	ee2 = np.histogram(e,bins=B)[0]
	DD=(ss1-ee2)[::-1]
	#return np.insert(np.cumsum(DD),0,0)[0:len(T)]
	return np.cumsum(DD)[0:len(T)] 

def get_VarValue_at_time(all_Times,Var_values,times_of_T_change_indexes,times_of_T_change,root_age):
	# times_of_T_change_indexes index 0 for times of T change	
	variable,ind=list(),list()	
	j =0
	Var_values = Var_values[::-1]
	
	for i in range(len(all_Times)):
		time_t = all_Times[i]

		if times_of_T_change_indexes[i]==0:
			 variable.append(Var_values[j])
			 j+=1
		elif time_t > max(times_of_T_change):
			 variable.append(Var_values[0])
		else: variable.append(Var_values[j])
	return array(variable)


def get_VarValue_at_timeMCDD(all_Times,Var_values,times_of_T_change_indexes,times_of_T_change,root_age,Curve_indexes,Focus_Curve_index):
	# times_of_T_change_indexes index 0 for times of T change	
	variable,ind=list(),list()	
	j =0
	Var_values = Var_values[::-1]
	
	for i in range(len(all_Times)):
		time_t = all_Times[i]

		if times_of_T_change_indexes[i]==0 and Curve_indexes[i]==Focus_Curve_index:
			variable.append(Var_values[j])
			j+=1
		elif time_t > max(times_of_T_change):
			variable.append(Var_values[0])
		elif j >= len(Var_values): # if the variable values do not reach the present assume most recent value
			variable.append(Var_values[len(Var_values)-1])
		else: variable.append(Var_values[j])
	return array(variable)









def trasfRate(r0,n,K,m,G):    # transforms a baseline rate r0 based on n taxa
	rate=r0+r0*(n*K)+r0*(m*G)  # and a correlation parameter K
	return np.amax(np.array((rate,zeros(len(rate))+small_number)),axis=0)

def trasfMultiRate(r0,Garray_clade,Dtraj):     # transforms a baseline rate r0 based number of taxa scaled by max diversity over all clades
	#mDtraj = Dtraj/np.mean(Dtraj, axis=0) # thus g is competition per species
	mDtraj = Dtraj/np.max(Dtraj)
	#mDtraj=(Dtraj-np.min(Dtraj, axis=0))/(np.max(Dtraj, axis=0)-np.min(Dtraj, axis=0))
	r_rate=r0 + np.sum(r0 * Garray_clade * mDtraj,axis=1)
	r_rate= np.amax(np.array((r_rate,zeros(len(r_rate))+small_number)),axis=0)
	return r_rate

def trasfMultiRateCladeScaling(r0,Garray_clade,Dtraj):    # transforms a baseline rate r0 based number of taxa scaled by max diversity for each clade
	mDtraj = Dtraj/np.max(Dtraj, axis=0)              # thus g is competition per clade
	r_rate=r0 + np.sum(r0 * Garray_clade * mDtraj,axis=1)
	r_rate= np.amax(np.array((r_rate,zeros(len(r_rate))+small_number)),axis=0)
	return r_rate

def trasfMultiRateND(r0,Garray_clade,mDtraj):    # curves not transformed
	#mDtraj=(Dtraj-np.min(Dtraj, axis=0))/(np.max(Dtraj, axis=0)-np.min(Dtraj, axis=0))
	r_rate=r0 + np.sum(r0 * Garray_clade * mDtraj,axis=1)
	r_rate= np.amax(np.array((r_rate,zeros(len(r_rate))+small_number)),axis=0)
	return r_rate

def trasfMultiRateND_exp(r0,Garray_clade,mDtraj):    # curves not transformed
	#mDtraj=(Dtraj-np.min(Dtraj, axis=0))/(np.max(Dtraj, axis=0)-np.min(Dtraj, axis=0))
	#r_rate=r0 + np.sum(r0 * Garray_clade * mDtraj,axis=1)
	r_rate= r0* exp(np.sum(Garray_clade*mDtraj,axis=1)) 
	r_rate= np.amax(np.array((r_rate,zeros(len(r_rate))+small_number)),axis=0)
	return r_rate




def trasfMultiRateN(r0,Garray_clade,Dtraj,Ntraj, Narray_clade):    # transforms a baseline rate r0 based on n taxa
	#mDtraj = Dtraj
	#mDtraj = Dtraj/np.max(Dtraj, axis=0)
	mNtraj = Ntraj/np.max(Ntraj)
	#print "\nmod", mDtraj
	#mNtraj = Ntraj/np.max(Ntraj) 
	#mDtraj=(Dtraj-np.min(Dtraj, axis=0))/(np.max(Dtraj, axis=0)-np.min(Dtraj, axis=0))
	r_rate=r0 + np.sum(r0 * Garray_clade * mDtraj,axis=1) + np.sum(r0 * mNtraj * Narray_clade,axis=1)
	r_rate= np.amax(np.array((r_rate,zeros(len(r_rate))+small_number)),axis=0)
	return r_rate


def trasfRateTemp(l0, alpha,Temp_at_events):
	mTemp_at_events = Temp_at_events #/np.mean(Temp_at_events, axis=0)
	r_rate= l0* exp(alpha*mTemp_at_events)
	r_rate= np.amax(np.array((r_rate,zeros(len(r_rate))+small_number)),axis=0)
	return r_rate

def trasfMultipleRateTemp(L0, Alpha,mTemp_at_events,Index_at_events):
	r_rate= L0[Index_at_events]* exp(Alpha[Index_at_events]*mTemp_at_events)
	r_rate= np.amax(np.array((r_rate,zeros(len(r_rate))+small_number)),axis=0)
	return r_rate

def trasfRateTempLinear(l0, alpha,Temp_at_events):
	mTemp_at_events = Temp_at_events #/np.mean(Temp_at_events, axis=0)
	r_rate= l0 + l0*alpha*mTemp_at_events
	r_rate= np.amax(np.array((r_rate,zeros(len(r_rate))+small_number)),axis=0)
	return r_rate

def trasfMultipleRateTempLinear(L0, Alpha,mTemp_at_events,Index_at_events):
	r_rate= L0[Index_at_events] + L0[Index_at_events]*Alpha[Index_at_events]*mTemp_at_events
	r_rate= np.amax(np.array((r_rate,zeros(len(r_rate))+small_number)),axis=0)
	return r_rate



def intRate_(t1,t2,events,R):            # R = rates_at_events 
	#                               #
	#  e1     e2    e3  e4      e5  #
	#     t1---------------t2       # t1,t2: speciation extinction times 
	#  e1 |---e2----e3--e4--|       # ind:      e2,e3,e4
	#  r1     r2    r3  r4      r5  # rates: r1,r2,r3,r4
	#                               #
	#     t1--e2----e3--e4-t2       # t_vector	
	ind=np.intersect1d(np.nonzero(events<t1)[0],np.nonzero(events>t2)[0])
	if len(ind)==0: rates=[R[len(events[events>=t1])-1]] # take rate at previous event
	else: rates=insert(R[ind],0,R[ind[0]-1])	
	t_vector=insert(events[ind], [0,len(events[ind])], [t1,t2])
	int_rate=abs(np.diff(t_vector))*rates # integral
	return rates[0],rates[len(rates)-1],sum(int_rate)

def intRate(t1,t2,events,R):            # R = rates_at_events (including t1,t2)
	#                               #
	#     t1-----------------t2     # t1,t2: speciation extinction times 
	#     |---e2----e3--e4---|      # ind:      e2,e3,e4
	#     r1  r2    r3  r4   r5     # rates: r1,r2,r3,r4
	#                               #
	#     t1--e2----e3--e4---t2     # t_vector	
	ind=np.intersect1d(np.nonzero(events<=t1)[0],np.nonzero(events>=t2)[0])
	rates=R[ind]
	t_vector=events[ind]
	int_rate=abs(np.diff(t_vector))*rates[0:len(rates)-1] # integral
	#print R[ind],rates[0],rates[len(rates)-1],sum(int_rate)
	return rates[0],rates[len(rates)-1],sum(int_rate)

def logDDBDlik(arg): # log lik
	[s,e,l_at_events,m_at_events,events]=arg 	
	l_s,l_e,l_int=intRate(s,e,events,l_at_events)
	m_s,m_e,m_int=intRate(s,e,events,m_at_events)	
	if l_s>0 and m_e>0:
		lik_l= log(l_s) + (-l_int)
		lik_m= log(m_e) + (-m_int)
	else: return -inf
	return lik_l+lik_m


def DDBDlik(arg): # lik
	[s,e,l_at_events,m_at_events,events]=arg 	
	l_s,l_e,l_int=intRate(s,e,events,l_at_events)
	m_s,m_e,m_int=intRate(s,e,events,m_at_events)	
	lik= l_s * exp(-l_int) * m_e * exp(-m_int)
	return lik

def logDDBDlik_no_loop(arg): # log lik
	[s,e,l_at_events,m_at_events,events]=arg
	l_s,l_e,l_int=intRate(s,e,events,l_at_events)
	m_s,m_e,m_int=intRate(s,e,events,m_at_events)	
	if l_s>0 and m_e>0:
		lik_l= (-l_int)
		lik_m= (-m_int)
	else: return -inf
	return lik_l+lik_m
                           

def get_temp_TI(k=10,a=0.3):
	K=k-1.        # K+1 categories
	k=array(np.arange(int(K+1)))
	beta=k/K
	alpha=a            # categories are beta distributed
	temperatures=beta**(1./alpha)
	temperatures[0]+= small_number # avoid exactly 0 temp
	temperatures=temperatures[::-1]
	return temperatures













