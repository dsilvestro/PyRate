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
lib_updates_priors = imp.load_source("lib_updates_priors", "pyrate_lib/lib_updates_priors.py")
lib_DD_likelihood = imp.load_source("lib_DD_likelihood", "pyrate_lib/lib_DD_likelihood.py")
lib_utilities = imp.load_source("lib_utilities", "pyrate_lib/lib_utilities.py")
from lib_updates_priors import *
from lib_DD_likelihood  import *
self_path=os.getcwd()

#### ARGS
p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('-A', type=int, help='algorithm: "0" parameter estimation, "1" TI', default=0, metavar=0) # 0: par estimation, 1: TI
p.add_argument('-d', type=str, help='data set', default="", metavar="<file>")
p.add_argument('-c', type=str, help='covariate data set', default="", metavar="<file>")
p.add_argument('-j', type=int, help='replicate', default=0, metavar=0)
p.add_argument('-m', type=int, help='model: "-1" constant rate, "0" exponential, "1" linear', default=0, metavar=0)
p.add_argument('-equal_G', type=int, help='model: "0" unconstrained G, "1" constrained G', default=0, metavar=0)
p.add_argument('-n', type=int, help='mcmc generations',default=1050000, metavar=1050000)
p.add_argument('-s', type=int, help='sample freq.', default=1000, metavar=1000)
p.add_argument('-p', type=int, help='print freq.', default=1000, metavar=1000)
p.add_argument('-r', type=float, help='rescale values (0 to scale in [0,1], 0.1 to reduce range 10x, 1 to leave unchanged)', default=0, metavar=0)
p.add_argument('-clade', type=int, help='clade analyzed', default=0, metavar=0)
p.add_argument('-b', type=float, help='burnin in *mcmc.log to generate input file', default=0.1, metavar=0.1)
p.add_argument('-w',  type=float, help='window sizes (bd rates, G)',  default=[1.4, .05], metavar=1.4, nargs=2)
p.add_argument('-ginput', type=str,help='generate input file from *mcmc.log', default="", metavar="<path_to_mcmc.log>")
p.add_argument('-tag', metavar='<*tag*.log>', type=str,help="Tag identifying files to be combined and plotted",default="")
p.add_argument('-mL',  type=str, help='calculate marginal likelihood',  default="", metavar="<path_to_log_files>")
p.add_argument('-stimes',  type=float, help='shift times',  default=[], metavar=0, nargs='+') 
p.add_argument('-extract_mcmc', type=int, help='Extract "cold" chain in separate log file', default=1, metavar=1)
p.add_argument("-DD",  help='Diversity Dependent Model', action='store_true', default=False)



args = p.parse_args()

mcmc_gen = args.n
sampling_freq = args.s
print_freq = args.p
dataset=args.d
cov_file=args.c
rescale_factor=args.r
focus_clade=args.clade
win_size=args.w
s_times=np.sort(np.array(args.stimes))[::-1]
equal_g = args.equal_G

if args.ginput != "":
	lib_utilities.write_ts_te_table(args.ginput, tag=args.tag, clade=focus_clade,burnin=args.b)
	quit()

if args.mL != "":
	lib_utilities.calc_marginal_likelihood(infile=args.mL,burnin=int(args.b),extract_mcmc=args.extract_mcmc)
	quit()

#t_file=np.genfromtxt(dataset, names=True, delimiter='\t', dtype=float)
t_file=np.loadtxt(dataset, skiprows=1)

ts=t_file[:,2+2*args.j]
te=t_file[:,3+2*args.j]

# assign short branch length to singletons (ts=te)
ind_singletons=(ts==te).nonzero()[0]
z=np.zeros(len(ts))
z[ind_singletons] =0.1
ts =ts+z
# if more than one clade only one is analyzed (flag -clade)
clade_ID=t_file[:,0].astype(int)
ts,te=ts[clade_ID==focus_clade],te[clade_ID==focus_clade]	


###____ TMP
#	r_times= np.load("rand_times.npy")
#	add_rand_ts = r_times[:,0+2*args.j]
#	add_rand_te = r_times[:,1+2*args.j]
#	ind_singletons=(ts==te).nonzero()[0]
#	for i in ind_singletons:
#		M = max(add_rand_ts[i],add_rand_te[i])
#		m = min(add_rand_ts[i],add_rand_te[i])
#		add_rand_ts[i] = M
#	        add_rand_te[i] = m
#	
#	add_rand_ts=add_rand_ts[0:len(ts)]
#	add_rand_te=add_rand_te[0:len(te)]
#	
#	ts += add_rand_ts
#	te[te>0] += add_rand_te[te>0]
###____ TMP




print len(ts),len(te[te>0]),sum(ts-te)

if args.DD is True:
	head_cov_file = ["","DD"]
	ts_te_vec = np.sort( np.concatenate((ts,te)) )[::-1]
	Dtraj = getDT(ts_te_vec,ts,te) + np.zeros(len(ts_te_vec))
	times_of_T_change =  ts_te_vec
	Temp_values = Dtraj
	for i in range(len(Temp_values)):
		print "%s\t%s" % (times_of_T_change[i],Temp_values[i])
	
else:
	tempfile=loadtxt(cov_file,skiprows=1)
	head_cov_file = next(open(cov_file)).split()
	times_of_T_change= tempfile[:,0] # array of times of Temp change
	Temp_values=       tempfile[:,1] # array of Temp values at times_of_T_change

# Temp_values= (Temp_values-Temp_values[0]) # so l0 and m0 are rates at the present
if rescale_factor > 0: Temp_values = Temp_values*rescale_factor
else: 
	denom = (max(Temp_values)-min(Temp_values))
	if denom==0: denom=1.
	Temp_values = Temp_values/denom


#for i in range(len(Temp_values)):
#	print "%s\t%s" % (times_of_T_change[i],Temp_values[i])

#print "BRL" , sum(ts-te)
#print "range:", max(Temp_values)-min(Temp_values)

# create matrix of all events sorted (1st row) with indexes 0: times_of_T_change, 1: ts, 2: te, 3: te=0
z=np.zeros(len(te))+2
z[te==0] = 3
all_events_temp= np.array([np.concatenate((times_of_T_change,ts,te),axis=0),
np.concatenate((np.zeros(len(times_of_T_change)),np.zeros(len(ts))+1,z),axis=0)])

idx = np.argsort(all_events_temp[0])[::-1] # get indexes of sorted events
all_events_temp2=all_events_temp[:,idx] # sort by time of event

times_of_T_change_tste=all_events_temp2[0,:]
times_of_T_change_indexes=all_events_temp2[1,:].astype(int)
#times_of_T_change_tste=sort(np.concatenate((times_of_T_change,ts,te),axis=0))[::-1]
#print shape(times_of_T_change_tste2),shape(times_of_T_change_tste),times_of_T_change_tste2[r],times_of_T_change_tste[r],times_of_T_change_indexes[r]
all_events=times_of_T_change_tste # events are speciation/extinction that change the diversity trajectory
#print len(all_events), len(all_events[times_of_T_change_indexes==1]),len(all_events[times_of_T_change_indexes==2])
ind_s = np.arange(len(all_events))[times_of_T_change_indexes==1]
ind_e = np.arange(len(all_events))[times_of_T_change_indexes==2]

n_events=len(all_events)
Dtraj=init_Dtraj(1,n_events)

# make trajectory curves for each clade
Dtraj[:,0]=getDT(all_events,ts,te)

#print "TIME", max(times_of_T_change), max(ts),Temp_values[-1]

Temp_at_events= get_VarValue_at_time(times_of_T_change_tste,Temp_values,times_of_T_change_indexes,times_of_T_change,max(ts))
#_print ind_s[3566:]
#print Temp_at_events[150:]
#_print "HERE",len(ind_s),len(ind_e)

### Get indexes of all events based on times of shift
shift_ind = np.zeros(len(times_of_T_change_tste)).astype(int)
if len(s_times)>0:
	bins_h = sort([max(times_of_T_change_tste)+1,-1] + list(s_times))
	# hist gives the number of events within each time bin (between shifts)
	hist=np.histogram(times_of_T_change_tste,bins=bins_h)[0][::-1]
	I=np.empty(0)
	for i in range(len(hist)): I=np.append(I,np.repeat(i,hist[i]))
	# shift_ind = [0,0,0,1,1,2,2,2,2,...N], where 0 is index of oldest bin, N of the most recent
	shift_ind =I.astype(int)

scaled_temp=np.zeros(len(Temp_at_events))
for i in range(len(np.unique(shift_ind))):
	Temp_values= Temp_at_events[shift_ind==i]
	#Temp_values= (Temp_values-Temp_values[-1]) # so l0 and m0 are rates at the end of the time bin
	Temp_values= (Temp_values-mean(Temp_values)) # so l0 and m0 are rates at the mean temp value
	scaled_temp[shift_ind==i]= Temp_values

Temp_at_events=scaled_temp
#print Temp_at_events[150:]
#for i in range(len(all_events)): #range(600,650):
#	print "%s\t%s" % (round(all_events[i],2),round(Temp_at_events[i],2))
#quit()

### INIT PARAMS
n_time_bins=len(np.unique(shift_ind))
GarrayA=np.zeros((2,n_time_bins)) # correlation parameters with Temp of lambda (GarrayA[0]) and mu (GarrayA[1])
l0A,m0A= init_BD(n_time_bins),init_BD(n_time_bins)
hypRA,hypGA= 1.,1.

output_wd = os.path.dirname(dataset)
if output_wd=="": output_wd= self_path

if len(s_times)>0: s_times_str = "s_" + '_'.join(s_times.astype("str"))
else: s_times_str=""

if equal_g==1: add_equal_g="EG"
else: add_equal_g=""

if args.m== -1: out_file_name="%s/%s_%s_%s_%sconst%s.log"  % (output_wd,os.path.splitext(os.path.basename(dataset))[0],head_cov_file[1],args.j,s_times_str,add_equal_g)
if args.m==  0: out_file_name="%s/%s_%s_%s_%sexp%s.log"    % (output_wd,os.path.splitext(os.path.basename(dataset))[0],head_cov_file[1],args.j,s_times_str,add_equal_g)
if args.m==  1: out_file_name="%s/%s_%s_%s_%slinear%s.log" % (output_wd,os.path.splitext(os.path.basename(dataset))[0],head_cov_file[1],args.j,s_times_str,add_equal_g)



logfile = open(out_file_name , "wb") 
wlog=csv.writer(logfile, delimiter='\t')

head="it\tposterior\tlikelihood\tprior" 
for i in range(n_time_bins): head+="\tlik_L_%s" % (i)
for i in range(n_time_bins): head+="\tlik_M_%s" % (i)
for i in range(n_time_bins): head+="\tl0_t%s" % (i)
for i in range(n_time_bins): head+="\tm0_t%s" % (i)
for j in range(n_time_bins): 
	head+="\tGl_t%s" % (j)
for j in range(n_time_bins): 
	head+="\tGm_t%s" % (j)
head+="\thp_rate"
head+="\thp_sig2"
head+="\tbeta"
wlog.writerow(head.split('\t'))
logfile.flush()

if args.A==0:
	scal_fac_TI=np.ones(1)
elif args.A==1:
	# parameters for TI are currently hard-coded (K=10, alpha=0.3)
	scal_fac_TI=get_temp_TI()

d1 = win_size[0]
d2 = win_size[1] # starting win size for Gl, Gm
list_d2=sort(exp(scal_fac_TI))**3*d2+(exp(1-np.array(scal_fac_TI))-1)*d2

## prep for lik calculation
abs_diff = abs(np.diff(all_events))
#abs_diff = np.concatenate((abs_diff,np.zeros(10)))
V1,V2,V3,V4 = list(),list(),list(),list()
for i in range(n_time_bins):
	v1 = (shift_ind==i).nonzero()[0]
	if i == n_time_bins-1: v_1 = v1[0:-1]
	else: v_1 = v1
	v_2 = v_1+1
	V1.append(v_1)
	V2.append(v_2)
	V3.append(np.intersect1d(ind_s,v_1))
	V4.append(np.intersect1d(ind_e,v_1))
	


scal_fac_ind=0
lik_pA=np.zeros(n_time_bins)
for iteration in range(mcmc_gen * len(scal_fac_TI)):	
	
	if (iteration+1) % (mcmc_gen+1) ==0: 
		print iteration, mcmc_gen  
		scal_fac_ind+=1
	
	hasting=0
	l0,m0=0+l0A,0+m0A
	Garray=0+GarrayA
	if iteration==0:
		likA,priorA,postA=0,0,0
	lik,priorBD=0,0
	
	# update values
	sampling_freqs=[.25,.27] # 0: rates, 1: hyper-priors, (2: correlation params)
	rr=np.random.uniform(0,1,3)
	if args.m== -1: rr[0]=0 # never update Garray
	
	GIBBS = 0
	if iteration>10:
		if rr[0]<sampling_freqs[0] or iteration<1000:
			if rr[1]>.5: 
				l0,U=update_multiplier_proposal(l0A,d1)
			else: 	
				m0,U=update_multiplier_proposal(m0A,d1)
			hasting=U
		elif rr[0]<sampling_freqs[1]:# Gibbs sampling
			GIBBS = 1
			# Gibbs sampler - Exponential + Gamma
			G_hp_alpha,G_hp_beta=2.,2.
			g_shape=G_hp_alpha+len(l0A)+len(m0A)
			g_rate=G_hp_beta+sum(l0A)+sum(m0A)
			hypRA = np.random.gamma(shape= g_shape, scale= 1./g_rate)
			#__ # Gibbs sampler - Normal(loc=0, tau) + Gamma
			#__ G_hp_alpha,G_hp_beta=1.,1.
			#__ g_shape=G_hp_alpha + len(GarrayA.flatten())/2.
			#__ g_rate=G_hp_beta + sum((GarrayA.flatten()-0)**2)/2.
			#__ hypGA = np.random.gamma(shape= g_shape, scale= 1./g_rate)
			# Gibbs sampler - Normal(loc=0, sig2) + InvGamma
			G_hp_alpha,G_hp_beta=1.,1.
			g_shape=G_hp_alpha + len(GarrayA.flatten())/2.
			g_rate=G_hp_beta + sum((GarrayA.flatten()-0)**2)/2.
			hypGA = 1./np.random.gamma(shape= g_shape, scale= 1./g_rate)
		else:
			if rr[2]>.5:
				if equal_g==0:
					Garray[0]=update_parameter_normal_2d(Garray[0],list_d2[scal_fac_ind]) 
				else:
					Garray[0,:]=update_parameter_normal(Garray[0,0],list_d2[scal_fac_ind])[0]
			else:
				if equal_g==0:
					Garray[1]=update_parameter_normal_2d(Garray[1],list_d2[scal_fac_ind]) 
				else:
					Garray[1,:]=update_parameter_normal(Garray[0,0],list_d2[scal_fac_ind])[0]
			
	if args.m==0: 
		l_at_events=trasfMultipleRateTemp(l0, Garray[0],Temp_at_events,shift_ind)
		m_at_events=trasfMultipleRateTemp(m0, Garray[1],Temp_at_events,shift_ind)
	if args.m==1: 
		l_at_events=trasfMultipleRateTempLinear(l0, Garray[0],Temp_at_events,shift_ind)
		m_at_events=trasfMultipleRateTempLinear(m0, Garray[1],Temp_at_events,shift_ind)
	if args.m== -1: 
		l_at_events=np.repeat(l0,len(Temp_at_events))
		m_at_events=np.repeat(m0,len(Temp_at_events))
	
	# Global likelihood
	#__ l_s1a=l_at_events[ind_s]
	#__ m_e1a=m_at_events[ind_e]
        #__ 
	#__ lik =  sum(log(l_s1a))-sum( abs(np.diff(all_events))*l_at_events[0:len(l_at_events)-1]*(Dtraj[:,0][1:len(l_at_events)])) \
	#__       +sum(log(m_e1a))-sum( abs(np.diff(all_events))*m_at_events[0:len(m_at_events)-1]*(Dtraj[:,0][1:len(l_at_events)])) 

	# partial likelihoods
	lik_p=np.zeros(n_time_bins*2)
	for i in range(n_time_bins):
		v_1 = V1[i]
		v_2 = V2[i]
		l_s1a=l_at_events[V3[i]]
		lik_p[i] = sum(log(l_s1a)) -sum( abs_diff[v_1] * l_at_events[v_1] * (Dtraj[v_2,0])) 

	for i in range(n_time_bins):
		v_1 = V1[i]
		v_2 = V2[i]
		m_e1a=m_at_events[V4[i]]
		lik_p[i+n_time_bins] = sum(log(m_e1a)) -sum( abs_diff[v_1] * m_at_events[v_1] * (Dtraj[v_2,0])) 
		           
	
	# Check likelihoods  
	#__ if iteration % 100 ==0:
	#__ 	print round(lik - sum(lik_p), 8)
	lik=sum(lik_p)
	
	lik_alter = lik * scal_fac_TI[scal_fac_ind]
	
	# Add hyper-prior + Gibbs sampling 
	prior= prior_normal(Garray,scale=sqrt(hypGA)) + prior_exponential(l0,rate=hypRA) + prior_exponential(m0,rate=hypRA)  # prior_normal_tau(Garray,precision=hypGA)
	
	if (lik_alter + prior + hasting) - postA >= log(rand.random()) or iteration==0 or GIBBS == 1:
		postA=lik_alter+prior
		likA=lik
		lik_pA=lik_p
		priorA=prior
		l0A=l0
                m0A=m0
		GarrayA=Garray
	if iteration % print_freq ==0: 
		print iteration, array([postA, likA,lik,prior]), hasting, scal_fac_TI[scal_fac_ind]
		print "l:",l0A, "\nm:", m0A, "\nG:", GarrayA
	if iteration % sampling_freq ==0:
		log_state=[iteration,postA,likA,priorA] + list(lik_pA) + list(l0A) + list(m0A) +list(GarrayA.flatten()) + [hypRA,hypGA] + [scal_fac_TI[scal_fac_ind]]
		wlog.writerow(log_state)
		logfile.flush()


quit()












