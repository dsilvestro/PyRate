#!/usr/bin/env python 
# Created by Daniele Silvestro on 02/03/2012 => pyrate.help@gmail.com 
import argparse, os,sys, platform, time, csv
import random as rand
import warnings
version= "      PyRate 0.570       "
build  = "        20140721         "
if platform.system() == "Darwin": sys.stdout.write("\x1b]2;%s\x07" % version)

citation= """Silvestro, D., Schnitzler, J., Liow, L.H., Antonelli, A. and Salamin, N. (2014)
Bayesian Estimation of Speciation and Extinction from Incomplete Fossil
Occurrence Data. Systematic Biology, 63, 349-367.

Silvestro, D., Salamin, N., Schnitzler, J. (in review)
PyRate: A new program to estimate speciation and extinction rates from
incomplete fossil record.
"""
print """
                       %s
                       %s

           Bayesian estimation of speciation and extinction
                  rates from fossil occurrence data        

               Daniele Silvestro, Jan Schnitzler et al.
                        pyrate.help@gmail.com

\n""" % (version, build)
# check python version
V=list(sys.version_info[0:3])
if V[0]>2: sys.exit("""\nPyRate currently runs only under python 2. Python 3.X is currently not supported.
	You can download to python 2.7 at: https://www.python.org/downloads/""")

# LOAD LIBRARIES
try:
	import argparse
except(ImportError):
	sys.exit("""\nError: argparse library not found.
	You can upgrade to python 2.7 at: https://www.python.org/downloads/ 
	or install argparse at: https://code.google.com/p/argparse/ \n""")

try: 
	from numpy import *
	import numpy as np
except(ImportError): 
	sys.exit("\nError: numpy library not found.\nYou can download numpy at: http://sourceforge.net/projects/numpy/files/ \n")

try:
	import scipy
	from scipy.special import gamma
	from scipy.special import beta as f_beta
	from scipy.special import gdtr, gdtrix
	from scipy.special import betainc
	import scipy.stats
except(ImportError): 
	sys.exit("\nError: scipy library not found.\nYou can download scipy at: http://sourceforge.net/projects/scipy/files/ \n")

try: 
	import multiprocessing, thread
	import multiprocessing.pool
	class NoDaemonProcess(multiprocessing.Process):
		# make 'daemon' attribute always return False
		def _get_daemon(self): return False			
		def _set_daemon(self, value): pass			
		daemon = property(_get_daemon, _set_daemon)

	class mcmcMPI(multiprocessing.pool.Pool):
		# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
		# because the latter is only a wrapper function, not a proper class.
		Process = NoDaemonProcess
	use_seq_lik=False
	if platform.system() == "Windows" or platform.system() == "Microsoft": use_seq_lik=True
except(ImportError): 
	print "\nWarning: library multiprocessing not found.\nPyRate will use (slower) sequential likelihood calculation. \n"
	use_seq_lik=True

if platform.system() == "Windows" or platform.system() == "Microsoft": use_seq_lik=True
### numpy print options ###
np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit

original_stderr = sys.stderr
NO_WARN = original_stderr #open('pyrate_warnings.log', 'w')
small_number= 1e-50

########################## CALC PROBS ##############################
def calcHPD(data, level) :
	assert (0 < level < 1)	
	d = list(data)
	d.sort()	
	nData = len(data)
	nIn = int(round(level * nData))
	if nIn < 2 :
		raise RuntimeError("not enough data")	
	i = 0
	r = d[i+nIn-1] - d[i]
	for k in range(len(d) - (nIn - 1)) :
		rk = d[k+nIn-1] - d[k]
		if rk < r :
			r = rk
			i = k
	assert 0 <= i <= i+nIn-1 < len(d)	
	return (d[i], d[i+nIn-1])

def check_burnin(b,I):
	#print b, I
	if b<1: burnin=int(b*I)
	else: burnin=int(b)
	if burnin>=(I-10):
		print "Warning: burnin too high! Excluding 10% instead."
		burnin=int(0.1*I)
	return burnin

def calc_model_probabilities(f,burnin):
	print "parsing log file...\n"
	t=loadtxt(f, skiprows=1)
	num_it=shape(t)[0]
	if num_it<10: sys.exit("\nNot enough samples in the log file!\n")
	burnin=check_burnin(burnin, num_it)
	print "First %s samples excluded as burnin.\n" % (burnin)
	file1=file(f, 'U')
	L=file1.readlines()
	head= L[0].split()
	PAR1=["k_birth","k_death"]
	k_ind= [head.index(s) for s in head if s in PAR1]
	z1=t[burnin:,k_ind[0]]  # list of shifts (lambda)
	z2=t[burnin:,k_ind[1]]  # list of shifts (mu)
	y1= max(max(z1),max(z2))
	print "Model           Probability"
	print "          Speciation  Extinction"
	for i in range(1,int(y1)+1):
		k_l=float(len(z1[z1==i]))/len(z1)
		k_m=float(len(z2[z2==i]))/len(z2)
		print "%s-rate    %s      %s" % (i,round(k_l,4),round(k_m,4)) 
	print "\n"
	quit()

def calc_ts_te(f, burnin):
	if f=="null": return FA,LO
	else:
		t_file=np.genfromtxt(f, delimiter='\t', dtype=None)
		shape_f=list(shape(t_file))
		if len(shape_f)==1: sys.exit("\nNot enough samples in the log file!\n")
		if shape_f[1]<10: sys.exit("\nNot enough samples in the log file!\n")
		#ind_ts0=np.where(t_file[0]=="Aceratherium_incisivum_TS")[0][0]
		#ind_te0=np.where(t_file[0]=="Aceratherium_incisivum_TE")[0][0]
		
		ind_start=np.where(t_file[0]=="tot_length")[0][0]
		indexes= np.array([ind_start+1, ind_start+(shape_f[1]-ind_start)/2])
		# fixes the case of missing empty column at the end of each row
		if t_file[0,-1] != "False": 
			indexes = indexes+np.array([0,1])
		ind_ts0 = indexes[0]
		ind_te0 = indexes[1]
		
		meanTS,meanTE=list(),list()
		burnin=check_burnin(burnin, shape_f[0])
		burnin+=1
		j=0
		for i in arange(ind_ts0,ind_te0):
			meanTS.append(mean(t_file[burnin:shape_f[0],i].astype(float)))
			meanTE.append(mean(t_file[burnin:shape_f[0],ind_te0+j].astype(float)))
			j+=1
		return array(meanTS),array(meanTE)

def calc_BF(f1, f2):
	input_file_raw = [os.path.basename(f1),os.path.basename(f2)]	
	def get_ML(FILE):
		file1=file(FILE, 'U')
		L=file1.readlines()	
		for i in range(len(L)):
			if "Marginal likelihood" in L[i]:
				x=L[i].split("Marginal likelihood: ")
				ML= float(x[1])
				return ML
	BF= 2*(get_ML(f1)-get_ML(f2))
	if abs(BF)<2: support="negligible"
	elif abs(BF)<6: support="positive"
	elif abs(BF)<10: support="strong"
	else: support="very strong"	
	if BF>0: best=0
	else: best=1	
	print "\nModel A: %s\nModelB: %s" % (input_file_raw[best],input_file_raw[abs(best-1)])
	print "\nModel A received %s support against Model B\nBayes Factor: %s\n\n" % (support, round(abs(BF), 4))


########################## PLOT RTT ##############################
def plot_RTT(files, stem_file, wd, burnin):
	print "parsing log files...",
	def print_R_vec(name,v):
		vec="%s=c(%s, " % (name,v[0])
		for j in range(1,len(v)-1): vec += "%s," % (v[j])
		vec += "%s)"  % (v[j+1])
		return vec
	if platform.system() == "Windows" or platform.system() == "Microsoft": out="%s\%sRTTplot.r" % (wd, stem_file)
	else: out="%s/%sRTTplot.r" % (wd, stem_file)
	newfile = open(out, "wb") 
	for PAR in ["l","m","r"]: # speciation/extinction/diversification
		mean_sp_m,mean_sp_M,hpd__sp_m,hpd__sp_M= list(),list(),list(),list()
		count=0
		for f in files: 
			t=loadtxt(f, skiprows=1)
			num_it=shape(t)[0]
			if PAR=="l": burnin=check_burnin(burnin, num_it)
			t=t[burnin:,:]
			file1=file(f, 'U')
			L=file1.readlines()
			head= L[0].split()
			sys.stdout.write(".")
			sys.stdout.flush()
			sp_ind= [head.index(s) for s in head if PAR in s]
			k=0	
			for j in sp_ind:
				col=t[:,j]
				hpd_val=array(calcHPD(col, .95))
				mean_val=mean(col)
				if count==0:
					mean_sp_m.append(mean_val)
					mean_sp_M.append(mean_val)
					hpd__sp_m.append(hpd_val[0])
					hpd__sp_M.append(hpd_val[1])
				else:
					mean_sp_m[k]=min(mean_val,mean_sp_m[k])
					mean_sp_M[k]=max(mean_val,mean_sp_M[k])
					hpd__sp_m[k]=min(hpd_val[0], hpd__sp_m[k])
					hpd__sp_M[k]=max(hpd_val[1], hpd__sp_M[k])						
				k+=1
			count +=1
			Rfile="\n# %s" % (f)		
		if PAR=="l":
			if platform.system() == "Windows" or platform.system() == "Microsoft":
				Rfile+= "\n\npdf(file='%s\%sRTTplot.pdf',width=7, height=7)" % (wd, stem_file) # \npar(mfrow=c(3,1))
			else: 
				Rfile+= "\n\npdf(file='%s/%sRTTplot.pdf',width=7, height=7)" % (wd, stem_file) # \npar(mfrow=c(3,1))
		Rfile+= print_R_vec('\nhpd_m',  hpd__sp_m)
		Rfile+= print_R_vec('\nhpd_M',  hpd__sp_M)
		Rfile+= print_R_vec('\nmean_m', mean_sp_m)
		Rfile+= print_R_vec('\nmean_M', mean_sp_M)
		Rfile+= "\nage=(0:(length(hpd_m)-1))* -1"
		if PAR=="l": rate,title="speciation",", main='%s' " % (stem_file)
		elif PAR=="m": rate,title="extinction", ""				
		elif PAR=="r": rate="net diversification"
		Rfile+= "\nplot(age,hpd_M,type = 'n', ylim = c(%s, %s), xlim = c(%s,0), ylab = '%s rate', xlab = 'Ma' %s)" % (min(hpd__sp_m),max(hpd__sp_M),-len(sp_ind), rate, title) # 
		Rfile+= """\npolygon(c(age, rev(age)), c(hpd_M, rev(hpd_m)), col = "#E5E4E2", border = NA)"""
		Rfile+= """\npolygon(c(age, rev(age)), c(mean_M, rev(mean_m)), col = "#504A4B", border = NULL)  """	
		if PAR=="r": 
			#Rfile+= """\nabline """ # add horizontal line
			Rfile+= "\nn<-dev.off()"
			Rfile+= "\ncat('\nThe RTT plot was saved as: %sRTTplot.pdf\n')" % (stem_file)
		newfile.writelines(Rfile)
	newfile.close()
	print "\n95% HPD calculated code from Biopy\n(https://www.cs.auckland.ac.nz/~yhel002/biopy/)"
	print "\nAn R script with the source for the RTT plot was saved as: %sRTTplot.pdf\n(in %s)" % (stem_file, wd)
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		cmd="cd %s; Rscript %s\%sRTTplot.r" % (wd,wd, stem_file)
	else: 
		cmd="cd %s; Rscript %s/%sRTTplot.r" % (wd,wd, stem_file)
	os.system(cmd)
	#print "\nThe RTT plot was saved as: %sRTTplot.pdf\n" % (wd, stem_file)

########################## INITIALIZE MCMC ##############################
def get_gamma_rates(a):
	b=a
	m = gdtrix(b,a,YangGammaQuant) # user defined categories
	s=args.ncat/sum(m) # multiplier to scale the so that the mean of the discrete distribution is one
	return array(m)*s # SCALED VALUES

def init_ts_te(FA,LO):
	ts=FA+np.random.exponential(.75, len(FA)) # exponential random starting point
	tt=np.random.beta(2.5, 1, len(LO)) # beta random starting point
	# te=LO-(.025*LO) IMPROVE INIT
	te=LO*tt
	if frac1==0: ts, te= FA,LO
	return ts, te

def init_BD(n):
	return np.random.exponential(.5, max(n-1,1))+.1

def init_times(m,time_framesL,time_framesM, tip_age):
	timesL=np.linspace(m,tip_age,time_framesL+1)
	timesM=np.linspace(m,tip_age,time_framesM+1)
	timesM[1:time_framesM] +=1
	timesL[time_framesL] =0
	timesM[time_framesM] =0
	return timesL, timesM

def init_alphas(): # p=1 for alpha1=alpha2
	return array([np.random.uniform(1,5),np.random.uniform(0,1)]),np.zeros(3)	

########################## UPDATES ######################################
def update_parameter(i, m, M, d, f): 
	#d=fabs(np.random.normal(d,d/2.)) # variable tuning prm
	if i>0 and rand.random()<=f:
		ii = i+(rand.random()-.5)*d
		if ii<m: ii=(ii-m)+m
		if ii>M: ii=(M-(ii-M))
		if ii<m: ii=i
	else: ii=i
	return ii

def update_parameter_normal(i, m, M, d):
	if d>0: ii = np.random.normal(i,d)
	else: ii=i
	if ii<m: ii=(ii-m)+m
	if ii>M: ii=(M-(ii-M))
	if ii<m: ii=i
	return ii

def update_rates(L,M,tot_L,mod_d3):
	Ln=zeros(len(L))
	Mn=zeros(len(M))
	#if random.random()>.5:
	for i in range(len(L)): Ln[i]=update_parameter(L[i],0, inf, mod_d3, f_rate)
	#else:
	for i in range(len(M)): Mn[i]=update_parameter(M[i],0, inf, mod_d3, f_rate)
	#Ln,Mn=Ln * scale_factor/tot_L , Mn * scale_factor/tot_L
	return Ln,Mn

def update_times(times, root, mod_d4):
	rS= zeros(len(times))
	rS[0]=root
	for i in range(1,len(times)): rS[i]=update_parameter(times[i],0, root, mod_d4, 1)
	y=sort(-rS)
	y=-1*y
	return y

def update_ts_te(ts, te, d1):
	tsn, ten= zeros(len(ts)), zeros(len(te))
	f1=np.random.random_integers(1,frac1) #int(frac1*len(FA)) #-np.random.random_integers(0,frac1*len(FA)-1)) 
	ind=np.random.random_integers(0,len(ts)-1,f1) 
	tsn[ind]=np.random.uniform(-d1,d1,len(tsn[ind]))
	tsn = abs(tsn+ts)		                         # reflection at min boundary (0)
	ind=(tsn<FA).nonzero()					 # indices of ts<FA (not allowed)
	tsn[ind] = abs(tsn[ind]-FA[ind])+FA[ind]         # reflection at max boundary (FA)
	tsn=abs(tsn)

	ind=np.random.random_integers(0,len(ts)-1,f1) 
	ten[ind]=np.random.uniform(-d1,d1,len(ten[ind]))
	ten=abs(ten+te)		                         # reflection at min boundary (0)
	ind2=(ten>LO).nonzero()                          
	ten[ind2] = abs(LO[ind2]-(ten[ind2]-LO[ind2]))   # reflection at max boundary (LO)
	ind2=(ten>LO).nonzero()                          
	ten[ind2] = te[ind2]                             # reflection at max boundary (LO)
	ind=(LO==0).nonzero()                            # indices of LO==0 (extant species)
	ten[ind]=0
	return tsn,ten

def seed_missing(x,m,s): # assigns random normally distributed trait values to missing data
	return np.isnan(x)*np.random.normal(m,s)+np.nan_to_num(x)

########################## PRIORS #######################################
try: 
	scipy.stats.gamma.logpdf(1, 1, scale=1./1,loc=0) 
	def prior_gamma(L,a,b): 
		return scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0)
	def prior_normal(L,sd): 
		return scipy.stats.norm.logpdf(L,loc=0,scale=sd)

except(AttributeError): # for older versions of scipy
	def prior_gamma(L,a,b):  
		return (a-1)*log(L)+(-b*L)-(log(b)*(-a)+ log(gamma(a)))
	def prior_normal(L,sd): 
		return -(x**2/(2*sd**2)) - log(sd*sqrt(2*np.pi))


def prior_times_frames(t, root, tip_age,a): # un-normalized Dirichlet (truncated)
	diff_t, min_t = abs(np.diff(t)), min(t)
	if min(diff_t)<=min_allowed_t: return -inf
	elif (min_t<=tip_age+min_allowed_t) and (min_t>0): return -inf
	else: 
		t_rel=diff_t/root
		return (a-1)*log(t_rel)

def prior_beta(x,a): 
	# return log(x)*(a-1)+log(1-x)*(a-1) # un-normalized beta
	return scipy.stats.beta.logpdf(x, a,a)
	
def prior_root_age(root, max_FA, l): # exponential (truncated)
	l=1./l
	if root>=max_FA: return log(l)-l*(root)
	else: return -inf

def prior_uniform(x,m,M): 
	if x>m and x<M: return log(1./(M-m))
	else: return -inf

def G_density(x,a,b):
	#return (1./b)**a * 1./gamma(a) * x**(a-1) *exp(-(1./b)*x)
	return scipy.stats.gamma.pdf(x, a, scale=1./b,loc=0)

def logPERT4_density(M,m,a,b,x): # relative 'stretched' LOG-PERT density: PERT4 * (s-e)
	return log((M-x)**(b-1) * (-m+x)**(a-1)) - log ((M-m)**4 * f_beta(a,b))

def PERT4_density(M,m,a,b,x):  # relative 'stretched' PERT density: PERT4 * (s-e) 
	return ((M-x)**(b-1) * (-m+x)**(a-1)) /((M-m)**4 * f_beta(a,b))

def logPERT4_density5(M,m,a,b,x): # relative LOG-PERT density: PERT4
	return log((M-x)**(b-1) * (-m+x)**(a-1)) - log ((M-m)**5 * f_beta(a,b))

########################## LIKELIHOODS ##################################
def get_hyper_priorBD(timesL,timesM,L,M,T):	
	if tot_extant==-1:
		return sum(prior_gamma(L,L_lam_r,L_lam_m))+sum(prior_gamma(M,M_lam_r,M_lam_m))
	else:
		def pNtvar(arg):
			T=arg[0]
			L=arg[1]
			M=arg[2]
			N=arg[3]
			Dt=-np.diff(T)
		        r_t = (L - M)*Dt
			Beta=  sum(exp((L - M)*Dt))
			Alpha= sum(L*exp((L - M)*Dt))
			lnBeta=  log(sum(exp((L - M)*Dt)))
			lnAlpha= log(sum(L*exp((L - M)*Dt)))
			#P   = (Beta/(Alpha*(1+Alpha))) *    (Alpha/(1+Alpha))**N
			if N>0: lnP = (lnBeta-(lnAlpha+log(1+Alpha))) + (lnAlpha-log(1+Alpha))*N
			else:	lnP = log((1+Alpha-Beta)/(Alpha*(1+Alpha)))
			return lnP

		### HYPER-PRIOR BD ###
		n0=1
		timesM[1:len(timesM)-1] = timesM[1:len(timesM)-1] +0.0001
		#if len(timesM)>2: 
		all_t_frames=sort(np.append(timesL, timesM[1:len(timesM)-1] ))[::-1] # merge time frames		

		#all_t_frames=sort(np.append(timesL, timesM[1:-1]+.001 ))[::-1] # merge time frames
		#else: all_t_frames=sort(np.append(timesL, timesM[1:-1] ))[::-1] # merge time frames
		sL=(np.in1d(all_t_frames,timesL[1:-1])+0).nonzero()[0] # indexes within 'all_t_frames' of shifts of L
		sM=(np.in1d(all_t_frames,timesM[1:-1])+0).nonzero()[0] # indexes within 'all_t_frames' of shifts of M
		sL[(sL-1>len(M)-1).nonzero()]=len(M)
		sM[(sM-1>len(L)-1).nonzero()]=len(L)
	
		nL=zeros(len(all_t_frames)-1)
		nM=zeros(len(all_t_frames)-1)
	
		Ln=insert(L,sM,L[sM-1]) # l rates for all_t_frames
		Mn=insert(M,sL,M[sL-1]) # m rates for all_t_frames
	
		return pNtvar([all_t_frames,Ln,Mn,tot_extant])
		
def BD_partial_lik(arg):
	[ts,te,up,lo,rate,lam_r,lam_m,lam_s, par, cov_par,q,m0]=arg
	# no. spec./ext. events
	# ts_inframe = np.intersect1d(ts[ts <= up], ts[ts > lo])
	# te_inframe = np.intersect1d(te[te <= up], te[te > lo])
	
	# indexes of the species within time frame
	if par=="l": i_events=np.intersect1d((ts <= up).nonzero()[0], (ts > lo).nonzero()[0])
	else: i_events=np.intersect1d((te <= up).nonzero()[0], (te > lo).nonzero()[0])
	# index of extant/extinct species
	# extinct_sp=(te > 0).nonzero()[0]
	# present_sp=(te == 0).nonzero()[0]
		
	# index species present in time frame
	n_all_inframe = np.intersect1d((ts >= lo).nonzero()[0], (te <= up).nonzero()[0])

	# tot br length within time frame
	n_t_ts,n_t_te=zeros(len(ts)),zeros(len(ts))

	n_t_ts[n_all_inframe]= ts[n_all_inframe]   # speciation events before time frame
	n_t_ts[(n_t_ts>up).nonzero()]=up           # for which length is accounted only from $up$ rather than from $ts$
	
	n_t_te[n_all_inframe]= te[n_all_inframe]   # extinction events in time frame
	n_t_te[np.intersect1d((n_t_te<lo).nonzero()[0], n_all_inframe)]=lo     # for which length is accounted only until $lo$ rather than to $te$

	# vector of br lengths within time frame  #(scaled by rho)
	n_S=((n_t_ts[n_all_inframe]-n_t_te[n_all_inframe])) #*rhos[n_all_inframe])

	if cov_par !=0: # covaring model: $r$ is vector of rates tranformed by trait
		r=exp(log(rate)+cov_par*(con_trait-parGAUS[0])) # exp(log(rate)+cov_par*(con_trait-mean(con_trait[all_inframe])))
		lik= sum(log(r[i_events])) + sum(-r[n_all_inframe]*n_S) #, cov_par
	else:           # constant rate model
		r=np.repeat(rate,len(i_events)) #*rhos[i_events]
		lik= sum(log(r)) + sum(-rate*n_S)
	return lik

def HOMPP_lik(arg):
	[m,M,shapeGamma,q_rate,i,cov_par, ex_rate]=arg
	i=int(i)
	x=fossil[i]
	lik=0
	k=len(x[x>0]) # no. fossils for species i
	if cov_par ==2: # transform preservation rate by trait value
		q=exp(log(q_rate)+cov_par*(con_trait[i]-parGAUS[0]))
	else: q=q_rate
	if argsG is True:
		YangGamma=get_gamma_rates(shapeGamma)
		qGamma= YangGamma*q
		lik1= -qGamma*(M-m) + log(qGamma)*k
		lik2= lik1-max(lik1)
		lik=log(sum(exp(lik2)*(1./args.ncat)))+max(lik1)
		return lik
	else: 	return -q*(M-m) + log(q)*k

def NHPP_lik(arg):
	[m,M,nothing,q_rate,i,cov_par, ex_rate]=arg
	i=int(i)
	x=fossil[i]
	lik=0
	k=len(x[x>0]) # no. fossils for species i
	x=sort(x)[::-1] # reverse
	xB1= -(x-M) # distance fossil-ts
	c=.5
	if cov_par !=0: # transform preservation rate by trait value
		q=exp(log(q_rate)+cov_par*(con_trait[i]-parGAUS[0]))
	else: q=q_rate

	if m==0: # data augmentation
		l=1./ex_rate
		#quant=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
		quant=[.125,.375,.625,.875] # quantiles gamma distribution (predicted te)
		#quant=[.5] 
		GM= -(-log(1-np.array(quant))/ex_rate) # get the x values from quantiles (exponential distribution)
		# GM=-array([gdtrix(ex_rate,1,jq) for jq in quant]) # get the x values from quantiles
		z=np.append(GM, [GM]*(k)).reshape(k+1,len(quant)).T
		xB=xB1/-(z-M) # rescaled fossil record from ts-te_DA to 0-1
		C=M-c*(M-GM)  # vector of modal values
		a = 1 + (4*(C-GM))/(M-GM) # shape parameters a,b=3,3
		b = 1 + (4*(-C+M))/(M-GM) # values will change in future implementations
		#print M, GM
		int_q = betainc(a,b,xB[:,k])* (M-GM)*q    # integral of beta(3,3) at time xB[k] (tranformed time 0)
		MM=np.zeros((len(quant),k))+M     # matrix speciation times of length quant x no. fossils
		aa=np.zeros((len(quant),k))+a[0]  # matrix shape parameters (3) of length quant x no. fossils
		bb=np.zeros((len(quant),k))+b[0]  # matrix shape parameters (3) of length quant x no. fossils
		X=np.append(x[x>0],[x[x>0]]*(len(quant)-1)).reshape(len(quant), k) # matrix of fossils of shape quant x no. fossils		
		if len(quant)>1:
			den = sum(G_density(-GM,1,l)) + small_number
			lik_temp= sum(exp(-(int_q) + np.sum((logPERT4_density(MM,z[:,0:k],aa,bb,X)+log(q)), axis=1) ) \
			* (G_density(-GM,1,l)/den) / (1-exp(-int_q))) / len(GM)			
			if lik_temp>0: lik=log(lik_temp)
			else: lik = -inf
		else: lik= sum(-(int_q) + np.sum((logPERT4_density(MM,z[:,0:k],aa,bb,X)+log(q)), axis=1))
	else:
		C=M-c*(M-m)
		a = 1+ (4*(C-m))/(M-m)
		b = 1+ (4*(-C+M))/(M-m)
		lik = -q*(M-m) + sum(logPERT4_density(M,m,a,b,x)+log(q)) - log(1-exp(-q*(M-m)))
	return lik

def NHPPgamma(arg):
	[m,M,shapeGamma,q_rate,i,cov_par, ex_rate]=arg
	i=int(i)
	x=fossil[i]

	k=len(x[x>0])   # no. fossils for species i
	x=sort(x)[::-1] # reverse
	xB1= -(x-M)     # distance fossil-ts

	if cov_par ==2: # transform preservation rate by trait value
		q=exp(log(q_rate)+cov_par*(con_trait[i]-parGAUS[0]))
	else: q=q_rate

	YangGamma=get_gamma_rates(shapeGamma)
	qGamma=YangGamma*q
	c=.5
	if len(x)>1 and m>0:
		C=M-.5*(M-m)
		a = 1+ (4*(C-m))/(M-m)
		b = 1+ (4*(-C+M))/(M-m)
		W=PERT4_density(M,m,a,b,x)
		PERT4_den=np.append(W, [W]*(args.ncat-1)).reshape(args.ncat,len(W)).T 
		#lik=log( sum( (exp(-qGamma*(M-m)) * np.prod((PERT4_den*qGamma), axis=0) / (1-exp(-qGamma*(M-m))))*(1./args.ncat)) )
		tempL=exp(-qGamma*(M-m))
		if max(tempL)<1:
			L=log(1-tempL)
			if np.isfinite(sum(L)):
				lik1=-qGamma*(M-m) + np.sum(log(PERT4_den*qGamma), axis=0) - L
				lik2=lik1-max(lik1)
				lik=log(sum(exp(lik2)*(1./args.ncat)))+max(lik1)
			else: lik=-100000
		else: lik=-100000
		
	else: lik=NHPP_lik(arg)
	return lik



def born_prm(times, R, ind, tse):
	#B=max(1./times[0], np.random.beta(1,(len(R)+1))) # avoid time frames < 1 My
	#B=np.random.beta(1,(len(R)+1))
	alpha=zeros(len(R)+1)+lam_s
	B=max(1./times[0], np.random.dirichlet(alpha,1)[0][0])
	Q=np.diff(times*(1-B))
	ADD=-B*times[0]
	Q1=insert(Q, ind,ADD)
	Q2=times[0]+cumsum(Q1)
	Q3=insert(Q2,0,times[0])
	Q3[len(Q3)-1]=0
	n_R= insert(R, ind, init_BD(1))
	#print "old", R, n_R,
	
	# better proposals for birth events
	#if len(R)>1: R_init=mean(R)
	#else: R_init=R
	#n_R= insert(R, ind,update_parameter(R_init,0,5,.2,1))
	
	
	#print "new", n_R
	n_times= sort(Q3)[::-1]
	go=True
	for j in range(len(n_times)-1):
		up,lo=n_times[j],n_times[j+1]
		if len(np.intersect1d((tse <= up).nonzero()[0], (tse > lo).nonzero()[0]))<1: 
			go=False
			#print len(np.intersect1d((tse <= up).nonzero()[0], (tse > lo).nonzero()[0])), up , lo

	if min(abs(np.diff(n_times)))<=1: return times, R
	elif go is False: return times, R
	else: return n_times, n_R

	#R= insert(R, ind, init_BD(1))
	#n_times= sort(Q3)[::-1]
	#return n_times, R

def kill_prm(times, R, ind):
	P=np.diff(times)         # time intervals
	Pi= abs(P[ind]/times[0]) 	
	P2= P/(1-Pi)
	P3=np.delete(P2, ind)    # remove interval
	Q2=times[0]+cumsum(P3)   # re-adjust length remaining time frames
	Q3=insert(Q2,0,times[0]) # add root
	Q3[len(Q3)-1]=0
	R=np.delete(R,ind)       # remove corresponding rate
	n_times= sort(Q3)[::-1]  # reverse array
	return n_times, R

def estimate_delta(likBDtemp, R,par,times, ts, te, cov_par, ind,deathRate,n_likBD,q,m0):
	# ESTIMATE DELTAS (individual death rates)		
	for temp in range(0,len(R)):
		if par=="l": 
			temp_l=temp
			cov_par_one=cov_par[0]
		else: 
			temp_l=temp+ind
			cov_par_one=cov_par[1]
		n_times, n_rates=kill_prm(times, R, temp)

		tempL=0
		for temp1 in range(len(n_times)-1):
			up, lo = n_times[temp1], n_times[temp1+1]
			l = n_rates[temp1]
			args=[ts, te, up, lo, l, L_lam_r,L_lam_m,lam_s, par, cov_par_one,q,m0]
			tempL+=BD_partial_lik(args)
		#print "LIK", 	tempL, sum(likBDtemp[ind:ind+len(R)])
		D=min(tempL-sum(likBDtemp[ind:ind+len(R)]), 100) # to avoid overflows
		deathRate[temp_l]=exp(D)
	return deathRate #, n_likBD


def Alg_3_1(arg):
	[it,likBDtemp, ts, te, L,M, timesL, timesM, cov_par,q,m0]=arg
	cont_time=0
	#T=max(ts)
	#priorBD= get_hyper_priorBD(timesL,timesM,L,M,T)	
	while cont_time<len_cont_time:
		#print cont_time, sum(likBDtemp), len(L),len(M) #, timesL, L
		deathRate=zeros(len(likBDtemp))
		n_likBD=zeros(len(likBDtemp))
		
		# ESTIMATE DELTAS (individual death rates)		
		if len(L)>1:  # SPECIATION RATES
			deathRate=estimate_delta(likBDtemp, L,"l",timesL, ts, te, cov_par, 0,deathRate,n_likBD,q,M[len(M)-1])
		if len(M)>1:  # EXTINCTION RATES
			deathRate=estimate_delta(likBDtemp, M,"m",timesM, ts, te, cov_par, len(L),deathRate,n_likBD,q,M[len(M)-1])
		deltaRate=sum(deathRate)
		#print "DELTA:", deltaRate, "\t", deathRate
		cont_time += np.random.exponential(1./min((deltaRate+birthRate), 100000))
		if cont_time>len_cont_time: break
			
		else: # UPDATE MODEL
			Pr_birth= birthRate/(birthRate+deltaRate)
			Pr_death= 1-Pr_birth
			#IND=-1
			#print sum(likBDtemp), Pr_birth, Pr_death, deathRate
			if rand.random()<Pr_birth or len(L)+len(M)==2: # ADD PARAMETER
				LL=len(L)+len(M)
				if rand.random()>.5:
					ind=np.random.random_integers(0,len(L))
					timesL, L = born_prm(timesL, L, ind, ts)
					IND=ind
				else: 
					ind=np.random.random_integers(0,len(M))
					timesM, M = born_prm(timesM, M, ind, te)
					IND=ind+len(timesL)-1
				if LL == len(L)+len(M): IND=-1
			else: # REMOVE PARAMETER
				probDeath=np.cumsum(deathRate/deltaRate) # cumulative prob (used to randomly sample one 
				r=rand.random()                          # parameter based on its deathRate)
				probDeath=sort(append(probDeath, r))
				ind=np.where(probDeath==r)[0][0] # just in case r==1
				if ind < len(L): timesL, L = kill_prm(timesL, L, ind)
				else: timesM, M = kill_prm(timesM, M, ind-len(L))
			# UPDATE LIKELIHOODS
			tempL=zeros(len(L)+len(M))
			tempP=zeros(len(L)+len(M))
			for temp_l in range(len(timesL)-1):
				up, lo = timesL[temp_l], timesL[temp_l+1]
				l = L[temp_l]
				args=[ts, te, up, lo, l, L_lam_r,L_lam_m,lam_s, 'l', cov_par[0],q,M[len(M)-1]]
				tempL[temp_l]=BD_partial_lik(args)
			for temp_m in range(len(timesM)-1):
				up, lo = timesM[temp_m], timesM[temp_m+1]
				m = M[temp_m]
				args=[ts, te, up, lo, m, M_lam_r,M_lam_m,lam_s, 'm', cov_par[1],q,M[len(M)-1]]
				tempL[len(timesL)-1+temp_m]=BD_partial_lik(args)
			likBDtemp=tempL
			
			#priorBDnew= get_hyper_priorBD(timesL,timesM,L,M,T)-priorBD
			#print IND, timesL, timesM
			#if IND > -1: likBDtemp[IND] += priorBDnew
			

	#if priorBDnew-priorBD >= log(rand.random()):
	#	return likBDtemp, L,M, timesL, timesM, cov_par
	#else:
	#	return arg[1],arg[4], arg[5],arg[6], arg[7],cov_par
	return likBDtemp, L,M, timesL, timesM, cov_par
		




########################## MCMC #########################################

def MCMC(all_arg):
	[it,n_proc, I,sample_freq, print_freq, temperatures, burnin, marginal_frames, arg]=all_arg
	if it==0: # initialize chain
		print "initializing chain..."
		tsA, teA = init_ts_te(FA,LO)
		if global_stop_update is True: tsA, teA = globalTS, globalTE
		timesLA, timesMA = init_times(max(tsA),time_framesL,time_framesM, min(teA))
		if len(fixed_times_of_shift)>0: timesLA[1:-1],timesMA[1:-1]=fixed_times_of_shift,fixed_times_of_shift
		LA = init_BD(len(timesLA))
		MA = init_BD(len(timesMA))
		alphasA,cov_parA = init_alphas() # use 1 for symmetric PERT
		if argsG is False: alphasA[0]=1
		SA=sum(tsA-teA)

	else: # restore values
		[itt, n_proc_,PostA, likA, priorA,tsA,teA,timesLA,timesMA,LA,MA,alphasA, cov_parA, lik_fossilA,likBDtempA]=arg
		SA=sum(tsA-teA)

	# start threads
	if num_processes>0: pool_lik = multiprocessing.Pool(num_processes) # likelihood
	if frac1>=0 and num_processes_ts>0: pool_ts = multiprocessing.Pool(num_processes_ts) # update ts, te
	tmp, marginal_lik, lik_tmp=0, zeros(len(temperatures)), 0

	while it<I:
		I_effective=I-burnin
		#if it==0: print (I_effective/len(temperatures))
		if it>0 and (it-burnin) % (I_effective/len(temperatures)) == 0 and it>burnin or it==I-1: # and it<I:
			if TDI==1:
				marginal_lik[tmp]=lik_tmp/((I_effective/len(temperatures))/sample_freq) # mean lik: Baele et al. 2012
				if it<I-1:
					tmp += 1
					lik_tmp=0
		temperature=temperatures[tmp]
		
		# update parameters
		ts,te=tsA, teA
		timesL,timesM=timesLA,timesMA
		
		# GLOBALLY CHANGE TRAIT VALUE
		if model_cov >0:
			global con_trait
			con_trait=seed_missing(trait_values,meanGAUS,sdGAUS)
			
		if global_stop_update is True: 
			rr=random.uniform(f_update_q,1)
			stop_update=0
			tsA, teA= globalTS, globalTE
			lik_fossilA=np.zeros(1)
		elif rand.random() < 1./freq_Alg_3_1 and it>start_Alg_3_1 and TDI==2:
			stop_update=inf
			rr=1.5 # no updates
		else:
			rr=random.uniform(0,1) #random.uniform(.8501, 1)
			stop_update=I+1

		alphas=zeros(2)
		cov_par=zeros(3)
		L,M=zeros(len(LA)),zeros(len(MA))
		tot_L=sum(tsA-teA)

		# autotuning
		if TDI != 1: tmp=0
		mod_d1= d1           # window size ts, te
		mod_d3= list_d3[tmp] # window size rates
		mod_d4= list_d4[tmp] # window size shift times
		
        
		if rr<f_update_se: # ts/te
			ts,te=update_ts_te(tsA,teA,mod_d1)
			tot_L=sum(ts-te)
		elif rr<f_update_q: # q/alpha
			if rand.random()>.5 and argsG is True: alphas[0]=update_parameter(alphasA[0],0,20,d2[1],1) # shape prm Gamma
			else: alphas[1]=update_parameter(alphasA[1],0,inf,d2[0],1) #  fossilization rate (q)
			
		elif rr < f_update_lm: # l/m
			if rand.random()<f_shift and len(LA)+len(MA)>2: 
				timesL=update_times(timesLA, max(ts),mod_d4)
				timesM=update_times(timesMA, max(ts),mod_d4)
			else: L,M=update_rates(LA,MA,3,mod_d3)

		elif rr<f_update_cov: # cov
			rcov=rand.random()
			if rcov < f_cov_par[0]: # cov lambda
				cov_par[0]=update_parameter_normal(cov_parA[0],-3,3,d5[0])
			elif rcov < f_cov_par[1]: # cov mu
				cov_par[1]=update_parameter_normal(cov_parA[1],-3,3,d5[1])
			else:
				cov_par[2]=update_parameter_normal(cov_parA[2],-3,3,d5[2])

		if constrain_time_frames is True: timesM=timesL
		alphas[(alphas==0).nonzero()]=alphasA[(alphas==0).nonzero()]
		L[(L==0).nonzero()]=LA[(L==0).nonzero()]
		M[(M==0).nonzero()]=MA[(M==0).nonzero()]
		cov_par[(cov_par==0).nonzero()]=cov_parA[(cov_par==0).nonzero()]
		timesL[0]=max(ts)
		timesM[0]=max(ts)
				
		# NHPP Lik: multi-thread computation (ts, te)
		# generate args lik (ts, te)
		ind1=range(0,len(fossil))
		ind2=[]
		if it>0 and rr<f_update_se: # recalculate likelihood only for ts, te that were updated
			ind1=(ts-te != tsA-teA).nonzero()[0]
			ind2=(ts-te == tsA-teA).nonzero()[0]
		lik_fossil=zeros(len(fossil))

		if len(ind1)>0 and it<stop_update:
			# generate args lik (ts, te)
			z=zeros(len(fossil)*7).reshape(len(fossil),7)
			z[:,0]=te
			z[:,1]=ts
			z[:,2]=alphas[0]   # shape prm Gamma
			z[:,3]=alphas[1]   # baseline foss rate (q)
			z[:,4]=range(len(fossil))
			z[:,5]=cov_par[2]  # covariance baseline foss rate
			z[:,6]=M[len(M)-1] # ex rate
			args=list(z[ind1])
			if num_processes_ts==0:
				for j in range(len(ind1)):
					i=ind1[j] # which species' lik
					if argsHPP is True or  frac1==0: lik_fossil[i] = HOMPP_lik(args[j])
					elif argsG is True: lik_fossil[i] = NHPPgamma(args[j]) 
					else: lik_fossil[i] = NHPP_lik(args[j])
			else:
				if argsHPP is True or  frac1==0: lik_fossil[ind1] = array(pool_ts.map(HOMPP_lik, args))
				elif argsG is True: lik_fossil[ind1] = array(pool_ts.map(NHPPgamma, args)) 
				else: lik_fossil[ind1] = array(pool_ts.map(NHPP_lik, args))

		
		if it>0: lik_fossil[ind2] = lik_fossilA[ind2]
		if it>=stop_update or stop_update==inf: lik_fossil = lik_fossilA

		# pert_prior defines gamma prior on alphas[1] - fossilization rate
		prior = prior_gamma(alphas[1],pert_prior[0],pert_prior[1]) + prior_uniform(alphas[0],0,20)

		# Birth-Death Lik: construct 2D array (args partial likelihood)
		# parameters of each partial likelihood and prior (l)
		if stop_update != inf:
			args=list()
			for temp_l in range(len(timesL)-1):
				up, lo = timesL[temp_l], timesL[temp_l+1]
				l = L[temp_l]
				args.append([ts, te, up, lo, l, L_lam_r,L_lam_m,lam_s, 'l', cov_par[0],alphas[1],M[len(M)-1]])			
			# parameters of each partial likelihood and prior (m)
			for temp_m in range(len(timesM)-1):
				up, lo = timesM[temp_m], timesM[temp_m+1]
				m = M[temp_m]
				args.append([ts, te, up, lo, m, M_lam_r,M_lam_m,lam_s, 'm', cov_par[1],alphas[1],M[len(M)-1]])
	
			
			if num_processes==0:
				likBDtemp=np.zeros(len(args))
				i=0
				for i in range(len(args)):
					likBDtemp[i]=BD_partial_lik(args[i])
					i+=1
			# multi-thread computation of lik and prior (rates)
			else: likBDtemp = array(pool_lik.map(BD_partial_lik, args))

			lik= sum(lik_fossil) + sum(likBDtemp)

		else: # run BD algorithm (Alg. 3.1)
			sys.stderr = NO_WARN
			args=[it, likBDtempA,tsA, teA, LA,MA, timesLA, timesMA, cov_parA,alphas[1],M[len(M)-1]]
			likBDtemp, L,M, timesL, timesM, cov_par = Alg_3_1(args)
			
			# NHPP Lik: needs to be recalculated after Alg 3.1
			# generate args lik (ts, te)
			ind1=range(0,len(fossil))
			lik_fossil=zeros(len(fossil))
			# generate args lik (ts, te)
			z=zeros(len(fossil)*7).reshape(len(fossil),7)
			z[:,0]=te
			z[:,1]=ts
			z[:,2]=alphas[0]   # shape prm Gamma
			z[:,3]=alphas[1]   # baseline foss rate (q)
			z[:,4]=range(len(fossil))
			z[:,5]=cov_par[2]  # covariance baseline foss rate
			z[:,6]=M[len(M)-1] # ex rate
			args=list(z[ind1])
			if num_processes_ts==0:
				for j in range(len(ind1)):
					i=ind1[j] # which species' lik
					if argsHPP is True or  frac1==0: lik_fossil[i] = HOMPP_lik(args[j])
					elif argsG is True: lik_fossil[i] = NHPPgamma(args[j]) 
					else: lik_fossil[i] = NHPP_lik(args[j])
			else:
				if argsHPP is True or frac1==0: lik_fossil[ind1] = array(pool_ts.map(HOMPP_lik, args))
				elif argsG is True: lik_fossil[ind1] = array(pool_ts.map(NHPPgamma, args))
				else: lik_fossil[ind1] = array(pool_ts.map(NHPP_lik, args))
			
			sys.stderr = original_stderr
			
			lik= sum(lik_fossil) + sum(likBDtemp)

		T= max(ts)
		prior += sum(prior_times_frames(timesL, max(tsA),min(teA), lam_s))
		prior += sum(prior_times_frames(timesM, max(tsA),min(teA), lam_s))

		priorBD= get_hyper_priorBD(timesL,timesM,L,M,T)
		prior += priorBD
		###
		if model_cov >0: prior+=sum(prior_normal(cov_par,covar_prior))

		# exponential prior on root age
		prior += prior_root_age(max(ts),max(FA),max(FA))
		
		if temperature==1: 
			tempMC3=1./(1+n_proc*temp_pr)
			lik_alter=lik
		else: 
			tempMC3=1
			lik_alter=sum(lik_fossil) + sum(likBDtemp)*temperature
		Post=lik_alter+prior
		if it==0: PostA=Post
		#print Post, PostA, alphasA #, lik, likA
		if Post>-inf and Post<inf:
			if Post*tempMC3-PostA*tempMC3 >= log(rand.random()) or stop_update==inf: # or it==0:
				likBDtempA=likBDtemp
				PostA=Post
				priorA=prior
				likA=lik
				timesLA=timesL
				timesMA=timesM
				LA,MA=L,M
				tsA,teA=ts,te
				SA=sum(tsA-teA)
				alphasA=alphas
				lik_fossilA=lik_fossil
				cov_parA=cov_par
			
		if it % print_freq ==0 or it==burnin:
			l=[round(y, 2) for y in [PostA, likA, priorA, SA]]
			if it>burnin and n_proc==0:
				print_out= "\n%s\tpost: %s lik: %s (%s, %s) prior: %s tot.l: %s" \
				% (it, l[0], l[1], round(sum(lik_fossilA), 2), round(sum(likBDtempA), 2),l[2], l[3])
				if TDI==1: print_out+=" beta: %s" % (round(temperature,4))
				if TDI==2: print_out+=" k: %s" % (len(LA)+len(MA))
				print print_out
				#if TDI==1: print "\tpower posteriors:", marginal_lik[0:10], "..."

				print "\tt.frames:", timesLA, "(sp.)"
				print "\tt.frames:", timesMA, "(ex.)"
				print "\tsp.rates:", LA, "\n\tex.rates:", MA
				
				if model_cov>=1: print "\tcov. (sp/ex/q):", cov_parA
 				print "\tq.rate:", round(alphasA[1], 3), "\tGamma.prm:", round(alphasA[0], 3)
				print "\tts:", tsA[0:5], "..."
				print "\tte:", teA[0:5], "..."
			if it<=burnin and n_proc==0: print "\n%s*\tpost: %s lik: %s prior: %s tot length %s" \
			% (it, l[0], l[1], l[2], l[3])

		if n_proc != 0: pass
		elif it % sample_freq ==0 and it>=burnin or it==0 and it>=burnin:
			s_max=max(tsA)
			if TDI<2: # normal MCMC or MCMC-TI
				log_state= [it,PostA, priorA, sum(lik_fossilA), likA-sum(lik_fossilA), alphasA[1], alphasA[0], cov_parA[0], cov_parA[1],cov_parA[2], temperature, s_max]
				log_state += list(LA)
				log_state += list(timesLA[1:-1])
				log_state += list(timesMA[1:-1])
			else: # BD-MCMC
				log_state= [it,PostA, priorA, sum(lik_fossilA), likA-sum(lik_fossilA), alphasA[1], alphasA[0], cov_parA[0], cov_parA[1],cov_parA[2], len(LA), len(MA), s_max]
                
			log_state += [SA]
			log_state += list(tsA)
			log_state += list(teA)
			wlog.writerow(log_state)
			logfile.flush()
			os.fsync(logfile)
			##else:
			#if TDI<2: # normal MCMC or MCMC-TI
			#	log_state="\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" \
			#	% (it,PostA, priorA, sum(lik_fossilA), likA-sum(lik_fossilA), alphasA[1], alphasA[0], cov_parA[0], cov_parA[1],cov_parA[2], temperature, s_max)
			#	for i in LA: log_state += "\t%s" % (i)
			#	for i in MA: log_state += "\t%s" % (i)			
			#	for i in range(1,time_framesL): log_state += "\t%s" % (timesLA[i])
			#	for i in range(1,time_framesM): log_state += "\t%s" % (timesMA[i])
			#else: # BD-MCMC
			#	log_state="\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" \
			#	% (it,PostA, priorA, sum(lik_fossilA), likA-sum(lik_fossilA), alphasA[1], alphasA[0], cov_parA[0], cov_parA[1],cov_parA[2], len(LA), len(MA), s_max)				
                        #
			#log_state += "\t%s" % (SA) 
			#for i in range(len(FA)): log_state += "\t%s" % (tsA[i])
			#for i in range(len(LO)): log_state += "\t%s" % (teA[i])
			##wlog.writerow()
			#logfile.writelines(log_state)
			#logfile.flush()
			#os.fsync(logfile)
			

			lik_tmp += sum(likBDtempA)
			if TDI !=1 and n_proc==0:
				margL=zeros(len(marginal_frames))
				margM=zeros(len(marginal_frames))
				for i in range(len(timesLA)-1): # indexes of the 1My bins within each timeframe
					ind=np.intersect1d(marginal_frames[marginal_frames<=timesLA[i]],marginal_frames[marginal_frames>=timesLA[i+1]])
					j=array(ind)
					margL[j]=LA[i]
				for i in range(len(timesMA)-1): # indexes of the 1My bins within each timeframe
					ind=np.intersect1d(marginal_frames[marginal_frames<=timesMA[i]],marginal_frames[marginal_frames>=timesMA[i+1]])
					j=array(ind)
					margM[j]=MA[i]
				marginal_rates(it, margL, margM, marginal_file, n_proc)
		it += 1
	if TDI==1 and n_proc==0: marginal_likelihood(marginal_file, marginal_lik, temperatures)
	if use_seq_lik is False:
		pool_lik.close()
		pool_lik.join()
		if frac1>=0: 
			pool_ts.close()
			pool_ts.join()
	return [it, n_proc,PostA, likA, priorA,tsA,teA,timesLA,timesMA,LA,MA,alphasA, cov_parA,lik_fossilA,likBDtempA]

def marginal_rates(it, margL,margM, marginal_file, run):
	#print len(margL), len(margM)
	#log_state="%s\t" % (it)
	#for i in margL: log_state += "%s\t" % (i)
	#for i in margM: log_state += "%s\t" % (i)
	#for i in range(len(margL)): log_state += "%s\t" % (margL[i]-margM[i])
	#for i in margM: log_state += "%s\t" % (i)
	#for i in range(len(margL)): log_state += "%s\t" % (margL[i]-margM[i])
	##marginal_file.write(log_state)
	#log_state=log_state.split('\t')
	log_state= [it]
	log_state += list(margL)
	log_state += list(margM)
	log_state += list(margL-margM)
	#marginal_file.write(log_state)
	#log_state=log_state.split('\t')
	wmarg.writerow(log_state)
	marginal_file.flush()
	os.fsync(marginal_file)

def marginal_likelihood(marginal_file, l, t):
	mL=0
	for i in range(len(l)-1): mL+=((l[i]+l[i+1])/2.)*(t[i]-t[i+1]) # Beerli and Palczewski 2010
	print "\n Marginal likelihood:", mL
	o= "\n Marginal likelihood: %s\n\nlogL: %s\nbeta: %s" % (mL,l,t)
	marginal_file.writelines(o)
	marginal_file.close()


########################## PARSE ARGUMENTS #######################################
self_path=os.getcwd()
p = argparse.ArgumentParser() #description='<input file>') 

p.add_argument('-v',         action='version', version='%(prog)s')
p.add_argument('-cite',      help='print PyRate citation', action='store_true', default=False)
p.add_argument('input_data', metavar='<input file>', type=str,help='Input python file - see template',default=[],nargs='*')
p.add_argument('-j',         type=int, help='number of data set in input file', default=1, metavar=1)
p.add_argument('-trait',     type=int, help='number of trait for Cov model', default=1, metavar=1)
p.add_argument("-N",         type=float, help='number of exant species') 
p.add_argument("-wd",        type=str, help='path to working directory', default="")
p.add_argument("-out",       type=str, help='output tag', default="")
p.add_argument('-plot',      metavar='<input file>', type=str,help="Input 'marginal_rates.log file(s)", nargs='+',default=[])
p.add_argument('-mProb',     type=str,help="Input 'mcmc.log file",default="")
p.add_argument('-BF',        type=str,help="Input 'marginal_likelihood.txt files",metavar='<2 input files>',nargs='+',default=[])

# MCMC SETTINGS
p.add_argument('-n',      type=int, help='mcmc generations',default=10000000, metavar=10000000)
p.add_argument('-s',      type=int, help='sample freq.', default=1000, metavar=1000)
p.add_argument('-p',      type=int, help='print freq.',  default=1000, metavar=1000)
p.add_argument('-b',      type=float, help='burnin', default=0, metavar=0)
p.add_argument('-thread', type=int, help='no. threads used for BD and NHPP likelihood respectively (set to 0 to bypass multi-threading)', default=[1,3], metavar=4, nargs=2)

# MCMC ALGORITHMS
p.add_argument('-A',   type=int, help='0) parameter estimation, 1) marginal likelihood, 2) BDMCMC', default=2, metavar=2)
p.add_argument('-r',   type=int,   help='MC3 - no. MCMC chains', default=1, metavar=1)
p.add_argument('-t',   type=float, help='MC3 - temperature', default=.03, metavar=.03)
p.add_argument('-sw',  type=float, help='MC3 - swap frequency', default=100, metavar=100)
p.add_argument('-M',   type=int,   help='BDMCMC - frequency of model update', default=10, metavar=10)
p.add_argument('-B',   type=int,   help='BDMCMC - birth rate', default=1, metavar=1)
p.add_argument('-T',   type=float, help='BDMCMC - time of model update', default=1.0, metavar=1.0)
p.add_argument('-S',   type=int,   help='BDMCMC - start model update', default=1000, metavar=1000)
p.add_argument('-k',   type=int,   help='TI - no. scaling factors', default=10, metavar=10)
p.add_argument('-a',   type=float, help='TI - shape beta distribution', default=.3, metavar=.3)

# PRIORS
p.add_argument('-pL',  type=float, help='Prior - speciation rate (Gamma <shape, rate>)', default=[1.1, 1.1], metavar=1.1, nargs=2)
p.add_argument('-pM',  type=float, help='Prior - extinction rate (Gamma <shape, rate>)', default=[1.1, 1.1], metavar=1.1, nargs=2)
p.add_argument('-pP',  type=float, help='Prior - preservation rate (Gamma <shape, rate>)', default=[1.5, 1.1], metavar=1.5, nargs=2)
p.add_argument('-pS',  type=float, help='Prior - time frames (Dirichlet <shape>)', default=2.5, metavar=2.5)
p.add_argument('-pC',  type=float, help='Prior - covariance parameters (Normal <standard deviation>)', default=1, metavar=1)

# MODEL
p.add_argument("-mHPP",  help='Model - Homogeneous Poisson process of preservation', action='store_true', default=False)
p.add_argument('-mL',    type=int, help='Model - no. (starting) time frames (speciation)', default=1, metavar=1)
p.add_argument('-mM',    type=int, help='Model - no. (starting) time frames (extinction)', default=1, metavar=1)
p.add_argument('-mC',    help='Model - constrain time frames (l,m)', action='store_true', default=False)
p.add_argument('-mCov',  type=int, help='COVAR model: 1) speciation, 2) extinction, 3) speciation & extinction, 4) preservation, 5) speciation & extinction & preservation', default=0, metavar=0)
p.add_argument("-mG",    help='Model - Gamma heterogeneity of preservation rate', action='store_true', default=False)
p.add_argument("-ncat",  type=int, help='Model - Number of categories for Gamma heterogeneity', default=4, metavar=4)
p.add_argument('-fixShift',metavar='<input file>', type=str,help="Input tab-delimited file",default="")
p.add_argument('-fixSE',metavar='<input file>', type=str,help="Input mcmc.log file",default="")

# TUNING
p.add_argument('-tT', type=float, help='Tuning - window size (ts, te)', default=1., metavar=1.)
p.add_argument('-nT', type=int,   help='Tuning - max number updated values (ts, te)', default=5, metavar=5)
p.add_argument('-tQ', type=float, help='Tuning - window sizes (q/alpha)', default=[0.33,3], nargs=2)
p.add_argument('-tR', type=float, help='Tuning - window size (rates)', default=.05, metavar=.05)
p.add_argument('-tS', type=float, help='Tuning - window size (time of shift)', default=1., metavar=1.)
p.add_argument('-fR', type=float, help='Tuning - fraction of updated values (rates)', default=1., metavar=1.)
p.add_argument('-fS', type=float, help='Tuning - fraction of updated values (shifts)', default=.7, metavar=.7)
p.add_argument('-tC', type=float, help='Tuning -window sizes cov parameters (l,m,q)', default=[.025, .025, .15], nargs=3)
p.add_argument('-fU', type=float, help='Tuning - update freq. (q/alpha,l/m,cov)', default=[.02, .18, .08], nargs=3)

args = p.parse_args()

if args.cite is True:
	sys.exit(citation)
############################ MODEL SETTINGS ############################
# PRIORS
L_lam_r,L_lam_m = args.pL # shape and scale parameters of Gamma prior on sp rates
M_lam_r,M_lam_m = args.pM # shape and scale parameters of Gamma prior on ex rates
lam_s = args.pS                              # shape parameter dirichlet prior on time frames
pert_prior = [args.pP[0],args.pP[1]] # gamma prior on foss. rate; beta on mode PERT distribution
covar_prior=args.pC # std of normal prior on th covariance parameters

# MODEL
time_framesL=args.mL          # no. (starting) time frames (lambda)
time_framesM=args.mM          # no. (starting) time frames (mu)
constrain_time_frames=args.mC # True/False
argsG=args.mG                 # gamma rates
if argsG is True:             # number of gamma categories
	YangGammaQuant=(np.linspace(0,1,args.ncat+1)-np.linspace(0,1,args.ncat+1)[1]/2)[1:]
model_cov=args.mCov           # boolean 0: no covariance 1: covariance (speciation,extinction) 2: covariance (speciation,extinction,preservation)

argsHPP=args.mHPP
############################ MCMC SETTINGS ############################
# GENERAL SETTINGS
TDI=args.A                  # 0: parameter estimation, 1: thermodynamic integration, 2: BD-MCMC
if constrain_time_frames is True or args.fixShift != "":
	if TDI==2:
		print "\nWarning: constrained shift times (-mC,-fixShift) cannot be used with BDMCMC alorithm. Using standard MCMC instead.\n"
		TDI = 0
mcmc_gen=args.n             # no. total mcmc generations
sample_freq=args.s
print_freq=args.p
burnin=args.b
num_processes = args.thread[0]    # BDlik
num_processes_ts = args.thread[1] # NHPPlik
if num_processes+num_processes_ts==0: use_seq_lik = True
if use_seq_lik is True: num_processes,num_processes_ts=0,0
min_allowed_t=1.

# TUNING
d1=args.tT                     # win-size (ts, te)
frac1= args.nT                 # max number updated values (ts, te)
d2=args.tQ                     # win-sizes (q,alpha)
d3=args.tR                     # win-size (rates)
f_rate=args.fR                 # fraction of updated values (rates)
d4=args.tS                     # win-size (time of shift)
f_shift=args.fS                # update frequency (time of shift) || will turn into 0 when no rate shifts
freq_list=args.fU              # generate update frequencies by parm category
d5=args.tC                     # win-size (cov)
if model_cov==0: freq_list[2]=0 
f_update_se=1-sum(freq_list)
if frac1==0: f_update_se=0
[f_update_q,f_update_lm,f_update_cov]=f_update_se+np.cumsum(array(freq_list))

# freq update CovPar
if model_cov==0: f_cov_par= [0  ,0  ,0 ]
if model_cov==1: f_cov_par= [1  ,0  ,0 ]
if model_cov==2: f_cov_par= [0  ,1  ,0 ]
if model_cov==3: f_cov_par= [.5 ,1  ,0 ]
if model_cov==4: f_cov_par= [0  ,0  ,1 ]
if model_cov==5: f_cov_par= [.33,.66,1 ]

if args.fixShift != "":          # fix times of rate shift
	try: 
		fixed_times_of_shift=sort(np.loadtxt(args.fixShift))[::-1]
		f_shift=0
		time_framesL=len(fixed_times_of_shift)+1
		time_framesM=len(fixed_times_of_shift)+1
		min_allowed_t=0
	except: 
		msg = "\nError in the input file %s.\n" % (args.fixShift)
		sys.exit(msg)
else: fixed_times_of_shift=[]

# BDMCMC & MCMC SETTINGS
runs=args.r              # no. parallel MCMCs (MC3)
if runs>1 and TDI>0: 
	print "\nWarning: MC3 algorithm is not available for TI and BDMCMC. Using a single chain instead.\n"
	runs=1
num_proc = runs          # processors MC3
temp_pr=args.t           # temperature MC3
IT=args.sw
freq_Alg_3_1=args.M      # frequency of model update
birthRate=args.B         # birthRate (=Poisson prior)
len_cont_time=args.T     # length continuous time of model update
start_Alg_3_1=args.S     # start sampling model after


if runs==1 or use_seq_lik is True: 
	IT=mcmc_gen

if TDI==1:                # Xie et al. 2011; Baele et al. 2012
	K=args.k-1.        # K+1 categories
	k=array(range(int(K+1)))
	beta=k/K
	alpha=args.a            # categories are beta distributed
	temperatures=list(beta**(1./alpha))
	temperatures[0]+= small_number # avoid exactly 0 temp
	temperatures.reverse()
	list_d3=sort(exp(temperatures))**2.5*d3+(exp(1-array(temperatures))-1)*d3
	list_d4=sort(exp(temperatures))**1.5*d4+exp(1-array(temperatures))-1
else:
	temperatures=[1]
	list_d3=[d3]
	list_d4=[d4]



############### PLOT RTT
list_files=sort(args.plot)
list_files_BF=sort(args.BF)

if len(list_files):
	input_file_raw = os.path.basename(list_files[0])
	stem_file = input_file_raw.split("marginal_rates.log")[0]
	if args.wd=="": 
		output_wd = os.path.dirname(list_files[0])
		if output_wd=="": output_wd= self_path
	else: output_wd=args.wd
	plot_RTT(list_files, stem_file, output_wd,burnin)
	quit()
elif args.mProb != "": calc_model_probabilities(args.mProb,burnin)
elif len(list_files_BF):
	if len(list_files_BF)<2: sys.exit("\n2 '*marginal_likelihood.txt' files required.\n")
	calc_BF(list_files_BF[0],list_files_BF[1])
	quit()
elif len(args.input_data)==0: sys.exit("\nInput file required. Use '-h' for command list.\n")


############################ LOAD INPUT DATA ############################
import imp
input_file_raw = os.path.basename(args.input_data[0])
input_file = os.path.splitext(input_file_raw)[0]  # file name without extension

if args.wd=="": 
	output_wd = os.path.dirname(args.input_data[0])
	if output_wd=="": output_wd= self_path
else: output_wd=args.wd

#print "\n",input_file, args.input_data, "\n"
try: input_data_module = imp.load_source(input_file, args.input_data[0])
except(IOError): sys.exit("\nInput file required. Use '-h' for command list.\n")

j=max(args.j-1,0)
try: fossil_complete=input_data_module.get_data(j)
except(IndexError): 
	fossil_complete=input_data_module.get_data(0)
	print "Warning: data set number %s not found. Using the first data set instead." % (args.j)
	j=0
fossil=list()
have_record=list()
for i in range(len(fossil_complete)):
	if len(fossil_complete[i])==1 and fossil_complete[i][0]==0: pass
	else: 
		have_record.append(i) # some (extant) species may have trait value but no fosil record
		fossil.append(fossil_complete[i])
		
out_name=input_data_module.get_out_name(j) +args.out

try: taxa_names=input_data_module.get_taxa_names()
except(AttributeError): 
	taxa_names=list()
	for i in range(len(fossil)): taxa_names.append("taxon_%s" % (i))

if argsG is True: out_name += "_G"
	
# Number of extant taxa (user specified)
if args.N>-1: tot_extant=args.N
else: 
	print "Number of extant species (-N) not specified. Using Gamma priors on the birth-death rates.\n"
	tot_extant = -1

FA,LO,N=np.zeros(len(fossil)),np.zeros(len(fossil)),np.zeros(len(fossil))
for i in range(len(fossil)):	
	FA[i]=max(fossil[i])
	LO[i]=min(fossil[i])
	N[i]=len(fossil[i])

if len(fixed_times_of_shift)>0: 
	fixed_times_of_shift=fixed_times_of_shift[fixed_times_of_shift<max(FA)]
	time_framesL=len(fixed_times_of_shift)+1
	time_framesM=len(fixed_times_of_shift)+1

if args.fixSE != "":          # fix TS, TE
	global_stop_update=True
	globalTS, globalTE= calc_ts_te(args.fixSE, burnin=args.b)
else: global_stop_update=False

# Get trait values (Cov model)
if model_cov>=1:
	try: trait_values=log(input_data_module.get_continuous(max(args.trait-1,0)))
	except: sys.exit("\nTrait data not found! Check input file.\n")
			
	MidPoints=np.zeros(len(fossil_complete))
	for i in range(len(fossil_complete)):
		MidPoints[i]=np.mean([max(fossil_complete[i]),min(fossil_complete[i])])
	
	# fit linear regression (for species with trait value - even if without fossil data)
	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(MidPoints[np.isfinite(trait_values)],trait_values[np.isfinite(trait_values)])
	
	# 
	ind_nan_trait= (np.isfinite(trait_values)==False).nonzero()
	meanGAUScomplete=np.zeros(len(fossil_complete))
	meanGAUScomplete[ind_nan_trait] = slope*MidPoints[ind_nan_trait] + intercept
	
	trait_values= trait_values[np.array(have_record)]
	meanGAUS= meanGAUScomplete[np.array(have_record)]
	
	sdGAUS = std_err
	regression_trait= "\n\nEstimated linear trend trait-value: \nslope=%s; sd. error= %s (intercept= %s; R2= %s; P-value= %s)" \
	% (round(slope,2), round(std_err,2), round(intercept,2), round(r_value,2), round(p_value,2))
	print regression_trait
	
	#print trait_values
	parGAUS=scipy.stats.norm.fit(trait_values[np.isfinite(trait_values)]) # fit normal distribution
	#global con_trait
	con_trait=seed_missing(trait_values,meanGAUS,sdGAUS) # fill the gaps (missing data)
	#print con_trait
	out_name += "_COV"

############################ MCMC OUTPUT ############################
try: os.mkdir(output_wd)
except(OSError): pass
path_dir = "%s/pyrate_mcmc_logs" % (output_wd)
folder_name="pyrate_mcmc_logs"
try: os.mkdir(path_dir) 
except(OSError): pass

suff_out=out_name
if TDI==1: suff_out+= "_TI"

# OUTPUT 0 SUMMARY AND SETTINGS
o0 = "\n%s build %s\n" % (version, build)
o1 = "\ninput: %s output: %s/%s" % (args.input_data, path_dir, out_name)
o2 = "\n\nPyRate was called as follows:\n%s\n" % (args)
if model_cov>=1: o2 += regression_trait
version_notes="""\n
Please cite: \n%s\n
OS: %s %s
Python version: %s\n
Numpy version: %s
Scipy version: %s
""" % (citation,platform.system(), platform.release(), sys.version, np.version.version, scipy.version.version)

o=''.join([o0,o1,o2,version_notes])
out_sum = "%s/%s_sum.txt" % (path_dir,suff_out)
sumfile = open(out_sum , "w",0) 
sumfile.writelines(o)
sumfile.close()

# OUTPUT 1 LOG MCMC
out_log = "%s/%s_mcmc.log" % (path_dir, suff_out) #(path_dir, output_file, out_run)
logfile = open(out_log , "w",0) 
if TDI<2:
	head="it\tposterior\tprior\tPP_lik\tBD_lik\tq_rate\talpha\tcov_sp\tcov_ex\tcov_q\tbeta\troot_age\t"
	for i in range(time_framesL): head += "lambda_%s\t" % (i)
	for i in range(time_framesM): head += "mu_%s\t" % (i)
	for i in range(1,time_framesL): head += "shift_sp_%s\t" % (i)
	for i in range(1,time_framesM): head += "shift_ex_%s\t" % (i)
else: head="it\tposterior\tprior\tPP_lik\tBD_lik\tq_rate\talpha\tcov_sp\tcov_ex\tcov_q\tk_birth\tk_death\troot_age\t"
head += "tot_length\t"
for i in taxa_names: head += "%s_TS\t" % (i)
for i in taxa_names: head += "%s_TE\t" % (i)
head=head.split('\t')
wlog=csv.writer(logfile, delimiter='\t')
wlog.writerow(head)

#logfile.writelines(head)
logfile.flush()
os.fsync(logfile)

# OUTPUT 2 MARGINAL RATES
if TDI!=1: # (path_dir, output_file, out_run)
	out_log_marginal = "%s/%s_marginal_rates.log" % (path_dir, suff_out) 
	marginal_file = open(out_log_marginal , "w") 
	head="it\t"
	for i in range(int(max(FA))+1): head += "l_%s\t" % i #int(fabs(int(max(FA))))
	for i in range(int(max(FA))+1): head += "m_%s\t" % i #int(fabs(int(max(FA))))
	for i in range(int(max(FA))+1): head += "r_%s\t" % i #int(fabs(int(max(FA))))
	head=head.split('\t')
	wmarg=csv.writer(marginal_file, delimiter='	')
	wmarg.writerow(head)
	marginal_file.flush()
	os.fsync(marginal_file)
	marginal_frames= array([int(fabs(i-int(max(FA)))) for i in range(int(max(FA))+1)])

# OUTPUT 3 MARGINAL LIKELIHOOD
else: 
	out_log_marginal_lik = "%s/%s_marginal_likelihood.txt" % (path_dir, suff_out) 
	marginal_file = open(out_log_marginal_lik , "w") 
	marginal_file.writelines(o)
	marginal_frames=0	

# OUTPUT 4 MARGINAL SHIFT TIMES	
#if TDI==2: 
#	out_log_marginal_time = "%s/%s_marginal_t_shifts.log" % (path_dir, suff_out) 
#	marginal_file_time = open(out_log_marginal_time , "wb") 
#	head="it\t"
#	for i in range(25): head += "t_l%s\t" % i #int(fabs(int(max(FA))))
#	for i in range(25): head += "t_m%s\t" % i #int(fabs(int(max(FA))))
#	head=head.split('\t')
#	wmarg_t=csv.writer(marginal_file_time, delimiter='	')
#	wmarg_t.writerow(head)
#	marginal_file.flush()


########################## START MCMC ####################################
if burnin<1 and burnin>0:
	burnin = int(burnin*mcmc_gen)

def start_MCMC(run):
	t1 = time.clock()
	print "started at:", time.ctime()
	# marginal_file is either for rates or for lik
	return MCMC([0,run, IT, sample_freq, print_freq, temperatures, burnin, marginal_frames, list()]) 

# Metropolis-coupled MCMC (Altekar, et al. 2004)	
if use_seq_lik is False and runs>1:
	marginal_frames= array([int(fabs(i-int(max(FA)))) for i in range(int(max(FA))+1)])
	pool = mcmcMPI(num_proc)
	res = pool.map(start_MCMC, range(runs))
	current_it=0
	swap_rate, attempts=0, 0
	while current_it<mcmc_gen:
		n1=np.random.random_integers(0,runs-1,2)		
		temp=1./(1+n1*temp_pr)
		[j,k]=n1
		#print "try: ", current_it, j,k #, res[j] #[2],res[0][2], res[1][2],res[2][2]
		r=(res[k][2]*temp[0]+res[j][2]*temp[1]) - (res[j][2]*temp[0]+res[k][2]*temp[1])
		if r>=log(random.random()) and j != k:
			args=list()
			best,ind=res[1][2],0
			#print current_it, "swap %s with chain %s [%s / %s] temp. [%s / %s]" % (j, k, res[j][2],res[k][2], temp[0],temp[1])
			swap_rate+=1
			res_temp1=res[j]
			res_temp2=res[k]
			res[j]=res_temp2
			res[k]=res_temp1
		current_it=res[0][0]
		res[0][0]=0
		args=list()
		for i in range(runs):
			seed=0
			args.append([current_it,i, current_it+IT, sample_freq, print_freq, temperatures, burnin, marginal_frames, res[i]])
		res = pool.map(MCMC, args)
		#except: print current_it,i, current_it+IT
		attempts+=1.
		#if attempts % 100 ==0: 
		#	print "swap freq.", swap_rate/attempts
		#	for i in range(runs): print "chain", i, "post:", res[i][2], sum(res[i][5]-res[i][6])
			
else: 
	if runs>1: print "\nWarning: MC3 algorithm requires multi-threading.\nUsing standard (BD)MCMC algorithm instead.\n"
	res=start_MCMC(0)
t1 = time.clock()
print "\nfinished at:", time.ctime(),"\n"
logfile.close()
marginal_file.close()

#cmd="cd %s && cd .. && tar -czf %s.tar.gz %s;" % (path_dir, folder_name, folder_name)
#print cmd
#os.system(cmd)
quit()