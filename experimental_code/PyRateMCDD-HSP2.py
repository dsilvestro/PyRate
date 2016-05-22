#!/usr/bin/env python 
import argparse, os,sys
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import scipy.special
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
from lib_utilities import calcHPD as calcHPD
from lib_utilities import print_R_vec as print_R_vec
from lib_utilities import get_mode as get_mode


#### DATA ###

p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('-d', type=str,   help='data set', default=0, metavar=0)
p.add_argument('-m', type=int,   help='model', default=0, metavar=0)
p.add_argument('-n', type=int,   help='MCMC iterations', default=5000000, metavar=5000000)
p.add_argument('-s', type=int,   help='sampling freq.', default=5000, metavar=5000)
p.add_argument('-p', type=int,   help='print freq.', default=5000000, metavar=5000000)
p.add_argument('-j', type=int,   help='replicate', default=0, metavar=0)
p.add_argument('-c', type=int, help='clade', default=0, metavar=0)
p.add_argument('-b', type=int, help='burnin (number of generations)', default=1, metavar=1)
p.add_argument('-T', type=float, help='Max time slice', default=np.inf, metavar=np.inf)
p.add_argument('-plot', type=str, help='Log file', default="", metavar="")
#p.add_argument('-bR',  type=float, help='Baseline speciation/extinction rates', default=[1., 1.], metavar=1., nargs=2)


args = p.parse_args()


dataset=args.d
n_iterations=args.n
sampling_freq=args.s
print_freq = args.p
#t_file=np.genfromtxt(dataset, names=True, delimiter='\t', dtype=float)
t_file=np.loadtxt(dataset, skiprows=1)

name_file = os.path.splitext(os.path.basename(dataset))[0]
wd = "%s" % os.path.dirname(dataset)



clade_ID=t_file[:,0]
clade_ID=clade_ID.astype(int)
ts=t_file[:,2+2*args.j]
te=t_file[:,3+2*args.j]

constr=args.m

single_focal_clade = True
fixed_focal_clade = args.c
clade_name = "_c%s" % (fixed_focal_clade)

burnin = args.b
beta_value = "_hsp"

all_events=sort(np.concatenate((ts,te),axis=0))[::-1] # events are speciation/extinction that change the diversity trajectory
n_clades,n_events=max(clade_ID)+1,len(all_events)
Dtraj=init_Dtraj(n_clades,n_events)

##### RTT PLOTS
summary_file = args.plot
if summary_file != "":
	plot_RTT = True
	# parse a log file to get baseline rates and G values
	print "parsing log file:", summary_file
	#fixed_focal_clade,baseline_L,baseline_M,Gl_focal_clade,Gm_focal_clade,est_kl,est_km = lib_utilities.parse_hsp_logfile(summary_file[0])
	try: fixed_focal_clade,baseline_L_list,baseline_M_list,Gl_focal_clade_list,Gm_focal_clade_list,est_kl,est_km = lib_utilities.parse_hsp_logfile_HPD(summary_file,burnin)
	except: sys.exit("Unable to parse file.")
		
else: plot_RTT = False




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
s_or_e_array[unsorted_events>args.T] = 4
unsorted_events[unsorted_events>args.T] = args.T

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




##### HORSESHOE PRIOR FUNCTIONS
def pdf_gamma(L,a,b): 
	return scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0)

def pdf_normal(L,sd): 
	return scipy.stats.norm.logpdf(L,loc=0,scale=sd)

def pdf_cauchy(x,s=1):
	return scipy.stats.cauchy.logpdf(x,scale=s,loc=0)
 

def sample_lam_mod(lam,beta,tau):
	eta=1./(lam**2)
	mu =beta/tau
	u =np.random.uniform(0, 1./(1+eta), len(eta))
	truncate = (1-u)/u
	# 2/(mu**2) = scale parameter
	new_eta = np.random.exponential( 2/(mu**2), len(mu)  )
	new_lam = np.zeros(len(lam))+lam
	new_lam[new_eta<truncate]= sqrt(1./new_eta[new_eta<truncate])
	return new_lam

def sample_tau_mod(lam,beta,tau):
	eta=1./(tau**2)
	u =np.random.uniform(0, 1./(1+eta))
	truncate = (1-u)/u
	theta = (beta/lam)
	a = (len(lam.flatten())+1)/2.
	b = sum((theta**2)/2)
	# 1./b = scale parameter = 2/sum(theta**2) || cf. 2/(mu**2) = scale parameter above
	new_eta = np.random.gamma( a, 1./b, len(tau)  )
	new_tau = np.zeros(len(tau))+tau
	new_tau[new_eta<truncate]= sqrt(1./new_eta[new_eta<truncate])
	return new_tau


# Scott 2010 arXiv:1010.5265v1
#eta = 1/(Tau^2)
#u = runif(1,0,1/(eta + 1))
#ub = (1-u)/u
#a = (p+1)/2
#b = sum(Theta^2)/2
#ub2 = pgamma(ub,a,rate=b)
#u2 = runif(1,0,ub2)
#eta = qgamma(u2,a,rate=b)
#Tau = 1/sqrt(eta)

#####
scaling =1

if scaling==0:	
	scale_factor = 1.
	MAX_G = np.inf #0.30/scale_factor # loc_shrinkage
	trasfRate_general = trasfMultiRateND
elif scaling == 1:
	scale_factor = 1./np.max(Dtraj)
	MAX_G = 0.30/scale_factor
	trasfRate_general = trasfMultiRate
elif scaling ==2:
	scale_factor = 1. #1./np.max(Dtraj, axis=0)
	MAX_G = 10.
	trasfRate_general = trasfMultiRateCladeScaling

print scale_factor





GarrayA=init_Garray(n_clades) # 3d array so:
                                 # Garray[i,:,:] is the 2d G for one clade
			         # Garray[0,0,:] is G_lambda, Garray[0,1,:] is G_mu for clade 0
if plot_RTT is True: pass
else:
	GarrayA[fixed_focal_clade,:,:] += np.random.normal(0,1,np.shape(GarrayA[fixed_focal_clade,:,:]))
	# setup log file
	out_file_name="%s_%s_m%s_MCDD%s%s.log" % (dataset,args.j,constr,clade_name,beta_value)
	logfile = open(out_file_name , "wb") 
	wlog=csv.writer(logfile, delimiter='\t')

	lik_head=""
	head="it\tposterior\tlikelihood\tprior"
	head+="\tl%s" % (fixed_focal_clade)
	head+="\tm%s" % (fixed_focal_clade)
	for j in range(n_clades): 
		head+="\tGl%s_%s" % (fixed_focal_clade,j)
	for j in range(n_clades): 
		head+="\tGm%s_%s" % (fixed_focal_clade,j)
	for j in range(n_clades): 
		head+="\tkl%s_%s" % (fixed_focal_clade,j)
	for j in range(n_clades): 
		head+="\tkm%s_%s" % (fixed_focal_clade,j)

	head+="\tLAM_mu"		
	head+="\tLAM_sd"		
	head+="\tTau"		
	head+="\thypR"
	wlog.writerow(head.split('\t'))
	logfile.flush()

LAM=init_Garray(n_clades)
LAM[fixed_focal_clade,:,:] = 1.
Constr_matrix=make_constraint_matrix(n_clades, constr)
l0A,m0A=init_BD(n_clades),init_BD(n_clades)

TauA=np.array([.5]) # np.ones(1) # P(G==0)
hypRA=np.ones(1)
Tau=TauA


########################## PLOT RTT ##############################
if plot_RTT is True: # NEW FUNCTION 2
	out="%s/%s_c%s_RTT.r" % (wd,name_file,fixed_focal_clade)
	newfile = open(out, "wb") 
	
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		r_script= "\n\npdf(file='%s\%s_c%s_RTT.pdf',width=0.6*20, height=0.6*10)\nlibrary(scales)\n" % (wd,name_file,fixed_focal_clade)
	else: 
		r_script= "\n\npdf(file='%s/%s_c%s_RTT.pdf',width=0.6*20, height=0.6*10)\nlibrary(scales)\n" % (wd,name_file,fixed_focal_clade)
	
	for i in range(n_clades):
		r_script+=lib_utilities.print_R_vec("\nclade_%s", Dtraj[:,i]) % (i+1)
	
	
	# get marginal rates
	print "Getting marginal rates..."
	
	variable_names = ["clade %s" % (i) for i in range(1,n_clades+1)]
	for i in range(-1, n_clades):
		marginal_L = list()
		marginal_M = list()
		Gl_temp,Gm_temp=0,0
		for j in range(len(baseline_L_list)): # loop over MCMC samples
			baseline_L = baseline_L_list[j]
			baseline_M = baseline_M_list[j]
			Gl_focal_clade = Gl_focal_clade_list[j,:]
			Gm_focal_clade = Gm_focal_clade_list[j,:]
			# G estimates are given per species but Dtraj are rescaled when:  scaling > 0 (default: scaling = 1)
			GarrayA=init_Garray(n_clades)
			GarrayA[fixed_focal_clade,0,:] += Gl_focal_clade/scale_factor 
			GarrayA[fixed_focal_clade,1,:] += Gm_focal_clade/scale_factor 
	
			if i==-1:
				G_temp = GarrayA+0
				#if j==0: print GarrayA[fixed_focal_clade,0,:] 
			else:
				G_temp = init_Garray(n_clades)
				G_temp[fixed_focal_clade,:,i] += GarrayA[fixed_focal_clade,:,i]
				Gl_temp+=G_temp[fixed_focal_clade,0,i]
				Gm_temp+=G_temp[fixed_focal_clade,1,i]
				#if j==0: print G_temp[fixed_focal_clade,0,:] 
					
	
			marginal_L.append(trasfRate_general(baseline_L,-G_temp[fixed_focal_clade,0,:],Dtraj))
			marginal_M.append(trasfRate_general(baseline_M, G_temp[fixed_focal_clade,1,:],Dtraj))


		if i== -1: print "Calculating mean rates and HPDs..."			
		else: print "Processing variable:", variable_names[i]
		
		marginal_L = np.array(marginal_L)
		marginal_M = np.array(marginal_M)
		#print np.shape(marginal_L)

		l_vec= np.zeros(np.shape(marginal_L)[1])
		m_vec= np.zeros(np.shape(marginal_L)[1])
		hpd_array_L= np.zeros((2,np.shape(marginal_L)[1]))
		hpd_array_M= np.zeros((2,np.shape(marginal_L)[1]))
		hpd_array_L50= np.zeros((2,np.shape(marginal_L)[1]))
		hpd_array_M50= np.zeros((2,np.shape(marginal_L)[1]))
		
		if i>=0:
			l_vec = np.mean(marginal_L, axis=0) # get_mode
			m_vec = np.mean(marginal_M, axis=0) # get_mode
		else:		
			for ii in range(np.shape(marginal_L)[1]): # loop over marginal rates
				l_vec[ii] = np.mean(marginal_L[:,ii]) # get_mode
				m_vec[ii] = np.mean(marginal_M[:,ii]) # get_mode
				hpd_array_L[:,ii] = calcHPD(marginal_L[:,ii])
				hpd_array_M[:,ii] = calcHPD(marginal_M[:,ii])
				hpd_array_L50[:,ii] = calcHPD(marginal_L[:,ii],0.75)
				hpd_array_M50[:,ii] = calcHPD(marginal_M[:,ii],0.75)

		r_script += lib_utilities.print_R_vec("\n\nt",all_events)
		r_script += "\ntime = -t"
		r_script += lib_utilities.print_R_vec("\nspeciation",l_vec)
		if i==-1:
			r_script += lib_utilities.print_R_vec("\nsp_hdp_m",hpd_array_L[0])
			r_script += lib_utilities.print_R_vec("\nsp_hdp_M",hpd_array_L[1])
			r_script += lib_utilities.print_R_vec("\nsp_hdp_m50",hpd_array_L50[0])
			r_script += lib_utilities.print_R_vec("\nsp_hdp_M50",hpd_array_L50[1])
		r_script += lib_utilities.print_R_vec("\nextinction",m_vec)
		if i==-1:
			r_script += lib_utilities.print_R_vec("\nex_hdp_m",hpd_array_M[0])
			r_script += lib_utilities.print_R_vec("\nex_hdp_M",hpd_array_M[1])
			r_script += lib_utilities.print_R_vec("\nex_hdp_m50",hpd_array_M50[0])
			r_script += lib_utilities.print_R_vec("\nex_hdp_M50",hpd_array_M50[1])
		

		if i==-1:
			r_script += """
clade_focal = clade_%s
par(mfrow=c(1,2))
YLIM = c(0,max(c(sp_hdp_M[clade_focal>0],ex_hdp_M[clade_focal>0])))
XLIM = c(min(time[clade_focal>0]),0)
YLIMsmall = c(0,max(c(sp_hdp_M50[clade_focal>0],ex_hdp_M50[clade_focal>0])))
plot(speciation[clade_focal>0] ~ time[clade_focal>0],type="l",col="#4c4cec", lwd=3,main="Speciation rates - Joint effects", ylim = YLIM,xlab="Time (Ma)",ylab="Speciation rates",xlim=XLIM)
polygon(c(time[clade_focal>0], rev(time[clade_focal>0])), c(sp_hdp_M[clade_focal>0], rev(sp_hdp_m[clade_focal>0])), col = alpha("#4c4cec",0.1), border = NA)	
polygon(c(time[clade_focal>0], rev(time[clade_focal>0])), c(sp_hdp_M50[clade_focal>0], rev(sp_hdp_m50[clade_focal>0])), col = alpha("#4c4cec",0.3), border = NA)	
abline(v=-c(65,200,251,367,445),lty=2,col="gray")
plot(extinction[clade_focal>0] ~ time[clade_focal>0],type="l",col="#e34a33",  lwd=3,main="Extinction rates - Joint effects", ylim = YLIM,xlab="Time (Ma)",ylab="Extinction rates",xlim=XLIM)
polygon(c(time[clade_focal>0], rev(time[clade_focal>0])), c(ex_hdp_M[clade_focal>0], rev(ex_hdp_m[clade_focal>0])), col = alpha("#e34a33",0.1), border = NA)	
polygon(c(time[clade_focal>0], rev(time[clade_focal>0])), c(ex_hdp_M50[clade_focal>0], rev(ex_hdp_m50[clade_focal>0])), col = alpha("#e34a33",0.3), border = NA)	
abline(v=-c(65,200,251,367,445),lty=2,col="gray")
""" % (fixed_focal_clade+1)
		else:
			r_script += """
par(mfrow=c(1,2))
plot(speciation[clade_focal>0] ~ time[clade_focal>0],type="l",col="darkblue", lwd=3,main="Effect of: %s", ylim = YLIMsmall,xlab="Time (Ma)",ylab="Speciation and extinction rates",xlim=XLIM)
mtext("Wl = %s, Wm = %s, Gl = %s, Gm = %s")
lines(extinction[clade_focal>0] ~ time[clade_focal>0], col="darkred", lwd=3)
abline(v=-c(65,200,251,367,445),lty=2,col="gray")
plot(clade_%s[clade_focal>0] ~ time[clade_focal>0],type="l", main = "Trajectory of variable: %s",xlab="Time (Ma)",ylab="Rescaled value",xlim=XLIM)
abline(v=-c(65,200,251,367,445),lty=2,col="gray")
""" % (variable_names[i],round(est_kl[i],2),round(est_km[i],2),round(Gl_temp/float(len(baseline_L_list)),2),round(Gm_temp/float(len(baseline_L_list)),2),i+1,variable_names[i])
			       
			

	r_script+="n<-dev.off()"
	newfile.writelines(r_script)
	newfile.close()
	print "\nAn R script with the source for the RTT plot was saved as: %s_c%s_RTT.r\n(in %s)" % (name_file,fixed_focal_clade,wd)
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		cmd="cd %s; Rscript %s\%s_c%s_RTT.r" % (wd,wd,name_file,fixed_focal_clade)
	else: 
		cmd="cd %s; Rscript %s/%s_c%s_RTT.r" % (wd,wd,name_file,fixed_focal_clade)
	os.system(cmd)
	print "done\n"	
	sys.exit("\n")

##############################################################



t1=time.time()
for iteration in range(n_iterations):	
	hasting=0
	gibbs_sampling=0
	if iteration==0:
		actualGarray=GarrayA*scale_factor
		likA,priorA,postA=np.zeros(n_clades),0,0
		
	l0,m0=l0A,m0A
	Garray=GarrayA
	Tau=TauA
	lik,priorBD=np.zeros(n_clades),0
	
	lik_test=np.zeros(n_clades)	
	
	
	if iteration==0:
		uniq_eve=np.unique(all_events,return_index=True)[1]  # indexes of unique values
		Garray_temp=Garray
		prior_r=0
		#for i in range(n_clades):
		i = fixed_focal_clade
		l_at_events=trasfRate_general(l0[i],-Garray_temp[i,0,:],Dtraj)
		m_at_events=trasfRate_general(m0[i],Garray_temp[i,1,:],Dtraj)
		l_s1a=l_at_events[idx_s[i]]
		m_e1a=m_at_events[idx_e[i]]
		lik[i] = (sum(log(l_s1a))-sum(abs(np.diff(all_events))*l_at_events[0:len(l_at_events)-1]*(Dtraj[:,i][1:len(l_at_events)])) \
		         +sum(log(m_e1a))-sum(abs(np.diff(all_events))*m_at_events[0:len(m_at_events)-1]*(Dtraj[:,i][1:len(l_at_events)])) )
		likA=lik

	else:	
		##### START FOCAL CLADE ONLY
		sampling_freqs=[.10,.60]		
		if iteration<1000: rr = np.random.uniform(0,sampling_freqs[1])
		else: rr = np.random.random()

		#if single_focal_clade is True and rr > sampling_freqs[1]: 
		focal_clade=fixed_focal_clade
		#else: focal_clade= np.random.random_integers(0,(n_clades-1),1)[0]
		
		if rr<sampling_freqs[0]:
			rr2 = np.random.random()
			if rr2<.25: 
				l0=np.zeros(n_clades)+l0A
				l0[focal_clade],hasting=update_multiplier_proposal(l0A[focal_clade],1.2)
			elif rr2<.5: 	
				m0=np.zeros(n_clades)+m0A
				m0[focal_clade],hasting=update_multiplier_proposal(m0A[focal_clade],1.2)
			#if iteration> 2000:
			#	Tau_t,hasting = update_multiplier_proposal(TauA,1.2)
			#	Tau = np.zeros(1)+Tau_t

		elif rr<sampling_freqs[1]: # update hypZ and hypR
			gibbs_sampling=1
			if  np.random.random() < 0.15:
				Tau = sample_tau_mod(LAM[focal_clade,:,:],GarrayA[focal_clade,:,:],TauA)
			else:
				# Gibbs sampler (slice-sampling, Scott 2011)
				LAM[focal_clade,0,:] = sample_lam_mod(LAM[focal_clade,0,:],GarrayA[focal_clade,0,:],Tau)
				LAM[focal_clade,1,:] = sample_lam_mod(LAM[focal_clade,1,:],GarrayA[focal_clade,1,:],Tau)
			# Gibbs sampler (Exponential + Gamma[2,2])
			G_hp_alpha,G_hp_beta=1.,.01
			g_shape=G_hp_alpha+len(l0A)+len(m0A)
			rate=G_hp_beta+sum(l0A)+sum(m0A)
			hypRA = np.random.gamma(shape= g_shape, scale= 1./rate, size=1)
		else: # update Garray (effect size) 
			Garray_temp= update_parameter_normal_2d_freq((GarrayA[focal_clade,:,:]),.35,m=-MAX_G,M=MAX_G)
			#Garray_temp,hasting= multiplier_normal_proposal_pos_neg_vec((GarrayA[focal_clade,:,:]),d1=.3,d2=1.2,f=.65)
			
			Garray=np.zeros(n_clades*n_clades*2).reshape(n_clades,2,n_clades)+GarrayA
			Garray[focal_clade,:,:]=Garray_temp
			#print GarrayA[focal_clade,:,:]-Garray[focal_clade,:,:]

		
		Garray_temp=Garray
		i=focal_clade 
		l_at_events=trasfRate_general(l0[i],-Garray_temp[i,0,:],Dtraj)
		m_at_events=trasfRate_general(m0[i], Garray_temp[i,1,:],Dtraj)
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
	log(TauA) * (1-sum_R_per_clade) + log(1-TauA)*(sum_R_per_clade))
	
	"""

	prior = sum(pdf_normal(Garray[fixed_focal_clade,:,:],sd=LAM[fixed_focal_clade,:,:]*Tau ))
	prior +=sum(pdf_cauchy(LAM[fixed_focal_clade,:,:]))
	prior +=sum(pdf_cauchy(Tau))	
	prior += prior_exponential(l0,hypRA)+prior_exponential(m0,hypRA)
	
	if (sum(lik) + prior) - postA + hasting >= log(rand.random()) or iteration==0 or gibbs_sampling==1:
		postA=sum(lik)+prior
		likA=lik
		priorA=prior
		l0A=l0
                m0A=m0
		GarrayA=Garray
		actualGarray=GarrayA[fixed_focal_clade,:,:]*scale_factor
		TauA=Tau
		#hypRA=hypR
	
	if iteration % print_freq ==0: 
		k= 1./(1+TauA**2 * LAM[fixed_focal_clade,:,:]**2) # Carvalho 2010 Biometrika, p. 471
		loc_shrinkage = (1-k) # so if loc_shrinkage > 0 is signal, otherwise it's noise (cf. Carvalho 2010 Biometrika, p. 474)
		print iteration, array([postA]), TauA, mean(LAM[fixed_focal_clade,:,:]), len(loc_shrinkage[loc_shrinkage>0.5]) #, sum(likA),sum(lik),prior, hasting
		#print likA
		#print "l:",l0A
		#print "m:", m0A
		#print "G:", actualGarray.flatten()
		#print "R:", RA.flatten()
		#print "Gr:", GarrayA.flatten()
		#print "Hmu:", TauA, 1./hypRA[0] #,1./hypRA[1],hypRA[2]
	if iteration % sampling_freq ==0:
		k= 1./(1+TauA**2 * LAM[fixed_focal_clade,:,:]**2) # Carvalho 2010 Biometrika, p. 471
		loc_shrinkage = (1-k) # so if loc_shrinkage > 0 is signal, otherwise it's noise (cf. Carvalho 2010 Biometrika, p. 474)
		#loc_shrinkage =LAM[fixed_focal_clade,:,:]**2
		log_state=[iteration,postA,sum(likA)]+[priorA]+[l0A[fixed_focal_clade]]+[m0A[fixed_focal_clade]]+list(actualGarray.flatten())+list(loc_shrinkage.flatten())+[mean(LAM[fixed_focal_clade,:,:]),std(LAM[fixed_focal_clade,:,:])] +list(TauA) +[hypRA[0]]
		wlog.writerow(log_state)
		logfile.flush()

print time.time()-t1
quit()










