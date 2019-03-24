#!/usr/bin/env python 
from numpy import *
import numpy as np
import scipy.stats
np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit
from scipy.special import beta as f_beta
from scipy.optimize import fmin_powell as Fopt # fmin
from scipy.special import betainc
import csv,os, argparse,sys


def NHPP_lik(x,q,s,e):
	k=len(x[x>0]) # no. fossils for species i
	xB1= -(x-s) # distance fossil-ts
	c=.5
	C=s-c*(s-e)
	a = 1+ (4*(C-e))/(s-e)
	b = 1+ (4*(-C+s))/(s-e)
	lik = -q*(s-e) + sum(logPERT4_density(s,e,a,b,x)+log(q)) - log(1-exp(-q*(s-e)))
	#lik = -q*(s-e) + sum(logPERT4_density(s,e,a,b,x)+log(q))+log(s-e) - log(1-exp(-q*(s-e)))
	lik += -sum(log(np.arange(1,k+1)))
	return lik

def logPERT4_density(s,e,a,b,x): # relative 'stretched' LOG-PERT density: PERT4 * (s-e)
	#return log((s-x)**(b-1) * (-e+x)**(a-1)) - log ((s-e)**4 * f_beta(a,b))
	return log(s-x)*(b-1) + log(-e+x)*(a-1) - (log(s-e)*4 + log(f_beta(a,b)))

### ML OPT SE
def optim_se_given_q_HPP(x,q,exp_se=0):
	# USE EXPECTED s,e: 
	if exp_se==1:
		s,e =max(x)+1./q,min(x)-1./q
	else:
		s,e =max(x),min(x)
	if e<0: e= 0.00000001
	k=len(x[x>0]) # no. fossils for species i
	lik= -q*(s-e) + log(q)*k - sum(log(np.arange(1,k+1))) - log(1-exp(-q*(s-e)))
	return [lik,s,e]

def optim_se_given_q_NHPP(x,q):		
	def NHPP_lik_ML(se):
		s=abs(se[0])+max(x)
		e=max(min(x)-abs(se[1]), 0.0001) # avoid extinction in the future
		k=len(x[x>0]) # no. fossils for species i
		c=.5
		C=s-c*(s-e)
		a = 1+ (4*(C-e))/(s-e)
		b = 1+ (4*(-C+s))/(s-e)
		lik1 = -q*(s-e)  
		lik2 = sum(logPERT4_density(s,e,a,b,x)+log(q))
		lik3 = - log(1-exp(-q*(s-e)))
		lik  = lik1+lik2+lik3 -sum(log(np.arange(1,k+1)))
		return -lik	
	se0 = [0.1*max(x), min(x)-0.1]
	optValues =Fopt(NHPP_lik_ML, se0, full_output=1, disp=0)
	params=abs(array(optValues[0]))
	lik= -(optValues[1])
	s = params[0]+max(x)
	e = min(x)-params[1]
	return [lik,s,e]

def exp_tpp(q,q_shift): # expected value under a multi-Exp distributio
	q_shift=sort(list(q_shift)+[0,inf])
	pdf_sample = []
	tot_samples = 10000
	for i in range(len(q)):
		r = np.random.exponential(1./q[i],tot_samples)
		temp= r[r<(q_shift[i+1]-q_shift[i])]
		#print q_shift[i+1]-q_shift[i]
		pdf_sample += list(temp+q_shift[i])
		tot_samples -= len(temp)
	return mean(pdf_sample)

def optim_se_given_q_TPP(x,q,occs_sp_bin_i,times_q_shift=[inf,0],exp_se=0):
	fa,la=max(x),min(x)
	# USE EXPECTED s,e
	if exp_se==1:
		# speciation time
		qS_ts = times_q_shift[times_q_shift>=fa]
		if len(qS_ts)==1: # no shift prior to max(x)
			s = fa + 1./q[0]
		else:
			q_rates_prior_fa = q[0:(len(qS_ts))]
			#print "q_rates_prior_fa",q_rates_prior_fa[::-1], (qS_ts-fa)[::-1]
			s = fa + exp_tpp(q_rates_prior_fa[::-1], (qS_ts-fa)[::-1])
		# extinction time
		qS_te = times_q_shift[times_q_shift<=la]
		if len(qS_te)==1: # no shift prior to max(x)
			e = la - 1./q[len(q)-1]
		else:
			q_rates_post_la = q[len(q)-len(qS_te):len(q)]
			#print "q_rates_post_la",q_rates_post_la,la-qS_te
			e = la - exp_tpp(q_rates_post_la,la-qS_te)
	else:
		s,e =fa,la
	if e<0: e= 0.00000001
	k=len(x[x>0]) # no. fossils for species i	
	k_vec = occs_sp_bin_i # no. occurrences per time bin per species
	# e.g. k_vec = [0,0,1,12.,3,0]
	h = np.histogram(np.array([s,e]),bins=sort(times_q_shift))[0][::-1]
	ind_tste= (h).nonzero()[0]
	#print "\n\n\nHIST",ind_tste, np.array([s,e]),sort(times_q_shift)
	#print k_vec
	#quit()
	ind_min=min(ind_tste)
	ind_max=max(ind_tste)
	ind=np.arange(len(times_q_shift))
	ind = ind[ind_min:(ind_max+1)] # indexes of time frames where lineage is present
	# calc time lived in each time frame
	t = times_q_shift[times_q_shift<s]
	t = t[t>e]
	t2 = np.array([s]+list(t)+[e])
	d = abs(np.diff(t2))
	
	q_rates=q
	
	lik = sum(-q_rates[ind]*d + log(q_rates[ind])*k_vec[ind]) - log(1-exp(sum(-q_rates[ind]*d))) -sum(log(np.arange(1,sum(k_vec)+1)))  
	#print [lik,s,e,max(x),min(x)]
	#print lik,s,e, x, i
	#quit()
	return [lik,s,e]

def range01(x, m=0, r=1):
	temp = (x-min(x))/(max(x)-min(x))
	temp = temp*r
	temp = temp+m
	return temp

def est_s_e_q(fossil_complete,occs_sp_bin,model=0,exp_se=0,q_shift_times=[],q0_init=[]):
 	def calc_tot_lik_given_q(q_arg):
		q=abs(q_arg[0])
		ml_est = []
		i=0
		for x in fossil_complete: # loop over species
			if model==0:
				ml_est.append(optim_se_given_q_HPP(x,q,exp_se))
			elif model==1:
				ml_est.append(optim_se_given_q_NHPP(x,q))
			elif model==2:
				ml_est.append(optim_se_given_q_TPP(x,abs(np.array(q_arg)),occs_sp_bin[i],q_shift_times,exp_se))
			i+=1
		ml_est = np.array(ml_est)
		tot_lik = sum(ml_est[:,0])
		return -tot_lik
	
	q0 = [1.1]
	if model==2: 
		if len(q0_init)> 0: q0 =q0_init
		else:	q0= np.ones(len(q_shift_times)-1)
	#print q0
	optValues =Fopt(calc_tot_lik_given_q, q0, full_output=1, disp=0)
	params=abs(array(optValues[0]))
	lik= -(optValues[1])
	return [lik, params]

def calcAICc(lik,df,s):
	return (2*df-2*lik + abs(2*df*(df+1)/(s-df-1.)))

def run_model_testing(Xdata,q_shift=0,min_n_fossils=2,verbose=1):
	# data are shifted by 100
	fossil_complete=[Xdata[i]+0 for i in range(len(Xdata)) if min(Xdata[i])>0 and len(Xdata[i])>=min_n_fossils] # remove extant, and few occs
	fossil_complete=[fossil_complete[i] for i in range(len(fossil_complete)) if max(fossil_complete[i])-min(fossil_complete[i])>0.1] # remove too short branches
	
	if len(fossil_complete) > 1:
		print "Using",len(fossil_complete),"species for model testing"
	else:
		sys.exit("The number of lineages meeting the requirements for model testing is insufficient.")
	
	max_time_range_fossils = max([max(i) for i in fossil_complete])
	min_time_range_fossils = min([min(i) for i in fossil_complete])
	
	if q_shift==0: 
		q_shift = [min_time_range_fossils+ (max_time_range_fossils-min_time_range_fossils)/2.]
	else: 
		q_shift = np.array(q_shift)+0
	
	times_q_shift = np.sort(np.array(list(q_shift)+ [ max(max(q_shift),max_time_range_fossils)*10 ] +[0]))[::-1]
	#print times_q_shift, [min_time_range_fossils+ (max_time_range_fossils-min_time_range_fossils)/2.]
	#print np.ones(len(times_q_shift)-1)

	occs_sp_bin =list()
	for i in range(len(fossil_complete)):
		occs_temp = fossil_complete[i]
		h = np.histogram(occs_temp[occs_temp>0],bins=sort( times_q_shift ))[0][::-1]
		occs_sp_bin.append(h)		
	# optimize rate
	# resHPP = est_s_e_q(fossil_complete,occs_sp_bin,model=0,exp_se=0)
	# if verbose ==1: print "HPP max likelihood:", resHPP[0], "q rate:", abs(resHPP[1])
	resHPPm = est_s_e_q(fossil_complete,occs_sp_bin,model=0,exp_se=1)
	if verbose ==1: print "HPP* max likelihood:", resHPPm[0], "q rate:", abs(resHPPm[1])
	# resTPP = est_s_e_q(fossil_complete,occs_sp_bin,model=2,q_shift_times=times_q_shift,exp_se=0)
	# if verbose ==1: print "TPP max likelihood:", resTPP[0],"q rates:", abs(np.array(resTPP[1]))
	resTPPm = est_s_e_q(fossil_complete,occs_sp_bin,model=2,q_shift_times=times_q_shift,exp_se=1)
	if verbose ==1: print "TPP* max likelihood:", resTPPm[0],"q rates:", abs(np.array(resTPPm[1]))
	resNHPP = est_s_e_q(fossil_complete,occs_sp_bin,model=1)
	#if verbose ==1: print "NHPP max likelihood:", resNHPP[0],"q rate:", abs(resNHPP[1])
	
	# ADD SAMPLING OF TS (NHPP)
	def update_multiplier_proposal(i,d=1.2):
		u = np.random.uniform(0,1)
		l = 2*log(d)
		m = exp(l*(u-.5))
	 	ii = i * m
		return ii, sum(log(m))

	def get_TSTEvalues(x,q=0.5,n=1000):
		min_te = min(x)
		addteA = 0.05*min_te
		max_ts = max(x)
		addtsA = 0.05*max_ts
		teAvector = []
		tsAvector = []
		postA = -np.inf
		hast=0
		for it in range(n):
			addte=0.
			addte, hast1 = update_multiplier_proposal(addteA,d=1.2)
			addts=0.
			addts, hast2 = update_multiplier_proposal(addtsA,d=1.2)
			hast = hast1+hast2
			lik = NHPP_lik(x,q,(max_ts+addts),(min_te-addte))		
			r = log(np.random.random())
			if lik - postA +hast >= r and (min_te-addte)>0: # only accept if te > 0
				#print lik,hast, postA, r
				postA=lik + 0
				addteA=addte + 0
				addtsA=addts + 0
			if it % 10==0 and it>100:
				#print it, postA,lik, s, min_te-addteA,q,addte
				teAvector.append(min_te-addteA)
				tsAvector.append(max_ts+addtsA)
		return  [mean(tsAvector),mean(teAvector)]

	liknhpp_Exp =0
	for i in range(len(fossil_complete)):
		x = fossil_complete[i]
		#est_s = optim_se_given_q_NHPP(x,resNHPP[1])[1]
		[ts,te] = get_TSTEvalues(x,q=resNHPP[1],n=10000)
		liknhpp_Exp += NHPP_lik(x,resNHPP[1],ts,te)
	if verbose ==1: print "NHPP* max likelihood:", liknhpp_Exp,"q rate:",abs(resNHPP[1]),"\n"
	
	# get AICc scores
	Liks = np.array([resHPPm[0],liknhpp_Exp,resTPPm[0]])
	DFs  = np.array([1, 1, len(resTPPm[1])])
	d_size = np.array([len(fossil_complete)]*3)
	AICs = calcAICc(Liks,DFs,d_size)
	models = np.array(["HPP","NHPP","TPP"])
	best_model = models[AICs==min(AICs)]
	other_models = models[models != best_model]
	print "models:", models
	print "AICc scores:", AICs
	deltaAICs = AICs-min(AICs)
	deltaAICs_ = deltaAICs[deltaAICs>0]
	# empirical thresholds
	dAIC_hpp  = [ [0,6.4,0],[0,17.4,0] ]
	dAIC_nhpp = [ [3.8,0,0],[8.,0,2.4]   ]
	dAIC_tpp  = [ [3.2,6.8,0],[10.6,23.3,0]]
	
	sig = ["",""]
	if best_model=="HPP":
		if deltaAICs[1] > dAIC_hpp[1][1]: sig = ["***", "***"] # significance at 1% vs NHPP and TPP
		elif deltaAICs[1] > dAIC_hpp[0][1]: sig = ["*", "***"]
		else: sig = ["","***"]
	if best_model=="NHPP":
		if deltaAICs[0] > dAIC_nhpp[1][0]: sig = ["***", "***"] # significance at 1% vs HPP and TPP
		elif deltaAICs[0] > dAIC_nhpp[0][0]: sig = ["*", "***"]
		else: sig = ["","***"]
	if best_model=="TPP":
		if deltaAICs[0] > dAIC_tpp[1][0]: sig = ["***"] # significance at 1% vs HPP 
		elif deltaAICs[0] > dAIC_tpp[0][0]: sig = ["*"]
		
		if deltaAICs[1] > dAIC_tpp[1][1]: sig += ["***"] # significance at 1% vs NHPP
		elif deltaAICs[1] > dAIC_tpp[0][1]: sig += ["*"]
	
	print """
	--------------------------------------
	Best model: %s
	
	dAIC - %s: %s %s
	dAIC - %s: %s %s
	
	*** indicates significance at P < 0.01
	*   indicates significance at P < 0.05
	--------------------------------------
	""" % (best_model[0], other_models[0], round(deltaAICs_[0],3), sig[0], \
	                      other_models[1], round(deltaAICs_[1],3), sig[1]  )
	
	



def run_model_testing_n_shifts(Xdata,q_shift=0,min_n_fossils=2,verbose=1):
	# data are shifted by 100
	fossil_complete=[Xdata[i]+0 for i in range(len(Xdata)) if min(Xdata[i])>0 and len(Xdata[i])>=min_n_fossils] # remove extant, and few occs
	fossil_complete=[fossil_complete[i] for i in range(len(fossil_complete)) if max(fossil_complete[i])-min(fossil_complete[i])>0.1] # remove too short branches
	
	if len(fossil_complete) > 1:
		print "Using",len(fossil_complete),"species for model testing"
	else:
		sys.exit("The number of lineages meeting the requirements for model testing is insufficient.")
	
	max_time_range_fossils = max([max(i) for i in fossil_complete])
	min_time_range_fossils = min([min(i) for i in fossil_complete])
	
	if q_shift==0: 
		q_shift = [min_time_range_fossils+ (max_time_range_fossils-min_time_range_fossils)/2.]
	else: 
		q_shift = np.array(q_shift)+0
	
	times_q_shift = np.sort(np.array(list(q_shift)+ [ max(max(q_shift),max_time_range_fossils)*10 ] +[0]))[::-1]
	d_size = len(fossil_complete)
	
	# full model
	occs_sp_bin =list()
	for i in range(len(fossil_complete)):
		occs_temp = fossil_complete[i]
		h = np.histogram(occs_temp[occs_temp>0],bins=sort( times_q_shift ))[0][::-1]
		occs_sp_bin.append(h)		
	resTPPm = est_s_e_q(fossil_complete,occs_sp_bin,model=2,q_shift_times=times_q_shift,exp_se=1,q0_init=[0.5]*(len(times_q_shift)-1))
	aic_temp = calcAICc(resTPPm[0],len(resTPPm[1]),d_size)
	print "\nLik:", resTPPm[0], "AICs:", aic_temp
	ml_est_rates = abs(np.array(resTPPm[1]))
	print "Q times:",times_q_shift, "Rates:", ml_est_rates
	
	aic_best = aic_temp
	def remove_one_shift(times_q_shift,ml_est_rates):
		list_shifts=[]
		list_rates =[]
		list_AICs = []
		for irm in range(1,len(times_q_shift)-1):
			times_temp = times_q_shift[times_q_shift != times_q_shift[irm]]
			occs_sp_bin =list()
			for i in range(len(fossil_complete)):
				occs_temp = fossil_complete[i]
				h = np.histogram(occs_temp[occs_temp>0],bins=sort( times_temp ))[0][::-1]
				occs_sp_bin.append(h)		
			# optimize rate
			q0 = ml_est_rates[ml_est_rates != ml_est_rates[irm]]
			print q0
			resTPPm = est_s_e_q(fossil_complete,occs_sp_bin,model=2,q_shift_times=times_temp,exp_se=1,q0_init = q0)
			aic_temp = calcAICc(resTPPm[0],len(resTPPm[1]),d_size)
			print "\nLik:", resTPPm[0],"AICs:", aic_temp
			print "Q times:",times_temp, "Rates:", abs(np.array(resTPPm[1]))
			list_AICs.append( aic_temp )
			list_shifts.append( times_temp )
			list_rates.append( abs(np.array(resTPPm[1])) )
		
		return list_AICs[np.argmin(list_AICs)], list_shifts[np.argmin(list_AICs)], list_rates[np.argmin(list_AICs)]
	
	while True:
		minAIC, bestTimes, bestRates = remove_one_shift(times_q_shift,ml_est_rates)
		print minAIC, aic_best
		if minAIC - aic_best > 4:
			print "break", minAIC, aic_best
			#if minAIC < aic_best: 
			aic_best = minAIC
			times_q_shift = bestTimes
			ml_est_rates = bestRates
