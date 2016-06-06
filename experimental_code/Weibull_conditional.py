from numpy import *
import numpy as np
import os
from scipy.special import gamma
np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit

for REP in range(10):
	# simulate
	R = 30.
	N = 50 #np.random.randint(25,50,1) # no. species
	E = 0.1   #np.random.uniform(0.05,0.25) # fraction of extant


	W_shape = 1.
	W_scale = 3.
	br_length = np.random.weibull(W_shape,N)*W_scale

	TS = np.random.uniform(br_length,R,N)
	ind_rand_extant = np.random.choice(np.arange(N), size=int(N*E), replace=False,p=br_length/sum(br_length))

	TS[ind_rand_extant]=np.random.uniform(0,br_length[ind_rand_extant],len(ind_rand_extant))
	TE = TS-br_length
	TE[TE<0]=0

	ts=TS
	te=TE

	# subsample
	q = 1.
	br = ts-te
	sampling_prob= 1 - exp(-q*br)

	#for i in range(len(br)):
	#	print br[i],sampling_prob[i]

	rr= np.random.uniform(0,1,N)
	index_sampled=[]
	for j in range(len(br)):
		if sampling_prob[j]>=rr[j]: index_sampled.append(j)




	def log_wei_pdf(x,W_shape,W_scale):
		log_wei_pdf = log(W_shape/W_scale) + (W_shape-1)*log(x/W_scale) - (x/W_scale)**W_shape
		return log_wei_pdf

	def update_multiplier_proposal(i,d=1.1):
		S=shape(i)
		u = np.random.uniform(0,1,S)
		l = 2*log(d)
		m = exp(l*(u-.5))
	 	ii = i * m
		return ii, sum(log(m))

	# pdf_Poi_sampling = 1 - exp(-q*v)


	ts=ts[index_sampled]	
	te=te[index_sampled]
	print len(te[te==0]), "extant", len(te[te>0]), "extinct",
	
	listW = []
	WA = np.ones(2) # shape, scale
	likA=-inf
	for i in range(1000):
		W,hast=update_multiplier_proposal(WA)
		lik=sum(log_wei_pdf(ts-te,W[0],W[1]))
		if lik-likA + hast >= log(np.random.random()):
			WA=W
			likA=lik
		if i > 100: listW.append(WA)
			
	print np.mean(np.array(listW),axis=0),


	#print len(ts)

	def cdf_Weibull(x,W_shape,W_scale):
		# Weibull cdf
		wei_cdf = 1 - exp(-(x/W_scale)**W_shape)
		return wei_cdf
	

	br =ts-te
	waiting_times = np.array([min(br)] + list(np.diff(sort(br))) )
	#print waiting_times
	#print list(log(1./(1-exp(-q*br))))
	#print log(exp(-q*np.cumsum(waiting_times)))
	
	def int_Poi_prob_Weibull(q,x,n_bins=1000):
		v= np.linspace(0.000001,x,n_bins)
		pdf_Poi_sampling = 1 - exp(-q*v)
		return sum(pdf_Poi_sampling) *(v[1]-v[0])
	
	def pdf_W_poi(W,q,x):
		return exp(log_wei_pdf(x,W[0],W[1]) + log(1-exp(-q*x)))
	
	
	
	listW = []
	WA = np.ones(2) # shape, scale
	likA=-inf
	for it in range(1000):
		W,hast=update_multiplier_proposal(WA)
		
		#x=  min(0.0512933/q, min(ts-te))
		#A1 =1-cdf_Weibull(x,W[0],W[1])
		lik1=sum(log_wei_pdf(br,W[0],W[1])) + sum(log(1-exp(-q*br)))
		v= np.linspace(0.000001,1000,10000)
		P = pdf_W_poi(W,q,v)
		#lik2= log(sum(P) *(v[1]-v[0]))
		
		
		lik2= log(sum(P) *(v[1]-v[0]))
		P_extant = [sum(pdf_W_poi(W,q,v[v<=i])) for i in ts[te==0]]
		P_extant = np.array(P_extant)*(v[1]-v[0]) 
		lik = lik1-(lik2)*len(te[te>0])-sum(log(P_extant))
		
		#lik_obs = 1 #sum(log_wei_pdf(br,W[0],W[1])) #+ sum(log(1-exp(-q*br)))     #+log(A1) 
		#
		#sorted_br = sort(br)
		#BR = np.array([0]+list(sorted_br))
		#
		#int1 = np.ones((len(BR),2))
		#for j in range(len(BR)-1):
		#	int1[j,0] = cdf_Weibull(BR[j+1],W[0],W[1]) - cdf_Weibull(BR[j],W[0],W[1]) # slice integral Weibull
		#	int1[j,1] = int_Poi_prob_Weibull(q,BR[j+1]) - int_Poi_prob_Weibull(q,BR[j]) # slice integral Weibull
		#	
		#log_int = log(int1+0.00001)
		#lik_unobs = sum(log_int)
		#lik= lik_unobs+lik_obs
		#lik += sum(log(1./(1-exp(-q*br))))
		
		lik = lik1-(lik2)*len(ts)
		
		if lik-likA + hast >= log(np.random.random()):
			WA=W
			likA=lik
		if it > 10: 
			listW.append(WA)
			#print likA, WA
			
	print np.mean(np.array(listW),axis=0)








