import numpy as np
from numpy import *

def TreePar_inter(time,t):
	res=1
	j=0
	i=1
	while (j==0):
		if (i == len(t)):
			j=1
		else:
			if (time<=t[i]): j=1 
			else: i = i+1
	if run_check==1: print "IIIII",i
	return i

def TreePar_q2(i,time,t,l,mu,rho):
	ind = i-1
	if (t[ind]>time): res = 0
	else:
		ci  =  TreePar_const(i,t,l,mu,rho)
		a   = (ci-1) * exp(-(l[ind]-mu[ind]) * t[ind])
		b   =  (mu[ind] - ci * l[ind]) * exp(-(l[ind]-mu[ind]) * time) 
		res =  (mu[ind] * a +b ) / (l[ind] * a + b )
	return res


def TreePar_g(time,t,l,mu,rho):
	i   = TreePar_inter(time,t)	
	ci  = TreePar_const(i,t,l,mu,rho)	
	i = i -1
	zah = exp(-(l[i]-mu[i]) * (t[i]+time))
	a   = (ci-1) * exp(-(l[i]-mu[i]) * t[i])
	b   = (mu[i] - ci * l[i]) * exp(-(l[i]-mu[i]) * time) 
	nen = l[i] * a +  b
	res = zah/(nen)**2
	return res

def TreePar_const(i,t,l,mu,rho):
	#print "IIIII",i
	if (i==1):
		ci = 1-rho[0]
	else:
		i= i-1
		ci = 1-rho[i] + rho[i] * TreePar_q2(i,t[i],t,l,mu,rho)
	return ci


def TreePar_lineages(x,t):
	lin = [len(x)+1]
	if (len(t)>1):
		for j in range(1,len(t)):
			lin.append( 1+len(x[x > t[j]]) )
	
	return lin



def TreePar_LikShifts(x,t,l,mu,sampling,posdiv=0,survival=1,groups=0):
	res = -10**12
	boundary = 0
	if 2<1: # various checks
		for i in range(len(l)):
			if (l[i]==mu[i] or mu[i]<0 or l[i]<0.0001 or l[i]>100 or (abs(l[i]-mu[i])<0.0001) ): boundary = 1
			if (posdiv==TRUE and (l[i]-mu[i])<0.0001 ): boundary = 1
	
		for i in range(len(sampling)):
			if (sampling[i]>1 or sampling[i]<=0 ): boundary = 1
	
	# if (boundary==0)
	# x = np.sort(x) # done only once
	n = TreePar_lineages(x,t)
	mrca = x[len(x)-1]
	res = n[0]*log(sampling[0]) + n[0]* log((l[0]-mu[0])**2) + 2 * log(TreePar_g(mrca,t,l,mu,sampling))
	if (survival == 1):
		res = res -2*log(1-TreePar_q2(TreePar_inter(mrca,t),mrca,t,l,mu,sampling ) )
	
	for j in range(0,(len(x)-1)):
		if run_check==1: print j, res
		ind  = TreePar_inter(x[j],t)-1
		res = res +log(2*l[ind]) + log(TreePar_g(x[j],t,l,mu,sampling))
	
	if (TreePar_inter(mrca,t)>0):
		for j in range(1,TreePar_inter(mrca,t)):
			res = res + n[j] * log(sampling[j] *(l[j]-mu[j])**2 * TreePar_g((t[j]),t,l,mu,sampling) )
	
	res = res-(len(x)-1)*log(2)	
	return res

run_check =0
if run_check==1:
	t = np.array([0.,2,4,8])

	l = np.array([0.2,0.4,0.1,0.5])
	mu = np.array([0.05,0.01,0.01,0.2])
	sampling = np.array([1.,1,1,1])

	x =np.array([0.10421359,  0.13004487,  0.37298092,  0.42911081,  0.57161158,
	        0.59774216,  0.75681057,  0.83263216,  0.94443039,  0.96978709,
	        0.9936791 ,  1.02544904,  1.38701433,  1.40730544,  1.66149102,
	        1.6851466 ,  1.71825988,  1.87455761,  1.9037728 ,  3.21374116,
	        3.3544458 ,  3.45178927,  3.60678163,  3.6780833 ,  4.04917188,
	        4.26055643,  4.92615207,  4.95757345,  5.20169177,  5.82124525,
	        6.18392766,  6.28772875,  6.38583129,  6.86042982,  7.26102691,
	        7.29864836,  7.36282462,  7.49450751,  7.61454435,  7.69944019,
	        8.04961634,  9.01729561,  9.49997352,  9.61148747, 10.03114318,
	       11.7092978 , 11.80412531, 12.78810896, 16.22804692, 24.22015229])


	print TreePar_LikShifts(x,t,l,mu,sampling)


