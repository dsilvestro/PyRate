from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
import scipy.stats
#np.random.seed(1234)

n_simulations = 100
if n_simulations==1:
	save_mcmc_samples = 1
else:
	save_mcmc_samples = 0
n_iterations = 50000
burnin = 10000
sampling_freq = 100


def binomial_pmf(x,n,p):
	# binomial distribution 
	" P(k | n, p)  = n!/(k! * (n-k)!) * p^k * (1-p)^(n-k)   "
	" scipy.stats.logpmf(x, n, p, loc=0) "
	return scipy.stats.binom.logpmf(x, n, p, loc=0)

def normal_pdf(x,n,p):
	return scipy.stats.norm.logpdf(x, n, p, loc=0)

def update_multiplier(q,d=1.1):
	u = np.random.uniform(0,1)
	l = 2*np.log(d)
	m = np.exp(l*(u-.5))
	new_q = q * m
	U=np.log(m)
	return new_q,U

def calcHPD(data, level=0.95) :
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
	return np.array([d[i], d[i+nIn-1]])


logfile = open("est_root_epochs_epsilon0.5.log", "w") 
text_str = "iteration\tlikelihood\tNobs\tmu0_true\troot_true\tq_med_true\tepsilon_true\troot_obs\troot\troot_m2\troot_M2\troot_m4\troot_M4\troot_m8\troot_M8"
logfile.writelines(text_str)
n_samples = 20

for replicate in range(n_simulations):
	# observed time bins
	true_root = np.random.uniform(50,200)
	mid_points = np.linspace(0,500,500/5)
	#mid_points = np.array([0]+list(mid_points))
	#mid_points= mid_points[0:-1] + np.diff(mid_points)/2.
	
	
	Nobs = np.random.randint(500,5000)
	x_0 = 1
	true_mu0 = (Nobs-x_0)/true_root

	# MAKE DIV TRAJECTORY
	Ntrue = Nobs - mid_points[mid_points<true_root]*true_mu0

	# add noise
	true_epsilon = np.random.uniform(0,0.5)
	Ntrue = np.rint(np.exp(np.random.normal(np.log(Ntrue),true_epsilon)))
	print( "Ntrue", Ntrue)


	true_q = np.exp( np.random.normal(-5,1,len(mid_points)))[mid_points<true_root]
	#np.random.gamma(1.01,.01,len(mid_points))[mid_points<true_root]
	# preserved occs
	x = np.rint(Ntrue*true_q)[::-1]
	print(x)
	# remove first x values if they are == 0
	j,c=0,0
	for i in range(len(x)):
		if x[i]==0 and j==0:
			c+=1
		else:
			break

	x = x[c:][::-1]
	age_oldest_obs_occ = mid_points[len(x)-1]
	print("true_q",true_q, np.max(true_q)/np.min(true_q))

	x_augmented = 0+x
	out_array = np.zeros( (n_samples, 3 ) )
	for root_index in range(n_samples):
		x_augmented = np.concatenate( (x_augmented, np.zeros(1))  )
		q_A = np.random.gamma(1.1,.01)
		q_augmented = np.repeat(q_A,len(x_augmented))
	
		root_A = mid_points[len(x_augmented)-1]
		mu0_A = (Nobs-x_0)/(root_A)
		Nest = Nobs - mid_points[mid_points<=(root_A)]*mu0_A
		likA = np.sum(binomial_pmf(x_augmented,Nest,q_augmented))
		j=0
		n_iterations=5000
	
		for iteration in range(n_iterations):
			q,hastings = update_multiplier(q_A)	
			Nest = Nobs - mid_points[mid_points<=(root_A)]*mu0_A
			q_augmented = np.repeat(q,len(x_augmented))	
			lik = np.sum(binomial_pmf(x_augmented,Nest,q_augmented))
			if (lik - likA)  > 0: 
				likA = lik
				q_A = q
		out_array[root_index] = np.array([ likA,q_A,root_A ])
	print(out_array)
	
	indx_max_lik = np.where(out_array[:,0]==np.max(out_array[:,0]))[0][0]
	max_lik = out_array[indx_max_lik,0]
	root_ml = out_array[indx_max_lik,2]
	min_max_range = np.array([i for i in range(n_samples) if np.max(out_array[:,0])-out_array[i,0]<2])
	root_min2 = np.min(out_array[min_max_range,2])
	root_max2 = np.max(out_array[min_max_range,2])
	min_max_range = np.array([i for i in range(n_samples) if np.max(out_array[:,0])-out_array[i,0]<4])
	root_min4 = np.min(out_array[min_max_range,2])
	root_max4 = np.max(out_array[min_max_range,2])		
	min_max_range = np.array([i for i in range(n_samples) if np.max(out_array[:,0])-out_array[i,0]<8])
	root_min8 = np.min(out_array[min_max_range,2])
	root_max8 = np.max(out_array[min_max_range,2])		
	
	
	
	
	text_str = "\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ( replicate, max_lik, Nobs, true_mu0, true_root, 
			np.median(true_q), true_epsilon, age_oldest_obs_occ, root_ml,root_min2,root_max2,root_min4,root_max4,root_min8,root_max8 )
						
	logfile.writelines(text_str)
	logfile.flush()
	