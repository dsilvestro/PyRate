from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
import scipy.stats
np.random.seed(1234)

n_simulations = 1000
if n_simulations==1:
	save_mcmc_samples = 1
else:
	save_mcmc_samples = 0

n_iterations = 50000
burnin = 10000
sampling_freq = 100


def approx_log_fact(n):
	# http://mathworld.wolfram.com/StirlingsApproximation.html
	return np.log(np.sqrt((2*n+1./3)*np.pi)) + n*np.log(n) -n

def get_log_factorial(n):
	if n < 100: return np.log(scipy.special.factorial(n))
	else: return approx_log_fact(n)

def get_log_binomial_coefficient(n, k):
	# np.log(scipy.special.binom(n, k))
	return get_log_factorial(n) - (get_log_factorial(k) + get_log_factorial(n-k))

def approx_log_binomiam_pmf(n, k, p):
	return get_log_binomial_coefficient(n, k) + k*np.log(p) + (n-k)*np.log(1-p)

def binomial_pmf(x,n,p):
	# binomial distribution 
	" P(k | n, p)  = n!/(k! * (n-k)!) * p^k * (1-p)^(n-k)   "
	" scipy.stats.logpmf(x, n, p, loc=0) "
	pmf = scipy.stats.binom.logpmf(x, n, p, loc=0)
	return pmf
	#if pmf > -np.inf:
	#	return pmf
	#else: 
	#	return approx_log_binomiam_pmf(x,n,p)




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


threshold_CI = 11
freq_zero_preservation = 0 #.25
log_q_mean = -5
log_q_std = 1
max_epsilon = 0.5
out_name = "rootest_q_%s_%s_epsilon_%s_fZero_%s_thr_%s.log" % (abs(log_q_mean),log_q_std,max_epsilon,freq_zero_preservation,threshold_CI)


logfile = open(out_name, "w") 
text_str = "iteration\tlikelihood\tNobs\tmu0_true\troot_true\tq_med_true\tepsilon_true\troot_obs\troot\troot_m2\troot_M2\troot_m4\troot_M4\troot_mth\troot_Mth\tdelta_lik"
logfile.writelines(text_str)
n_samples = 20

for replicate in range(n_simulations):
	# observed time bins
	true_root = np.random.uniform(50,200)
	mid_points = np.linspace(2.5,500,int(500/5))
	#print(mid_points)
	#mid_points = np.array([0]+list(mid_points))
	#mid_points= mid_points[0:-1] + np.diff(mid_points)/2.
	
	
	Nobs = np.random.randint(500,5000)
	x_0 = 1
	true_mu0 = (Nobs-x_0)/true_root

	# MAKE DIV TRAJECTORY
	Ntrue = Nobs - mid_points[mid_points<true_root]*true_mu0

	# add noise
	true_epsilon = np.random.uniform(0,max_epsilon)
	Ntrue = np.rint(np.exp(np.random.normal(np.log(Ntrue),true_epsilon)))
	print( "Ntrue", Ntrue)


	true_q = np.exp( np.random.normal(log_q_mean,log_q_std,len(mid_points)))[mid_points<true_root]
	true_q = true_q * np.random.binomial(1,1-freq_zero_preservation,len(true_q))
	true_q[true_q>0.1] = 0.1
	#np.random.gamma(1.01,.01,len(mid_points))[mid_points<true_root]
	# preserved occs
	x = np.rint(Ntrue*true_q)[::-1] # order is temporarily reversed
	# remove first x values if they are == 0
	j,c=0,0
	for i in range(len(x)):
		if x[i]==0 and j==0:
			c+=1
		else:
			break

	x = x[c:][::-1]
	print(x)
	age_oldest_obs_occ = mid_points[len(x)-1]
	print(true_root, age_oldest_obs_occ)
	print("true_q",true_q, np.max(true_q)/np.min(true_q[true_q>0]))

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
		#print(x_augmented,Nest,q_augmented)
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
	
	indx_max_lik = np.argmax(out_array[:,0])
	max_lik = out_array[indx_max_lik,0]
	root_ml = out_array[indx_max_lik,2]
	min_max_range = np.array([i for i in range(n_samples) if max_lik-out_array[i,0]<2])
	root_min2 = np.min(out_array[min_max_range,2])
	root_max2 = np.max(out_array[min_max_range,2])
	min_max_range = np.array([i for i in range(n_samples) if max_lik-out_array[i,0]<4])
	root_min4 = np.min(out_array[min_max_range,2])
	root_max4 = np.max(out_array[min_max_range,2])	
	min_max_range = np.array([i for i in range(n_samples) if max_lik-out_array[i,0]<threshold_CI])
	root_min8 = np.min(out_array[min_max_range,2])
	root_max8 = np.max(out_array[min_max_range,2])
	index_binned_true_root = np.argmin((np.abs(out_array[:,2]-true_root)))
	delta_lik_true_to_est_root = max_lik - out_array[index_binned_true_root,0]	
	
	text_str = "\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ( replicate, max_lik, Nobs, true_mu0, true_root, 
			np.median(true_q), true_epsilon, age_oldest_obs_occ, root_ml,root_min2,root_max2,root_min4,root_max4,root_min8,root_max8,delta_lik_true_to_est_root )
	print(root_ml, delta_lik_true_to_est_root)				
	logfile.writelines(text_str)
	logfile.flush()
	