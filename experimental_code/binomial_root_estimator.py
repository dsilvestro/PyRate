from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
import scipy.stats
#np.random.seed(1234)

n_simulations = 1000
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


logfile = open("est_root_epochs.log", "w") 
if save_mcmc_samples:
	text_str = "iteration\tlikelihood\tmu0\troot\tq"
	logfile.writelines(text_str)
else:
	text_str = "iteration\tlikelihood\tNobs\tmu0_true\troot_true\tq_med_true\tepsilon_true\troot_obs\tmu0\troot\troot_m\troot_M\tq"
	logfile.writelines(text_str)
	n_samples = int((n_iterations - burnin)/sampling_freq)

for replicate in range(n_simulations):
	# observed time bins
	true_root = np.random.uniform(50,200)
	mid_points = np.loadtxt("/Users/danielesilvestro/Software/PyRate_github/example_files/epochs_q.txt")
	#mid_points = np.array([0]+list(mid_points))
	#mid_points= mid_points[0:-1] + np.diff(mid_points)/2.
	
	
	Nobs = np.random.randint(500,5000)
	x_0 = 1
	true_mu0 = (Nobs-x_0)/true_root

	# MAKE DIV TRAJECTORY
	Ntrue = Nobs - mid_points[mid_points<true_root]*true_mu0
	# add noise
	true_epsilon = np.random.gamma(2.,0.12)
	Ntrue = np.rint(np.exp(np.random.normal(np.log(Ntrue),true_epsilon)))
	print(Ntrue)

	# MAKE UP FOSSIL DATA
	# sampling probability
	true_q = np.exp( np.random.normal(-5,0.5,len(mid_points)))[mid_points<true_root]
	#np.random.gamma(1.01,.01,len(mid_points))[mid_points<true_root]
	# preserved occs
	x = np.rint(Ntrue*true_q)
	# remove first x values if they are == 0
	j,c=0,0
	for i in range(len(x)):
		if x[i]==0 and j==0:
			c+=1
		else:
			break

	x = x[c:]
	age_oldest_obs_occ = np.sort(mid_points)[len(x)-1]
	print(x)
	print("true_q",true_q, np.max(true_q)/np.min(true_q))


	# init parameters
	root_A = 1
	mu0_A = (Nobs-x_0)/(root_A+age_oldest_obs_occ)
	q_A = np.random.gamma(1.1,.01)
	Nest = Nobs - mid_points[mid_points<(root_A+age_oldest_obs_occ)]*mu0_A

	x_augmented = np.concatenate( (np.zeros(len(Nest)-len(x)), x  )  )
	q_augmented = np.repeat(q_A,len(x_augmented))

	likA = np.sum(binomial_pmf(x_augmented,Nest,q_augmented))
	if save_mcmc_samples==0:
		out_array = np.zeros( (n_samples, 4 ) )
	
	j=0
	for iteration in range(n_iterations):
		q,hastings = update_multiplier(q_A)	
		root = np.fabs(np.random.normal(root_A,1) )
		mu0 = (Nobs-x_0)/(root+age_oldest_obs_occ)
		Nest = Nobs - mid_points[mid_points<(root+age_oldest_obs_occ)]*mu0
		x_augmented = np.concatenate( (np.zeros(len(Nest)-len(x)), x  )  )
		q_augmented = np.repeat(q,len(x_augmented))	
		lik = np.sum(binomial_pmf(x_augmented,Nest,q_augmented))
		if (lik - likA) + hastings  > np.log(np.random.random()):
			likA = lik
			q_A = q
			root_A =root
			mu0_A = mu0
	
		if iteration % sampling_freq ==0:
			if save_mcmc_samples:
				text_str = "\n%s\t%s\t%s\t%s\t%s" % (iteration,likA,mu0_A,root_A+age_oldest_obs_occ,q_A)
				logfile.writelines(text_str)
				logfile.flush()
			elif iteration >= burnin:
				out_array[j] = np.array([likA,mu0_A,root_A+age_oldest_obs_occ,q_A])
				j+=1
	#print(out_array)
	if save_mcmc_samples==0:
		mean_values = np.mean(out_array,0)
		print(mean_values)
		root_hpd = calcHPD(out_array[:,2])
		print(true_root, root_hpd, age_oldest_obs_occ,"\n")
		
		text_str = "iteration\tlikelihood\tNobs\tmu0_true\troot_true\tq_med_true\tepsilon_true\troot_obs\tmu0\troot\troot_m\troot_M\tq"
		text_str = "\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ( replicate, mean_values[0],Nobs, true_mu0, true_root, np.median(true_q), \
				true_epsilon, age_oldest_obs_occ, mean_values[1],mean_values[2],root_hpd[0],root_hpd[1],mean_values[3] )
		logfile.writelines(text_str)
		logfile.flush()
	