import numpy as np
from numpy import *
import sys, argparse, os, csv
import importlib.util
import scipy.stats
from scipy.special import gamma
from scipy.special import gdtr, gdtrix
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  


p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('-d',      type=str,   help='pyrate input file (.py)')
p.add_argument('-m',      type=str,   help='mcmc file (mcmc.log)')
p.add_argument('-q',      type=str,   help='qShift file (same file used for PyRate analyses)')
p.add_argument('-n',      type=int,   help='n. MCMC iterations', default = 250000)
p.add_argument('-s',      type=int,   help='sampling freq', default = 500)
p.add_argument('-p',      type=int,   help='print freq', default = 1000)
p.add_argument('-b',      type=int,   help='number of bins', default = 20)
p.add_argument('-j',      type=int,   help='max PyRate replicate', default = 1)
p.add_argument('-v',      type=int,   help='verbose', default = 0)
p.add_argument('-N',      type=int,   help='specify diversity in first bin', default = -1)
p.add_argument('-rescale',type=int,   help='rescale time axis', default = 1)


args = p.parse_args()

pyrate_file = args.d
logfile = args.m

path_infile = os.path.dirname(pyrate_file)
input_file = os.path.splitext(os.path.basename(pyrate_file))[0]

q_shifts = np.sort(np.loadtxt(args.q))[::-1] # reversed order from old to young to match PyRate
rescale_factor = args.rescale # assumes only q_rates must be rescaled (not q_shift and fossil occcs)
n_bins = args.b               # mcmc logfile output (reversed in get_q_rates_time_bins())

verbose = args.v 
modern_diversity = args.N




def parse_pyrate_file(j, time_bins): 
	# get occurrences
	x = input_data_module.get_data(j)
	for i in range(len(x)):
		x[i] = x[i]
		
	# merge all occs and get occs count per time bin
	occs=[]
	for i in x: 
		occs += list(i)
	
	occs = np.array(occs)
	occs= np.sort(occs)
	occs = occs[occs>0]
	time_zero = floor(min(occs))
	hist_occurrences = np.histogram(occs,bins=time_bins)[0]
	
	# get number of sampled species per time bin
	hsp = np.zeros(len(hist_occurrences))
	root_age,tips_age = 0,np.inf
	for i in range(len(x)):
		x_temp = x[i]
		x_temp = x_temp[x_temp>0]
		hsp_temp = np.histogram(x_temp,bins=time_bins)[0]
		hsp_temp[hsp_temp>0] = 1
		hsp +=hsp_temp
		if np.max(x[i]) > root_age:
			root_age = np.max(x[i])
		if np.min(x[i]) < tips_age:
			tips_age = np.min(x[i])
	
	actual_n_sampled_species = hsp + 0
	return ([actual_n_sampled_species, time_bins, x, hist_occurrences,time_zero,root_age,tips_age])

def get_q_rates(logfile):
	tbl = np.loadtxt(logfile,skiprows=1)
	lines = np.shape(tbl)[0]
	btbl = tbl[int(0.2*lines):,:]	
	head = next(open(logfile)).split()
	q_rates_index,alpha_index = [head.index(i) for i in head if "q_" in i],  (np.array(head)=="alpha").nonzero()[0]
	root_index = (np.array(head)=="root_age").nonzero()[0]
	tips_time_index = (np.array(head)=="death_age").nonzero()[0]
	q_rates = btbl[:,q_rates_index] * rescale_factor	
	alphas = btbl[:,alpha_index] 
	roots = btbl[:,root_index]
	tips_time = btbl[:,tips_time_index]
	return np.median(q_rates,axis=0),np.median(alphas),np.min(roots),np.max(tips_time),q_rates, alphas

def get_q_rates_time_bins(q_rates,q_shifts,time_bins):
	q_rates_bins = np.zeros(len(time_bins)-1)
	indx = np.digitize(time_bins,q_shifts)
	q_rates_bins = q_rates[indx] # from recent to old
	return q_rates_bins

def update_multiplier_freq(q,d=1.1,f=0.25):
	S=np.shape(q)
	ff=np.random.binomial(1,f,S)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
	m[ff==0] = 1.
	new_q = q * m
	U=sum(log(m))
	return new_q,U

def update_multiplier(q,d=1.1):
	u = np.random.uniform(0,1)
	l = 2*np.log(d)
	m = np.exp(l*(u-.5))
	new_q = q * m
	U=np.log(m)
	return new_q,U



q_rates, alpha, root, tips_time, res_q, res_alpha = get_q_rates(logfile)
time_bins = np.sort(np.array(list(np.linspace(tips_time,root,n_bins)) + list(q_shifts)))  # from recent to old
if verbose: print( "rates",q_rates, alpha )

input_file_raw = os.path.basename(pyrate_file)
input_file = os.path.splitext(input_file_raw)[0] # file name without extension
# import pyrate input file
test_spec = importlib.util.spec_from_file_location(input_file,pyrate_file)
input_data_module = importlib.util.module_from_spec(test_spec)
test_spec.loader.exec_module(input_data_module)

res = parse_pyrate_file(0, time_bins)

x_obs = res[0] # from recent to old
root, tips_time = res[5],res[6]

q_shifts = q_shifts[q_shifts<root]
q_shifts = q_shifts[q_shifts>tips_time]


time_bins = np.sort(np.array(list(np.linspace(tips_time,root,n_bins)) + list(q_shifts)))  # from recent to old
dTime = np.diff(time_bins)

# reload PyRate file
res = parse_pyrate_file(0, time_bins)
x_obs = res[0] # from recent to old


if verbose: print("time_bins",time_bins)

# init diversity multipliers assuming const diversity
n_multi_A = np.max(x_obs)-x_obs + 1
if modern_diversity >= 0:
	n_multi_A[0] = 0
n_est = x_obs + n_multi_A

# init q multiplier
q_multi_A = 1.

def get_q_rate_samples():
	q_rates, alpha = res_q[np.random.choice(range(len(res_alpha))),:], res_alpha[np.random.choice(range(len(res_alpha)))]
	if verbose: print("resample:",q_rates, alpha)
	q_rates_bins = get_q_rates_time_bins(q_rates,q_shifts,time_bins)
	rGamma = np.random.gamma(alpha,1./alpha,(10000,len(q_rates_bins)-1))
	q_rates_bins_G = rGamma*(q_rates_bins[:-1]*q_multi_A)
	if verbose: print(q_rates_bins)
	if verbose: print(np.mean(q_rates_bins_G,0))
	rho_bins_array = 1 - np.exp(-(q_rates_bins_G)*dTime)
	rho_bins = np.mean(rho_bins_array,0)
	if verbose: print(1 - np.exp(-(q_rates_bins[:-1])*dTime))
	if verbose: print(rho_bins)
	ncat=100
	YangGammaQuant=(np.linspace(0,1,ncat+1)-np.linspace(0,1,ncat+1)[1]/2)[1:]
	rGamma = gdtrix(alpha,alpha,YangGammaQuant) # user defined categories
	rGamma = np.repeat(rGamma,len(q_rates_bins)-1).reshape((ncat,len(q_rates_bins)-1))
	return q_rates_bins, rGamma, alpha



q_rates_bins, rGamma, alpha = get_q_rate_samples()

q_rates_bins_G = rGamma*(q_rates_bins[:-1]*q_multi_A)
if verbose: print(np.mean(q_rates_bins_G,0))
rho_bins_array = 1 - np.exp(-(q_rates_bins_G)*dTime)
rho_bins = np.mean(rho_bins_array,0)
if verbose: print(1 - np.exp(-(q_rates_bins[:-1])*dTime))
if verbose: print(rho_bins)

#print(dTime)
#print(time_bins)
#quit()

sig2_A = 10.
prior_A = np.sum(scipy.stats.norm.logpdf(  np.diff(np.log(n_est)), 0, np.sqrt(sig2_A*dTime[:-1]) )) 
prior_A += scipy.stats.gamma.logpdf(sig2_A,1,scale=1) #+ scipy.stats.norm.logpdf(np.log(q_multi_A),0,1) #+ 
prior_A += scipy.stats.gamma.logpdf(q_multi_A,1,scale=1) 
lik_A   = np.sum(scipy.stats.binom.logpmf( x_obs, n_est, rho_bins ))

# init mcmc file
div_output_file = "%s/%s_mcmcdiv.log" % (path_infile, input_file)
output_logfile = open(div_output_file , "w") 
wlog=csv.writer(output_logfile, delimiter='\t')
head = ["it","posterior","likelihood","prior","q_multi","sig2_hp"]
mid_points = (time_bins[1:] + time_bins[:-1]) / 2
for i in mid_points: 
	head.append("t_%s" % round(i,3))

wlog.writerow(head)
output_logfile.flush()

q_multi = q_multi_A

freq_resample_q = 2500
freq_resample_occs = 1000

print(x_obs)

for iteration in range(args.n):
	rr= np.random.random()
	if rr < 0.001:
		# reload PyRate file
		res = parse_pyrate_file(np.random.choice(np.arange(args.j)), time_bins)
		x_obs = res[0] # from recent to old
	elif rr < 0.002:
		q_rates_bins, rGamma, alpha = get_q_rate_samples()
		n_multi, sig2 = n_multi_A, sig2_A 
		u1,u2,u3 = 0,0,0
	else:
		n_multi, u1 = update_multiplier_freq(n_multi_A,f=0.4)
		q_multi, u2 = q_multi_A, 0 #update_multiplier(q_multi_A,1.3)
		sig2, u3     = update_multiplier(sig2_A,1.2)
		if modern_diversity >= 0:
			n_multi[0]   = 0
		n_est = x_obs + n_multi
	
	q_rates_bins_G = rGamma*(q_rates_bins[:-1]*q_multi)
	rho_bins_array = 1 - np.exp(-(q_rates_bins_G)*dTime)
	rho_bins = np.mean(rho_bins_array,0)
	
	prior = np.sum(scipy.stats.norm.logpdf(  np.diff(np.log(n_est)), 0, np.sqrt(sig2*dTime[:-1]) ))
	prior += scipy.stats.gamma.logpdf(sig2,1,scale=1) #+ scipy.stats.norm.logpdf(np.log(q_multi),0,1) # +
	prior += scipy.stats.gamma.logpdf(q_multi,1,scale=1) 
	lik   = np.sum(scipy.stats.binom.logpmf( x_obs, n_est, rho_bins ))
	if ( prior-prior_A + lik-lik_A + u1+u2+u3 ) >= np.log(np.random.random()) or rr < 0.002: # or iteration % freq_resample_q==0:
		n_multi_A = n_multi
		q_multi_A = q_multi
		sig2_A = sig2
		prior_A = prior
		lik_A = lik
	
	if iteration % args.p ==0:
		print(iteration, lik_A,q_multi_A, sig2_A, alpha )
	if iteration % args.s ==0:
		mcmc_sample = [iteration,lik_A+prior_A, lik_A,prior_A,q_multi_A, sig2_A ] + list(x_obs + n_multi_A)
		wlog.writerow(mcmc_sample)
		output_logfile.flush()
		


n_est = x_obs + n_multi_A
if verbose: print(n_est)



