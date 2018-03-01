#!/usr/bin/env python 
import argparse, os,sys,csv
from numpy import *
import numpy as np
import scipy, scipy.stats
from scipy.special import gamma
np.set_printoptions(suppress= 1) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit


p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('-A',         type=int,   help='set 1 to estimate marginal likelihood', default=0, metavar=0)
p.add_argument('-d',         type=str,   help='input file', default="", metavar="<inputfile.txt>")
p.add_argument('-n',         type=int,   help='mcmc generations',default=100000, metavar=100000)
p.add_argument('-s',         type=int,   help='sample freq.', default=100, metavar=100)
p.add_argument('-p',         type=int,   help='print freq.', default=1000, metavar=1000)
p.add_argument('-hp',        type=int,   help='set to 0) for fixed (flat) rate parameter (G prior on tau), 1) for single estimated prior for all traits, \
2) for one estimated prior for each trait (only works if -A 0)', default=2, metavar=2)
p.add_argument('-t0',        type=float, help='max age time slice', default=np.inf, metavar=np.inf)
p.add_argument('-t1',        type=float, help='min age time slice', default=0, metavar=0)
p.add_argument('-seed',      type=int,   help='seed (if -1 -> random)', default=-1, metavar=-1)
p.add_argument('-mu_species',type=int,   help='set to 1 to save extinction rates foreach trait combination', default=0, metavar=0)
p.add_argument('-traits',    type=int ,  help="index trait(s)",metavar='',nargs='+',default=[1])
p.add_argument('-const',     type=int ,  help="if set to 1: trait independent rate",metavar=0,default=0)



args = p.parse_args()

use_HP = args.hp 
n_iterations = args.n
s_freq = args.s
p_freq = args.p
max_age = args.t0
min_age = args.t1

if args.d =="": sim_data=1
else: sim_data=0

if args.seed>=0: rseed = args.seed
else: rseed = rseed=np.random.randint(0,9999)
np.random.seed(rseed)

# FUNCTIONS
def rDir(params):
	sample = np.random.gamma(params,1,len(params))
	return sample/sum(sample)

def rDirE(params):
	return params/sum(params)

def get_mle_mu(ts,te):
	return len(te[te>0])/sum(ts-te)

def get_death_lik(mu,bl,ex,tr=[]):
	if len(tr)==0: tr = np.zeros(len(bl)).astype(int)
	return ex*log(mu[tr]) - (bl*mu[tr])

def update_multiplier(q,d=1.1,f=0.5):
	S=np.shape(q)
	ff=np.random.binomial(1,f,S)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
	m[ff==0] = 1.
 	new_q = q * m
	U=sum(log(m))
	return new_q,U

def update_normal(q,d=0.5,f=0.5):
	S=np.shape(q)
	ff=np.random.binomial(1,f,S)
	n = np.random.normal(0,d,S)
	n[ff==0] = 0
 	new_q = q + n
	return new_q

def prior_gamma(L,a=1,b=0.01):
	return scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0)

def get_death_lik_lineage(mu,bl,ex):
	return ex*log(mu) - (bl*mu)

def sample_s2hp_gibbs(Yvec,a,b):
	invGamma_a = a + 1./2 # one observation for each value (1 Y for 1 s2)
	invGamma_b = b + ((Yvec-0.)**2)/2.
	return 1./np.random.gamma(invGamma_a, scale=1./invGamma_b)

def sample_tau_hp_gibbs(Yvec,a,b):
	Gamma_a = a + 1./2 # one observation for each value (1 Y for 1 s2)
	Gamma_b = b + ((Yvec-0.)**2)/2.
	tau = np.random.gamma(Gamma_a, scale=1./Gamma_b)
	return sqrt(1./tau) # sqrt(variance): return std

def prior_normal(x,m=0,sd=1): 
	return scipy.stats.norm.logpdf(x,loc=m,scale=sd)

def sample_G_rate_gibbs(sd,a):
	# prior is on precision tau
	tau_list = []
	for s in sd: tau_list += list(1./(s**2))
	tau = np.array(tau_list)
	alpha_0 = 1.
	beta_0 = .1
	conjugate_a = alpha_0 + len(tau)*a
	conjugate_b = beta_0 + sum(1./tau)
	return np.random.gamma(conjugate_a,1./conjugate_b)

if sim_data==1:
	out_file_name = "mTraitBD.log"
	# MAKE UP DATA
	trait_categories_list = [4,2]
	#mu_list = [[0.2,0.5,0.05,1.2], [0.1,0.75]]
	mean_rate = 0.75
	mu_list_init = [np.array([0.1,0.5,5,0.5]), np.array([1.5,0.5])]
	mu_list_init = [np.array([0.1,0.5,3,0.5]), np.array([1.5,0.5])]
	#mu_list = [np.array([0.25,0.25,0.25,0.25])*mean_rate, np.array([0.5,0.5])*mean_rate]
	mu_list=[]
	for i in mu_list_init:
		j = i/sum(i)
		mu_list.append(j)
		print j/mean(j), j #, rDirE(i)
	#_ 
	n_traits = len(trait_categories_list)
	n_lineages =20
	#mu_by_lineage
	print "mu_list",mu_list

	print "true rescaled multi"
	for i in mu_list:
		print i/mean(i)


	# simulate traits
	tr_list = []
	root = 5
	for i in range(n_lineages):
		tr = []
		for j in range(len(trait_categories_list)):
			tr_i = np.random.randint(trait_categories_list[j])
			tr  += [tr_i]
		# add traits
		tr_list.append(tr)

	tr_list= np.array(tr_list).astype(int)


	# get rates per lineage
	sum_rates = np.zeros(n_lineages)
	for i in range(n_traits):
		transform_trait_multiplier = rDirE(mu_list[i])
	
		print transform_trait_multiplier
		r_temp =  transform_trait_multiplier # * trait_categories_list[i] 
		sum_rates += r_temp[tr_list[:,i]] #* (1./n_traits)

	sum_rates = sum_rates/sum(sum_rates) * len(sum_rates)
	prior_m = 0 # uniform prior on mean rate (can be replaced by something else)
	rates = mean_rate * sum_rates
	print "mean rate:",round(mean(rates),2), mean_rate, mean(sum_rates)

	# simulate lineages
	ts_list = []
	te_list = []

	root = 5
	for i in range(n_lineages):
		mu = rates[i]
		ts=np.ones(1)*root
		te=ts-np.random.exponential(1/mu,1)
		te[te<0]=0
		mle_mu = get_mle_mu(ts,te)
		ts_list += list(ts)
		te_list += list(te)

	ts_list= np.array(ts_list)
	te_list= np.array(te_list)

else:
	##### READ DATA
	head = next(open(args.d)).split()
	print head
	data = np.loadtxt(args.d, delimiter='\t', skiprows=1, usecols=range(1,len(head)))
	ts_list = data[:,0]
	te_list = data[:,1]
	tr_name_list = head[3:]
	if min(args.traits) < 1 or max(args.traits) > len(tr_name_list):
		sys.exit("Trait not found.")

	traits_indx = np.array(args.traits)+1
	print traits_indx, np.shape(data)
	tr_list = data[:,traits_indx].astype(int)
	print "Siza total dataset:", len(ts_list),len(te_list),len(tr_list), tr_name_list
	
	# subset data
	old_sp = (te_list<max_age).nonzero()[0]
	young_sp = (ts_list>min_age).nonzero()[0]
	in_species = np.intersect1d(old_sp,young_sp)
	ts_list = ts_list[in_species]
	te_list = te_list[in_species]
	tr_list = tr_list[in_species]

	ts_list[ts_list>max_age] = max_age
	ts_list -= min_age
	te_list[te_list<=min_age] = 0.
	if use_HP==0: m_tag = ""
	elif use_HP==1: m_tag ="_hp1"
	else: m_tag ="hp2"
	trait_tag=""
	for i in args.traits: 
		trait_tag+= "%s_" % (tr_name_list[i-1])
	print trait_tag
	if args.const== 1: trait_tag = "const_"
	if max_age < np.inf or min_age > 0:
		out_file_name = "trait_%s%s-%s%s.log" % (trait_tag,max_age,min_age,m_tag)
	else:
		out_file_name = "trait_%s%s.log" % (trait_tag,m_tag)

print "Siza dataset after filtering time slice:", len(ts_list),len(te_list),len(tr_list)
print "time range", max(ts_list), min(te_list)
n_traits = len(tr_list[0])
trait_categories_list = []
mu_list = []
for i in range(n_traits):
	trait_categories_list.append(len(np.unique(tr_list[:,i])))
	print "states", np.unique(tr_list[:,i])
	mu_list.append( np.ones(len(np.unique(tr_list[:,i])))/len(np.unique(tr_list[:,i])) )

n_lineages = len(ts_list)
print mu_list, n_traits
print np.shape(tr_list)
print "states", trait_categories_list
#quit()

print "extinction rate (mle):", get_mle_mu(ts_list,te_list)

bl = ts_list-te_list # branch length
ex = (te_list>0)*1   # extinction ID (0: extant, 1: extinct)
#print ex,(te_list>0)
#quit()

# get indexes of one species for each trait observed combination
def list_to_string(l):
	s ="m"
	for i in l: s+= "%s" % (i)
	return s

trait_comb_all = [] # labels for each species (e.g. [['m1', 'm1', 'm3', ... ])
			  #, only used if saving sp-specific ex.rates
unique_trait_comb = [] # unique labels of each trait combination (e.g. ['m0', 'm1', 'm2' ... ])
unique_trait_comb_indx = [] # index of 1 species for each trait combination (only used to log final ex.rate)
for i in range(n_lineages):
	s=list_to_string(tr_list[i,:])
	trait_comb_all.append(s)
	if s in unique_trait_comb: pass
	else: 
		unique_trait_comb.append(s)
		unique_trait_comb_indx.append(i)
print "unique_trait_comb", len(unique_trait_comb_indx)
print unique_trait_comb
#quit()
print trait_comb_all, len(trait_comb_all)
print tr_list


# count n. of species in each combination of character states
trait_comb_all = np.array(trait_comb_all)

unique_trait_comb_name = []
for i in unique_trait_comb:
	unique_trait_comb_name.append(i+"_%s" % (len(trait_comb_all[trait_comb_all==i])))

# init model parameters
Y_vecA = [np.zeros(i) for i in trait_categories_list] #
Gamma_a_prior = 1.5  # Gamma hyperprior: for a>1, mean = b / (a-1) | invGamma
Gamma_b_prior = 10. # Gamma on tau: mean on s2 -> 1./(a/b)
sd_hp = [sample_tau_hp_gibbs(Y_vecA[i],Gamma_a_prior,Gamma_b_prior) for i in range(n_traits)]  #
multipA = np.array([get_mle_mu(ts_list,te_list)])
likA = -np.inf
prior_Y = [sum(prior_normal(Y_vecA[i],m=0,sd=sd_hp[i])) for i in range(n_traits)]
prior_m = 0 # uniform prior on mean rate (can be replaced by something else)
priorA = sum(prior_Y) + prior_m
if use_HP==2: Gamma_b_prior = [10 for i in range(n_traits)]

# MCMC settings

#multipA = np.array([0.75])
ws_multi = 1.1
Y_vecA = mu_list
print "mu_list: ",mu_list
iteration = 0 

# init MCMC log file
out_file_name = "test_"+out_file_name
logfile = open(out_file_name , "wb") 
wlog=csv.writer(logfile, delimiter='\t')

head="it\tposterior\tlikelihood\tprior\tmean_r"
for i in range(n_traits):
	for j in range(trait_categories_list[i]):
		head+= "\tm_%s_%s" % (args.traits[i],j)
for i in range(n_traits):
	for j in range(trait_categories_list[i]):
		head+= "\tsd_%s_%s" % (args.traits[i],j)

if use_HP==1: head += "\tbeta_hp"

elif use_HP==2: 
	for i in range(n_traits):
		head += "\tbeta_hp_%s" % (args.traits[i])

if args.mu_species==1:
	for i in unique_trait_comb_name:
		head+= "\t%s" % (i)

if args.A==1: 
	head +=  "\ttemperature"

wlog.writerow(head.split('\t'))
logfile.flush()

ncat=10
K=ncat-1.        # K+1 categories
k=np.array(range(int(K+1)))
beta=k/K
alpha=0.3            # categories are beta distributed
temperatures=list(beta**(1./alpha))
temperatures[0]+= 0.00005 # avoid exactly 0 temp
temperatures.reverse()
it_change = np.cumsum(np.zeros(ncat)+n_iterations).astype(int)
ind_temp = 0 
temperature=temperatures[0]
if args.A==1: 
	n_iterations = max(it_change)
	sd_hp = [np.ones(len(Y_vecA[i])) for i in range(n_traits)]  #
	freq_update_Ys = 0.6
	freq_update_m  = 0.4
	ws_multi = 1.2
else:
	freq_update_Ys = 0.75
	freq_update_m  = 0.15

if args.const == 1: # constant rate model
	freq_update_Ys = 0
	freq_update_m  = 1



# RUN MCMC
while iteration <= n_iterations:
	if iteration == it_change[ind_temp]+1:
		ind_temp+=1
		temperature = temperatures[ind_temp]
		print iteration, temperature
	
	gibbs = 0
	rr = np.random.random()
	if rr < freq_update_Ys:
		indx_updated_Y = np.random.choice(range(n_traits))
		Y_vec = []+Y_vecA
		Y_vec[indx_updated_Y] = update_normal(Y_vec[indx_updated_Y])
		multip,hasting = multipA,0
	elif rr <(freq_update_Ys+freq_update_m):
		multip,hasting = update_multiplier(multipA,ws_multi,1)
		Y_vec = Y_vecA
	elif args.A==0: # further during TDI stop sampling priors
		gibbs = 1
		if use_HP==1: 
			sd_hp = [sample_tau_hp_gibbs(Y_vecA[i],Gamma_a_prior,Gamma_b_prior) for i in range(n_traits)]
			Gamma_b_prior = sample_G_rate_gibbs(sd_hp,Gamma_a_prior)
		elif use_HP==2: 
			sd_hp = [sample_tau_hp_gibbs(Y_vecA[i],Gamma_a_prior,Gamma_b_prior[i]) for i in range(n_traits)]
			Gamma_b_prior = [sample_G_rate_gibbs([sd_hp[i]],Gamma_a_prior) for i in range(n_traits)]
		multip,hasting = multipA,0
		Y_vec = Y_vecA
	
	# lineage-sp rates
	sum_rates,prior_Y = np.zeros(n_lineages),0
	for i in range(n_traits):
		transform_trait_multiplier = exp(Y_vec[i])/sum(exp(Y_vec[i])) #
		# create array of multipliers (1 per species) based on trait states
		# using trait states as indexes (must start from 0 and be sequential!)
		sum_rates += transform_trait_multiplier[tr_list[:,i]]
		prior_Y += sum(prior_normal(Y_vec[i],m=0,sd=sd_hp[i]))
	
	prior_m = prior_gamma(multip[0],a=1.,b=1.) # exponential prior
	sum_rates = sum_rates/sum(sum_rates) * len(sum_rates)
	rates = multip * sum_rates
	lik =  sum(get_death_lik_lineage(rates,bl,ex))
	prior = prior_Y + prior_m
	
	if (lik - likA)*temperature + (prior - priorA) + hasting >= log(np.random.random()) or gibbs==1:
		Y_vecA = Y_vec
		multipA = multip
		likA = lik
		priorA = prior
		ratesA = rates

	if iteration % p_freq ==0:
		print iteration,likA, multipA
		post_temp=[]
		for i in range(n_traits):
			transform_trait_multiplier = exp(Y_vec[i])/sum(exp(Y_vec[i]))
			post_temp += list(transform_trait_multiplier) #/mean(transform_trait_multiplier)) 
		print np.array(post_temp)
	if iteration % s_freq ==0:
		post_temp,post_sd=[],[]
		for i in range(n_traits):
			#transform_trait_multiplier_temp = Y_vec[i] # exp(Y_vec[i])/sum(exp(Y_vec[i]))
			#post_temp += list(transform_trait_multiplier_temp)
			transform_trait_multiplier_temp = exp(Y_vec[i])/sum(exp(Y_vec[i]))
			post_temp += list(transform_trait_multiplier_temp) #/mean(transform_trait_multiplier_temp)) 
			post_sd   += list(sd_hp[i])
		post_log = [iteration, likA+priorA, likA, priorA,multipA[0]] + post_temp + post_sd 
		if use_HP==1: post_log += [Gamma_b_prior]
		if use_HP==2: post_log += Gamma_b_prior
		if args.mu_species==1: post_log += list(ratesA[unique_trait_comb_indx])
		if args.A==1: post_log += [temperature]
		wlog.writerow(post_log)
		logfile.flush()
	
	iteration+=1


