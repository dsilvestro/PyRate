#!/usr/bin/env python 
import argparse, os,sys,csv
#from numpy import *
import numpy as np
import scipy, scipy.stats
from scipy.special import gamma
np.set_printoptions(suppress= 1) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit
import copy
# CALC BAYS FACTORS BASED ON PROB pI
def F(p=0.05, BF_threshold=6):
	A = exp(BF_threshold/2.)*(p/(1-p))
	return A/(A+1)

p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('-A',          type=int,   help='set 1 to estimate marginal likelihood', default=0, metavar=0)
p.add_argument('-d',          type=str,   help='input file', default="", metavar="<inputfile.txt>")
p.add_argument('-n',          type=int,   help='mcmc generations',default=100000, metavar=100000)
p.add_argument('-s',          type=int,   help='sample freq.', default=100, metavar=100)
p.add_argument('-p',          type=int,   help='print freq.', default=1000, metavar=1000)
p.add_argument('-t0',         type=float, help='max age time slice', default=np.inf, metavar=np.inf)
p.add_argument('-t1',         type=float, help='min age time slice', default=0, metavar=0)
p.add_argument('-seed',       type=int,   help='seed (if -1 -> random)', default=-1, metavar=-1)
p.add_argument('-mu_species', type=int,   help='set to 1 to save extinction rates foreach trait combination', default=0, metavar=0)
p.add_argument('-traits',     type=int ,  help="index trait(s)",metavar='',nargs='+',default=[])
p.add_argument('-const',      type=int ,  help="if set to 1: trait independent rate",metavar=0,default=0)
p.add_argument('-out',        type=str,   help='output string', default="", metavar="")
p.add_argument('-bvs',        type=int,   help='use Bayesian Variable Selection', default=1, metavar=1)
p.add_argument('-pI',         type=float, help='prior on indicators', default=0.05, metavar=0.05)



args = p.parse_args()

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

# 'Exponential' hyper-prior on rate parameter of Gamma prior on Precision
alpha_0 = 1.
beta_0 = .1
# Shape Gamma on tau
Gamma_a_prior = 1.5  



# FUNCTIONS
def rDir(params):
	sample = np.random.gamma(params,1,len(params))
	return sample/np.sum(sample)

def rDirE(params):
	return params/np.sum(params)

def get_mle_mu(ts,te):
	return len(te[te>0])/np.sum(ts-te)

def get_death_lik(mu,bl,ex,tr=[]):
	if len(tr)==0: tr = np.zeros(len(bl)).astype(int)
	return ex*np.log(mu[tr]) - (bl*mu[tr])

def update_multiplier(q,d=1.1,f=0.5):
	S = np.shape(q)
	ff = np.random.binomial(1,f,S)
	u = np.random.uniform(0,1,S)
	l = 2*np.log(d)
	m = np.exp(l*(u-.5))
	m[ff==0] = 1.
	new_q = q * m
	U=np.sum(np.log(m))
	return new_q,U

def update_normal(q,d=0.5,f=0.5):
	S=np.shape(q)
	ff=np.random.binomial(1,f,S)
	n = np.random.normal(0,d,S)
	n[ff==0] = 0
	new_q = q + n
	return new_q

def update_normal_first_zero(q,d=0.5,f=0.5):
	# the first is redundant and fixed to 0!
	S=np.shape(q)
	ff=np.random.binomial(1,f,S)
	ff[0] = 0
	n = np.random.normal(0,d,S)
	n[ff==0] = 0
	new_q = q + n
	return new_q

def update_indicators(I_vecA):
	r= np.random.choice(np.arange(n_traits))
	I_vec_temp = 0 + I_vecA
	I_vec_temp[r] = 1 - I_vec_temp[r] # if 0 return(1), if 1 return(0)
	return I_vec_temp

def prior_gamma(L,a=1,b=0.01):
	return scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0)

def get_death_lik_lineage(mu,bl,ex):
	return ex*np.log(mu) - (bl*mu)

def sample_std_hp_gibbs_rate(x,a,b,mu=0):
	Gamma_a = a + len(x)/2. # one observation for each value (1 Y for 1 s2)
	Gamma_b = b + np.sum((x-mu)**2)/2.
	tau = np.random.gamma(Gamma_a, scale=1./Gamma_b)
	return 1/np.sqrt(tau) 

def prior_normal(x,m=0,sd=1): 
	return scipy.stats.norm.logpdf(x,loc=m,scale=sd)

def sample_G_rate_gibbs(sd,a):
	# prior is on precision tau
	tau = 1./(sd**2) #np.array(tau_list)
	conjugate_a = alpha_0 + len(tau)*a
	conjugate_b = beta_0 + np.sum(tau)
	return np.random.gamma(conjugate_a,scale=1./conjugate_b)


##### READ DATA
head = next(open(args.d)).split()
print(head)
data = np.loadtxt(args.d, delimiter='\t', skiprows=1, usecols=list(range(1,len(head))))
ts_list = data[:,0]
te_list = data[:,1]
tr_name_list = head[3:]

traits_indx = np.array(args.traits)+1
if len(traits_indx)==0:
	traits_indx = np.arange(2,len(head)-1)
print(traits_indx, np.shape(data))
tr_list = data[:,traits_indx].astype(int)
print("Size total dataset:", len(ts_list),len(te_list),len(tr_list), tr_name_list)

# subset data
old_sp = (te_list<max_age).nonzero()[0]
young_sp = (ts_list>min_age).nonzero()[0]
in_species = np.intersect1d(old_sp,young_sp)
ts_list = ts_list[in_species]
te_list = te_list[in_species]
tr_list = tr_list[in_species]

ts_list[ts_list>max_age] = max_age
ts_list -= min_age
te_list -= min_age
te_list[te_list<=0] = 0.
m_tag = ""
trait_tag=""
trait_name_vec = []
print(tr_name_list,traits_indx)
for i in traits_indx:
	trait_name_vec.append(tr_name_list[i-2])

# output directory
self_path= [os.path.dirname(sys.argv[0]) , os.getcwd()]
output_wd = os.path.dirname(args.d)
if output_wd=="": output_wd= max(self_path)
name_file = os.path.splitext(os.path.basename(args.d))[0]

trait_tag = args.out
if args.const== 1: trait_tag = "const_"
if max_age < np.inf or min_age > 0:
	out_file_name = "%s/%s_%s%s-%s%s.log" % (output_wd,name_file,trait_tag,max_age,min_age,m_tag)
else:
	out_file_name = "%s/%s_%s%s.log" % (output_wd,name_file,trait_tag,m_tag)
	


print("Size dataset after filtering time slice:", len(ts_list),len(te_list),len(tr_list))
print("time range", max(ts_list), min(te_list))
n_traits = len(tr_list[0])
trait_categories_list = []
mu_list = []
tr_list_transf = tr_list+0
# transform traits to avoid gaps
for i in range(n_traits):
	trait_categories_list.append(len(np.unique(tr_list[:,i])))
	print("states:           ", np.unique(tr_list[:,i]))
	mu_list.append( np.ones(len(np.unique(tr_list[:,i])))/len(np.unique(tr_list[:,i])) )
	
	state_count = 0
	tr_temp = tr_list_transf[:,i]
	for j in np.sort(np.unique(tr_list[:,i])):
		tr_temp[tr_temp==j] = state_count
		state_count+=1
	tr_list_transf[:,i] = tr_temp
	print("states tranformed:", np.unique(tr_list_transf[:,i]))

n_lineages = len(ts_list)
print(mu_list, n_traits)
print(np.shape(tr_list))
print("states", trait_categories_list)
print("extinction rate (mle):", get_mle_mu(ts_list,te_list))

print(len(te_list[te_list>0]), (ts_list-te_list))

bl = ts_list-te_list # branch length
ex = (te_list>0)*1   # extinction ID (0: extant, 1: extinct)

# get indexes of one species for each trait observed combination
def list_to_string(l):
	s ="m"
	for i in l: s+= "%s" % (i)
	return s

trait_comb_all = [] # labels for each species (e.g. [['m1', 'm1', 'm3', ... ])
			  #, only used if saving sp-specific ex.rates
unique_trait_comb = [] # unique labels of each trait combination (e.g. ['m0', 'm1', 'm2' ... ])
unique_trait_comb_indx = [] # index of 1 species for each trait combination (only used to np.log final ex.rate)
for i in range(n_lineages):
	s=list_to_string(tr_list[i,:])
	trait_comb_all.append(s)
	if s in unique_trait_comb: pass
	else: 
		unique_trait_comb.append(s)
		unique_trait_comb_indx.append(i)
print("unique_trait_comb", len(unique_trait_comb_indx))
# print unique_trait_comb
# #quit()
# print trait_comb_all, len(trait_comb_all)
# print tr_list


# count n. of species in each combination of character states
trait_comb_all = np.array(trait_comb_all)

unique_trait_comb_name = []
for i in unique_trait_comb:
	unique_trait_comb_name.append(i+"_%s" % (len(trait_comb_all[trait_comb_all==i])))

# init model parameters
Y_vecA = [np.random.normal(0,0.1,i) for i in trait_categories_list]
I_vecA = np.zeros(n_traits) # trait-specific indicators
Gamma_b_prior = np.ones(n_traits) # shrinking rate -> larger gamma, larger tau, smaller std normal
std_HP = np.array([sample_std_hp_gibbs_rate(Y_vecA[i],Gamma_a_prior,Gamma_b_prior[i]) for i in range(n_traits)])  #
Gamma_b_prior = sample_G_rate_gibbs(std_HP,Gamma_a_prior)
multipA = np.array([get_mle_mu(ts_list,te_list)]) # mean extinction rate
likA = -np.inf
prior_Y = [np.sum(prior_normal(Y_vecA[i],m=0,sd=std_HP[i])) for i in range(n_traits)]
prior_m = 0 # uniform prior on mean rate (can be replaced by something else)
priorA = np.sum(prior_Y) + prior_m

# MCMC settings

#multipA = np.array([0.75])
ws_multi = 1.1
#Y_vecA = mu_list
print("mu_list: ",mu_list)
iteration = 0 

# init MCMC np.log file
logfile = open(out_file_name , "w") 
wlog=csv.writer(logfile, delimiter='\t')

head="it\tposterior\tlikelihood\tprior\tmean_r"
print(trait_name_vec)
for i in range(n_traits):
	head+= "\tI_%s" % (trait_name_vec[i])
for i in range(n_traits):
	for j in np.sort(np.unique(tr_list[:,i])):
		head+= "\tm_%s_%s" % (trait_name_vec[i],j)
for i in range(n_traits):
	head+= "\tsd_%s" % (trait_name_vec[i])

head += "\tbeta_hp"

if args.mu_species==1:
	for i in unique_trait_comb_name:
		head+= "\t%s" % (i)


wlog.writerow(head.split('\t'))
logfile.flush()

ncat=10
K=ncat-1.        # K+1 categories
k=np.array(list(range(int(K+1))))
beta=k/K
alpha=0.3            # categories are beta distributed
temperatures=list(beta**(1./alpha))
temperatures[0]+= 0.00005 # avoid exactly 0 temp
temperatures.reverse()
it_change = np.cumsum(np.zeros(ncat)+n_iterations).astype(int)
ind_temp = 0 
temperature=temperatures[0]

if args.const == 1: # constant rate model
	freq_update_Ys = 0
	freq_update_m  = 1
	freq_update_I = 0
else:
	if args.bvs == 1:
		freq_update_Ys = 0.45
		freq_update_m  = 0.01
		freq_update_I = 0.45
	else:
		freq_update_Ys = 0.90
		freq_update_m  = 0.05
		freq_update_I = 0
		I_vecA = np.ones(n_traits)


# RUN MCMC
while iteration <= n_iterations:
	gibbs = 0
	rr = np.random.random()
	
	# reset temp values
	multip,hasting = 0+multipA,0
	I_vec = I_vecA
	Y_vec = copy.deepcopy(Y_vecA)
	
	#random update
	# update state-spec Dirichlet multipliers
	if rr < freq_update_Ys: 
		indx_updated_Y = np.random.choice(list(range(n_traits)))
		Y_vec[indx_updated_Y] = update_normal_first_zero(Y_vec[indx_updated_Y])
	
	# update mean extinction rate
	elif rr <(freq_update_Ys+freq_update_m): 
		multip,hasting = update_multiplier(multipA,ws_multi,1)

	# update Indicators
	elif rr <(freq_update_Ys+freq_update_m+freq_update_I) and iteration>1000: 
		I_vec = update_indicators(I_vecA)
	
	# update hyper-priors
	else:
		gibbs = 1
		if np.random.random()>0.5:
			Gamma_b_prior = sample_G_rate_gibbs(std_HP,Gamma_a_prior)
		else:
			std_HP = np.array([sample_std_hp_gibbs_rate(Y_vecA[i],Gamma_a_prior,Gamma_b_prior) for i in range(n_traits)]) 
	
	# lineage-sp rates
	np.sum_rates,prior_Y,prior_Tau,prior_Beta = np.zeros(n_lineages),0,0,0
	for i in range(n_traits):
		# if indicator is =0, the trait is not doing anything to the extinction rate
		if I_vec[i]==0:
			transform_trait_multiplier = np.ones(len(Y_vec[i]))/len(Y_vec[i])
		else:
			transform_trait_multiplier = np.exp(Y_vec[i])/np.sum(np.exp(Y_vec[i])) #
		# create array of multipliers (1 per species) based on trait states
		# using trait states as indexes (must start from 0 and be sequential!)
		np.sum_rates += transform_trait_multiplier[tr_list_transf[:,i]]
		# calculate priors
		prior_Y    += np.sum(prior_normal(Y_vec[i],m=0,sd=std_HP[i]))
		prior_Tau  += prior_gamma(1./(std_HP[i]**2), Gamma_a_prior,Gamma_b_prior)
	
	prior_Beta += prior_gamma(Gamma_b_prior, alpha_0, beta_0)
	prior_m = prior_gamma(multip[0],a=1.,b=1.) # exponential prior on mean extinction rate
	np.sum_rates = np.sum_rates/np.sum(np.sum_rates) * len(np.sum_rates)
	rates = multip * np.sum_rates
	lik =  np.sum(get_death_lik_lineage(rates,bl,ex))
	prior = prior_Y + prior_m + prior_Tau + prior_Beta
	prior += np.sum(np.log(args.pI)*I_vec)
	
	if (lik - likA) + (prior - priorA) + hasting >= np.log(np.random.random()) or gibbs==1:
		Y_vecA = Y_vec
		multipA = multip
		likA = lik
		I_vecA = I_vec
		priorA = prior
		ratesA = rates




	if iteration % p_freq ==0:
		print(iteration,likA, multipA)
		print(prior_Y, prior_m, prior_Tau, prior_Beta, gibbs)
		#post_temp=[]
		#for i in range(n_traits):
		#	if I_vecA[i]==0:
		#		transform_trait_multiplier = np.ones(len(Y_vec[i]))/len(Y_vec[i])
		#	else:
		#		transform_trait_multiplier = exp(Y_vec[i])/np.sum(exp(Y_vec[i])) #
		#	post_temp += list(transform_trait_multiplier) #/mean(transform_trait_multiplier)) 
		#print np.array(post_temp)
		print(I_vecA)
		#print "Y ARRAY:", Y_vecA
	if iteration % s_freq ==0:
		post_temp,post_sd=[],[]
		for i in range(n_traits):
			#transform_trait_multiplier_temp = Y_vec[i] # exp(Y_vec[i])/np.sum(exp(Y_vec[i]))
			#post_temp += list(transform_trait_multiplier_temp)
			if I_vecA[i]==0:
				transform_trait_multiplier_temp = np.ones(len(Y_vecA[i]))/len(Y_vecA[i])
			else:
				transform_trait_multiplier_temp = np.exp(Y_vecA[i])/np.sum(np.exp(Y_vecA[i])) #
			
			transform_trait_multiplier_temp = np.exp(Y_vecA[i])/np.sum(np.exp(Y_vecA[i]))
			post_temp += list(transform_trait_multiplier_temp) #/mean(transform_trait_multiplier_temp)) 
		post_sd   = list(std_HP)
		post_log = [iteration, likA+priorA, likA, priorA,multipA[0]] + list(I_vecA) + post_temp + post_sd
		post_log += [Gamma_b_prior]
		if args.mu_species==1: post_log += list(ratesA[unique_trait_comb_indx])
		wlog.writerow(post_log)
		logfile.flush()
	
	iteration+=1


