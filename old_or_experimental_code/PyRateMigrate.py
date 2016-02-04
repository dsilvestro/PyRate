#!/usr/bin/env python 
import argparse, os,sys
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import platform, time
import csv
import scipy.stats
from scipy.optimize import fmin_powell as Fopt1 

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
self_path=os.getcwd()


### LIK/PRIOR FUNCTIONS
try: 
	scipy.stats.gamma.logpdf(1, 1, scale=1./1,loc=0) 
	def prior_gamma(L,a,b): 
		return scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0)
	def prior_normal(L,sd): 
		return scipy.stats.norm.logpdf(L,loc=0,scale=sd)
	def prior_cauchy(x,s):
		return scipy.stats.cauchy.logpdf(x,scale=s,loc=0)
except(AttributeError): # for older versions of scipy
	def prior_gamma(L,a,b):  
		return (a-1)*log(L)+(-b*L)-(log(b)*(-a)+ log(gamma(a)))
	def prior_normal(L,sd): 
		return -(x**2/(2*sd**2)) - log(sd*sqrt(2*np.pi))
	def prior_cauchy(x,s):
		return -log(np.pi*s * (1+ (x/s)**2))

def PP_lik_global(n_events,n_lineages,rates): # calculates BD_partial_lik for a vector of rates
	# number of lineages
	n_S= n_lineages #*grid_size
	# likelihood
	lik= log(rates)*n_events -rates*n_S
	return sum(lik)




####### BEGIN FUNCTIONS for DIRICHLET PROCESS PRIOR #######
def cond_alpha_proposal(hp_gamma_shape,hp_gamma_rate,current_alpha,k,n):
	z = [current_alpha + 1.0, float(n)]
	f = np.random.dirichlet(z,1)[0]
	eta = f[0]	
	u = np.random.uniform(0,1,1)[0]
	x = (hp_gamma_shape + k - 1.0) / ((hp_gamma_rate - np.log(eta)) * n)
	if (u / (1.0-u)) < x: new_alpha = np.random.gamma( (hp_gamma_shape+k), (1./(hp_gamma_rate-np.log(eta))) )
	else: new_alpha = np.random.gamma( (hp_gamma_shape+k-1.), 1./(hp_gamma_rate-np.log(eta)) )
	return new_alpha

def random_choice_P(vector):
	probDeath=np.cumsum(vector/sum(vector)) # cumulative prob (used to randomly sample one 
	r=np.random.random()                          # parameter based on its deathRate)
	probDeath=sort(append(probDeath, r))
	ind=np.where(probDeath==r)[0][0] # just in case r==1
	return [vector[ind], ind]

def calc_rel_prob(log_lik):
	rel_prob=exp(log_lik-max(log_lik))
	return rel_prob/sum(rel_prob)
	
def G0(alpha=2,beta=3,n=1):
	#return np.array([np.random.random()])
	#return np.random.gamma(shape=alpha,scale=1./beta,size=n)
	return init_BD(n)

def PP_partial_lik_vec(i_events,i_n_lineages,rate): # calculates BD_partial_lik for a vector of rates
	# number of lineages
	n_S= i_n_lineages #*grid_size
	# likelihood
	lik= log(rate)*i_events -rate*n_S
	return lik


def DDP_gibbs_sampler(args): # rate_type = "l" or "m" (for speciation/extinction respectively)
	[parA,ind,mt_numbers,n_lineages,hp_gamma_rate,alpha_par_Dir] = args
	# par: parameters for each category
	n_time_bins=len(ind)
	# GIBBS SAMPLER for NUMBER OF CATEGORIES - Algorithm 4. (Neal 2000)
	par=parA # parameters for each category
	eta = np.array([len(ind[ind==j]) for j in range(len(par))]) # number of elements in each category
	u1 = np.random.uniform(0,1,n_time_bins) # init random numbers
	new_lik_vec=np.zeros(n_time_bins) # store new sampled likelihoods
	new_alpha_par_Dir = 0 + cond_alpha_proposal(hp_gamma_shape,hp_gamma_rate,alpha_par_Dir,len(par),n_time_bins)
	for i in range(0,n_time_bins):
		#up=time_frames[i]
		#lo=time_frames[i+1]
		k1 = len(par)

		if len(ind[ind==ind[i]])==1: # is singleton
			k1 = k1 - 1
			par_k1 = par			
			if u1[i]<= k1/(k1+1.): pass
			else: ind[i] = k1 + 1 # this way n_ic for singleton is not 0
		else: # is not singleton
			par_k1 = np.concatenate((par,G0()), axis=0)
		
		# construct prob vector FAST!
		lik_vec=PP_partial_lik_vec(mt_numbers[i],n_lineages[i],par_k1)
		rel_lik = calc_rel_prob(lik_vec)
		if len(par_k1)>len(eta): # par_k1 add one element only when i is not singleton
			eta[ind[i]] -= 1
			eta_temp=np.append(eta,new_alpha_par_Dir/(k1+1.))
		else: eta_temp = eta
		P=eta_temp*rel_lik
	
		# randomly sample a new value for indicator ind[i]
		IND = random_choice_P(P)[1] 
		ind[i] = IND # update ind vector
		if IND==(len(par_k1)-1): par = par_k1 # add category


		# Change the state to contain only those par are now associated with an observation
		# create vector of number of elements per category
		eta = np.array([len(ind[ind==j]) for j in range(len(par))])
		# remove parameters for which there are no elements
		par = par[eta>0]
		# rescale indexes
		ind_rm = (eta==0).nonzero()[0] # which category has no elements
		if len(ind_rm)>0: ind[ind>=ind_rm] = ind[ind>=ind_rm]-1
		# update eta
		eta = np.delete(eta,ind_rm)

		# Update lik vec
		new_lik_vec[i]=lik_vec[IND]

	likA = sum(new_lik_vec)
	parA = par
	return likA,parA, ind,new_alpha_par_Dir


def get_rate_HP(n,target_k,hp_gamma_shape):
	def estK(alpha,N): 
		return sum([alpha/(alpha+i-1) for i in range(1,int(N+1))])
	
	def opt_gamma_rate(a):
		a= abs(a[0])
		ea =estK(a,n)
		return exp(abs( ea-target_k ))
	# from scipy.optimize import fmin_powell as Fopt1 
	opt = Fopt1(opt_gamma_rate, [np.array(0.001)], full_output=1, disp=0)
	expected_cp=abs(opt[0])
	hp_gamma_rate = expected_cp/hp_gamma_shape 
	return hp_gamma_rate

####### END FUNCTIONS for DIRICHLET PROCESS PRIOR #######

####### INIT FUNCTIONS
def init_BD(n):
	return np.random.exponential(.2, max(n-1,1))+.1

def update_multiplier_proposal(i,d):
	S=np.shape(i)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
 	ii = i * m
	U=sum(log(m))
	return ii, U


def MCMC(mt_numbers,n_lineages):
	m_ratesA = init_BD(1) # init 1 rate
	indDPP  = np.zeros(len(mt_numbers)).astype(int) # init category indexes
	alpha_par_Dir = np.random.uniform(0,1) # init concentration parameters
	hp_gamma_rate  = get_rate_HP(len(mt_numbers),target_k,hp_gamma_shape)
	print "hp_gamma_rate:",hp_gamma_rate, len(indDPP),len(m_ratesA[indDPP])
	
	for iteration in range(IT):
		if np.random.random()< .2:
			gibbs, hasting=1,0
			args = [m_ratesA,indDPP,mt_numbers,n_lineages,hp_gamma_rate,alpha_par_Dir]
			lik, m_rates, indDPP, alpha_par_Dir = DDP_gibbs_sampler(args)
		else:
			gibbs=0
			m_rates,hasting = update_multiplier_proposal(m_ratesA, 1.2) 
			lik = PP_lik_global(mt_numbers,n_lineages,m_rates[indDPP])

		prior = sum(prior_gamma(m_rates,a=2,b=3))
		
		Post = lik+prior
		if iteration==0: PostA=Post
		
		if Post-PostA + hasting >= log(np.random.random()) or gibbs==1:
			PostA=Post
			priorA=prior
			likA=lik
			m_ratesA=m_rates
		
		# screen output
		if iteration % print_freq ==0:
			l=[round(y, 2) for y in [PostA, likA, priorA]]
			print "\n%s\tpost: %s lik: %s prior: %s k: %s" % (iteration, l[0], l[1], l[2], len(m_rates)), lik
			#print m_rates
			#print "ind", indDPP
		
		if iteration % sample_freq ==0:
			log_state= [iteration,PostA,likA,priorA,len(m_rates),alpha_par_Dir]	
			marg_rates=zeros(len(marginal_frames))
			for i in range(len(indDPP)): # indexes of the 1 time unit within each timeframe
				ind=np.intersect1d(marginal_frames[marginal_frames<=fixed_times_of_shift[i]],marginal_frames[marginal_frames>=fixed_times_of_shift[i+1]])
				j=array(ind)
				marg_rates[j]=m_ratesA[indDPP[i]]
			
			log_state += list(marg_rates)
			wlog.writerow(log_state)
			logfile.flush()
			os.fsync(logfile)
			
	


# CALC MT
def get_mt_numbers(migration_times,grid_size):	
	fixed_times_of_shift=np.arange(0,100,grid_size)[::-1] # make a grid
	fixed_times_of_shift=fixed_times_of_shift[fixed_times_of_shift<max(migration_times)]
	B=np.append(max(fixed_times_of_shift)+grid_size, fixed_times_of_shift)
	mt_numbers = np.histogram(migration_times,bins=sort(B))[0]
	mt_numbers = mt_numbers[::-1] # number of events per grid cell
	return mt_numbers,B


# MCMC PARAMETERS
p = argparse.ArgumentParser() #description='<input file>') 

p.add_argument('-v',         action='version', version='%(prog)s')
p.add_argument('-n',         type=int,   help='MCMC iterations', default=10000, metavar=10000)
p.add_argument("-g",         type=float, help='grid size', default=2., metavar=2.) 
p.add_argument('-M',         type=int,   help='no. migration events', default=50, metavar=50)
p.add_argument('-L',         type=int,   help='no. lineages', default=100, metavar=100)
p.add_argument('-p',         type=int,   help='print freq', default=1000, metavar=1000)
p.add_argument('-s',         type=int,   help='sample freq.', default=10, metavar=10)
p.add_argument('-k',         type=int,   help='target K', default=2, metavar=2)

args = p.parse_args()

grid_size      = args.g
IT             = args.n
print_freq     = args.p
sample_freq    = args.s
target_k       = args.k
N_migr_events  = args.M
N_lineages     = args.L
hp_gamma_shape = 2.

# DATA
# ### make up some migration data
# poi1 = np.random.poisson(2.5*5,20)
# poi2 = np.random.poisson(2.5*5,20)
# migration_times=[]
# for i in range(1,20):
# 	if i <10 or i > 16: migration_times += list(np.random.uniform(i-1,i,poi1[i]))
# 	else: migration_times += list(np.random.uniform(i-1,i,poi2[i]))
# 	
# migration_times = np.array(migration_times)
# #migration_times = np.random.exponential(5,500)

migration_times = np.linspace(0,19,N_migr_events)

migration_times=np.sort(migration_times)[::-1]
mt_numbers,fixed_times_of_shift = get_mt_numbers(migration_times,grid_size)

lineages_times = np.linspace(0,19,N_lineages)
n_lineages,temp = get_mt_numbers(lineages_times,grid_size)
print n_lineages
n_lineages = np.ones(len(mt_numbers)) # number of lineages per grid cell (if 1 -> standard Poi process)
#n_lineages = np.sort(np.random.geometric(0.1,len(mt_numbers)))
print n_lineages

#print mt_numbers
#print n_lineages 
#print fixed_times_of_shift
#print sum(mt_numbers), sum(n_lineages)

# OUPUT FILES
out_log = "migration_mcmc.log" 
logfile = open(out_log , "w",0) 
head="it\tpost\tlik\tprior\tk\talpha\t"
for i in range(int(max(fixed_times_of_shift))+1): head += "m_%s\t" % i 
head=head.split('\t')
wlog=csv.writer(logfile, delimiter='\t')
wlog.writerow(head)
logfile.flush()
os.fsync(logfile)
marginal_frames= array([int(fabs(i-int(max(fixed_times_of_shift)))) for i in range(int(max(fixed_times_of_shift))+1)])
#print marginal_frames


t1 = time.time()

MCMC(mt_numbers,n_lineages)

print "\nelapsed time:", np.round(time.time()-t1,2), "\n"

