#!/usr/bin/env python 
# Created by Daniele Silvestro on 04/04/2018 => pyrate.help@gmail.com 
import os,csv,platform
import argparse, os,sys, time
import math
from numpy import *
import numpy as np
import scipy
import scipy.linalg
linalg = scipy.linalg
import scipy.stats
import random as rand
import nlopt
from scipy.integrate import odeint
try: 
	import multiprocessing, thread
	import multiprocessing.pool
	use_seq_lik=False
	if platform.system() == "Windows" or platform.system() == "Microsoft": use_seq_lik=True
except(ImportError): 
	print("\nWarning: library multiprocessing not found.\nPyRateDES will use (slower) sequential likelihood calculation. \n")
	use_seq_lik=True

self_path=os.getcwd()

# DES libraries
self_path= os.path.dirname(sys.argv[0])
import imp

try: 
	self_path= os.path.dirname(sys.argv[0])
	des_model_lib = imp.load_source("des_model_lib", "%s/pyrate_lib/des_model_lib.py" % (self_path))
	mcmc_lib = imp.load_source("mcmc_lib", "%s/pyrate_lib/des_mcmc_lib.py" % (self_path))
	lib_DD_likelihood = imp.load_source("lib_DD_likelihood", "%s/pyrate_lib/lib_DD_likelihood.py" % (self_path))
	lib_utilities = imp.load_source("lib_utilities", "%s/pyrate_lib/lib_utilities.py" % (self_path))
except:
	self_path=os.getcwd()
	des_model_lib = imp.load_source("des_model_lib", "%s/pyrate_lib/des_model_lib.py" % (self_path))
	mcmc_lib = imp.load_source("mcmc_lib", "%s/pyrate_lib/des_mcmc_lib.py" % (self_path))
	lib_DD_likelihood = imp.load_source("lib_DD_likelihood", "%s/pyrate_lib/lib_DD_likelihood.py" % (self_path))
	lib_utilities = imp.load_source("lib_utilities", "%s/pyrate_lib/lib_utilities.py" % (self_path))

from des_model_lib import *
from mcmc_lib import *
from lib_utilities import *

np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit
small_number= 1e-5

citation= """\nThe DES method is described in:\nSilvestro, D., Zizka A., Bacon C. D., Cascales-Minana B. and Salamin, N., Antonelli, A. (2016)
Fossil Biogeography: A new model to infer dispersal, extinction and sampling from paleontological data.
Phil. Trans. R. Soc. B 371: 20150225.\n
"""


p = argparse.ArgumentParser() #description='<input file>') 

p.add_argument('-v',        action='version', version='%(prog)s')
p.add_argument('-cite',     help='print DES citation', action='store_true', default=False)
p.add_argument('-A',        type=int, help='algorithm - 0: parameter estimation, 1: TI, 2: ML 3: ML (subplex algorithm; experimental; time dependent and simple mechanistic models)', default=0, metavar=0) # 0: par estimation, 1: TI
p.add_argument('-k',        type=int,   help='TI - no. scaling factors', default=10, metavar=10)
p.add_argument('-a',        type=float, help='TI - shape beta distribution', default=.3, metavar=.3)
p.add_argument('-hp',       help='Use hyper-prior on rates', action='store_true', default=False)
p.add_argument('-pw',       type=float, help='Exponent acceptance ratio (ML)', default=58, metavar=58) # accept 0.95 post ratio with 5% prob
p.add_argument('-seed',     type=int, help='seed (set to -1 to make it random)', default= 1, metavar= 1)
p.add_argument('-out',      type=str, help='output string',   default="", metavar="")

p.add_argument('-d',        type=str, help='input data set',   default="", metavar="<input file>")
p.add_argument('-n',        type=int, help='mcmc generations',default=100000, metavar=100000)
p.add_argument('-s',        type=int, help='sample freq.', default=100, metavar=100)
p.add_argument('-p',        type=int, help='print freq.',  default=100, metavar=100)
p.add_argument('-b',        type=int, help='burnin',  default=0)
p.add_argument('-thread',   type=int, help='no threads',  default=0)
p.add_argument('-ver',      type=int, help='verbose',   default=0, metavar=0)
p.add_argument('-pade',     type=int, help='0) Matrix decomposition 1) Use Pade approx (slow)', default=0, metavar=0)
p.add_argument('-qtimes',   type=float, help='shift time (Q)',  default=[], metavar=0, nargs='+') # '*'
p.add_argument('-sum',      type=str, help='Summarize results (provide log file)',  default="", metavar="log file")
p.add_argument('-symd',     help='symmetric dispersal rates', action='store_true', default=False)
p.add_argument('-syme',     help='symmetric extinction rates', action='store_true', default=False)
p.add_argument('-symq',     help='symmetric preservation rates', action='store_true', default=False)
p.add_argument('-symCov',   type=int,  help='symmetric correlations',  default=[], metavar=0, nargs='+') # '*'
p.add_argument('-constq',   type=int,  help='if 1 (or 2): constant q in area 1 (or 2)', default=0)
p.add_argument('-constr',   type=int, help='Contraints on covar parameters',  default=[], metavar=0, nargs='+') # '*'
#p.add_argument('-constA',   type=int, help='Contraints on covar parameters',  default=[], metavar=0, nargs='+') # '*'
p.add_argument('-data_in_area', type=int,  help='if data only in area 1 set to 1 (set to 2 if data only in area 2)', default=0)
p.add_argument('-varD',      type=str, help='Time variable file DISPERSAL (e.g. PhanerozoicTempSmooth.txt)',  default="", metavar="")
p.add_argument('-varE',      type=str, help='Time variable file EXTINCTION (e.g. PhanerozoicTempSmooth.txt)',  default="", metavar="")
p.add_argument('-red',      type=int, help='if -red 1: reduce dataset to taxa with occs in both areas', default=0)
p.add_argument('-DivdD',    help='Use Diversity dependent Dispersal',  action='store_true', default=False)
p.add_argument('-DivdE',    help='Use Diversity dependent Extinction', action='store_true', default=False)
p.add_argument('-DdE',      help='Use Dispersal dependent Extinction', action='store_true', default=False)
p.add_argument('-DisdE',    help='Use Dispersal-rate dependent Extinction', action='store_true', default=False)
p.add_argument('-TdD',      help='Use Time dependent Dispersal',  action='store_true', default=False)
p.add_argument('-TdE',      help='Use Time dependent Extinction', action='store_true', default=False)
p.add_argument('-lgD',      help='Use logistic correlation Dispersal',  action='store_true', default=False)
p.add_argument('-lgE',      help='Use logistic correlation Extinction', action='store_true', default=False)
p.add_argument('-linE',      help='Use linear correlation Extinction', action='store_true', default=False)
p.add_argument('-cov_and_dispersal', help='Model with symmetric extinction covarying with both a proxy and dispersal', action='store_true', default=False)
#p.add_argument('-const', type=int, help='Constant d/e rate ()',default=0)
p.add_argument('-fU',     type=float, help='Tuning - update freq. (d, e, s)', default=[0, 0, 0], nargs=3)
p.add_argument("-mG", help='Model - Gamma heterogeneity of preservation rate', action='store_true', default=False)
p.add_argument("-ncat", type=int, help='Model - Number of categories for Gamma heterogeneity', default=4, metavar=4)



### simulation settings ###
p.add_argument('-sim_d',  type=float, help='dispersal rates',  default=[.4, .1], metavar=1.1, nargs=2)
p.add_argument('-sim_e',  type=float, help='extinction rates', default=[.1, .05], metavar=.1, nargs=2)
p.add_argument('-sim_q',  type=float, help='preservation rates',   default=[1.,1.,], metavar=1, nargs=2)
p.add_argument('-i',      type=int,   help='simulation number',  default=0)
p.add_argument('-ran',    help='random settings', action='store_true', default=False)
p.add_argument('-t',      type=int, help='no taxa',  default=50, metavar=50)
# number of bins used to code the (simulated) geographic ranges
p.add_argument('-n_bins',  type=int, help='no bins',  default=20, metavar=20)
# number of bins to approximate continuous time DES process when simulating the data
p.add_argument('-n_sim_bins',  type=int, help='no bins for simulation',  default=1000,metavar=1000)
p.add_argument("-wd",        type=str, help='path to working directory', default="")

args = p.parse_args()
if args.cite is True:
	sys.exit(citation)
simulation_no = args.i

if args.seed==-1:
	rseed=np.random.randint(0,9999)
else: rseed=args.seed	
random.seed(rseed)
np.random.seed(rseed)

print "Random seed: ", rseed

burnin= args.b
n_taxa= args.t
num_processes=args.thread
verbose= args.ver
n_bins= args.n_bins
n_sim_bins= args.n_sim_bins
sim_d_rate = np.array(args.sim_d)
sim_e_rate = np.array(args.sim_e)
q_rate = args.sim_q
nareas=2
sampling_prob_per_bin=q_rate
input_data= args.d
Q_times=np.sort(args.qtimes)
output_wd = args.wd
if output_wd=="": output_wd= self_path

equal_d = args.symd
equal_e = args.syme
equal_q = args.symq
constraints_covar = np.array(args.constr)
const_q = args.constq
if args.cov_and_dispersal:
	model_DUO= 1
else: model_DUO= 0
argsG = args.mG
pp_gamma_ncat = args.ncat

# if args.const==1:
# 	equal_d = True # this makes them symmatric not equal!
# 	equal_e = True # this makes them symmatric not equal!
# 	args.TdD = True
# 	args.TdE = True

### MCMC SETTINGS
if args.A ==2: 
	runMCMC = 0 # approximate ML estimation
	n_generations   = 100000
	if sum(args.fU)==0:
		update_freq     = [.4,.8,1]
	else: update_freq = args.fU
	sampling_freq   = 10
	max_ML_iterations = 5000
else: 
	runMCMC = 1
	n_generations   = args.n
	if sum(args.fU)==0:
		update_freq     = [.4,.8,1]
		if args.hp == True: 
			update_freq = [.3,.6,.9]
	else: update_freq = args.fU
	sampling_freq   = args.s

print_freq      = args.p
map_power       = args.pw
hp_alpha        = 2.
hp_beta         = 2.
use_Pade_approx = args.pade
scale_proposal  = 1

#### SUMMARIZE RESULTS
if args.sum !="":
	f=args.sum
	if burnin==0: 
		print """Burnin was set to 0. Use command -b to specify a higher burnin
	(e.g. -b 100 will exclude the first 100 samples)."""
	t=loadtxt(f, skiprows=max(1,burnin))

	head = next(open(f)).split()
	start_column = 4
	j=0

	outfile=os.path.dirname(f)+"/"+os.path.splitext(os.path.basename(f))[0]+"_sum.txt"
	out=open(outfile, "wb")

	out.writelines("parameter\tmean\tmode\tHPDm\tHPDM\n")
	for i in range(start_column,len(head)-1):
		par = t[:,i]
		hpd = np.around(calcHPD(par, .95), decimals=3)
		mode = round(get_mode(par),3)
		mean_par = round(mean(par),3)
		if i==start_column: out_str= "%s\t%s\t%s\t%s\t%s\n" % (head[i], mean_par,mode,hpd[0],hpd[1])
		else: out_str= "%s\t%s\t%s\t%s\t%s\n" % (head[i], mean_par,mode,hpd[0],hpd[1])
		out.writelines(out_str)
		j+=1

	s= "\nA summary file was saved as: %s\n\n" % (outfile)
	sys.exit(s)

### INIT SIMULATION SETTINGS
# random settings
if args.ran is True:
	sim_d_rate = np.round(np.random.uniform(.025,.2, 2),2)
	sim_e_rate = np.round(np.random.uniform(0,sim_d_rate, 2),2)
	n_taxa = np.random.randint(20,75)
	q_rate = np.round(np.random.uniform(.05,1, 2),2)

print sim_d_rate,sim_e_rate

# sampling rates
TimeSpan = 50.
sim_bin_size = TimeSpan/n_sim_bins
# 1 minus prob of no occurrences (Poisson waiting time) = prob of at least 1 occurrence [assumes homogenenous Poisson process]
# bin size:   [ 10.   5.    2.5   1.    0.5 ]
# q = 1.0   P=[ 1.    0.99  0.92  0.63  0.39]
# q = 0.5   P=[ 0.99  0.92  0.71  0.39  0.22]
# q = 0.1   P=[ 0.63  0.39  0.22  0.1   0.05]
#sampling_prob_per_bin = np.round(np.array([1-exp(-q_rate[0]*bin_size), 1-exp(-q_rate[1]*bin_size)]),2)
sampling_prob_per_sim_bin = np.array([1-exp(-q_rate[0]*sim_bin_size), 1-exp(-q_rate[1]*sim_bin_size)])

#########################################
######       DATA SIMULATION       ######
#########################################
if input_data=="":
	print "simulating data..."
	simulate_dataset(simulation_no,sim_d_rate,sim_e_rate,n_taxa,n_sim_bins,output_wd)
	RHO_sampling = np.array(sampling_prob_per_sim_bin)
	time.sleep(1)
	input_data = "%s/sim_%s_%s_%s_%s_%s_%s.txt" % (output_wd,simulation_no,n_taxa,sim_d_rate[0],sim_d_rate[1],sim_e_rate[0],sim_e_rate[1]) 
	nTaxa, time_series, obs_area_series, OrigTimeIndex = parse_input_data(input_data,RHO_sampling,verbose,n_sampled_bins=n_bins)
	print obs_area_series
	if args.A==1: ti_tag ="_TI"
	else: ti_tag=""
	out_log = "%s/simContinuous_%s_b_%s_q_%s_mcmc_%s_%s_%s_%s_%s_%s_%s%s.log" \
	% (output_wd,simulation_no,n_bins,q_rate[0],n_taxa,sim_d_rate[0],sim_d_rate[1],sim_e_rate[0],sim_e_rate[1],q_rate[0],q_rate[1],ti_tag)
	time_series = np.sort(time_series)[::-1]
	
else:
	print "parsing input data..."
	RHO_sampling= np.ones(2)
	nTaxa, time_series, obs_area_series, OrigTimeIndex = parse_input_data(input_data,RHO_sampling,verbose,n_sampled_bins=0,reduce_data=args.red)
	name_file = os.path.splitext(os.path.basename(input_data))[0]
	if len(Q_times)>0: Q_times_str = "_q_" + '_'.join(Q_times.astype("str"))
	else: Q_times_str=""
	if args.A==1: ti_tag ="_TI"
	else: ti_tag=""
	output_wd = os.path.dirname(input_data)
	if output_wd=="": output_wd= self_path
	model_tag=""
	if args.TdD: model_tag+= "_TdD"
	elif args.DivdD: model_tag+= "_DivdD"
	else: model_tag+= "_Dexp"
	if args.TdE: model_tag+= "_TdE"
	elif args.DivdE: model_tag+= "_DivdE"
	elif args.DisdE: model_tag+= "_DisdE"
	elif args.DdE: model_tag+= "_DdE"
	else: model_tag+= "_Eexp"

	if args.lgD: model_tag+= "_lgD"
	if args.lgE: model_tag+= "_lgE"
	if args.linE: model_tag+= "_linE"
	# constraints
	if equal_d is True: model_tag+= "_symd"
	if equal_e is True: model_tag+= "_syme"
	if equal_q is True: model_tag+= "_symq"
	if const_q > 0: model_tag+= "_constq%s" % (const_q)
	if runMCMC == 0: model_tag+= "_ML"
	if len(constraints_covar)>0: model_tag+= "_constr"
	for i in constraints_covar: model_tag+= "_%s" % (i)
	if len(args.symCov)>0: model_tag+= "_symCov"
	for i in args.symCov: model_tag+= "_%s" % (i)
	if argsG is True: model_tag+= "_G" 
	if args.A == 3: model_tag+= "_Mlsbplx"
		
	out_log ="%s/%s_%s%s%s%s%s.log" % (output_wd,name_file,simulation_no,Q_times_str,ti_tag,model_tag,args.out)
	out_rates ="%s/%s_%s%s%s%s%s_marginal_rates.log" % (output_wd,name_file,simulation_no,Q_times_str,ti_tag,model_tag,args.out)
	time_series = np.sort(time_series)[::-1] # the order of the time vector is only used to assign the different Q matrices
	                                         # to the correct time bin. Q_list[0] = root age, Q_list[n] = most recent

if verbose ==1: 
	print time_series
	print obs_area_series
	
#############################################
######            INIT MODEL           ######
#############################################
print "initializing model..."
delta_t= abs(np.diff(time_series))
bin_size = delta_t[0]
possible_areas= list(powerset(np.arange(nareas)))

present_data=obs_area_series[:,-1] # last element

rho_at_present_LIST=[]
r_vec_indexes_LIST=[]
sign_list_LIST=[]


list_taxa_index =[]
for l in range(nTaxa):
	rho_at_present=np.zeros(len(possible_areas))
	try: 
		rho_at_present[possible_areas.index(present_data[l])]=1 # assign prob 1 for observed range at present, 0 for all others
		list_taxa_index.append(l)
	except: rho_at_present=np.zeros(len(possible_areas)) # NaN at present (entire species not preserved)
	rho_at_present_LIST.append(rho_at_present)
	# INIT PARMS
	r_vec_indexes,sign_list=build_list_rho_index_vec(obs_area_series[l],nareas,possible_areas)
	r_vec_indexes_LIST.append(r_vec_indexes)
	sign_list_LIST.append(sign_list)
#####	

dis_rate_vec= np.array([.1,.1]  ) #__ np.zeros(nareas)+.5 # np.random.uniform(0,1,nareas)
ext_rate_vec= np.array([.005,.005]) #__ np.zeros(nareas)+.05 # np.random.uniform(0,1,nareas)
r_vec= np.array([0]+list(np.zeros(nareas)+0.001) +[1])
# where r[1] = prob not obs in A; r[2] = prob not obs in B
# r[0] = 0 (for impossible ranges); r[3] = 1 (for obs ranges)





scal_fac_TI=np.ones(1)
if args.A==1:
	# parameters for TI are currently hard-coded (K=10, alpha=0.3)
	scal_fac_TI=get_temp_TI(args.k,args.a)


#############################################
######       MULTIPLE Q MATRICES       ######
#############################################
"""
Q_list = [Q1,Q2,...]
Q_index = [0,0,0,1,1,1,1,....] # Q index for each time bin
so that at time t: Q = Q_list[t]

d= [[d1,d2]
    [d1,d2]
    ......
           ]
NOTE that Q_list[0] = root age, Q_list[n] = most recent
"""

# INIT PARAMETERS
n_Q_times=len(Q_times)+1
dis_rate_vec=np.random.uniform(0.1,0.2,nareas*1).reshape(1,nareas)
if args.TdD: 
	dis_rate_vec=np.random.uniform(0.05,0.15,nareas*n_Q_times).reshape(n_Q_times,nareas)
	#dis_rate_vec = np.array([0.194,0.032,0.007,0.066,0.156,0.073]).reshape(n_Q_times,nareas)
ext_rate_vec=np.random.uniform(0.01,0.05,nareas*1).reshape(1,nareas)
if args.TdE: 
	ext_rate_vec=np.random.uniform(0.01,0.05,nareas*n_Q_times).reshape(n_Q_times,nareas)
	#ext_rate_vec = np.array([0.222,0.200,0.080,0.225,0.216,0.435]).reshape(n_Q_times,nareas)
if equal_d is True:
	d_temp=dis_rate_vec[:,0]
	dis_rate_vec = array([d_temp,d_temp]).T
if equal_e is True:
	e_temp=ext_rate_vec[:,0]
	ext_rate_vec = array([e_temp,e_temp]).T

r_vec= np.zeros((n_Q_times,nareas+2)) 
r_vec[:,1:3]=0.25
r_vec[:,3]=1
#q_rate_vec = np.array([0.317,1.826,1.987,1.886,2.902,2.648])
#r_vec_temp  = exp(-q_rate_vec*bin_size).reshape(n_Q_times,nareas)
#r_vec[:,1:3]=r_vec_temp





# where r[1] = prob not obs in A; r[2] = prob not obs in B
# r[0] = 0 (for impossible ranges); r[3] = 1 (for obs ranges)
#for i in range(len(time_series)):
ind_shift=[]

for i in Q_times: ind_shift.append(np.argmin(abs(time_series-i)))

ind_shift.append(len(time_series))
ind_shift= np.sort(ind_shift)[::-1]
# Q_index = [0,0,0,1,1,1,1,....] # Q index for each time bin
# Note that Q_index also provides the index for r_vec
if verbose ==1: print ind_shift,time_series
Q_index=np.zeros(len(time_series))
i,count=0,0
for j in ind_shift[::-1]:
	print i, j, count
	Q_index[i:j]=count
	i=j
	count+=1

Q_index =Q_index.astype(int) 
if verbose ==1: print Q_index, shape(dis_rate_vec)

prior_exp_rate = 1.

if args.data_in_area == 1:
	ext_rate_vec[:,1] = 0
	dis_rate_vec[:,0] = 0
	r_vec[:,2] = small_number
elif  args.data_in_area == 2:
	ext_rate_vec[:,0] = 0
	dis_rate_vec[:,1] = 0
	r_vec[:,1] = small_number

print np.shape(ext_rate_vec)

YangGammaQuant = array([1.])
if argsG is True:
        YangGammaQuant = (np.linspace(0,1,pp_gamma_ncat+1)-np.linspace(0,1,pp_gamma_ncat+1)[1]/2)[1:]
alpha = array([10.]) # little sampling heterogeneity 

#############################################
######               MCMC              ######
#############################################

logfile = open(out_log , "w",0) 
head="it\tposterior\tprior\tlikelihood"
for i in range(len(dis_rate_vec)): head+= "\td12_t%s\td21_t%s" % (i,i)
for i in range(len(ext_rate_vec)): head+= "\te1_t%s\te2_t%s" % (i,i)
for i in range(n_Q_times): head+= "\tq1_t%s\tq2_t%s" % (i,i)
if args.lgD: head += "\tk_d12\tk_d21\tx0_d12\tx0_d21"
else: head += "\tcov_d12\tcov_d21"
if args.lgE: head += "\tk_e1\tk_e2\tx0_e1\tx0_e2"
else: head += "\tcov_e1\tcov_e2"
if argsG is True:
        head+= "\talpha"
head+="\thp_rate\tbeta"

head=head.split("\t")
wlog=csv.writer(logfile, delimiter='\t')
wlog.writerow(head)

print "data size:", len(list_taxa_index), nTaxa, len(time_series)

print "starting MCMC..."
if use_seq_lik is True: num_processes=0
if num_processes>0: pool_lik = multiprocessing.Pool(num_processes) # likelihood
start_time=time.time()

update_rate_freq = max(0.1, 1.5/sum(np.size(dis_rate_vec)))
print "Origination time (binned):", OrigTimeIndex, delta_t # update_rate_freq, update_freq
l=1
recursive = np.arange(OrigTimeIndex[l],len(delta_t))[::-1]
print recursive
print shape(r_vec_indexes_LIST[l]),shape(sign_list_LIST[l])
#quit()
covar_par_A =np.zeros(4)
if args.DivdD:
	covar_par_A[0:2] = np.array([nTaxa * 1., nTaxa * 1.])
x0_logistic_A =np.zeros(4)


# EMPIRICAL INIT EANA

#dis_rate_vec=np.array([0.014585816,0.019537033,0.045284283,0.011647877,0.077365681,0.072545101]).reshape(n_Q_times,nareas)
#    q_rate_vec = np.array([1.581652245,1.517955917 ,1.601288529 ,1.846235859 ,2.664323002 ,3.438584756])
#    r_vec_temp  = exp(-q_rate_vec*bin_size).reshape(n_Q_times,nareas)
#    r_vec[:,1:3]=r_vec_temp
#    
#    # start for TempD TempE
#    ext_rate_vec = np.array([0.121 , 0.121]).reshape(1,nareas)
#    dis_rate_vec = np.array([.053 , .053]).reshape(1,nareas)
#    covar_par_A = np.array([-0.0118503199,-0.0118503199,-0.104669685,-0.104669685])
#    
#    ## start for TempD DdE
#    # ./PyRateDES2.py -d /Users/danielesilvestro/Documents/Projects/DES_II/Carnivora/Carnivora_raw_data/neogene/EANAng3/EANAng3_rep1.txt -var /Users/danielesilvestro/Documents/Projects/DES_II/Carnivora/Carnivora_raw_data/neogene/NeogeneTempSmooth.txt -qtimes 5.3 2.6 -DdE -symCov 1 -A 2 -symd
#    #ext_rate_vec = np.array([0.043327715,0.127886299]).reshape(1,nareas)
#    #ext_rate_vec = np.array([0.12,0.12]).reshape(1,nareas)
#    #dis_rate_vec = np.array([.05,.05]).reshape(1,nareas)
#    #covar_par_A = np.array([-0.15,-0.15,0.279991284,0.070409858])
#    covar_par_A = np.array([-0.011 ,-0.011 ,0 , 0])
#    ext_rate_vec=np.array([0.151 , 0.151 , 0.159 , 0.159 , 0.327 , 0.327]).reshape(n_Q_times,nareas)
#
#	d: [ 0.096  0.096] e: [ 0.121  0.121] q: [ 1.603  1.513  1.61   1.785  2.634  3.465]
#	a/k: [-0.022 -0.022  0.143  0.143] x0: [ 0.  0.  0.  0.]
#	d: [ 0.053  0.053] e: [ 0.151  0.151  0.159  0.159  0.327  0.327] q: [ 1.598  1.521  1.595  1.757  2.688  3.455]
#	a/k: [-0.011 -0.011  0.     0.   ] x0: [ 0.  0.  0.  0.]



#-0.009073347	-0.138451368	-0.062747722	-0.059015597
#dis_rate_vec=np.array([0.014585816,0.019537033,0.045284283,0.011647877,0.077365681,0.072545101]).reshape(n_Q_times,nareas)
#q_rate_vec = np.array([0.32033397,1.958145806,1.948256733,1.835880139,2.542124041,2.720067697])
#r_vec_temp  = exp(-q_rate_vec*bin_size).reshape(n_Q_times,nareas)
#r_vec[:,1:3]=r_vec_temp
#
#dis_rate_vec=np.array([.18,.05,.18,0.1,.18,.05]).reshape(n_Q_times,nareas)
#ext_rate_vec = np.array([0.25,0.25]).reshape(1,nareas)
##dis_rate_vec = np.array([.18,.05]).reshape(1,nareas)
#covar_par_A = np.array([0,0,-0.05,-0.05])
#





#############################################
#####    time variable Q (no shifts)    #####
#############################################
#try:
#	time_var_temp = get_binned_continuous_variable(time_series, args.varD)
#	# SHIFT TIME VARIABLE ()
#	time_var = time_var_temp-time_var_temp[len(delta_t)-1]
#	print "Rescaled variable",time_var, time_series
#	if args.varE=="":
#		time_varE=time_var		
#	else:
#		## TEMP FOR EXTINCTION
#		time_var_temp = get_binned_continuous_variable(time_series, args.varE)
#		# SHIFT TIME VARIABLE ()
#		time_varE = time_var_temp-time_var_temp[len(delta_t)-1]
#		
#except:
#	time_var = np.ones(len(time_series)-1)
#	print "Covariate-file not found"
if args.varD != "" and args.varE != "":
	time_var_temp = get_binned_continuous_variable(time_series, args.varD)	
	#time_varD = time_var_temp-time_var_temp[len(delta_t)-1]
	time_varD = time_var_temp - np.mean(time_var_temp)
	print "Rescaled variable dispersal",time_varD, time_series
	time_var_temp = get_binned_continuous_variable(time_series, args.varE)	
	#time_varE = time_var_temp-time_var_temp[len(delta_t)-1]
	time_varE = time_var_temp - np.mean(time_var_temp)
	print "Rescaled variable extinction",time_varE, time_series
elif args.varD != "":
	time_var_temp = get_binned_continuous_variable(time_series, args.varD)	
	#time_varD = time_var_temp-time_var_temp[len(delta_t)-1]
	time_varD = time_var_temp - np.mean(time_var_temp)
	print "Rescaled variable dispersal",time_varD, time_series
	time_varE = np.ones(len(time_series)-1)
elif args.varE != "":
	time_varD = np.ones(len(time_series)-1)
	time_var_temp = get_binned_continuous_variable(time_series, args.varE)	
	#time_varE = time_var_temp-time_var_temp[len(delta_t)-1]
	time_varE = time_var_temp - np.mean(time_var_temp)
	print "Rescaled variable extinction",time_varE, time_series
else:
	time_varD = np.ones(len(time_series)-1)
	time_varE = time_varD
	print "Covariate-file not found"

bound_covar = 25.
range_time_varD = np.max(time_varD) - np.min(time_varD)
range_time_varE = np.max(time_varE) - np.min(time_varE)
bound_covar_d = (1. + small_number) / (range_time_varD + small_number) * bound_covar
bound_covar_e = (1. + small_number) / (range_time_varE + small_number) * bound_covar	
#x0_logistic_A = np.array([np.mean(time_varD), np.mean(time_varD), np.mean(time_varE), np.mean(time_varE)])



ratesfile = open(out_rates , "w",0) 
head="it"
for i in range(len(time_varD)): head+= "\td12_%s" % (i)
for i in range(len(time_varD)): head+= "\td21_%s" % (i)
for i in range(len(time_varD)): head+= "\te1_%s" % (i)
for i in range(len(time_varD)): head+= "\te2_%s" % (i)
head=head.split("\t")
rlog=csv.writer(ratesfile, delimiter='\t')
rlog.writerow(head)


# DIVERSITY TRAJECTORIES
tbl = np.genfromtxt(input_data, dtype=str, delimiter='\t')
tbl_temp=tbl[1:,1:]
data_temp=tbl_temp.astype(float)
# remove empty taxa (absent throughout)
ind_keep = (np.sum(data_temp,axis=1) != 0).nonzero()[0]
data_temp = data_temp[ind_keep]

#print data_temp

# area 1
data_temp1 = 0.+data_temp
data_temp1[data_temp1==2]=0
data_temp1[data_temp1==3]=1
div_traj_1 = np.nansum(data_temp1,axis=0)[0:-1]

# area 2
data_temp2 = 0.+data_temp
data_temp2[data_temp2==1]=0
data_temp2[data_temp2==2]=1
data_temp2[data_temp2==3]=1
div_traj_2 = np.nansum(data_temp2,axis=0)[0:-1]
print "Diversity trajectories", div_traj_1,div_traj_2

#def smooth_trajectory(div_traj):
#	for i in range(len(div_traj)):
#		j=len(a)-i-1
#		print j, a[j]

# Get area and time for the first record of each species
# Could there be area 0? The example data have this but why?
bin_first_occ = np.zeros(nTaxa, dtype = int)
first_area = np.zeros(nTaxa)
first_time = np.zeros(nTaxa)
for i in range(nTaxa):
	bin_first_occ[i] = np.min( np.where( np.in1d(data_temp[i,:], [1., 2., 3.]) ) )
	first_area[i] = data_temp[i, bin_first_occ[i]]
	first_time[i] = time_series[bin_first_occ[i]]
	
#print "First bin", bin_first_occ	
#print "First area", first_area
#print "First time", first_time

Q_index_first_occ = Q_index[bin_first_occ]
Q_index_first_occ = Q_index_first_occ.astype(int)
len_time_series = len(time_series)

# Max diversity of area AB
offset_dis_div1 = 0.
offset_dis_div2 = 0.
if args.DivdD:
	data_temp3 = 0.+data_temp
	data_temp3[data_temp3==1]=0
	data_temp3[data_temp3==2]=0
	data_temp3[data_temp3==3]=1
	div_traj_3 = np.nansum(data_temp3,axis=0)[0:-1]
	max_div_traj_3 = np.max(div_traj_3)
	offset_dis_div1 = max_div_traj_3
	offset_dis_div2 = max_div_traj_3
	
# Median of diversity	
offset_ext_div1 = 0.
offset_ext_div2 = 0.
if args.DivdE:
	offset_ext_div1 = np.median(div_traj_1)
	offset_ext_div2 = np.median(div_traj_2)
	if 1 in args.symCov or 3 in args.symCov:
		offset_ext_div = np.median( np.concatenate((div_traj_1, div_traj_2)) )
		offset_ext_div1 = offset_ext_div
		offset_ext_div2 = offset_ext_div

argsDivdD = args.DivdD
argsDivdE = args.DivdE

# Use an ode solver to approximate the diversity trajectories
def div_dt(div, t, d12, d21, mu1, mu2, k_d1, k_d2, k_e1, k_e2):
	div1 = div[0]
	div2 = div[1]
	div3 = div[2]
	lim_d2 = max(0, 1 - (div1 + div3)/k_d1) # Limit dispersal into area 2
  	lim_d1 = max(0, 1 - (div2 + div3)/k_d2) # Limit dispersal into area 1
  	lim_e1 = max(1e-05, 1 - (div1 + div3)/k_e1) # Increases extinction in area 2
  	lim_e2 = max(1e-05, 1 - (div2 + div3)/k_e2) # Increases extinction in area 1
  	dS = np.zeros(3)
	dS[0] = -mu1/lim_e1 * div1 + mu2 * div2 - d12 * div1 * lim_d1 
  	dS[1] = -mu2/lim_e2 * div2 + mu1 * div3 - d21 * div2 * lim_d2 
  	dS[2] = -(mu1/lim_e1 + mu2/lim_e2) * div3 + d21 * div2 * lim_d2 + d12 * div1 * lim_d1
	return dS	


def approx_div_traj(nTaxa, dis_rate_vec, ext_rate_vec, 
                    argsDivdD, argsDivdE, argsG,
                    r_vec, alpha, YangGammaQuant, pp_gamma_ncat, bin_size, Q_index, Q_index_first_occ, 
                    covar_par, offset_dis_div1, offset_dis_div2, offset_ext_div1, offset_ext_div2,
		    time_series, len_time_series, bin_first_occ, first_area):	
	if argsG:
		YangGamma = get_gamma_rates(alpha, YangGammaQuant, pp_gamma_ncat)
		sa = np.zeros((pp_gamma_ncat, nTaxa))
		sb = np.zeros((pp_gamma_ncat, nTaxa))
		for i in range(pp_gamma_ncat):
			sa[i,:] = 1. - exp(-bin_size * YangGamma[i] * -log(r_vec[Q_index_first_occ, 1])/bin_size)  
			sb[i,:] = 1. - exp(-bin_size * YangGamma[i] * -log(r_vec[Q_index_first_occ, 2])/bin_size)
		sa = sa * weight_per_taxon.T
		sb = sb * weight_per_taxon.T
		sa = np.nansum(sa, axis = 0)
		sb = np.nansum(sb, axis = 0)
	else:	
		sa = 1. - r_vec[Q_index_first_occ, 1]	
		sb = 1. - r_vec[Q_index_first_occ, 2]
	sa[first_area == 2.] = 0. # Gives false absence in area 1 (observed 2 cannot be in area 1) 
	sb[first_area == 1.] = 0. # Gives false absence in area 2 (observed 1 cannot be in area 2)
	# Add artificial bin before start of the time series
	padded_time = time_series[0] + time_series[0] - time_series[1]
	time_series_pad = np.concatenate((padded_time, time_series[0:-1]), axis = None)
	div_1 = np.zeros(len_time_series)
	div_2 = np.zeros(len_time_series)
	div_3 = np.zeros(len_time_series)	
	for i in range(len_time_series - 1):		
		Q_index_i = Q_index[i]
		
		if argsDivdD:
			dis_rate_vec_i = dis_rate_vec[0, ]
			k_d1 = covar_par[0]
			k_d2 = covar_par[1]
			dis_rate_vec_i = dis_rate_vec_i / (1.- [offset_dis_div2, offset_dis_div1]/covar_par[0:2])			
		else:
			dis_rate_vec_i = dis_rate_vec[Q_index_i, ]
			k_d1 = np.inf
			k_d2 = np.inf
			
		if argsDivdE:
			ext_rate_vec_i = ext_rate_vec[0, ] 
			k_e1 = covar_par[2]
			k_e2 = covar_par[3]	
			ext_rate_vec_i = ext_rate_vec_i * (1 - ([offset_ext_div1, offset_ext_div2]/covar_par[2:4]))			
		else:
			ext_rate_vec_i = ext_rate_vec[Q_index_i, ]
			k_e1 = np.inf
			k_e2 = np.inf	
		
		d12 = dis_rate_vec_i[0] 
		d21 = dis_rate_vec_i[1]
		mu1 = ext_rate_vec_i[0]
		mu2 = ext_rate_vec_i[1]
				
		occ_i = bin_first_occ == i # Only taxa occuring at that time for the first time
		sa_i = sa[occ_i]
		sa_i = sa_i[sa_i != 0.] # Only taxa in area 1
		sb_i = sb[occ_i]
		sb_i = sb_i[sb_i != 0.]		
		len_sa_i = len(sa_i) # Number of new taxa observed in area 1
		len_sb_i = len(sb_i) 
		sum_sa_i = sum(sa_i) # Summed probability of false absences in area 1
		sum_sb_i = sum(sb_i) 
		new_1 = len_sa_i - sum_sa_i 
		new_2 = len_sb_i - sum_sb_i
		new_3 = sum_sa_i + sum_sb_i
		
		dt = [0., time_series_pad[i] - time_series_pad[i + 1] ]
		div_t = np.zeros(3)
		div_t[0] = div_1[i] + new_1
		div_t[1] = div_2[i] + new_2
		div_t[2] = div_3[i] + new_3

		div_int = odeint(div_dt, div_t, dt, args = (d12, d21, mu1, mu2, k_d1, k_d2, k_e1, k_e2))
		div_1[i + 1] = div_int[1,0]
	        div_2[i + 1] = div_int[1,1]
		div_3[i + 1] = div_int[1,2]
		
	div_13 = div_1 + div_3
	div_23 = div_2 + div_3
	
	return div_13[1:], div_23[1:]	
	
	
def get_num_dispersals(dis_rate_vec,r_vec):
	Pr1 = 1- r_vec[Q_index[0:-1],1] # remove last value (time zero)
	Pr2 = 1- r_vec[Q_index[0:-1],2] 
	# get dispersal rates through time
	#d12 = dis_rate_vec[0][0] *exp(covar_par[0]*time_var)
	#d21 = dis_rate_vec[0][1] *exp(covar_par[1]*time_var)
	dr = dis_rate_vec
	d12 = dr[:,0]
	d21 = dr[:,1]
	
	numD12 = (div_traj_1/Pr1)*d12
	numD21 = (div_traj_2/Pr2)*d21
	
	return numD12,numD21

def get_est_div_traj(r_vec):
	Pr1 = 1- r_vec[Q_index[0:-1],1] # remove last value (time zero)
	Pr2 = 1- r_vec[Q_index[0:-1],2] 
	
	numD1 = (div_traj_1/Pr1)
	numD2 = (div_traj_2/Pr2)
	numD1res = rescale_vec_to_range(numD1, r=10., m=0)
	numD2res = rescale_vec_to_range(numD2, r=10., m=0)
	
	return numD1res,numD2res
	

# initialize weight per gamma cat per species to estimate diversity trajectory with heterogeneous preservation
weight_per_taxon = np.ones((nTaxa, pp_gamma_ncat)) / pp_gamma_ncat


#print Q_index
# print dis_rate_vec
# print time_series, time_varD

#get_num_dispersals(d12,d21,r_vec)

###################################################################################
# Avoid code redundancy in mcmc and maximum likelihood
def lik_DES(Q_list, w_list, vl_list, vl_inv_list, delta_t, r_vec, rho_at_present_LIST, r_vec_indexes_LIST, sign_list_LIST, OrigTimeIndex,Q_index, alpha, YangGammaQuant, pp_gamma_ncat, num_processes, use_Pade_approx):
	# weight per gamma cat per species: multiply 
	weight_per_taxon = np.zeros((nTaxa, pp_gamma_ncat)) 
	if num_processes==0:
		if use_Pade_approx==0:
			#t1= time.time()
			lik=0
			#print "Q_list", Q_list		
			if argsG is False:
				for l in list_taxa_index:
					Q_index_temp = np.array(range(0,len(w_list)))				
					l_temp = calc_likelihood_mQ_eigen([delta_t,r_vec,w_list,vl_list,vl_inv_list,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index,Q_index_temp])
					#print l,  l_temp
					lik +=l_temp 
			else:				
				for l in list_taxa_index:
					Q_index_temp = np.array(range(0,len(w_list)))
					YangGamma = get_gamma_rates(alpha, YangGammaQuant, pp_gamma_ncat)
	                	        lik_vec = np.zeros(pp_gamma_ncat)
                                        for i in range(pp_gamma_ncat): 
                                        	r_vec_Gamma = exp(-bin_size * YangGamma[i] * -log(r_vec)/bin_size) # convert to probability scale
                                                r_vec_Gamma[:,0] = 0
                                                r_vec_Gamma[:,3] = 1
                                                if args.data_in_area == 1:	                                               
	                                                r_vec_Gamma[:,2] = small_number
                                                elif  args.data_in_area == 2:
	                                                r_vec_Gamma[:,1] = small_number
	                                        lik_vec[i] =  calc_likelihood_mQ_eigen([delta_t,r_vec_Gamma,w_list,vl_list,vl_inv_list,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index,Q_index_temp])
                                        lik_vec_max = np.max(lik_vec)
                                        lik2 = lik_vec - lik_vec_max
                                        lik += log(sum(exp(lik2))/pp_gamma_ncat) + lik_vec_max
                                        weight_per_taxon[l,:] = lik_vec / sum(lik_vec) 
			#print "elapsed time:", time.time()-t1
		else:
			#t1= time.time()
			#lik=0
			#for l in list_taxa_index:
			#	lik += calc_likelihood_mQ([delta_t,r_vec,Q_list_old,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index,Q_index])
			#print "lik1:", lik
			lik=0
			if argsG is False:
				for l in list_taxa_index:
					Q_index_temp = np.array(range(0,len(Q_list)))
					lik += calc_likelihood_mQ([delta_t,r_vec,Q_list,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index,Q_index_temp])
			else:
				for l in list_taxa_index:
					Q_index_temp = np.array(range(0,len(Q_list)))
		                        YangGamma = get_gamma_rates(alpha, YangGammaQuant, pp_gamma_ncat)
	                	        lik_vec = np.zeros(pp_gamma_ncat)
	                	        for i in range(pp_gamma_ncat): 
                                                r_vec_Gamma = exp(-bin_size * YangGamma[i] * -log(r_vec)/bin_size) # convert to probability scale
                                                r_vec_Gamma[:,0] = 0
                                                r_vec_Gamma[:,3] = 1
                                                if args.data_in_area == 1:	                                               
	                                                r_vec_Gamma[:,2] = small_number
                                                elif  args.data_in_area == 2:
	                                                r_vec_Gamma[:,1] = small_number
                                                lik_vec[i] = calc_likelihood_mQ([delta_t,r_vec_Gamma,Q_list,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index,Q_index_temp])
                                        lik_vec_max = np.max(lik_vec)
                                        lik2 = lik_vec - lik_vec_max
                                        lik += log(sum(exp(lik2))/pp_gamma_ncat) + lik_vec_max
			#print "lik2", lik
		
			
	else: # multi=processing
		sys.exit("Multi-threading not available")
		if use_Pade_approx==0:
			#t1= time.time()
			w_list,vl_list,vl_inv_list = get_eigen_list(Q_list)
			Q_index_temp = np.array(range(0,len(w_list)))
			if argsG is False:				
				args_mt_lik = [ [delta_t,r_vec,w_list,vl_list,vl_inv_list,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index,Q_index_temp] for l in list_taxa_index ]
				lik= sum(np.array(pool_lik.map(calc_likelihood_mQ_eigen, args_mt_lik)))
			else:
				YangGamma = get_gamma_rates(alpha, YangGammaQuant, pp_gamma_ncat)
                                liktmp = np.zeros((pp_gamma_ncat, nTaxa)) # row: ncat column: species
                                for i in range(pp_gamma_ncat): 
                                        r_vec_Gamma = exp(-bin_size * YangGamma[i] * -log(r_vec)/bin_size) # convert to probability scale
                                        r_vec_Gamma[:,0] = 0
                                        r_vec_Gamma[:,3] = 1
                                        if args.data_in_area == 1:	                                               
	                                        r_vec_Gamma[:,2] = small_number
                                        elif  args.data_in_area == 2:
	                                        r_vec_Gamma[:,1] = small_number
                                        args_mt_lik = [ [delta_t,r_vec_Gamma,w_list,vl_list,vl_inv_list,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index,Q_index_temp] for l in list_taxa_index ]
                                        liktmp[i,:] = np.array(pool_lik.map(calc_likelihood_mQ_eigen, args_mt_lik))
                                liktmpmax = np.amax(liktmp, axis = 0)
                                liktmp2 = liktmp - liktmpmax
                                lik = sum(log(sum( exp(liktmp2), axis = 0 )/pp_gamma_ncat)+liktmpmax)
			#print "l3",lik
			#print "elapsed time:", time.time()-t1
		else:
			#t1= time.time()
			if argsG is False:
				args_mt_lik = [ [delta_t,r_vec,Q_list,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index] for l in list_taxa_index ]
				lik= sum(np.array(pool_lik.map(calc_likelihood_mQ, args_mt_lik)))
			else:
				YangGamma = get_gamma_rates(alpha, YangGammaQuant, pp_gamma_ncat)
                                liktmp = np.zeros((pp_gamma_ncat, nTaxa)) # row: ncat column: species
                                for i in range(pp_gamma_ncat): 
                                        r_vec_Gamma = exp(-bin_size * YangGamma[i] * -log(r_vec)/bin_size) # convert to probability scale
                                        r_vec_Gamma[:,0] = 0
                                        r_vec_Gamma[:,3] = 1
                                        if args.data_in_area == 1:	                                               
	                                        r_vec_Gamma[:,2] = small_number
                                        elif  args.data_in_area == 2:
	                                        r_vec_Gamma[:,1] = small_number
					args_mt_lik = [ [delta_t,r_vec_Gamma,Q_list,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index] for l in list_taxa_index ]
					liktmp[i,:] = np.array(pool_lik.map(calc_likelihood_mQ, args_mt_lik))
         			liktmpmax = np.amax(liktmp, axis = 0)
                                liktmp2 = liktmp - liktmpmax
                                lik = sum(log(sum( exp(liktmp2), axis = 0 )/pp_gamma_ncat)+liktmpmax)
			#print "l4",lik
			#print "elapsed time:", time.time()-t1
	return lik, weight_per_taxon


# Likelihood for a set of parameters 
# Uses elements of the global environment but there is no other way?!     
def lik_opt(x, grad):
	covar_par = np.zeros(4) + 0.	
	x0_logistic = np.zeros(4) + 0.
	# Sampling	
	r_vec = np.zeros((n_Q_times,nareas+2)) 
	r_vec[:,3]=1
	if args.data_in_area == 1:
		r_vec[:,1] = x[opt_ind_r_vec]
		r_vec[:,2] = small_number
	elif  args.data_in_area == 2:
		r_vec[:,1] = small_number
		r_vec[:,2] = x[opt_ind_r_vec]	
	else:
		r_vec[:,1:3] = np.array(x[opt_ind_r_vec]).reshape(n_Q_times,nareas)
	# Dispersal
	dis_vec = np.zeros((n_Q_times,nareas))
	if args.data_in_area == 1:
		dis_vec[:,1] = x[opt_ind_dis]
	elif  args.data_in_area == 2:
		dis_vec[:,0] = x[opt_ind_dis]
	else:
		dis_vec = np.array(x[opt_ind_dis]).reshape(n_Q_times_dis,nareas)
		
	do_approx_div_traj = 0
	if args.TdD: # time dependent D
		dis_vec = dis_vec[Q_index,:] 
		dis_vec = dis_vec[0:-1] 
		transf_d = 0
		time_var_d1,time_var_d2=time_varD,time_varD	
	elif args.DivdD: # Diversity dependent D
		transf_d = 4 # 1
		#time_var_d1,time_var_d2 = get_est_div_traj(r_vec)
		do_approx_div_traj = 1		
	else: # temp dependent D	
		transf_d=1
		time_var_d1,time_var_d2=time_varD,time_varD
	if args.lgD: transf_d = 2
	
	# Extinction
	ext_vec = np.zeros((n_Q_times,nareas))
	if args.data_in_area == 1:
		ext_vec[:,0] = x[opt_ind_ext]
	elif  args.data_in_area == 2:
		ext_vec[:,1] = x[opt_ind_ext]	
	else:
		ext_vec = np.array(x[opt_ind_ext]).reshape(n_Q_times_ext,nareas)	
	if args.TdE:			
		ext_vec = ext_vec[Q_index,:] 
		ext_vec = ext_vec[0:-1] 
		transf_e = 0
		time_var_e1,time_var_e2=time_varD,time_varD		
	elif args.DivdE: # Diversity dep Extinction
		# NOTE THAT extinction in 1 depends diversity in 1
		transf_e = 4
 		#time_var_e1,time_var_e2 = get_est_div_traj(r_vec)
 		do_approx_div_traj = 1
	else: # Temp dependent Extinction
		transf_e = 1
		time_var_e1,time_var_e2=time_varE,time_varE
	if args.lgE: transf_e = 2
		                                          
	alpha = 10.
	if argsG:
		alpha = x[alpha_ind]	
	
	if transf_d > 0:
		covar_par[[0,1]] = x[opt_ind_covar_dis]
		if args.data_in_area != 0:
		        covar_par[[args.data_in_area - 1]] = x[opt_ind_covar_dis]
		if transf_d == 2:
			x0_logistic[[0,1]] = x[opt_ind_x0_log_dis]
			if args.data_in_area != 0:
		        	x0_logistic[[args.data_in_area - 1]] = x[opt_ind_x0_log_dis]
	if transf_e > 0:
		covar_par[[2,3]] = x[opt_ind_covar_ext]
		if args.data_in_area != 0:
		        covar_par[[args.data_in_area + 2]] = x[opt_ind_covar_ext]
		if transf_e == 2:
			x0_logistic[[2,3]] = x[opt_ind_x0_log_ext]
			if args.data_in_area != 0:
		        	x0_logistic[[args.data_in_area + 2]] = x[opt_ind_x0_log_ext]
		        	
	if do_approx_div_traj == 1:
		approx_d1,approx_d2 = approx_div_traj(nTaxa, dis_vec, ext_vec, 
		                                      argsDivdD, argsDivdE, argsG,
                    				      r_vec, alpha, YangGammaQuant, pp_gamma_ncat, bin_size, Q_index, Q_index_first_occ, 
                    				      covar_par, offset_dis_div1, offset_dis_div2, offset_ext_div1, offset_ext_div2,
		                                      time_series, len_time_series, bin_first_occ, first_area)		    
		if args.DivdD:
			time_var_d2 = approx_d1 # Limits dispersal into 1
			time_var_d1 = approx_d2 # Limits dispersal into 2
		if args.DivdE:
			time_var_e1 = approx_d1
			time_var_e2 = approx_d2
		#print "Approx d1", time_var_d1
		#print "Approx d2", time_var_d2		        	
			
        Q_list, marginal_rates_temp= make_Q_Covar4VDdE(dis_vec,ext_vec,time_var_d1,time_var_d2,time_var_e1,time_var_e2,covar_par,x0_logistic,transf_d,transf_e, offset_dis_div1, offset_dis_div2, offset_ext_div1, offset_ext_div2)
        if use_Pade_approx==0:
		w_list,vl_list,vl_inv_list = get_eigen_list(Q_list)
        lik, weight_per_taxon = lik_DES(Q_list, w_list, vl_list, vl_inv_list, delta_t, r_vec, rho_at_present_LIST, r_vec_indexes_LIST, sign_list_LIST,OrigTimeIndex,Q_index, alpha, YangGammaQuant, pp_gamma_ncat, num_processes, use_Pade_approx)
	print "lik", lik, x
        return lik

# Maximize likelihood
if args.A == 3:
	n_generations   = 1
	sampling_freq   = 1
	print_freq      = 1
	# Define which initial value for the optimizer corresponds to which parameter
	# opt_ind_xxx: indices for parameters to optimize
	# x0: initial values
	# xxx_bounds: bounds for optimization
	
	if args.TdD is True:
	        n_Q_times_dis = n_Q_times	        
	else:
		n_Q_times_dis = 1
	opt_ind_dis = np.arange(0, n_Q_times_dis*nareas) 
	if equal_d is True:
		opt_ind_dis = np.repeat(np.arange(0, n_Q_times_dis), 2)
	if args.data_in_area != 0:
		opt_ind_dis = np.arange(0, n_Q_times_dis)
	
	if args.TdE is True:
	        n_Q_times_ext = n_Q_times
	else:
		n_Q_times_ext = 1		
	opt_ind_ext = np.max(opt_ind_dis) + 1 + np.arange(0, n_Q_times_ext*nareas) 
	if equal_e is True:
		opt_ind_ext = np.max(opt_ind_dis) + 1 + np.repeat(np.arange(0, n_Q_times_ext), 2)
	if args.data_in_area != 0:
		opt_ind_ext = np.max(opt_ind_dis) + 1 + np.arange(0, n_Q_times_ext)
			
	opt_ind_r_vec = np.max(opt_ind_ext) + 1 + np.arange(0, n_Q_times*nareas)
	if equal_q is True:
		opt_ind_r_vec = np.max(opt_ind_ext) + 1 + np.repeat(np.arange(0, n_Q_times), 2) 
	if const_q ==1: # Needs bin1 = 01 bin2 = 02 bin3 = 03 bin4 = 04 bin5 = 05
		opt_ind_r_vec = np.max(opt_ind_ext) + 1 + np.array([np.zeros(nareas, dtype = int), np.arange(1,nareas+1)]).T.flatten() 
	if const_q ==2: # Needs bin1 = 01 bin2 = 21 bin3 = 31 bin4 = 41 bin5 = 51
		opt_ind_r_vec = np.array([np.arange(1,nareas+1), np.ones(nareas, dtype = int)]).T.flatten()
		opt_ind_r_vec[0] = 0
		opt_ind_r_vec = np.max(opt_ind_ext) + 1 + opt_ind_r_vec 
	if args.data_in_area != 0:
		if const_q ==1 or const_q ==2:
			opt_ind_r_vec = np.max(opt_ind_ext) + 1	+ np.zeros(n_Q_times, dtype = int)
		else: 
			opt_ind_r_vec = np.max(opt_ind_ext) + 1 + np.arange(0, n_Q_times)
	
	
	x0 = np.random.uniform(0.01,0.1, 1 + np.max(opt_ind_r_vec)) # Initial values
	lower_bounds = [small_number]*len(x0)
	upper_bounds = [100]*len(x0)
	upper_bounds[-len(opt_ind_r_vec):] = [1 - small_number] * len(opt_ind_r_vec)
	
	# Preservation heterogeneity
	ind_counter = np.max(opt_ind_r_vec) + 1 		
	if argsG:
		alpha_ind = ind_counter
		ind_counter += 1
		x0 = np.concatenate((x0, 10.), axis = None) 
		lower_bounds = lower_bounds + [small_number]
		upper_bounds = upper_bounds + [100]
		
	# Covariates
	if args.TdD is False or args.DivdD:
		opt_ind_covar_dis = np.array([ind_counter, ind_counter + 1])
		ind_counter += 2		
		if args.DivdD:
			x0 = np.concatenate((x0, nTaxa + 0., nTaxa + 0.), axis = None)
			lower_bounds = lower_bounds + [np.max(div_traj_2)] + [np.max(div_traj_1)]
			upper_bounds = upper_bounds + [np.inf] + [np.inf]
		else:
			x0 = np.concatenate((x0, 0., 0.), axis = None)
			lower_bounds = lower_bounds + [-bound_covar_d] + [-bound_covar_d]
			upper_bounds = upper_bounds + [bound_covar_d] + [bound_covar_d]
		if 1 in args.symCov or args.data_in_area != 0:
	                opt_ind_covar_dis = opt_ind_covar_dis[0:-1]
	                ind_counter = ind_counter - 1
	                x0 = x0[0:-1]
	                lower_bounds = lower_bounds[0:-1]
	                upper_bounds = upper_bounds[0:-1]	        
		if args.lgD:
			opt_ind_x0_log_dis = np.array([ind_counter, ind_counter + 1])
			ind_counter += 2
			x0 = np.concatenate((x0, np.mean(time_varD), np.mean(time_varD)), axis = None)
			lower_bounds = lower_bounds + [np.min(time_varD).tolist()] + [np.min(time_varD).tolist()]
			upper_bounds = upper_bounds + [np.max(time_varD).tolist()] + [np.max(time_varD).tolist()]	       
	                if 1 in args.symCov or args.data_in_area != 0:
	                        opt_ind_x0_log_dis = opt_ind_x0_log_dis[0:-1]
	                        ind_counter = ind_counter - 1
	                        x0 = x0[0:-1]
	                        lower_bounds = lower_bounds[0:-1]
	                        upper_bounds = upper_bounds[0:-1]
	                
	if args.TdE is False or args.DivdE:
		opt_ind_covar_ext = np.array([ind_counter, ind_counter + 1])
		ind_counter += 2 
		if args.DivdE:
			x0 = np.concatenate((x0, nTaxa + 0., nTaxa + 0.), axis = None)
			lower_bounds = lower_bounds + [np.max(div_traj_1)] + [np.max(div_traj_2)]
			upper_bounds = upper_bounds + [np.inf] + [np.inf]
		else:
			x0 = np.concatenate((x0, 0., 0.), axis = None)
			lower_bounds = lower_bounds + [-bound_covar_e] + [-bound_covar_e]
			upper_bounds = upper_bounds + [bound_covar_e] + [bound_covar_e]
		if 3 in args.symCov or args.data_in_area != 0:
	                opt_ind_covar_ext = opt_ind_covar_ext[0:-1]
	                ind_counter = ind_counter - 1
	                x0 = x0[0:-1]
	                lower_bounds = lower_bounds[0:-1]
	                upper_bounds = upper_bounds[0:-1]
		if args.lgE:
			opt_ind_x0_log_ext = np.array([ind_counter, ind_counter + 1])
			ind_counter += 2
			x0 = np.concatenate((x0, np.mean(time_varE), np.mean(time_varE)), axis = None)
			lower_bounds = lower_bounds + [np.min(time_varE).tolist()] + [np.min(time_varE).tolist()]
			upper_bounds = upper_bounds + [np.max(time_varE).tolist()] + [np.max(time_varE).tolist()]		
	                if 3 in args.symCov or args.data_in_area != 0:
	                        opt_ind_x0_log_ext = opt_ind_x0_log_ext[0:-1]
	                        ind_counter = ind_counter - 1
	                        x0 = x0[0:-1]
	                        lower_bounds = lower_bounds[0:-1]
	                        upper_bounds = upper_bounds[0:-1]
	
        # Maximize likelihood
	opt = nlopt.opt(nlopt.LN_SBPLX, len(x0))
	opt.set_lower_bounds(lower_bounds) 
	opt.set_upper_bounds(upper_bounds) 
	opt.set_max_objective(lik_opt)	
	opt.set_xtol_rel(1e-3)
	#opt.set_ftol_abs(1e-2)
	#opt.set_maxeval(100) # Only for checking quickly the result
	x = opt.optimize(x0) 
	minf = opt.last_optimum_value()
	
	# Format output
	dis_rate_vec = np.zeros((n_Q_times_dis,nareas))
	if args.data_in_area == 1:
		dis_rate_vec[:,1] = x[opt_ind_dis]
	elif  args.data_in_area == 2:
		dis_rate_vec[:,0] = x[opt_ind_dis]
	else:
		dis_rate_vec = np.array(x[opt_ind_dis]).reshape(n_Q_times_dis,nareas)
		
	ext_rate_vec = np.zeros((n_Q_times_ext,nareas))
	if args.data_in_area == 1:
		ext_rate_vec[:,0] = x[opt_ind_ext]
	elif  args.data_in_area == 2:
		ext_rate_vec[:,1] = x[opt_ind_ext]	
	else:
		ext_rate_vec = np.array(x[opt_ind_ext]).reshape(n_Q_times_ext,nareas)
		
	r_vec = np.zeros((n_Q_times,nareas+2)) 
	r_vec[:,3]=1
	if args.data_in_area == 1:
		r_vec[:,1] = x[opt_ind_r_vec]
		r_vec[:,2] = small_number
	elif  args.data_in_area == 2:
		r_vec[:,1] = small_number
		r_vec[:,2] = x[opt_ind_r_vec]	
	else:
		r_vec[:,1:3] = np.array(x[opt_ind_r_vec]).reshape(n_Q_times,nareas)
	if argsG: 
		alpha = x[alpha_ind]
		alpha = alpha.flatten()
	covar_par_A = np.zeros(4) + 0.
	x0_logistic_A = np.zeros(4) + 0.
	if args.TdD is False or args.DivdD:
		covar_par_A[[0,1]] = x[opt_ind_covar_dis]
		if args.lgD:
			x0_logistic_A[[0,1]] = x[opt_ind_x0_log_dis]
	if args.TdE is False or args.DivdE:
		covar_par_A[[2,3]] = x[opt_ind_covar_ext]
		if args.lgE:
			x0_logistic_A[[2,3]] = x[opt_ind_x0_log_ext]	
	log_to_file = 1

#############################################	
do_approx_div_traj = 0
ml_it=0
scal_fac_ind=0
for it in range(n_generations * len(scal_fac_TI)):		
	if (it+1) % (n_generations+1) ==0: 
		print it, n_generations  
		scal_fac_ind+=1
	if it ==0: 
		dis_rate_vec_A= dis_rate_vec
		ext_rate_vec_A= ext_rate_vec
		r_vec_A= r_vec 
		# fixed starting values 
		# dis_rate_vec_A= np.array([[0.032924117],[0.045818755]]).T #dis_rate_vec
		# ext_rate_vec_A= np.array([[0.446553889],[0.199597008]]).T #ext_rate_vec
		# covar_par_A=np.array([-0.076944388,-0.100211345,-0.161531353,-0.059495477])
		# r_vec_A=     exp(-np.array([0,1.548902792,1.477082486,1,0,1.767267039,1.981165598,1,0,2.726853331,3.048116889,1])*.5).reshape(3,4) # r_vec  
		likA=-inf
		priorA=-inf
		alphaA = alpha
	
	dis_rate_vec = dis_rate_vec_A + 0.
	ext_rate_vec = ext_rate_vec_A + 0.
	covar_par =    covar_par_A + 0.
	r_vec=         r_vec_A + 0.
	x0_logistic=   x0_logistic_A + 0.	
	hasting = 0
	gibbs_sample = 0
	if it>0: 
		if runMCMC == 1:
			r= np.random.random(3)
		elif it % 10==0:
			r= np.random.random(3)
	else: r = np.ones(3)+1
	
	if it<100: r[1]=1
	
	if r[0] < update_freq[0]: # DISPERSAL UPDATES
		if args.TdD is False and r[1] < .5: # update covar
			if args.lgD and r[2] < .5: # update logistic mid point
				x0_logistic=update_parameter_uni_2d_freq(x0_logistic_A,d=0.1*scale_proposal,f=0.5,m=-3,M=3)
			else:	
				covar_par=update_parameter_uni_2d_freq(covar_par_A,d=0.1*scale_proposal,f=0.5,m=-3,M=3)
		else: # update dispersal rates
			if equal_d is True:
				d_temp,hasting = update_multiplier_proposal_freq(dis_rate_vec_A[:,0],d=1+.1*scale_proposal,f=update_rate_freq)
				dis_rate_vec = array([d_temp,d_temp]).T
			else:
				dis_rate_vec,hasting=update_multiplier_proposal_freq(dis_rate_vec_A,d=1+.1*scale_proposal,f=update_rate_freq)
			
	elif r[0] < update_freq[1]: # EXTINCTION RATES
		if args.TdE is False and r[1] < .5: 
			if args.lgE and r[2] < .5: # update logistic mid point
				x0_logistic=update_parameter_uni_2d_freq(x0_logistic_A,d=0.1*scale_proposal,f=0.5,m=-3,M=3)
			else:	
				covar_par=update_parameter_uni_2d_freq(covar_par_A,d=0.1*scale_proposal,f=0.5,m=-3,M=3)				
			
		else:
			if equal_e is True:
				e_temp,hasting = update_multiplier_proposal_freq(ext_rate_vec_A[:,0],d=1+.1*scale_proposal,f=update_rate_freq)
				ext_rate_vec = array([e_temp,e_temp]).T
			else:
				ext_rate_vec,hasting=update_multiplier_proposal_freq(ext_rate_vec_A,d=1+.1*scale_proposal,f=update_rate_freq)
	
	elif r[0] <=update_freq[2]: # SAMPLING RATES
		r_vec=update_parameter_uni_2d_freq(r_vec_A,d=0.1*scale_proposal,f=update_rate_freq)
		if argsG is True:
			alpha,hasting=update_multiplier_proposal(alphaA,d=1.1)
		r_vec[:,0]=0
		#--> CONSTANT Q IN AREA 2
                if const_q ==1: r_vec[:,1] = r_vec[1,1]
		#--> CONSTANT Q IN AREA 2
                if const_q ==2: r_vec[:,2] = r_vec[1,2]
		#--> SYMMETRIC SAMPLING
		if equal_q is True: r_vec[:,2] = r_vec[:,1]		
		r_vec[:,3]=1
		
		# CHECK THIS: CHANGE TO VALUE CLOSE TO 1? i.e. for 'ghost' area 
		if args.data_in_area == 1: r_vec[:,2] = small_number 
		elif  args.data_in_area == 2: r_vec[:,1] = small_number
	elif it>0:
		gibbs_sample = 1
		prior_exp_rate = gibbs_sampler_hp(np.concatenate((dis_rate_vec,ext_rate_vec)),hp_alpha,hp_beta)
	
	# enforce constraints if any
	if len(constraints_covar)>0:
		covar_par[constraints_covar] = 0
		x0_logistic[constraints_covar] = 0
		dis_rate_vec[:,constraints_covar[constraints_covar<2]] = dis_rate_vec[0,constraints_covar[constraints_covar<2]]
		ext_rate_vec[:,constraints_covar[constraints_covar>=2]-2] = ext_rate_vec[0,constraints_covar[constraints_covar>=2]-2]
	if len(args.symCov)>0:
		args_symCov = np.array(args.symCov)
		covar_par[args_symCov] = covar_par[args_symCov-1] 
		x0_logistic[args_symCov] = x0_logistic[args_symCov-1] 
			
	#dis_rate_vec[1:3,0] = dis_rate_vec[2,0]
	
	## CHANGE HERE TO FIRST OPTIMIZE DISPERSAL AND THEN EXTINCTION
	if args.DdE and it < 100:
		covar_par[2:4]=0
	
	#x0_logistic = np.array([0,0,5,5])
	### GET LIST OF Q MATRICES ###
	if args.TdD: # time dependent D
		covar_par[0:2]=0
		x0_logistic[0:2]=0
		dis_vec = dis_rate_vec[Q_index,:]
		dis_vec = dis_vec[0:-1]
		transf_d=0
		time_var_d1,time_var_d2=time_varD,time_varD
	elif args.DivdD: # Diversity dependent D
		transf_d=4
		dis_vec = dis_rate_vec		
		time_var_d1,time_var_d2 = get_est_div_traj(r_vec)
		do_approx_div_traj = 1
	else: # temp dependent D	
		transf_d=1
		dis_vec = dis_rate_vec
		time_var_d1,time_var_d2=time_varD,time_varD

	if args.lgD: transf_d = 2
	
	if args.data_in_area == 1: 
		covar_par[[0,3]] = 0
		x0_logistic[[0,3]] = 0
	elif  args.data_in_area == 2: 
		covar_par[[1,2]] = 0
		x0_logistic[[1,2]] = 0
	

	marginal_dispersal_rate_temp = get_dispersal_rate_through_time(dis_vec,time_var_d1,time_var_d2,covar_par,x0_logistic,transf_d)
	numD12,numD21 =  get_num_dispersals(marginal_dispersal_rate_temp,r_vec)
	rateD12,rateD21 = marginal_dispersal_rate_temp[:,0], marginal_dispersal_rate_temp[:,1]

	if args.DdE: # Dispersal dep Extinction		
		#numD12res = rescale_vec_to_range(log(1+numD12), r=10., m=0)
		#numD21res = rescale_vec_to_range(log(1+numD21), r=10., m=0)		
		div_traj1,div_traj2 = get_est_div_traj(r_vec)
		
		numD12res = rescale_vec_to_range((1+numD12)/(1+div_traj2), r=10., m=0)
		numD21res = rescale_vec_to_range((1+numD21)/(1+div_traj1), r=10., m=0)		

 		
		
		# NOTE THAT no. dispersals from 1=>2 affects extinction in 2 and vice versa
		transf_e=1
 		time_var_e2,time_var_e1 = numD12res, numD21res
		ext_vec = ext_rate_vec
	elif args.DisdE: # Dispersal RATE dep Extinction		
		rateD12res = log(rateD12)
		rateD21res = log(rateD21)
		transf_e=1
 		time_var_e2,time_var_e1 = rateD12res,rateD21res
		ext_vec = ext_rate_vec		
	elif args.DivdE: # Diversity dep Extinction
		# NOTE THAT extinction in 1 depends diversity in 1
		transf_e=4
 		time_var_e1,time_var_e2 = get_est_div_traj(r_vec)
 		do_approx_div_traj = 1
		ext_vec = ext_rate_vec
	elif args.TdE: # Time dep Extinction
		covar_par[2:4]=0
		x0_logistic[2:4]=0
		ext_vec = ext_rate_vec[Q_index,:]
		ext_vec = ext_vec[0:-1]
		transf_e=0
		time_var_e1,time_var_e2=time_varD,time_varD
	else: # Temp dependent Extinction
		ext_vec = ext_rate_vec
		transf_e=1
		time_var_e1,time_var_e2=time_varE,time_varE
	
	if do_approx_div_traj == 1:		
		approx_d1,approx_d2 = approx_div_traj(nTaxa, dis_vec, ext_vec, 
		                                      argsDivdD, argsDivdE, argsG,
                    				      r_vec, alpha, YangGammaQuant, pp_gamma_ncat, bin_size, Q_index, Q_index_first_occ, 
                    				      covar_par, offset_dis_div1, offset_dis_div2, offset_ext_div1, offset_ext_div2,
		                                      time_series, len_time_series, bin_first_occ, first_area)
		if args.DivdD:
			time_var_d2 = approx_d1 # Limits dispersal into 1
			time_var_d1 = approx_d2 # Limits dispersal into 2
		if args.DivdE:
			time_var_e1 = approx_d1
			time_var_e2 = approx_d2
		
	if args.lgE: transf_e = 2
	if args.linE: transf_e = 3

	if model_DUO:
		numD12res = rescale_vec_to_range(log(1+numD12), r=10., m=0)
		numD21res = rescale_vec_to_range(log(1+numD21), r=10., m=0)		
		# NOTE THAT no. dispersals from 1=>2 affects extinction in 2 and vice versa
 		time_var_e2two, time_var_e1two=  numD12res, numD21res
		ext_vec = ext_rate_vec	
		# LINEAR		
		#_ Q_list, marginal_rates_temp= make_Q_Covar4VDdEDOUBLE(dis_vec,ext_vec,time_var_d1,time_var_d2,time_var_e1,time_var_e2,time_var_e1two,time_var_e2two,covar_par,x0_logistic,transf_d,transf_e=3)
		# EXPON
		Q_list, marginal_rates_temp= make_Q_Covar4VDdEDOUBLE(dis_vec,ext_vec,time_var_d1,time_var_d2,time_var_e1,time_var_e2,time_var_e1two,time_var_e2two,covar_par,x0_logistic,transf_d,transf_e=1)
	else:
		#print "dis_vec", dis_vec
		#print "ext_vec", ext_vec
		#print "time_var_d1", time_var_d1
		#print "time_var_d2", time_var_d2
		#print "time_var_e1",
		#print "time_var_e2",
		#print "covar_par", covar_par
		#print "x0_logistic", x0_logistic
		#print "transf_d", transf_d
		#print "transf_e", transf_e	        	
		Q_list, marginal_rates_temp= make_Q_Covar4VDdE(dis_vec,ext_vec,time_var_d1,time_var_d2,time_var_e1,time_var_e2,covar_par,x0_logistic,transf_d,transf_e, offset_dis_div1, offset_dis_div2, offset_ext_div1, offset_ext_div2)
	
		
	
	#__      
	#__      
	#__      
	#__      else:
	#__      		time_var_e1,time_var_e2 = get_est_div_traj(r_vec) # diversity dependent extinction
	#__      		Q_list= make_Q_Covar4VDdE(dis_vec[0:-1],ext_rate_vec,time_varD,time_var_e1,time_var_e2,covar_par,transf_d=0)	
	#__      	
	#__      	
	#__      elif args.DdE:
	#__      	# TRANSFORM Q MATRIX DeE MODEL	
	#__      	# NOTE THAT no. dispersals from 1=>2 affects extinction in 2 and vice versa
 	#__      	time_var_e2,time_var_e1 = get_num_dispersals(dis_rate_vec,r_vec)
	#__      	#print time_var_e1,time_var_e2
	#__      	Q_list= make_Q_Covar4VDdE(dis_rate_vec,ext_rate_vec,time_varD,time_var_e1,time_var_e2,covar_par)
	#__      	#for i in [1,10,20]:
	#__      	#	print Q_list[i]
	#__      	#quit()
	#__      elif args.DivdE:
	#__      	# TRANSFORM Q MATRIX DeE MODEL	
	#__      	# NOTE THAT extinction in 1 depends diversity in 1
 	#__      	time_var_e1,time_var_e2 = get_est_div_traj(r_vec)
	#__      	#print time_var_e1,time_var_e2
	#__      	Q_list= make_Q_Covar4VDdE(dis_rate_vec,ext_rate_vec,time_varD,time_var_e1,time_var_e2,covar_par)
	#__      else:
	#__      	# TRANSFORM Q MATRIX	
	#__      	Q_list= make_Q_Covar4V(dis_rate_vec,ext_rate_vec,time_varD,covar_par)
	#__      	#print "Q2", Q_list[0], covar_par
	#__      	#print Q_list[3]
	
	
	#if it % print_freq == 0: 
	#	print it,  Q_list[0],Q_list_old,covar_par
	if r[0] < update_freq[1] or it==0:
				w_list,vl_list,vl_inv_list = get_eigen_list(Q_list)
	lik, weight_per_taxon = lik_DES(Q_list, w_list, vl_list, vl_inv_list, delta_t, r_vec, rho_at_present_LIST, r_vec_indexes_LIST, sign_list_LIST,OrigTimeIndex,Q_index, alpha, YangGammaQuant, pp_gamma_ncat, num_processes, use_Pade_approx)
		
	prior= sum(prior_exp(dis_rate_vec,prior_exp_rate))+sum(prior_exp(ext_rate_vec,prior_exp_rate))+prior_normal(covar_par,0,1)
	
	lik_alter = lik * scal_fac_TI[scal_fac_ind]

	accepted_state = 0
	if args.A == 3: 
		accepted_state = 1
		lik = minf
	if np.isfinite((lik_alter+prior+hasting)) == True:
		if it==0: 
			likA=lik_alter+0.
			MLik=likA-1
			ml_it=0
		if runMCMC == 1:
			# MCMC
			if (lik_alter-(likA* scal_fac_TI[scal_fac_ind]) + prior-priorA +hasting >= log(np.random.uniform(0,1))) or (gibbs_sample == 1) :
				accepted_state=1
		else:
			# ML (approx maximum a posteriori algorithm)
			if ml_it==0: rr = 0 # only accept improvements
			else: rr = log(np.random.uniform(0,1))
			if (lik_alter-(likA* scal_fac_TI[scal_fac_ind]))*map_power >= rr:
				accepted_state=1
	elif it==0: 
		MLik=likA-1
	if accepted_state:
		dis_rate_vec_A= dis_rate_vec
		ext_rate_vec_A= ext_rate_vec
		r_vec_A=        r_vec
		likA=lik
		priorA=prior
		covar_par_A=covar_par
		x0_logistic_A=x0_logistic
		marginal_rates_A = marginal_rates_temp
		numD12A,numD21A = numD12,numD21
		alphaA = alpha
	
	log_to_file=0	
	if it % print_freq == 0:
		sampling_prob = r_vec_A[:,1:len(r_vec_A[0])-1].flatten()
		q_rates = -log(sampling_prob)/bin_size
		print it,"\t",likA,lik,scal_fac_TI[scal_fac_ind]
		if argsDivdD:
			dis_rate_vec_A[0,:] = dis_rate_vec_A[0,:] / (1. - ([offset_dis_div2, offset_dis_div1]/covar_par[0:2]))
		if argsDivdE:
			ext_rate_vec_A[0,:] = ext_rate_vec_A[0,:] * (1. - ([offset_ext_div2, offset_ext_div1]/covar_par[2:4]))	
		print "\td:", dis_rate_vec_A.flatten(), "e:", ext_rate_vec_A.flatten(),"q:",q_rates,"alpha:",alphaA		
		print "\ta/k:",covar_par_A,"x0:",x0_logistic_A
	if it % sampling_freq == 0 and it >= burnin and runMCMC == 1:
		log_to_file=1
		# sampling_prob = r_vec_A[:,1:len(r_vec_A[0])-1].flatten()
		# q_rates = -log(sampling_prob)/bin_size
		# log_state= [it,likA+priorA, priorA,likA]+list(dis_rate_vec_A.flatten())+list(ext_rate_vec_A.flatten())+list(q_rates)
		# log_state = log_state+list(covar_par_A[0:2])	
		# if args.lgD: log_state = log_state+list(x0_logistic_A[0:2])
		# log_state = log_state+list(covar_par_A[2:4])	
		# if args.lgE: log_state = log_state+list(x0_logistic_A[2:4])
		# log_state = log_state+[prior_exp_rate]+[scal_fac_TI[scal_fac_ind]]
		# wlog.writerow(log_state)
		# logfile.flush()
		# os.fsync(logfile)
	if runMCMC == 0:
		if likA>MLik: 
			log_to_file=1
			# sampling_prob = r_vec_A[:,1:len(r_vec_A[0])-1].flatten()
			# q_rates = -log(sampling_prob)/bin_size
			# log_state= [it,likA+priorA, priorA,likA]+list(dis_rate_vec_A.flatten())+list(ext_rate_vec_A.flatten())+list(q_rates)
			# log_state = log_state+list(covar_par_A[0:2])	
			# if args.lgD: log_state = log_state+list(x0_logistic_A[0:2])
			# log_state = log_state+list(covar_par_A[2:4])	
			# if args.lgE: log_state = log_state+list(x0_logistic_A[2:4])
			# log_state = log_state+[prior_exp_rate]+[scal_fac_TI[scal_fac_ind]]
			# wlog.writerow(log_state)
			# logfile.flush()
			# os.fsync(logfile)
			MLik=likA+0.
			ML_dis_rate_vec = dis_rate_vec_A+0.
			ML_ext_rate_vec = ext_rate_vec_A+0.
			ML_covar_par    = covar_par_A   +0.
			ML_r_vec        = r_vec_A       +0.
			ML_alpha        = alphaA        +0.
			ml_it=0
		else:	ml_it +=1
		# exit when ML convergence reached
		if ml_it==100:
			print "restore ML estimates", MLik, likA+0.
			dis_rate_vec_A = ML_dis_rate_vec
			ext_rate_vec_A = ML_ext_rate_vec
			covar_par_A    = ML_covar_par   
			r_vec_A        = ML_r_vec   
			alphaA         = ML_alpha    
			#if ml_it==100:
			print scale_proposal, "changed to:",
			scale_proposal = 0.1
			print scale_proposal
			#update_rate_freq = 0.15
		#	if ml_it==200:
		#		print scale_proposal, "changed to:",
		#		scale_proposal = 0.5
		#if ml_it>8:
		#	scale_proposal = 0 
		#	print lik, likA, MLik
		if ml_it>max_ML_iterations:
			msg= "Convergence reached. ML = %s (%s iterations)" % (round(MLik,3),it)
			sys.exit(msg)
	if log_to_file:
		sampling_prob = r_vec_A[:,1:len(r_vec_A[0])-1].flatten()
		q_rates = -log(sampling_prob)/bin_size
		if argsDivdD and args.A != 3:
			dis_rate_vec_A[0,:] = dis_rate_vec_A[0,:] / (1. - ([offset_dis_div2, offset_dis_div1]/covar_par[0:2]))
		if argsDivdE and args.A != 3:
			ext_rate_vec_A[0,:] = ext_rate_vec_A[0,:] * (1. - ([offset_ext_div2, offset_ext_div1]/covar_par[2:4]))	
		log_state= [it,likA+priorA, priorA,likA]+list(dis_rate_vec_A.flatten())+list(ext_rate_vec_A.flatten())+list(q_rates)		
		log_state = log_state+list(covar_par_A[0:2])	
		if args.lgD: log_state = log_state+list(x0_logistic_A[0:2])
		log_state = log_state+list(covar_par_A[2:4])	
		if args.lgE: log_state = log_state+list(x0_logistic_A[2:4])
		if argsG: log_state = log_state + list(alpha)
		log_state = log_state+[prior_exp_rate]+[scal_fac_TI[scal_fac_ind]]
		wlog.writerow(log_state)
		logfile.flush()
		
	
	log_marginal_rates = 1	
	log_n_dispersals = 0
	if log_marginal_rates and log_to_file == 1:
		if log_n_dispersals:
			temp_marginal_d12 = list(numD12A[::-1])
			temp_marginal_d21 = list(numD21A[::-1])
		else:			
			temp_marginal_d12 = list(marginal_rates_A[0][:,0][::-1])
			temp_marginal_d21 = list(marginal_rates_A[0][:,1][::-1])
		temp_marginal_e1  = list(marginal_rates_A[1][:,0][::-1])
		temp_marginal_e2  = list(marginal_rates_A[1][:,1][::-1])
		log_state = [it]+temp_marginal_d12+temp_marginal_d21+temp_marginal_e1+temp_marginal_e2	
		rlog.writerow(log_state)
		ratesfile.flush()
		os.fsync(ratesfile)
		

print "elapsed time:", time.time()-start_time

if num_processes>0:
	pool_lik.close()
	pool_lik.join()


quit()

