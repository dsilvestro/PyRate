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
p.add_argument('-A',        type=int, help='algorithm - 0: parameter estimation, 1: TI, 2: ML', default=0, metavar=0) # 0: par estimation, 1: TI
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

if argsG == True and args.DdE or argsG == True and args.cov_and_dispersal:
	sys.exit("Preservation heterogeneity not compatible with dispersal dependent extinction")
if argsG == True and args.DivdD or argsG == True and args.DivdE:
	sys.exit("Preservation heterogeneity not compatible with dispersal dependence")

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
try:
	time_var_temp = get_binned_continuous_variable(time_series, args.varD)
	# SHIFT TIME VARIABLE ()
	time_var = time_var_temp-time_var_temp[len(delta_t)-1]
	print "Rescaled variable",time_var, time_series
	if args.varE=="":
		time_varE=time_var
	else:
		## TEMP FOR EXTINCTION
		time_var_temp = get_binned_continuous_variable(time_series, args.varE)
		# SHIFT TIME VARIABLE ()
		time_varE = time_var_temp-time_var_temp[len(delta_t)-1]
		
except:
	time_var = np.ones(len(time_series)-1)
	print "Covariate-file not found"





ratesfile = open(out_rates , "w",0) 
head="it"
for i in range(len(time_var)): head+= "\td12_%s" % (i)
for i in range(len(time_var)): head+= "\td21_%s" % (i)
for i in range(len(time_var)): head+= "\te1_%s" % (i)
for i in range(len(time_var)): head+= "\te2_%s" % (i)
head=head.split("\t")
rlog=csv.writer(ratesfile, delimiter='\t')
rlog.writerow(head)


# DIVERSITY TRAJECTORIES
tbl = np.genfromtxt(input_data, dtype=str, delimiter='\t')
tbl_temp=tbl[1:,1:]
data_temp=tbl_temp.astype(float)

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

#print Q_index
# print dis_rate_vec
# print time_series, time_var

#get_num_dispersals(d12,d21,r_vec)


#############################################
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
		dis_vec = dis_rate_vec[Q_index,:]
		dis_vec = dis_vec[0:-1]
		transf_d=0
		time_var_d1,time_var_d2=time_var,time_var
	elif args.DivdD: # Diversity dependent D
		transf_d=1
		dis_vec = dis_rate_vec
		time_var_d2,time_var_d1 = get_est_div_traj(r_vec)
	else: # temp dependent D	
		transf_d=1
		dis_vec = dis_rate_vec
		time_var_d1,time_var_d2=time_var,time_var

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
		transf_e=1
 		time_var_e1,time_var_e2 = get_est_div_traj(r_vec)
		ext_vec = ext_rate_vec
	elif args.TdE: # Time dep Extinction
		covar_par[2:4]=0
		ext_vec = ext_rate_vec[Q_index,:]
		ext_vec = ext_vec[0:-1]
		transf_e=0
		time_var_e1,time_var_e2=time_var,time_var
	else: # Temp dependent Extinction
		ext_vec = ext_rate_vec
		transf_e=1
		time_var_e1,time_var_e2=time_varE,time_varE
		
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
		Q_list, marginal_rates_temp= make_Q_Covar4VDdE(dis_vec,ext_vec,time_var_d1,time_var_d2,time_var_e1,time_var_e2,covar_par,x0_logistic,transf_d,transf_e)
	
		
	
	#__      
	#__      
	#__      
	#__      else:
	#__      		time_var_e1,time_var_e2 = get_est_div_traj(r_vec) # diversity dependent extinction
	#__      		Q_list= make_Q_Covar4VDdE(dis_vec[0:-1],ext_rate_vec,time_var,time_var_e1,time_var_e2,covar_par,transf_d=0)	
	#__      	
	#__      	
	#__      elif args.DdE:
	#__      	# TRANSFORM Q MATRIX DeE MODEL	
	#__      	# NOTE THAT no. dispersals from 1=>2 affects extinction in 2 and vice versa
 	#__      	time_var_e2,time_var_e1 = get_num_dispersals(dis_rate_vec,r_vec)
	#__      	#print time_var_e1,time_var_e2
	#__      	Q_list= make_Q_Covar4VDdE(dis_rate_vec,ext_rate_vec,time_var,time_var_e1,time_var_e2,covar_par)
	#__      	#for i in [1,10,20]:
	#__      	#	print Q_list[i]
	#__      	#quit()
	#__      elif args.DivdE:
	#__      	# TRANSFORM Q MATRIX DeE MODEL	
	#__      	# NOTE THAT extinction in 1 depends diversity in 1
 	#__      	time_var_e1,time_var_e2 = get_est_div_traj(r_vec)
	#__      	#print time_var_e1,time_var_e2
	#__      	Q_list= make_Q_Covar4VDdE(dis_rate_vec,ext_rate_vec,time_var,time_var_e1,time_var_e2,covar_par)
	#__      else:
	#__      	# TRANSFORM Q MATRIX	
	#__      	Q_list= make_Q_Covar4V(dis_rate_vec,ext_rate_vec,time_var,covar_par)
	#__      	#print "Q2", Q_list[0], covar_par
	#__      	#print Q_list[3]
	
	
	#if it % print_freq == 0: 
	#	print it,  Q_list[0],Q_list_old,covar_par
	
	if num_processes==0:
		if use_Pade_approx==0:
			#t1= time.time()
			lik=0
			if r[0] < update_freq[1] or it==0:
				w_list,vl_list,vl_inv_list = get_eigen_list(Q_list)
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
                                        lik2= lik_vec-np.max(lik_vec)
                                        lik += sum(log(sum(exp(lik2))/pp_gamma_ncat)+np.max(lik_vec))
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
                                        lik2= lik_vec-np.max(lik_vec)
                                        lik += sum(log(sum(exp(lik2))/pp_gamma_ncat)+np.max(lik_vec))
			#print "lik2",lik
			#print "elapsed time:", time.time()-t1
		
			
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


		
	prior= sum(prior_exp(dis_rate_vec,prior_exp_rate))+sum(prior_exp(ext_rate_vec,prior_exp_rate))+prior_normal(covar_par,0,1)
	
	lik_alter = lik * scal_fac_TI[scal_fac_ind]

	accepted_state = 0
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

