#!/usr/bin/env python 
# Created by Daniele Silvestro on 02/03/2012 => dsilvestro@senckenberg.de 
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

p.add_argument('-v',  action='version', version='%(prog)s')
p.add_argument('-cite',      help='print DES citation', action='store_true', default=False)
p.add_argument('-A',  type=int, help='algorithm - 0: parameter estimation, 1: TI, 2: ML', default=0, metavar=0) # 0: par estimation, 1: TI
p.add_argument('-k',  type=int,   help='TI - no. scaling factors', default=10, metavar=10)
p.add_argument('-a',  type=float, help='TI - shape beta distribution', default=.3, metavar=.3)
p.add_argument('-hp',      help='Use hyper-prior on rates', action='store_true', default=False)
p.add_argument('-pw',  type=float, help='Exponent acceptance ratio (ML)', default=58, metavar=58) # accept 0.95 post ratio with 5% prob

p.add_argument('-d',  type=str, help='input data set',   default="", metavar="<input file>")
p.add_argument('-n',      type=int, help='mcmc generations',default=100000, metavar=100000)
p.add_argument('-s',      type=int, help='sample freq.', default=100, metavar=100)
p.add_argument('-p',      type=int, help='print freq.',  default=100, metavar=100)
p.add_argument('-b',  type=int, help='burnin',  default=0)
p.add_argument('-thread',  type=int, help='no threads',  default=0)
p.add_argument('-ver',  type=int, help='verbose',   default=0, metavar=0)
p.add_argument('-pade',  type=int, help='0) Matrix decomposition 1) Use Pade approx (slow)', default=0, metavar=0)
p.add_argument('-qtimes',  type=float, help='shift time (Q)',  default=[], metavar=0, nargs='+') # '*'
p.add_argument('-sum',  type=str, help='Summarize results (provide log file)',  default="", metavar="log file")
p.add_argument('-symd',      help='symmetric dispersal rates', action='store_true', default=False)
p.add_argument('-syme',      help='symmetric extinction rates', action='store_true', default=False)
p.add_argument('-symq',      help='symmetric preservation rates', action='store_true', default=False)
p.add_argument('-data_in_area', type=int,  help='if data only in area 1 set to 1 (set to 2 if data only in area 2)', default=0)
p.add_argument('-red', type=int,  help='if -red 1: reduce dataset to taxa with occs in both areas', default=0)


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
p.add_argument("-mG", help='Model - Gamma heterogeneity of preservation rate', action='store_true', default=False)
p.add_argument("-ncat", type=int, help='Model - Number of categories for Gamma heterogeneity', default=4, metavar=4)

args = p.parse_args()
if args.cite is True:
	sys.exit(citation)
simulation_no = args.i
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
argsG = args.mG
pp_gamma_ncat = args.ncat

### MCMC SETTINGS
if args.A ==2: 
	runMCMC = 0 # approximate ML estimation
	n_generations   = 10000
	update_freq     = [.33,.66,1]
	sampling_freq   = 10
	max_ML_iterations = 250	
else: 
	runMCMC = 1
	n_generations   = args.n
	update_freq     = [.33,.66,1]
	if args.hp == True: 
		update_freq = [.3,.6,.9]
	sampling_freq   = args.s
print_freq      = args.p
map_power       = args.pw
hp_alpha        = 2.
hp_beta         = 2.
use_Pade_approx = args.pade

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
	out_log = "%s/sim_%s_b_%s_q_%s_mcmc_%s_%s_%s_%s_%s_%s_%s%s.log" \
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
	if equal_d is True: model_tag+= "_symd"
	if equal_e is True: model_tag+= "_syme"
	if equal_q is True: model_tag+= "_symq"
	if runMCMC == 0: model_tag+= "_ML"
	if argsG is True: model_tag+= "_G" 
	out_log ="%s/%s_%s%s%s%s.log" % (output_wd,name_file,simulation_no,Q_times_str,ti_tag,model_tag)
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
dis_rate_vec=np.random.uniform(0.1,0.2,nareas*n_Q_times).reshape(n_Q_times,nareas)
ext_rate_vec=np.random.uniform(0.01,0.05,nareas*n_Q_times).reshape(n_Q_times,nareas)
if equal_d is True:
	d_temp=dis_rate_vec[:,0]
	dis_rate_vec = array([d_temp,d_temp]).T
if equal_e is True:
	e_temp=ext_rate_vec[:,0]
	ext_rate_vec = array([e_temp,e_temp]).T

r_vec= np.zeros((n_Q_times,nareas+2)) 
r_vec[:,1:3]=0.25
r_vec[:,3]=1
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
# define time labels for log file
time_slices = sort([max(time_series)+1,0] + list(Q_times))[::-1]
time_bin_label=[]
for i in range(1,len(time_slices)): time_bin_label.append("%s-%s" % (round(time_slices[i-1],1),round(time_slices[i],1)))

logfile = open(out_log , "w",0) 
head="it\tposterior\tprior\tlikelihood"
for i in range(n_Q_times): head+= "\td12_%s\td21_%s" % (time_bin_label[i],time_bin_label[i])
for i in range(n_Q_times): head+= "\te1_%s\te2_%s" %   (time_bin_label[i],time_bin_label[i])
for i in range(n_Q_times): head+= "\tq1_%s\tq2_%s" %   (time_bin_label[i],time_bin_label[i])

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

update_rate_freq = max(0.2, 1./sum(np.shape(dis_rate_vec)))
print "Origination time (binned):", OrigTimeIndex, delta_t # update_rate_freq, update_freq
l=1
recursive = np.arange(OrigTimeIndex[l],len(delta_t))[::-1]
print recursive
print shape(r_vec_indexes_LIST[l]),shape(sign_list_LIST[l])
#quit()
	
scal_fac_ind=0
for it in range(n_generations * len(scal_fac_TI)):		
	if (it+1) % (n_generations+1) ==0: 
		print it, n_generations  
		scal_fac_ind+=1
	if it ==0: 
		dis_rate_vec_A= dis_rate_vec
		ext_rate_vec_A= ext_rate_vec
		r_vec_A=        r_vec
		likA=-inf
		priorA=-inf
		alphaA = alpha 

	dis_rate_vec= dis_rate_vec_A
	ext_rate_vec= ext_rate_vec_A
	r_vec=        r_vec_A
	hasting = 0
	gibbs_sample = 0
	if it>=0: r= np.random.random()
	#else: r = 2
	if r < update_freq[0]: 
		if equal_d is True:
			#--> SYMMETRIC DISPERSAL
			d_temp,hasting = update_multiplier_proposal_freq(dis_rate_vec_A[:,0],d=1.1,f=update_rate_freq)
			dis_rate_vec = array([d_temp,d_temp]).T
			#--> CONSTANT DISPERSAL IN AREA 2
			# d_temp,hasting = update_multiplier_proposal_freq(dis_rate_vec_A,d=1.1,f=update_rate_freq)
			# d_temp[:,1] = d_temp[0,1]
			# dis_rate_vec = d_temp
			#--> CONSTANT DISPERSAL IN AREA 2, 0 in the Miocene
			# d_temp,hasting = update_multiplier_proposal_freq(dis_rate_vec_A,d=1.1,f=update_rate_freq)
			# d_temp[:,1] = d_temp[2,1]
			# d_temp[0,1] = small_number
			# dis_rate_vec = d_temp
			#--> 0 DISPERSAL in Miocene and Pliocene
			# d_temp,hasting = update_multiplier_proposal_freq(dis_rate_vec_A,d=1.1,f=update_rate_freq)
			# d_temp[0,:] = small_number
			# d_temp[1,:] = small_number
			# dis_rate_vec = d_temp
		else:
			dis_rate_vec,hasting=update_multiplier_proposal_freq(dis_rate_vec_A,d=1.1,f=update_rate_freq)
	elif r < update_freq[1]: 
		if equal_e is True:
			e_temp,hasting = update_multiplier_proposal_freq(ext_rate_vec_A[:,0],d=1.1,f=update_rate_freq)
			ext_rate_vec = array([e_temp,e_temp]).T
		else:
			ext_rate_vec,hasting=update_multiplier_proposal_freq(ext_rate_vec_A,d=1.1,f=update_rate_freq)
	elif r<=update_freq[2]: 
		r_vec=update_parameter_uni_2d_freq(r_vec_A,d=0.1,f=update_rate_freq)
		if argsG is True:
			alpha,hasting=update_multiplier_proposal(alphaA,d=1.1)
		r_vec[:,0]=0
		#--> SYMMETRIC SAMPLING
		if equal_q is True: r_vec[:,2] = r_vec[:,1]
		#--> CONSTANT Q IN AREA 2
                # if equal_q is True: r_vec[:,2] = r_vec[1,2]
		
		r_vec[:,3]=1
		# CHECK THIS: CHANGE TO VALUE CLOSE TO 1? i.e. for 'ghost' area 
		if args.data_in_area == 1: r_vec[:,2] = small_number 
		elif  args.data_in_area == 2: r_vec[:,1] = small_number
		
	else:
		gibbs_sample = 1
		prior_exp_rate = gibbs_sampler_hp(np.concatenate((dis_rate_vec,ext_rate_vec)),hp_alpha,hp_beta)

	Q_list= make_Q_list(dis_rate_vec,ext_rate_vec)
	#print Q_list, len(delta_t)	
	

	if num_processes==0:
		if use_Pade_approx==0:
			#t1= time.time()
			lik=0
			if r < update_freq[1] or it==0:
				w_list,vl_list,vl_inv_list = get_eigen_list(Q_list)
		    	if argsG is False: 
			        for l in list_taxa_index:
				        l_temp = calc_likelihood_mQ_eigen([delta_t,r_vec,w_list,vl_list,vl_inv_list,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index,Q_index])
				        #print l,  l_temp
				        lik +=l_temp
			                #print "elapsed time:", time.time()-t1
			else:
                                for l in list_taxa_index:
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
                                                lik_vec[i]  = calc_likelihood_mQ_eigen([delta_t,r_vec_Gamma,w_list,vl_list,vl_inv_list,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index,Q_index])
                                        lik2= lik_vec-np.max(lik_vec)
                                        lik += sum(log(sum(exp(lik2))/pp_gamma_ncat)+np.max(lik_vec))
		else:
			#t1= time.time()
			lik=0
			if argsG is False:
			        for l in list_taxa_index:
				        lik += calc_likelihood_mQ([delta_t,r_vec,Q_list,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index,Q_index])
                        else:
                                for l in list_taxa_index:
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
                                                lik_vec[i] = calc_likelihood_mQ([delta_t,r_vec_Gamma,Q_list,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index,Q_index])
                                        lik2= lik_vec-np.max(lik_vec)
                                        lik += sum(log(sum(exp(lik2))/pp_gamma_ncat)+np.max(lik_vec))
			#print "l2",lik
			#print "elapsed time:", time.time()-t1
		
			
	else: # multi=processing
		if use_Pade_approx==0:
			#t1= time.time()
			w_list,vl_list,vl_inv_list = get_eigen_list(Q_list)
                        if argsG is False:
			        args_mt_lik = [ [delta_t,r_vec,w_list,vl_list,vl_inv_list,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index,Q_index] for l in list_taxa_index ]
			        lik= sum(np.array(pool_lik.map(calc_likelihood_mQ_eigen, args_mt_lik)))
			        #print "l3",lik
			        #print "elapsed time:", time.time()-t1
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
                                        args_mt_lik = [ [delta_t,r_vec_Gamma,w_list,vl_list,vl_inv_list,rho_at_present_LIST[l],r_vec_indexes_LIST[l],sign_list_LIST[l],OrigTimeIndex[l],Q_index,Q_index] for l in list_taxa_index ]
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


		
	prior= sum(prior_exp(dis_rate_vec,prior_exp_rate))+sum(prior_exp(ext_rate_vec,prior_exp_rate))
	
	lik_alter = lik * scal_fac_TI[scal_fac_ind]
	
	if np.isfinite((lik_alter+prior+hasting)) == True:
		if it==0: 
			likA=lik_alter+0.
			MLik=likA
			ml_it=0
		if runMCMC == 1:
			if (lik_alter-(likA* scal_fac_TI[scal_fac_ind]) + prior-priorA +hasting >= log(np.random.uniform(0,1))) or (gibbs_sample == 1) :
				dis_rate_vec_A= dis_rate_vec
				ext_rate_vec_A= ext_rate_vec
				r_vec_A=        r_vec
				likA=lik
				priorA=prior
				alphaA = alpha
		else:
			# ML (approx maximum a posteriori algorithm)
			if ((lik_alter-(likA* scal_fac_TI[scal_fac_ind]))*map_power >= log(np.random.uniform(0,1))):
				dis_rate_vec_A= dis_rate_vec
				ext_rate_vec_A= ext_rate_vec
				r_vec_A=        r_vec
				likA=lik
				priorA=prior
				alphaA = alpha
		
	if it % print_freq == 0:
		sampling_prob = r_vec_A[:,1:len(r_vec_A[0])-1].flatten()
		q_rates = -log(sampling_prob)/bin_size
		print it,"\t",likA,"\t",  dis_rate_vec_A.flatten(),ext_rate_vec_A.flatten(),scal_fac_TI[scal_fac_ind], q_rates, alpha
	if it % sampling_freq == 0 and it >= burnin and runMCMC == 1:
		sampling_prob = r_vec_A[:,1:len(r_vec_A[0])-1].flatten()
		q_rates = -log(sampling_prob)/bin_size
		# This feels unneccesary complicated. Maybe easier to log a constant alpha?
		if argsG is False:
			log_state= [it,likA+priorA, priorA,likA]+list(dis_rate_vec_A.flatten())+list(ext_rate_vec_A.flatten())+ list(q_rates) +[prior_exp_rate]+[scal_fac_TI[scal_fac_ind]]
                else:
                        log_state= [it,likA+priorA, priorA,likA]+list(dis_rate_vec_A.flatten())+list(ext_rate_vec_A.flatten())+ list(q_rates)+ list(alpha) +[prior_exp_rate]+[scal_fac_TI[scal_fac_ind]]
		wlog.writerow(log_state)
		logfile.flush()
		os.fsync(logfile)
	if runMCMC == 0:
		if likA>MLik:
			sampling_prob = r_vec_A[:,1:len(r_vec_A[0])-1].flatten()
			q_rates = -log(sampling_prob)/bin_size
			# This feels unneccesary complicated. Maybe easier to log a constant alpha?
			if argsG is False:
				log_state= [it,likA+priorA, priorA,likA]+list(dis_rate_vec_A.flatten())+list(ext_rate_vec_A.flatten())+ list(q_rates) +[prior_exp_rate]+[scal_fac_TI[scal_fac_ind]]
			else:
				log_state= [it,likA+priorA, priorA,likA]+list(dis_rate_vec_A.flatten())+list(ext_rate_vec_A.flatten())+ list(q_rates)+ list(alpha) +[prior_exp_rate]+[scal_fac_TI[scal_fac_ind]]
			wlog.writerow(log_state)
			logfile.flush()
			os.fsync(logfile)
			MLik=likA
			ml_it=0
		else:	ml_it +=1
		# exit when ML convergence reached
		if ml_it>max_ML_iterations:
			msg= "Convergence reached. ML = %s" % (round(MLik,3))
			sys.exit(msg)
			


print "elapsed time:", time.time()-start_time

if num_processes>0:
	pool_lik.close()
	pool_lik.join()


quit()

