from numpy import *
import numpy as np
import scipy
import csv, os
np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit
import random as rand
from itertools import *

def powerset(iterable):
	# powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def build_rho_index_vec(obs_state,nareas,possible_areas,verbose=0):
	obs_state=set(obs_state)
	r_vec_index=[]
	r_neg_index=np.zeros((len(possible_areas),nareas))
	if verbose ==1: print "anc_state\tobs_state\ttemp\tr_neg"
	for i in range(len(possible_areas)):
		anc_state= set(possible_areas[i])
		if 1>2: pass #len(anc_state)==0: temp = np.repeat(0,nareas)
		else:
			temp=[]
			for j in range(nareas): 
				if j in anc_state and j in obs_state: 
				#	temp.append(nareas+1) # nareas+1 is index of r_vec = 1 
					temp.append(j+1) # j+1 should be the index of r_{j}
					r_neg_index[i,j]=1 # this will transform it in (1 - r_{j})
					
				elif j in anc_state and j not in obs_state: # area present in anc_states but not observed
				#	temp.append(nareas+1)
					temp.append(j+1)  # j+1 should be the index of r_{j}
				elif j not in anc_state and j not in obs_state: # area absent in anc_states AND not observed
					#temp.append(j+1) # j+1 should be the index of r_{j}
					#r_neg_index[i,j]=1 # this will transform it in (1 - r_{j})
					temp.append(nareas+1) # this option replaces (1 - r_{j}) with 1 | nareas+1 is index of r_vec = 1 
				elif j not in anc_state and j in obs_state:
					temp.append(0) # 0 is index of r_vec = 0
				else: print "Warning: problem in function <build_rho_index_vec>"
		if verbose ==1: 
			r_vec= np.array([0]+list(np.zeros(nareas)+0.25) +[1])
			r_vec[1]=0.33
			print anc_state,"\t",obs_state,"\t",temp,"\t",r_neg_index[i], abs(r_neg_index[i]-r_vec[temp]),np.prod(abs(r_neg_index[i]-r_vec[temp]))
		r_vec_index.append(temp)	
	#print "\nFINAL",np.array(r_vec_index),r_neg_index	, "\nEND"
	return np.array(r_vec_index),r_neg_index

def build_list_rho_index_vec(obs_area_series,nareas,possible_areas,verbose=0):
	r_vec_indexes= list()
	sign_list= list()
	for n in obs_area_series:
		r_ind,sign= build_rho_index_vec(n,nareas,possible_areas,verbose)  # this is fixed (but different) for each node
		r_vec_indexes.append(r_ind)
		sign_list.append(sign)
		#print n,r_ind,sign
	return r_vec_indexes,sign_list

def make_Q(dv,ev): # construct Q matrix
	D=0
	[d1,d2] = dv # d1 A->B; d2 B->A;
	[e1,e2] = ev
	Q= np.array([
		[D, 0, 0, 0 ],
		[e1,D, 0, d1],
		[e2,0, D, d2],
		[0 ,e1,e2,D ]	
	])
	# fill diagonal values
	np.fill_diagonal(Q, -np.sum(Q,axis=1))
	return Q

def make_Q_list(dv_list,ev_list): # construct list of Q matrices
	Q_list=[]
	#print len(dv_list), dv_list
	for i in range(len(dv_list)):
		D=0
		[d1,d2] = dv_list[i] # d1 A->B; d2 B->A;
		[e1,e2] = ev_list[i]
		Q= np.array([
			[D, 0, 0, 0 ],
#			[D, d1, d2, 0 ],
			[e1,D, 0, d1],
			[e2,0, D, d2],
			[0 ,e1,e2,D ]	
		])
		# fill diagonal values
		np.fill_diagonal(Q, -np.sum(Q,axis=1))
		Q_list.append(Q)
	return Q_list
		

def make_Q3A(dv,ev): # construct Q matrix
	D=0
	[d1,d2,d3] = dv # d1 A<->B; d2 B<->C; d3 A<->C
	[e1,e2,e3] = ev
	Q= np.array([
		# 0   A   B   C   AB  BC  AC  ABC
		[ D,  0,  0,  0,  0,  0,  0,  0    ],  # 0
		[e1,  D,  0,  0, d1,  0, d3,  0    ],  # A
		[e2,  0,  D,  0, d1,  d2, 0,  0    ],  # B
		[e3,  0,  0,  D,  0,  d2,d3,  0    ],  # C
		[ 0, e2, e1,  0,  D,  0,  0,  d3+d2],  # AB
		[ 0,  0, e3, e2,  0,  D,  0,  d1+d3],  # BC
		[ 0, e3,  0, e1,  0,  0,  D,  d1+d2],  # AC
		[ 0,  0,  0,  0, e3, e1, e2,  D    ],  # ABC
	])
	# fill diagonal values
	np.fill_diagonal(Q, -np.sum(Q,axis=1))
	return Q



######################################
########      SIMULATOR      #########
######################################
def simulate_dataset(no_sim,d,e,n_taxa,n_bins=20,wd=""):
	n_bins +=1
	def random_choice_P(vector):
		probDeath=np.cumsum(vector/sum(vector)) # cumulative prob (used to randomly sample one 
		r=rand.random()                          # parameter based on its deathRate)
		probDeath=sort(append(probDeath, r))
		ind=np.where(probDeath==r)[0][0] # just in case r==1
		return [vector[ind], ind]
	
	Q = make_Q(d,e)
	D = -np.diagonal(Q) # waiting times
	
	outfile="%s/sim_%s_%s_%s_%s_%s_%s.txt" % (wd,no_sim,n_taxa,d[0],d[1],e[0],e[1]) 
	newfile = open(outfile, "wb") 
	wlog=csv.writer(newfile, delimiter='\t')
	
	TimeSpan = 50
	# origin age of taxa
	OrigTimes = np.zeros(n_taxa) # all extant at the present
	#OrigTimes = np.random.geometric(0.3,n_taxa)-1 # geometric distrib 
	OrigTimes = np.random.uniform(0,TimeSpan,n_taxa) # uniform distribution of speciation times
	OrigTimes[OrigTimes>45]=0 # avoid taxa appearing later than 5 Ma 
	
	#Times = sort(np.random.uniform(0,50,10))  #
	Times_mod = sort(np.linspace(0,TimeSpan,n_bins))
	SimStates = np.zeros(n_bins) + nan
	#deltaTimes= np.diff(Times)
	#Times_mod = sort(np.append(Times,0))
	wlog.writerow(list(Times_mod))
	newfile.flush()

	J=0
	while J < n_taxa:
		AncState = np.random.randint(1,4,1)[0] # 0 -> 0, 1 -> A, 2 -> B, 3 -> AB
		current_state=AncState
		SimStates[0] = current_state
		t = OrigTimes[J]
		SimStates[np.nonzero(Times_mod < t)[0]] = nan
		while True:
			#print J
			if D[current_state]>0: rand_t = np.random.exponential(1/D[current_state])
			else: rand_t = inf # lineage extinct
			passed_ind = np.intersect1d(np.nonzero(Times_mod < t+rand_t)[0],np.nonzero(Times_mod > t)[0])
			#print t, rand_t
			t += rand_t
			SimStates[passed_ind] = current_state
	
			if t < max(Times_mod): # did not exceed max length
				Qrow = Q[current_state]+0
				Qrow[Qrow<0]=0
				[rate, ind] = random_choice_P(Qrow)
				current_state = ind
				#print rate, ind, current_state
				#__ if ind == 0: # lineage is extinct
				#__ 	break
			else: 
				log_state=list(SimStates)
				wlog.writerow(log_state)
				newfile.flush()
				J+=1
				break
	os.fsync(newfile)
	newfile.close()

######## PARSE INPUT FILES

def transform_Array_Tuple(A): # works ONLY for 2 areas
	A=np.array(A)
	A[isnan(A)]=4 # 4 means NaN
	A =np.ndarray.astype(A,dtype=int)
	z=np.empty(np.shape(A),dtype=object)
	translation = np.array([(),(0,),(1,),(0,1),(np.nan,)])
	z = translation[A]
	return z

def parse_input_data(input_file_name,RHO_sampling=np.ones(2),verbose=0,n_sampled_bins=0):
	try:
		DATA = np.loadtxt(input_file_name)
	except:
		tbl = np.genfromtxt(input_file_name, dtype=str, delimiter='\t')
		tbl_temp=tbl[:,1:]
		DATA=tbl_temp.astype(float)
		# remove empty taxa (absent throughout)
		ind_keep = (np.sum(DATA,axis=1) != 0).nonzero()[0]
		DATA = DATA[ind_keep]
		

	time_series = DATA[0]
	obs_area_series = transform_Array_Tuple(DATA[1:])
	nTaxa = len(obs_area_series)
	print time_series
	print len(time_series),shape(obs_area_series)

	###### SIMULATE SAMPLING ######
	sampled_data=list()  #np.empty(nTaxa, dtype=object)
	OrigTimeIndex=np.zeros(nTaxa)
	for l in range(nTaxa):
		obs= obs_area_series[l]
		new_obs=[]
	
		for i in range(len(obs)): 
			if i == len(obs)-1: sampling_fraction =np.ones(2) # full sampling at the present
			else: sampling_fraction = RHO_sampling
		
			if len(obs[i]) == 0: new_obs.append(()) # if no areas it means taxon is extinct for good
		
			elif isnan(obs[i][0]): 
				#__ if species doesn't exist yet NaN (assuming origination time is known)
				new_obs.append((np.nan,)) 
				OrigTimeIndex[l] = i+1
				#__ new_obs.append(()) # if we assume that origination time is unknown
		
			else:
				sampling_fraction = sampling_fraction[np.array(obs[i])] # RHO[0] if area (0), RHO[1] if area (1), RHO[0,1] if area (0,1)
				if verbose ==1: print sampling_fraction, np.array(obs[i])
				r=np.random.uniform(0,1,len(obs[i]))
				ind=np.nonzero(r<sampling_fraction)[0]
				if len(ind)>0: new_obs.append(tuple(np.array(obs[i])[ind]))	
				else: new_obs.append(())
			#print i, obs[i], new_obs[i]
	
		new_obs2,NA=[],1
		for i in range(len(new_obs)): # add NaN until first observed appearance (i.e. not true time of origin)
			if NA==1:
				if new_obs[i]==() or isnan(new_obs[i][0]):
					 new_obs2.append((np.nan,))
				else: 
					new_obs2.append(new_obs[i])
					NA = 0
					OrigTimeIndex[l] = i
			else:
				new_obs2.append(new_obs[i])
				NA=0
		new_obs2[-1]=new_obs[-1] # present state is always sampled

		sampled_data.append(new_obs2)
		#_ print "TAXON" ,l
		#_ print new_obs
		#_ print new_obs2
		#_ print "ORIG",OrigTimeIndex[l], new_obs2[int(OrigTimeIndex[l])]
		
	# code data into binned time
	if n_sampled_bins > 0:
		Binned_time = sort(np.linspace(0,max(time_series),n_sampled_bins))
		#_ print Binned_time
		binned_DATA = list()
		binned_OrigTimeIndex=np.zeros(nTaxa)
		#
		for j in range(nTaxa):
			binned_states=list()
			samples_taxon_j = np.array(sampled_data[j])
			for i in range(1,n_sampled_bins):
				t1=Binned_time[i-1]
				t2=Binned_time[i]
				ind=np.intersect1d(np.nonzero(time_series>=t1)[0],np.nonzero(time_series<t2)[0])
				samples_taxon_j_time_t = samples_taxon_j[ind]
		
				if (0,1) in list(samples_taxon_j_time_t): 
					binned_states.append((0,1))
				elif (1,) in list(samples_taxon_j_time_t) and (0,) in list(samples_taxon_j_time_t): 
					binned_states.append((0,1))
				elif (1,) in list(samples_taxon_j_time_t): 
					binned_states.append((1,))
				elif (0,) in list(samples_taxon_j_time_t): 
					binned_states.append((0,))
				elif () in list(samples_taxon_j_time_t): 
					binned_states.append(())
				else:
					binned_states.append((nan,))
					binned_OrigTimeIndex[j] = i

			# add present state
			binned_states.append(samples_taxon_j[-1])
			binned_DATA.append(binned_states)

		#_ print len(binned_states),len(sampled_data[0]),len(Binned_time)
	        #_ 
		#_ print "\n",binned_states
		#_ 
		#_ print OrigTimeIndex.astype(int)
		#_ print binned_OrigTimeIndex.astype(int)
		binned_obs_area_series = np.array(binned_DATA)
		binned_OrigTimeIndex   = binned_OrigTimeIndex.astype(int)	
	else:
		print "Empirical bins"
		binned_obs_area_series = np.array(sampled_data)
		binned_OrigTimeIndex   = OrigTimeIndex.astype(int)
		Binned_time = time_series
		print Binned_time
					
	return nTaxa, Binned_time, binned_obs_area_series, binned_OrigTimeIndex

