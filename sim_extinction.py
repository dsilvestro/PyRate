#!/usr/bin/env python 
# Created by Daniele Silvestro on 20/01/2017 => daniele.silvestro@bioenv.gu.se
from numpy import *
import numpy as np
import sys, os
print "Birth-Death Sampler 18\n"


##########################################################################
###########                 SIMULATION SETTINGS                 ##########
##########################################################################

n_reps = 100 # number of simulations

# CONSTRAINTS on DATA SIZE (simulations will run until size requirements are met)
s_species=5827   # number of starting species
minSP=5827-351     # min size data set
maxSP=5827     # max size data set
minEX_SP=0    # minimum number of extinct lineages allowed

# SETTINGS for BD-SHIFT SIMULATIONS
root_age = 30
speciation_rates = [0] # if using rate shifts, the first rate is that closest to the root age
extinction_rates = np.array([1.8573897183238192e-08, 6.175032717446494e-08])*1000000/10. # 

scale=100.
print "iteration	ext_time_My	sampled_mu	n_extinct_species"
sim=0
ex_rates = np.random.uniform(extinction_rates[0],extinction_rates[1], n_reps)/scale

while sim != n_reps:
	ts=list()
	te=list()
	
	l,root=speciation_rates[0]/scale,-int(root_age*scale)
	m = ex_rates[sim]

	for i  in range(s_species): 
		ts.append(root)
		te.append(0)
	
	for t in range(root,0): # time
		TE=len(te)
		if TE>maxSP: 
			break
		for j in range(TE): # extant lineages
			if te[j]==0:
				ran=random.random()
				if ran<l: 
					te.append(0) # add species
					ts.append(t) # sp time
				elif ran>l and ran < (l+m): # extinction
					te[j]=t
		te_temp = np.array(te)
		if len(te_temp[te_temp==0]) == (5827-351): 
			print "%s\t%s\t%s\t%s" % (sim+1, -(root-t)/1000., m*scale, len(te_temp[te_temp<0]))
			sim +=1
			break
		if len(te_temp[te_temp==0]) < (5827-351):
			
			break
	te=array(te)
