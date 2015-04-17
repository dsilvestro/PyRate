#!/usr/bin/env python 
# Created by Daniele Silvestro on 02/03/2012 => dsilvestro@senckenberg.de 
import argparse, os,sys
from numpy import *
import numpy as np
import os, csv, glob
try: from biopy.bayesianStats import hpd as calcHPD
except(ImportError): pass
np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3)   # rounds all array elements to 3rd digit
import collections
from scipy import stats



def write_ts_te_table(path_dir, clade=0,burnin=0.1):
	# = infile
	#sys.path.append(infile)
	direct="%s/*mcmc.log" % path_dir
	files=glob.glob(direct)
	files=sort(files)
	print "found", len(files), "log files...\n"
	count=0

	name_file = os.path.splitext(os.path.basename(files[0]))[0]
	name_file = name_file.split("_mcmc")[0]
	
	outfile="%s/%s_se_est.txt" % (path_dir, name_file)
	newfile = open(outfile, "wb") 
	wlog=csv.writer(newfile, delimiter='\t')

	head="clade\tspecies"+ ("\tts\tte"*len(files))
	wlog.writerow(head.split('\t'))
	newfile.flush()

	for f in files:
		t_file=np.genfromtxt(f, delimiter='\t', dtype=None)
		input_file = os.path.basename(f)
		name_file = os.path.splitext(input_file)[0]
		path_dir = "%s/" % os.path.dirname(f)
		shape_f=list(shape(t_file))
		print "%s" % (name_file)
		
		if count==0:
			head = next(open(f)).split()
			w=[x for x in head if 'TS' in x]
			ind_ts0 = head.index(w[0])
			y=[x for x in head if 'TE' in x]
			ind_te0 = head.index(y[0])
		
		j=0
		out_list=list()
		if burnin<1: burnin = int(burnin*shape_f[0])
			
		for i in arange(ind_ts0,ind_te0):
			meanTS= mean(t_file[burnin:shape_f[0],i].astype(float))
			meanTE= mean(t_file[burnin:shape_f[0],ind_te0+j].astype(float))
			j+=1
			if count==0: out_list.append(array([clade, j, meanTS, meanTE]))
			else: out_list.append(array([meanTS, meanTE]))
		
			#print i-ind_ts0, array([meanTS,meanTE])

		out_list=array(out_list)
		if count==0: out_array=out_list
		else: out_array=np.hstack((out_array, out_list))
		count+=1
	#print shape(out_array)
	#print out_array[1:5,:]

	for i in range(len(out_array[:,0])):
		log_state=list(out_array[i,:])
		wlog.writerow(log_state)
		newfile.flush()


	newfile.close()
	print "\nFile saved as:", outfile


# import lib_utilities
# lib_utilities.write_ts_te_table("/Users/daniele/Desktop/try/tries",0)
