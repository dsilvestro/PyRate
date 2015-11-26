#!/usr/bin/env python 
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



def write_ts_te_table(path_dir, tag="",clade=0,burnin=0.1):
	# = infile
	#sys.path.append(infile)
	direct="%s/*%s*mcmc.log" % (path_dir,tag)
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
		try:
			t_file=np.genfromtxt(f, delimiter='\t', dtype=None)
			input_file = os.path.basename(f)
			name_file = os.path.splitext(input_file)[0]
			path_dir = "%s/" % os.path.dirname(f)
			shape_f=list(shape(t_file))
			print "%s" % (name_file),
		
			#if count==0:
			head = next(open(f)).split()
			w=[x for x in head if 'TS' in x]
			ind_ts0 = head.index(w[0])
			y=[x for x in head if 'TE' in x]
			ind_te0 = head.index(y[0])
			print len(w), "species", np.shape(t_file)
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
		except: print "Could not read file:",name_file
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



def calc_marginal_likelihood(infile,burnin,extract_mcmc=1):
	sys.path.append(infile)
	direct="%s/*.log" % infile # PyRateContinuous
	files=glob.glob(direct)
	files=sort(files)
	if len(files)==0:
		print "log files not found."
		quit()
	else: print "found", len(files), "log files...\n"	
	out_file="%s/marginal_likelihoods.txt" % (infile)
	newfile =open(out_file,'wb')
	tbl_header = "file_name\treplicate\tmodel"
	for f in files:
		try: 
		#if 2>1:
			t=loadtxt(f, skiprows=max(1,burnin))
			input_file = os.path.basename(f)
			name_file = os.path.splitext(input_file)[0]
			try: replicate = int(name_file.split('_')[-5])
			except: replicate = 0
			model = name_file.split('_')[-1]
			head = np.array(next(open(f)).split()) # should be faster
			head_l = list(head)
			
			L_index = [head_l.index(i) for i in head_l if "lik" in i]
			
			for i in L_index: tbl_header+= ("\t"+head[i])
			
			if f==files[0]: newfile.writelines(tbl_header)
			
			
			#try:
			#	like_index = np.where(head=="BD_lik")[0][0]
			#except(IndexError): 
			#	like_index = np.where(head=="likelihood")[0][0] 
			try: 
				temp_index = np.where(head=="temperature")[0][0]
			except(IndexError): 
				temp_index = np.where(head=="beta")[0][0]
	
			z=t[:,temp_index] # list of all temps
			y=collections.Counter(z)
			x=list(y) # list of unique values
			x=sort(x)[::-1]
			l=zeros(len(x))
			s=zeros(len(x))
			l_str,s_str="",""
			for like_index in L_index:
				for i in range(len(x)): 
					l[i]=mean(t[:,like_index][(t[:,temp_index]==x[i]).nonzero()[0]]) 
					s[i]=std(t[:,like_index][(t[:,temp_index]==x[i]).nonzero()[0]]) 
					#l_str += "\t%s" % round(l[i],3)
					#s_str += "\t%s" % round(s[i],3)
				mL=0
				for i in range(len(l)-1): mL+=((l[i]+l[i+1])/2.)*(x[i]-x[i+1]) # Beerli and Palczewski 2010
				#ml_str="\n%s\t%s\t%s\t%s%s%s" % (name_file,round(mL,3),replicate,model    ,l_str,s_str)
				if like_index == min(L_index): ml_str="\n%s\t%s\t%s\t%s" % (name_file,replicate,model,round(mL,3))
				else: ml_str += "\t%s" % (round(mL,3))
			newfile.writelines(ml_str)
			
			if extract_mcmc==1:
				out_name="%s/%s_cold.log" % (infile,name_file)
				newfile2 =open(out_name,'wb')
				
				t_red = t[(t[:,temp_index]==1).nonzero()[0]]
				newfile2.writelines( "\t".join(head_l)  )
				for l in range(shape(t_red)[0]):
					line= ""
					for i in t_red[l,:]: line+= "%s\t" % (i)
					newfile2.writelines( "\n"+line  )
				#
				#l[i]=mean(t[:,like_index][(t[:,temp_index]==x[i]).nonzero()[0]]) 
			        #
				newfile2.close()
			
			
			
			
			sys.stdout.write(".")
			sys.stdout.flush()
		except:
			print "\n WARNING: cannot read file:", f, "\n\n"

	newfile.close()
