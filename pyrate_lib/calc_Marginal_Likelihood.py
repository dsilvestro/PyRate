#!/usr/bin/env python 
# Created by Daniele Silvestro on 18/12/2014
import argparse, os,sys
from numpy import *
import numpy as np
import os, csv, glob
np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3)   # rounds all array elements to 3rd digit
import collections

# PATH TO LOG FILES
print """\n\nEnter the path to the log files (from TI analyses).
Marginal likelihoods will be calculated for all files with extension '.log'"""
infile=raw_input("> ")
if infile.endswith(' ')==True: infile = infile[:-1]
path_dir = infile
sys.path.append(infile)
direct="%s/*.log" % infile # standard MCC or MCC2 VERSION
files=glob.glob(direct)
files=sort(files)
if len(files)==0:
	print "log files not found."
	quit()
else: print "found", len(files), "log files...\n"

# SET BURNIN (number of discarded samples)
print """Set burnin (default: 1000)"""
try: burnin=input("> ")
except: burnin=1000




for f in files:
	try: 
		t=loadtxt(f, skiprows=max(1,burnin))
		input_file = os.path.basename(f)
		name_file = os.path.splitext(input_file)[0]
		head = np.array(next(open(f)).split()) # should be faster
		try:
			like_index = np.where(head=="BD_lik")[0][0]
		except(IndexError): 
			like_index = np.where(head=="likelihood")[0][0] 
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
		for i in range(len(x)): 
			l[i]=mean(t[:,like_index][(t[:,temp_index]==x[i]).nonzero()[0]]) 
			s[i]=std(t[:,like_index][(t[:,temp_index]==x[i]).nonzero()[0]]) 
		mL=0
		for i in range(len(l)-1): mL+=((l[i]+l[i+1])/2.)*(x[i]-x[i+1]) # Beerli and Palczewski 2010
		print name_file
	#	print "temp:", x
	#	print "lik: ", l
	#	print "std: ", s
		print "Marginal likelihood:", mL, "\n"
	except:
		print "\n WARNING: cannot read file:", f, "\n\n"