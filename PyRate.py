#!/usr/bin/env python
# Created by Daniele Silvestro on 02/03/2012 => pyrate.help@gmail.com
import argparse, os,sys, platform, time, csv, glob
import random as rand
import warnings, imp

version= "PyRate"
build  = "v2.0 - 20191013"
if platform.system() == "Darwin": sys.stdout.write("\x1b]2;%s\x07" % version)

citation= """Silvestro, D., Schnitzler, J., Liow, L.H., Antonelli, A. and Salamin, N. (2014)
Bayesian Estimation of Speciation and Extinction from Incomplete Fossil
Occurrence Data. Systematic Biology, 63, 349-367.

Silvestro, D., Salamin, N., Schnitzler, J. (2014)
PyRate: A new program to estimate speciation and extinction rates from
incomplete fossil record. Methods in Ecology and Evolution, 5, 1126-1131.
"""
print("""
                 %s - %s

          Bayesian estimation of origination,
           extinction and preservation rates
              from fossil occurrence data

                 pyrate.help@gmail.com

\n""" % (version, build))
# check python version
V=list(sys.version_info[0:3])
if V[0]>2: sys.exit("""\nPyRate currently runs only under python 2. Python 3.X is currently not supported.
	You can download to python 2.7 at: https://www.python.org/downloads/""")

# LOAD LIBRARIES
try:
	import argparse
except(ImportError):
	sys.exit("""\nError: argparse library not found.
	You can upgrade to python 2.7 at: https://www.python.org/downloads/
	or install argparse at: https://code.google.com/p/argparse/ \n""")

try:
	from numpy import *
	import numpy as np
except(ImportError):
	sys.exit("\nError: numpy library not found.\nYou can download numpy at: http://sourceforge.net/projects/numpy/files/ \n")

try:
	import scipy
	from scipy.special import gamma
	from scipy.special import beta as f_beta
	from scipy.special import gdtr, gdtrix
	from scipy.special import betainc
	import scipy.stats
	from scipy.optimize import fmin_powell as Fopt1
except(ImportError):
	sys.exit("\nError: scipy library not found.\nYou can download scipy at: http://sourceforge.net/projects/scipy/files/ \n")

try:
	import multiprocessing, thread
	import multiprocessing.pool
	class NoDaemonProcess(multiprocessing.Process):
		# make 'daemon' attribute always return False
		def _get_daemon(self): return False
		def _set_daemon(self, value): pass
		daemon = property(_get_daemon, _set_daemon)

	class mcmcMPI(multiprocessing.pool.Pool):
		# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
		# because the latter is only a wrapper function, not a proper class.
		Process = NoDaemonProcess
	use_seq_lik= 0
	if platform.system() == "Windows" or platform.system() == "Microsoft": use_seq_lik= 1
except(ImportError):
	print("\nWarning: library multiprocessing not found.\nPyRate will use (slower) sequential likelihood calculation. \n")
	use_seq_lik= 1

if platform.system() == "Windows" or platform.system() == "Microsoft": use_seq_lik= 1

version_details="PyRate %s; OS: %s %s; Python version: %s; Numpy version: %s; Scipy version: %s" \
% (build, platform.system(), platform.release(), sys.version, np.version.version, scipy.version.version)

### numpy print options ###
np.set_printoptions(suppress= 1) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit

original_stderr = sys.stderr
NO_WARN = original_stderr #open('pyrate_warnings.log', 'w')
small_number= 1e-50

def get_self_path():
	self_path = -1
	path_list = [os.path.dirname(sys.argv[0]) , os.getcwd()]
	for path in path_list:
		try:
			self_path=path
			lib_updates_priors = imp.load_source("lib_updates_priors", "%s/pyrate_lib/lib_updates_priors.py" % (self_path))
			break
		except:
			self_path = -1
	if self_path== -1:
		print os.getcwd(), os.path.dirname(sys.argv[0])
		sys.exit("pyrate_lib not found.\n")
	return self_path


# Search for the module
hasFoundPyRateC = 0
try:
	self_path = get_self_path()
	if platform.system()=="Darwin": os_spec_lib="macOS"
	elif platform.system() == "Windows" or platform.system() == "Microsoft": os_spec_lib="Windows"
	else: os_spec_lib = "Other"

	c_lib_path = "pyrate_lib/fastPyRateC/%s" % (os_spec_lib)
	sys.path.append(os.path.join(self_path,c_lib_path))
	#print self_path, sys.path

	from _FastPyRateC import PyRateC_BD_partial_lik, PyRateC_HOMPP_lik, PyRateC_setFossils, \
						   PyRateC_getLogGammaPDF, PyRateC_initEpochs, PyRateC_HPP_vec_lik, \
													 PyRateC_NHPP_lik, PyRateC_FBD_T4
	hasFoundPyRateC = 1
	print("Module FastPyRateC was loaded.")
  # Set that to true to enable sanity check (comparing python and c++ results)
	sanityCheckForPyRateC = 0
	sanityCheckThreshold = 1e-10
	if sanityCheckForPyRateC == 1:
		print "Sanity check for FastPyRateC is enabled."
		print "Python and C results will be compared and any divergence greater than ", sanityCheckThreshold, " will be reported."
except:
	print("Module FastPyRateC was not found.")
	hasFoundPyRateC = 0
	sanityCheckForPyRateC = 0

########################## CALC PROBS ##############################
def calcHPD(data, level) :
	assert (0 < level < 1)
	d = list(data)
	d.sort()
	nData = len(data)
	nIn = int(round(level * nData))
	if nIn < 2 :
		raise RuntimeError("not enough data")
	i = 0
	r = d[i+nIn-1] - d[i]
	for k in range(len(d) - (nIn - 1)) :
		rk = d[k+nIn-1] - d[k]
		if rk < r :
			r = rk
			i = k
	assert 0 <= i <= i+nIn-1 < len(d)
	return (d[i], d[i+nIn-1])

def check_burnin(b,I):
	#print b, I
	if b<1: burnin=int(b*I)
	else: burnin=int(b)
	if burnin>=(I-10):
		print("Warning: burnin too high! Excluding 10% instead.")
		burnin=int(0.1*I)
	return burnin

def calc_model_probabilities(f,burnin):
	print("parsing log file...\n")
	t=loadtxt(f, skiprows=1)
	num_it=shape(t)[0]
	if num_it<10: sys.exit("\nNot enough samples in the log file!\n")
	burnin=check_burnin(burnin, num_it)
	print("First %s samples excluded as burnin.\n" % (burnin))
	file1=file(f, 'U')
	L=file1.readlines()
	head= L[0].split()
	PAR1=["k_birth","k_death"]
	k_ind= [head.index(s) for s in head if s in PAR1]
	if len(k_ind)==0: k_ind =[head.index(s) for s in head if s in ["K_l","K_m"]]
	z1=t[burnin:,k_ind[0]]  # list of shifts (lambda)
	z2=t[burnin:,k_ind[1]]  # list of shifts (mu)
	y1= max(max(z1),max(z2))
	print("Model		   Probability")
	print("		  Speciation  Extinction")
	for i in range(1,int(y1)+1):
		k_l=float(len(z1[z1==i]))/len(z1)
		k_m=float(len(z2[z2==i]))/len(z2)
		print("%s-rate	%s	  %s" % (i,round(k_l,4),round(k_m,4)))
	print("\n")

	try:
		import collections,os
		d = collections.OrderedDict()

		def count_BD_config_freq(A):
			for a in A:
				t = tuple(a)
				if t in d: d[t] += 1
				else: d[t] = 1

			result = []
			for (key, value) in d.items(): result.append(list(key) + [value])
			return result

		BD_config = t[burnin:,np.array(k_ind)]
		B = np.asarray(count_BD_config_freq(BD_config))
		B[:,2]=B[:,2]/sum(B[:,2])
		B = B[B[:,2].argsort()[::-1]]
		cum_prob = np.cumsum(B[:,2])
		print "Best BD/ID configurations (rel.pr >= 0.05)"
		print "   B/I	D	  Rel.pr"
		print B[(np.round(B[:,2],3)>=0.05).nonzero()[0],:]

	except: pass
	quit()

def calc_ts_te(f, burnin):
	if f=="null": return FA,LO
	else:
		t_file=np.genfromtxt(f, delimiter='\t', dtype=None)
		shape_f=list(shape(t_file))
		if len(shape_f)==1: sys.exit("\nNot enough samples in the log file!\n")
		if shape_f[1]<10: sys.exit("\nNot enough samples in the log file!\n")

		ind_start=np.where(t_file[0]=="tot_length")[0][0]
		indexes= np.array([ind_start+1, ind_start+(shape_f[1]-ind_start)/2])
		# fixes the case of missing empty column at the end of each row
		if t_file[0,-1] != "False":
			indexes = indexes+np.array([0,1])
		ind_ts0 = indexes[0]
		ind_te0 = indexes[1]

		meanTS,meanTE=list(),list()
		burnin=check_burnin(burnin, shape_f[0])
		burnin+=1
		j=0
		for i in arange(ind_ts0,ind_te0):
			meanTS.append(mean(t_file[burnin:shape_f[0],i].astype(float)))
			meanTE.append(mean(t_file[burnin:shape_f[0],ind_te0+j].astype(float)))
			j+=1
		return array(meanTS),array(meanTE)

def calc_BF(f1, f2):
	input_file_raw = [os.path.basename(f1),os.path.basename(f2)]
	def get_ML(FILE):
		file1=file(FILE, 'U')
		L=file1.readlines()
		for i in range(len(L)):
			if "Marginal likelihood" in L[i]:
				x=L[i].split("Marginal likelihood: ")
				ML= float(x[1])
				return ML
	BF= 2*(get_ML(f1)-get_ML(f2))
	if abs(BF)<2: support="negligible"
	elif abs(BF)<6: support="positive"
	elif abs(BF)<10: support="strong"
	else: support="very strong"
	if BF>0: best=0
	else: best=1
	print("\nModel A: %s\nModelB: %s" % (input_file_raw[best],input_file_raw[abs(best-1)]))
	print("\nModel A received %s support against Model B\nBayes Factor: %s\n\n" % (support, round(abs(BF), 4)))



def calc_BFlist(f1):
	#f1 = os.path.basename(f1)
	#print f1
	tbl = np.genfromtxt(f1,"str")
	file_list = tbl[:,tbl[0]=="file_name"]

	f_list, l_list = list(), list()

	for i in range(1, len(file_list)):
		fn = str(file_list[i][0])
		f_list.append(fn)
		l_list.append(float(tbl[i,tbl[0]=="likelihood"][0]))

	ml = l_list[l_list.index(max(l_list))]

	l_list = np.array(l_list)
	bf = 2*(ml - l_list)

	print "Found %s models:" % (len(bf))
	for i in range(len(bf)): print "model %s: '%s'" % (i,f_list[i])

	print "\nBest model:", f_list[(bf==0).nonzero()[0][0]], ml, "\n"


	for i in range(len(bf)):
		BF = bf[i]
		if abs(BF)==0: pass
		else:
			if abs(BF)<2: support="negligible"
			elif abs(BF)<6: support="positive"
			elif abs(BF)<10: support="strong"
			else: support="very strong"
			print "Support in favor of model %s: %s (%s)" % (i, BF, support)

def get_DT(T,s,e): # returns the Diversity Trajectory of s,e at times T (x10 faster)
	B=np.sort(np.append(T,T[0]+1))+.000001 # the + .0001 prevents problems with identical ages
	ss1 = np.histogram(s,bins=B)[0]
	ee2 = np.histogram(e,bins=B)[0]
	DD=(ss1-ee2)[::-1]
	#return np.insert(np.cumsum(DD),0,0)[0:len(T)]
	return np.cumsum(DD)[0:len(T)]


########################## PLOT RTT ##############################
def plot_RTT(infile,burnin, file_stem="",one_file= 0, root_plot=0,plot_type=1):
	burnin = int(burnin)
	if burnin<=1:
		print("Burnin must be provided in terms of number of samples to be excluded.")
		print("E.g. '-b 100' will remove the first 100 samples.")
		print("Assuming burnin = 1.\n")
	def print_R_vec(name,v):
		new_v=[]
		for j in range(0,len(v)):
			value=v[j]
			if isnan(v[j]): value="NA"
			new_v.append(value)

		vec="%s=c(%s, " % (name,new_v[0])
		for j in range(1,len(v)-1): vec += "%s," % (new_v[j])
		vec += "%s)"  % (new_v[j+1])
		return vec


	path_dir = infile
	sys.path.append(infile)
	plot_title = file_stem.split('_')[0]
	print "FILE STEM:",file_stem, plot_title
	if file_stem=="": direct="%s/*_marginal_rates.log" % infile
	else: direct="%s/*%s*marginal_rates.log" % (infile,file_stem)
	files=glob.glob(direct)
	files=sort(files)
	if one_file== 1: files=["%s/%smarginal_rates.log" % (infile,file_stem)]

	stem_file=files[0]
	name_file = os.path.splitext(os.path.basename(stem_file))[0]

	wd = "%s" % os.path.dirname(stem_file)
	#print(name_file, wd)
	print "found", len(files), "log files...\n"

	########################################################
	######		   DETERMINE MIN ROOT AGE		   ######
	########################################################
	if root_plot==0:
		min_age=np.inf
		print "determining min age...",
		for f in files:
			file_name =  os.path.splitext(os.path.basename(f))[0]
			sys.stdout.write(".")
			sys.stdout.flush()
			head = next(open(f)).split() # should be faster
			sp_ind= [head.index(s) for s in head if "l_" in s]
			min_age=min(min_age,len(sp_ind))

		print "Min root age:", min_age
		max_ind=min_age-1
	else: max_ind = int(root_plot-1)

	print max_ind, root_plot
	########################################################
	######			COMBINE ALL LOG FILES		   ######
	########################################################
	print "\ncombining all files...",
	file_n=0
	for f in files:
		file_name =  os.path.splitext(os.path.basename(f))[0]
		print file_name
		try:
			t=loadtxt(f, skiprows=max(1,burnin))
			sys.stdout.write(".")
			sys.stdout.flush()
			head = next(open(f)).split()
			l_ind= [head.index(s) for s in head if "l_" in s]
			m_ind= [head.index(s) for s in head if "m_" in s]
			r_ind= [head.index(s) for s in head if "r_" in s]
			l_ind=l_ind[0:max_ind]
			m_ind=m_ind[0:max_ind]
			r_ind=r_ind[0:max_ind]

			if file_n==0:
				L_tbl=t[:,l_ind]
				M_tbl=t[:,m_ind]
				R_tbl=t[:,r_ind]
				file_n=1
				#if np.min([np.max(L_tbl),np.max(M_tbl)])>0.1: no_decimals = 3
				#elif np.min([np.max(L_tbl),np.max(M_tbl)])>0.01: no_decimals = 5
				#else:
				no_decimals = 15
			else:
				L_tbl=np.concatenate((L_tbl,t[:,l_ind]),axis=0)
				M_tbl=np.concatenate((M_tbl,t[:,m_ind]),axis=0)
				R_tbl=np.concatenate((R_tbl,t[:,r_ind]),axis=0)
		except:
			print "skipping file:", f

	########################################################
	######			   CALCULATE HPDs			   ######
	########################################################
	print "\ncalculating HPDs...",
	def get_HPD(threshold=.95):
		L_hpd_m,L_hpd_M=[],[]
		M_hpd_m,M_hpd_M=[],[]
		R_hpd_m,R_hpd_M=[],[]
		sys.stdout.write(".")
		sys.stdout.flush()
		for time_ind in range(shape(L_tbl)[1]):
			hpd1=np.around(calcHPD(L_tbl[:,time_ind],threshold),decimals=no_decimals)
			hpd2=np.around(calcHPD(M_tbl[:,time_ind],threshold),decimals=no_decimals)
			if len(r_ind)>0:
				hpd3=np.around(calcHPD(R_tbl[:,time_ind],threshold),decimals=no_decimals)
			else:
				hpd3 =  hpd1- hpd2
			L_hpd_m.append(hpd1[0])
			L_hpd_M.append(hpd1[1])
			M_hpd_m.append(hpd2[0])
			M_hpd_M.append(hpd2[1])
			R_hpd_m.append(hpd3[0])
			R_hpd_M.append(hpd3[1])

		return [L_hpd_m,L_hpd_M,M_hpd_m,M_hpd_M,R_hpd_m,R_hpd_M]

	def get_CI(threshold=.95):
		threshold = (1-threshold)/2.
		L_hpd_m,L_hpd_M=[],[]
		M_hpd_m,M_hpd_M=[],[]
		R_hpd_m,R_hpd_M=[],[]
		sys.stdout.write(".")
		sys.stdout.flush()
		for time_ind in range(shape(R_tbl)[1]):
			l=np.sort(L_tbl[:,time_ind])
			m=np.sort(M_tbl[:,time_ind])
			r=np.sort(R_tbl[:,time_ind])
			hpd1=np.around(np.array([l[int(threshold*len(l))] , l[int(len(l) - threshold*len(l))] ]),decimals=no_decimals)
			hpd2=np.around(np.array([m[int(threshold*len(m))] , m[int(len(m) - threshold*len(m))] ]),decimals=no_decimals)
			hpd3=np.around(np.array([r[int(threshold*len(r))] , r[int(len(r) - threshold*len(r))] ]),decimals=no_decimals)

			L_hpd_m.append(hpd1[0])
			L_hpd_M.append(hpd1[1])
			M_hpd_m.append(hpd2[0])
			M_hpd_M.append(hpd2[1])
			R_hpd_m.append(hpd3[0])
			R_hpd_M.append(hpd3[1])
		return [L_hpd_m,L_hpd_M,M_hpd_m,M_hpd_M,R_hpd_m,R_hpd_M]



	hpds95 =  np.array(get_HPD(threshold=.95))
	hpds50 =  np.array(get_CI(threshold=.50))
	#hpds10 =  get_CI(threshold=.10)

	L_tbl_mean=np.around(np.mean(L_tbl,axis=0),no_decimals)
	M_tbl_mean=np.around(np.mean(M_tbl,axis=0),no_decimals)
	if len(r_ind)>0: R_tbl_mean=np.around(np.mean(R_tbl,axis=0),no_decimals)
	else:
		R_tbl_mean= L_tbl_mean-M_tbl_mean
	mean_rates=np.array([L_tbl_mean,L_tbl_mean,M_tbl_mean,M_tbl_mean,R_tbl_mean,R_tbl_mean] )

	nonzero_rate = L_tbl_mean+ M_tbl_mean
	NA_ind = (nonzero_rate==0).nonzero()[0]

	hpds95[:,NA_ind] = np.nan
	#hpds50[:,NA_ind] = np.nan
	print mean_rates
	mean_rates[:,NA_ind] = np.nan
	print "HPD", hpds95
	#print(np.shape(np.array(hpds50)	), np.shape(L_tbl_mean))

	########################################################
	######				  PLOT RTTs				 ######
	########################################################
	print "\ngenerating R file...",
	out="%s/%s_RTT.r" % (wd,name_file)
	newfile = open(out, "wb")
	Rfile="# %s files combined:\n" % (len(files))
	for f in files: Rfile+="# \t%s\n" % (f)
	Rfile+= """\n# 95% HPDs calculated using code from Biopy (https://www.cs.auckland.ac.nz/~yhel002/biopy/)"""

	if plot_type==1: n_plots=4
	else: n_plots=3

	if platform.system() == "Windows" or platform.system() == "Microsoft":
		wd_forward = os.path.abspath(wd).replace('\\', '/')
		Rfile+= "\n\npdf(file='%s/%s_RTT.pdf',width=10.8, height=8.4)\npar(mfrow=c(2,2))" % (wd_forward,name_file)
	else:
		Rfile+= "\n\npdf(file='%s/%s_RTT.pdf',width=10.8, height=8.4)\npar(mfrow=c(2,2))" % (wd,name_file)

	Rfile+= "\nlibrary(scales)"

	if plot_type==2: Rfile+= """\nplot_RTT <- function (age,hpd_M,hpd_m,mean_m,color){
	N=100
	beta=(1:(N-1))/N
	alpha_shape=0.25
	cat=1-(beta^(1./alpha_shape))
	for (i in 1:(N-1)){
		trans= 1/N + 2/N
		polygon(c(age, rev(age)), c(hpd_M-((hpd_M-mean_m)*cat[i]), rev(hpd_m+((mean_m-hpd_m)*cat[i]))), col = alpha(color,trans), border = NA)
	}
	lines(rev(age), rev(mean_m), col = color, lwd=3)\n}
	"""

	def RTT_plot_in_R(args, alpha):
		count=0
		data=""

		name=['95','_mean'] # ,'50'
		for hpd_list in args:
			sys.stdout.write(".")
			sys.stdout.flush()
			[L_hpd_m,L_hpd_M,M_hpd_m,M_hpd_M,R_hpd_m,R_hpd_M]=hpd_list
			if name[count]=="_mean":
				data += print_R_vec('\nL_mean',L_hpd_m)
				data += print_R_vec('\nM_mean',M_hpd_m)
				data += print_R_vec('\nR_mean',R_hpd_m)
			else:
				data += print_R_vec('\nL_hpd_m%s',L_hpd_m) % name[count]
				data += print_R_vec('\nL_hpd_M%s',L_hpd_M) % name[count]
				data += print_R_vec('\nM_hpd_m%s',M_hpd_m) % name[count]
				data += print_R_vec('\nM_hpd_M%s',M_hpd_M) % name[count]
				data += print_R_vec('\nR_hpd_m%s',R_hpd_m) % name[count]
				data += print_R_vec('\nR_hpd_M%s',R_hpd_M) % name[count]
			if count==0:
				max_x_axis,min_x_axis = -len(L_hpd_m), 0 # root to the present
 				max_x_axis,min_x_axis = -(len(L_hpd_m)+.05*len(L_hpd_m)), -(len(L_hpd_m)-len(L_hpd_m[np.isfinite(L_hpd_m)]))+.05*len(L_hpd_m)
				plot_L = "\ntrans=%s\nage=(0:(%s-1))* -1" % (alpha, len(L_hpd_m))
				plot_L += "\nplot(age,age,type = 'n', ylim = c(%s, %s), xlim = c(%s,%s), ylab = 'Speciation rate', xlab = 'Ma',main='%s' )" \
					% (0,1.1*np.nanmax(L_hpd_M),max_x_axis,min_x_axis,plot_title)
				plot_M  = "\nplot(age,age,type = 'n', ylim = c(%s, %s), xlim = c(%s,%s), ylab = 'Extinction rate', xlab = 'Ma' )" \
					% (0,1.1*np.nanmax(M_hpd_M),max_x_axis,min_x_axis)
				plot_R  = "\nplot(age,age,type = 'n', ylim = c(%s, %s), xlim = c(%s,%s), ylab = 'Net diversification rate', xlab = 'Ma' )" \
					% (-abs(1.1*np.nanmin(R_hpd_m)),1.1*np.nanmax(R_hpd_M),max_x_axis,min_x_axis)
				plot_R += """\nabline(h=0,lty=2,col="darkred")""" # \nabline(v=-c(65,200,251,367,445),lty=2,col="darkred")

			if name[count]=="_mean":
				plot_L += """\nlines(rev(age), rev(L_mean), col = "#4c4cec", lwd=3)"""
				plot_M += """\nlines(rev(age), rev(M_mean), col = "#e34a33", lwd=3)"""
				plot_R += """\nlines(rev(age), rev(R_mean), col = "#504A4B", lwd=3)"""
			else:
				if plot_type==1:
					plot_L += """\npolygon(c(age, rev(age)), c(L_hpd_M%s, rev(L_hpd_m%s)), col = alpha("#4c4cec",trans), border = NA)""" % (name[count],name[count])
					plot_M += """\npolygon(c(age, rev(age)), c(M_hpd_M%s, rev(M_hpd_m%s)), col = alpha("#e34a33",trans), border = NA)""" % (name[count],name[count])
					plot_R += """\npolygon(c(age, rev(age)), c(R_hpd_M%s, rev(R_hpd_m%s)), col = alpha("#504A4B",trans), border = NA)""" % (name[count],name[count])
				elif plot_type==2:
					plot_L += """\nplot_RTT(age,L_hpd_M95,L_hpd_m95,L_mean,"#4c4cec")"""
					plot_M += """\nplot_RTT(age,M_hpd_M95,M_hpd_m95,M_mean,"#e34a33")"""
					plot_R += """\nplot_RTT(age,R_hpd_M95,R_hpd_m95,R_mean,"#504A4B")"""

			count+=1

		R_code=data+plot_L+plot_M+plot_R

		if plot_type==1:
			R_code += "\nplot(age,rev(1/M_mean),type = 'n', xlim = c(%s,%s), ylab = 'Longevity (Myr)', xlab = 'Ma' )" % (max_x_axis,min_x_axis)
			R_code += """\nlines(rev(age), rev(1/M_mean), col = "#504A4B", lwd=3)"""
			#R_code += """\npolygon(c(age, rev(age)), c((1/M_hpd_m95), rev(1/M_hpd_M95)), col = alpha("#504A4B",trans), border = NA)"""

		return R_code

	Rfile += RTT_plot_in_R([hpds95,mean_rates],.5) # ,hpds50

	Rfile += "\nn <- dev.off()"
	newfile.writelines(Rfile)
	newfile.close()
	print "\nAn R script with the source for the RTT plot was saved as: %s_RTT.r\n(in %s)" % (name_file, wd)
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		cmd="cd %s & Rscript %s_RTT.r" % (wd,name_file)
	else:
		cmd="cd %s; Rscript %s/%s_RTT.r" % (wd,wd,name_file)
	os.system(cmd)
	print "done\n"


def plot_ltt(tste_file,plot_type=1,rescale= 1,step_size=1.): # change rescale to change bin size
	# plot_type=1 : ltt + min/max range
	# plot_type=2 : log10 ltt + min/max range
	#step_size=int(step_size)
	# read data
	print "Processing data..."
	tbl = np.loadtxt(tste_file,skiprows=1)
	j_max=(np.shape(tbl)[1]-1)/2
	j_range=np.arange(j_max)
	ts = tbl[:,2+2*j_range]*rescale
	te = tbl[:,3+2*j_range]*rescale
	time_vec = np.sort(np.linspace(np.min(te),np.max(ts),int((np.max(ts)-np.min(te))/float(step_size)) ))

	# create out file
	wd = "%s" % os.path.dirname(tste_file)
	out_file_name = os.path.splitext(os.path.basename(tste_file))[0]
	out_file="%s/%s" % (wd,out_file_name+"_ltt.txt")
	ltt_file = open(out_file , "w",0)
	ltt_log=csv.writer(ltt_file, delimiter='\t')


	# calc ltt
	print time_vec
	dtraj = []
	for rep in j_range:
		sys.stdout.write(".")
		sys.stdout.flush()
		dtraj.append(get_DT(time_vec,ts[:,rep],te[:,rep])[::-1])
	dtraj = np.array(dtraj)
	div_mean = np.mean(dtraj,axis=0)
	div_m	= np.min(dtraj,axis=0)
	div_M	= np.max(dtraj,axis=0)

	Ymin,Ymax,yaxis = 0,max(div_M)+1,""
	if min(div_m)>5: Ymin = min(div_m)-1

	if plot_type==2:
		div_mean = log10(div_mean)
		div_m	= log10(div_m   )
		div_M	= log10(div_M   )
		Ymin,Ymax,yaxis = min(div_m),max(div_M), " (Log10)"

	# write to file
	if plot_type==1 or plot_type==2:
		ltt_log.writerow(["time","diversity","m_div","M_div"])
		for i in range(len(time_vec)):
			ltt_log.writerow([time_vec[i]/rescale,div_mean[i],div_m[i],div_M[i]])
		ltt_file.close()
		plot2 = """polygon(c(time, rev(time)), c(tbl$M_div, rev(tbl$m_div)), col = alpha("#504A4B",0.5), border = NA)"""

	# write multiple LTTs to file
	if plot_type==3:
		header = ["time","diversity"]+["rep%s" % (i) for i in j_range]
		ltt_log.writerow(header)
		plot2=""
		for i in range(len(time_vec)):
			d = dtraj[:,i]
			ltt_log.writerow([time_vec[i]/rescale,div_mean[i]]+list(d))
			plot2 += """\nlines(time,tbl$rep%s, type="l",lwd = 1,col = alpha("#504A4B",0.5))""" % (i)
		ltt_file.close()

	###### R SCRIPT
	R_file_name="%s/%s" % (wd,out_file_name+"_ltt.R")
	R_file=open(R_file_name, "wb")
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		tmp_wd = os.path.abspath(wd).replace('\\', '/')
	else: tmp_wd = wd
	R_script = """
	setwd("%s")
	tbl = read.table(file = "%s_ltt.txt",header = T)
	pdf(file='%s_ltt.pdf',width=12, height=9)
	time = -tbl$time
	library(scales)
	plot(time,tbl$diversity, type="n",ylab= "Number of lineages%s", xlab="Time (Ma)", main="Range-through diversity through time", ylim=c(%s,%s),xlim=c(min(time),0))
	%s
	lines(time,tbl$diversity, type="l",lwd = 2)
	n<-dev.off()
	""" % (tmp_wd, out_file_name,out_file_name, yaxis, Ymin,Ymax,plot2)

	R_file.writelines(R_script)
	R_file.close()
	print "\nAn R script with the source for the stat plot was saved as: \n%s" % (R_file_name)
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		cmd="cd %s & Rscript %s" % (wd,out_file_name+"_ltt.R")
	else:
		cmd="cd %s; Rscript %s" % (wd,out_file_name+"_ltt.R")
	os.system(cmd)
	sys.exit("done\n")




########################## PLOT TS/TE STAT ##############################
def plot_tste_stats(tste_file, EXT_RATE, step_size,no_sim_ex_time,burnin,rescale,ltt_only=1):
	step_size=int(step_size)
	# read data
	print "Processing data..."
	tbl = np.loadtxt(tste_file,skiprows=1)
	j_max=(np.shape(tbl)[1]-1)/2
	j=np.arange(j_max)
	ts = tbl[:,2+2*j]*rescale
	te = tbl[:,3+2*j]*rescale
	root = int(np.max(ts)+1)

	if EXT_RATE==0:
		EXT_RATE = len(te[te>0])/sum(ts-te) # estimator for overall extinction rate
		print "estimated extinction rate:", EXT_RATE

	wd = "%s" % os.path.dirname(tste_file)
	# create out file
	out_file_name = os.path.splitext(os.path.basename(tste_file))[0]
	out_file="%s/%s" % (wd,out_file_name+"_stats.txt")
	out_file=open(out_file, "wb")

	out_file.writelines("time\tdiversity\tm_div\tM_div\tmedian_age\tm_age\tM_age\tturnover\tm_turnover\tM_turnover\tlife_exp\tm_life_exp\tM_life_exp\t")

	no_sim_ex_time = int(no_sim_ex_time)
	def draw_extinction_time(te,EXT_RATE):
		te_mod = np.zeros(np.shape(te))
		ind_extant = (te[:,0]==0).nonzero()[0]
		te_mod[ind_extant,:] = -np.random.exponential(1/EXT_RATE,(len(ind_extant),len(te[0]))) # sim future extinction
		te_mod += te
		return te_mod

	def calc_median(arg):
		if len(arg)>1: return np.median(arg)
		else: return np.nan

	extant_at_time_t_previous = [0]
	for i in range(0,root+1,step_size):
		time_t = root-i
		up = time_t+step_size
		lo = time_t
		extant_at_time_t = [np.intersect1d((ts[:,rep] >= lo).nonzero()[0], (te[:,rep] <= up).nonzero()[0]) for rep in j]
		extinct_in_time_t =[np.intersect1d((te[:,rep] >= lo).nonzero()[0], (te[:,rep] <= up).nonzero()[0]) for rep in j]
		diversity = [len(extant_at_time_t[rep]) for rep in j]
		try:
			#turnover = [1-len(np.intersect1d(extant_at_time_t_previous[rep],extant_at_time_t[rep]))/float(len(extant_at_time_t[rep])) for rep in j]
			turnover = [(len(extant_at_time_t[rep])-len(np.intersect1d(extant_at_time_t_previous[rep],extant_at_time_t[rep])))/float(len(extant_at_time_t[rep])) for rep in j]
		except:
			turnover = [np.nan for rep in j]

		if min(diversity)<=1:
			age_current_taxa = [np.nan for rep in j]
		else:
			ext_age = [calc_median(ts[extinct_in_time_t[rep],rep]-te[extinct_in_time_t[rep],rep]) for rep in j]
			age_current_taxa = [calc_median(ts[extant_at_time_t[rep],rep]-time_t) for rep in j]

		# EMPIRICAL/PREDICTED LIFE EXPECTANCY
		life_exp=list()
		try:
			ex_rate = [float(EXT_RATE)]
			r_ind = np.repeat(0,no_sim_ex_time)
		except(ValueError):
			t=loadtxt(EXT_RATE, skiprows=max(1,int(burnin)))
			head = next(open(EXT_RATE)).split()
			m_ind= [head.index(s) for s in head if "m_0" in s]
			ex_rate= [mean(t[:,m_ind])]
			r_ind = np.random.randint(0,len(ex_rate),no_sim_ex_time)

		if min(diversity)<=1:
			life_exp.append([np.nan for rep in j])
		else:
			for sim in range(no_sim_ex_time):
				#print ex_rate[r_ind[sim]]
				te_mod = draw_extinction_time(te,ex_rate[r_ind[sim]])
				te_t = [te_mod[extant_at_time_t[rep],:] for rep in j]
				life_exp.append([median(time_t-te_t[rep]) for rep in j])

		life_exp= np.array(life_exp)
		STR= "\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" \
		% (time_t, median(diversity),min(diversity),max(diversity),
		median(age_current_taxa),min(age_current_taxa),max(age_current_taxa),
		median(turnover),min(turnover),max(turnover),
		median(life_exp),np.min(life_exp),np.max(life_exp))
		extant_at_time_t_previous = extant_at_time_t
		STR = STR.replace("nan","NA")
		sys.stdout.write(".")
		sys.stdout.flush()
		out_file.writelines(STR)
	out_file.close()

	###### R SCRIPT
	R_file_name="%s/%s" % (wd,out_file_name+"_stats.R")
	R_file=open(R_file_name, "wb")
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		tmp_wd = os.path.abspath(wd).replace('\\', '/')
	else: tmp_wd = wd
	R_script = """
	setwd("%s")
	tbl = read.table(file = "%s_stats.txt",header = T)
	pdf(file='%s_stats.pdf',width=12, height=9)
	time = -tbl$time
	par(mfrow=c(2,2))
	library(scales)
	plot(time,tbl$diversity, type="l",lwd = 2, ylab= "Number of lineages", xlab="Time (Ma)", main="Diversity through time", ylim=c(0,max(tbl$M_div,na.rm =T)+1),xlim=c(min(time),0))
	polygon(c(time, rev(time)), c(tbl$M_div, rev(tbl$m_div)), col = alpha("#504A4B",0.5), border = NA)
	plot(time,tbl$median_age, type="l",lwd = 2, ylab = "Median age", xlab="Time (Ma)", main= "Taxon age", ylim=c(0,max(tbl$M_age,na.rm =T)+1),xlim=c(min(time),0))
	polygon(c(time, rev(time)), c(tbl$M_age, rev(tbl$m_age)), col = alpha("#504A4B",0.5), border = NA)
	plot(time,tbl$turnover, type="l",lwd = 2, ylab = "Fraction of new taxa", xlab="Time (Ma)", main= "Turnover", ylim=c(0,max(tbl$M_turnover,na.rm =T)+.1),xlim=c(min(time),0))
	polygon(c(time, rev(time)), c(tbl$M_turnover, rev(tbl$m_turnover)), col = alpha("#504A4B",0.5), border = NA)
	plot(time,tbl$life_exp, type="l",lwd = 2, ylab = "Median longevity", xlab="Time (Ma)", main= "Taxon (estimated) longevity", ylim=c(0,max(tbl$M_life_exp,na.rm =T)+1),xlim=c(min(time),0))
	polygon(c(time, rev(time)), c(tbl$M_life_exp, rev(tbl$m_life_exp)), col = alpha("#504A4B",0.5), border = NA)
	n<-dev.off()
	""" % (tmp_wd, out_file_name,out_file_name)
	R_file.writelines(R_script)
	R_file.close()
	print "\nAn R script with the source for the stat plot was saved as: \n%s" % (R_file_name)
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		cmd="cd %s & Rscript %s" % (wd,out_file_name+"_stats.R")
	else:
		cmd="cd %s; Rscript %s" % (wd,out_file_name+"_stats.R")
	os.system(cmd)
	print "done\n"



########################## COMBINE LOG FILES ##############################
def comb_rj_rates(infile, files,tag, resample, rate_type):
	j=0
	for f in files:
		f_temp = open(f,'r')
		x_temp = [line for line in f_temp.readlines()]
		x_temp = x_temp[max(1,int(burnin)):]
		x_temp =array(x_temp)
		if 2>1: #try:
			if resample>0:
				r_ind= sort(np.random.randint(0,len(x_temp),resample))
				x_temp = x_temp[r_ind]
			if j==0: 
				comb = x_temp
			else:
				comb = np.concatenate((comb,x_temp))
			j+=1
		#except: 
		#	print "Could not process file:",f
	
	outfile = "%s/combined_%s%s_%s.log" % (infile,len(files),tag,rate_type)	
	with open(outfile, 'w') as f:
		for i in comb: f.write(i)

def comb_mcmc_files(infile, files,burnin,tag,resample,col_tag,file_type=""): 
	j=0
	for f in files:
		if platform.system() == "Windows" or platform.system() == "Microsoft":
			f = f.replace("\\","/")
		
		if 2>1: #try:
			file_name =  os.path.splitext(os.path.basename(f))[0]
			print file_name,			
			t_file=loadtxt(f, skiprows=max(1,int(burnin)))
			shape_f=shape(t_file)
			print shape_f
			#t_file = t[burnin:shape_f[0],:]#).astype(str)
			# only sample from cold chain
			
			head = np.array(next(open(f)).split()) # should be faster\
			if j == 0:
				tbl_header = '\t'.join(head)	
			if "temperature" in head or "beta" in head:
				try: 
					temp_index = np.where(head=="temperature")[0][0]
				except(IndexError): 
					temp_index = np.where(head=="beta")[0][0]
			
				temp_values = t_file[:,temp_index]
				t_file = t_file[temp_values==1,:]
				print "removed heated chains:",np.shape(t_file)
				
				
			# exclude preservation rates under TPP model (they can mismatch)
			if len(col_tag) == 0:
				q_ind = np.array([i for i in range(len(head)) if "q_" in head[i]])
				if len(q_ind)>0:
					mean_q = np.mean(t_file[:,q_ind],axis=1)
					t_file = np.delete(t_file,q_ind,axis=1)
					t_file = np.insert(t_file,q_ind[0],mean_q,axis=1)
				
			shape_f=shape(t_file)
			
			if resample>0:
				r_ind= sort(np.random.randint(0,shape_f[0],resample))
				t_file = t_file[r_ind,:]
			
			

		#except: print "ERROR in",f	
		if len(col_tag) == 0:
			if j==0: 
				head_temp = np.array(next(open(f)).split())
				head_temp = np.delete(head_temp,q_ind)
				head_temp = np.insert(head_temp,q_ind[0],"mean_q")
				tbl_header=""
				for i in head_temp: tbl_header = tbl_header + "\t" + i
				tbl_header+="\n"
				comb = t_file
			else:
				comb = np.concatenate((comb,t_file),axis=0)
		else: 
			head_temp = next(open(f)).split() # should be faster
			sp_ind_list=[]
			for TAG in col_tag:
				if TAG in head_temp:
					sp_ind_list+=[head_temp.index(s) for s in head_temp if s == TAG]
			
			try: 
				col_tag_ind = np.array([int(tag_i) for tag_i in col_tag])
				sp_ind= np.array(col_tag_ind)
			except:
				sp_ind= np.array(sp_ind_list)

			#print "COLTAG",col_tag, sp_ind, head_temp
			#sys.exit()	
			

			#print "INDEXES",sp_ind
			if j==0: 
				head_temp= np.array(head_temp)
				head_t= ["%s\t" % (i) for i in head_temp[sp_ind]]
				tbl_header="it\t"
				for i in head_t: tbl_header+=i
				tbl_header+="\n"
				print "found", len(head_t), "columns"
				comb = t_file[:,sp_ind]
			else:
				comb = np.concatenate((comb,t_file[:,sp_ind]),axis=0)
			
		j+=1

	#print shape(comb)	
	if len(col_tag) == 0:
		sampling_freq= comb[1,0]-comb[0,0]
		comb[:,0] = (np.arange(0,len(comb))+1)*sampling_freq
		fmt_list=['%i']
		for i in range(1,np.shape(comb)[1]): fmt_list.append('%4f')
	else: 
		fmt_list=['%i']
		for i in range(1,np.shape(comb)[1]+1): fmt_list.append('%4f')
		comb = np.concatenate((np.zeros((len(comb[:,0]),1)),comb),axis=1)
	comb[:,0] = (np.arange(0,len(comb)))

	print np.shape(comb), len(fmt_list)
	
	outfile = "%s/combined_%s%s_%s.log" % (infile,len(files),tag,file_type)
	
	with open(outfile, 'w') as f:
		f.write(tbl_header)
		if platform.system() == "Windows" or platform.system() == "Microsoft":
			np.savetxt(f, comb, delimiter="\t",fmt=fmt_list,newline="\r") #)
		else:
			np.savetxt(f, comb, delimiter="\t",fmt=fmt_list,newline="\n") #)

def comb_log_files_smart(path_to_files,burnin=0,tag="",resample=0,col_tag=[]):
	infile=path_to_files
	sys.path.append(infile)
	direct="%s/*%s*.log" % (infile,tag)
	files=glob.glob(direct)
	files=sort(files)
	print "found", len(files), "log files...\n"
	if len(files)==0: quit()
	j=0
	burnin = int(burnin)	
	
	# RJ rates files
	files_temp = [f for f in files if "_sp_rates.log" in os.path.basename(f)]
	if len(files_temp)>1: 
		print "processing %s *_sp_rates.log files" % (len(files_temp))
		comb_rj_rates(infile, files_temp,tag, resample, rate_type="sp_rates")
	
	files_temp = [f for f in files if "_ex_rates.log" in os.path.basename(f)]
	if len(files_temp)>1: 
		print "processing %s *_ex_rates.log files" % (len(files_temp))
		comb_rj_rates(infile, files_temp,tag, resample, rate_type="ex_rates")

	# MCMC files
	files_temp = [f for f in files if "_mcmc.log" in os.path.basename(f)]
	if len(files_temp)>1: 
		print "processing %s *_mcmc.log files" % (len(files_temp))
		comb_mcmc_files(infile, files_temp,burnin,tag,resample,col_tag,file_type="mcmc")
	files_temp = [f for f in files if "_marginal_rates.log" in os.path.basename(f)]
	if len(files_temp)>1: 
		print "processing %s *_marginal_rates.log files" % (len(files_temp))
		comb_mcmc_files(infile, files_temp,burnin,tag,resample,col_tag,file_type="marginal_rates")
	
	



def comb_log_files(path_to_files,burnin=0,tag="",resample=0,col_tag=[]):
	infile=path_to_files
	sys.path.append(infile)
	direct="%s/*%s*.log" % (infile,tag)
	files=glob.glob(direct)
	files=sort(files)
	print "found", len(files), "log files...\n"
	if len(files)==0: quit()
	j=0


	burnin = int(burnin)

	if "_sp_rates.log" in os.path.basename(files[0]) or "_ex_rates.log" in os.path.basename(files[0]):
		for f in files:
			f_temp = open(f,'r')
			x_temp = [line for line in f_temp.readlines()]
			x_temp = x_temp[max(1,int(burnin)):]
			x_temp =array(x_temp)
			try:
				if resample>0:
					r_ind= sort(np.random.randint(0,len(x_temp),resample))
					x_temp = x_temp[r_ind]
				if j==0:
					comb = x_temp
				else:
					comb = np.concatenate((comb,x_temp))
				j+=1
			except:
				print "Could not process file:",f

		outfile = "%s/combined_%s%s.log" % (infile,len(files),tag)
		with open(outfile, 'w') as f:
			#
			for i in comb: f.write(i)
			 #fmt_list=['%i']
			 #if platform.system() == "Windows" or platform.system() == "Microsoft":
			 #	np.savetxt(f, comb, delimiter="\t",fmt=fmt_list,newline="\r") #)
			 #else:
			 #	np.savetxt(f, comb, delimiter="\t",fmt=fmt_list,newline="\n") #)
			   #
		sys.exit("done")


	for f in files:
		if platform.system() == "Windows" or platform.system() == "Microsoft":
			f = f.replace("\\","/")

		try:
			file_name =  os.path.splitext(os.path.basename(f))[0]
			print file_name,
			t_file=loadtxt(f, skiprows=max(1,int(burnin)))
			shape_f=shape(t_file)
			print shape_f
			#t_file = t[burnin:shape_f[0],:]#).astype(str)
			# only sample from cold chain

			head = np.array(next(open(f)).split()) # should be faster\
			#txt_tbl = np.genfromtxt(f, delimiter="\t")
			#print "TRY", txt_tbl[0:],np.shape(txt_tbl), head
			if j == 0:
				tbl_header = '\t'.join(head)
			if "temperature" in head or "beta" in head:
				try:
					temp_index = np.where(head=="temperature")[0][0]
				except(IndexError):
					temp_index = np.where(head=="beta")[0][0]

				temp_values = t_file[:,temp_index]
				t_file = t_file[temp_values==1,:]
				print "removed heated chains:",np.shape(t_file)
			shape_f=shape(t_file)

			if resample>0:
				r_ind= sort(np.random.randint(0,shape_f[0],resample))
				t_file = t_file[r_ind,:]



		except: print "ERROR in",f
		if len(col_tag) == 0:
			if j==0:
				tbl_header = next(open(f))#.split()
				comb = t_file
			else:
				comb = np.concatenate((comb,t_file),axis=0)
		else:
			head_temp = next(open(f)).split() # should be faster
			sp_ind_list=[]
			for TAG in col_tag:
				if TAG in head_temp:
					sp_ind_list+=[head_temp.index(s) for s in head_temp if s == TAG]

			try:
				col_tag_ind = np.array([int(tag_i) for tag_i in col_tag])
				sp_ind= np.array(col_tag_ind)
			except:
				sp_ind= np.array(sp_ind_list)

			#print "COLTAG",col_tag, sp_ind, head_temp
			#sys.exit()


			#print "INDEXES",sp_ind
			if j==0:
				head_temp= np.array(head_temp)
				head_t= ["%s\t" % (i) for i in head_temp[sp_ind]]
				tbl_header="it\t"
				for i in head_t: tbl_header+=i
				tbl_header+="\n"
				print "found", len(head_t), "columns"
				comb = t_file[:,sp_ind]
			else:
				comb = np.concatenate((comb,t_file[:,sp_ind]),axis=0)

		j+=1

	#print shape(comb)
	if len(col_tag) == 0:
		sampling_freq= comb[1,0]-comb[0,0]
		comb[:,0] = (np.arange(0,len(comb))+1)*sampling_freq
		fmt_list=['%i']
		for i in range(1,np.shape(comb)[1]): fmt_list.append('%4f')
	else:
		fmt_list=['%i']
		for i in range(1,np.shape(comb)[1]+1): fmt_list.append('%4f')
		comb = np.concatenate((np.zeros((len(comb[:,0]),1)),comb),axis=1)
	comb[:,0] = (np.arange(0,len(comb)))

	print np.shape(comb), len(fmt_list)

	outfile = "%s/combined_%s%s.log" % (infile,len(files),tag)

	with open(outfile, 'w') as f:
		f.write(tbl_header)
		if platform.system() == "Windows" or platform.system() == "Microsoft":
			np.savetxt(f, comb, delimiter="\t",fmt=fmt_list,newline="\r") #)
		else:
			np.savetxt(f, comb, delimiter="\t",fmt=fmt_list,newline="\n") #)

########################## INITIALIZE MCMC ##############################
def get_gamma_rates(a):
	b=a
	m = gdtrix(b,a,YangGammaQuant) # user defined categories
	s=pp_gamma_ncat/sum(m) # multiplier to scale the so that the mean of the discrete distribution is one
	return array(m)*s # SCALED VALUES

def init_ts_te(FA,LO):
	#ts=FA+np.random.exponential(.75, len(FA)) # exponential random starting point
	#tt=np.random.beta(2.5, 1, len(LO)) # beta random starting point
	#ts=FA+(.025*FA) #IMPROVE INIT
	#te=LO-(.025*LO) #IMPROVE INIT

	brl = FA-LO
	q = N/(brl+0.1)

	ts= FA+np.random.exponential(1./q,len(q))
	te= LO-np.random.exponential(1./q,len(q))

	if max(ts) > boundMax:
		ts[ts>boundMax] = np.random.uniform(FA[ts>boundMax],boundMax,len(ts[ts>boundMax])) # avoit init values outside bounds
	if min(te) < boundMin:
		te[te<boundMin] = np.random.uniform(boundMin,LO[te<boundMin],len(te[te<boundMin])) # avoit init values outside bounds
	#te=LO*tt
	if frac1==0: ts, te= FA,LO
	try:
		ts[SP_not_in_window] = boundMax
		te[EX_not_in_window] = boundMin
	except(NameError): pass
	return ts, te

def init_BD(n):
	#return np.repeat(0.5,max(n-1,1))
	return np.random.exponential(.2, max(n-1,1))+.1

def init_times(root_age,time_framesL,time_framesM,tip_age):
	timesL=np.linspace(root_age,tip_age,time_framesL+1)
	timesM=np.linspace(root_age,tip_age,time_framesM+1)
	timesM[1:time_framesM] +=1
	timesL[time_framesL] =0
	timesM[time_framesM] =0
	#print timesL,timesM
	#ts,te=sort(ts)[::-1],sort(te)[::-1]
	#indL = np.linspace(len(ts),0,time_framesL).astype(int)
	#indM = np.linspace(len(te[te>0]),0,time_framesM).astype(int)
	#
	#print indL[1:time_framesL], indM[1:time_framesM]
	#print ts[indL[1:time_framesL]],te[indM[1:time_framesM]]
	#b= [0]+ list(te[indM[1:time_framesM]])+[max(ts)]
	#print np.histogram(ts,bins=b)
	#quit()

	return timesL, timesM

def init_q_rates(): # p=1 for alpha1=alpha2
	return array([np.random.uniform(.5,1),np.random.uniform(0.25,1)]),np.zeros(3)

########################## UPDATES ######################################
def update_parameter(i, m, M, d, f):
	#d=fabs(np.random.normal(d,d/2.)) # variable tuning prm
	if i>0 and np.random.random()<=f:
		ii = i+(np.random.random()-.5)*d
		if ii<m: ii=(ii-m)+m
		if ii>M: ii=(M-(ii-M))
		if ii<m: ii=i
	else: ii=i
	return ii

def update_parameter_normal(i, m, M, d):
	if d>0: ii = np.random.normal(i,d)
	else: ii=i
	if ii<m: ii=(ii-m)+m
	if ii>M: ii=(M-(ii-M))
	if ii<m: ii=i
	return ii

def update_multiplier_proposal(i,d):
	S=shape(i)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
 	ii = i * m
	return ii, sum(log(m))

def update_rates_sliding_win(L,M,tot_L,mod_d3):
	Ln=zeros(len(L))
	Mn=zeros(len(M))
	#if random.random()>.5:
	for i in range(len(L)): Ln[i]=update_parameter(L[i],0, inf, mod_d3, f_rate)
	#else:
	for i in range(len(M)): Mn[i]=update_parameter(M[i],0, inf, mod_d3, f_rate)
	#Ln,Mn=Ln * scale_factor/tot_L , Mn * scale_factor/tot_L
	return Ln,Mn, 1

def update_parameter_normal_vec(oldL,d,f=.25):
	S = np.shape(oldL)
	ii = np.random.normal(0,d,S)
	ff = np.random.binomial(1,f,S)
	s= oldL + ii*ff
	return s


def update_rates_multiplier(L,M,tot_L,mod_d3):
	if use_ADE_model == 0 and use_Death_model == 0:
		# UPDATE LAMBDA
		S=np.shape(L)
		#print L, S
		ff=np.random.binomial(1,f_rate,S)
		#print ff
		d=1.2
		u = np.random.uniform(0,1,S)
		l = 2*log(mod_d3)
		m = exp(l*(u-.5))
		m[ff==0] = 1.
	 	newL = L * m
		U=sum(log(m))
	else: U,newL = 0,L

	# UPDATE MU
	S=np.shape(M)
	ff=np.random.binomial(1,f_rate,S)
	d=1.2
	u = np.random.uniform(0,1,S) #*np.rint(np.random.uniform(0,f,S))
	l = 2*log(mod_d3)
	m = exp(l*(u-.5))
	m[ff==0] = 1.
 	newM = M * m
	U+=sum(log(m))
	return newL,newM,U

def update_q_multiplier(q,d=1.1,f=0.75):
	S=np.shape(q)
	ff=np.random.binomial(1,f,S)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
	m[ff==0] = 1.
 	new_q = q * m
	U=sum(log(m))
	return new_q,U

def update_times(times, max_time,min_time, mod_d4,a,b):
	rS= times+zeros(len(times))
	rS[0]=max_time
	if np.random.random()< 1.5:
		for i in range(a,b): rS[i]=update_parameter(times[i],min_time, max_time, mod_d4, 1)
	else:
		i = np.random.choice(range(a,b))
		rS[i]=update_parameter(times[i],min_time, max_time, mod_d4*3, 1)
	y=sort(-rS)
	y=-1*y
	return y

def update_ts_te(ts, te, d1):
	tsn, ten= zeros(len(ts))+ts, zeros(len(te))+te
	f1=np.random.random_integers(1,frac1) #int(frac1*len(FA)) #-np.random.random_integers(0,frac1*len(FA)-1))
	ind=np.random.choice(SP_in_window,f1) # update only values in SP/EX_in_window
	tsn[ind] = ts[ind] + (np.random.uniform(0,1,len(ind))-.5)*d1
	M = np.inf #boundMax
	tsn[tsn>M]=M-(tsn[tsn>M]-M)
	m = FA
	tsn[tsn<m]=(m[tsn<m]-tsn[tsn<m])+m[tsn<m]
	tsn[tsn>M] = ts[tsn>M]

	ind=np.random.choice(EX_in_window,f1)
	ten[ind] = te[ind] + (np.random.uniform(0,1,len(ind))-.5)*d1
	M = LO
	ten[ten>M]=M[ten>M]-(ten[ten>M]-M[ten>M])
	m = 0 #boundMin
	ten[ten<m]=(m-ten[ten<m])+m
	ten[ten>M] = te[ten>M]
	ten[LO==0]=0									 # indices of LO==0 (extant species)
	S= tsn-ten
	if min(S)<=0: print S
	tsn[SP_not_in_window] = max([boundMax, max(tsn[SP_in_window])])
	ten[EX_not_in_window] = min([boundMin, min(ten[EX_in_window])])
	return tsn,ten


#### GIBBS SAMPLER S/E
def draw_se_gibbs(fa,la,q_rates_L,q_rates_M,q_times):
	t = np.sort(np.array([fa, la] + list(q_times)))[::-1]
	# sample ts
	prior_to_fa = np.arange(len(q_times))[q_times>fa]
	tfa = (q_times[prior_to_fa]-fa)[::-1] # time since fa
	qfa = q_rates_L[prior_to_fa][::-1] # rates before fa
	ts_temp=0
	for i in range(len(qfa)):
		q = qfa[i]
		deltaT = np.random.exponential(1./q)
		ts_temp = min(ts_temp+deltaT, tfa[i])
		if ts_temp < tfa[i]:
			break

	ts= ts_temp+fa
	#print "TS:", ts, fa
	#print q_times
	#print la
	if la>0:
		# sample te
		after_la = np.arange(len(q_times))[q_times<la]
		tla = (la-q_times[after_la]) # time after la
		qla = q_rates_M[after_la-1] # rates after la
		#print "QLA", qla, tla
		te_temp=0
		i,attempt=0,0
		while True:
			q = qla[i]
			deltaT = np.random.exponential(1./q)
			te_temp = min(te_temp+deltaT, tla[i])
			#print attempt,i,te_temp,len(qla)
			if te_temp < tla[i]:
				break
			i+=1
			attempt+=1
			if i == len(qla):
				i= 0 # try again
			if attempt==100:
				te_temp = np.random.uniform(0,la)
				break

		te= la-te_temp
		#print "TE:", te
	else:
		te=0
	return (ts,te)

def gibbs_update_ts_te(q_rates_L,q_rates_M,q_time_frames):
	#print q_rates,q_time_frames
	q_times= q_time_frames+0
	q_times[0] = np.inf
	new_ts = []
	new_te = []
	for sp_indx in range(0,len(FA)):
		#print "sp",sp_indx
		s,e = draw_se_gibbs(FA[sp_indx],LO[sp_indx],q_rates_L,q_rates_M,q_times)
		new_ts.append(s)
		new_te.append(e)
	return np.array(new_ts), np.array(new_te)




def seed_missing(x,m,s): # assigns random normally distributed trait values to missing data
	return np.isnan(x)*np.random.normal(m,s)+np.nan_to_num(x)

def cond_alpha_proposal(hp_gamma_shape,hp_gamma_rate,current_alpha,k,n):
	z = [current_alpha + 1.0, float(n)]
	f = np.random.dirichlet(z,1)[0]
	eta = f[0]
	u = np.random.uniform(0,1,1)[0]
	x = (hp_gamma_shape + k - 1.0) / ((hp_gamma_rate - np.log(eta)) * n)
	if (u / (1.0-u)) < x: new_alpha = np.random.gamma( (hp_gamma_shape+k), (1./(hp_gamma_rate-np.log(eta))) )
	else: new_alpha = np.random.gamma( (hp_gamma_shape+k-1.), 1./(hp_gamma_rate-np.log(eta)) )
	return new_alpha


def get_post_sd(N,HP_shape=2,HP_rate=2,mean_Norm=0): # get sd of Normal from sample N and hyperprior G(a,b)
	n= len(N)
	G_shape = HP_shape + n*.5
	G_rate  = HP_rate  + sum((N-mean_Norm)**2)*.5
	tau = np.random.gamma(shape=G_shape,scale=1./G_rate)
	sd= sqrt(1./tau)
	return(sd)

########################## PRIORS #######################################
try:
	scipy.stats.gamma.logpdf(1, 1, scale=1./1,loc=0)
	def prior_gamma(L,a,b):
		if hasFoundPyRateC:
			return PyRateC_getLogGammaPDF(L, a, 1./b)#scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0)
		else:
			return scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0)
	def prior_normal(L,sd):
		return scipy.stats.norm.logpdf(L,loc=0,scale=sd)
	def prior_cauchy(x,s):
		return scipy.stats.cauchy.logpdf(x,scale=s,loc=0)
except(AttributeError): # for older versions of scipy
	def prior_gamma(L,a,b):
		return (a-1)*log(L)+(-b*L)-(log(b)*(-a)+ log(gamma(a)))
	def prior_normal(L,sd):
		return -(L**2/(2*sd**2)) - log(sd*sqrt(2*np.pi))
	def prior_cauchy(x,s):
		return -log(np.pi*s * (1+ (x/s)**2))

def prior_times_frames(t, root, tip_age,a): # un-normalized Dirichlet (truncated)
	diff_t, min_t = abs(np.diff(t)), np.min(t)
	if np.min(diff_t)<=min_allowed_t: return -inf
	elif (min_t<=tip_age+min_allowed_t) and (min_t>0): return -inf
	else:
		t_rel=diff_t/root
		return (a-1)*log(t_rel)


def get_min_diffTime(times):
	diff_t = abs(np.diff(times))
	if fix_edgeShift==1: # min and max bounds
		diff_t = diff_t[1:-1]
	elif fix_edgeShift==2: # max bound
		diff_t = diff_t[1]
	elif fix_edgeShift==3: # min bound
		diff_t = diff_t[-1]
	return np.min(diff_t)


def prior_sym_beta(x,a):
	# return log(x)*(a-1)+log(1-x)*(a-1) # un-normalized beta
	return scipy.stats.beta.logpdf(x, a,a)

def prior_root_age(root, max_FA, l): # exponential (truncated)
	l=1./l
	if root>=max_FA: return log(l)-l*(root)
	else: return -inf

def prior_uniform(x,m,M):
	if x>m and x<M: return log(1./(M-m))
	else: return -inf

def G_density(x,a,b):
	#return (1./b)**a * 1./gamma(a) * x**(a-1) *exp(-(1./b)*x)
	return scipy.stats.gamma.pdf(x, a, scale=1./b,loc=0)

def logPERT4_density(M,m,a,b,x): # relative 'stretched' LOG-PERT density: PERT4 * (s-e)
	return log((M-x)**(b-1) * (-m+x)**(a-1)) - log ((M-m)**4 * f_beta(a,b))

def PERT4_density(M,m,a,b,x):  # relative 'stretched' PERT density: PERT4 * (s-e)
	return ((M-x)**(b-1) * (-m+x)**(a-1)) /((M-m)**4 * f_beta(a,b))

def logPERT4_density5(M,m,a,b,x): # relative LOG-PERT density: PERT4
	return log((M-x)**(b-1) * (-m+x)**(a-1)) - log ((M-m)**5 * f_beta(a,b))

########################## LIKELIHOODS ##################################
def HPBD1(timesL,timesM,L,M,T,s):
	return sum(prior_cauchy(L,s[0]))+sum(prior_cauchy(M,s[1]))

def HPBD2(timesL,timesM,L,M,T,s):
	return sum(prior_gamma(L,L_lam_r,s[0])) + sum(prior_gamma(M,M_lam_r,s[1]))

def HPBD3(timesL,timesM,L,M,T,s):
	def pNtvar(arg):
		T=arg[0]
		L=arg[1]
		M=arg[2]
		N=arg[3]
		Dt=-np.diff(T)
		r_t = (L - M)*Dt
		Beta=  sum(exp((L - M)*Dt))
		Alpha= sum(L*exp((L - M)*Dt))
		lnBeta=  log(sum(exp((L - M)*Dt)))
		lnAlpha= log(sum(L*exp((L - M)*Dt)))
		#P   = (Beta/(Alpha*(1+Alpha))) *	(Alpha/(1+Alpha))**N
		if N>0: lnP = (lnBeta-(lnAlpha+log(1+Alpha))) + (lnAlpha-log(1+Alpha))*N
		else:	lnP = log((1+Alpha-Beta)/(Alpha*(1+Alpha)))
		return lnP

	### HYPER-PRIOR BD ###
	n0=1
	timesMtemp = np.zeros(len(timesM)) + timesM
	timesMtemp[1:len(timesM)-1] = timesMtemp[1:len(timesM)-1] +0.0001
	#if len(timesM)>2:
	all_t_frames=sort(np.append(timesL, timesMtemp[1:len(timesMtemp)-1] ))[::-1] # merge time frames

	#all_t_frames=sort(np.append(timesL, timesM[1:-1]+.001 ))[::-1] # merge time frames
	#else: all_t_frames=sort(np.append(timesL, timesM[1:-1] ))[::-1] # merge time frames
	sL=(np.in1d(all_t_frames,timesL[1:-1])+0).nonzero()[0] # indexes within 'all_t_frames' of shifts of L
	sM=(np.in1d(all_t_frames,timesMtemp[1:-1])+0).nonzero()[0] # indexes within 'all_t_frames' of shifts of M
	sL[(sL-1>len(M)-1).nonzero()]=len(M)
	sM[(sM-1>len(L)-1).nonzero()]=len(L)

	nL=zeros(len(all_t_frames)-1)
	nM=zeros(len(all_t_frames)-1)

	Ln=insert(L,sM,L[sM-1]) # l rates for all_t_frames
	Mn=insert(M,sL,M[sL-1]) # m rates for all_t_frames

	return pNtvar([all_t_frames,Ln,Mn,tot_extant])

# BIRTH-DEATH MODELS

#---- reconstructed BD
def p0(t,l,m,rho):
	return 1 - (rho*(l-m) / (rho*l + (l*(1-rho) -m) * exp(-(l-m)*t)))

def p1(t,l,m,rho):
	return  rho*(l-m)**2 * exp(-(l-m)*t)/(rho*l + (l*(1-rho) -m)*exp(-(l-m)*t))**2

def treeBDlikelihood(x,l,m,rho,root=1,survival=1):
	#_ lik = (root + 1) * log(p1(x[0], l, m, rho))
	#_ for i in range(1, len(x)) :
	#_	 lik = lik + log(l * p1(x[i], l, m, rho))
	#_ if survival == 1:
	#_ 	lik = lik - (root + 1) * log(1 - p0(x[0], l, m, rho))
	#_ return lik
	lik1= (root + 1) * log(p1(x[0], l, m, rho))
	lik2= sum(log(l * p1(x[1:], l, m, rho)))
	lik3= - (root + 1) * log(1 - p0(x[0], l, m, rho))
	return lik1+lik2+lik3

#----

def BD_lik_discrete_trait(arg):
	[ts,te,L,M]=arg
	S = ts-te
	lik0 =  sum(log(L)*lengths_B_events )	#
	lik1 = -sum(L*sum(S))				   # assumes that speiation can arise from any trait state
	#lik1 = -sum([L[i]*sum(S[ind_trait_species==i]) for i in range(len(L))])
	lik2 =  sum(log(M)*lengths_D_events)										# Trait specific extinction
	lik3 = -sum([M[i]*sum(S[ind_trait_species==i]) for i in range(len(M))]) # only species with a trait state can go extinct
	return sum(lik0+lik1+lik2+lik3)

def BD_lik_discrete_trait_continuous(arg):
	[ts,te,L,M,cov_par]=arg
	S = ts-te
	# speciation
	sp_rate=exp(log(L)+cov_par[0]*(con_trait-parGAUS[0]))
	lik01 = sum(log(sp_rate)) + sum(-sp_rate*S) #, cov_par
	# extinction
	ex_rate = exp(log(M[ind_trait_species])+cov_par[0]*(con_trait-parGAUS[0]))
	lik2 =  sum(log(ex_rate[te>0]))   # only count extinct species
	lik3 = -sum([M[i]*sum(S[ind_trait_species==i]) for i in range(len(M))]) # only species with a trait state can go extinct
	return sum(lik01+lik2+lik3)


def BPD_lik_vec_times(arg):
	[ts,te,time_frames,L,M]=arg
	if fix_SE == 0 or fix_Shift == 0:
		BD_lik = 0
		B = sort(time_frames)+0.000001 # add small number to avoid counting extant species as extinct
		ss1 = np.histogram(ts,bins=B)[0][::-1]
		ee2 = np.histogram(te,bins=B)[0][::-1]

		for i in range(len(time_frames)-1):
			up, lo = time_frames[i], time_frames[i+1]
			len_sp_events=ss1[i]
			if i==0: len_sp_events = len_sp_events-no_starting_lineages
			len_ex_events=ee2[i]
			inTS = np.fmin(ts,up)
			inTE = np.fmax(te,lo)
			S	= inTS-inTE
			# speciation
			if use_poiD == 0:
				lik1 = log(L[i])*len_sp_events
				lik0 = -sum(L[i]*S[S>0]) # S < 0 when species outside up-lo range
			else:
				lik1 = log(L[i])*len_sp_events
				lik0 = -sum(L[i]*(up-lo)) # S < 0 when species outside up-lo range

			# extinction
			lik2 = log(M[i])*len_ex_events
			lik3 = -sum(M[i]*S[S>0]) # S < 0 when species outside up-lo range
			BD_lik += lik0+lik1+lik2+lik3
			#print "len",sum(S[S>0]),-sum(L[i]*S[S>0]), -L[i]*sum(S[S>0])
	else:
		lik0 =  log(L)*len_SS1
		lik1 = -(L* S_time_frame)
		lik2 =  log(M)*len_EE1
		lik3 = -(M* S_time_frame)
		BD_lik = lik0+lik1+lik2+lik3

	return BD_lik


def get_sp_in_frame_br_length(ts,te,up,lo):
	# index species present in time frame
	n_all_inframe = np.intersect1d((ts >= lo).nonzero()[0], (te <= up).nonzero()[0])

	# tot br length within time frame
	n_t_ts,n_t_te=zeros(len(ts)),zeros(len(ts))

	n_t_ts[n_all_inframe]= ts[n_all_inframe]   # speciation events before time frame
	n_t_ts[(n_t_ts>up).nonzero()]=up		   # for which length is accounted only from $up$ rather than from $ts$

	n_t_te[n_all_inframe]= te[n_all_inframe]   # extinction events in time frame
	n_t_te[np.intersect1d((n_t_te<lo).nonzero()[0], n_all_inframe)]=lo	 # for which length is accounted only until $lo$ rather than to $te$

	# vector of br lengths within time frame  #(scaled by rho)
	n_S=((n_t_ts[n_all_inframe]-n_t_te[n_all_inframe])) #*rhos[n_all_inframe])
	return n_all_inframe, n_S


def BD_partial_lik(arg):
	[ts,te,up,lo,rate,par, cov_par,n_frames_L]=arg
	# indexes of the species within time frame
	if par=="l": i_events=np.intersect1d((ts <= up).nonzero()[0], (ts > lo).nonzero()[0])
	else: i_events=np.intersect1d((te <= up).nonzero()[0], (te > lo).nonzero()[0])
	# index of extant/extinct species
	# extinct_sp=(te > 0).nonzero()[0]
	# present_sp=(te == 0).nonzero()[0]
	n_all_inframe, n_S = get_sp_in_frame_br_length(ts,te,up,lo)

	if cov_par !=0: # covaring model: $r$ is vector of rates tranformed by trait
		r=exp(log(rate)+cov_par*(con_trait-parGAUS[0])) # exp(log(rate)+cov_par*(con_trait-mean(con_trait[all_inframe])))
		lik= sum(log(r[i_events])) + sum(-r[n_all_inframe]*n_S) #, cov_par
	else:		   # constant rate model
		no_events = len(i_events)
		if par=="l" and up==max_age_fixed_ts:
			no_events = no_events-no_starting_lineages
		lik= log(rate)*no_events -rate*sum(n_S) #log(rate)*len(i_events) +sum(-rate*n_S)
	return lik

def BD_partial_lik_bounded(arg):
	[ts,te,up,lo,rate,par, cov_par,n_frames_L]=arg
	# indexes of the species within time frame
	if par=="l":
		i_events=np.intersect1d((ts[SP_in_window] <= up).nonzero()[0], (ts[SP_in_window] > lo).nonzero()[0])
	else:
		i_events=np.intersect1d((te[EX_in_window] <= up).nonzero()[0], (te[EX_in_window] > lo).nonzero()[0])
	n_all_inframe, n_S = get_sp_in_frame_br_length(ts,te,up,lo)
	if cov_par !=0: # covaring model: $r$ is vector of rates tranformed by trait
		r=exp(log(rate)+cov_par*(con_trait-parGAUS[0])) # exp(log(rate)+cov_par*(con_trait-mean(con_trait[all_inframe])))
		lik= sum(log(r[i_events])) + sum(-r[n_all_inframe]*n_S) #, cov_par
	else:		   # constant rate model
		#print par, len(i_events), len(te)
		lik= log(rate)*len(i_events) -rate*sum(n_S) #log(rate)*len(i_events) +sum(-rate*n_S)
	return lik


# ADE model
def cdf_WR(W_shape,W_scale,x):
	return (x/W_scale)**(W_shape)

def log_wr(t,W_shape,W_scale): # return log extinction rate at time t based on ADE model
	return log(W_shape/W_scale)+(W_shape-1)*log(t/W_scale)

def log_wei_pdf(x,W_shape,W_scale): # log pdf Weibull
	return log(W_shape/W_scale) + (W_shape-1)*log(x/W_scale) - (x/W_scale)**W_shape

def wei_pdf(x,W_shape,W_scale): # pdf Weibull
	return W_shape/W_scale * (x/W_scale)**(W_shape-1) *exp(-(x/W_scale)**W_shape)

def pdf_W_poi(W_shape,W_scale,q,x): # exp log Weibull + Q_function
	return exp(log_wei_pdf(x,W_shape,W_scale) + log(1-exp(-q*x)))

def pdf_W_poi_nolog(W_shape,W_scale,q,x): # Weibull + Q_function
	return wei_pdf(x,W_shape,W_scale) * (1-exp(-q*x))

def cdf_Weibull(x,W_shape,W_scale): # Weibull cdf
	return 1 - exp(-(x/W_scale)**W_shape)

def integrate_pdf(P,v,d,upper_lim):
	if upper_lim==0: return 0
	else: return sum(P[v<upper_lim])*d

# integration settings //--> Add to command list
nbins  = 1000
xLim   = 50
x_bins = np.linspace(0.0000001,xLim,nbins)
x_bin_size = x_bins[1]-x_bins[0]

def BD_bd_rates_ADE_lik(arg):
	[s,e,W_shape,W_scale]=arg
	# fit BD model
	birth_lik = len(s)*log(l)-l*sum(d) # replace with partial lik function
	d = s-e
	de = d[e>0] #takes only the extinct species times
	death_lik_de = sum(log_wr(de, W_shape, W_scale)) # log probability of death event
	death_lik_wte = sum(-cdf_WR(W_shape,W_scale, d[te==0])) 
	# analytical integration
	death_lik_wte = sum(-m0*cdf_WR(W_shape,W_scale, d)) # log probability of waiting time until death event
	lik = birth_lik + death_lik_de + death_lik_wte
	return lik

def BD_age_partial_lik(arg):
	[ts,te,up,lo, rate,par,  cov_par,   W_shape,q]=arg
	W_scale = rate
	ind_ex_events=np.intersect1d((te <= up).nonzero()[0], (te >= lo).nonzero()[0])
	ts_time = ts[ind_ex_events]
	te_time = te[ind_ex_events]
	br = ts_time-te_time
	#br=ts-te
	lik1=(log_wei_pdf(br[te_time>0],W_shape,W_scale)) + (log(1-exp(-q*br[te_time>0])))
	v=x_bins
	# numerical integration + analytical for right tail
	P = pdf_W_poi(W_shape,W_scale,q,v)				 # partial integral (0 => xLim) via numerical integration
	d= x_bin_size
	const_int = (1- cdf_Weibull(xLim,W_shape,W_scale)) # partial integral (xLim => Inf) via CDF_weibull
	lik2 = log( sum(P)*d  + const_int )
	lik_extant = [log(sum(P[v>i])*d + const_int)-lik2 for i in ts_time[te_time==0] ] # P(x > ts | W_shape, W_scale, q)
	# this is equal to log(1- (sum(P[v<=i]) *(v[1]-v[0]) / exp(lik2)))
	lik_extinct = sum(lik1-lik2)
	lik = lik_extinct + sum(lik_extant)
	return lik

######## W-MEAN
def get_fraction_per_bin(ts,te,time_frames):
	len_time_intervals=len(time_frames)-1
	n=len(ts)
	tot_br = ts-te
	in_bin_br =np.zeros(n*len_time_intervals).reshape(n,len_time_intervals)
	in_bin_0_br =np.zeros(n)
	te_temp = np.fmax(te,np.ones(n)*time_frames[1])
	br_temp = ts-te_temp
	in_bin_0_br[br_temp>0] = br_temp[br_temp>0]
	in_bin_br[:,0]= in_bin_0_br
	for i in range(1,len_time_intervals):
		t_i, t_i1 = time_frames[i], time_frames[i+1]
		ts_temp = np.fmin(ts,np.ones(n)*t_i)
		te_temp = np.fmax(te,np.ones(n)*t_i1)
		br_temp = ts_temp-te_temp
		in_bin_i_br = br_temp
		in_bin_i_br[br_temp<0] = 0
		in_bin_br[:,i]= in_bin_i_br
	return in_bin_br.T/tot_br

def BD_age_lik_vec_times(arg):
	[ts,te,time_frames,W_shape,W_scales,q_rates,q_time_frames]=arg
	len_time_intervals=len(time_frames)-1
	if TPP_model == 0: # because in this case q_rates[0] is alpha par of gamma model (set to 1)
		q_rates = np.zeros(len_time_intervals)+q_rates[1]

	#Weigths = get_fraction_per_bin(ts,te,time_frames)
	#W_scale_species = np.sum(W_scales*Weigths.T,axis=1)
	W_scale_species = np.zeros(len(ts))+W_scales[0]
	#W_scale_species = np.zeros(len(ts))
	#W_scale_species = W_scales[np.round(Weigths).astype(int)][1]

	qWeigths = get_fraction_per_bin(ts,te,q_time_frames)
	q_rate_species  = np.sum(q_rates*qWeigths.T,axis=1)

	br=ts-te
	#print W_scales
	#print "O:",Weigths
	#print "T:", W_scale_species
	lik1=(log_wei_pdf(br[te>0],W_shape,W_scale_species[te>0])) + (log(1-exp(-q_rate_species[te>0]*br[te>0])))
	# the q density using the weighted mean is equal to (log(1-exp(-np.sum(q_rates*(br*Weigths).T,axis=1))))
	# numerical integration + analytical for right tail
	v=np.zeros((len(ts),len(x_bins)))+x_bins
	P = pdf_W_poi(W_shape,W_scale_species,q_rate_species,v.T)				 # partial integral (0 => xLim) via numerical integration
	d= x_bin_size
	const_int = (1- cdf_Weibull(xLim,W_shape,W_scale_species)) # partial integral (xLim => Inf) via CDF_weibull
	lik2 = log( np.sum(P,axis=0)*d  + const_int )
	ind_extant = (te==0).nonzero()[0]
	lik_extant =[log(sum(P[x_bins>br[i],i])*d + const_int[i])-lik2[i] for i in ind_extant] # P(x > ts | W_shape, W_scale, q)
	# this is equal to log(1- (sum(P[v<=i]) *(v[1]-v[0]) / exp(lik2)))
	lik_extinct = sum(lik1-lik2[te>0])
	lik = lik_extinct + sum(lik_extant)
	return lik

def pure_death_shift(arg):
	[ts,te,time_frames,L,M,Q]=arg
	BD_lik = 0
	B = sort(time_frames)+0.000001 # add small number to avoid counting extant species as extinct
	ss1 = np.histogram(ts,bins=B)[0][::-1]
	ee2 = np.histogram(te,bins=B)[0][::-1]

	for i in range(len(time_frames)-1):
		up, lo = time_frames[i], time_frames[i+1]
		len_sp_events=ss1[i]
		len_ex_events=ee2[i]
		inTS = np.fmin(ts,up)
		inTE = np.fmax(te,lo)
		S	= inTS-inTE
		# prob waiting times: qExp(S,mu) * samplingP(S,q) # CHECK for ANALYTICAL SOLUTION?
		# prob extinction events: dExp(S[inTE>lo],mu) * samplingD(S[inTE>lo],mu)
		# prob extant taxa: 1 - (dExp(S[inTE>lo],mu) * samplingD(S[inTE>lo],mu))/exp(P_waiting_time)


		BD_lik += lik0+lik1+lik2+lik3

def BDI_partial_lik(arg):
	[ts,te,up,lo,rate,par, cov_par,n_frames_L]=arg
	ind_in_time = np.intersect1d((all_events_array[0] <= up).nonzero()[0], (all_events_array[0] > lo).nonzero()[0])
	traj_T=div_trajectory[ind_in_time]
	all_events_temp2_T=all_events_array[:,ind_in_time]

	L = np.zeros(len(traj_T))+rate * (1-model_BDI) # if model_BDI=0: BD, if model_BDI=1: ID
	M = np.zeros(len(traj_T))+rate
	I = np.zeros(len(traj_T))+rate * model_BDI
	k=traj_T
	event_at_state_k= all_events_temp2_T[1]-1 # events=0: speciation; =1: extinction
	Tk = dT_events[ind_in_time]
	Uk = 1-event_at_state_k
	Dk = event_at_state_k

	if par=="l":
		# calc likelihood only when diversity > 0
		lik = sum(log(L[k>0]*k[k>0]+I[k>0])*Uk[k>0] - (L[k>0]*k[k>0]+I[k>0])*Tk[k>0])
	else:
		# calc likelihood only when diversity > 0
		lik = sum(log(M[k>0]*k[k>0])*Dk[k>0] -(M[k>0]*k[k>0]*Tk[k>0]))
	return lik

def PoiD_partial_lik(arg):
	[ts,te,up,lo,rate,par, cov_par,n_frames_L]=arg
	if par=="l":
		i_events=np.intersect1d((ts <= up).nonzero()[0], (ts > lo).nonzero()[0])
		n_i_events = len(i_events)
		lik = log(rate)*n_i_events - (rate * (up-lo))
	else:
		i_events=np.intersect1d((te <= up).nonzero()[0], (te > lo).nonzero()[0])
		n_i_events = len(i_events)
		n_all_inframe, n_S = get_sp_in_frame_br_length(ts,te,up,lo)
		lik= log(rate)*n_i_events  -rate*sum(n_S)
	return lik

# PRESERVATION
def HPP_vec_lik(arg):
	[te,ts,time_frames,q_rates,i,alpha]=arg
	i=int(i) # species number
	k_vec = occs_sp_bin[i] # no. occurrences per time bin per species
	# e.g. k_vec = [0,0,1,12.,3,0]
	if ts in time_frames and ts != time_frames[0]: # if max(ts)<max(q_shift), ignore max(ts)
		time_frames = time_frames[time_frames != ts]
	h = np.histogram(np.array([ts,te]),bins=sort(time_frames))[0][::-1]
	ind_tste= (h).nonzero()[0]
	ind_min=min(ind_tste)
	ind_max=max(ind_tste)
	ind=np.arange(len(time_frames))
	ind = ind[ind_min:(ind_max+1)] # indexes of time frames where lineage is present
	# calc time lived in each time frame
	t = time_frames[time_frames<ts]
	t = t[t>te]
	t2 = np.array([ts]+list(t)+[te])
	d = abs(np.diff(t2))

	if argsG == 1 and sum(k_vec)>1: # for singletons no Gamma
		# loop over gamma categories
		YangGamma=get_gamma_rates(alpha)
		lik_vec = np.zeros(pp_gamma_ncat)
		for i in range(pp_gamma_ncat):
			qGamma= YangGamma[i]*q_rates
			lik_vec[i] = sum(-qGamma[ind]*d + log(qGamma[ind])*k_vec[ind]) - log(1-exp(sum(-qGamma[ind]*d))) -sum(log(np.arange(1,sum(k_vec)+1)))

		#print lik_vec
		lik2= lik_vec-np.max(lik_vec)
		lik = sum(log(sum(exp(lik2))/pp_gamma_ncat)+np.max(lik_vec))
	else:
		lik = sum(-q_rates[ind]*d + log(q_rates[ind])*k_vec[ind]) - log(1-exp(sum(-q_rates[ind]*d))) -sum(log(np.arange(1,sum(k_vec)+1)))
	return lik

def HOMPP_lik(arg):
	[m,M,shapeGamma,q_rate,i,cov_par, ex_rate]=arg
	i=int(i)
	x=fossil[i]
	lik=0
	k=len(x[x>0]) # no. fossils for species i
	br_length = M-m
	if useBounded_BD == 1:
		br_length = min(M,boundMax)-max(m, boundMin)
	if cov_par ==2: # transform preservation rate by trait value
		q=exp(log(q_rate)+cov_par*(con_trait[i]-parGAUS[0]))
	else: q=q_rate
	if argsG == 1:
		YangGamma=get_gamma_rates(shapeGamma)
		qGamma= YangGamma*q
		lik1= -qGamma*(br_length) + log(qGamma)*k - sum(log(np.arange(1,k+1)))  -log(1-exp(-qGamma*(br_length)))
		maxLik1 = max(lik1)
		lik2= lik1-maxLik1
		lik=log(sum(exp(lik2)*(1./pp_gamma_ncat)))+maxLik1
		return lik
	else:	return -q*(br_length) + log(q)*k - sum(log(np.arange(1,k+1))) - log(1-exp(-q*(br_length)))

def NHPP_lik(arg):
	[m,M,shapeGamma,q_rate,i,cov_par, ex_rate]=arg
	i=int(i)
	x=fossil[i]
	lik=0
	k=len(x[x>0]) # no. fossils for species i
	x=sort(x)[::-1] # reverse
	xB1= -(x-M) # distance fossil-ts
	c=.5
	if cov_par !=0: # transform preservation rate by trait value
		q=exp(log(q_rate)+cov_par*(con_trait[i]-parGAUS[0]))
	else: q=q_rate
	if m==0 and use_DA == 1: # data augmentation
		l=1./ex_rate
		#quant=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
		quant=[.125,.375,.625,.875] # quantiles gamma distribution (predicted te)
		#quant=[.5]
		GM= -(-log(1-np.array(quant))/ex_rate) # get the x values from quantiles (exponential distribution)
		# GM=-array([gdtrix(ex_rate,1,jq) for jq in quant]) # get the x values from quantiles
		z=np.append(GM, [GM]*(k)).reshape(k+1,len(quant)).T
		xB=xB1/-(z-M) # rescaled fossil record from ts-te_DA to 0-1
		C=M-c*(M-GM)  # vector of modal values
		a = 1 + (4*(C-GM))/(M-GM) # shape parameters a,b=3,3
		b = 1 + (4*(-C+M))/(M-GM) # values will change in future implementations
		#print M, GM
		int_q = betainc(a,b,xB[:,k])* (M-GM)*q	# integral of beta(3,3) at time xB[k] (tranformed time 0)
		MM=np.zeros((len(quant),k))+M	 # matrix speciation times of length quant x no. fossils
		aa=np.zeros((len(quant),k))+a[0]  # matrix shape parameters (3) of length quant x no. fossils
		bb=np.zeros((len(quant),k))+b[0]  # matrix shape parameters (3) of length quant x no. fossils
		X=np.append(x[x>0],[x[x>0]]*(len(quant)-1)).reshape(len(quant), k) # matrix of fossils of shape quant x no. fossils
		if len(quant)>1:
			den = sum(G_density(-GM,1,l)) + small_number
			#lik_temp= sum(exp(-(int_q) + np.sum((logPERT4_density(MM,z[:,0:k],aa,bb,X)+log(q)), axis=1) ) \
			#* (G_density(-GM,1,l)/den) / (1-exp(-int_q))) / len(GM)
			#if lik_temp>0: lik=log(lik_temp)
			#else: lik = -inf
			# LOG TRANSF
			log_lik_temp = (-(int_q) + np.sum((logPERT4_density(MM,z[:,0:k],aa,bb,X)+log(q)), axis=1) )  \
			+ log(G_density(-GM,1,l)/den) - log(1-exp(-int_q))
			maxLogLikTemp = max(log_lik_temp)
			log_lik_temp_scaled = log_lik_temp-maxLogLikTemp
			lik = log(sum(exp(log_lik_temp_scaled))/ len(GM))+maxLogLikTemp
		else: lik= sum(-(int_q) + np.sum((logPERT4_density(MM,z[:,0:k],aa,bb,X)+log(q)), axis=1))
		lik += -sum(log(np.arange(1,k+1)))
	elif m==0: lik = HOMPP_lik(arg)
	else:
		C=M-c*(M-m)
		a = 1+ (4*(C-m))/(M-m)
		b = 1+ (4*(-C+M))/(M-m)
		lik = -q*(M-m) + sum(logPERT4_density(M,m,a,b,x)+log(q)) - log(1-exp(-q*(M-m)))
		lik += -sum(log(np.arange(1,k+1)))
	#if m==0: print i, lik, q, k, min(x),sum(exp(-(int_q)))
	return lik

def NHPPgamma(arg):
	[m,M,shapeGamma,q_rate,i,cov_par, ex_rate]=arg
	i=int(i)
	x=fossil[i]

	k=len(x[x>0])   # no. fossils for species i
	x=sort(x)[::-1] # reverse
	xB1= -(x-M)	 # distance fossil-ts

	if cov_par ==2: # transform preservation rate by trait value
		q=exp(log(q_rate)+cov_par*(con_trait[i]-parGAUS[0]))
	else: q=q_rate

	YangGamma=get_gamma_rates(shapeGamma)
	qGamma=YangGamma*q
	c=.5
	if len(x)>1 and m>0:
		C=M-.5*(M-m)
		a = 1+ (4*(C-m))/(M-m)
		b = 1+ (4*(-C+M))/(M-m)
		W=PERT4_density(M,m,a,b,x)
		PERT4_den=np.append(W, [W]*(pp_gamma_ncat-1)).reshape(pp_gamma_ncat,len(W)).T
		#lik=log( sum( (exp(-qGamma*(M-m)) * np.prod((PERT4_den*qGamma), axis=0) / (1-exp(-qGamma*(M-m))))*(1./pp_gamma_ncat)) )
		tempL=exp(-qGamma*(M-m))
		if max(tempL)<1:
			L=log(1-tempL)
			if np.isfinite(sum(L)):
				lik1=-qGamma*(M-m) + np.sum(log(PERT4_den*qGamma), axis=0) - L
				maxLogLik1 = max(lik1)
				lik2=lik1-maxLogLik1
				lik=log(sum(exp(lik2)*(1./pp_gamma_ncat)))+maxLogLik1
			else: lik=-100000
		else: lik=-100000
	elif m==0 and use_DA == 0: lik = HOMPP_lik(arg)
	else: lik=NHPP_lik(arg)
	return lik



###### BEGIN FUNCTIONS for FBD Range ########
def init_ts_te_FBDrange(FA,LO):
	ts,te = init_ts_te(FA,LO)
	min_dt = min(get_DT_FBDrange(ts,ts,te)[1:])
	print """\n Using the FBD-range likelihood function \n(Warnock, Heath, and Stadler; Paleobiology, in press)\n"""
	while min_dt <= 1:
		ts = ts+0.01*ts
		#ts = np.random.exponential(5, len(FA)) + FA
		#te = LO - np.random.exponential(5, len(LO))
		#te[te<0] = np.random.uniform(LO[te<0],0)
		dt = get_DT_FBDrange(ts,ts,te)
		min_dt = min(dt[1:])
	
	return ts, te

def get_DT_FBDrange(T,s,e): # returns the Diversity Trajectory of s,e at times T (x10 faster)
	T_list = np.array(list(T) + [np.max(T)+1])
	B=np.sort(T_list)-.000001 # the - .0001 prevents problems with identical ages
	#print "B", B
	#print "T", T
	#quit()
	ss1 = np.histogram(s,bins=B)[0]
	ee2 = np.histogram(e,bins=B)[0]
	DD=(ss1-ee2)[::-1]
	return np.cumsum(DD)[0:len(T)].astype(float)

def get_k(array_all_fossils, times):
	ff = np.histogram(array_all_fossils,bins=np.sort(times))[0]
	return ff[::-1]

def get_times_n_rates(timesQ, timesL, timesM, q_rates, Lt,Mt):
	Ltemp = 0+Lt
	Mtemp = 0+Mt
	merged = list(timesQ[1:])+ list(timesL[1:])+ list(timesM[1:])
	times = np.unique(merged)[::-1]
	psi = q_rates[np.digitize(times,timesQ[1:])]
	if len(Ltemp)>1: lam = Ltemp[np.digitize(times,timesL[1:])]
	else: lam = np.zeros(len(psi))+Ltemp[0]
	if len(Mtemp)>1: mu  = Mtemp[np.digitize(times,timesM[1:])]
	else: mu = np.zeros(len(psi))+Mtemp[0]
	times =np.insert(times,0, timesL[0])
	return times, psi, lam, mu

def calcAi(lam,mu,psi):
	Ai = abs(sqrt((lam-mu-psi)**2 + 4*lam*psi))
	return Ai

def calc_q(i, t, args):
	[intervalAs, lam, mu, psi, times, l, rho] = args
	intervalBs = np.zeros(l)
	intervalPs = np.zeros(l)

	def calc_p(i, t): # ti:
		if t ==0: return 1.
		ti = times[i+1]
		Ai = intervalAs[i]
		Bi = ((1 -2*(1-rho[i])* calc_p(i+1, ti)) * lam[i] +mu[i]+psi[i]) /Ai
		p = lam[i] + mu[i] + psi[i]
		p -= Ai * ( ((1+Bi) -(1-Bi) * exp(-Ai*(t-ti)) ) / ((1+Bi) +(1-Bi) * exp(-Ai*(t-ti) )) )
		p = p/(2. * lam[i])
		intervalBs[i] = Bi
		intervalPs[i] = p
		return p

	p = calc_p(i, t)
	Ai_t = intervalAs[i]*(t-times[i+1])
	qi_t = (log(4)-Ai_t) - (2* log( exp(-Ai_t) *(1-intervalBs[i]) + (1+intervalBs[i]) ) )
	return qi_t

def calc_qt(i, t, args):
	[intervalAs, lam, mu, psi, times, l, rho] = args
	qt = .5 * ( calc_q(i, t, args) - (lam[i]+mu[i]+psi[i])*(t-times[i+1]) )
	return qt

def likelihood_rangeFBD(times, psi, lam, mu, ts, te, k=[], intervalAs=[], int_indx=[], div_traj=[], rho=0):
	l  = len(times)-1 # number of intervals (combination of qShift, Kl, Km)
	if rho: pass
	else: rho = np.zeros(l)

	# only recompute if updating lam, mu, or psi
	if len(intervalAs) > 0: pass
	else: intervalAs = calcAi(lam,mu,psi)

	# only recompute if updating times or ts/te
	if len(int_indx) > 0:
		bint = int_indx[0]
		oint = int_indx[1]
		dint = int_indx[2]
	else:
		bint = np.digitize(ts, times)-1 # which interval ts happens
		oint = np.digitize(FA, times)-1 # which interval FA happens
		dint = np.digitize(te, times)-1 # which interval te happens
		bint[bint<0] = 0
		int_indx = [bint, oint, dint]

	# only need to recompute div_traj when updating ts/te
	if len(div_traj) > 0: pass
	else: div_traj = get_DT_FBDrange(ts,ts,te)
	# print div_traj
	# print np.sort(ts)
	# print np.sort(te)

	# only need to update when changing times
	if len(k) > 0: pass
	else: k = get_k(array_all_fossils, times)

	term0 = -log(lam[0])
	if rho[0]>0: term0 += log(rho[0])*(n-m)

	term1 = np.sum(k*log(psi))

	term2 = log(lam[bint])

	term3 = log(mu[dint])
	term3[te==0] = 0. # only counts for extinct lineages

	gamma_i = div_traj-1. # attachment points: diversity -1
	#gamma_i[gamma_i<=0] = 1
	gamma_i[0] = 1. # oldest range gets gamma= 1
	if np.min(gamma_i)<1: return [-np.inf, intervalAs, int_indx, div_traj, k]

	term4 = 0
	term4_c = 0

	if hasFoundPyRateC: # We use the C version for term 4
		term4_c = PyRateC_FBD_T4(tot_number_of_species, bint, dint, oint, intervalAs, lam, mu, psi, rho, gamma_i, times, ts, te, FA)

	if not hasFoundPyRateC or sanityCheckForPyRateC: # We use the python version if PyRateC not found or if sanity check is asked
		log_gamma_i = log(gamma_i)
		args = [intervalAs, lam, mu, psi, times, l, rho]

		for i in range(tot_number_of_species):

			term4_q  = calc_q(bint[i],ts[i],args)-calc_q(oint[i], FA[i],args)
			term4_qt = calc_qt(oint[i], FA[i],args)-calc_qt(dint[i], te[i],args)

			qj_1= 0
			for j in range(bint[i], oint[i]):
				qj_1 += calc_q(j+1, times[j+1], args)

			qtj_1= 0
			for j in range(oint[i], dint[i]):
				qtj_1 += calc_qt(j+1, times[j+1], args)

			term4_qj = qj_1 + qtj_1
			term4 += log_gamma_i[i] + term4_q + term4_qt + term4_qj # + term3[i] + term4[i]

		if hasFoundPyRateC and sanityCheckForPyRateC: # Sanity check only done if needed
			absDivergence = abs(term4 - term4_c)
			if absDivergence > sanityCheckThreshold:
				print "[WARNING] PyRateC_FBD_T4 diverged for more than ", sanityCheckThreshold, " (", absDivergence, ")"

	if hasFoundPyRateC:
		term4 = term4_c

	likelihood = np.sum([term1, term0+sum(term2),sum(term3), term4])
	res = [likelihood, intervalAs, int_indx, div_traj, k]

	return res

######  END FUNCTIONS for FBD Range  ########



###### BEGIN FUNCTIONS for BDMCMC ########

def born_prm(times, R, ind, tse):
	#B=max(1./times[0], np.random.beta(1,(len(R)+1))) # avoid time frames < 1 My
	#B=np.random.beta(1,(len(R)+1))
	alpha=zeros(len(R)+1)+lam_s
	B=max(1./times[0], np.random.dirichlet(alpha,1)[0][0])
	Q=np.diff(times*(1-B))
	ADD=-B*times[0]
	Q1=insert(Q, ind,ADD)
	Q2=times[0]+cumsum(Q1)
	Q3=insert(Q2,0,times[0])
	Q3[len(Q3)-1]=0
	#n_R= insert(R, ind, R[max(ind-1,0)]) # R[np.random.randint(0,len(R))]
	#print "old", R, n_R,

	# better proposals for birth events
	#if len(R)>1: R_init=mean(R)
	#else: R_init=R
	if np.random.random()>.5:
		n_R= insert(R, ind, init_BD(1))
	else:
		R_init = R[max(ind-1,0)]
		n_R= insert(R, ind,update_parameter(R_init,0,5,.05,1))


	#print "new", n_R
	n_times= sort(Q3)[::-1]
	go= 1
	for j in range(len(n_times)-1):
		up,lo=n_times[j],n_times[j+1]
		if len(np.intersect1d((tse <= up).nonzero()[0], (tse > lo).nonzero()[0]))<=1:
			go= 0
			#print len(np.intersect1d((tse <= up).nonzero()[0], (tse > lo).nonzero()[0])), up , lo

	if min(abs(np.diff(n_times)))<=1: return times, R
	elif go == 0: return times, R
	else: return n_times, n_R

	#R= insert(R, ind, init_BD(1))
	#n_times= sort(Q3)[::-1]
	#return n_times, R

def kill_prm(times, R, ind):
	P=np.diff(times)		 # time intervals
	Pi= abs(P[ind]/times[0])
	P2= P/(1-Pi)
	P3=np.delete(P2, ind)	# remove interval
	Q2=times[0]+cumsum(P3)   # re-adjust length remaining time frames
	Q3=insert(Q2,0,times[0]) # add root
	Q3[len(Q3)-1]=0
	R=np.delete(R,ind)	   # remove corresponding rate
	n_times= sort(Q3)[::-1]  # reverse array
	#ind_rm= max(1,ind)
	#n_times= np.delete(times,ind_rm)
	return n_times, R

def estimate_delta(likBDtemp, R,par,times, ts, te, cov_par, ind,deathRate,n_likBD,len_L):
	# ESTIMATE DELTAS (individual death rates)
	#print "now",R,times
	for temp in range(0,len(R)):
		if par=="l":
			temp_l=temp
			cov_par_one=cov_par[0]
		else:
			temp_l=temp+ind
			cov_par_one=cov_par[1]
		n_times, n_rates=kill_prm(times, R, temp)
		#print temp,n_rates,n_times

		tempL=0
		for temp1 in range(len(n_times)-1):
			up, lo = n_times[temp1], n_times[temp1+1]
			l = n_rates[temp1]
			#print up,lo,l,n_rates
			args=[ts, te, up, lo, l, par, cov_par_one,len_L]
			tempL+=BPD_partial_lik(args)
		#print "LIK", 	tempL, sum(likBDtemp[ind:ind+len(R)])
		D=min(tempL-sum(likBDtemp[ind:ind+len(R)]), 100) # to avoid overflows
		deathRate[temp_l]=exp(D)
	return deathRate #, n_likBD

def Alg_3_1(arg):
	[it,likBDtemp, ts, te, L,M, timesL, timesM, cov_par,len_L]=arg
	cont_time=0
	#T=max(ts)
	#priorBD= get_hyper_priorBD(timesL,timesM,L,M,T)
	while cont_time<len_cont_time:
		#print cont_time, sum(likBDtemp), len(L),len(M) #, timesL, L
		deathRate=zeros(len(likBDtemp))
		n_likBD=zeros(len(likBDtemp))

		# ESTIMATE DELTAS (individual death rates)
		if len(L)>1 :  # SPECIATION RATES
			deathRate=estimate_delta(likBDtemp, L,"l",timesL, ts, te, cov_par, 0,deathRate,n_likBD,len(L))
		if len(M)>1:  # EXTINCTION RATES
			deathRate=estimate_delta(likBDtemp, M,"m",timesM, ts, te, cov_par, len(L),deathRate,n_likBD,len(L))
		deltaRate=sum(deathRate)
		#print it, "DELTA:", round(deltaRate,3), "\t", deathRate, len(L), len(M),round(sum(likBDtemp),3)
		cont_time += np.random.exponential(1./min((deltaRate+birthRate), 100000))
		if cont_time>len_cont_time: break

		else: # UPDATE MODEL
			Pr_birth= birthRate/(birthRate+deltaRate)
			Pr_death= 1-Pr_birth
			#IND=-1
			#print sum(likBDtemp), Pr_birth, Pr_death, deathRate
			if np.random.random()<Pr_birth or len(L)+len(M)==2: # ADD PARAMETER
				LL=len(L)+len(M)
				if np.random.random()>.5 and use_Death_model == 0:
					ind=np.random.random_integers(0,len(L))
					timesL, L = born_prm(timesL, L, ind, ts[SP_in_window])
					IND=ind
				else:
					ind=np.random.random_integers(0,len(M))
					timesM, M = born_prm(timesM, M, ind, te[EX_in_window])
					IND=ind+len(timesL)-1
				if LL == len(L)+len(M): IND=-1
			else: # REMOVE PARAMETER
				probDeath=np.cumsum(deathRate/deltaRate) # cumulative prob (used to randomly sample one
				r=np.random.random()						  # parameter based on its deathRate)
				probDeath=sort(append(probDeath, r))
				ind=np.where(probDeath==r)[0][0] # just in case r==1
				if ind < len(L): timesL, L = kill_prm(timesL, L, ind)
				else: timesM, M = kill_prm(timesM, M, ind-len(L))
			# UPDATE LIKELIHOODS
			tempL=zeros(len(L)+len(M))
			tempP=zeros(len(L)+len(M))
			for temp_l in range(len(timesL)-1):
				up, lo = timesL[temp_l], timesL[temp_l+1]
				l = L[temp_l]
				args=[ts, te, up, lo, l, 'l', cov_par[0],len(L)]
				tempL[temp_l]=BPD_partial_lik(args)
			for temp_m in range(len(timesM)-1):
				up, lo = timesM[temp_m], timesM[temp_m+1]
				m = M[temp_m]
				args=[ts, te, up, lo, m, 'm', cov_par[1],len(L)]
				tempL[len(timesL)-1+temp_m]=BPD_partial_lik(args)
			likBDtemp=tempL

			#priorBDnew= get_hyper_priorBD(timesL,timesM,L,M,T)-priorBD
			#print IND, timesL, timesM
			#if IND > -1: likBDtemp[IND] += priorBDnew


	#if priorBDnew-priorBD >= log(np.random.random()):
	#	return likBDtemp, L,M, timesL, timesM, cov_par
	#else:
	#	return arg[1],arg[4], arg[5],arg[6], arg[7],cov_par
	return likBDtemp, L,M, timesL, timesM, cov_par

######	END FUNCTIONS for BDMCMC ######

####### BEGIN FUNCTIONS for RJMCMC #######
def random_choice(vector):
	ind = np.random.choice(range(len(vector)))
	return [vector[ind], ind]

def add_DoubleShift_RJ_rand_gamma(rates,times):
	r_time, r_time_ind = random_choice(np.diff(times))
	delta_t_prime	  = np.random.uniform(0,r_time,2)
	t_prime			= times[r_time_ind] + delta_t_prime
	times_prime		= np.sort(np.array(list(times)+list(t_prime)))[::-1]
	a,b				= shape_gamma_RJ,rate_gamma_RJ
	rate_prime		 = np.random.gamma(a,scale=1./b,size=2)
	log_q_prob		 = -sum(prior_gamma(rate_prime,a,b)) +log(abs(r_time)) # prob latent parameters: Gamma pdf, - (Uniform pdf )
	#print "PROB Q", prior_gamma(rate_prime,a,b), -log(1/abs(r_time))
	rates_prime		= np.insert(rates,r_time_ind+1,rate_prime)
	Jacobian		   = 0 # log(1)
	return rates_prime,times_prime,log_q_prob+Jacobian

def remove_DoubleShift_RJ_rand_gamma(rates,times):
	rm_shift_ind  = np.random.choice(range(2,len(times)-1))
	rm_shift_ind  = np.array([rm_shift_ind-1,rm_shift_ind])
	rm_shift_time = times[rm_shift_ind]
	dT			= abs(times[rm_shift_ind[1]+1]-times[rm_shift_ind[0]-1]) # if rm t_i: U[t_i-1, t_i+1]
	times_prime   = np.setdiff1d(times, rm_shift_time)[::-1]
	rm_rate	   = rates[rm_shift_ind]
	a,b		   = shape_gamma_RJ,rate_gamma_RJ
	log_q_prob	= sum(prior_gamma(rm_rate,a,b)) -log(dT) # log_q_prob_rm = 1/(log_q_prob_add)
	rates_prime   = np.delete(rates,rm_shift_ind)
	Jacobian	  = 0 # log(1)
	return rates_prime,times_prime,log_q_prob+Jacobian



def add_shift_RJ_rand_gamma(rates,times):
	if fix_edgeShift==1: # min and max bounds
		random_indx = np.random.choice(range(1,len(times)-2))
		r_time, r_time_ind = np.diff(times)[random_indx],random_indx
	elif fix_edgeShift==2: # max bound
		random_indx = np.random.choice(range(1,len(times)-1))
		r_time, r_time_ind = np.diff(times)[random_indx],random_indx
	elif fix_edgeShift==3: # min bound
		random_indx = np.random.choice(range(0,len(times)-2))
		r_time, r_time_ind = np.diff(times)[random_indx],random_indx
	else:
		r_time, r_time_ind = random_choice(np.diff(times))
	delta_t_prime	  = np.random.uniform(0,r_time)
	t_prime			= times[r_time_ind] + delta_t_prime
	times_prime		= np.sort(np.array(list(times)+[t_prime]))[::-1]
	a,b				= shape_gamma_RJ,rate_gamma_RJ
	rate_prime		 = np.random.gamma(a,scale=1./b)
	log_q_prob		 = -prior_gamma(rate_prime,a,b) +log(abs(r_time)) # prob latent parameters: Gamma pdf, - (Uniform pdf )
	#print "PROB Q", prior_gamma(rate_prime,a,b), -log(1/abs(r_time))
	rates_prime		= np.insert(rates,r_time_ind+1,rate_prime)
	Jacobian		   = 0 # log(1)
	return rates_prime,times_prime,log_q_prob+Jacobian

def remove_shift_RJ_rand_gamma(rates,times):
	if fix_edgeShift==1:  # min and max bounds
		random_indx = np.random.choice(range(2,len(times)-2))
	elif fix_edgeShift==2: # max bound
		random_indx = np.random.choice(range(2,len(times)-1))
	elif fix_edgeShift==3: # min bound
		random_indx = np.random.choice(range(1,len(times)-2))
	else:
		random_indx = np.random.choice(range(1,len(times)-1))
	rm_shift_ind  = random_indx
	rm_shift_time = times[rm_shift_ind]
	dT			= abs(times[rm_shift_ind+1]-times[rm_shift_ind-1]) # if rm t_i: U[t_i-1, t_i+1]
	times_prime   = times[times != rm_shift_time]
	rm_rate	   = rates[rm_shift_ind] ## CHECK THIS: could also be rates[rm_shift_ind-1] ???
	a,b		   = shape_gamma_RJ,rate_gamma_RJ
	log_q_prob	= prior_gamma(rm_rate,a,b) -log(dT) # log_q_prob_rm = 1/(log_q_prob_add)
	rates_prime   = rates[rates != rm_rate]
	Jacobian	  = 0 # log(1)
	return rates_prime,times_prime,log_q_prob+Jacobian

def add_shift_RJ_weighted_mean(rates,times ):
	if fix_edgeShift==1: # min and max bounds
		random_indx = np.random.choice(range(1,len(times)-2))
		r_time, r_time_ind = np.diff(times)[random_indx],random_indx
	elif fix_edgeShift==2: # max bound
		random_indx = np.random.choice(range(1,len(times)-1))
		r_time, r_time_ind = np.diff(times)[random_indx],random_indx
	elif fix_edgeShift==3: # min bound
		random_indx = np.random.choice(range(0,len(times)-2))
		r_time, r_time_ind = np.diff(times)[random_indx],random_indx
	else:
		r_time, r_time_ind = random_choice(np.diff(times))
	delta_t_prime		   = np.random.uniform(0,r_time)
	t_prime				 = times[r_time_ind] + delta_t_prime
	times_prime			 = np.sort(np.array(list(times)+[t_prime]))[::-1]
	time_i1				 = times[r_time_ind]
	time_i2				 = times[r_time_ind+1]
	p1 = (time_i1-t_prime)/(time_i1-time_i2)
	p2 = (t_prime-time_i2)/(time_i1-time_i2)
	u = np.random.beta(shape_beta_RJ,shape_beta_RJ)  #np.random.random()
	rate_i				  = rates[r_time_ind]
	rates_prime1			= exp( log(rate_i)-p2*log((1-u)/u) )
	rates_prime2			= exp( log(rate_i)+p1*log((1-u)/u) )
	rates_prime			 = np.insert(rates,r_time_ind+1,rates_prime2)
	#print p1+p2
	#print u,rates_prime1, rate_i,rates_prime2
	#print time_i1,times_prime,time_i2
	rates_prime[r_time_ind] = rates_prime1
	log_q_prob			  = log(abs(r_time))-prior_sym_beta(u,shape_beta_RJ) # prob latent parameters: Gamma pdf
	Jacobian				= 2*log(rates_prime1+rates_prime2)-log(rate_i)
	return rates_prime,times_prime,log_q_prob+Jacobian

def remove_shift_RJ_weighted_mean(rates,times):
	if fix_edgeShift==1:  # min and max bounds
		random_indx = np.random.choice(range(2,len(times)-2))
	elif fix_edgeShift==2: # max bound
		random_indx = np.random.choice(range(2,len(times)-1))
	elif fix_edgeShift==3: # min bound
		random_indx = np.random.choice(range(1,len(times)-2))
	else:
		random_indx = np.random.choice(range(1,len(times)-1))
	rm_shift_ind  = random_indx
	t_prime	   = times[rm_shift_ind]
	time_i1	   = times[rm_shift_ind-1]
	time_i2	   = times[rm_shift_ind+1]
	dT			= abs(times[rm_shift_ind+1]-times[rm_shift_ind-1]) # if rm t_i: U[t_i-1, t_i+1]
	times_prime   = times[times != t_prime]
	p1 = (time_i1-t_prime)/(time_i1-time_i2)
	p2 = (t_prime-time_i2)/(time_i1-time_i2)
	rate_i1	   = rates[rm_shift_ind-1]
	rate_i2	   = rates[rm_shift_ind]
	rate_prime	= exp(p1 *log(rate_i1) + p2 *log(rate_i2))
	#print p1,p2
	#print rate_i1, rate_i2,rate_prime
	#print t_prime, times_prime
	rm_rate	   = rates[rm_shift_ind]
	rates_prime   = rates[rates != rm_rate]
	rates_prime[rm_shift_ind-1] = rate_prime
	#print rates
	#print rates_prime
	u			 = 1./(1+rate_i2/rate_i1) # == rate_i1/(rate_i1+rate_i2)
	log_q_prob	= -log(dT)+prior_sym_beta(u,shape_beta_RJ) # log_q_prob_rm = 1/(log_q_prob_add)
	Jacobian	  = log(rate_prime)-(2*log(rate_i1+rate_i2))
	return rates_prime,times_prime,log_q_prob+Jacobian

def RJMCMC(arg):
	[L,M, timesL, timesM]=arg
	r=np.random.random(3)
	newL,newtimesL,log_q_probL = L,timesL,0
	newM,newtimesM,log_q_probM = M,timesM,0

	if r[0]>sample_shift_mu:
		# ADD/REMOVE SHIFT LAMBDA
		if r[1]>0.5:
			if r[2]>0.5 or allow_double_move==0:
				newL,newtimesL,log_q_probL = add_shift_RJ(L,timesL)
			else:
				newL,newtimesL,log_q_probL = add_DoubleShift_RJ_rand_gamma(L,timesL)
		# if 1-rate model this won't do anything, keeping the frequency of add/remove equal
		elif len(L)> min_allowed_n_rates: # defined for the edgeShift model
			if r[2]>0.5 or allow_double_move==0:
				newL,newtimesL,log_q_probL = remove_shift_RJ(L,timesL)
			elif len(L)>2:
				newL,newtimesL,log_q_probL = remove_DoubleShift_RJ_rand_gamma(L,timesL)
	else:
		# ADD/REMOVE SHIFT MU
		if r[1]>0.5:
			if r[2]>0.5 or allow_double_move==0:
				newM,newtimesM,log_q_probM = add_shift_RJ(M,timesM)
			else:
				newM,newtimesM,log_q_probM = add_DoubleShift_RJ_rand_gamma(M,timesM)
		# if 1-rate model this won't do anything, keeping the frequency of add/remove equal
		elif len(M)> min_allowed_n_rates: # defined for the edgeShift model
			if r[2]>0.5 or allow_double_move==0:
				newM,newtimesM,log_q_probM = remove_shift_RJ(M,timesM)
			elif len(M)>2:
				newM,newtimesM,log_q_probM = remove_DoubleShift_RJ_rand_gamma(M,timesM)

	return newL,newtimesL,newM,newtimesM,log_q_probL+log_q_probM

def get_post_rj_HP(xl,xm):
	G_shape_rjHP = 2. # 1.1
	G_rate_rjHP  = 1. # 0.1 # mode at 1
	n = 2 # sp, ex
	a = G_shape_rjHP + xl + xm
	b = G_rate_rjHP + n
	Poi_lambda_rjHP = np.random.gamma(a,1./b)
	#print "Mean Poi_lambda:", a/b
	return Poi_lambda_rjHP

def Poisson_prior(k,rate):
	return k*log(rate) - rate - sum(log(np.arange(1,k+1)))

####### BEGIN FUNCTIONS for DIRICHLET PROCESS PRIOR #######

def random_choice_P(vector):
	probDeath=np.cumsum(vector/sum(vector)) # cumulative prob (used to randomly sample one
	r=np.random.random()						  # parameter based on its deathRate)
	probDeath=sort(append(probDeath, r))
	ind=np.where(probDeath==r)[0][0] # just in case r==1
	return [vector[ind], ind]

def calc_rel_prob(log_lik):
	rel_prob=exp(log_lik-max(log_lik))
	return rel_prob/sum(rel_prob)

def G0(alpha=1.5,beta=5,n=1):
	#return np.array([np.random.random()])
	return np.random.gamma(shape=alpha,scale=1./beta,size=n)
	#return init_BD(n)


def DDP_gibbs_sampler(arg): # rate_type = "l" or "m" (for speciation/extinction respectively)
	[ts,te,parA,ind,time_frames,alpha_par_Dir,rate_type]=arg
	# par: parameters for each category
	n_data=len(ind)
	# GIBBS SAMPLER for NUMBER OF CATEGORIES - Algorithm 4. (Neal 2000)
	par=parA # parameters for each category
	eta = np.array([len(ind[ind==j]) for j in range(len(par))]) # number of elements in each category
	u1 = np.random.uniform(0,1,n_data) # init random numbers
	new_lik_vec=np.zeros(n_data) # store new sampled likelihoods
	new_alpha_par_Dir = 0 + cond_alpha_proposal(hp_gamma_shape,hp_gamma_rate,alpha_par_Dir,len(par),n_data)
	for i in range(0,n_data):
		up=time_frames[i]
		lo=time_frames[i+1]
		k1 = len(par)

		if len(ind[ind==ind[i]])==1: # is singleton
			k1 = k1 - 1
			par_k1 = par
			if u1[i]<= k1/(k1+1.): pass
			else: ind[i] = k1 + 1 # this way n_ic for singleton is not 0
		else: # is not singleton
			par_k1 = np.concatenate((par,G0()), axis=0)

		# construct prob vector FAST!
		lik_vec=BPD_partial_lik([ts,te,up,lo,par_k1,rate_type,0,1])
		rel_lik = calc_rel_prob(lik_vec)
		if len(par_k1)>len(eta): # par_k1 add one element only when i is not singleton
			eta[ind[i]] -= 1
			eta_temp=np.append(eta,new_alpha_par_Dir/(k1+1.))
		else: eta_temp = eta
		P=eta_temp*rel_lik

		# randomly sample a new value for indicator ind[i]
		IND = random_choice_P(P)[1]  # numpy.random.choice(a, size=None, replace= 1, p=None)
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

def get_init_values(mcmc_log_file,taxa_names):
	tbl = np.loadtxt(mcmc_log_file,skiprows=1)
	last_row = np.shape(tbl)[0]-1
	head = next(open(mcmc_log_file)).split()
	ts_index_temp = [head.index(i) for i in head if "_TS" in i]
	te_index_temp = [head.index(i) for i in head if "_TE" in i]
	ts_index,te_index = [],[]
	for ts_i in ts_index_temp:
		sp=head[ts_i].split("_TS")[0]
		if sp in taxa_names: ts_index.append(ts_i)
	for te_i in te_index_temp:
		sp=head[te_i].split("_TE")[0]
		if sp in taxa_names: te_index.append(te_i)
	if len(ts_index) != len(ts_index_temp):
		print "Excluded", len(ts_index) - len(ts_index_temp), "taxa"

	alpha_pp=1
	try:
		q_rates_index = np.array([head.index("alpha"), head.index("q_rate")])
		q_rates = tbl[last_row,q_rates_index]
	except:
		q_rates_index = [head.index(i) for i in head if "q_" in i]
		q_rates = tbl[last_row,q_rates_index]
		try:
			alpha_pp = tbl[last_row,head.index("alpha")]
		except: pass
	ts = tbl[last_row,ts_index]
	te = tbl[last_row,te_index]
	if len(fixed_times_of_shift)>0: # fixShift
		try:
			hyp_index = [head.index("hypL"), head.index("hypM")]
			l_index = [head.index(i) for i in head if "lambda_" in i]
			m_index = [head.index(i) for i in head if "mu_" in i]
			lam = tbl[last_row,l_index]
			mu  = tbl[last_row,m_index]
			hyp = tbl[last_row,hyp_index]
		except:
			lam = np.array([float(len(ts))/sum(ts-te)	  ])	# const rate ML estimator
			mu  = np.array([float(len(te[te>0]))/sum(ts-te)])	# const rate ML estimator
			hyp = np.ones(2)
	else:
		lam = np.array([float(len(ts))/sum(ts-te)	  ])	# const rate ML estimator
		mu  = np.array([float(len(te[te>0]))/sum(ts-te)])	# const rate ML estimator
		hyp = np.ones(2)
	#print

	return [ts,te,q_rates,lam,mu,hyp,alpha_pp]

########################## MCMC #########################################

def MCMC(all_arg):
	[it,n_proc, I,sample_freq, print_freq, temperatures, burnin, marginal_frames, arg]=all_arg
	if it==0: # initialize chain
		print("initializing chain...")
		if fix_SE == 1: tsA, teA = fixed_ts, fixed_te
		elif FBDrange==0: tsA, teA = init_ts_te(FA,LO)
		else:
			tsA, teA = init_ts_te_FBDrange(FA,LO)
			res_FBD_A = []
		if restore_chain == 1:
			tsA_temp, teA_temp = init_ts_te(FA,LO)
			tsA, teA = restore_init_values[0], restore_init_values[1]
			# avoid incompatibilities due to age randomizations
			# print len(tsA[tsA<FA]),len(tsA)
			tsA[tsA<FA]=FA[tsA<FA]+1
			teA[teA>LO]=LO[teA>LO]-1
			teA[teA<0]=teA_temp[teA<0]
		maxTSA = max(tsA)
		timesLA, timesMA = init_times(maxTSA,time_framesL,time_framesM, min(teA))
		if len(fixed_times_of_shift)>0: timesLA[1:-1],timesMA[1:-1]=fixed_times_of_shift,fixed_times_of_shift
		if fix_edgeShift > 0:
			print edgeShifts,fix_edgeShift
			if fix_edgeShift == 1:
				timesLA, timesMA = init_times(edgeShifts[0],time_framesL,time_framesM, edgeShifts[1]) # starting shift tims within allowed window
				timesLA[0],timesMA[0]= maxTSA,maxTSA
				timesLA[1],timesMA[1]= edgeShifts[0],edgeShifts[0]
				timesLA[-2],timesMA[-2]= edgeShifts[1],edgeShifts[1]
			elif fix_edgeShift == 2: # max age edge shift
				timesLA, timesMA = init_times(edgeShifts[0],time_framesL,time_framesM, 0) # starting shift tims within allowed window
				timesLA[1],timesMA[1]= edgeShifts[0],edgeShifts[0]
			elif fix_edgeShift == 3: # min age edge shift
				timesLA, timesMA = init_times(maxTSA,time_framesL,time_framesM, edgeShifts[0]) # starting shift tims within allowed window
				timesLA[-2],timesMA[-2]= edgeShifts[0],edgeShifts[0]
			# if len(edgeShifts)==1:
			# 	if args.edgeShifts[0] < np.inf:
			# 		timesLA, timesMA = init_times(edgeShifts[0],time_framesL,time_framesM, 0) # starting shift tims within allowed window
			# 		timesLA[1],timesMA[1]= edgeShifts[0],edgeShifts[0]
			# 	else:
			# 		timesLA, timesMA = init_times(maxTSA,time_framesL,time_framesM, edgeShifts[0]) # starting shift tims within allowed window
			# 		timesLA[1],timesMA[1]= edgeShifts[0],edgeShifts[0]
			# if len(edgeShifts)>1:
			# 	timesLA, timesMA = init_times(edgeShifts[0],time_framesL,time_framesM, edgeShifts[1]) # starting shift tims within allowed window
			# 	timesLA[0],timesMA[0]= maxTSA,maxTSA
			# 	timesLA[1],timesMA[1]= edgeShifts[0],edgeShifts[0]
			# 	timesLA[-2],timesMA[-2]= edgeShifts[1],edgeShifts[1]
			print "times", timesLA
			print "times", timesMA
				#quit()
		if TDI<3:
			LA = init_BD(len(timesLA))
			MA = init_BD(len(timesMA))
			if restore_chain == 1:
				LAt = restore_init_values[3]
				MAt = restore_init_values[4]
				if len(LAt) == len(LA): LA = LAt # if restored mcmc has different number of rates ignore them
				if len(MAt) == len(MA): MA = MAt
			if use_ADE_model >= 1: MA = np.random.uniform(3,5,len(timesMA)-1)
			if use_Death_model == 1: LA = np.ones(1)
			if useDiscreteTraitModel == 1:
				LA = init_BD(len(lengths_B_events)+1)
				MA = init_BD(len(lengths_D_events)+1)

		elif TDI==3 : ### DPP
			LA = init_BD(1) # init 1 rate
			MA = init_BD(1) # init 1 rate
			indDPP_L = np.zeros(len(timesLA)-1).astype(int) # init category indexes
			indDPP_M = np.zeros(len(timesLA)-1).astype(int) # init category indexes
			alpha_par_Dir_L = np.random.uniform(0,1) # init concentration parameters
			alpha_par_Dir_M = np.random.uniform(0,1) # init concentration parameters
		else:
			LA = init_BD(len(timesLA))
			MA = init_BD(len(timesMA))
			rj_cat_HP= 1

		q_ratesA,cov_parA = init_q_rates() # use 1 for symmetric PERT
		alpha_pp_gammaA = 1.
		if TPP_model == 1: # init multiple q rates
			q_ratesA = np.zeros(time_framesQ)+q_ratesA[1]
		if restore_chain == 1:
			q_ratesA = restore_init_values[2]
			if TPP_model == 1:
				if len(q_ratesA) != time_framesQ:
					q_ratesA=np.zeros(time_framesQ)+mean(q_ratesA)

		if est_COVAR_prior == 1:
			covar_prior = 1.
			cov_parA = np.random.random(3)*f_cov_par # f_cov_par is 0 or >0 depending on COVAR model
		else: covar_prior = covar_prior_fixed


		#if fix_hyperP == 0:	hyperPA=np.ones(2)
		hyperPA = hypP_par
		if restore_chain == 1: hyperPA = restore_init_values[5]

		if argsG == 0 and TPP_model == 0: q_ratesA[0]=1
		if argsG == 1 and TPP_model == 1 and restore_chain == 1: alpha_pp_gammaA=restore_init_values[6]
		SA=sum(tsA-teA)
		W_shapeA=1.

		if analyze_tree >=1:
			MA = LA*np.random.random()
			r_treeA = np.random.random()
			m_treeA = np.random.random()
			if analyze_tree==4:
				r_treeA = np.random.random(len(phylo_times_of_shift))+2.
				m_treeA = np.random.random(len(phylo_times_of_shift))
				if args_bdc:
					r_treeA = np.ones(len(phylo_times_of_shift))*0.8

	else: # restore values
		[itt, n_proc_,PostA, likA, priorA,tsA,teA,timesLA,timesMA,LA,MA,q_ratesA, cov_parA, lik_fossilA,likBDtempA]=arg
		SA=sum(tsA-teA)

	# start threads
	if num_processes>0: pool_lik = multiprocessing.Pool(num_processes) # likelihood
	if frac1>=0 and num_processes_ts>0: pool_ts = multiprocessing.Pool(num_processes_ts) # update ts, te
	tmp, marginal_lik, lik_tmp=0, zeros(len(temperatures)), 0

	while it<I:
		I_effective=I-burnin
		#if it==0: print (I_effective/len(temperatures))
		if it>0 and (it-burnin) % (I_effective/len(temperatures)) == 0 and it>burnin or it==I-1: # and it<I:
			if TDI==1:
				marginal_lik[tmp]=lik_tmp/((I_effective/len(temperatures))/sample_freq) # mean lik: Baele et al. 2012
				if it<I-1:
					tmp += 1
					lik_tmp=0
		temperature=temperatures[tmp]

		# update parameters
		ts,te=tsA, teA
		timesL,timesM=timesLA,timesMA
		try: # to make it work with MC3
			hyperP=hyperPA
			W_shape=W_shapeA
		except:
			hyperPA,W_shapeA=[1,1],[1]
			hyperP,W_shape=hyperPA,W_shapeA


		# GLOBALLY CHANGE TRAIT VALUE
		if model_cov >0:
			global con_trait
			con_trait=seed_missing(trait_values,meanGAUS,sdGAUS)

		if fix_SE == 1:
			rr=random.uniform(f_update_q,1)
			stop_update=0
			tsA, teA= fixed_ts, fixed_te
			lik_fossilA=np.zeros(1)
		elif it < fast_burnin:
			rr=random.uniform(f_update_se*0.95,1) # change update freq
			stop_update=I+1
		else:
			rr=random.uniform(0,1) #random.uniform(.8501, 1)
			stop_update=I+1

		if np.random.random() < 1./freq_Alg_3_1 and it>start_Alg_3_1 and TDI in [2,4]:
			stop_update=inf
			rr=1.5 # no updates

		if np.random.random() < 1./freq_dpp and TDI==3 and it > 1000: ### DPP
			stop_update=inf
			rr=1.5 # no updates

		if it>0 and (it-burnin) % (I_effective/len(temperatures)) == 0 and it>burnin or it==I-1: rr=1.5 # no updates when changing temp

		q_rates=zeros(len(q_ratesA))
		alpha_pp_gamma=alpha_pp_gammaA
		cov_par=zeros(3)
		L,M=zeros(len(LA)),zeros(len(MA))
		tot_L=sum(tsA-teA)
		hasting=0

		# autotuning
		if TDI != 1: tmp=0
		mod_d1= d1		   # window size ts, te
		mod_d3= list_d3[tmp] # window size rates
		mod_d4= list_d4[tmp] # window size shift times

		move_type = 0 # move types: 1) ts/te; 2) q rates; 3) timesL/M; 4) L/M rates;

		if rr<f_update_se: # ts/te
			move_type = 1
			ts,te=update_ts_te(tsA,teA,mod_d1)
			if use_gibbs_se_sampling or it < fast_burnin:
				if sum(timesL[1:-1])==sum(times_q_shift):
					ts,te = gibbs_update_ts_te(q_ratesA+LA,q_ratesA+MA,np.sort(np.array([np.inf,0]+times_q_shift))[::-1])
				else:
					times_q_temp = np.sort(np.array([np.inf,0]+times_q_shift))[::-1]
					q_temp_time = np.sort(np.unique(list(times_q_shift)+list(timesLA[1:])+list(timesMA[1:])))[::-1]
					q_rates_temp =  q_ratesA[np.digitize(q_temp_time,times_q_temp[1:])]
					if len(LA)==1:
						q_rates_temp_L = q_rates_temp + LA[0]
					else:
						q_rates_temp_L = q_rates_temp + LA[np.digitize(q_temp_time,timesLA[1:])]
					if len(MA)==1:
						q_rates_temp_M = q_rates_temp + MA[0]
					else:
						q_rates_temp_M = q_rates_temp + MA[np.digitize(q_temp_time,timesMA[1:])]
					ts,te = gibbs_update_ts_te(q_rates_temp_L,q_rates_temp_M,times_q_temp)

			tot_L=sum(ts-te)
		elif rr<f_update_q: # q/alpha
			move_type = 2
			q_rates=np.zeros(len(q_ratesA))+q_ratesA
			if TPP_model == 1:
				q_rates, hasting = update_q_multiplier(q_ratesA,d=d2[1],f=f_qrate_update)
				if np.random.random()> 1./len(q_rates) and argsG == 1:
					alpha_pp_gamma, hasting2 = update_multiplier_proposal(alpha_pp_gammaA,d2[0]) # shape prm Gamma
					hasting += hasting2
			elif np.random.random()>.5 and argsG == 1:
				q_rates[0], hasting=update_multiplier_proposal(q_ratesA[0],d2[0]) # shape prm Gamma
			else:
				q_rates[1], hasting=update_multiplier_proposal(q_ratesA[1],d2[1]) #  preservation rate (q)

		elif rr < f_update_lm: # l/m
			if np.random.random()<f_shift and len(LA)+len(MA)>2:
				move_type = 3
				if fix_edgeShift > 0:
					if fix_edgeShift == 1:
						timesL=update_times(timesLA, edgeShifts[0],edgeShifts[1],mod_d4,2,len(timesL)-2)
						timesM=update_times(timesMA, edgeShifts[0],edgeShifts[1],mod_d4,2,len(timesM)-2)
					elif fix_edgeShift == 2: # max age edge shift
						timesL=update_times(timesLA, edgeShifts[0],min(te),mod_d4,2,len(timesL)-1)
						timesM=update_times(timesMA, edgeShifts[0],min(te),mod_d4,2,len(timesM)-1)
					elif fix_edgeShift == 3: # min age edge shift
						timesL=update_times(timesLA,max(ts),edgeShifts[0],mod_d4,1,len(timesL)-2)
						timesM=update_times(timesMA,max(ts),edgeShifts[0],mod_d4,1,len(timesM)-2)

				else:
					maxTS = max(ts)
					minTE = min(te)
					timesL=update_times(timesLA, maxTS,minTE,mod_d4,1,len(timesL))
					timesM=update_times(timesMA, maxTS,minTE,mod_d4,1,len(timesM))
			else:
				move_type = 4
				if TDI<2: #
					if np.random.random()<.95 or est_hyperP == 0 or fix_hyperP == 1:
						L,M,hasting=update_rates(LA,MA,3,mod_d3)
						update_W_shape =1 
						if use_ADE_model == 1 and update_W_shape:
							W_shape, hasting2 = update_multiplier_proposal(W_shapeA,1.1)
							hasting+=hasting2
					else:
						hyperP,hasting = update_multiplier_proposal(hyperPA,d_hyperprior)
				else: # DPP or BDMCMC
						L,M,hasting=update_rates(LA,MA,3,mod_d3)

		elif rr<f_update_cov: # cov
			rcov=np.random.random()
			if est_COVAR_prior == 1 and rcov<0.05:
				covar_prior = get_post_sd(cov_parA[cov_parA>0]) # est hyperprior only based on non-zero rates
				stop_update=inf
			elif rcov < f_cov_par[0]: # cov lambda
				cov_par[0]=update_parameter_normal(cov_parA[0],-1000,1000,d5[0])
			elif rcov < f_cov_par[1]: # cov mu
				cov_par[1]=update_parameter_normal(cov_parA[1],-1000,1000,d5[1])
			else:
				cov_par[2]=update_parameter_normal(cov_parA[2],-1000,1000,d5[2])

		if constrain_time_frames == 1: timesM=timesL
		q_rates[(q_rates==0).nonzero()]=q_ratesA[(q_rates==0).nonzero()]
		L[(L==0).nonzero()]=LA[(L==0).nonzero()]
		M[(M==0).nonzero()]=MA[(M==0).nonzero()]
		cov_par[(cov_par==0).nonzero()]=cov_parA[(cov_par==0).nonzero()]
		max_ts = max(ts)
		timesL[0]=max_ts
		timesM[0]=max_ts
		if fix_SE == 0:
			if TPP_model == 1:
				q_time_frames = np.sort(np.array([max_ts,0]+times_q_shift))[::-1]

		# NHPP Lik: multi-thread computation (ts, te)
		# generate args lik (ts, te)
		if fix_SE == 0 and FBDrange==0:
			ind1=range(0,len(fossil))
			ind2=[]
			if it>0 and rr<f_update_se: # recalculate likelihood only for ts, te that were updated
				ind1=((ts-te != tsA-teA).nonzero()[0]).tolist()
				ind2=(ts-te == tsA-teA).nonzero()[0]
			lik_fossil=zeros(len(fossil))

			if len(ind1)>0 and it<stop_update and fix_SE == 0:
				# generate args lik (ts, te)
				z=zeros(len(fossil)*7).reshape(len(fossil),7)
				z[:,0]=te
				z[:,1]=ts
				z[:,2]=q_rates[0]   # shape prm Gamma
				z[:,3]=q_rates[1]   # baseline foss rate (q)
				z[:,4]=range(len(fossil))
				z[:,5]=cov_par[2]  # covariance baseline foss rate
				z[:,6]=M[len(M)-1] # ex rate
				if useDiscreteTraitModel == 1: z[:,6] = mean(M)
				args=list(z[ind1])

				if hasFoundPyRateC:
					if TPP_model == 1:
						# This uses the median for gamma rates
						YangGamma = [1]
						if argsG :
							YangGamma=get_gamma_rates(alpha_pp_gamma)

						lik_fossil = np.array(PyRateC_HPP_vec_lik(ind1, ts, te, q_time_frames, q_rates, YangGamma))
						# This uses the mean for gamma rates
						#lik_fossil2 = np.array(PyRateC_HPP_vec_lik(ind1, ts, te, q_time_frames, q_rates, pp_gamma_ncat, alpha_pp_gamma))

						# Check correctness of results by comparing with python version
						if sanityCheckForPyRateC == 1:
							lik_fossil2 = zeros(len(fossil))
							for j in range(len(ind1)):
								i=ind1[j] # which species' lik
								lik_fossil2[i] = HPP_vec_lik([te[i],ts[i],q_time_frames,q_rates,i,alpha_pp_gamma])

							absDivergence = abs(sum(lik_fossil2) - sum(lik_fossil))
							if absDivergence > sanityCheckThreshold:
								print "[WARNING] HPP_vec_lik diverged for more than ", sanityCheckThreshold, " (", absDivergence, ")"

					elif argsHPP == 1:
						YangGamma = [1]
						if argsG :
							YangGamma=get_gamma_rates(q_rates[0])

						lik_fossil = np.array(PyRateC_HOMPP_lik(ind1, ts, te, q_rates[1], YangGamma, cov_par[2], M[len(M)-1]))

						# Check correctness of results by comparing with python version
						if sanityCheckForPyRateC == 1:
							lik_fossil2 = zeros(len(fossil))
							for j in range(len(ind1)):
								i=ind1[j] # which species' lik
								lik_fossil2[i] = HOMPP_lik(args[j])

							absDivergence = abs(sum(lik_fossil2) - sum(lik_fossil))
							if absDivergence > sanityCheckThreshold:
								print "[WARNING] PyRateC_HOMPP_lik diverged for more than ", sanityCheckThreshold, " (", absDivergence, ")"

					else:
						YangGamma = [1]
						if argsG :
							YangGamma=get_gamma_rates(q_rates[0])
						lik_fossil = np.array(PyRateC_NHPP_lik(use_DA==1, ind1, ts, te, q_rates[1], YangGamma, cov_par[2], M[len(M)-1]))

						# Check correctness of results by comparing with python version
						if sanityCheckForPyRateC == 1:
							lik_fossil2 = zeros(len(fossil))
							for j in range(len(ind1)):
								i=ind1[j] # which species' lik
								if argsG == 1: lik_fossil2[i] = NHPPgamma(args[j])
								else: lik_fossil2[i] = NHPP_lik(args[j])

							absDivergence = abs(sum(lik_fossil2) - sum(lik_fossil))
							if absDivergence > sanityCheckThreshold:
								print "[WARNING] PyRateC_NHPP_lik diverged for more than ", sanityCheckThreshold, " (", absDivergence, ")"

				else:
					if num_processes_ts==0:
						for j in range(len(ind1)):
							i=ind1[j] # which species' lik
							if TPP_model == 1:  lik_fossil[i] = HPP_vec_lik([te[i],ts[i],q_time_frames,q_rates,i,alpha_pp_gamma])
							else:
								if argsHPP == 1 or  frac1==0: lik_fossil[i] = HOMPP_lik(args[j])
								elif argsG == 1: lik_fossil[i] = NHPPgamma(args[j])
								else: lik_fossil[i] = NHPP_lik(args[j])
					else:
						if TPP_model == 1: sys.exit("TPP_model model can only run on a signle processor")
						if argsHPP == 1 or  frac1==0: lik_fossil[ind1] = array(pool_ts.map(HOMPP_lik, args))
						elif argsG == 1: lik_fossil[ind1] = array(pool_ts.map(NHPPgamma, args))
						else: lik_fossil[ind1] = array(pool_ts.map(NHPP_lik, args))

			if it>0:
				lik_fossil[ind2] = lik_fossilA[ind2]

		# FBD range likelihood
		elif FBDrange==1:
			move_type = 0
			stop_update = 0
			if np.random.random()<0.01 and TDI==4:
				rj_cat_HP = get_post_rj_HP(len(LA),len(MA)) # est Poisson hyperprior on number of rates (RJMCMC)
				stop_update=inf
			elif np.random.random()< 0.3 and TDI==4:
				stop_update=0
				L,timesL,M,timesM,hasting2 = RJMCMC([LA,MA, timesLA, timesMA])
				hasting += hasting2
				move_type=0

			if it==0: stop_update=1
			# move types: 1) ts/te; 2) q rates; 3) timesL/M; 4) L/M rates;
			if it==0 or len(res_FBD_A)==0: move_type=0

			# only need to update if changed timesL/timesM or psi, lam, mu
			if move_type in [0,2,3,4] or len(res_FBD_A)==0 or np.max(ts) != maxTSA:
				times_fbd_temp, psi_fbd_temp, lam_fbd_temp, mu_fbd_temp = get_times_n_rates(q_time_frames, timesL, timesM, q_rates, L, M)
			else:
				[times_fbd_temp, psi_fbd_temp, lam_fbd_temp, mu_fbd_temp] = FBD_temp_A

			# times, psi, lam, mu, ts, te, k=0, intervalAs=0, int_indx=0, div_traj=0, rho=0
			if move_type == 1: # updated ts/te
				res_FBD = likelihood_rangeFBD(times_fbd_temp, psi_fbd_temp, lam_fbd_temp, mu_fbd_temp, ts, te, intervalAs=res_FBD_A[1], k=res_FBD_A[4])
			elif move_type in [2,4]: # updated q/s/e rates
				res_FBD = likelihood_rangeFBD(times_fbd_temp, psi_fbd_temp, lam_fbd_temp, mu_fbd_temp, ts, te,  int_indx=res_FBD_A[2], div_traj=res_FBD_A[3], k=res_FBD_A[4])
			elif move_type in [3]: # updated times
				res_FBD = likelihood_rangeFBD(times_fbd_temp, psi_fbd_temp, lam_fbd_temp, mu_fbd_temp, ts, te,  intervalAs=res_FBD_A[1], div_traj=res_FBD_A[3])
			else:
				res_FBD = likelihood_rangeFBD(times_fbd_temp, psi_fbd_temp, lam_fbd_temp, mu_fbd_temp, ts, te)
			# res = [likelihood, intervalAs, int_indx, div_traj, k]
			# if it % 1000==0:
			# 	print times_fbd_temp
			# 	print psi_fbd_temp
			# 	print lam_fbd_temp
			# 	print mu_fbd_temp
			# 	print res_FBD[3]
			# 	print res_FBD[4]

			lik_fossil = res_FBD[0]
			#if it>1: print lik_fossil,lik_fossilA


		else: lik_fossil=np.zeros(1)

		if FBDrange==0:
			if it>=stop_update or stop_update==inf: lik_fossil = lik_fossilA

		# pert_prior defines gamma prior on q_rates[1] - fossilization rate
		if TPP_model == 1:
			if pert_prior[1]>0:
				prior = sum(prior_gamma(q_rates,pert_prior[0],pert_prior[1]))+ prior_uniform(alpha_pp_gamma,0,20)
			else: # use hyperprior on Gamma rate on q
				hpGammaQ_shape = 1.01 # hyperprior is essentially flat
				hpGammaQ_rate =  0.1
				post_rate_prm_Gq = np.random.gamma( shape=hpGammaQ_shape+pert_prior[0]*len(q_rates), scale=1./(hpGammaQ_rate+sum(q_rates)) )
				prior = sum(prior_gamma(q_rates,pert_prior[0],post_rate_prm_Gq)) + prior_uniform(alpha_pp_gamma,0,20)
		else: prior = prior_gamma(q_rates[1],pert_prior[0],pert_prior[1]) + prior_uniform(q_rates[0],0,20)			
		if est_hyperP == 1: prior += ( prior_uniform(hyperP[0],0,20)+prior_uniform(hyperP[1],0,20) ) # hyperprior on BD rates


		### DPP begin
		if TDI==3:
			likBDtemp=0
			if stop_update != inf: # standard MCMC
				likBDtemp = BPD_lik_vec_times([ts,te,timesL,L[indDPP_L],M[indDPP_M]])
				#n_data=len(indDPP_L)
				#for time_frame_i in range(n_data):
				#	up=timesL[time_frame_i]
				#	lo=timesL[time_frame_i+1]
				#	likBDtemp+= BPD_partial_lik_vec([ts,te,up,lo,L[indDPP_L[time_frame_i]], "l"])
				#	likBDtemp+= BPD_partial_lik_vec([ts,te,up,lo,M[indDPP_M[time_frame_i]], "m"])
			else: ### RUN DPP GIBBS SAMPLER
				lik1, L, indDPP_L, alpha_par_Dir_L = DDP_gibbs_sampler([ts,te,L,indDPP_L,timesL,alpha_par_Dir_L,"l"])
				lik2, M, indDPP_M, alpha_par_Dir_M = DDP_gibbs_sampler([ts,te,M,indDPP_M,timesL,alpha_par_Dir_M,"m"])
				likBDtemp = lik1+lik2
		### DPP end

		elif FBDrange == 0:
			if TDI==4 and np.random.random()<0.01:
				rj_cat_HP = get_post_rj_HP(len(LA),len(MA)) # est Poisson hyperprior on number of rates (RJMCMC)
				stop_update=inf

			# Birth-Death Lik: construct 2D array (args partial likelihood)
			# parameters of each partial likelihood and prior (l)
			if stop_update != inf:
				if useDiscreteTraitModel == 1:
					if twotraitBD == 1:
						likBDtemp = BD_lik_discrete_trait_continuous([ts,te,L,M,cov_par])
					else:
						likBDtemp = BD_lik_discrete_trait([ts,te,L,M])

				elif use_ADE_model >= 1 and TPP_model == 1:
					likBDtemp = BD_age_lik_vec_times([ts,te,timesL,W_shape,M,q_rates,q_time_frames])
				elif fix_Shift == 1:
					if use_ADE_model == 0: likBDtemp = BPD_lik_vec_times([ts,te,timesL,L,M])
					else: likBDtemp = BD_age_lik_vec_times([ts,te,timesL,W_shape,M,q_rates])
				else:
					args=list()
					if use_ADE_model == 0: # speciation rate is not used under ADE model
						for temp_l in range(len(timesL)-1):
							up, lo = timesL[temp_l], timesL[temp_l+1]
							l = L[temp_l]
							args.append([ts, te, up, lo, l, 'l', cov_par[0],1])
					# parameters of each partial likelihood and prior (m)
					for temp_m in range(len(timesM)-1):
						up, lo = timesM[temp_m], timesM[temp_m+1]
						m = M[temp_m]
						if use_ADE_model == 0:
							args.append([ts, te, up, lo, m, 'm', cov_par[1],1])
						elif use_ADE_model >= 1:
							args.append([ts, te, up, lo, m, 'm', cov_par[1],W_shape,q_rates[1]])

					if hasFoundPyRateC and model_cov==0 and use_ADE_model == 0:
						likBDtemp = PyRateC_BD_partial_lik(ts, te, timesL, timesM, L, M)

						# Check correctness of results by comparing with python version
						if sanityCheckForPyRateC == 1:
							likBDtemp2=np.zeros(len(args))
							i=0
							for i in range(len(args)):
								likBDtemp2[i]=BPD_partial_lik(args[i])
								i+=1

							absDivergence = abs(sum(likBDtemp) - sum(likBDtemp2))
							if absDivergence > sanityCheckThreshold:
								print "[WARNING] BPD_partial_lik diverged for more than ", sanityCheckThreshold, " (", absDivergence, ")"

					else:
						if num_processes==0:
							likBDtemp=np.zeros(len(args))
							i=0
							for i in range(len(args)):
								likBDtemp[i]=BPD_partial_lik(args[i])
								i+=1
						# multi-thread computation of lik and prior (rates)
						else: likBDtemp = array(pool_lik.map(BPD_partial_lik, args))
					likBDtemp = likBDtemp
					#print likBDtemp - BD_lik_vec_times([ts,te,timesL,L,M])

			else:
				if TDI==2: # run BD algorithm (Alg. 3.1)
					sys.stderr = NO_WARN
					args=[it, likBDtempA,tsA, teA, LA,MA, timesLA, timesMA, cov_parA,len(LA)]
					likBDtemp, L,M, timesL, timesM, cov_par = Alg_3_1(args)
					sys.stderr = original_stderr

				elif TDI==4 and FBDrange==0: # run RJMCMC
					stop_update = 0
					L,timesL,M,timesM,hasting = RJMCMC([LA,MA, timesLA, timesMA])
					#print  L,timesL,M,timesM #,hasting
					args=list()
					for temp_l in range(len(timesL)-1):
						up, lo = timesL[temp_l], timesL[temp_l+1]
						l = L[temp_l]
						args.append([ts, te, up, lo, l, 'l', cov_par[0],1])
					for temp_m in range(len(timesM)-1):
						up, lo = timesM[temp_m], timesM[temp_m+1]
						m = M[temp_m]
						args.append([ts, te, up, lo, m, 'm', cov_par[1],1])

					if hasFoundPyRateC and model_cov==0:
						likBDtemp = PyRateC_BD_partial_lik(ts, te, timesL, timesM, L, M)

						# Check correctness of results by comparing with python version
						if sanityCheckForPyRateC == 1:
							likBDtemp2=np.zeros(len(args))
							i=0
							for i in range(len(args)):
								likBDtemp2[i]=BPD_partial_lik(args[i])
								i+=1

							absDivergence = abs(sum(likBDtemp) - sum(likBDtemp2))
							if absDivergence > sanityCheckThreshold:
								print "[WARNING] BPD_partial_lik diverged for more than ", sanityCheckThreshold, " (", absDivergence, ")"

					else:
						if num_processes==0:
							likBDtemp=np.zeros(len(args))
							i=0
							for i in range(len(args)):
								likBDtemp[i]=BPD_partial_lik(args[i])
								i+=1
						# multi-thread computation of lik and prior (rates)
						else: likBDtemp = array(pool_lik.map(BPD_partial_lik, args))
						#print sum(likBDtemp)-sum(likBDtempA),hasting,get_hyper_priorBD(timesL,timesM,L,M,T,hyperP)+(-log(max(ts)\
						# -min(te))*(len(L)-1+len(M)-1))-(get_hyper_priorBD(timesLA,timesMA,LA,MA,T,hyperP)+(-log(max(tsA)\
						# -min(teA))*(len(LA)-1+len(MA)-1))), len(L),len(M)

				# NHPP Lik: needs to be recalculated after Alg 3.1 or RJ (but only if NHPP+DA)
				if fix_SE == 0 and TPP_model == 0 and argsHPP == 0 and use_DA == 1:
					# NHPP calculated only if not -fixSE
					# generate args lik (ts, te)
					ind1=range(0,len(fossil))
					lik_fossil=zeros(len(fossil))
					# generate args lik (ts, te)
					z=zeros(len(fossil)*7).reshape(len(fossil),7)
					z[:,0]=te
					z[:,1]=ts
					z[:,2]=q_rates[0]   # shape prm Gamma
					z[:,3]=q_rates[1]   # baseline foss rate (q)
					z[:,4]=range(len(fossil))
					z[:,5]=cov_par[2]  # covariance baseline foss rate
					z[:,6]=M[len(M)-1] # ex rate
					args=list(z[ind1])
					if num_processes_ts==0:
						for j in range(len(ind1)):
							i=ind1[j] # which species' lik
							if argsG == 1: lik_fossil[i] = NHPPgamma(args[j])
							else: lik_fossil[i] = NHPP_lik(args[j])
					else:
						if argsG == 1: lik_fossil[ind1] = array(pool_ts.map(NHPPgamma, args))
						else: lik_fossil[ind1] = array(pool_ts.map(NHPP_lik, args))

		elif FBDrange == 1:
			likBDtemp = 0 # alrady included in lik_fossil


		lik= sum(lik_fossil) + sum(likBDtemp) + PoiD_const

		maxTs= max(ts)
		minTe= min(te)
		if TDI < 3:
			prior += sum(prior_times_frames(timesL, maxTs, minTe, lam_s))
			prior += sum(prior_times_frames(timesM, maxTs, minTe, lam_s))
		if TDI ==4:
			#prior_old = -log(max(ts)-max(te))*len(L-1)  #sum(prior_times_frames(timesL, max(ts),min(te), 1))
			#prior_old += -log(max(ts)-max(te))*len(M-1)  #sum(prior_times_frames(timesM, max(ts),min(te), 1))
			prior += -log(maxTs-minTe)*(len(L)-1+len(M)-1)
			prior += Poisson_prior(len(L),rj_cat_HP)+Poisson_prior(len(M),rj_cat_HP)
			#if it % 100 ==0: print len(L),len(M), prior_old, -log(max(ts)-min(te))*(len(L)-1+len(M)-1), hasting
						
			if get_min_diffTime(timesL)<=min_allowed_t or get_min_diffTime(timesM)<=min_allowed_t: prior = -np.inf

		priorBD= get_hyper_priorBD(timesL,timesM,L,M,maxTs,hyperP)
		if use_ADE_model >= 1:
			# M in this case is the vector of Weibull scales
			priorBD = sum(prior_normal(log(W_shape),2)) # Normal prior on log(W_shape): highest prior pr at W_shape=1

		prior += priorBD
		###
		if model_cov >0: prior+=sum(prior_normal(cov_par,covar_prior))

		# exponential prior on root age
		maxFA = max(FA)
		prior += prior_root_age(maxTs,maxFA,maxFA)

		# add tree likelihood
		if analyze_tree ==1: # independent rates model
			r_tree, h1 = update_multiplier_proposal(r_treeA,1.1) # net diversification
			m_tree, h2 = update_multiplier_proposal(m_treeA,1.1) # extinction rate
			l_tree = m_tree+r_tree
			tree_lik = treeBDlikelihood(tree_node_ages,l_tree,m_tree,rho=tree_sampling_frac)
			hasting = hasting+h1+h2
		elif analyze_tree ==2: # compatible model (BDC)
			r_tree = update_parameter(r_treeA, m=0, M=1, d=0.1, f=1)
			l_tree = (M[0]*r_tree) + (L[0]-M[0])
			m_tree = M[0]*r_tree
			tree_lik = treeBDlikelihood(tree_node_ages,l_tree,m_tree,rho=tree_sampling_frac)
		elif analyze_tree ==3: # equal rate model
			r_tree = 0
			l_tree = L[0]
			m_tree = M[0]
			tree_lik = treeBDlikelihood(tree_node_ages,l_tree,m_tree,rho=tree_sampling_frac)
		elif analyze_tree ==4: # skyline independent model
			m_tree,r_tree,h1,h2 = m_treeA+0., r_treeA+0.,0,0
			if args_bdc: # BDC model
				ind = np.random.choice(range(len(m_tree)))
				r_tree[ind] = update_parameter(r_treeA[ind], 0, 1, 0.1, 1)
				l_tree = L[::-1] - M[::-1] + M[::-1]*r_tree
				m_tree = M[::-1]*r_tree #  so mu > 0
			else:
				m_tree, h2 = update_q_multiplier(m_treeA,d=1.1,f=0.5) # extinction rate
				r_tree, h1 = update_q_multiplier(r_treeA,d=1.1,f=0.5) # speciation rate
				l_tree = r_tree*m_tree # this allows extinction > speciation
				# args = (x,t,l,mu,sampling,posdiv=0,survival=1,groups=0)

			if np.min(l_tree)<=0:
				tree_lik = -np.inf
			else:
				tree_lik = treeBDlikelihoodSkyLine(tree_node_ages,phylo_times_of_shift,l_tree,m_tree,tree_sampling_frac)
				hasting = hasting+h1+h2
				prior += sum(prior_gamma(l_tree,1.1,1)) + sum(prior_gamma(m_tree,1.1,1))
		else:
			tree_lik = 0

		if temperature==1:
			tempMC3=1./(1+n_proc*temp_pr)
			lik_alter=lik
		else:
			tempMC3=1
			lik_alter=(sum(lik_fossil)+ PoiD_const) + (sum(likBDtemp)+ PoiD_const)*temperature
		Post=lik_alter+prior+tree_lik
		accept_it = 0
		if it==0:
			accept_it = 1
			PostA = Post
		if it>0 and (it-burnin) % (I_effective/len(temperatures)) == 0 and it>burnin or it==I-1:
			accept_it = 1 # when temperature changes always accept first iteration
			PostA = Post

		if rr<f_update_se and use_gibbs_se_sampling==1:
			accept_it = 1

		if rr<f_update_se and it < fast_burnin:
			accept_it = 1

		#print Post, PostA, q_ratesA, sum(lik_fossil), sum(likBDtemp),  prior
		#print sum(lik_fossil), sum(likBDtemp), PoiD_const
		if Post>-inf and Post<inf:
			r_acc = log(np.random.random())
			if Post*tempMC3-PostA*tempMC3 + hasting >= r_acc or stop_update==inf and TDI in [2,3,4] or accept_it==1: #
				likBDtempA=likBDtemp
				PostA=Post
				priorA=prior
				likA=lik
				timesLA=timesL
				timesMA=timesM
				LA,MA=L,M
				hyperPA=hyperP
				tsA,teA=ts,te
				SA=sum(tsA-teA)
				q_ratesA=q_rates
				alpha_pp_gammaA=alpha_pp_gamma
				lik_fossilA=lik_fossil
				cov_parA=cov_par
				W_shapeA=W_shape
				tree_likA = tree_lik
				if analyze_tree >=1:
					r_treeA = r_tree
					m_treeA = m_tree
				if FBDrange==1:
					res_FBD_A = res_FBD
					FBD_temp_A = [times_fbd_temp, psi_fbd_temp, lam_fbd_temp, mu_fbd_temp]


		if it % print_freq ==0 or it==burnin:
			try: l=[round(y, 2) for y in [PostA, likA, priorA, SA]]
			except:
				print "An error occurred."
				print PostA,Post,lik, prior, prior_root_age(max(ts),max(FA),max(FA)), priorBD, max(ts),max(FA)
				print prior_gamma(q_rates[1],pert_prior[0],pert_prior[1]) + prior_uniform(q_rates[0],0,20)
				quit()
			if it>burnin and n_proc==0:
				print_out= "\n%s\tpost: %s lik: %s (%s, %s) prior: %s tot.l: %s" \
				% (it, l[0], l[1], round(sum(lik_fossilA), 2), round(sum(likBDtempA)+ PoiD_const, 2),l[2], l[3])
				if TDI==1: print_out+=" beta: %s" % (round(temperature,4))
				if TDI in [2,3,4]: print_out+=" k: %s" % (len(LA)+len(MA))
				print(print_out)
				#if TDI==1: print "\tpower posteriors:", marginal_lik[0:10], "..."
				if TDI==3:
					print "\tind L", indDPP_L
					print "\tind M", indDPP_M
				else:
					print "\tt.frames:", timesLA, "(sp.)"
					print "\tt.frames:", timesMA, "(ex.)"
				if use_ADE_model >= 1:
					print "\tWeibull.shape:", round(W_shapeA,3)
					print "\tWeibull.scale:", MA, 1./MA
				else:
					print "\tsp.rates:", LA
					print "\tex.rates:", MA
				if analyze_tree ==1:
					print np.array([tree_likA, r_treeA+m_treeA, m_treeA])
				if analyze_tree==2:
					ltreetemp,mtreetemp = (M[0]*r_tree) + (L[0]-M[0]), M[0]*r_tree
					print np.array([tree_likA,ltreetemp,mtreetemp,ltreetemp-mtreetemp,LA[0]-MA[0]])
				if analyze_tree==4:
					ltreetemp,mtreetemp = list(r_treeA[::-1]*m_treeA[::-1]), list(m_treeA[::-1])
					print np.array([tree_likA] + ltreetemp + mtreetemp)

				if est_hyperP == 1: print "\thyper.prior.par", hyperPA


				if model_cov>=1:
					print "\tcov. (sp/ex/q):", cov_parA
					if est_COVAR_prior == 1: print "\tHP_covar:",round(covar_prior,3)
 				if fix_SE == 0:
					if TPP_model == 1:
						print "\tq.rates:", q_ratesA, "\n\tGamma.prm:", round(alpha_pp_gammaA,3)
					else: print "\tq.rate:", round(q_ratesA[1], 3), "\tGamma.prm:", round(q_ratesA[0], 3)
					print "\tts:", tsA[0:5], "..."
					print "\tte:", teA[0:5], "..."
			if it<=burnin and n_proc==0: print("\n%s*\tpost: %s lik: %s prior: %s tot length %s" \
			% (it, l[0], l[1], l[2], l[3]))

		if n_proc != 0: pass
		elif it % sample_freq ==0 and it>=burnin or it==0 and it>=burnin:
			s_max=np.max(tsA)
			if fix_SE == 0:
				if TPP_model == 0: log_state = [it,PostA, priorA, sum(lik_fossilA), likA-sum(lik_fossilA), q_ratesA[1], q_ratesA[0]]
				else:
					log_state= [it,PostA, priorA, sum(lik_fossilA), likA-sum(lik_fossilA)] + list(q_ratesA) + [alpha_pp_gammaA]
					if pert_prior[1]==0:
						log_state += [post_rate_prm_Gq]
			else:
				log_state= [it,PostA, priorA, likA-sum(lik_fossilA)]

			if model_cov>=1:
				log_state += cov_parA[0], cov_parA[1],cov_parA[2]
				if est_COVAR_prior == 1: log_state += [covar_prior]

			if TDI<2: # normal MCMC or MCMC-TI
				log_state += s_max,np.min(teA)
				if TDI==1: log_state += [temperature]
				if est_hyperP == 1: log_state += list(hyperPA)
				if use_ADE_model == 0:
					log_state += list(LA)
				elif use_ADE_model == 1:
					log_state+= [W_shapeA]
				elif use_ADE_model == 2:
					# this correction is for the present (recent sp events are unlikely to show up)
					xtemp = np.linspace(0,5,101)
					pdf_q_sampling = np.round(1-exp(-q_ratesA[1]*xtemp),2)
					#try:
					#	q95 = np.min([xtemp[pdf_q_sampling==0.75][0],0.25*s_max]) # don't remove more than 25% of the time window
					#except: q95 = 0.25*s_max
					q95 = min(tsA[tsA>0])
					# estimate sp rate based on ex rate and ratio between observed sp and ex events
					corrSPrate = float(len(tsA[tsA>q95]))/max(1,len(teA[teA>q95])) * 1./MA
					log_state+= list(corrSPrate)

				if use_ADE_model <= 1:
					log_state += list(MA) # This is W_scale in the case of ADE models
				if use_ADE_model == 2:
					log_state += list(1./MA) # when using model 2 shape = 1, and 1/scale = extinction rate

				if use_ADE_model >= 1:
					log_state+= list(MA * gamma(1 + 1./W_shapeA))
				if fix_Shift== 0:
					log_state += list(timesLA[1:-1])
					log_state += list(timesMA[1:-1])
				if analyze_tree ==1:
					log_state += [tree_likA, r_treeA+m_treeA, m_treeA]
				if analyze_tree ==2:
					log_state += [tree_likA, (MA[0]*r_treeA) + (LA[0]-MA[0]), MA[0]*r_treeA]
				if analyze_tree ==3:
					log_state += [tree_likA, LA[0], MA[0]]
				if analyze_tree ==4:
					if args_bdc: # BDC model
						ltreetemp = (MA[::-1]*r_treeA) + (LA[::-1]-MA[::-1])
						mtreetemp =  MA[::-1]*r_treeA
						ltreetemp = list(ltreetemp[::-1])
						mtreetemp = list(mtreetemp[::-1])
					else:
						ltreetemp,mtreetemp = list(r_treeA[::-1]*m_treeA[::-1]), list(m_treeA[::-1])
					log_tree_lik_temp = [tree_likA] + ltreetemp + mtreetemp
					log_state += log_tree_lik_temp

			elif TDI == 2: # BD-MCMC
				log_state+= [len(LA), len(MA), s_max,min(teA)]
			elif TDI == 3: # DPP
				log_state+= [len(LA), len(MA), alpha_par_Dir_L,alpha_par_Dir_M, s_max,min(teA)]
			elif TDI ==4: # RJMCMC
				log_state+= [len(LA), len(MA),rj_cat_HP, s_max,min(teA)]

			if useDiscreteTraitModel == 1:
				for i in range(len(lengths_B_events)): log_state += [sum(tsA[ind_trait_species==i]-teA[ind_trait_species==i])]
			log_state += [SA]
			if fix_SE == 0:
				log_state += list(tsA)
				log_state += list(teA)
			wlog.writerow(log_state)
			logfile.flush()
			os.fsync(logfile)

			lik_tmp += sum(likBDtempA)

			if log_marginal_rates_to_file==1:
				if TDI in [0,2,4] and n_proc==0 and use_ADE_model == 0 and useDiscreteTraitModel == 0:
					margL=np.zeros(len(marginal_frames))
					margM=np.zeros(len(marginal_frames))
					if useBounded_BD == 1: min_marginal_frame = boundMin
					else: min_marginal_frame = min(LO)

					for i in range(len(timesLA)-1): # indexes of the 1My bins within each timeframe
						ind=np.intersect1d(marginal_frames[marginal_frames<=timesLA[i]],marginal_frames[marginal_frames>=max(min_marginal_frame,timesLA[i+1])])
						j=array(ind)
						margL[j]=LA[i]
					for i in range(len(timesMA)-1): # indexes of the 1My bins within each timeframe
						ind=np.intersect1d(marginal_frames[marginal_frames<=timesMA[i]],marginal_frames[marginal_frames>=max(min_marginal_frame,timesMA[i+1])])
						j=array(ind)
						margM[j]=MA[i]
					marginal_rates(it, margL, margM, marginal_file, n_proc)
				if n_proc==0 and TDI==3: # marg rates DPP | times of shift are fixed and equal for L and M
					margL=zeros(len(marginal_frames))
					margM=zeros(len(marginal_frames))
					for i in range(len(timesLA)-1): # indexes of the 1My bins within each timeframe
						ind=np.intersect1d(marginal_frames[marginal_frames<=timesLA[i]],marginal_frames[marginal_frames>=timesLA[i+1]])
						j=array(ind)
						margL[j]=LA[indDPP_L[i]]
						margM[j]=MA[indDPP_M[i]]
					marginal_rates(it, margL, margM, marginal_file, n_proc)
			elif TDI in [0,2,4]:
				w_marg_sp.writerow(list(LA) + list(timesLA[1:len(timesLA)-1]))
				marginal_sp_rate_file.flush()
				os.fsync(marginal_sp_rate_file)
				w_marg_ex.writerow(list(MA) + list(timesMA[1:len(timesMA)-1]))
				marginal_ex_rate_file.flush()
				os.fsync(marginal_ex_rate_file)




		it += 1
	if TDI==1 and n_proc==0: marginal_likelihood(marginal_file, marginal_lik, temperatures)
	if use_seq_lik == 0:
		pool_lik.close()
		pool_lik.join()
		if frac1>=0:
			pool_ts.close()
			pool_ts.join()
	return [it, n_proc,PostA, likA, priorA,tsA,teA,timesLA,timesMA,LA,MA,q_ratesA, cov_parA,lik_fossilA,likBDtempA]

def marginal_rates(it, margL,margM, marginal_file, run):
	log_state= [it]
	log_state += list(margL)
	log_state += list(margM)
	log_state += list(margL-margM)
	wmarg.writerow(log_state)
	marginal_file.flush()
	os.fsync(marginal_file)

def marginal_likelihood(marginal_file, l, t):
	mL=0
	for i in range(len(l)-1): mL+=((l[i]+l[i+1])/2.)*(t[i]-t[i+1]) # Beerli and Palczewski 2010
	print "\n Marginal likelihood:", mL
	o= "\n Marginal likelihood: %s\n\nlogL: %s\nbeta: %s" % (mL,l,t)
	marginal_file.writelines(o)
	marginal_file.close()

########################## PARSE ARGUMENTS #######################################
p = argparse.ArgumentParser() #description='<input file>')

p.add_argument('-v',		 action='version', version=version_details)
p.add_argument('-seed',	  type=int, help='random seed', default=-1,metavar=-1)
p.add_argument('-useCPPlib', type=int, help='Use C++ library if available (boolean)', default=1,metavar=1)
p.add_argument('-cite',	  help='print PyRate citation', action='store_true', default=False)
p.add_argument('input_data', metavar='<input file>', type=str,help='Input python file - see template',default=[],nargs='*')
p.add_argument('-j',		 type=int, help='number of data set in input file', default=1, metavar=1)
p.add_argument('-trait',	 type=int, help='number of trait for Cov model', default=1, metavar=1)
p.add_argument('-logT',	  type=int, help='Transform trait (or rates for -plotRJ): 0) False, 1) Ln(x), 2) Log10(x)', default=0, metavar=0)
p.add_argument("-N",		 type=float, help='number of exant species')
p.add_argument("-wd",		type=str, help='path to working directory', default="")
p.add_argument("-out",	   type=str, help='output tag', default="")
p.add_argument('-singleton', type=float, help='Remove singletons (min no. occurrences)', default=0, metavar=0)
p.add_argument('-frac_sampled_singleton', type=float, help='Random fraction of singletons not removed', default=0, metavar=0)
p.add_argument("-rescale",   type=float, help='Rescale data (e.g. -rescale 1000: 1 -> 1000, time unit = 1Ky)', default=1, metavar=1)
p.add_argument("-translate", type=float, help='Shift data (e.g. -translate 10: 1My -> 10My)', default=0, metavar=0)
p.add_argument('-d',		 type=str,help="Load SE table",metavar='<input file>',default="")
p.add_argument('-clade',	 type=int, help='clade analyzed (set to -1 to analyze all species)', default=-1, metavar=-1)
p.add_argument('-trait_file',type=str,help="Load trait table",metavar='<input file>',default="")
p.add_argument('-restore_mcmc',type=str,help="Load mcmc.log file",metavar='<input file>',default="")
p.add_argument('-filter',	 type=float,help="Filter lineages with all occurrences within time range ",default=[inf,0], metavar=inf, nargs=2)
p.add_argument('-filter_taxa',type=str,help="Filter lineages within list (drop all others) ",default="", metavar="taxa_file")
p.add_argument('-initDiv',	type=int, help='Number of initial lineages (option only available with -d SE_table or -fixSE)', default=0, metavar=0)
p.add_argument('-PPmodeltest',help='Likelihood testing among preservation models', action='store_true', default=False)
p.add_argument('-log_marginal_rates',type=int,help='0) save summary file, default for -A 4; 1) save marginal rate file, default for -A 0,2 ', default=-1,metavar=-1)
# phylo test
p.add_argument('-tree',	   type=str,help="Tree file (NEXUS format)",default="", metavar="")
p.add_argument('-sampling',   type=float,help="Taxon sampling (phylogeny)",default=1., metavar=1.)
p.add_argument('-bdc',	  help='Run BDC:Compatible model', action='store_true', default=False)
p.add_argument('-eqr',	  help='Run BDC:Equal rate model', action='store_true', default=False)

# PLOTS AND OUTPUT
p.add_argument('-plot',	   metavar='<input file>', type=str,help="RTT plot (type 1): provide path to 'marginal_rates.log' files or 'marginal_rates' file",default="")
p.add_argument('-plot2',	  metavar='<input file>', type=str,help="RTT plot (type 2): provide path to 'marginal_rates.log' files or 'marginal_rates' file",default="")
p.add_argument('-plot3',	  metavar='<input file>', type=str,help="RTT plot for fixed number of shifts: provide 'mcmc.log' file",default="")
p.add_argument('-plotRJ',	 metavar='<input file>', type=str,help="RTT plot for runs with '-log_marginal_rates 0': provide path to 'mcmc.log' files",default="")
p.add_argument('-plotQ',	  metavar='<input file>', type=str,help="Plot preservation rates through time: provide 'mcmc.log' file and '-qShift' argument ",default="")
p.add_argument('-grid_plot',  type=float, help='Plot resolution in Myr (only for plot3 and plotRJ commands). If set to 0: 100 equal time bins', default=0, metavar=0)
p.add_argument('-root_plot',  type=float, help='User-defined root age for RTT plots', default=0, metavar=0)
p.add_argument('-min_age_plot',type=float, help='User-defined minimum age for RTT plots (only with plotRJ option)', default=0, metavar=0)
p.add_argument('-tag',        metavar='<*tag*.log>', type=str,help="Tag identifying files to be combined and plotted (-plot and -plot2) or summarized in SE table (-ginput)",default="")
p.add_argument('-ltt',        type=int,help='1) Plot lineages-through-time; 2) plot Log10(LTT)', default=0, metavar=0)
p.add_argument('-mProb',      type=str,help="Input 'mcmc.log' file",default="")
p.add_argument('-BF',         type=str,help="Input 'marginal_likelihood.txt' files",metavar='<2 input files>',nargs='+',default=[])
p.add_argument("-data_info",  help='Summary information about an input data', action='store_true', default=False)
p.add_argument('-SE_stats',   type=str,help="Calculate and plot stats from SE table:",metavar='<extinction rate at the present, bin_size, #_simulations>',nargs='+',default=[])
p.add_argument('-ginput',     type=str,help='generate SE table from *mcmc.log files', default="", metavar="<path_to_mcmc.log>")
p.add_argument('-combLog',    type=str,help='Combine (and resample) log files', default="", metavar="<path_to_log_files>")
p.add_argument('-combLogRJ',  type=str,help='Combine (and resample) all log files form RJMCMC', default="", metavar="<path_to_log_files>")
p.add_argument('-resample',   type=int,help='Number of samples for each log file (-combLog). Use 0 to keep all samples.', default=0, metavar=0)
p.add_argument('-col_tag',	type=str,help='Columns to be combined using combLog', default=[], metavar="column names",nargs='+')
p.add_argument('-check_names',type=str,help='Automatic check for typos in taxa names (provide SpeciesList file)', default="", metavar="<*_SpeciesList.txt file>")
p.add_argument('-reduceLog',  type=str,help='Reduce file size (mcmc.log) to quickly assess convergence', default="", metavar="<*_mcmc.log file>")

# MCMC SETTINGS
p.add_argument('-n',	  type=int, help='mcmc generations',default=10000000, metavar=10000000)
p.add_argument('-s',	  type=int, help='sample freq.', default=1000, metavar=1000)
p.add_argument('-p',	  type=int, help='print freq.',  default=1000, metavar=1000)
p.add_argument('-b',	  type=float, help='burnin', default=0, metavar=0)
p.add_argument('-fast_burnin',	  type=float, help='n. fast-burnin generations', default=0, metavar=0)
p.add_argument('-thread', type=int, help='no. threads used for BD and NHPP likelihood respectively (set to 0 to bypass multi-threading)', default=[0,0], metavar=4, nargs=2)

# MCMC ALGORITHMS
p.add_argument('-A',		type=int, help='0) parameter estimation, 1) marginal likelihood, 2) BDMCMC, 3) DPP, 4) RJMCMC', default=4, metavar=4)
p.add_argument("-use_DA",   help='Use data augmentation for NHPP likelihood opf extant taxa', action='store_true', default=False)
p.add_argument('-r',		type=int,   help='MC3 - no. MCMC chains', default=1, metavar=1)
p.add_argument('-t',		type=float, help='MC3 - temperature', default=.03, metavar=.03)
p.add_argument('-sw',	   type=float, help='MC3 - swap frequency', default=100, metavar=100)
p.add_argument('-M',		type=int,   help='BDMCMC/RJMCMC - frequency of model update', default=10, metavar=10)
p.add_argument('-B',		type=int,   help='BDMCMC - birth rate', default=1, metavar=1)
p.add_argument('-T',		type=float, help='BDMCMC - time of model update', default=1.0, metavar=1.0)
p.add_argument('-S',		type=int,   help='BDMCMC - start model update', default=1000, metavar=1000)
p.add_argument('-k',		type=int,   help='TI - no. scaling factors', default=10, metavar=10)
p.add_argument('-a',		type=float, help='TI - shape beta distribution', default=.3, metavar=.3)
p.add_argument('-dpp_f',	type=float, help='DPP - frequency ', default=500, metavar=500)
p.add_argument('-dpp_hp',   type=float, help='DPP - shape of gamma HP on concentration parameter', default=2., metavar=2.)
p.add_argument('-dpp_eK',   type=float, help='DPP - expected number of rate categories', default=2., metavar=2.)
p.add_argument('-dpp_grid', type=float, help='DPP - size of time bins',default=1.5, metavar=1.5)
p.add_argument('-dpp_nB',   type=float, help='DPP - number of time bins',default=0, metavar=0)
p.add_argument('-rj_pr',	   type=float, help='RJ - proposal (0: Gamma, 1: Weighted mean) ', default=1, metavar=1)
p.add_argument('-rj_Ga',	   type=float, help='RJ - shape of gamma proposal (if rj_pr 0)', default=1.5, metavar=1.5)
p.add_argument('-rj_Gb',	   type=float, help='RJ - rate of gamma proposal (if rj_pr 0)',  default=3., metavar=3.)
p.add_argument('-rj_beta',	 type=float, help='RJ - shape of beta multiplier (if rj_pr 1)',default=10, metavar=10)
p.add_argument('-rj_dm',	   type=float, help='RJ - allow double moves (0: no, 1: yes)',default=0, metavar=0)
p.add_argument('-rj_bd_shift', type=float, help='RJ - 0: only sample shifts in speciation; 1: only sample shifts in extinction',default=0.5, metavar=0.5)
p.add_argument('-se_gibbs',	help='Use aprroximate S/E Gibbs sampler', action='store_true', default=False)

# PRIORS
p.add_argument('-pL',	  type=float, help='Prior - speciation rate (Gamma <shape, rate>) | (if shape=n,rate=0 -> rate estimated)', default=[1.1, 1.1], metavar=1.1, nargs=2)
p.add_argument('-pM',	  type=float, help='Prior - extinction rate (Gamma <shape, rate>) | (if shape=n,rate=0 -> rate estimated)', default=[1.1, 1.1], metavar=1.1, nargs=2)
p.add_argument('-pP',	  type=float, help='Prior - preservation rate (Gamma <shape, rate>) | (if shape=n,rate=0 -> rate estimated)', default=[1.5, 1.1], metavar=1.5, nargs=2)
p.add_argument('-pS',	  type=float, help='Prior - time frames (Dirichlet <shape>)', default=2.5, metavar=2.5)
p.add_argument('-pC',	  type=float, help='Prior - Covar parameters (Normal <standard deviation>) | (if pC=0 -> sd estimated)', default=1, metavar=1)
p.add_argument("-cauchy",  type=float, help='Prior - use hyper priors on sp/ex rates (if 0 -> estimated)', default=[-1, -1], metavar=-1, nargs=2)
p.add_argument("-min_dt",  type=float, help='Prior - minimum allowed distance between rate shifts', default=1., metavar=1)

# MODEL
p.add_argument("-mHPP",	help='Model - Homogeneous Poisson process of preservation', action='store_true', default=False)
#p.add_argument("-TPP_model",help='Model - Poisson process of preservation with shifts', action='store_true', default=False)
p.add_argument('-mL',	  type=int, help='Model - no. (starting) time frames (speciation)', default=1, metavar=1)
p.add_argument('-mM',	  type=int, help='Model - no. (starting) time frames (extinction)', default=1, metavar=1)
p.add_argument('-mC',	  help='Model - constrain time frames (l,m)', action='store_true', default=False)
p.add_argument('-mCov',	type=int, help='COVAR model: 1) speciation, 2) extinction, 3) speciation & extinction, 4) preservation, 5) speciation & extinction & preservation', default=0, metavar=0)
p.add_argument("-mG",	  help='Model - Gamma heterogeneity of preservation rate', action='store_true', default=False)
p.add_argument('-mPoiD',   help='Poisson-death diversification model', action='store_true', default=False)
p.add_argument('-mDeath',  help='Pure-death model', action='store_true', default=False)
p.add_argument("-mBDI",	type=int, help='BDI sub-model - 0) birth-death, 1) immigration-death', default=-1, metavar=-1)
p.add_argument("-ncat",	type=int, help='Model - Number of categories for Gamma heterogeneity', default=4, metavar=4)
p.add_argument('-fixShift',metavar='<input file>', type=str,help="Input tab-delimited file",default="")
p.add_argument('-qShift',  metavar='<input file>', type=str,help="Poisson process of preservation with shifts (Input tab-delimited file)",default="")
p.add_argument('-fixSE',   metavar='<input file>', type=str,help="Input mcmc.log file",default="")
p.add_argument('-ADE',	 type=int, help='ADE model: 0) no age dependence 1) estimated age dep', default=0, metavar=0)
p.add_argument('-discrete',help='Discrete-trait-dependent BD model (requires -trait_file)', action='store_true', default=False)
p.add_argument('-twotrait',help='Discrete-trait-dependent extinction + Covar', action='store_true', default=False)
p.add_argument('-bound',   type=float, help='Bounded BD model', default=[np.inf, 0], metavar=0, nargs=2)
p.add_argument('-edgeShift',type=float, help='Fixed times of shifts at the edges (when -mL/-mM > 3)', default=[np.inf, 0], metavar=0, nargs=2)
p.add_argument('-qFilter', type=int, help='if set to zero all shifts in preservation rates are kept, even if outside observed timerange', default=1, metavar=1)
p.add_argument('-FBDrange', type=int, help='use FBDrange likelihood (experimental)', default=0, metavar=0)


# TUNING
p.add_argument('-tT',	 type=float, help='Tuning - window size (ts, te)', default=1., metavar=1.)
p.add_argument('-nT',	 type=int,   help='Tuning - max number updated values (ts, te)', default=5, metavar=5)
p.add_argument('-tQ',	 type=float, help='Tuning - window sizes (q/alpha: 1.2 1.2)', default=[1.2,1.2], nargs=2)
p.add_argument('-tR',	 type=float, help='Tuning - window size (rates)', default=1.2, metavar=1.2)
p.add_argument('-tS',	 type=float, help='Tuning - window size (time of shift)', default=1., metavar=1.)
p.add_argument('-fR',	 type=float, help='Tuning - fraction of updated values (rates)', default=.5, metavar=.5)
p.add_argument('-fS',	 type=float, help='Tuning - fraction of updated values (shifts)', default=.7, metavar=.7)
p.add_argument('-fQ',	 type=float, help='Tuning - fraction of updated values (q rates, TPP)', default=.5, metavar=.5)
p.add_argument('-tC',	 type=float, help='Tuning - window sizes cov parameters (l,m,q)', default=[.2, .2, .15], nargs=3)
p.add_argument('-fU',	 type=float, help='Tuning - update freq. (q: .02, l/m: .18, cov: .08)', default=[.02, .18, .08], nargs=3)
p.add_argument('-multiR', type=int,   help='Tuning - Proposals for l/m: 0) sliding win 1) muliplier ', default=1, metavar=1)
p.add_argument('-tHP',	type=float, help='Tuning - window sizes hyperpriors on l and m', default=[1.2, 1.2], nargs=2)

args = p.parse_args()
t1=time.time()

if args.seed==-1:
	rseed=np.random.randint(0,9999)
else: rseed=args.seed
rand.seed(rseed)  # set as argument/ use get seed function to get it and save it to sum.txt file
random.seed(rseed)
np.random.seed(rseed)

if  args.FBDrange==0: FBDrange = 0
else: FBDrange = 1

if args.useCPPlib==1 and hasFoundPyRateC == 1:
	#print("Loaded module FastPyRateC")
	CPPlib="\nUsing module FastPyRateC"
else:
	hasFoundPyRateC= 0
	CPPlib=""

if args.cite:
	sys.exit(citation)
############################ MODEL SETTINGS ############################
# PRIORS
L_lam_r,L_lam_m = args.pL # shape and rate parameters of Gamma prior on sp rates
M_lam_r,M_lam_m = args.pM # shape and rate parameters of Gamma prior on ex rates
lam_s = args.pS							  # shape parameter dirichlet prior on time frames
pert_prior = [args.pP[0],args.pP[1]] # gamma prior on foss. rate; beta on mode PERT distribution
covar_prior_fixed=args.pC # std of normal prior on th covariance parameters

# MODEL
time_framesL=args.mL		  # no. (starting) time frames (lambda)
time_framesM=args.mM		  # no. (starting) time frames (mu)
constrain_time_frames=args.mC # True/False
pp_gamma_ncat=args.ncat			  # args.ncat
if args.mG:			 # number of gamma categories
	argsG = 1
	YangGammaQuant=(np.linspace(0,1,pp_gamma_ncat+1)-np.linspace(0,1,pp_gamma_ncat+1)[1]/2)[1:]
else: argsG = 0
model_cov=args.mCov		   # boolean 0: no covariance 1: covariance (speciation,extinction) 2: covariance (speciation,extinction,preservation)

if args.mHPP: argsHPP=1
else: argsHPP=0
############################ MCMC SETTINGS ############################
# GENERAL SETTINGS
TDI=args.A				  # 0: parameter estimation, 1: thermodynamic integration, 2: BD-MCMC
if constrain_time_frames == 1 or args.fixShift != "":
	if TDI in [2,4]:
		print("\nConstrained shift times (-mC,-fixShift) cannot be used with BD/RJ MCMC alorithms. Using standard MCMC instead.\n")
		TDI = 0
if args.ADE>=1 and TDI>1:
	print("\nADE models (-ADE 1) cannot be used with BD/RJ MCMC alorithms. Using standard MCMC instead.\n")
	TDI = 0
mcmc_gen=args.n			 # no. total mcmc generations
sample_freq=args.s
print_freq=args.p
burnin=args.b
num_processes = args.thread[0]	# BDlik
num_processes_ts = args.thread[1] # NHPPlik
if num_processes+num_processes_ts==0: use_seq_lik = 1
if use_seq_lik == 1: num_processes,num_processes_ts=0,0
min_allowed_t=args.min_dt

# RJ arguments
addrm_proposal_RJ = args.rj_pr	  # 0: random Gamma; 1: weighted mean
shape_gamma_RJ	= args.rj_Ga
rate_gamma_RJ	 = args.rj_Gb
shape_beta_RJ	 = args.rj_beta
if addrm_proposal_RJ == 0:
	add_shift_RJ	= add_shift_RJ_rand_gamma
	remove_shift_RJ = remove_shift_RJ_rand_gamma
elif addrm_proposal_RJ == 1:
	add_shift_RJ	= add_shift_RJ_weighted_mean
	remove_shift_RJ = remove_shift_RJ_weighted_mean
allow_double_move = args.rj_dm



# TUNING
d1=args.tT					 # win-size (ts, te)
frac1= args.nT				 # max number updated values (ts, te)
d2=args.tQ					 # win-sizes (q,alpha)
d3=args.tR					 # win-size (rates)
f_rate=args.fR				 # fraction of updated values (rates)
d4=args.tS					 # win-size (time of shift)
f_shift=args.fS				# update frequency (time of shift) || will turn into 0 when no rate shifts
f_qrate_update =args.fQ		# update frequency (preservation rates under TPP model)
freq_list=args.fU			  # generate update frequencies by parm category
d5=args.tC					 # win-size (cov)
d_hyperprior=np.array(args.tHP)		  # win-size hyper-priors onf l/m (or W_scale)
if model_cov==0: freq_list[2]=0
f_update_se=1-sum(freq_list)
if frac1==0: f_update_se=0
[f_update_q,f_update_lm,f_update_cov]=f_update_se+np.cumsum(array(freq_list))


if args.se_gibbs: use_gibbs_se_sampling = 1
else: use_gibbs_se_sampling = 0

fast_burnin =args.fast_burnin


multiR = args.multiR
if multiR==0:
	update_rates =  update_rates_sliding_win
else:
	update_rates = update_rates_multiplier
	d3 = max(args.tR,1.01) # avoid win size < 1


if args.ginput != "" or args.check_names != "" or args.reduceLog != "":
	try:
	 	self_path = get_self_path()
	 	pyrate_lib_path = "pyrate_lib"
	 	sys.path.append(os.path.join(self_path,pyrate_lib_path))
		import lib_DD_likelihood
		import lib_utilities
		import check_species_names

 	except:
		sys.exit("""\nWarning: library pyrate_lib not found.\nMake sure PyRate.py and pyrate_lib are in the same directory.
		You can download pyrate_lib here: <https://github.com/dsilvestro/PyRate> \n""")

	if args.ginput != "":
		lib_utilities.write_ts_te_table(args.ginput, tag=args.tag, clade=-1,burnin=int(burnin)+1)
	elif args.check_names != "":
		SpeciesList_file = args.check_names
		check_species_names.run_name_check(SpeciesList_file)
	elif args.reduceLog != "":
		lib_utilities.reduce_log_file(args.reduceLog,max(1,int(args.b)))
	quit()


if args.use_DA: use_DA = 1
else: use_DA = 0



# freq update CovPar
if model_cov==0: f_cov_par= [0  ,0  ,0 ]
if model_cov==1: f_cov_par= [1  ,0  ,0 ]
if model_cov==2: f_cov_par= [0  ,1  ,0 ]
if model_cov==3: f_cov_par= [.5 ,1  ,0 ]
if model_cov==4: f_cov_par= [0  ,0  ,1 ]
if model_cov==5: f_cov_par= [.33,.66,1 ]

if covar_prior_fixed==0: est_COVAR_prior = 1
else: est_COVAR_prior = 0

if args.fixShift != "" or TDI==3:	 # fix times of rate shift or DPP
	try:
		try: fixed_times_of_shift=sort(np.loadtxt(args.fixShift))[::-1]
		except: fixed_times_of_shift=np.array([np.loadtxt(args.fixShift)])
		f_shift=0
		time_framesL=len(fixed_times_of_shift)+1
		time_framesM=len(fixed_times_of_shift)+1
		min_allowed_t=0
		fix_Shift = 1
	except:
		if TDI==3:
			fixed_times_of_shift=np.arange(0,10000,args.dpp_grid)[::-1] # run fixed_times_of_shift[fixed_times_of_shift<max(FA)] below
			fixed_times_of_shift=fixed_times_of_shift[:-1]			  # after loading input file
			f_shift=0
			time_framesL=len(fixed_times_of_shift)+1
			time_framesM=len(fixed_times_of_shift)+1
			min_allowed_t=0
			fix_Shift = 1
		else:
			msg = "\nError in the input file %s.\n" % (args.fixShift)
			sys.exit(msg)
else:
	fixed_times_of_shift=[]
	fix_Shift = 0

if args.edgeShift[0] != np.inf or args.edgeShift[1] != 0:
	edgeShifts = []
	if args.edgeShift[0] != np.inf: # max boundary
		edgeShifts.append(args.edgeShift[0])
		fix_edgeShift = 2
		min_allowed_n_rates = 2
	if args.edgeShift[1] != 0: # min boundary
		edgeShifts.append(args.edgeShift[1])
		fix_edgeShift = 3
		min_allowed_n_rates = 2
	if len(edgeShifts)==2: # min and max boundaries
		fix_edgeShift = 1
		min_allowed_n_rates = 3
	time_framesL = max(min_allowed_n_rates,args.mL) # change number of starting rates based on edgeShifts
	time_framesM = max(min_allowed_n_rates,args.mM) # change number of starting rates based on edgeShifts
	edgeShifts = np.array(edgeShifts)*args.rescale+args.translate
else:
	fix_edgeShift = 0
	min_allowed_n_rates = 1

# BDMCMC & MCMC SETTINGS
runs=args.r			  # no. parallel MCMCs (MC3)
if runs>1 and TDI>0:
	print("\nWarning: MC3 algorithm is not available for TI and BDMCMC. Using a single chain instead.\n")
	runs,TDI=1,0
num_proc = runs		  # processors MC3
temp_pr=args.t		   # temperature MC3
IT=args.sw
freq_Alg_3_1=args.M	  # frequency of model update
birthRate=args.B		 # birthRate (=Poisson prior)
len_cont_time=args.T	 # length continuous time of model update
start_Alg_3_1=args.S	 # start sampling model after


if runs==1 or use_seq_lik == 1:
	IT=mcmc_gen

if TDI==1:				# Xie et al. 2011; Baele et al. 2012
	K=args.k-1.		# K+1 categories
	k=array(range(int(K+1)))
	beta=k/K
	alpha=args.a			# categories are beta distributed
	temperatures=list(beta**(1./alpha))
	temperatures[0]+= small_number # avoid exactly 0 temp
	temperatures.reverse()
	if multiR==0: # tune win sizes only if sliding win proposals
		list_d3=sort(exp(temperatures))**2.5*d3+(exp(1-array(temperatures))-1)*d3
	else:
		list_d3=np.repeat(d3,len(temperatures))
	list_d4=sort(exp(temperatures))**1.5*d4+exp(1-array(temperatures))-1
else:
	temperatures=[1]
	list_d3=[d3]
	list_d4=[d4]

# ARGS DPP
freq_dpp	   = args.dpp_f
hp_gamma_shape = args.dpp_hp
target_k	   = args.dpp_eK

############### PLOT RTT
path_dir_log_files=""
if args.plot != "":
	path_dir_log_files=args.plot
	plot_type=1
elif args.plot2 != "":
	path_dir_log_files=args.plot2
	plot_type=2
elif args.plot3 != "":
	path_dir_log_files=args.plot3
	plot_type=3
elif args.plotRJ != "":
	path_dir_log_files=args.plotRJ
	plot_type=4
elif args.plotQ != "":
	path_dir_log_files=args.plotQ
	plot_type=5
print args.plotQ

list_files_BF=sort(args.BF)
file_stem=args.tag
root_plot=args.root_plot
grid_plot = args.grid_plot
if path_dir_log_files != "":
	self_path = get_self_path()
	if plot_type>=3:
		lib_DD_likelihood = imp.load_source("lib_DD_likelihood", "%s/pyrate_lib/lib_DD_likelihood.py" % (self_path))
		lib_utilities = imp.load_source("lib_utilities", "%s/pyrate_lib/lib_utilities.py" % (self_path))
		rtt_plot_bds = imp.load_source("rtt_plot_bds", "%s/pyrate_lib/rtt_plot_bds.py" % (self_path))
		if plot_type==3:
			if grid_plot==0: grid_plot=1
			rtt_plot_bds.RTTplot_high_res(path_dir_log_files,grid_plot,int(burnin),root_plot)
		elif plot_type==4:
			rtt_plot_bds = rtt_plot_bds.plot_marginal_rates(path_dir_log_files,name_tag=file_stem,bin_size=grid_plot,burnin=burnin,min_age=args.min_age_plot,max_age=root_plot,logT=args.logT)
		elif plot_type== 5:
			rtt_plot_bds = rtt_plot_bds.RTTplot_Q(path_dir_log_files,args.qShift,burnin=burnin,max_age=root_plot)
		#except: sys.exit("""\nWarning: library pyrate_lib not found.\nMake sure PyRate.py and pyrate_lib are in the same directory.
		#You can download pyrate_lib here: <https://github.com/dsilvestro/PyRate> \n""")

	else:
		#path_dir_log_files=sort(path_dir_log_files)
		# plot each file separately
		print root_plot
		if file_stem == "":
			path_dir_log_files = os.path.abspath(path_dir_log_files)
			direct="%s/*marginal_rates.log" % path_dir_log_files
			files=glob.glob(direct)
			files=sort(files)
			if len(files)==0:
				if 2>1: #try:
					name_file = os.path.splitext(os.path.basename(str(path_dir_log_files)))[0]
					path_dir_log_files = os.path.dirname(str(path_dir_log_files))
					name_file = name_file.split("marginal_rates")[0]
					one_file= 1
					plot_RTT(path_dir_log_files, burnin, name_file,one_file,root_plot,plot_type)
				#except: sys.exit("\nFile or directory not recognized.\n")
			else:
				for f in files:
					name_file = os.path.splitext(os.path.basename(f))[0]
					name_file = name_file.split("marginal_rates")[0]
					one_file = 0
					plot_RTT(path_dir_log_files, burnin, name_file,one_file,root_plot,plot_type)
		else:
			one_file = 0
			plot_RTT(path_dir_log_files, burnin, file_stem,one_file,root_plot,plot_type)
	quit()
elif args.mProb != "": calc_model_probabilities(args.mProb,burnin)
elif len(list_files_BF):
	print list_files_BF[0]
	if len(list_files_BF)==1: calc_BFlist(list_files_BF[0])
	else: calc_BF(list_files_BF[0],list_files_BF[1])
	 	#
	#	sys.exit("\n2 '*marginal_likelihood.txt' files required.\n")
	quit()
elif args.combLog != "": # COMBINE LOG FILES
	comb_log_files(args.combLog,burnin,args.tag,resample=args.resample,col_tag=args.col_tag)
	sys.exit("\n")
elif args.combLogRJ != "": # COMBINE LOG FILES
	comb_log_files_smart(args.combLogRJ,burnin,args.tag,resample=args.resample,col_tag=args.col_tag)
	sys.exit("\n")
elif len(args.input_data)==0 and args.d == "": sys.exit("\nInput file required. Use '-h' for command list.\n")

use_se_tbl = 0
if args.d != "":
	use_se_tbl = 1
	se_tbl_file  = args.d

if len(args.SE_stats)>0:
	if use_se_tbl == 0: sys.exit("\nProvide an SE table using command -d\n")
	#if len(args.SE_stats)<1: sys.exit("\nExtinction rate at the present\n")
	#else:
	try: EXT_RATE  = float(args.SE_stats[0])
	except: EXT_RATE = 0
	if EXT_RATE==0: print "\nExtinction rate set to 0: using estimator instead.\n"
	if len(args.SE_stats)>1: step_size = args.SE_stats[1]
	else: step_size = 1
	if len(args.SE_stats)>2: no_sim_ex_time = args.SE_stats[2]
	else: no_sim_ex_time = 100
	plot_tste_stats(se_tbl_file, EXT_RATE, step_size,no_sim_ex_time,burnin,args.rescale)
	quit()

if args.ltt>0:
	grid_plot = args.grid_plot
	if grid_plot==0: grid_plot=0.1
	plot_ltt(se_tbl_file,plot_type=args.ltt,rescale=args.rescale, step_size=grid_plot)

twotraitBD = 0
if args.twotrait == 1:
	twotraitBD = 1

############################ LOAD INPUT DATA ############################
match_taxa_trait = 0
if use_se_tbl==0:
	import imp
	input_file_raw = os.path.basename(args.input_data[0])
	input_file = os.path.splitext(input_file_raw)[0]  # file name without extension

	if args.wd=="":
		output_wd = os.path.dirname(args.input_data[0])
		if output_wd=="": output_wd= get_self_path()
	else: output_wd=args.wd

	print "\n",input_file, args.input_data, "\n"
	try: input_data_module = imp.load_source(input_file, args.input_data[0])
	except(IOError): sys.exit("\nInput file required. Use '-h' for command list.\n")

	j=max(args.j-1,0)
	try: fossil_complete=input_data_module.get_data(j)
	except(IndexError):
		fossil_complete=input_data_module.get_data(0)
		print("Warning: data set number %s not found. Using the first data set instead." % (args.j))
		j=0

	if args.filter_taxa != "":
		list_included_taxa = [line.rstrip() for line in open(args.filter_taxa)]
		print "Included taxa:"

	fossil=list()
	have_record=list()
	singletons_excluded = list()
	taxa_included = list()
	for i in range(len(fossil_complete)):
		if len(fossil_complete[i])==1 and fossil_complete[i][0]==0: pass # exclude taxa with no fossils

		elif max(fossil_complete[i]) > max(args.filter) or min(fossil_complete[i]) < min(args.filter):
			print "excluded taxon with age range:",round(max(fossil_complete[i]),3), round(min(fossil_complete[i]),3)

		elif args.singleton == -1: # exclude extant taxa (if twotraitBD == 1: extant (re)moved later)
			if min(fossil_complete[i])==0 and twotraitBD == 0: singletons_excluded.append(i)
			else:
				have_record.append(i)
				fossil.append(fossil_complete[i]*args.rescale+args.translate)
				taxa_included.append(i)

		elif args.translate < 0: # exclude recent taxa after 'translating' records towards zero
			if max(fossil_complete[i]*args.rescale+args.translate)<=0: singletons_excluded.append(i)
			else:
				have_record.append(i)
				fossil_occ_temp = fossil_complete[i]*args.rescale+args.translate
				fossil_occ_temp[fossil_occ_temp<0] = 0.0
				fossil.append(np.unique(fossil_occ_temp[fossil_occ_temp>=0]))
				taxa_included.append(i)


		elif args.singleton > 0: # min number of occurrences
			if len(fossil_complete[i]) <= args.singleton and np.random.random() >= args.frac_sampled_singleton:
				singletons_excluded.append(i)
			else:
				have_record.append(i) # some (extant) species may have trait value but no fossil record
				fossil.append(fossil_complete[i]*args.rescale+args.translate)
				taxa_included.append(i)
		elif args.filter_taxa != "": # keep only taxa within list
			taxa_names_temp=input_data_module.get_taxa_names()
			if taxa_names_temp[i] in list_included_taxa:
				have_record.append(i) # some (extant) species may have trait value but no fossil record
				fossil.append(fossil_complete[i]*args.rescale+args.translate)
				taxa_included.append(i)
				print taxa_names_temp[i]
			else: singletons_excluded.append(i)
		else:
			have_record.append(i) # some (extant) species may have trait value but no fossil record
			fossil.append(fossil_complete[i]*args.rescale+args.translate)
			taxa_included.append(i)
	if len(singletons_excluded)>0 and args.data_info == 0: print "The analysis includes %s species (%s were excluded)" % (len(fossil),len(singletons_excluded))
	else: print "\nThe analysis includes %s species (%s were excluded)" % (len(fossil),len(fossil_complete)-len(fossil))
	out_name=input_data_module.get_out_name(j) +args.out

	try:
		taxa_names=input_data_module.get_taxa_names()
		match_taxa_trait = 1
	except(AttributeError):
		taxa_names=list()
		for i in range(len(fossil)): taxa_names.append("taxon_%s" % (i))

	#print singletons_excluded
	taxa_included = np.array(taxa_included)
	taxa_names = np.array(taxa_names)
	taxa_names = taxa_names[taxa_included]

	FA,LO,N=np.zeros(len(fossil)),np.zeros(len(fossil)),np.zeros(len(fossil))
	array_all_fossils = []
	for i in range(len(fossil)):
		FA[i]=max(fossil[i])
		LO[i]=min(fossil[i])
		N[i]=len(fossil[i])
		array_all_fossils = array_all_fossils + list(fossil[i])
	array_all_fossils = np.array(array_all_fossils)

else:
	print se_tbl_file
	t_file=np.loadtxt(se_tbl_file, skiprows=1)
	print np.shape(t_file)
	j=max(args.j-1,0)
	FA=t_file[:,2+2*j]*args.rescale+args.translate
	LO=t_file[:,3+2*j]*args.rescale+args.translate
	focus_clade=args.clade
	clade_ID=t_file[:,0].astype(int)
	if focus_clade>=0: FA,LO=FA[clade_ID==focus_clade],LO[clade_ID==focus_clade]
	print j, len(FA), "species"
	fix_SE= 1
	fixed_ts, fixed_te=FA, LO
	output_wd = os.path.dirname(se_tbl_file)
	if output_wd=="": output_wd= get_self_path()
	out_name="%s_%s_%s"  % (os.path.splitext(os.path.basename(se_tbl_file))[0],j,args.out)
	if focus_clade>=0: out_name+= "_c%s" % (focus_clade)

if args.restore_mcmc != "":
	restore_init_values = get_init_values(args.restore_mcmc,taxa_names)
	restore_chain = 1
else: restore_chain = 0

###### SET UP BD MODEL WITH STARTING NUMBER OF LINEAGES > 1
no_starting_lineages = args.initDiv
max_age_fixed_ts = max(FA)

if no_starting_lineages>0:
	if use_se_tbl==0:
		sys.exit("Starting lineages > 1 only allowed with -d option")
	if model_cov>=1:
		sys.exit("Starting lineages > 1 only allowed with -d option")
	#print sort(fixed_ts)
	fixed_ts_ordered = np.sort(fixed_ts+0.)[::-1]
	fixed_ts_ordered_not_speciation = fixed_ts_ordered[0:no_starting_lineages]
	ind = np.array([i for i in range(len(fixed_ts)) if fixed_ts[i] in fixed_ts_ordered_not_speciation])
	fixed_ts[ind] = max_age_fixed_ts
	#print sort(fixed_ts)

################

if argsG == 1: out_name += "_G"
if args.se_gibbs: out_name += "_seGibbs"

############################ SET BIRTH-DEATH MODEL ############################

# Number of extant taxa (user specified)
if args.N>-1: tot_extant=args.N
else: tot_extant = -1


if len(fixed_times_of_shift)>0:
	fixed_times_of_shift=fixed_times_of_shift[fixed_times_of_shift<max(FA)]
	# fixed number of dpp bins
	if args.dpp_nB>0:
		t_bin_set = np.linspace(0,max(FA),args.dpp_nB+1)[::-1]
		fixed_times_of_shift = t_bin_set[1:len(t_bin_set)-1]
	time_framesL=len(fixed_times_of_shift)+1
	time_framesM=len(fixed_times_of_shift)+1
	# estimate DPP hyperprior
	hp_gamma_rate  = get_rate_HP(time_framesL,target_k,hp_gamma_shape)



if args.fixSE != "" or use_se_tbl==1:		  # fix TS, TE
	if use_se_tbl==1: pass
	else:
		fix_SE=1
		fixed_ts, fixed_te= calc_ts_te(args.fixSE, burnin=args.b)
else: fix_SE=0

if args.discrete == 1: useDiscreteTraitModel = 1
else: useDiscreteTraitModel = 0

useBounded_BD = 0
if args.bound[0] != np.inf or args.bound[1] != 0:
	useBounded_BD = 1
boundMax = max(args.bound) # if not specified it is set to Inf
boundMin = min(args.bound) # if not specified it is set to 0

# Get trait values (COVAR and DISCRETE models)
if model_cov>=1 or useDiscreteTraitModel == 1 or useBounded_BD == 1:
	if 2>1: #try:
		if args.trait_file != "": # Use trait file
			traitfile=file(args.trait_file, 'U')

			L=traitfile.readlines()
			head= L[0].split()

			if useBounded_BD == 1: # columns: taxon_name, SP, EX (SP==1 if speciate in window)
				trait_val=[l.split()[1:3] for l in L][1:]
			else:
				if len(head)==2: col=1
				elif len(head)==3: col=2 ####  CHECK HERE: WITH BOUNDED-BD 3 columns but two traits!
				else: sys.exit("\nNo trait data found!")
				trait_val=[l.split()[col] for l in L][1:]

			if useBounded_BD == 1:
				trait_values = np.array(trait_val)
				trait_values = trait_values.astype(float)
			else:
				trait_values = np.zeros(len(trait_val))
				trait_count = 0
				for i in range(len(trait_val)):
					try:
						trait_values[i] = float(trait_val[i])
						trait_count+=1
					except:
						trait_values[i] = np.nan
				if trait_count==0: sys.exit("\nNo trait data found.")

			if match_taxa_trait == 1:
				trait_taxa=np.array([l.split()[0] for l in L][1:])
				#print taxa_names
				#print sort(trait_taxa)
				matched_trait_values = []
				for i in range(len(taxa_names)):
					taxa_name=taxa_names[i]
					matched_val = trait_values[trait_taxa==taxa_name]
					if len(matched_val)>0:
						matched_trait_values.append(matched_val[0])
						print "matched taxon: %s\t%s\t%s" % (taxa_name, matched_val[0], max(fossil[i])-min(fossil[i]))
					else:
						if useBounded_BD == 1: # taxa not specified originate/go_extinct in window
							matched_trait_values.append([1,1])
						else:
							matched_trait_values.append(np.nan)
						#print taxa_name, "did not have data"
				trait_values= np.array(matched_trait_values)
				#print trait_values

		else:			 # Trait data from .py file
			trait_values=input_data_module.get_continuous(max(args.trait-1,0))
		#
		if twotraitBD == 1:
			trait_values=input_data_module.get_continuous(0)
			discrete_trait=input_data_module.get_continuous(1)
			discrete_trait=discrete_trait[taxa_included]

			#print discrete_trait
			#print len(np.isfinite(discrete_trait).nonzero()[0])
			if args.singleton == -1:
				ind_extant_sp =(LO==0).nonzero()[0]
				print "Treating %s extant taxa as missing discrete data" % (len(ind_extant_sp))
				#temp_trait = discrete_trait+0 #np.zeros(len(discrete_trait))*discrete_trait
				discrete_trait[ind_extant_sp]=np.nan
				#discrete_trait = temp_trait
				#print len(np.isfinite(discrete_trait).nonzero()[0])
				#print discrete_trait

		if model_cov>=1:
			if args.logT==0: pass
			elif args.logT==1: trait_values = log(trait_values)
			else: trait_values = np.log10(trait_values)
	#except: sys.exit("\nTrait data not found! Check input file.\n")

	if model_cov>=1:
		# Mid point age of each lineage
		if use_se_tbl==1: MidPoints = (fixed_ts+fixed_te)/2.
		else:
			MidPoints=np.zeros(len(fossil_complete))
			for i in range(len(fossil_complete)):
				MidPoints[i]=np.mean([max(fossil_complete[i]),min(fossil_complete[i])])

		MidPoints = MidPoints[taxa_included]
		# fit linear regression (for species with trait value - even if without fossil data)
		print len(trait_values), len(np.isfinite(trait_values)), len(taxa_names), len(MidPoints), len(have_record)
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(MidPoints[np.isfinite(trait_values)],trait_values[np.isfinite(trait_values)])

		#
		ind_nan_trait= (np.isfinite(trait_values)== 0).nonzero()
		meanGAUScomplete=np.zeros(len(MidPoints))
		meanGAUScomplete[ind_nan_trait] = slope*MidPoints[ind_nan_trait] + intercept

		if use_se_tbl==1 or args.trait_file != "": meanGAUS= meanGAUScomplete
		else:
			trait_values= trait_values[np.array(have_record)]
			meanGAUS= meanGAUScomplete[np.array(have_record)]

		sdGAUS = std_err
		regression_trait= "\n\nEstimated linear trend trait-value: \nslope=%s; sd. error= %s (intercept= %s; R2= %s; P-value= %s\nTrait data for %s of %s taxa)" \
		% (round(slope,2), round(std_err,2), round(intercept,2), round(r_value,2), round(p_value,2), len(trait_values)-len(ind_nan_trait[0]), len(trait_values))
		print(regression_trait)

		#print trait_values
		parGAUS=scipy.stats.norm.fit(trait_values[np.isfinite(trait_values)]) # fit normal distribution
		#global con_trait
		con_trait=seed_missing(trait_values,meanGAUS,sdGAUS) # fill the gaps (missing data)
		#print con_trait
		if est_COVAR_prior == 1: out_name += "_COVhp"
		else: out_name += "_COV"

	if useDiscreteTraitModel == 1:
		if twotraitBD == 0:
			discrete_trait = trait_values
		ind_nan_trait= (np.isfinite(discrete_trait)== 0).nonzero()
		regression_trait= "\n\nDiscrete trait data for %s of %s taxa" \
		% (len(trait_values)-len(ind_nan_trait[0]), len(trait_values))
		print(regression_trait)
	else: print "\n"


if useDiscreteTraitModel == 1:
	ind_trait_species = discrete_trait
	print ind_trait_species
	ind_trait_species[np.isnan(ind_trait_species)]=np.nanmax(ind_trait_species)+1
	print ind_trait_species
	ind_trait_species = ind_trait_species.astype(int)
	len_trait_values = len(np.unique(discrete_trait))
	lengths_B_events=[]
	lengths_D_events=[]
	for i in range(len_trait_values):
		lengths_B_events.append(len(discrete_trait[discrete_trait==i]))
		lo_temp = LO[discrete_trait==i]
		lengths_D_events.append(len(lo_temp[lo_temp>0]))
	lengths_B_events = np.array([sum(lengths_B_events)]) # ASSUME CONST BIRTH RATE
	#lengths_B_events = np.array(lengths_D_events)
	lengths_D_events = np.array(lengths_D_events)
	#ind_trait_species = ind_trait_species-ind_trait_species
	print lengths_B_events, lengths_D_events
	obs_S = [sum(FA[ind_trait_species==i]-LO[ind_trait_species==i]) for i in range(len(lengths_B_events))]
	print obs_S
	TDI = 0

use_poiD=args.mPoiD
if use_poiD == 1:
	BPD_partial_lik = PoiD_partial_lik
	PoiD_const = - (sum(log(np.arange(1,len(FA)+1))))
elif useBounded_BD == 1:
	BPD_partial_lik = BD_partial_lik_bounded
	PoiD_const = 0
	SP_in_window = (trait_values[:,0]==1).nonzero()[0]
	EX_in_window = (trait_values[:,1]==1).nonzero()[0]
	SP_not_in_window = (trait_values[:,0]==0).nonzero()[0]
	EX_not_in_window = (trait_values[:,1]==0).nonzero()[0]
	#print SP_in_window, EX_in_window
	###### NEXT: change function update_ts_te() so that only SP/EX_in_win are updated
	# make sure the starting points are set to win boundaries for the other species and
	# within boundaries for SP/EX_in_win
	argsHPP = 1 # only HPP can be used with bounded BD
else:
	BPD_partial_lik = BD_partial_lik
	PoiD_const = 0
	SP_in_window = np.arange(len(FA)) # all ts/te can be updated
	EX_in_window = np.arange(len(LO))
	SP_not_in_window = []
	EX_not_in_window = []

if args.mDeath == 1: use_Death_model = 1
else: use_Death_model= 0


# USE BDI subMODELS
model_BDI=args.mBDI
if model_BDI >=0:
	try: ts,te = fixed_ts, fixed_te
	except: sys.exit("\nYou must use options -fixSE or -d to run BDI submodels.")
	z=np.zeros(len(te))+2
	z[te==0] = 3
	te_orig = te+0.
	te= te[te>0]  # ignore extant
	z = z[z==2]   # ignore extant
	all_events_temp= np.array([np.concatenate((ts,te),axis=0),np.concatenate((np.zeros(len(ts))+1,z),axis=0)])
	idx = np.argsort(all_events_temp[0])[::-1] # get indexes of sorted events
	all_events_array=all_events_temp[:,idx] # sort by time of event
	#print all_events_array
	all_events = all_events_array[0,:]
	dT_events= -(np.diff(np.append(all_events,0)))

	#div_trajectory =get_DT(np.append(all_events,0),ts,te_orig)
	#div_trajectory =div_trajectory[1:]
	div_traj = np.zeros(len(ts)+len(te))
	current_div,j = 0,0
	for i in all_events_array[1]:
		if i == 1: current_div+=1
		if i == 2: current_div-=1
		div_traj[j] = current_div
		j+=1

	#j=0
	#for i in all_events_array[0]:
	#	print i, "\t", div_trajectory[j],  "\t", div_traj[j], "\t",dT_events[j]
	#	j+=1
	div_trajectory=div_traj
	#print div_traj
	BPD_partial_lik = BDI_partial_lik
	if model_BDI==0: out_name += "BD"
	if model_BDI==1: out_name += "ID"
	if TDI<2: out_name = "%s%s%s" % (out_name,time_framesL,time_framesM)


# SET UO AGE DEP. EXTINCTION MODEL
use_ADE_model = 0
if args.ADE == 1:
	use_ADE_model = 1
	BPD_partial_lik = BD_age_partial_lik
	out_name += "_ADE"
	argsHPP = 1

if args.ADE == 2:
	use_ADE_model = 2
	BPD_partial_lik = BD_age_partial_lik
	out_name += "_CorrBD"
	argsHPP = 1
	#list_all_occs = []
	#for i in range(len(fossil)):
	#	f =fossil[i]
	#	list_all_occs = list_all_occs + list(f[f>0])
	#
	#list_all_occs = np.sort(np.array(list_all_occs))
	#print (np.diff(list_all_occs))
	#
	#quit()
	  #
	#n_sampled_species_bins = np.linspace(0,max(FA),100)
	#dT_sampled_sp_bins = n_sampled_species_bins[1]
	#n_sampled_species = np.zeros(len(n_sampled_species_bins)-1)
	#for i in range(len(fossil)):
	#	occs_temp = fossil[i]
	#	hist = np.histogram(occs_temp[occs_temp>0],bins=sort( n_sampled_species_bins ))
	#	h = hist[0][::-1]
	#	h[h>1] = 1
	#	print hist
	#	n_sampled_species += h
	#print log(n_sampled_species), sum(n_sampled_species)
	#sampling_prob = 1 - exp(-dT_sampled_sp_bins*0.5)
	#est_true_sp = log(1 + n_sampled_species + n_sampled_species*sampling_prob)
	#print est_true_sp[::-1]
	#print np.diff(n_sampled_species)[::-1], mean(np.diff(n_sampled_species)[::-1])/dT_sampled_sp_bins
	#print np.mean(np.diff(est_true_sp)[::-1])/dT_sampled_sp_bins
	#print np.mean(np.diff(log(n_sampled_species+1))[::-1])
	#quit()

est_hyperP = 0
use_cauchy = 0
fix_hyperP = 0
if sum(args.cauchy) >= 0:
	hypP_par = np.ones(2)
	use_cauchy = 1
	est_hyperP = 1
	if sum(args.cauchy) > 0:
		fix_hyperP = 1
		hypP_par = np.array(args.cauchy) # scale of Cauchy distribution
else:
	hypP_par = np.array([L_lam_m,M_lam_m]) # rate of Gamma distribution
	if min([L_lam_m,M_lam_m])==0:
		est_hyperP = 1
		hypP_par = np.ones(2)

if use_ADE_model >= 1:
	hypP_par[1]=0.1
	tot_extant = -1
	d_hyperprior[0]=1 # first hyper-prior on sp.rates is not used under ADE, thus not updated (multiplier update =1)

qFilter=args.qFilter # if set to zero all times of shifts (and preservation rates) are kept, even if they don't have occurrences
if args.qShift != "":
	if 2>1: #try:
		try: times_q_shift=np.sort(np.loadtxt(args.qShift))[::-1]*args.rescale + args.translate
		except: times_q_shift=np.array([np.loadtxt(args.qShift)])*args.rescale + args.translate
		# filter qShift times based on observed time frame
		if qFilter == 1:
			times_q_shift=times_q_shift[times_q_shift<max(FA)]
			times_q_shift=list(times_q_shift[times_q_shift>min(LO)])
		else: # q outside observed range (sampled from the prior)
			times_q_shift = list(times_q_shift)
		time_framesQ=len(times_q_shift)+1
		occs_sp_bin =list()
		temp_times_q_shift = np.array(list(times_q_shift)+[max(FA)+1]+[0])
		for i in range(len(fossil)):
			occs_temp = fossil[i]
			h = np.histogram(occs_temp[occs_temp>0],bins=sort( temp_times_q_shift ))[0][::-1]
			occs_sp_bin.append(h)
		argsHPP = 1
		TPP_model = 1
		print times_q_shift, max(FA), min(LO)
	#except:
	#	msg = "\nError in the input file %s.\n" % (args.qShift)
	#	sys.exit(msg)
else: TPP_model = 0


if fix_Shift == 1 and use_ADE_model == 0: est_hyperP = 1
# define hyper-prior function for BD rates
if tot_extant==-1 or TDI ==3 or use_poiD == 1:
	if use_ADE_model == 0 and fix_Shift == 1 and TDI < 3 or use_cauchy == 1:
		if est_hyperP == 0 or fix_hyperP == 1:
			prior_setting= "Using Cauchy priors on the birth-death rates (C_l[0,%s],C_l[0,%s]).\n" % (hypP_par[0],hypP_par[1])
		else:
				prior_setting= "Using Cauchy priors on the birth-death rates (C_l[0,est],C_l[0,est]).\n"
 		get_hyper_priorBD = HPBD1 # cauchy with hyper-priors
	else:
		if est_hyperP == 0:
			prior_setting= "Using Gamma priors on the birth-death rates (G_l[%s,%s], G_m[%s,%s]).\n" % (L_lam_r,hypP_par[0],M_lam_r,hypP_par[1])
		else:
			prior_setting= "Using Gamma priors on the birth-death rates (G_l[%s,est], G_m[%s,est]).\n" % (L_lam_r,M_lam_r)
		get_hyper_priorBD = HPBD2 # gamma
else:
	prior_setting= "Priors on the birth-death rates based on extant diversity (N = %s).\n" % (tot_extant)
	get_hyper_priorBD = HPBD3 # based on no. extant
print prior_setting

if use_poiD == 1:
	if model_cov>=1:
		print "PoiD not available with trait correlation. Using BD instead."
		BPD_partial_lik = BD_partial_lik
		PoiD_const = 0
	if fix_SE== 0:
		print "PoiD not available with SE estimation. Using BD instead."
		BPD_partial_lik = BD_partial_lik
		PoiD_const = 0
	if hasFoundPyRateC:
		print "PoiD not available using FastPyRateC library. Using Python version instead."
		hasFoundPyRateC = 0


##### SETFREQ OF PROPOSING B/D shifts (RJMCMC)
sample_shift_mu = args.rj_bd_shift # 0: updates only lambda; 1: only mu; default: 0.5



#### ANALYZE PHYLOGENY
analyze_tree = 0
if args.tree != "":
	try:
		import dendropy
	except:
		sys.exit("Library 'dendropy' not found!\n")
	try:
		tree_list = dendropy.TreeList.get_from_path(args.tree, schema="nexus", preserve_underscores= 1)
		tree=tree_list[0]
		tree.resolve_polytomies(update_bipartitions= 1)
		#tree_node_ages = sort(tree.calc_node_ages(ultrametricity_precision=0.001,is_return_internal_node_ages_only= 1))[::-1]
		tree.calc_node_ages(ultrametricity_precision=0.001) #
		nd= tree.ageorder_node_iter(include_leaves=False, filter_fn=None, descending= 1)
		ages=list()
		for n, node in enumerate(nd): ages.append(node.age)
		tree_node_ages = sort(np.array(ages))[::-1]
	except:
		sys.exit("Tree format not recognized (NEXUS file required). \n")
	tree_sampling_frac = args.sampling
	analyze_tree = 1
	if args.bdc: analyze_tree = 2
	if args.eqr: analyze_tree = 3

	if fix_Shift == 1:
		print "Using Skyline indepdent model"
		phylo_bds_likelihood = imp.load_source("phylo_bds_likelihood", "%s/pyrate_lib/phylo_bds_likelihood.py" % (self_path))
		analyze_tree = 4
		treeBDlikelihoodSkyLine = phylo_bds_likelihood.TreePar_LikShifts
		# args = (x,t,l,mu,sampling,posdiv=0,survival=1,groups=0)
		tree_node_ages = np.sort(tree_node_ages)
		phylo_times_of_shift = np.sort(np.array(list(fixed_times_of_shift) + [0]))
		tree_sampling_frac = np.array([tree_sampling_frac] + list(np.ones(len(fixed_times_of_shift))))
		print phylo_times_of_shift
		print tree_node_ages
		if args.bdc: args_bdc = 1
		else: args_bdc = 0
		# print tree_sampling_frac
		#quit()


	TDI = 0




# GET DATA SUMMARY INFO
if args.data_info == 1:
	print "\nDATA SUMMARY\n"
	if len(singletons_excluded)>0: print "%s taxa excluded" % (len(singletons_excluded))
	print "%s taxa included in the analysis" % (len(fossil))
	one_occ_sp,all_occ,extant_sp,n_occs_list  = 0,0,0,list()
	for i in fossil:
		if len(i[i>0])==1: one_occ_sp+=1
		all_occ += len(i[i>0])
		n_occs_list.append(len(i[i>0]))
		if min(i)==0: extant_sp+=1
	print "%s taxa have a single occurrence, %s taxa are extant" % (one_occ_sp,extant_sp)
	j=0
	m_ages,M_ages=[],[]
	while True:
		try: fossil_complete=input_data_module.get_data(j)
		except(IndexError): break
		min_age, max_age = np.inf, 0
		sp_indtemp = 0
		for i in fossil_complete:
			if sp_indtemp in taxa_included:
				a,b = min(i[i>0]), max(i)
				if a < min_age: min_age=a
				if b > max_age: max_age=b
			sp_indtemp+=1
		m_ages.append(min_age)
		M_ages.append(max_age)
		j+=1
	print "%s fossil occurrences (%s replicates), ranging from %s (+/- %s) to %s (+/- %s) Ma\n" % \
	(all_occ, j, round(mean(M_ages),3), round(std(M_ages),3),round(mean(m_ages),3), round(std(m_ages),3))
	# print species FA,LO
	print "Taxon\tFA\tLA"
	for i in range(len(fossil)):
		foss_temp = fossil[i]
		if min(foss_temp)==0: status_temp="extant"
		else: status_temp=""
		print "%s\t%s\t%s\t%s" % (taxa_names[i], round(max(foss_temp),3),round(min(foss_temp[foss_temp>0]),3),status_temp)

	# print histogram
	n_occs_list = np.array(n_occs_list)
	hist = np.histogram(n_occs_list,bins = np.arange(np.max(n_occs_list)+1)+1)[0]
	hist2 = hist.astype(float)/max(hist) * 50
	#print "occs.\ttaxa\thistogram"
	#for i in range(len(hist)): print "%s\t%s\t%s" % (i+1,int(hist[i]),"*"*int(hist2[i]))
	sys.exit("\n")

# RUN PP-MODEL TEST
if args.PPmodeltest== 1:
 	self_path = get_self_path()
 	pyrate_lib_path = "pyrate_lib"
 	sys.path.append(os.path.join(self_path,pyrate_lib_path))
	import PPmodeltest
	if TPP_model== 0: times_q_shift = 0
	PPmodeltest.run_model_testing(fossil,q_shift=times_q_shift,min_n_fossils=2,verbose=1)
	quit()


# CREATE C++ OBJECTS
if hasFoundPyRateC:
	if use_se_tbl==1:
		pass
	else: PyRateC_setFossils(fossil) # saving all fossil data as C vector

	if args.qShift != "":  # q_shift times
		tmpEpochs = np.sort(np.array(list(times_q_shift)+[max(FA)+1]+[0]))[::-1]
		PyRateC_initEpochs(tmpEpochs)

############################ MCMC OUTPUT ############################
try: os.mkdir(output_wd)
except(OSError): pass
path_dir = "%s/pyrate_mcmc_logs" % (output_wd)
folder_name="pyrate_mcmc_logs"
try: os.mkdir(path_dir)
except(OSError): pass

suff_out=out_name
if args.eqr: suff_out+= "_EQR"
elif args.bdc: suff_out+= "_BDC"
else:
	if TDI<=1: suff_out+= "BD%s-%s" % (args.mL,args.mM)
	if TDI==1: suff_out+= "_TI"
	if TDI==3: suff_out+= "dpp"
	if TDI==4: suff_out+= "rj"

# OUTPUT 0 SUMMARY AND SETTINGS
o0 = "\n%s build %s\n" % (version, build)
o1 = "\ninput: %s output: %s/%s" % (args.input_data, path_dir, out_name)
o2 = "\n\nPyRate was called as follows:\n%s\n" % (args)
if model_cov>=1 or useDiscreteTraitModel == 1: o2 += regression_trait
if TDI==3: o2 += "\n\nHyper-prior on concentration parameter (Gamma shape, rate): %s, %s\n" % (hp_gamma_shape, hp_gamma_rate)
if len(fixed_times_of_shift)>0:
	o2 += "\nUsing birth-death model with fixed times of rate shift: "
	for i in fixed_times_of_shift: o2 += "%s " % (i)
o2+= "\n"+prior_setting

if use_se_tbl != 1:
	if argsHPP == 1:
		if TPP_model == 0:
			o2+="Using Homogeneous Poisson Process of preservation (HPP)."
		else:
			o2 += "\nUsing Time-variable Poisson Process of preservation (TPP) at: "
			for i in times_q_shift: o2 += "%s " % (i)

	else: o2+="Using Non-Homogeneous Poisson Process of preservation (NHPP)."

version_notes="""\n
Please cite: \n%s\n
Feedback and support: pyrate.help@gmail.com
OS: %s %s
Python version: %s\n
Numpy version: %s
Scipy version: %s\n
Random seed: %s %s
""" % (citation,platform.system(), platform.release(), sys.version, np.version.version, scipy.version.version,rseed,CPPlib)



o=''.join([o0,o1,o2,version_notes])
out_sum = "%s/%s_sum.txt" % (path_dir,suff_out)
sumfile = open(out_sum , "wb",0)
sumfile.writelines(o)
sumfile.close()

# OUTPUT 1 LOG MCMC
out_log = "%s/%s_mcmc.log" % (path_dir, suff_out) #(path_dir, output_file, out_run)
logfile = open(out_log , "wb",0)
if fix_SE == 0:
	if TPP_model == 0:
		head="it\tposterior\tprior\tPP_lik\tBD_lik\tq_rate\talpha\t"
	else:
		head="it\tposterior\tprior\tPP_lik\tBD_lik\t"
		for i in range(time_framesQ): head += "q_%s\t" % (i)
		head += "alpha\t"
		if pert_prior[1]==0: head +="hypQ\t"
else:
	head="it\tposterior\tprior\tBD_lik\t"

if model_cov>=1:
	head += "cov_sp\tcov_ex\tcov_q\t"
	if est_COVAR_prior == 1: head+="cov_hp\t"
if TDI<2:
	head += "root_age\tdeath_age\t"
	if TDI==1: head += "beta\t"
	if est_hyperP == 1:
		head += "hypL\thypM\t"

	if use_ADE_model == 0 and useDiscreteTraitModel == 0:
		for i in range(time_framesL): head += "lambda_%s\t" % (i)
		for i in range(time_framesM): head += "mu_%s\t" % (i)
	elif use_ADE_model >= 1:
		if use_ADE_model == 1:
			head+="w_shape\t"
			for i in range(time_framesM): head += "w_scale_%s\t" % (i)
		elif use_ADE_model == 2:
			head+="corr_lambda\t"
			for i in range(time_framesM): head += "corr_mu_%s\t" % (i)
		for i in range(time_framesM): head += "mean_longevity_%s\t" % (i)
	elif useDiscreteTraitModel == 1:
		for i in range(len(lengths_B_events)): head += "lambda_%s\t" % (i)
		for i in range(len(lengths_D_events)): head += "mu_%s\t" % (i)

	if fix_Shift== 0:
		for i in range(1,time_framesL): head += "shift_sp_%s\t" % (i)
		for i in range(1,time_framesM): head += "shift_ex_%s\t" % (i)

	if analyze_tree >=1:
		if analyze_tree==4:
			head += "tree_lik\t"
			for i in range(time_framesL): head += "tree_sp_%s\t" % (i)
			for i in range(time_framesL): head += "tree_ex_%s\t" % (i)
		else:
			head += "tree_lik\ttree_sp\ttree_ex\t"

elif TDI == 2: head+="k_birth\tk_death\troot_age\tdeath_age\t"
elif TDI == 3: head+="k_birth\tk_death\tDPP_alpha_L\tDPP_alpha_M\troot_age\tdeath_age\t"
elif TDI == 4: head+="k_birth\tk_death\tRJ_hp\troot_age\tdeath_age\t"


if useDiscreteTraitModel == 1:
	for i in range(len(lengths_B_events)): head += "S_%s\t" % (i)
head += "tot_length"
head=head.split('\t')
if use_se_tbl == 0: tot_number_of_species = len(taxa_names)

if fix_SE == 0:
	for i in taxa_names: head.append("%s_TS" % (i))
	for i in taxa_names: head.append("%s_TE" % (i))
wlog=csv.writer(logfile, delimiter='\t')
wlog.writerow(head)

#logfile.writelines(head)
logfile.flush()
os.fsync(logfile)

# OUTPUT 2 MARGINAL RATES
if args.log_marginal_rates == -1: # default values
	if TDI==4 or use_ADE_model != 0: log_marginal_rates_to_file = 0
	else: log_marginal_rates_to_file = 1
else:
	log_marginal_rates_to_file = args.log_marginal_rates

# save regular marginal rate file
if TDI!=1 and use_ADE_model == 0 and useDiscreteTraitModel == 0 and log_marginal_rates_to_file==1: # (path_dir, output_file, out_run)
	if useBounded_BD == 1: max_marginal_frame = boundMax+1
	else: max_marginal_frame = max(FA)
	marginal_frames= array([int(fabs(i-int(max_marginal_frame))) for i in range(int(max_marginal_frame)+1)])
	if log_marginal_rates_to_file==1:
		out_log_marginal = "%s/%s_marginal_rates.log" % (path_dir, suff_out)
		marginal_file = open(out_log_marginal , "wb")
		head="it\t"
		for i in range(int(max_marginal_frame)+1): head += "l_%s\t" % i #int(fabs(int(max(FA))))
		for i in range(int(max_marginal_frame)+1): head += "m_%s\t" % i #int(fabs(int(max(FA))))
		for i in range(int(max_marginal_frame)+1): head += "r_%s\t" % i #int(fabs(int(max(FA))))
		head=head.split('\t')
		wmarg=csv.writer(marginal_file, delimiter='	')
		wmarg.writerow(head)
		marginal_file.flush()
		os.fsync(marginal_file)

# save files with sp/ex rates and times of shift
elif log_marginal_rates_to_file==0:
	marginal_sp_rate_file_name = "%s/%s_sp_rates.log" % (path_dir, suff_out)
	marginal_sp_rate_file = open(marginal_sp_rate_file_name , "w")
	w_marg_sp=csv.writer(marginal_sp_rate_file, delimiter='\t')
	marginal_sp_rate_file.flush()
	os.fsync(marginal_sp_rate_file)
	marginal_ex_rate_file_name = "%s/%s_ex_rates.log" % (path_dir, suff_out)
	marginal_ex_rate_file = open(marginal_ex_rate_file_name , "w")
	w_marg_ex=csv.writer(marginal_ex_rate_file, delimiter='\t')
	marginal_ex_rate_file.flush()
	os.fsync(marginal_ex_rate_file)
	marginal_frames=0

# OUTPUT 3 MARGINAL LIKELIHOOD
elif TDI==1:
	out_log_marginal_lik = "%s/%s_marginal_likelihood.txt" % (path_dir, suff_out)
	marginal_file = open(out_log_marginal_lik , "wb")
	marginal_file.writelines(o)
	marginal_frames=0
else: marginal_frames=0
# OUTPUT 4 MARGINAL SHIFT TIMES
#if TDI==2:
#	out_log_marginal_time = "%s/%s_marginal_t_shifts.log" % (path_dir, suff_out)
#	marginal_file_time = open(out_log_marginal_time , "wb")
#	head="it\t"
#	for i in range(25): head += "t_l%s\t" % i #int(fabs(int(max(FA))))
#	for i in range(25): head += "t_m%s\t" % i #int(fabs(int(max(FA))))
#	head=head.split('\t')
#	wmarg_t=csv.writer(marginal_file_time, delimiter='	')
#	wmarg_t.writerow(head)
#	marginal_file.flush()
if fix_SE == 1 and fix_Shift == 1:
	time_frames  = sort(np.array(list(fixed_times_of_shift) + [0,max(fixed_ts)]))
	B = sort(time_frames)+0.000001 # add small number to avoid counting extant species as extinct
	ss1 = np.histogram(fixed_ts,bins=B)[0][::-1]
	ss1[0] = ss1[0]-no_starting_lineages
	ee2 = np.histogram(fixed_te,bins=B)[0][::-1]
	len_SS1,len_EE1 = list(),list()
	S_time_frame =list()
	time_frames = time_frames[::-1]
	for i in range(len(time_frames)-1):
		up, lo = time_frames[i], time_frames[i+1]
		len_SS1.append(ss1[i])
		len_EE1.append(ee2[i])
		inTS = np.fmin(fixed_ts,up)
		inTE = np.fmax(fixed_te,lo)
		temp_S = inTS-inTE
		S_time_frame.append(sum(temp_S[temp_S>0]))

	len_SS1 = np.array(len_SS1)
	len_EE1 = np.array(len_EE1)
	S_time_frame = np.array(S_time_frame)

########################## START MCMC ####################################
t1 = time.time()
if burnin<1 and burnin>0:
	burnin = int(burnin*mcmc_gen)

def start_MCMC(run):
	t1 = time.clock()
	print "started at: ",time.ctime()
	# marginal_file is either for rates or for lik
	return MCMC([0,run, IT, sample_freq, print_freq, temperatures, burnin, marginal_frames, list()])

# Metropolis-coupled MCMC (Altekar, et al. 2004)
if use_seq_lik == 0 and runs>1:
	marginal_frames= array([int(fabs(i-int(max(FA)))) for i in range(int(max(FA))+1)])
	pool = mcmcMPI(num_proc)
	res = pool.map(start_MCMC, range(runs))
	current_it=0
	swap_rate, attempts=0, 0
	while current_it<mcmc_gen:
		n1=np.random.random_integers(0,runs-1,2)
		temp=1./(1+n1*temp_pr)
		[j,k]=n1
		#print "try: ", current_it, j,k #, res[j] #[2],res[0][2], res[1][2],res[2][2]
		r=(res[k][2]*temp[0]+res[j][2]*temp[1]) - (res[j][2]*temp[0]+res[k][2]*temp[1])
		if r>=log(random.random()) and j != k:
			args=list()
			best,ind=res[1][2],0
			#print current_it, "swap %s with chain %s [%s / %s] temp. [%s / %s]" % (j, k, res[j][2],res[k][2], temp[0],temp[1])
			swap_rate+=1
			res_temp1=res[j]
			res_temp2=res[k]
			res[j]=res_temp2
			res[k]=res_temp1
		current_it=res[0][0]
		res[0][0]=0
		args=list()
		for i in range(runs):
			seed=0
			args.append([current_it,i, current_it+IT, sample_freq, print_freq, temperatures, burnin, marginal_frames, res[i]])
		res = pool.map(MCMC, args)
		#except: print current_it,i, current_it+IT
		attempts+=1.
		#if attempts % 100 ==0:
		#	print "swap freq.", swap_rate/attempts
		#	for i in range(runs): print "chain", i, "post:", res[i][2], sum(res[i][5]-res[i][6])

else:
	if runs>1: print("\nWarning: MC3 algorithm requires multi-threading.\nUsing standard (BD)MCMC algorithm instead.\n")
	res=start_MCMC(0)
print "\nfinished at:", time.ctime(), "\nelapsed time:", round(time.time()-t1,2), "\n"
logfile.close()

quit()
