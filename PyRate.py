#!/usr/bin/env python 
# Created by Daniele Silvestro on 02/03/2012 => pyrate.help@gmail.com 
import argparse, os,sys, platform, time, csv, glob
import random as rand
import warnings
version= "PyRate"
build  = "20161005"
if platform.system() == "Darwin": sys.stdout.write("\x1b]2;%s\x07" % version)

citation= """Silvestro, D., Schnitzler, J., Liow, L.H., Antonelli, A. and Salamin, N. (2014)
Bayesian Estimation of Speciation and Extinction from Incomplete Fossil
Occurrence Data. Systematic Biology, 63, 349-367.

Silvestro, D., Salamin, N., Schnitzler, J. (2014)
PyRate: A new program to estimate speciation and extinction rates from
incomplete fossil record. Methods in Ecology and Evolution, 5, 1126-1131.

Silvestro D., Cascales-Minana B., Bacon C. D., Antonelli A. (2015)
Revisiting the origin and diversification of vascular plants through a
comprehensive Bayesian analysis of the fossil record. New Phytologist,
doi:10.1111/nph.13247. 
"""
print("""
                           %s - %s

           Bayesian estimation of speciation and extinction
                  rates from fossil occurrence data        

               Daniele Silvestro, Jan Schnitzler et al.
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
	use_seq_lik=False
	if platform.system() == "Windows" or platform.system() == "Microsoft": use_seq_lik=True
except(ImportError): 
	print("\nWarning: library multiprocessing not found.\nPyRate will use (slower) sequential likelihood calculation. \n")
	use_seq_lik=True

if platform.system() == "Windows" or platform.system() == "Microsoft": use_seq_lik=True

version_details="PyRate %s; OS: %s %s; Python version: %s; Numpy version: %s; Scipy version: %s" \
% (build, platform.system(), platform.release(), sys.version, np.version.version, scipy.version.version)

### numpy print options ###
np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit

original_stderr = sys.stderr
NO_WARN = original_stderr #open('pyrate_warnings.log', 'w')
small_number= 1e-50

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
	z1=t[burnin:,k_ind[0]]  # list of shifts (lambda)
	z2=t[burnin:,k_ind[1]]  # list of shifts (mu)
	y1= max(max(z1),max(z2))
	print("Model           Probability")
	print("          Speciation  Extinction")
	for i in range(1,int(y1)+1):
		k_l=float(len(z1[z1==i]))/len(z1)
		k_m=float(len(z2[z2==i]))/len(z2)
		print("%s-rate    %s      %s" % (i,round(k_l,4),round(k_m,4)))
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
		print "   B/I    D      Rel.pr"
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

def get_DT(T,s,e): # returns the Diversity Trajectory of s,e at times T (x10 faster)
	B=np.sort(np.append(T,T[0]+1))+.000001 # the + .0001 prevents problems with identical ages
	ss1 = np.histogram(s,bins=B)[0]
	ee2 = np.histogram(e,bins=B)[0]
	DD=(ss1-ee2)[::-1]
	#return np.insert(np.cumsum(DD),0,0)[0:len(T)]
	return np.cumsum(DD)[0:len(T)] 


########################## PLOT RTT ##############################
def plot_RTT(infile,burnin, file_stem="",one_file=False, root_plot=0, plot_type=1):
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
	if one_file==True: files=["%s/%smarginal_rates.log" % (infile,file_stem)]
	
	stem_file=files[0]
	name_file = os.path.splitext(os.path.basename(stem_file))[0]

	wd = "%s" % os.path.dirname(stem_file)
	#print(name_file, wd)
	print "found", len(files), "log files...\n"

	########################################################
	######           DETERMINE MIN ROOT AGE           ######
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
	######            COMBINE ALL LOG FILES           ######
	########################################################
	print "\ncombining all files...",
	file_n=0
	for f in files:
		file_name =  os.path.splitext(os.path.basename(f))[0]
		#print file_name
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
	print(shape(R_tbl))

	########################################################
	######               CALCULATE HPDs               ######
	########################################################
	print "\ncalculating HPDs...",
	def get_HPD(threshold=.95):
		L_hpd_m,L_hpd_M=[],[]
		M_hpd_m,M_hpd_M=[],[]
		R_hpd_m,R_hpd_M=[],[]
		sys.stdout.write(".")
		sys.stdout.flush()
		for time_ind in range(shape(R_tbl)[1]):
			hpd1=np.around(calcHPD(L_tbl[:,time_ind],threshold),decimals=no_decimals)
			hpd2=np.around(calcHPD(M_tbl[:,time_ind],threshold),decimals=no_decimals)
			hpd3=np.around(calcHPD(R_tbl[:,time_ind],threshold),decimals=no_decimals)
				
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
	R_tbl_mean=np.around(np.mean(R_tbl,axis=0),no_decimals)
	mean_rates=np.array([L_tbl_mean,L_tbl_mean,M_tbl_mean,M_tbl_mean,R_tbl_mean,R_tbl_mean] )
	
	nonzero_rate = L_tbl_mean+ M_tbl_mean
	NA_ind = (nonzero_rate==0).nonzero()[0]
	
	hpds95[:,NA_ind] = np.nan
	#hpds50[:,NA_ind] = np.nan
	mean_rates[:,NA_ind] = np.nan
	#print(np.shape(np.array(hpds50)	), np.shape(L_tbl_mean))

	########################################################
	######                  PLOT RTTs                 ######
	########################################################
	print "\ngenerating R file...",
	out="%s/%s_RTT.r" % (wd,name_file)
	newfile = open(out, "wb") 
	Rfile="# %s files combined:\n" % (len(files))
	for f in files: Rfile+="# \t%s\n" % (f)	
	Rfile+= """\n# 95% HPDs calculated using code from Biopy (https://www.cs.auckland.ac.nz/~yhel002/biopy/)"""
		
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		Rfile+= "\n\npdf(file='%s\%s_RTT.pdf',width=0.6*9, height=0.6*21)\npar(mfrow=c(3,1))" % (wd,name_file) # 9
	else: 
		Rfile+= "\n\npdf(file='%s/%s_RTT.pdf',width=0.6*9, height=0.6*21)\npar(mfrow=c(3,1))" % (wd,name_file) # 9

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
		return R_code

	Rfile += RTT_plot_in_R([hpds95,mean_rates],.5) # ,hpds50

	Rfile += "\nn <- dev.off()"
	newfile.writelines(Rfile)
	newfile.close()
	print "\nAn R script with the source for the RTT plot was saved as: %sRTT.r\n(in %s)" % (name_file, wd)
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		cmd="cd %s; Rscript %s\%s_RTT.r" % (wd,wd,name_file)
	else: 
		cmd="cd %s; Rscript %s/%s_RTT.r" % (wd,wd,name_file)
	os.system(cmd)
	print "done\n"



########################## PLOT TS/TE STAT ##############################
def plot_tste_stats(tste_file, EXT_RATE, step_size,no_sim_ex_time,burnin,rescale):
	step_size=int(step_size)
	# read data
	print "Processing data..."
	tbl = np.loadtxt(tste_file,skiprows=1)
	j_max=(np.shape(tbl)[1]-1)/2
	j=np.arange(j_max)
	ts = tbl[:,2+2*j]*rescale
	te = tbl[:,3+2*j]*rescale
	wd = "%s" % os.path.dirname(tste_file)
	# create out file
	out_file_name = os.path.splitext(os.path.basename(tste_file))[0]
	out_file="%s/%s" % (wd,out_file_name+"_stats.txt")
	out_file=open(out_file, "wb")
	out_file.writelines("time\tdiversity\tm_div\tM_div\tmedian_genus_age\tm_age\tM_age\tturnover\tm_turnover\tM_turnover\tlife_exp\tm_life_exp\tM_life_exp\t")
	
	root = int(np.max(ts)+1)
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
	R_script = """
	setwd("%s")
	tbl = read.table(file = "%s_stats.txt",header = T)
	pdf(file='f%s_stats.pdf',width=12, height=9)
	time = -tbl$time
	par(mfrow=c(2,2))
	library(scales)
	plot(time,tbl$diversity, type="l",lwd = 2, ylab= "Number of lineages", xlab="Time (Ma)", main="Diversity through time", ylim=c(0,max(tbl$M_div,na.rm =T)+1),xlim=c(min(time),0))
	polygon(c(time, rev(time)), c(tbl$M_div, rev(tbl$m_div)), col = alpha("#504A4B",0.5), border = NA)
	plot(time,tbl$median_genus_age, type="l",lwd = 2, ylab = "Median age", xlab="Time (Ma)", main= "Taxon age", ylim=c(0,max(tbl$M_age,na.rm =T)+1),xlim=c(min(time),0))
	polygon(c(time, rev(time)), c(tbl$M_age, rev(tbl$m_age)), col = alpha("#504A4B",0.5), border = NA)
	plot(time,tbl$turnover, type="l",lwd = 2, ylab = "Fraction of new taxa", xlab="Time (Ma)", main= "Turnover", ylim=c(0,max(tbl$M_turnover,na.rm =T)+.1),xlim=c(min(time),0))
	polygon(c(time, rev(time)), c(tbl$M_turnover, rev(tbl$m_turnover)), col = alpha("#504A4B",0.5), border = NA)
	plot(time,tbl$life_exp, type="l",lwd = 2, ylab = "Median longevity", xlab="Time (Ma)", main= "Taxon (estimated) longevity", ylim=c(0,max(tbl$M_life_exp,na.rm =T)+1),xlim=c(min(time),0))
	polygon(c(time, rev(time)), c(tbl$M_life_exp, rev(tbl$m_life_exp)), col = alpha("#504A4B",0.5), border = NA)
	n<-dev.off()
	""" % (wd, out_file_name,out_file_name)
	R_file.writelines(R_script)
	R_file.close()
	print "\nAn R script with the source for the stat plot was saved as: \n%s" % (R_file_name)
	cmd="cd %s; Rscript %s" % (wd,out_file_name+"_stats.R")
	os.system(cmd)
	print "done\n"



########################## COMBINE LOG FILES ##############################
def comb_log_files(path_to_files,burnin=0,tag="",resample=0,col_tag=[]):
	infile=path_to_files
	sys.path.append(infile)
	direct="%s/*%s*.log" % (infile,tag)
	files=glob.glob(direct)
	files=sort(files)
	print "found", len(files), "log files...\n"
	j=0
	burnin = int(burnin)
	for f in files:
		if 2>1: #try:
			file_name =  os.path.splitext(os.path.basename(f))[0]
			print file_name,
			t_file=loadtxt(f, skiprows=max(1,int(burnin)))
			shape_f=shape(t_file)
			print shape_f
			#t_file = t[burnin:shape_f[0],:]#).astype(str)
			if resample>0:
				r_ind= sort(np.random.randint(0,shape_f[0],resample))
				t_file = t_file[r_ind,:]

		#except: print "ERROR in",f		
		if len(col_tag) == 0:
			if j==0: 
				head = next(open(f))#.split()
				comb = t_file
			else:
				comb = np.concatenate((comb,t_file),axis=0)
		else: 
			head_temp = next(open(f)).split() # should be faster
			sp_ind_list=[]
			for TAG in col_tag:
				sp_ind_list+=[head_temp.index(s) for s in head_temp if s == TAG]
			
			sp_ind= np.array(sp_ind_list)
			#print "INDEXES",sp_ind
			if j==0: 
				head_temp= np.array(head_temp)
				head_t= ["%s\t" % (i) for i in head_temp[sp_ind]]
				head=""
				for i in head_t: head+=i
				head+="\n"
				print "found", len(head_t), "columns"
				comb = t_file[:,sp_ind]
			else:
				comb = np.concatenate((comb,t_file[:,sp_ind]),axis=0)
			
		j+=1

	#print shape(comb)	
	if col_tag == "":
		sampling_freq= comb[1,0]-comb[0,0]
		comb[:,0] = (np.arange(0,len(comb))+1)*sampling_freq
		fmt_list=['%i']
	else: fmt_list=['%i']
	for i in range(1,np.shape(comb)[1]): fmt_list.append('%4f')
	outfile = "%s/combined_%s%s_files.log" % (infile,len(files),tag)
	comb[:,0] = np.arange(len(comb[:,0]))
	
	with open(outfile, 'wb') as f:
		f.write(head)
		np.savetxt(f, comb, delimiter="\t",fmt=fmt_list,newline="\r") #)

########################## INITIALIZE MCMC ##############################
def get_gamma_rates(a):
	b=a
	m = gdtrix(b,a,YangGammaQuant) # user defined categories
	s=args.ncat/sum(m) # multiplier to scale the so that the mean of the discrete distribution is one
	return array(m)*s # SCALED VALUES

def init_ts_te(FA,LO):
	ts=FA+np.random.exponential(.75, len(FA)) # exponential random starting point
	tt=np.random.beta(2.5, 1, len(LO)) # beta random starting point
	ts=FA+(.025*FA) #IMPROVE INIT
	if max(ts) > boundMax:
		ts[ts>boundMax] = np.random.uniform(FA[ts>boundMax],boundMax,len(ts[ts>boundMax])) # avoit init values outside bounds
	te=LO-(.025*LO) #IMPROVE INIT
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


def update_rates_multiplier(L,M,tot_L,mod_d3):
	if use_ADE_model is False and use_Death_model is False:
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


def update_ts_te_old(ts, te, d1):
	tsn, ten= zeros(len(ts)), zeros(len(te))
	f1=np.random.random_integers(1,frac1) #int(frac1*len(FA)) #-np.random.random_integers(0,frac1*len(FA)-1)) 
	ind=np.random.choice(SP_in_window,f1) # update only values in SP/EX_in_window
	#ind=np.random.random_integers(0,len(ts)-1,f1) 
	tsn[ind]=np.random.uniform(-d1,d1,len(tsn[ind]))
	tsn = abs(tsn+ts)		                         # reflection at min boundary (0)
	ind=(tsn<FA).nonzero()					 # indices of ts<FA (not allowed)
	tsn[ind] = abs(tsn[ind]-FA[ind])+FA[ind]         # reflection at max boundary (FA)
	tsn=abs(tsn)

	ind=np.random.choice(EX_in_window,f1)
	#ind=np.random.random_integers(0,len(ts)-1,f1) 
	ten[ind]=np.random.uniform(-d1,d1,len(ten[ind]))
	ten=abs(ten+te)		                         # reflection at min boundary (0)
	ind2=(ten>LO).nonzero()                          
	ten[ind2] = abs(LO[ind2]-(ten[ind2]-LO[ind2]))   # reflection at max boundary (LO)
	ind2=(ten>LO).nonzero()                          
	ten[ind2] = te[ind2]                             # reflection at max boundary (LO)
	ind=(LO==0).nonzero()                            # indices of LO==0 (extant species)
	ten[ind]=0
	return tsn,ten

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
	ten[LO==0]=0                                     # indices of LO==0 (extant species)
	S= tsn-ten
	if min(S)<=0: print S
	tsn[SP_not_in_window] = max([boundMax, max(tsn[SP_in_window])])
	ten[EX_not_in_window] = min([boundMin, min(ten[EX_in_window])])
	return tsn,ten

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

def prior_times_frames(t, root, tip_age,a): # un-normalized Dirichlet (truncated)
	diff_t, min_t = abs(np.diff(t)), min(t)
	if min(diff_t)<=min_allowed_t: return -inf
	elif (min_t<=tip_age+min_allowed_t) and (min_t>0): return -inf
	else: 
		t_rel=diff_t/root
		return (a-1)*log(t_rel)

def prior_beta(x,a): 
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
		#P   = (Beta/(Alpha*(1+Alpha))) *    (Alpha/(1+Alpha))**N
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
def BD_lik_discrete_trait(arg):
	[ts,te,L,M]=arg
	S = ts-te
	lik0 =  sum(log(L)*lengths_B_events )    #
	lik1 = -sum(L*sum(S))                   # assumes that speiation can arise from any trait state
	#lik1 = -sum([L[i]*sum(S[ind_trait_species==i]) for i in range(len(L))])
	lik2 =  sum(log(M)*lengths_D_events)                                        # Trait specific extinction
	lik3 = -sum([M[i]*sum(S[ind_trait_species==i]) for i in range(len(M))]) # only species with a trait state can go extinct
	return sum(lik0+lik1+lik2+lik3)


def BPD_lik_vec_times(arg):
	[ts,te,time_frames,L,M]=arg
	if fix_SE is False or fix_Shift is False:
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
			S    = inTS-inTE
			# speciation
			if use_poiD is False:
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
	n_t_ts[(n_t_ts>up).nonzero()]=up           # for which length is accounted only from $up$ rather than from $ts$
	
	n_t_te[n_all_inframe]= te[n_all_inframe]   # extinction events in time frame
	n_t_te[np.intersect1d((n_t_te<lo).nonzero()[0], n_all_inframe)]=lo     # for which length is accounted only until $lo$ rather than to $te$

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
	else:           # constant rate model
		lik= log(rate)*len(i_events) -rate*sum(n_S) #log(rate)*len(i_events) +sum(-rate*n_S)
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
	else:           # constant rate model
		#print par, len(i_events), len(te)
		lik= log(rate)*len(i_events) -rate*sum(n_S) #log(rate)*len(i_events) +sum(-rate*n_S)
	return lik


# WORK IN PROGRESS
def cdf_WR(W_shape,W_scale,x):
	return (x/W_scale)**(W_shape)

def log_wr(t,W_shape,W_scale):
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
	death_lik_de = sum(log_wr(de, W_shape, W_scale)) # log probability of death event
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
	P = pdf_W_poi(W_shape,W_scale,q,v)                 # partial integral (0 => xLim) via numerical integration
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
	if multiHPP is False: # because in this case q_rates[0] is alpha par of gamma model (set to 1)
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
	P = pdf_W_poi(W_shape,W_scale_species,q_rate_species,v.T)                 # partial integral (0 => xLim) via numerical integration
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
		S    = inTS-inTE
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
	[te,ts,time_frames,q_rates,i]=arg	
	i=int(i) # species number
	k_vec = occs_sp_bin[i] # no. occurrences per time bin per species
	# e.g. k_vec = [0,0,1,12.,3,0]
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
	lik = sum(-q_rates[ind]*d + log(q_rates[ind])*k_vec[ind]) - log(1-exp(sum(-q_rates[ind]*d))) -sum(log(np.arange(1,sum(k_vec)+1))) 
	q=q_rates[0]
	k=sum(k_vec)
	#print -q*(ts-te) + log(q)*k - sum(log(np.arange(1,k+1))) - log(1-exp(-q*(ts-te))), lik
	#print -q*(ts-te) + log(q)*k,  sum(-q_rates[ind]*d + log(q_rates[ind])*k_vec[ind])
	return lik

def HPP_vec_lik_(arg):
	[te,ts,time_frames,q_rates,i]=arg	
	i=int(i) # species number
	k_vec = occs_sp_bin[i] # no. occurrences per time bin per species
	# e.g. k_vec = [0,0,1,12.,3,0]
	
	lik=0
	TotTime=0
	for j in range(1, len(time_frames)):
		if ts<time_frames[j] or te>time_frames[j-1]: pass
		else:
			up = min(ts, time_frames[j-1])
			lo = max(te, time_frames[j])
			dT = up-lo
			k = k_vec[j-1]
			q=q_rates[j-1]
			lik_temp = -q*(dT) + log(q)*k  - log(1-exp(-q*(dT)))
			lik+= lik_temp
			TotTime+=dT
	#print ts-te
	#print -q*(ts-te) + log(q)*k,  sum(-q_rates[ind]*d + log(q_rates[ind])*k_vec[ind])
	return lik



def HOMPP_lik(arg):
	[m,M,shapeGamma,q_rate,i,cov_par, ex_rate]=arg
	i=int(i)
	x=fossil[i]
	lik=0
	k=len(x[x>0]) # no. fossils for species i
	br_length = M-m
	if useBounded_BD is True: 
		br_length = min(M,boundMax)-max(m, boundMin)
	if cov_par ==2: # transform preservation rate by trait value
		q=exp(log(q_rate)+cov_par*(con_trait[i]-parGAUS[0]))
	else: q=q_rate
	if argsG is True:
		YangGamma=get_gamma_rates(shapeGamma)
		qGamma= YangGamma*q
		lik1= -qGamma*(br_length) + log(qGamma)*k - sum(log(np.arange(1,k+1)))  -log(1-exp(-qGamma*(br_length)))
		lik2= lik1-max(lik1)
		lik=log(sum(exp(lik2)*(1./args.ncat)))+max(lik1)
		return lik
	else: 	return -q*(br_length) + log(q)*k - sum(log(np.arange(1,k+1))) - log(1-exp(-q*(br_length)))

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
	if m==0 and use_DA is True: # data augmentation
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
		int_q = betainc(a,b,xB[:,k])* (M-GM)*q    # integral of beta(3,3) at time xB[k] (tranformed time 0)
		MM=np.zeros((len(quant),k))+M     # matrix speciation times of length quant x no. fossils
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
			log_lik_temp_scaled = log_lik_temp-max(log_lik_temp)
			lik = log(sum(exp(log_lik_temp_scaled))/ len(GM))+max(log_lik_temp)
		else: lik= sum(-(int_q) + np.sum((logPERT4_density(MM,z[:,0:k],aa,bb,X)+log(q)), axis=1))
	elif m==0: lik = HOMPP_lik(arg)		
	else:
		C=M-c*(M-m)
		a = 1+ (4*(C-m))/(M-m)
		b = 1+ (4*(-C+M))/(M-m)
		lik = -q*(M-m) + sum(logPERT4_density(M,m,a,b,x)+log(q)) - log(1-exp(-q*(M-m)))
	#if m==0: print i, lik, q, k, min(x),sum(exp(-(int_q)))
	return lik

def NHPPgamma(arg):
	[m,M,shapeGamma,q_rate,i,cov_par, ex_rate]=arg
	i=int(i)
	x=fossil[i]

	k=len(x[x>0])   # no. fossils for species i
	x=sort(x)[::-1] # reverse
	xB1= -(x-M)     # distance fossil-ts

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
		PERT4_den=np.append(W, [W]*(args.ncat-1)).reshape(args.ncat,len(W)).T 
		#lik=log( sum( (exp(-qGamma*(M-m)) * np.prod((PERT4_den*qGamma), axis=0) / (1-exp(-qGamma*(M-m))))*(1./args.ncat)) )
		tempL=exp(-qGamma*(M-m))
		if max(tempL)<1:
			L=log(1-tempL)
			if np.isfinite(sum(L)):
				lik1=-qGamma*(M-m) + np.sum(log(PERT4_den*qGamma), axis=0) - L
				lik2=lik1-max(lik1)
				lik=log(sum(exp(lik2)*(1./args.ncat)))+max(lik1)
			else: lik=-100000
		else: lik=-100000
	elif m==0 and use_DA is False: lik = HOMPP_lik(arg)
	else: lik=NHPP_lik(arg)
	return lik

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
	go=True
	for j in range(len(n_times)-1):
		up,lo=n_times[j],n_times[j+1]
		if len(np.intersect1d((tse <= up).nonzero()[0], (tse > lo).nonzero()[0]))<=1: 
			go=False
			#print len(np.intersect1d((tse <= up).nonzero()[0], (tse > lo).nonzero()[0])), up , lo

	if min(abs(np.diff(n_times)))<=1: return times, R
	elif go is False: return times, R
	else: return n_times, n_R

	#R= insert(R, ind, init_BD(1))
	#n_times= sort(Q3)[::-1]
	#return n_times, R

def kill_prm(times, R, ind):
	P=np.diff(times)         # time intervals
	Pi= abs(P[ind]/times[0]) 	
	P2= P/(1-Pi)
	P3=np.delete(P2, ind)    # remove interval
	Q2=times[0]+cumsum(P3)   # re-adjust length remaining time frames
	Q3=insert(Q2,0,times[0]) # add root
	Q3[len(Q3)-1]=0
	R=np.delete(R,ind)       # remove corresponding rate
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
				if np.random.random()>.5 and use_Death_model is False:
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
				r=np.random.random()                          # parameter based on its deathRate)
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
def add_shift_RJ(rates,times):
	r_time, r_time_ind = random_choice_P(np.diff(times))
	delta_t_prime      = np.random.uniform(0,r_time)
	t_prime            = times[r_time_ind] + delta_t_prime
	times_prime        = np.sort(np.array(list(times)+[t_prime]))[::-1]
	a,b                = 1,0.2
	rate_prime         = np.random.gamma(a,b)
	log_q_prob         = -prior_gamma(rate_prime,a,b) -log(max(times)-min(times)) # log(/prob(proposing_prime_values)) + log(1/timespan)
	rates_prime        = np.insert(rates,r_time_ind+1,rate_prime)
	return rates_prime,times_prime,log_q_prob

def remove_shift_RJ(rates,times):
	rm_shift_ind  = np.random.choice(range(1,len(times)-1))
	rm_shift_time = times[rm_shift_ind]
	times_prime   = times[times != rm_shift_time]
	rm_rate       = rates[rm_shift_ind]
	a,b           = 1,0.2
	log_q_prob    = prior_gamma(rm_rate,a,b) +log(max(times)-min(times)) # log(prob(proposing_prime_values)) - log(timespan)
	rates_prime   = rates[rates != rm_rate]
	return rates_prime,times_prime,log_q_prob
	

def RJMCMC(arg):
	[L,M, timesL, timesM]=arg
	r=np.random.random(2)
	newL,newtimesL,log_q_probL = L,timesL,0
	newM,newtimesM,log_q_probM = M,timesM,0
	
	if np.random.random()>0.5:
		# ADD/REMOVE SHIFT LAMBDA
		if r[0]>0.5: newL,newtimesL,log_q_probL = add_shift_RJ(L,timesL)
		# if 1-rate model this won't do anything, keeping the frequency of add/remove equal
		elif len(L)>1: newL,newtimesL,log_q_probL = remove_shift_RJ(L,timesL) 
	else:
		# ADD/REMOVE SHIFT LAMBDA
		if r[1]>0.5: newM,newtimesM,log_q_probM = add_shift_RJ(M,timesM)
		# if 1-rate model this won't do anything, keeping the frequency of add/remove equal
		elif len(M)>1: newM,newtimesM,log_q_probM = remove_shift_RJ(M,timesM) 
	
	return newL,newtimesL,newM,newtimesM,log_q_probL+log_q_probM


####### BEGIN FUNCTIONS for DIRICHLET PROCESS PRIOR #######

def random_choice_P(vector):
	probDeath=np.cumsum(vector/sum(vector)) # cumulative prob (used to randomly sample one 
	r=np.random.random()                          # parameter based on its deathRate)
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
		IND = random_choice_P(P)[1]  # numpy.random.choice(a, size=None, replace=True, p=None)
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

def get_init_values(mcmc_log_file):
	tbl = np.loadtxt(mcmc_log_file,skiprows=1)
	last_row = np.shape(tbl)[0]-1
	head = next(open(mcmc_log_file)).split()
	ts_index = [head.index(i) for i in head if "_TS" in i]
	te_index = [head.index(i) for i in head if "_TE" in i]
	try:
		q_rates_index = [head.index("alpha"), head.index("q_rate")]
		q_rates = tbl[last_row,q_rates_index]
	except:
		q_rates_index = [head.index(i) for i in head if "q_" in i]
		q_rates = tbl[last_row,q_rates_index]	
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
			lam = np.array([float(len(ts))/(ts-te)      ])    # const rate ML estimator
			mu  = np.array([float(len(te[te>0]))/(ts-te)])    # const rate ML estimator
			hyp = np.ones(2)	
	else:
		lam = np.array([float(len(ts))/(ts-te)      ])    # const rate ML estimator
		mu  = np.array([float(len(te[te>0]))/(ts-te)])    # const rate ML estimator
		hyp = np.ones(2)	
	return [ts,te,q_rates,lam,mu,hyp]





########################## MCMC #########################################

def MCMC(all_arg):
	[it,n_proc, I,sample_freq, print_freq, temperatures, burnin, marginal_frames, arg]=all_arg
	if it==0: # initialize chain
		print("initializing chain...")
		if fix_SE is True: tsA, teA = fixed_ts, fixed_te
		else: tsA, teA = init_ts_te(FA,LO)
		if restore_chain is True: tsA, teA = restore_init_values[0], restore_init_values[1]
		timesLA, timesMA = init_times(max(tsA),time_framesL,time_framesM, min(teA))
		if len(fixed_times_of_shift)>0: timesLA[1:-1],timesMA[1:-1]=fixed_times_of_shift,fixed_times_of_shift
		if fix_edgeShift is True: 
			if len(edgeShifts)==1: 
				timesLA, timesMA = init_times(edgeShifts[0],time_framesL,time_framesM, 0) # starting shift tims within allowed window
				timesLA[1],timesMA[1]= edgeShifts[0],edgeShifts[0]
			if len(edgeShifts)>1: 
				timesLA, timesMA = init_times(edgeShifts[0],time_framesL,time_framesM, edgeShifts[1]) # starting shift tims within allowed window
				timesLA[0],timesMA[0]= max(tsA),max(tsA)
				timesLA[1],timesMA[1]= edgeShifts[0],edgeShifts[0]
				timesLA[-2],timesMA[-2]= edgeShifts[1],edgeShifts[1]
				print timesLA
				print timesMA
				#quit()
		if TDI<3:
			LA = init_BD(len(timesLA))
			MA = init_BD(len(timesMA))
			if restore_chain is True: 
				LAt = restore_init_values[3]
				MAt = restore_init_values[4]
				if len(LAt) == len(LA): LA = LAt # if restored mcmc has different number of rates ignore them
				if len(MAt) == len(MA): MA = MAt
			if use_ADE_model is True: MA = np.random.uniform(3,5,len(timesMA)-1)
			if use_Death_model is True: LA = np.ones(1)
			if useDiscreteTraitModel is True: 
				LA = init_BD(len(lengths_B_events)+1)
				MA = init_BD(len(lengths_D_events)+1)
				
		else : ### DPP
			LA = init_BD(1) # init 1 rate
			MA = init_BD(1) # init 1 rate
			indDPP_L = np.zeros(len(timesLA)-1).astype(int) # init category indexes
			indDPP_M = np.zeros(len(timesLA)-1).astype(int) # init category indexes
			alpha_par_Dir_L = np.random.uniform(0,1) # init concentration parameters
			alpha_par_Dir_M = np.random.uniform(0,1) # init concentration parameters
		
		q_ratesA,cov_parA = init_q_rates() # use 1 for symmetric PERT
		if multiHPP is True: # init multiple q rates
			q_ratesA = np.zeros(time_framesQ)+q_ratesA[1]
		if restore_chain is True: q_ratesA = restore_init_values[2]
		
		if est_COVAR_prior is True: 
			covar_prior = 1.
			cov_parA = np.random.random(3)*f_cov_par # f_cov_par is 0 or >0 depending on COVAR model
		else: covar_prior = covar_prior_fixed
			
			
		#if fix_hyperP is False:	hyperPA=np.ones(2)
		hyperPA = hypP_par
		if restore_chain is True: hyperPA = restore_init_values[5]
		
		if argsG is False and multiHPP is False: q_ratesA[0]=1
		SA=sum(tsA-teA)
		W_shapeA=1.

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
			
		if fix_SE is True: 
			rr=random.uniform(f_update_q,1)
			stop_update=0
			tsA, teA= fixed_ts, fixed_te
			lik_fossilA=np.zeros(1)
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
		cov_par=zeros(3)
		L,M=zeros(len(LA)),zeros(len(MA))
		tot_L=sum(tsA-teA)
		hasting=0

		# autotuning
		if TDI != 1: tmp=0
		mod_d1= d1           # window size ts, te
		mod_d3= list_d3[tmp] # window size rates
		mod_d4= list_d4[tmp] # window size shift times
		
        
		if rr<f_update_se: # ts/te
			ts,te=update_ts_te(tsA,teA,mod_d1)
			tot_L=sum(ts-te)
		elif rr<f_update_q: # q/alpha
			q_rates=np.zeros(len(q_ratesA))+q_ratesA
			if multiHPP is True: q_rates, hasting = update_q_multiplier(q_ratesA,d=1.1,f=0.5)
			elif np.random.random()>.5 and argsG is True: 
				q_rates[0], hasting=update_multiplier_proposal(q_ratesA[0],d2[0]) # shape prm Gamma
			else:
				q_rates[1], hasting=update_multiplier_proposal(q_ratesA[1],d2[1]) #  preservation rate (q)

		elif rr < f_update_lm: # l/m
			if np.random.random()<f_shift and len(LA)+len(MA)>2: 
				if fix_edgeShift is True:
					if len(edgeShifts)>1:
						timesL=update_times(timesLA, edgeShifts[0],edgeShifts[1],mod_d4,2,len(timesL)-2)
						timesM=update_times(timesMA, edgeShifts[0],edgeShifts[1],mod_d4,2,len(timesM)-2)
					else:
						timesL=update_times(timesLA, edgeShifts[0],min(te),mod_d4,2,len(timesL)-1)
						timesM=update_times(timesMA, edgeShifts[0],min(te),mod_d4,2,len(timesM)-1)
						
				else:
					timesL=update_times(timesLA, max(ts),min(te),mod_d4,1,len(timesL))
					timesM=update_times(timesMA, max(ts),min(te),mod_d4,1,len(timesM))
			else: 
				if TDI<2: # 
					if np.random.random()<.95 or est_hyperP is False or fix_hyperP is True:
						L,M,hasting=update_rates(LA,MA,3,mod_d3)
						if use_ADE_model is True: 
							W_shape, hasting2 = update_multiplier_proposal(W_shapeA,1.1)
							hasting+=hasting2
					else:
						hyperP,hasting = update_multiplier_proposal(hyperPA,d_hyperprior)
				else: # DPP or BDMCMC
						L,M,hasting=update_rates(LA,MA,3,mod_d3)

		elif rr<f_update_cov: # cov
			rcov=np.random.random()
			if est_COVAR_prior is True and rcov<0.05: 
				covar_prior = get_post_sd(cov_parA[cov_parA>0]) # est hyperprior only based on non-zero rates
				stop_update=inf
			elif rcov < f_cov_par[0]: # cov lambda
				cov_par[0]=update_parameter_normal(cov_parA[0],-1000,1000,d5[0])
			elif rcov < f_cov_par[1]: # cov mu
				cov_par[1]=update_parameter_normal(cov_parA[1],-1000,1000,d5[1])
			else:
				cov_par[2]=update_parameter_normal(cov_parA[2],-1000,1000,d5[2])

		if constrain_time_frames is True: timesM=timesL
		q_rates[(q_rates==0).nonzero()]=q_ratesA[(q_rates==0).nonzero()]
		L[(L==0).nonzero()]=LA[(L==0).nonzero()]
		M[(M==0).nonzero()]=MA[(M==0).nonzero()]
		cov_par[(cov_par==0).nonzero()]=cov_parA[(cov_par==0).nonzero()]
		max_ts = max(ts)
		timesL[0]=max_ts
		timesM[0]=max_ts
		if fix_SE is False:
			if multiHPP is True: 
				if it==0:  q_time_frames = np.sort(np.array([max_ts,0]+times_q_shift))[::-1]
				else: q_time_frames[0]= max_ts

		# NHPP Lik: multi-thread computation (ts, te)
		# generate args lik (ts, te)
		if fix_SE is False:
			ind1=range(0,len(fossil))
			ind2=[]
			if it>0 and rr<f_update_se: # recalculate likelihood only for ts, te that were updated
				ind1=(ts-te != tsA-teA).nonzero()[0]
				ind2=(ts-te == tsA-teA).nonzero()[0]
			lik_fossil=zeros(len(fossil))

			if len(ind1)>0 and it<stop_update and fix_SE is False:
				# generate args lik (ts, te)
				z=zeros(len(fossil)*7).reshape(len(fossil),7)
				z[:,0]=te
				z[:,1]=ts
				z[:,2]=q_rates[0]   # shape prm Gamma
				z[:,3]=q_rates[1]   # baseline foss rate (q)
				z[:,4]=range(len(fossil))
				z[:,5]=cov_par[2]  # covariance baseline foss rate
				z[:,6]=M[len(M)-1] # ex rate
				if useDiscreteTraitModel is True: z[:,6] = mean(M)
				args=list(z[ind1])
				if num_processes_ts==0:
					for j in range(len(ind1)):
						i=ind1[j] # which species' lik
						if multiHPP is True:  lik_fossil[i] = HPP_vec_lik([te[i],ts[i],q_time_frames,q_rates,i])
						else:
							if argsHPP is True or  frac1==0: lik_fossil[i] = HOMPP_lik(args[j])
							elif argsG is True: lik_fossil[i] = NHPPgamma(args[j]) 
							else: lik_fossil[i] = NHPP_lik(args[j])
				else:
					if argsHPP is True or  frac1==0: lik_fossil[ind1] = array(pool_ts.map(HOMPP_lik, args))
					elif argsG is True: lik_fossil[ind1] = array(pool_ts.map(NHPPgamma, args)) 
					else: lik_fossil[ind1] = array(pool_ts.map(NHPP_lik, args))

		
			if it>0: lik_fossil[ind2] = lik_fossilA[ind2]



		else: lik_fossil=zeros(1)

		if it>=stop_update or stop_update==inf: lik_fossil = lik_fossilA

		# pert_prior defines gamma prior on q_rates[1] - fossilization rate
		if multiHPP is True: prior = sum(prior_gamma(q_rates,pert_prior[0],pert_prior[1]))
		else: prior = prior_gamma(q_rates[1],pert_prior[0],pert_prior[1]) + prior_uniform(q_rates[0],0,20)			
		if est_hyperP is True: prior += ( prior_uniform(hyperP[0],0,20)+prior_uniform(hyperP[1],0,20) )


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
		
		else:
			# Birth-Death Lik: construct 2D array (args partial likelihood)
			# parameters of each partial likelihood and prior (l)
			if stop_update != inf:
				if useDiscreteTraitModel is True:
					likBDtemp = BD_lik_discrete_trait([ts,te,L,M])
						
				elif use_ADE_model is True and multiHPP is True: 
					likBDtemp = BD_age_lik_vec_times([ts,te,timesL,W_shape,M,q_rates,q_time_frames])
				elif fix_Shift == True:
					if use_ADE_model is False: likBDtemp = BPD_lik_vec_times([ts,te,timesL,L,M])
					else: likBDtemp = BD_age_lik_vec_times([ts,te,timesL,W_shape,M,q_rates])
				else:	
					args=list()
					if use_ADE_model is False: # speciation rate is not used under ADE model
						for temp_l in range(len(timesL)-1):
							up, lo = timesL[temp_l], timesL[temp_l+1]
							l = L[temp_l]
							args.append([ts, te, up, lo, l, 'l', cov_par[0],1])
					# parameters of each partial likelihood and prior (m)
					for temp_m in range(len(timesM)-1):
						up, lo = timesM[temp_m], timesM[temp_m+1]
						m = M[temp_m]
						if use_ADE_model is False:
							args.append([ts, te, up, lo, m, 'm', cov_par[1],1])
						elif use_ADE_model is True:
							args.append([ts, te, up, lo, m, 'm', cov_par[1],W_shape,q_rates[1]])
			
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
			
					# NHPP Lik: needs to be recalculated after Alg 3.1
					if fix_SE is False:
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
								if multiHPP is True:  lik_fossil[i] = HPP_vec_lik([te[i],ts[i],q_time_frames,q_rates,i])
								else:
									if argsHPP is True or  frac1==0: lik_fossil[i] = HOMPP_lik(args[j])
									elif argsG is True: lik_fossil[i] = NHPPgamma(args[j]) 
									else: lik_fossil[i] = NHPP_lik(args[j])
						else:
							if argsHPP is True or frac1==0: lik_fossil[ind1] = array(pool_ts.map(HOMPP_lik, args))
							elif argsG is True: lik_fossil[ind1] = array(pool_ts.map(NHPPgamma, args))
							else: lik_fossil[ind1] = array(pool_ts.map(NHPP_lik, args))
			
					sys.stderr = original_stderr
				
				elif TDI==4: # run RJMCMC
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
					
					if num_processes==0:
						likBDtemp=np.zeros(len(args))
						i=0
						for i in range(len(args)):
							likBDtemp[i]=BPD_partial_lik(args[i])
							i+=1
					# multi-thread computation of lik and prior (rates)
					else: likBDtemp = array(pool_lik.map(BPD_partial_lik, args))
					#print sum(likBDtempA), sum(likBDtemp)
					
		
		lik= sum(lik_fossil) + sum(likBDtemp) + PoiD_const

		T= max(ts)
		if TDI < 3:
			prior += sum(prior_times_frames(timesL, max(ts),min(te), lam_s))
			prior += sum(prior_times_frames(timesM, max(ts),min(te), lam_s))
		if TDI ==4: 
			prior += -log(max(ts)-max(te))*len(L-1)  #sum(prior_times_frames(timesL, max(ts),min(te), 1))
			prior += -log(max(ts)-max(te))*len(M-1)  #sum(prior_times_frames(timesM, max(ts),min(te), 1))
		
		priorBD= get_hyper_priorBD(timesL,timesM,L,M,T,hyperP)
		if use_ADE_model is True:
			# M in this case is the vector of Weibull scales
			priorBD+= sum(prior_normal(log(W_shape),2)) # Normal prior on log(W_shape): highest prior pr at W_shape=1
		
		
		prior += priorBD
		###
		if model_cov >0: prior+=sum(prior_normal(cov_par,covar_prior))

		# exponential prior on root age
		prior += prior_root_age(max(ts),max(FA),max(FA))
		
		if temperature==1: 
			tempMC3=1./(1+n_proc*temp_pr)
			lik_alter=lik
		else: 
			tempMC3=1
			lik_alter=(sum(lik_fossil)+ PoiD_const) + (sum(likBDtemp)+ PoiD_const)*temperature
		Post=lik_alter+prior
		if it==0: PostA=Post
		if it>0 and (it-burnin) % (I_effective/len(temperatures)) == 0 and it>burnin or it==I-1: 
			PostA=Post # when temperature changes always accept first iteration
		
		#print Post, PostA, q_ratesA, sum(lik_fossil), sum(likBDtemp),  prior
		if Post>-inf and Post<inf:
			if Post*tempMC3-PostA*tempMC3 + hasting >= log(np.random.random()) or stop_update==inf and TDI in [2,3]: # 
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
				lik_fossilA=lik_fossil
				cov_parA=cov_par
				W_shapeA=W_shape
			
		if it % print_freq ==0 or it==burnin:
			try: l=[round(y, 2) for y in [PostA, likA, priorA, SA]]
			except: 
				print "An error occurred."
				print PostA,lik, prior, prior_root_age(max(ts),max(FA),max(FA)), priorBD, max(ts),max(FA)
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
				if use_ADE_model is True: 
					print "\tWeibull.shape:", round(W_shapeA,3)
					print "\tWeibull.scale:", MA
				else: 
					print "\tsp.rates:", LA
					print "\tex.rates:", MA
				if est_hyperP is True: print "\thyper.prior.par", hyperPA

				
				if model_cov>=1: 
					print "\tcov. (sp/ex/q):", cov_parA
					if est_COVAR_prior is True: print "\tHP_covar:",round(covar_prior,3) 
 				if fix_SE ==False: 
					if multiHPP is True: print "\tq.rates:", q_ratesA
					else: print "\tq.rate:", round(q_ratesA[1], 3), "\tGamma.prm:", round(q_ratesA[0], 3)
					print "\tts:", tsA[0:5], "..."
					print "\tte:", teA[0:5], "..."
			if it<=burnin and n_proc==0: print("\n%s*\tpost: %s lik: %s prior: %s tot length %s" \
			% (it, l[0], l[1], l[2], l[3]))

		if n_proc != 0: pass
		elif it % sample_freq ==0 and it>=burnin or it==0 and it>=burnin:
			s_max=max(tsA)
			if fix_SE ==False:
				if multiHPP is False: log_state = [it,PostA, priorA, sum(lik_fossilA), likA-sum(lik_fossilA), q_ratesA[1], q_ratesA[0]]
				else: log_state= [it,PostA, priorA, sum(lik_fossilA), likA-sum(lik_fossilA)] + list(q_ratesA)
			else:
				log_state= [it,PostA, priorA, likA-sum(lik_fossilA)]

			if model_cov>=1: 
				log_state += cov_parA[0], cov_parA[1],cov_parA[2]
				if est_COVAR_prior is True: log_state += [covar_prior]

			if TDI<2: # normal MCMC or MCMC-TI
				log_state += s_max,min(teA)
				if TDI==1: log_state += [temperature]
				if est_hyperP is True: log_state += list(hyperPA)
				if use_ADE_model is False: log_state += list(LA)
				else: log_state+= [W_shapeA]
				log_state += list(MA) # This is W_scale in the case of ADE models
				if use_ADE_model is True: log_state+= list(MA * gamma(1 + 1./W_shapeA))
				if fix_Shift== False:
					log_state += list(timesLA[1:-1])
					log_state += list(timesMA[1:-1])
			elif TDI in [2,4]: # BD-MCMC
				log_state+= [len(LA), len(MA), s_max,min(teA)]
			else: # DPP
				log_state+= [len(LA), len(MA), alpha_par_Dir_L,alpha_par_Dir_M, s_max,min(teA)]
			 					
			if useDiscreteTraitModel is True: 
				for i in range(len(lengths_B_events)): log_state += [sum(tsA[ind_trait_species==i]-teA[ind_trait_species==i])]
			log_state += [SA]
			if fix_SE ==False:
				log_state += list(tsA)
				log_state += list(teA)
			wlog.writerow(log_state)
			logfile.flush()
			os.fsync(logfile)

			lik_tmp += sum(likBDtempA)
			if TDI in [0,2,4] and n_proc==0 and use_ADE_model is False and useDiscreteTraitModel is False:
				margL=zeros(len(marginal_frames))
				margM=zeros(len(marginal_frames))
				if useBounded_BD is True: min_marginal_frame = boundMin
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
			
		it += 1
	if TDI==1 and n_proc==0: marginal_likelihood(marginal_file, marginal_lik, temperatures)
	if use_seq_lik is False:
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
self_path=os.getcwd()
p = argparse.ArgumentParser() #description='<input file>') 

p.add_argument('-v',         action='version', version=version_details)
p.add_argument('-cite',      help='print PyRate citation', action='store_true', default=False)
p.add_argument('input_data', metavar='<input file>', type=str,help='Input python file - see template',default=[],nargs='*')
p.add_argument('-j',         type=int, help='number of data set in input file', default=1, metavar=1)
p.add_argument('-trait',     type=int, help='number of trait for Cov model', default=1, metavar=1)
p.add_argument('-logT',      type=int, help='Transform trait: 0) False, 1) Ln(x), 2) Log10(x)', default=2, metavar=2)
p.add_argument("-N",         type=float, help='number of exant species') 
p.add_argument("-wd",        type=str, help='path to working directory', default="")
p.add_argument("-out",       type=str, help='output tag', default="")
p.add_argument('-singleton', type=float, help='Remove singletons (min no. occurrences)', default=0, metavar=0)
p.add_argument('-frac_sampled_singleton', type=float, help='Random fraction of singletons not removed', default=0, metavar=0)
p.add_argument("-rescale",   type=float, help='Rescale data (e.g. -rescale 0.001: 1My -> 1Ky)', default=1, metavar=1)
p.add_argument("-translate", type=float, help='Shift data (e.g. -translate 10: 1My -> 10My)', default=0, metavar=0)
p.add_argument('-d',         type=str,help="Load SE table",metavar='<input file>',default="")
p.add_argument('-clade',     type=int, help='clade analyzed (set to -1 to analyze all species)', default=-1, metavar=-1)
p.add_argument('-trait_file',type=str,help="Load trait table",metavar='<input file>',default="")
p.add_argument('-restore_mcmc',type=str,help="Load mcmc.log file",metavar='<input file>',default="")
p.add_argument('-filter',     type=float,help="Filter lineages with all occurrences within time range ",default=[inf,0], metavar=inf, nargs=2)
p.add_argument('-filter_taxa',type=str,help="Filter lineages within list (drop all others) ",default="", metavar="taxa_file")

# PLOTS AND OUTPUT
p.add_argument('-plot',       metavar='<input file>', type=str,help="RTT plot (type 1): provide path to 'marginal_rates.log' files or 'marginal_rates' file",default="")
p.add_argument('-plot2',      metavar='<input file>', type=str,help="RTT plot (type 2): provide path to 'marginal_rates.log' files or 'marginal_rates' file",default="")
p.add_argument('-plot3',      metavar='<input file>', type=str,help="RTT plot (type 3; fixed number of shifts): provide 'mcmc.log' file",default="")
p.add_argument('-grid_plot3', type=float, help='Plot resolution in Myr (only for -plot3 command)', default=0.1, metavar=0.1)
p.add_argument('-root_plot',  type=float, help='User define root age for RTT plots', default=0, metavar=0)
p.add_argument('-tag',        metavar='<*tag*.log>', type=str,help="Tag identifying files to be combined and plotted (-plot) or summarized in SE table (-ginput)",default="")
p.add_argument('-mProb',      type=str,help="Input 'mcmc.log' file",default="")
p.add_argument('-BF',         type=str,help="Input 'marginal_likelihood.txt' files",metavar='<2 input files>',nargs='+',default=[])
p.add_argument("-data_info",  help='Summary information about an input data', action='store_true', default=False)
p.add_argument('-SE_stats',   type=str,help="Calculate and plot stats from SE table:",metavar='<marginal_rates.log file bin_size #_simulations>',nargs='+',default=[])
p.add_argument('-ginput',     type=str,help='generate SE table from *mcmc.log files', default="", metavar="<path_to_mcmc.log>")
p.add_argument('-combLog',    type=str,help='Combine (and resample) log files', default="", metavar="<path_to_log_files>")
p.add_argument('-resample',   type=int,help='Number of samples for each log file (-combLog). Use 0 to keep all samples.', default=0, metavar=0)
p.add_argument('-col_tag',    type=str,help='Columns to be combined using combLog', default=[], metavar="column names",nargs='+')
p.add_argument('-check_names',type=str,help='Automatic check for typos in taxa names (provide SpeciesList file)', default="", metavar="<*_SpeciesList.txt file>")
p.add_argument('-reduceLog',  type=str,help='Reduce file size (mcmc.log) to quickly assess convergence', default="", metavar="<*_mcmc.log file>")

# MCMC SETTINGS
p.add_argument('-n',      type=int, help='mcmc generations',default=10000000, metavar=10000000)
p.add_argument('-s',      type=int, help='sample freq.', default=1000, metavar=1000)
p.add_argument('-p',      type=int, help='print freq.',  default=1000, metavar=1000)
p.add_argument('-b',      type=float, help='burnin', default=0, metavar=0)
p.add_argument('-thread', type=int, help='no. threads used for BD and NHPP likelihood respectively (set to 0 to bypass multi-threading)', default=[0,0], metavar=4, nargs=2)

# MCMC ALGORITHMS
p.add_argument('-A',        type=int, help='0) parameter estimation, 1) marginal likelihood, 2) BDMCMC', default=2, metavar=2)
p.add_argument("-use_DA",   help='Use data augmentation for NHPP likelihood opf extant taxa', action='store_true', default=False)
p.add_argument('-r',        type=int,   help='MC3 - no. MCMC chains', default=1, metavar=1)
p.add_argument('-t',        type=float, help='MC3 - temperature', default=.03, metavar=.03)
p.add_argument('-sw',       type=float, help='MC3 - swap frequency', default=100, metavar=100)
p.add_argument('-M',        type=int,   help='BDMCMC - frequency of model update', default=10, metavar=10)
p.add_argument('-B',        type=int,   help='BDMCMC - birth rate', default=1, metavar=1)
p.add_argument('-T',        type=float, help='BDMCMC - time of model update', default=1.0, metavar=1.0)
p.add_argument('-S',        type=int,   help='BDMCMC - start model update', default=1000, metavar=1000)
p.add_argument('-k',        type=int,   help='TI - no. scaling factors', default=10, metavar=10)
p.add_argument('-a',        type=float, help='TI - shape beta distribution', default=.3, metavar=.3)
p.add_argument('-dpp_f',    type=float, help='DPP - frequency ', default=500, metavar=500)
p.add_argument('-dpp_hp',   type=float, help='DPP - shape of gamma HP on concentration parameter', default=2., metavar=2.)
p.add_argument('-dpp_eK',   type=float, help='DPP - expected number of rate categories', default=2., metavar=2.)
p.add_argument('-dpp_grid', type=float, help='DPP - size of time frames',default=1.5, metavar=1.5)

# PRIORS
p.add_argument('-pL',      type=float, help='Prior - speciation rate (Gamma <shape, rate>) | (if shape=n,rate=0 -> rate estimated)', default=[1.1, 1.1], metavar=1.1, nargs=2)
p.add_argument('-pM',      type=float, help='Prior - extinction rate (Gamma <shape, rate>) | (if shape=n,rate=0 -> rate estimated)', default=[1.1, 1.1], metavar=1.1, nargs=2)
p.add_argument('-pP',      type=float, help='Prior - preservation rate (Gamma <shape, rate>)', default=[1.5, 1.1], metavar=1.5, nargs=2)
p.add_argument('-pS',      type=float, help='Prior - time frames (Dirichlet <shape>)', default=2.5, metavar=2.5)
p.add_argument('-pC',      type=float, help='Prior - Covar parameters (Normal <standard deviation>) | (if pC=0 -> sd estimated)', default=1, metavar=1)
p.add_argument("-cauchy",  type=float, help='Prior - use hyper priors on sp/ex rates (if 0 -> estimated)', default=[-1, -1], metavar=-1, nargs=2)

# MODEL
p.add_argument("-mHPP",    help='Model - Homogeneous Poisson process of preservation', action='store_true', default=False)
#p.add_argument("-multiHPP",help='Model - Poisson process of preservation with shifts', action='store_true', default=False)
p.add_argument('-mL',      type=int, help='Model - no. (starting) time frames (speciation)', default=1, metavar=1)
p.add_argument('-mM',      type=int, help='Model - no. (starting) time frames (extinction)', default=1, metavar=1)
p.add_argument('-mC',      help='Model - constrain time frames (l,m)', action='store_true', default=False)
p.add_argument('-mCov',    type=int, help='COVAR model: 1) speciation, 2) extinction, 3) speciation & extinction, 4) preservation, 5) speciation & extinction & preservation', default=0, metavar=0)
p.add_argument("-mG",      help='Model - Gamma heterogeneity of preservation rate', action='store_true', default=False)
p.add_argument('-mPoiD',   help='Poisson-death diversification model', action='store_true', default=False)
p.add_argument('-mDeath',  help='Pure-death model', action='store_true', default=False)
p.add_argument("-mBDI",    type=int, help='BDI sub-model - 0) birth-death, 1) immigration-death', default=-1, metavar=-1)
p.add_argument("-ncat",    type=int, help='Model - Number of categories for Gamma heterogeneity', default=4, metavar=4)
p.add_argument('-fixShift',metavar='<input file>', type=str,help="Input tab-delimited file",default="")
p.add_argument('-qShift',  metavar='<input file>', type=str,help="Poisson process of preservation with shifts (Input tab-delimited file)",default="")
p.add_argument('-fixSE',   metavar='<input file>', type=str,help="Input mcmc.log file",default="")
p.add_argument('-ADE',     type=int, help='ADE model: 0) no age dependence 1) estimated age dep', default=0, metavar=0)
p.add_argument('-discrete',help='Discrete-trait-dependent BD model (requires -trait_file)', action='store_true', default=False)
p.add_argument('-bound',   type=float, help='Bounded BD model', default=[np.inf, 0], metavar=0, nargs=2)
p.add_argument('-edgeShift',type=float, help='Fixed times of shifts at the edges (when -mL/-mM > 3)', default=[np.inf, 0], metavar=0, nargs=2)


# TUNING
p.add_argument('-tT',     type=float, help='Tuning - window size (ts, te)', default=1., metavar=1.)
p.add_argument('-nT',     type=int,   help='Tuning - max number updated values (ts, te)', default=5, metavar=5)
p.add_argument('-tQ',     type=float, help='Tuning - window sizes (q/alpha)', default=[1.2,1.2], nargs=2)
p.add_argument('-tR',     type=float, help='Tuning - window size (rates)', default=1.2, metavar=1.2)
p.add_argument('-tS',     type=float, help='Tuning - window size (time of shift)', default=1., metavar=1.)
p.add_argument('-fR',     type=float, help='Tuning - fraction of updated values (rates)', default=1., metavar=1.)
p.add_argument('-fS',     type=float, help='Tuning - fraction of updated values (shifts)', default=.7, metavar=.7)
p.add_argument('-tC',     type=float, help='Tuning - window sizes cov parameters (l,m,q)', default=[.2, .2, .15], nargs=3)
p.add_argument('-fU',     type=float, help='Tuning - update freq. (q/alpha,l/m,cov)', default=[.02, .18, .08], nargs=3)
p.add_argument('-multiR', type=int,   help='Tuning - Proposals for l/m: 0) sliding win 1) muliplier ', default=1, metavar=1)
p.add_argument('-tHP',    type=float, help='Tuning - window sizes hyperpriors on l and m', default=[1.2, 1.2], nargs=2)

args = p.parse_args()
t1=time.time()

if args.cite is True:
	sys.exit(citation)
############################ MODEL SETTINGS ############################ 
# PRIORS
L_lam_r,L_lam_m = args.pL # shape and rate parameters of Gamma prior on sp rates
M_lam_r,M_lam_m = args.pM # shape and rate parameters of Gamma prior on ex rates
lam_s = args.pS                              # shape parameter dirichlet prior on time frames
pert_prior = [args.pP[0],args.pP[1]] # gamma prior on foss. rate; beta on mode PERT distribution
covar_prior_fixed=args.pC # std of normal prior on th covariance parameters

# MODEL
time_framesL=args.mL          # no. (starting) time frames (lambda)
time_framesM=args.mM          # no. (starting) time frames (mu)
constrain_time_frames=args.mC # True/False
argsG=args.mG                 # gamma rates
if argsG is True:             # number of gamma categories
	YangGammaQuant=(np.linspace(0,1,args.ncat+1)-np.linspace(0,1,args.ncat+1)[1]/2)[1:]
model_cov=args.mCov           # boolean 0: no covariance 1: covariance (speciation,extinction) 2: covariance (speciation,extinction,preservation)

argsHPP=args.mHPP
############################ MCMC SETTINGS ############################
# GENERAL SETTINGS
TDI=args.A                  # 0: parameter estimation, 1: thermodynamic integration, 2: BD-MCMC
if constrain_time_frames is True or args.fixShift != "":
	if TDI in [2,4]:
		print("\nConstrained shift times (-mC,-fixShift) cannot be used with BDMCMC alorithm. Using standard MCMC instead.\n")
		TDI = 0
if args.ADE==1 and TDI>1: 
	print("\nADE models (-ADE 1) cannot be used with BDMCMC alorithm. Using standard MCMC instead.\n")
	TDI = 0
mcmc_gen=args.n             # no. total mcmc generations
sample_freq=args.s
print_freq=args.p
burnin=args.b
num_processes = args.thread[0]    # BDlik
num_processes_ts = args.thread[1] # NHPPlik
if num_processes+num_processes_ts==0: use_seq_lik = True
if use_seq_lik is True: num_processes,num_processes_ts=0,0
min_allowed_t=1.

# TUNING
d1=args.tT                     # win-size (ts, te)
frac1= args.nT                 # max number updated values (ts, te)
d2=args.tQ                     # win-sizes (q,alpha)
d3=args.tR                     # win-size (rates)
f_rate=args.fR                 # fraction of updated values (rates)
d4=args.tS                     # win-size (time of shift)
f_shift=args.fS                # update frequency (time of shift) || will turn into 0 when no rate shifts
freq_list=args.fU              # generate update frequencies by parm category
d5=args.tC                     # win-size (cov)
d_hyperprior=np.array(args.tHP)          # win-size hyper-priors onf l/m (or W_scale)
if model_cov==0: freq_list[2]=0 
f_update_se=1-sum(freq_list)
if frac1==0: f_update_se=0
[f_update_q,f_update_lm,f_update_cov]=f_update_se+np.cumsum(array(freq_list))

multiR = args.multiR
if multiR==0:
	update_rates =  update_rates_sliding_win
else:
	update_rates = update_rates_multiplier
	d3 = max(args.tR,1.05) # avoid win size < 1


if args.ginput != "" or args.check_names != "" or args.reduceLog != "":
	import imp
	try:
		self_path= os.path.dirname(sys.argv[0])
		try: 
			print "attempt 1"
			lib_DD_likelihood = imp.load_source("lib_DD_likelihood", "%s/pyrate_lib/lib_DD_likelihood.py" % (self_path))
			lib_utilities = imp.load_source("lib_utilities", "%s/pyrate_lib/lib_utilities.py" % (self_path))
		except:
			print "attempt 2"
			lib_DD_likelihood = imp.load_source("lib_DD_likelihood", "%s\pyrate_lib\lib_DD_likelihood.py" % (self_path))
			lib_utilities = imp.load_source("lib_utilities", "%s\pyrate_lib\lib_utilities.py" % (self_path))
	except: 
		self_path=os.getcwd()
		try: 
			print "attempt 3"
			lib_DD_likelihood = imp.load_source("lib_DD_likelihood", "%s/pyrate_lib/lib_DD_likelihood.py" % (self_path))
			lib_utilities = imp.load_source("lib_utilities", "%s/pyrate_lib/lib_utilities.py" % (self_path))
		except:
			print "attempt 4"
			lib_DD_likelihood = imp.load_source("lib_DD_likelihood", "%s\pyrate_lib\lib_DD_likelihood.py" % (self_path))
			lib_utilities = imp.load_source("lib_utilities", "%s\pyrate_lib\lib_utilities.py" % (self_path))
						
		
	#	sys.exit("""\nWarning: library pyrate_lib not found.\nMake sure PyRate.py and pyrate_lib are in the same directory.
	#You can download pyrate_lib here: <https://github.com/dsilvestro/PyRate> \n""")
	if args.ginput != "":
		lib_utilities.write_ts_te_table(args.ginput, tag=args.tag, clade=-1,burnin=int(burnin)+1)
	elif args.check_names != "":
		SpeciesList_file = args.check_names
		lib_utilities.check_taxa_names(SpeciesList_file)
	elif args.reduceLog != "":
		lib_utilities.reduce_log_file(args.reduceLog,max(1,int(args.b)))
	quit()


use_DA = False
if args.use_DA is True: use_DA = True



# freq update CovPar
if model_cov==0: f_cov_par= [0  ,0  ,0 ]
if model_cov==1: f_cov_par= [1  ,0  ,0 ]
if model_cov==2: f_cov_par= [0  ,1  ,0 ]
if model_cov==3: f_cov_par= [.5 ,1  ,0 ]
if model_cov==4: f_cov_par= [0  ,0  ,1 ]
if model_cov==5: f_cov_par= [.33,.66,1 ]

if covar_prior_fixed==0: est_COVAR_prior = True
else: est_COVAR_prior = False

if args.fixShift != "" or TDI==3:     # fix times of rate shift or DPP
	try: 
		try: fixed_times_of_shift=sort(np.loadtxt(args.fixShift))[::-1]
		except(IndexError): fixed_times_of_shift=np.array([np.loadtxt(args.fixShift)])
		f_shift=0
		time_framesL=len(fixed_times_of_shift)+1
		time_framesM=len(fixed_times_of_shift)+1
		min_allowed_t=0
		fix_Shift = True
	except: 
		if TDI==3:
			fixed_times_of_shift=np.arange(0,10000,args.dpp_grid)[::-1] # run fixed_times_of_shift[fixed_times_of_shift<max(FA)] below 
			fixed_times_of_shift=fixed_times_of_shift[:-1]              # after loading input file
			f_shift=0
			time_framesL=len(fixed_times_of_shift)+1
			time_framesM=len(fixed_times_of_shift)+1
			min_allowed_t=0
			fix_Shift = True
		else:
			msg = "\nError in the input file %s.\n" % (args.fixShift)
			sys.exit(msg)
else: 
	fixed_times_of_shift=[]
	fix_Shift = False
if args.edgeShift[0] != np.inf: # and args.edgeShift[1] != 0:
	fix_edgeShift = True
	if args.edgeShift[1] != 0: edgeShifts = args.edgeShift
	else: edgeShifts = [args.edgeShift[0]]
else: fix_edgeShift = False
# BDMCMC & MCMC SETTINGS
runs=args.r              # no. parallel MCMCs (MC3)
if runs>1 and TDI>0: 
	print("\nWarning: MC3 algorithm is not available for TI and BDMCMC. Using a single chain instead.\n")
	runs,TDI=1,0
num_proc = runs          # processors MC3
temp_pr=args.t           # temperature MC3
IT=args.sw
freq_Alg_3_1=args.M      # frequency of model update
birthRate=args.B         # birthRate (=Poisson prior)
len_cont_time=args.T     # length continuous time of model update
start_Alg_3_1=args.S     # start sampling model after


if runs==1 or use_seq_lik is True: 
	IT=mcmc_gen

if TDI==1:                # Xie et al. 2011; Baele et al. 2012
	K=args.k-1.        # K+1 categories
	k=array(range(int(K+1)))
	beta=k/K
	alpha=args.a            # categories are beta distributed
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
freq_dpp       = args.dpp_f
hp_gamma_shape = args.dpp_hp
target_k       = args.dpp_eK

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
list_files_BF=sort(args.BF)
file_stem=args.tag
root_plot=args.root_plot
if path_dir_log_files != "":
	if plot_type==3:
		if 2>1: #try:
			self_path= os.path.dirname(sys.argv[0])
			import imp
			lib_DD_likelihood = imp.load_source("lib_DD_likelihood", "%s/pyrate_lib/lib_DD_likelihood.py" % (self_path))
			lib_utilities = imp.load_source("lib_utilities", "%s/pyrate_lib/lib_utilities.py" % (self_path))
			rtt_plot_bds = imp.load_source("rtt_plot_bds", "%s/pyrate_lib/rtt_plot_bds.py" % (self_path))
			rtt_plot_bds.RTTplot_high_res(path_dir_log_files,args.grid_plot3,int(burnin),root_plot)		
		#except: sys.exit("""\nWarning: library pyrate_lib not found.\nMake sure PyRate.py and pyrate_lib are in the same directory.
		#You can download pyrate_lib here: <https://github.com/dsilvestro/PyRate> \n""")
	else:
		#path_dir_log_files=sort(path_dir_log_files)
		# plot each file separately
		print root_plot 
		if file_stem == "":
			direct="%s/*marginal_rates.log" % path_dir_log_files
			files=glob.glob(direct)
			files=sort(files)		
			if len(files)==0:
				if 2>1: #try:
					name_file = os.path.splitext(os.path.basename(str(path_dir_log_files)))[0]
					path_dir_log_files = os.path.dirname(str(path_dir_log_files))
					name_file = name_file.split("marginal_rates")[0]
					one_file=True
					plot_RTT(path_dir_log_files, burnin, name_file,one_file,root_plot,plot_type)
				#except: sys.exit("\nFile or directory not recognized.\n")
			else:
				for f in files:
					name_file = os.path.splitext(os.path.basename(f))[0]
					name_file = name_file.split("marginal_rates")[0]
					one_file =False
					plot_RTT(path_dir_log_files, burnin, name_file,one_file,root_plot,plot_type)
		else:
			one_file =False
			plot_RTT(path_dir_log_files, burnin, file_stem,one_file,root_plot,plot_type)
	quit()
elif args.mProb != "": calc_model_probabilities(args.mProb,burnin)
elif len(list_files_BF):
	if len(list_files_BF)<2: sys.exit("\n2 '*marginal_likelihood.txt' files required.\n")
	calc_BF(list_files_BF[0],list_files_BF[1])
	quit()
elif args.combLog != "": # COMBINE LOG FILES
	comb_log_files(args.combLog,burnin,args.tag,resample=args.resample,col_tag=args.col_tag)
	sys.exit("\n")
elif len(args.input_data)==0 and args.d == "": sys.exit("\nInput file required. Use '-h' for command list.\n")

use_se_tbl = False
if args.d != "":
	use_se_tbl = True
	se_tbl_file  = args.d

if len(args.SE_stats)>0:
	if use_se_tbl is False: sys.exit("\nProvide an SE table using command -d\n")
	if len(args.SE_stats)<1: sys.exit("\nAt least 1 argument required: Extinction rate at the present or *marginal_rates.log file\n")
	else:
		EXT_RATE  = args.SE_stats[0]
		if len(args.SE_stats)>1: step_size = args.SE_stats[1]
		else: step_size = 5
		if len(args.SE_stats)>2: no_sim_ex_time = args.SE_stats[2]
		else: no_sim_ex_time = 100
		plot_tste_stats(se_tbl_file, EXT_RATE, step_size,no_sim_ex_time,burnin,args.rescale)
		quit()



if args.restore_mcmc != "":
	restore_init_values = get_init_values(args.restore_mcmc)
	restore_chain = True
else: restore_chain = False

############################ LOAD INPUT DATA ############################
match_taxa_trait = False
if use_se_tbl==False:
	import imp
	input_file_raw = os.path.basename(args.input_data[0])
	input_file = os.path.splitext(input_file_raw)[0]  # file name without extension

	if args.wd=="": 
		output_wd = os.path.dirname(args.input_data[0])
		if output_wd=="": output_wd= self_path
	else: output_wd=args.wd

	#print "\n",input_file, args.input_data, "\n"
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
			

		elif args.singleton == -1: # exclude extant taxa
			if min(fossil_complete[i])==0: singletons_excluded.append(i)
			else:
				have_record.append(i) 
				fossil.append(fossil_complete[i]*args.rescale+args.translate)
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
	if len(singletons_excluded)>0 and args.data_info is False: print "The analysis includes %s species (%s were excluded)" % (len(fossil),len(singletons_excluded))	
	else: print "\nThe analysis includes %s species (%s were excluded)" % (len(fossil),len(fossil_complete)-len(fossil)) 
	out_name=input_data_module.get_out_name(j) +args.out

	try: 
		taxa_names=input_data_module.get_taxa_names()
		match_taxa_trait = True
	except(AttributeError): 
		taxa_names=list()
		for i in range(len(fossil)): taxa_names.append("taxon_%s" % (i))
	
	#print singletons_excluded
	taxa_included = np.array(taxa_included)
	taxa_names = np.array(taxa_names)
	taxa_names = taxa_names[taxa_included]

	FA,LO,N=np.zeros(len(fossil)),np.zeros(len(fossil)),np.zeros(len(fossil))
	for i in range(len(fossil)):
		FA[i]=max(fossil[i])
		LO[i]=min(fossil[i])
		N[i]=len(fossil[i])

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
	fix_SE=True
	fixed_ts, fixed_te=FA, LO	
	output_wd = os.path.dirname(se_tbl_file)
	if output_wd=="": output_wd= self_path
	out_name="%s_%s_%s"  % (os.path.splitext(os.path.basename(se_tbl_file))[0],j,args.out)
	if focus_clade>=0: out_name+= "_c%s" % (focus_clade)
	
	
	
	

if argsG is True: out_name += "_G"

############################ SET BIRTH-DEATH MODEL ############################

# Number of extant taxa (user specified)
if args.N>-1: tot_extant=args.N
else: tot_extant = -1	


if len(fixed_times_of_shift)>0: 
	fixed_times_of_shift=fixed_times_of_shift[fixed_times_of_shift<max(FA)]
	time_framesL=len(fixed_times_of_shift)+1
	time_framesM=len(fixed_times_of_shift)+1
	# estimate DPP hyperprior
	hp_gamma_rate  = get_rate_HP(time_framesL,target_k,hp_gamma_shape)


if args.fixSE != "" or use_se_tbl==True:          # fix TS, TE
	if use_se_tbl==True: pass
	else:
		fix_SE=True
		fixed_ts, fixed_te= calc_ts_te(args.fixSE, burnin=args.b)
else: fix_SE=False

if args.discrete is True: useDiscreteTraitModel = True
else: useDiscreteTraitModel = False

useBounded_BD = False
if args.bound[0] != np.inf or args.bound[1] != 0: 
	useBounded_BD = True
boundMax = max(args.bound) # if not specified it is set to Inf
boundMin = min(args.bound) # if not specified it is set to 0


# Get trait values (COVAR and DISCRETE models)
if model_cov>=1 or useDiscreteTraitModel is True or useBounded_BD is True:
	if 2>1: #try:
		if args.trait_file != "": # Use trait file
			traitfile=file(args.trait_file, 'U')
			
			L=traitfile.readlines()
			head= L[0].split()
			
			if useBounded_BD is True: # columns: taxon_name, SP, EX (SP==1 if speciate in window)
				trait_val=[l.split()[1:3] for l in L][1:]
			else:
				if len(head)==2: col=1
				elif len(head)==3: col=2 ####  CHECK HERE: WITH BOUNDED-BD 3 columns but two traits!
				else: sys.exit("\nNo trait data found!")
				trait_val=[l.split()[col] for l in L][1:]
			
			if useBounded_BD is True:
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
			
			if match_taxa_trait is True:
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
						if useBounded_BD is True: # taxa not specified originate/go_extinct in window 
							matched_trait_values.append([1,1])
						else: 
							matched_trait_values.append(np.nan)
						#print taxa_name, "did not have data"
				trait_values= np.array(matched_trait_values)
				#print trait_values
			
		else:             # Trait data from .py file
			trait_values=input_data_module.get_continuous(max(args.trait-1,0))
		if model_cov>=1:
			if args.logT==0: pass
			elif args.logT==1: trait_values = log(trait_values)
			else: trait_values = np.log10(trait_values)		
	#except: sys.exit("\nTrait data not found! Check input file.\n")
	
	if model_cov>=1:
		# Mid point age of each lineage
		if use_se_tbl==True: MidPoints = (fixed_ts+fixed_te)/2.
		else:
			MidPoints=np.zeros(len(fossil_complete))
			for i in range(len(fossil_complete)):
				MidPoints[i]=np.mean([max(fossil_complete[i]),min(fossil_complete[i])])
	
		# fit linear regression (for species with trait value - even if without fossil data)
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(MidPoints[np.isfinite(trait_values)],trait_values[np.isfinite(trait_values)])
	
		# 
		ind_nan_trait= (np.isfinite(trait_values)==False).nonzero()
		meanGAUScomplete=np.zeros(len(MidPoints))
		meanGAUScomplete[ind_nan_trait] = slope*MidPoints[ind_nan_trait] + intercept
	
		if use_se_tbl==True: meanGAUS= meanGAUScomplete
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
		if est_COVAR_prior is True: out_name += "_COVhp"
		else: out_name += "_COV"
	elif useDiscreteTraitModel is True: 
		con_trait = trait_values
		ind_nan_trait= (np.isfinite(trait_values)==False).nonzero()
		regression_trait= "\n\nDiscrete trait data for %s of %s taxa" \
		% (len(trait_values)-len(ind_nan_trait[0]), len(trait_values))
		print(regression_trait)
	else: print "\n"
		

if useDiscreteTraitModel is True:
	ind_trait_species = con_trait
	ind_trait_species[np.isnan(ind_trait_species)]=np.nanmax(ind_trait_species)+1
	ind_trait_species = ind_trait_species.astype(int)
	len_trait_values = len(np.unique(con_trait)) 
	lengths_B_events=[]
	lengths_D_events=[]
	for i in range(len_trait_values):
		lengths_B_events.append(len(con_trait[con_trait==i]))
		lo_temp = LO[con_trait==i]
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
if use_poiD is True:
	BPD_partial_lik = PoiD_partial_lik
	PoiD_const = - (sum(log(np.arange(1,len(FA)+1))))
elif useBounded_BD is True:
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
	argsHPP = True # only HPP can be used with bounded BD
else:
	BPD_partial_lik = BD_partial_lik
	PoiD_const = 0
	SP_in_window = np.arange(len(FA)) # all ts/te can be updated
	EX_in_window = np.arange(len(LO))
	SP_not_in_window = []
	EX_not_in_window = []

if args.mDeath is True: use_Death_model =True
else: use_Death_model=False


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
use_ADE_model = False
if args.ADE == 1:
	use_ADE_model = True
	BPD_partial_lik = BD_age_partial_lik
	out_name += "_ADE"
	argsHPP = True
	

est_hyperP = False
use_cauchy = False
fix_hyperP = False
if sum(args.cauchy) >= 0:
	hypP_par = np.ones(2)
	use_cauchy = True
	est_hyperP = True
	if sum(args.cauchy) > 0:
		fix_hyperP = True
		hypP_par = np.array(args.cauchy) # scale of Cauchy distribution
else:
	hypP_par = np.array([L_lam_m,M_lam_m]) # rate of Gamma distribution
	if min([L_lam_m,M_lam_m])==0:
		est_hyperP = True
		hypP_par = np.ones(2)

if use_ADE_model is True: 
	hypP_par[1]=0.1
	tot_extant = -1
	d_hyperprior[0]=1 # first hyper-prior on sp.rates is not used under ADE, thus not updated (multiplier update =1)

if args.qShift != "":
	try: 
		try: times_q_shift=np.sort(np.loadtxt(args.qShift))[::-1]
		except(IndexError): times_q_shift=np.array([np.loadtxt(args.qShift)])		
		times_q_shift=list(times_q_shift[times_q_shift<max(FA)])
		time_framesQ=len(times_q_shift)+1
		occs_sp_bin =list()
		temp_times_q_shift = np.array(list(times_q_shift)+[max(FA)+1]+[0])
		for i in range(len(fossil)):
			occs_temp = fossil[i]
			h = np.histogram(occs_temp[occs_temp>0],bins=sort( temp_times_q_shift ))[0][::-1]
			occs_sp_bin.append(h)
		argsHPP = True
		multiHPP = True
	except: 
		msg = "\nError in the input file %s.\n" % (args.qShift)
		sys.exit(msg)
else: multiHPP = False 


if fix_Shift is True and use_ADE_model is False: est_hyperP = True
# define hyper-prior function for BD rates
if tot_extant==-1 or TDI ==3 or use_poiD is True:
	if use_ADE_model is False and fix_Shift is True and TDI < 3 or use_cauchy is True: 
		if est_hyperP is False or fix_hyperP is True:
			prior_setting= "Using Cauchy priors on the birth-death rates (C_l[0,%s],C_l[0,%s]).\n" % (hypP_par[0],hypP_par[1])
		else:
				prior_setting= "Using Cauchy priors on the birth-death rates (C_l[0,est],C_l[0,est]).\n" 
 		get_hyper_priorBD = HPBD1 # cauchy with hyper-priors
	else: 
		if est_hyperP is False:
			prior_setting= "Using Gamma priors on the birth-death rates (G_l[%s,%s], G_m[%s,%s]).\n" % (L_lam_r,hypP_par[0],M_lam_r,hypP_par[1])
		else:
			prior_setting= "Using Gamma priors on the birth-death rates (G_l[%s,est], G_m[%s,est]).\n" % (L_lam_r,M_lam_r)
		get_hyper_priorBD = HPBD2 # gamma
else: 
	prior_setting= "Priors on the birth-death rates based on extant diversity (N = %s).\n" % (tot_extant)
	get_hyper_priorBD = HPBD3 # based on no. extant
print prior_setting

if use_poiD is True:
	if model_cov>=1: 
		print "PoiD not available with trait correlation. Using BD instead."
		BPD_partial_lik = BD_partial_lik
		PoiD_const = 0
	if fix_SE==False: 
		print "PoiD not available with SE estimation. Using BD instead."
		BPD_partial_lik = BD_partial_lik
		PoiD_const = 0



# GET DATA SUMMARY INFO
if args.data_info is True:
	print "\nDATA SUMMARY\n"
	if len(singletons_excluded)>0: print "%s species excluded as singletons (observed life span < %s Myr)" % (len(singletons_excluded), args.singleton)
	print "%s species included in the data set" % (len(fossil))
	one_occ_sp,all_occ,extant_sp,n_occs_list  = 0,0,0,list()
	for i in fossil:
		if len(i[i>0])==1: one_occ_sp+=1
		all_occ += len(i[i>0])
		n_occs_list.append(len(i[i>0]))
		if min(i)==0: extant_sp+=1
	print "%s species have a single occurrence, %s species are extant" % (one_occ_sp,extant_sp)
	j=0
	m_ages,M_ages=[],[]
	while True:
		try: fossil_complete=input_data_module.get_data(j)
		except(IndexError): break
		min_age, max_age = np.inf, 0
		for i in fossil_complete:
			a,b = min(i[i>0]), max(i)
			if a < min_age: min_age=a
			if b > max_age: max_age=b
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
	

############################ MCMC OUTPUT ############################
try: os.mkdir(output_wd)
except(OSError): pass
path_dir = "%s/pyrate_mcmc_logs" % (output_wd)
folder_name="pyrate_mcmc_logs"
try: os.mkdir(path_dir) 
except(OSError): pass

suff_out=out_name
if TDI<=1: suff_out+= "BD%s-%s" % (args.mL,args.mM)
if TDI==1: suff_out+= "_TI"
if TDI==3: suff_out+= "_dpp"
if TDI==4: suff_out+= "_rj"

# OUTPUT 0 SUMMARY AND SETTINGS
o0 = "\n%s build %s\n" % (version, build)
o1 = "\ninput: %s output: %s/%s" % (args.input_data, path_dir, out_name)
o2 = "\n\nPyRate was called as follows:\n%s\n" % (args)
if model_cov>=1 or useDiscreteTraitModel is True: o2 += regression_trait
if TDI==3: o2 += "\n\nHyper-prior on concentration parameter (Gamma shape, rate): %s, %s\n" % (hp_gamma_shape, hp_gamma_rate)
if len(fixed_times_of_shift)>0:
	o2 += "\nUsing the following fixed time frames: "
	for i in fixed_times_of_shift: o2 += "%s " % (i)
o2+= "\n"+prior_setting
if argsHPP is True: 
	if multiHPP is False: 
		o2+="Using Homogeneous Poisson Process of preservation (HPP)."
	else: 
		o2 += "\nUsing Homogeneous Poisson Process of preservation with shifts (HPPS) at: "
		for i in times_q_shift: o2 += "%s " % (i)
	
else: o2+="Using Non-Homogeneous Poisson Process of preservation (NHPP)."

version_notes="""\n
Please cite: \n%s\n
Feedback and support: pyrate.help@gmail.com
OS: %s %s
Python version: %s\n
Numpy version: %s
Scipy version: %s
""" % (citation,platform.system(), platform.release(), sys.version, np.version.version, scipy.version.version)

o=''.join([o0,o1,o2,version_notes])
out_sum = "%s/%s_sum.txt" % (path_dir,suff_out)
sumfile = open(out_sum , "wb",0) 
sumfile.writelines(o)
sumfile.close()

# OUTPUT 1 LOG MCMC
out_log = "%s/%s_mcmc.log" % (path_dir, suff_out) #(path_dir, output_file, out_run)
logfile = open(out_log , "wb",0) 
if fix_SE == False:
	if multiHPP is False:
		head="it\tposterior\tprior\tPP_lik\tBD_lik\tq_rate\talpha\t"
	else: 
		head="it\tposterior\tprior\tPP_lik\tBD_lik\t"
		for i in range(time_framesQ): head += "q_%s\t" % (i)
else: 
	head="it\tposterior\tprior\tBD_lik\t"
	
if model_cov>=1: 
	head += "cov_sp\tcov_ex\tcov_q\t"
	if est_COVAR_prior is True: head+="cov_hp\t"
if TDI<2:
	head += "root_age\tdeath_age\t"
	if TDI==1: head += "beta\t"
	if est_hyperP is True: 
		head += "hypL\thypM\t"
	
	if use_ADE_model is False and useDiscreteTraitModel is False: 
		for i in range(time_framesL): head += "lambda_%s\t" % (i)
		for i in range(time_framesM): head += "mu_%s\t" % (i)
	elif use_ADE_model is True: 
		head+="w_shape\t"
		for i in range(time_framesM): head += "w_scale_%s\t" % (i)
		for i in range(time_framesM): head += "mean_longevity_%s\t" % (i)
	elif useDiscreteTraitModel is True:
		for i in range(len(lengths_B_events)): head += "lambda_%s\t" % (i)
		for i in range(len(lengths_D_events)): head += "mu_%s\t" % (i)		
	
	if fix_Shift== False:
		for i in range(1,time_framesL): head += "shift_sp_%s\t" % (i)
		for i in range(1,time_framesM): head += "shift_ex_%s\t" % (i)
elif TDI in [2,4]: head+="k_birth\tk_death\troot_age\tdeath_age\t"
else:        head+="k_birth\tk_death\tDPP_alpha_L\tDPP_alpha_M\troot_age\tdeath_age\t"


if useDiscreteTraitModel is True: 
	for i in range(len(lengths_B_events)): head += "S_%s\t" % (i)
head += "tot_length"
head=head.split('\t')

if fix_SE == False:
	for i in taxa_names: head.append("%s_TS" % (i))
	for i in taxa_names: head.append("%s_TE" % (i))
wlog=csv.writer(logfile, delimiter='\t')
wlog.writerow(head)

#logfile.writelines(head)
logfile.flush()
os.fsync(logfile)

# OUTPUT 2 MARGINAL RATES
if TDI!=1 and use_ADE_model is False and useDiscreteTraitModel is False: # (path_dir, output_file, out_run)
	out_log_marginal = "%s/%s_marginal_rates.log" % (path_dir, suff_out) 
	marginal_file = open(out_log_marginal , "wb") 
	head="it\t"
	if useBounded_BD is True: max_marginal_frame = boundMax+1
	else: max_marginal_frame = max(FA)
	for i in range(int(max_marginal_frame)+1): head += "l_%s\t" % i #int(fabs(int(max(FA))))
	for i in range(int(max_marginal_frame)+1): head += "m_%s\t" % i #int(fabs(int(max(FA))))
	for i in range(int(max_marginal_frame)+1): head += "r_%s\t" % i #int(fabs(int(max(FA))))
	head=head.split('\t')
	wmarg=csv.writer(marginal_file, delimiter='	')
	wmarg.writerow(head)
	marginal_file.flush()
	os.fsync(marginal_file)
	marginal_frames= array([int(fabs(i-int(max_marginal_frame))) for i in range(int(max_marginal_frame)+1)])

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
if fix_SE is True and fix_Shift is True:
	time_frames  = sort(np.array(list(fixed_times_of_shift) + [0,max(fixed_ts)]))
	B = sort(time_frames)+0.000001 # add small number to avoid counting extant species as extinct
	ss1 = np.histogram(fixed_ts,bins=B)[0][::-1]
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
if use_seq_lik is False and runs>1:
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
marginal_file.close()

#cmd="cd %s && cd .. && tar -czf %s.tar.gz %s;" % (path_dir, folder_name, folder_name)
#print cmd
#os.system(cmd)
quit()