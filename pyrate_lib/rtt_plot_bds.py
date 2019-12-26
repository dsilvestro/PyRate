from numpy import *
import numpy as np
import os,platform,glob,sys
import lib_utilities as util
import csv 

def get_marginal_rates_plot3(times,rates,grid):
	mr = np.zeros(len(grid))
	for i in range(len(times)-1):
		grid_ind = np.intersect1d((grid<=times[i]).nonzero()[0],(grid>=times[i+1]).nonzero()[0])
		mr[grid_ind.astype(int)] = rates[i]
	return mr
		

def r_plot_code(res,wd,name_file,alpha=0.5,plot_title="RTT plot"):
	data  = "library(scales)\ntrans=%s" % (alpha)
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		wd_forward = os.path.abspath(wd).replace('\\', '/')
		data+= "\n\npdf(file='%s/%s_RTT.pdf',width=0.6*9, height=0.6*14)\npar(mfrow=c(2,1))" % (wd_forward,name_file) # 9
	else: 
		data+= "\n\npdf(file='%s/%s_RTT.pdf',width=0.6*9, height=0.6*14)\npar(mfrow=c(2,1))" % (wd,name_file) # 9
	data += util.print_R_vec('\nage',   -res[:,0])
	data += util.print_R_vec('\nL_mean', res[:,1])
	data += util.print_R_vec('\nL_hpd_m',res[:,2])
	data += util.print_R_vec('\nL_hpd_M',res[:,3])
	data += util.print_R_vec('\nM_mean', res[:,4])
	data += util.print_R_vec('\nM_hpd_m',res[:,5])
	data += util.print_R_vec('\nM_hpd_M',res[:,6])
	max_x_axis,min_x_axis = -max(res[:,0]),-min(res[:,0])
	data += "\nplot(age,age,type = 'n', ylim = c(0, %s), xlim = c(%s,%s), ylab = 'Speciation rate', xlab = 'Ma',main='%s' )" \
		% (1.1*max(res[:,3]),max_x_axis,min_x_axis,plot_title) 
	data += """\nlines(rev(age), rev(L_mean), col = "#4c4cec", lwd=3)""" 
	data += """\npolygon(c(age, rev(age)), c(L_hpd_M, rev(L_hpd_m)), col = alpha("#4c4cec",trans), border = NA)""" 
	data += "\nplot(age,age,type = 'n', ylim = c(0, %s), xlim = c(%s,%s), ylab = 'Extinction rate', xlab = 'Ma' )" \
		% (1.1*max(res[:,6]),max_x_axis,min_x_axis)	
	data += """\nlines(rev(age), rev(M_mean), col = "#e34a33", lwd=3)""" 
	data += """\npolygon(c(age, rev(age)), c(M_hpd_M, rev(M_hpd_m)), col = alpha("#e34a33",trans), border = NA)"""
	data += "\nn <- dev.off()" 
	return data


def RTTplot_high_res(f,grid_cell_size=1.,burnin=0,max_age=0):
	wd = "%s" % os.path.dirname(f)
	name_file=os.path.splitext(os.path.basename(f))[0]
	t=loadtxt(f, skiprows=max(1,int(burnin)))
	head = np.array(next(open(f)).split()) # should be faster
	#print np.where(head=="beta")[0], np.where(head=="temperature")[0]
	if "temperature" in head or "beta" in head:
		if "temperature" in head: 
			temp_index = np.where(head=="temperature")[0][0]
		else: 
			temp_index = np.where(head=="beta")[0][0]
		temp_values = t[:,temp_index]
		t = t[temp_values==1,:]
		print "removed heated chains:",np.shape(t)
	
	head= list(head)
	sp_ind = [head.index(s) for s in head if "lambda_" in s]
	ex_ind = [head.index(s) for s in head if "mu_" in s]
	root_ind  = head.index("root_age")
	death_ind = head.index("death_age")
	sp_shift_ind = [head.index(s) for s in head if "shift_sp_" in s]
	ex_shift_ind = [head.index(s) for s in head if "shift_ex_" in s]
	min_root_age = min(t[:,root_ind])
	if max_age> 0: min_root_age=max_age
	max_death_age= max(t[:,death_ind])
	n_bins= int((min_root_age-max_death_age)/grid_cell_size)
	grid = np.linspace(min_root_age,max_death_age,n_bins)
	n_samples = np.shape(t)[0]
	m_sp_matrix = np.zeros((n_samples,n_bins))
	m_ex_matrix = np.zeros((n_samples,n_bins))
	print "Extracting marginal rates..."
	for i in range(n_samples):
		l_shift_times = np.array([min_root_age]+ list(t[i,sp_shift_ind])+[max_death_age])
		l_rates = t[i,sp_ind]
		m_sp_matrix[i] = get_marginal_rates_plot3(l_shift_times,l_rates,grid)
		m_shift_times = np.array([min_root_age]+ list(t[i,ex_shift_ind])+[max_death_age])
		m_rates = t[i,ex_ind]
		m_ex_matrix[i] = get_marginal_rates_plot3(m_shift_times,m_rates,grid)

	res = np.zeros((n_bins,7)) # times, l, lM, lm, m, mM, mm
	print "Calculating HPDs..."
	for i in range(n_bins):
		l_HPD  = list(util.calcHPD(m_sp_matrix[:,i],0.95))
		l_mean = mean(m_sp_matrix[:,i])
		m_HPD  = list(util.calcHPD(m_ex_matrix[:,i],0.95))
		m_mean = mean(m_ex_matrix[:,i])
		res[i] = np.array([grid[i],l_mean]+l_HPD+[m_mean]+m_HPD)
	
	Rfile=r_plot_code(res,wd,name_file)
	out="%s/%s_RTT.r" % (wd,name_file)
	newfile = open(out, "w") 
	newfile.writelines(Rfile)
	newfile.close()
	print "\nAn R script with the source for the RTT plot was saved as: %sRTT.r\n(in %s)" % (name_file, wd)
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		cmd="cd %s & Rscript %s_RTT.r" % (wd,name_file)
	else: 
		cmd="cd %s; Rscript %s/%s_RTT.r" % (wd,wd,name_file)
	os.system(cmd)
	
	save_HR_log_file=1
	if save_HR_log_file==1:
		print "Saving log file..."
		out="%s/%s_HR_marginal.log" % (wd,name_file)
		logfile = open(out, "w") 
		head = ["iteration"]+["l_%s" % (i) for i in range(n_bins)]
		head +=["m_%s" % (i) for i in range(n_bins)]
		wlog=csv.writer(logfile, delimiter='\t')
		wlog.writerow(head)
		for i in range(n_samples):
			l = [i]+ list(m_sp_matrix[i,:]) + list(m_ex_matrix[i,:])
			wlog.writerow(l)
			logfile.flush()
			
	
	
	print "done\n"


# args
#burnin=0
#grid_cell_size=.1 # in Myr (approximated)
#f= "/Users/daniele/Desktop/try/catalina_temp/par_est/combined_10mcmc_files.log"
#max_age=23.03
#RTTplot_high_res(f,grid_cell_size,burnin,max_age)

def RTTplot_Q(f,q_shift_file,burnin=0,max_age=0):
	wd = "%s" % os.path.dirname(f)
	name_file=os.path.splitext(os.path.basename(f))[0]
	t=loadtxt(f, skiprows=max(1,int(burnin)))
	head = np.array(next(open(f)).split()) # should be faster
	#print np.where(head=="beta")[0], np.where(head=="temperature")[0]
	if "temperature" in head or "beta" in head:
		if "temperature" in head: 
			temp_index = np.where(head=="temperature")[0][0]
		else: 
			temp_index = np.where(head=="beta")[0][0]
		temp_values = t[:,temp_index]
		t = t[temp_values==1,:]
		print "removed heated chains:",np.shape(t)
	
	head= list(head)
	q_ind = [head.index(s) for s in head if "q_" in s]
	root_ind  = head.index("root_age")
	death_ind = head.index("death_age")
	min_root_age = min(t[:,root_ind])
	if max_age> 0: min_root_age=max_age
	max_death_age= max(t[:,death_ind])

	try: times_q_shift=np.sort(np.loadtxt(q_shift_file))[::-1]
	except: times_q_shift=np.array([np.loadtxt(q_shift_file)])
	
	times_q_shift = times_q_shift[times_q_shift>max_death_age]
	times_q_shift = times_q_shift[times_q_shift<min_root_age]
	times_q_shift = np.sort(np.array(list(times_q_shift) + [max_death_age,min_root_age]))[::-1]
	print times_q_shift
	
	means = []
	hpdM  = []
	hpdm  = []
	data  = "library(scales)\n" 
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		wd_forward = os.path.abspath(wd).replace('\\', '/')
		data+= "\n\npdf(file='%s/%s_RTT_Qrates.pdf',width=0.6*9, height=0.6*7)\n" % (wd,name_file) # 9
	else: 
		data+= "\n\npdf(file='%s/%s_RTT_Qrates.pdf',width=0.6*9, height=0.6*7)\n" % (wd,name_file) # 9
	
	
	max_y_axis,max_x_axis,min_x_axis = np.max(t[:,q_ind]),-np.max(times_q_shift),-np.min(times_q_shift)
	
	for i in range(len(q_ind)):
		qtemp = t[:,q_ind[i]]
		hpdtemp = util.calcHPD(qtemp,0.95)
		#means.append(np.mean(qtemp),)
		#hpdM.append(hpdtemp[1],hpdtemp[1])
	      #hpdm.append(hpdtemp[0],hpdtemp[0])
		time_slice = np.array([times_q_shift[i],times_q_shift[i+1]])
		data += '\nage = c(%s, %s)' % (-time_slice[0],-time_slice[1])
		data += '\nQ_mean = %s' %  np.mean(qtemp)
		data += '\nQ_hpd_m = %s' % hpdtemp[0]
		data += '\nQ_hpd_M = %s' % hpdtemp[1]
		if i==0:
			data += "\nplot(age,age,type = 'n', ylim = c(0, %s), xlim = c(%s,%s), ylab = 'Preservation rate', xlab = 'Ma',main='%s' )" \
				% (max_y_axis,max_x_axis,min_x_axis,"Preservation rates") 			
		else:
			data += """\nsegments(x0=age[1], y0 = %s, x1 = age[1], y1 = Q_mean, col = "#756bb1", lwd=3)""" % (Q_mean_previous)
		Q_mean_previous = np.mean(qtemp)
		data += """\nsegments(x0=age[1], y0 = Q_mean, x1 = age[2], y1 = Q_mean, col = "#756bb1", lwd=3)""" 
		data += """\npolygon( c(age, rev(age)), c(Q_hpd_m, Q_hpd_m, Q_hpd_M, Q_hpd_M), col = alpha("#756bb1",0.5), border = NA)""" 
	data += "\nn <- dev.off()" 
		
	out="%s/%s_RTT_Qrates.r" % (wd,name_file)
	newfile = open(out, "w") 
	newfile.writelines(data)
	newfile.close()
	print "\nAn R script with the source for the RTT plot was saved as: %s_RTT_Qrates.r\n(in %s)" % (name_file, wd)
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		cmd="cd %s & Rscript %s_RTT_Qrates.r" % (wd,name_file)
	else: 
		cmd="cd %s; Rscript %s/%s_RTT_Qrates.r" % (wd,wd,name_file)
	os.system(cmd)
	
	print "done\n"



# functionsto plot RTT when '-log_marginal_rates 0'
def calcBF(threshold,empirical_prior):
	A = exp(threshold/2)*empirical_prior/(1-empirical_prior)
	return A/(A+1)


def get_prior_shift(t_start,t_end,bins_histogram):
	times_of_shift = []
	sampled_K = []
	# Gamma hyper-prior
	G_shape = 2. # currently hard-coded
	G_rate = 1.  # mode at 1
	min_time_frame_size = 1
	iteration=0.
	print "Computing empirical priors on rate shifts..."
	for rep in range(100000):
		if rep % 10000 ==0:
			sys.stdout.write(".")
			sys.stdout.flush()	
		try:
			# Poisson prior
			Poi_lambda = np.random.gamma(G_shape,1./G_rate)
			n_rates_temp = np.random.poisson(Poi_lambda,1000)
			n_rates = n_rates_temp[n_rates_temp>0][0]
			shift_times = list(np.random.uniform(t_end-min_time_frame_size,t_start+min_time_frame_size,n_rates-1))
			time_frames = np.sort([t_start, t_end]+shift_times)	
			if min(np.diff(time_frames))<min_time_frame_size:
				pass
			else:
				iteration+=1
				times_of_shift += shift_times
				sampled_K.append(n_rates)
		except(IndexError): pass
	expectedK = np.array(sampled_K)
	prior_s = np.mean(np.histogram(times_of_shift,bins=bins_histogram)[0]/iteration)
	bf2 = calcBF(2,prior_s)
	bf6 = calcBF(6,prior_s)
	print np.array([prior_s,bf2,bf6])
	return [prior_s,bf2,bf6]




def get_marginal_rates(f_name,min_age,max_age,nbins=0,burnin=0.2):
	# returns a list of 5 items:
	# 1. a vector of times (age of each marginal rate)
	# 2-4. mean, min and max marginal rates (95% HPD)
	# 5. a vector of times of rate shift
	f = file(f_name,'U')
	if nbins==0:
		nbins = int(max_age-0)
	post_rate=f.readlines()
	bins_histogram = np.linspace(min_age,max_age,nbins+1)	
	marginal_rates_list = []
	times_of_shift = []
	
	if burnin<1: # define burnin as a fraction
		burnin=min(int(burnin*len(post_rate)),int(0.9*len(post_rate)))
	else: burnin = int(burnin)
	
	for i in range(burnin,len(post_rate)):
		row = np.array(post_rate[i].split()).astype(float)
		
		if len(row)==0:
			continue	
		elif len(row)==1: 
			marginal_rates = np.zeros(nbins)+row[0]
		else:
			ind_rates = np.arange(0,int(np.ceil(len(row)/2.)))
			ind_shifts = np.arange(int(np.ceil(len(row)/2.)),len(row))
			rates = row[ind_rates]
			shifts = row[ind_shifts]
			#shifts = shifts[shifts>min_age]
			h = np.histogram(row[ind_shifts],bins =bins_histogram)[0][::-1]
			marginal_rates = rates[np.cumsum(h)]
			times_of_shift += list(shifts)
		
		marginal_rates_list.append(marginal_rates)
	
	marginal_rates_list = np.array(marginal_rates_list)
	mean_rates= np.median(marginal_rates_list,axis=0)
	min_rates,max_rates=[],[]
	for i in range(nbins):
		hpd = util.calcHPD(marginal_rates_list[:,i],0.95)
		min_rates += [hpd[0]]
		max_rates += [hpd[1]]
	
	time_frames = bins_histogram-abs(bins_histogram[1]-bins_histogram[0])/2.
	#print time_frames, min(times_of_shift), min_age
	#quit()
	time_frames = time_frames[1:]
	#print len(time_frames),len(mean_rates), 
	n_mcmc_samples = len(post_rate)-burnin # number of samples used to normalize frequencies of rate shifts
	return [time_frames,mean_rates,np.array(min_rates),np.array(max_rates),np.array(times_of_shift),n_mcmc_samples, marginal_rates_list]

def get_r_plot(res,col,parameter,min_age,max_age,plot_title,plot_log,run_simulation=1, plot_shifts=1, line_wd=2):
	
	times = res[0]
	rates = res[1][::-1]
	rates_m = res[2][::-1]
	rates_M = res[3][::-1]
	shifts = res[4]
	
	indx = np.intersect1d((times>=min_age).nonzero()[0], (times<=max_age).nonzero()[0])
	times = times[indx]
	#print indx,times, rates
	rates   = rates[indx]
	rates_m = rates_m[indx]
	rates_M = rates_M[indx]
	try:
		shifts= shifts[shifts>min_age]
		shifts= shifts[shifts<max_age]
	except: plot_shifts=0
	
	out_str = "\n"
	out_str += util.print_R_vec("\ntime",-times)
	out_str += util.print_R_vec("\nrate",rates)
	out_str += util.print_R_vec("\nminHPD",rates_m)
	out_str += util.print_R_vec("\nmaxHPD",rates_M)
	if plot_log==0:
		out_str += "\nplot(time,time,type = 'n', ylim = c(%s, %s), xlim = c(%s,%s), ylab = '%s', xlab = 'Time',main='%s' )" \
			% (0,1.1*np.nanmax(rates_M),-max_age,-min_age,parameter,plot_title) 
		out_str += "\npolygon(c(time, rev(time)), c(maxHPD, rev(minHPD)), col = alpha('%s',0.3), border = NA)" % (col)
		out_str += "\nlines(time,rate, col = '%s', lwd=%s)" % (col, line_wd)
	else:
		out_str += "\nplot(time,time,type = 'n', ylim = c(%s, %s), xlim = c(%s,%s), ylab = 'Log10 %s', xlab = 'Time',main='%s' )" \
			% (np.nanmin(np.log10(0.9*rates_m)),np.nanmax(np.log10(1.1*rates_M)),-max_age,-min_age,parameter,plot_title) 
		out_str += "\npolygon(c(time, rev(time)), c(log10(maxHPD), rev(log10(minHPD))), col = alpha('%s',0.3), border = NA)" % (col)
		out_str += "\nlines(time,log10(rate), col = '%s', lwd=%s)" % (col, line_wd)
		
	if plot_shifts:
		# add barplot rate shifts
		bins_histogram = np.linspace(0,max_age,len(res[0]))
		if len(shifts)>1: # rate shift sampled at least once
			h = np.histogram(shifts,bins =bins_histogram) #,density=1)
		else:
			h = [np.zeros(len(bins_histogram)-1),bins_histogram]
		a = h[1]
		mids = (a-a[1]/2.)[1:]
		out_str += util.print_R_vec("\nmids",-mids)
		out_str += util.print_R_vec("\ncounts",h[0]/float(res[5]))
		out_str += "\nplot(mids,counts,type = 'h', xlim = c(%s,%s), ylim=c(0,%s), ylab = 'Frequency of rate shift', xlab = 'Time',lwd=5,col='%s')" \
		    % (-max_age,-min_age,max(max(h[0]/float(res[5])),0.2),col)
		# get BFs
		if run_simulation==1:
			BFs = get_prior_shift(min_age,max_age,bins_histogram)
			out_str += "\nbf2 = %s\nbf6 = %s" % (BFs[1],BFs[2])
		out_str += "\nabline(h=bf2, lty=2)"
		out_str += "\nabline(h=bf6, lty=2)"
	return out_str


def plot_net_rate(resS,resE,col,min_age,max_age,plot_title,n_bins):
	#computes and plots net RATES
	resS_marginal_rate = resS[6]
	resE_marginal_rate = resE[6]
	# in case they have different number of samples
	max_indx = np.min([resS_marginal_rate.shape[0], resE_marginal_rate.shape[0]])
	marginal_rates_list	= resS_marginal_rate[0:max_indx,:] - resE_marginal_rate[0:max_indx,:]
	
	mean_rates= np.mean(marginal_rates_list,axis=0)
	min_rates,max_rates=[],[]
	time_ax = []
	for i in range(n_bins):
		hpd = util.calcHPD(marginal_rates_list[:,i],0.95)
		min_rates += [hpd[0]]
		max_rates += [hpd[1]]
	
	times = abs(resS[0])
	indx = np.intersect1d((times>=min_age).nonzero()[0], (times<=max_age).nonzero()[0])
	times = times[indx]

	out_str = "\n#Net Diversification Rate"
  	out_str += util.print_R_vec("\ntime",-times)
	minXaxis,maxXaxis= min_age,max_age
	
	mean_rates = np.array(mean_rates)[::-1]
	min_rates  = np.array(min_rates )[::-1]
	max_rates  = np.array(max_rates )[::-1]
	
	mean_rates = mean_rates[indx]
	min_rates  = min_rates[indx]
	max_rates  = max_rates[indx]
	
	out_str += util.print_R_vec("\nnet_rate",mean_rates)
	out_str += util.print_R_vec("\nnet_minHPD",np.array(min_rates))
	out_str += util.print_R_vec("\nnet_maxHPD",np.array(max_rates))
	out_str += "\nplot(time,time,type = 'n', ylim = c(%s, %s), xlim = c(%s,%s), ylab = 'Net Rate', xlab = 'Time',lwd=2, main='%s', col= '%s' )" \
			% (min(0,1.1*np.nanmin(min_rates)),1.1*np.nanmax(max_rates),-maxXaxis,-minXaxis,plot_title,col) 
	out_str += "\npolygon(c(time, rev(time)), c(net_maxHPD, rev(net_minHPD)), col = alpha('%s',0.3), border = NA)" % (col)
	out_str += "\nlines(time,net_rate, col = '%s', lwd=2)" % (col)
	out_str += "\nabline(h=0,lty=2)\n"
		
	return out_str


def plot_marginal_rates(path_dir,name_tag="",bin_size=0.,burnin=0.2,min_age=0,max_age=0,logT=0):
	direct="%s/*%s*mcmc.log" % (path_dir,name_tag)
	files=glob.glob(direct)
	files=np.sort(files)
	stem_file=files[0]
	wd = "%s" % os.path.dirname(stem_file)
	#print(name_file, wd)
	print "found", len(files), "log files...\n"
	if logT==1: outname = "Log_"
	else: outname = ""
	if max_age>0: outname+= "t%s" % (int(max_age))
	if min_age>0: outname+= "-%s" % (int(min_age))
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		wd_forward = os.path.abspath(wd).replace('\\', '/')
		r_str = "\n\npdf(file='%s/%sRTT_plots.pdf',width=10, height=15)\npar(mfrow=c(3,2))\nlibrary(scales)" % (wd_forward,outname)
	else:
		r_str = "\n\npdf(file='%s/%sRTT_plots.pdf',width=10, height=15)\npar(mfrow=c(3,2))\nlibrary(scales)" % (wd,outname)
	for mcmc_file in files:
		if 2>1: #try:
			name_file = os.path.splitext(os.path.basename(mcmc_file))[0]		
			tbl=np.loadtxt(mcmc_file, skiprows=1)
			head = next(open(mcmc_file)).split() 
			max_age_t = np.min(tbl[:,head.index("root_age")])
			min_age_t = np.max(tbl[:,head.index("death_age")])
			print "\nAge range:",max_age_t, min_age_t
			if max_age==0: max_age=max_age_t
			print bin_size 
			if bin_size>0:
				nbins = int((max_age_t-min_age_t)/float(bin_size))
			else:
				nbins = 100
				bin_size = (min(max_age,max_age_t)-max(min_age_t,min_age))/100.
			print bin_size, nbins
			colors = ["#4c4cec","#e34a33","#504A4B","#756bb1"] # sp and ex rate and net div rate
			# sp file
			f_name = mcmc_file.replace("mcmc.log","sp_rates.log")
			resS = get_marginal_rates(f_name,min_age_t,max_age_t,nbins,burnin)
			fig_title = "Speciation (%s)" % ( name_file)
			r_str += get_r_plot(resS,col=colors[0],parameter="Speciation rate",min_age=max(min_age_t,min_age),max_age=min(max_age,max_age_t),plot_title=fig_title,plot_log=logT,run_simulation=1)
			# ex file
			f_name = mcmc_file.replace("mcmc.log","ex_rates.log")
			resE = get_marginal_rates(f_name,min_age_t,max_age_t,nbins,burnin)
			r_str += get_r_plot(resE,col=colors[1],parameter="Extinction rate",min_age=max(min_age_t,min_age),max_age=min(max_age,max_age_t),plot_title="Extinction",plot_log=logT,run_simulation=0)
			# net div rate
			r_str += plot_net_rate(resS,resE,col=colors[2],min_age=max(min_age_t,min_age),max_age=min(max_age,max_age_t),plot_title="Net diversification",n_bins= nbins)
			# longevity
			lon_avg = 1./np.mean(resE[6],axis=0)	
			r_str += get_r_plot([resE[0],lon_avg,lon_avg,lon_avg,0],col=colors[3],parameter="Mean longevity",min_age=max(min_age_t,min_age),max_age=min(max_age,max_age_t),plot_title="Longevity",plot_log=logT,run_simulation=0,plot_shifts=0,line_wd=4)
			
		#except:
		#	print "Could not read file:", mcmc_file
	r_str += "\n\nn <- dev.off()"
	out="%s/%sRTT_plots.r" % (wd,outname)
	outfile = open(out, "wb") 
	outfile.writelines(r_str)
	outfile.close()
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		cmd="cd '%s' & Rscript %sRTT_plots.r" % (wd,outname)
	else:
		cmd="cd '%s'; Rscript %sRTT_plots.r" % (wd,outname)
	print "Plots saved in %s (%sRTT_plots)" % (wd,outname)
	os.system(cmd)






# x = get_prior_shift(20,0,20)
# x = get_prior_shift(100,0,0)
# res = get_prior_shift(1300,0,130) 
# 
# 
# 
#res = get_prior_shift(1300,0,130) 
#x = res[0]
#empirical_prior = mean(x[x>0])
#
