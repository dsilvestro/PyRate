from numpy import *
import numpy as np
import os,platform
import lib_utilities as util
import csv 

def get_marginal_rates(times,rates,grid):
	mr = np.zeros(len(grid))
	for i in range(len(times)-1):
		grid_ind = np.intersect1d((grid<=times[i]).nonzero()[0],(grid>=times[i+1]).nonzero()[0])
		mr[grid_ind.astype(int)] = rates[i]
	return mr
		

def r_plot_code(res,wd,name_file,alpha=0.5,plot_title="RTT plot"):
	data  = "library(scales)\ntrans=%s" % (alpha)
	if platform.system() == "Windows" or platform.system() == "Microsoft":
		data+= "\n\npdf(file='%s\%s_RTT.pdf',width=0.6*9, height=0.6*14)\npar(mfrow=c(2,1))" % (wd,name_file) # 9
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
	t=loadtxt(f, skiprows=max(1,burnin))
	head = next(open(f)).split() # should be faster
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
		m_sp_matrix[i] = get_marginal_rates(l_shift_times,l_rates,grid)
		m_shift_times = np.array([min_root_age]+ list(t[i,ex_shift_ind])+[max_death_age])
		m_rates = t[i,ex_ind]
		m_ex_matrix[i] = get_marginal_rates(m_shift_times,m_rates,grid)

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
		cmd="cd %s; Rscript %s\%s_RTT.r" % (wd,wd,name_file)
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