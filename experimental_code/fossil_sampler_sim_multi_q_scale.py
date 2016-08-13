from numpy import *
import numpy as np
import os
from scipy.special import gamma
np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit


# fossilSampler .
#os.chdir(r'C:\Users\oskar\Documents\Dropbox\PyRate_Age-Dependency_and_Beyond\Toy_Datasets_TreeSimGM\BAT_simulator\testingzone')
#os.chdir('/Users/daniele/Dropbox-personal/Dropbox/PyRate_Age-Dependency_and_Beyond/Toy_Datasets_TreeSimGM/BAT_simulator/testingzone')
os.chdir("/Users/daniele/Desktop/try/pyrate_ade/multiHPP")

n_simulations=1
W_shape_vec = np.round(np.random.uniform(1.5,1.5,n_simulations),3) # shapes[sim_no]
q_rate_vec  = np.round(np.random.uniform(0.5,1.5,2*n_simulations),3).reshape((n_simulations,2)) # preservation rate
W_scale_vec = np.round(np.random.uniform(4,7,2*n_simulations),3).reshape((n_simulations,2))
W_scale_vec[:,1]=W_scale_vec[:,0]
generate_output = True
shift_time = 15.

for sim_no in range(0,n_simulations):
	def mean_ext_WR(x,W_shape,W_scale):
		return mean((W_shape/W_scale)*(x/W_scale)**(W_shape-1)), W_scale * gamma(1 + 1./W_shape)


	R = 30. # root age
	N = np.random.randint(100,150) # no. species
	E = 1 # if 0 all extinct, if 1 includes extant
	#shapes = [0.5,1.,1.5]
	q_rate  = q_rate_vec[sim_no]   #np.round(np.random.uniform(0.5,1.5,2),3) # preservation rate
	W_shape = W_shape_vec[sim_no]  #np.round(np.random.uniform(0.5,1.5),3) # shapes[sim_no]
	W_scale = W_scale_vec[sim_no]  #np.round(np.random.uniform(3,7,2),3)
	v= np.linspace(0.0001,10,1000)
	#mean_ext_WR(v,W_shape,W_scale)
	# CI
	#br_length = sort(np.random.weibull(W_shape,10000)*W_scale)
	#print br_length[int(0.025*len(br_length))],br_length[int(0.975*len(br_length))]


	
	# make all lineages extinct
	if E==0: R = max(br_length)+1.
	
	TS = np.sort(np.random.uniform(0,R,N))[::-1]
	TE = np.zeros(N)+R
	
	# species originating prior to the shift time
	br_length_0 = np.random.weibull(W_shape,N)*W_scale[0]
	#print mean(br_length_0),W_scale[0]
	# species going extinct after shift time
	TE[TS>shift_time] = TS[TS>shift_time]- br_length_0[TS>shift_time]
	ind_TE_after = (TE<shift_time).nonzero()[0]
	
	for i in ind_TE_after:
		patial_br = shift_time-TE[i]
		TE[i] = shift_time-patial_br/(W_scale[0]/W_scale[1])
		#__ br_temp = np.random.weibull(W_shape,1)*W_scale[1]
		#__ while br_temp <= (TS[i]-shift_time): br_temp = np.random.weibull(W_shape,1)*W_scale[1]
		#__ TE[i]= TS[i]-br_temp
	
	
	# species originating AFTER the shift time
	br_length_1 = np.random.weibull(W_shape,N)*W_scale[1]
	TE[TS<shift_time] = TS[TS<shift_time]- br_length_1[TS<shift_time]
	
	TE[TE<0]=0
	
	for i in range(len(TS)):
		print i, round(TS[i],3), round(TE[i],3)
        
	print len(TE[TE==0]), N, len(TE[TE==0])/float(N)
	#print TS[TE==0]
	
	if generate_output is True:
		filename="sim_%s_%s_%s.txt" % (W_shape,W_scale[0],W_scale[1])

		outfile = open(filename, "wb") 
		outfile.writelines("clade	species	ts	te\n")
		for i in range(N):
			outfile.writelines("0\t%s\t%s\t%s\n" % (i,TS[i],TE[i]))

		outfile.close()



		# reading simulated file
		filename="sim_%s_%s_%s" % (W_shape,W_scale[0],W_scale[1])
		simi = np.loadtxt(fname=filename+".txt", skiprows=1)
		TS=simi[:,2]
		TE=simi[:,3]


		## SIMULAITON FUNCTIONS
		def write_to_file(f, o):
			sumfile = open(f , "wb") 
			sumfile.writelines(o)
			sumfile.close()

		# N=repeat(0,len(TS)) ## DANIELE im no sure about this....

		def resample_simulation(TS,TE, verbose=1):
			# uniform sample
			br = TS-TE
			N = []

			FAbeta, LObeta=zeros(len(TS)), zeros(len(TS))
			all_records=list()
			
			no_data=0.
			for i in range(len(br)): #looping over species
				ts= TS[i]
				te= TE[i]
				samples=[]
				# entire sp prior to shift time
				if ts>shift_time and te>shift_time:
					n = np.random.poisson(q_rate[0]*(ts-te))
					if n>0: samples=np.random.uniform(te,ts,n)
				# species crossing
				elif ts>shift_time:
					n_0 = np.random.poisson(q_rate[0]*(ts-shift_time))						
					samples_0 = list(np.random.uniform(shift_time,ts,n_0))
					n_1 = np.random.poisson(q_rate[1]*(shift_time))						
					samples_1=list(np.random.uniform(0,shift_time,n_1))
					n=n_0+n_1
					if n>0: samples=np.array(samples_0+samples_1)
				# species after shift time
				else:
					n = np.random.poisson(q_rate[1]*(ts-te))
					if n>0: samples=np.random.uniform(te,ts,n)

				N.append(n)

				if len(samples)>0: 
					samples = np.sort(samples)[::-1]
					FAbeta[i]=max(samples)
					LObeta[i]=min(samples)
					all_records.append(samples)
				elif TE[i]==0: no_data+=1 # extant species with no data
		
	
			g,a,b,e,f="no. records:\n", "\nts (true):\n","\nte (true):\n","\nts (obs):\n","\nte (obs):\n"
			for i in range(len(TS)):
				g += "%s\t" % (N[i])	
				a += "%s\t" % (TS[i])	
				b += "%s\t" % (TE[i])	
				e += "%s\t" % (FAbeta[i])	
				f += "%s\t" % (LObeta[i])
			data="\ndata \n%s"  % (all_records)		
			
			#for i in range(len(all_records)):
			#	if len(all_records[i])==0: no_data+=1.
	
	
			a1= "\n\nTotal species: %s - %s observed (%s); %s extant; %s no data)" % \
			(len(TE), int(len(all_records)-no_data), round((len(all_records)-no_data)/len(TE),2),len(TE[TE==0]), no_data)

			a1+="\ntotal length: %s (observed: %s,q: %s,%s)\n" % \
			(sum(TS-TE), sum(FAbeta-LObeta), q_rate[0],q_rate[1])
	
			print "\n", a1
	
			o=''.join([a1,g,a,b,e,f]) # a0,
	
			taxon_sampling = round((len(all_records)-no_data)/float(len(TE)),2)
			return [all_records, o,taxon_sampling]






		# RUN SIMULATION
		sim_data = resample_simulation(TS,TE)
		all_records = sim_data[0]

		# final taxon sampling 
		rho = sim_data[2]
		# simulatiom name
		sim_output = "sim_%s" % (sim_no)
		# names log files                                # extant, extinct, q0, q1, shape, scale0, scale1
		output_name = "sim%s_%s_%s_%s_%s_%s_%s_%s" % (sim_no, N,len(TE[TE==0]),q_rate[0],q_rate[1],W_shape,W_scale[0],W_scale[1])



		#print "\n\n", all_records, len(all_records)
		#filtering sizes
		if len(all_records) >= 20 and len(all_records) <=200:
			data="#!/usr/bin/env python\nfrom numpy import * \n\n"
			d="\nd=[sim_data]"
			names="\nnames=['%s']" % (output_name)
			data += "\nsim_data = %s"  % (all_records)
			taxa_names="\ntaxa_names=["
			for i in range(len(all_records)): 
				taxa_names+= "'sp_%s'" % (i)
				if i<len(all_records)-1: taxa_names +=","
			taxa_names += "]\ndef get_taxa_names(): return taxa_names\n"                     
			f="\ndef get_data(i): return d[i]\ndef get_out_name(i): return names[i]"
			all_d=data+d+names+taxa_names+f
			#write_to_file(r"\fossils\%s.py" % output, all_d) 	
			#write_to_file(r"\fossils\%s_summary.txt" % output, sim_data[1]) 	
			write_to_file("%s.py" % sim_output, all_d) 	
			write_to_file("%s_summary.txt" % sim_output, sim_data[1]) 	
		else:
			print("Skipping "+ filename + " : too big or too small")

		print "saved as: ", sim_output

quit()
