from numpy import *
import numpy as np
import os
from scipy.special import gamma
np.set_printoptions(suppress=True) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit


os.chdir("/Users/daniele/Desktop/try/pyrate_ade/HPP_extant")

for sim_no in range(1):
	def mean_ext_WR(x,W_shape,W_scale):
		return mean((W_shape/W_scale)*(x/W_scale)**(W_shape-1)), W_scale * gamma(1 + 1./W_shape)


	R = 15. # root age
	N = np.random.randint(100,150) # no. species
	E = 1 #round(np.random.uniform(0,0.25),3) # fraction of extant
	q_rate = round(np.random.uniform(0.5,1.5),3) # preservation rate
	generate_output = True
	shapes = [0.5,1.,1.5]
	W_shape = round(np.random.uniform(1.5,1.5),3) # shapes[sim_no]
	W_scale = round(np.random.uniform(2.5,5),3)
	v= np.linspace(0.0001,10,1000)
	mean_ext_WR(v,W_shape,W_scale)
	# CI
	br_length = sort(np.random.weibull(W_shape,10000)*W_scale)
	#print br_length[int(0.025*len(br_length))],br_length[int(0.975*len(br_length))]


	br_length = np.random.weibull(W_shape,N)*W_scale
	
	# make all lineages extinct
	if E==0: R = max(br_length)+1.
	
	TS = np.random.uniform(0,R,N)
	#ind_rand_extant = np.random.choice(np.arange(N), size=int(N*E), replace=False,p=br_length/sum(br_length))

	#TS[ind_rand_extant]=np.random.uniform(0,br_length[ind_rand_extant],len(ind_rand_extant))
	TE = TS-br_length
	TE[TE<0]=0

	print len(TE[TE==0]), N, len(TE[TE==0])/float(N)
	#print TS[TE==0]
	
	if generate_output is True:
		filename="sim_%s_%s.txt" % (W_shape,W_scale)

		outfile = open(filename, "wb") 
		outfile.writelines("clade	species	ts	te\n")
		for i in range(N):
			outfile.writelines("0\t%s\t%s\t%s\n" % (i,TS[i],TE[i]))

		outfile.close()



		# reading simulated file
		filename="sim_%s_%s" % (W_shape,W_scale)
		simi = np.loadtxt(fname=filename+".txt", skiprows=1)
		TS=simi[:,2]
		TE=simi[:,3]


		## SIMULAITON FUNCTIONS
		def write_to_file(f, o):
			sumfile = open(f , "wb") 
			sumfile.writelines(o)
			sumfile.close()

		# N=repeat(0,len(TS)) ## DANIELE im no sure about this....

		def resample_simulation(TS,TE, beta_par=3,q=1,rho=1,minFO=0,verbose=1):
			# uniform sample
			n = TS-TE
			N = np.random.poisson(q*n)
			if q==0: N =np.random.poisson(3*n)
			if minFO>0: N=N+1
	
			if rho <1:
				eliminate=int((1-rho)*len(N))
				DEL=np.random.choice(len(N), eliminate, replace=False)
				N[DEL]=0
		
			FAbeta, LObeta=zeros(len(n)), zeros(len(n))
			all_records=list()
			
			no_data=0.
			for i in range(len(n)): #looping over species
		
				if TE[i]==0: 
					## setting extinctiom of extant taxa as getting extinct at the present
					rW = np.random.weibull(float(filename.split("_")[1]),1)*float(filename.split("_")[2])
					#with the following while we avoid extinction times that are before the present
					while rW <= TS[i]: rW = np.random.weibull(float(filename.split("_")[1]),1)*float(filename.split("_")[2])

					m=TS[i]-rW # m is equal to the future projected time to live....
					n_DA=(TS[i]-m)
					N[i]=np.random.poisson(q*n_DA) 
					#if q==0: N[i] =np.random.poisson(3*n_DA)
					#samples= TS[i] - np.random.beta(beta_par,beta_par,N[i])*n_DA # + m
					#samples=samples[samples>0] # avoid negative
					# UNIFORM SAMPLING
					samples=np.random.uniform(0,TS[i],np.random.poisson(q*TS[i]))
					if len(samples)>0:
						samples=np.concatenate((samples,array([0]))) #, axis=1)
					else: samples = []
					#print samples
				elif N[i]>0:
					#samples=np.random.beta(beta_par,beta_par,N[i]) *n[i] +TE[i]
					# UNIFORM SAMPLING
					samples=np.random.uniform(TE[i],TS[i],N[i]) # *n[i] +TE[i]
				else: samples=[] # no record

				if len(samples)>0: 
					samples = np.sort(samples)[::-1]
					FAbeta[i]=max(samples)
					LObeta[i]=min(samples)
					all_records.append(samples)
				elif TE[i]==0: no_data+=1 # extant species with no data
		
			if q==0: # trend 95-05%
				x=all_records
				NTAX=list()
				for i in range(len(x)):
					s_max=max(x[i])
					rnd = np.random.uniform(0,1,len(x[i]))
					th=array(.95*(1-x[i]/s_max) +.05)
					in2 = (rnd<th).nonzero()[0]
					sub=x[i][in2] #) [I]
					if len(x[i])==1 and x[i][0]==0: NTAX.append(x[i])
					else:
						if min(x[i])==0:
							if len(sub)==0: sub=np.append(sub, 0)
							elif min(sub)>0: sub=np.append(sub, 0)
						if len(sub)>=1: NTAX.append(sub)
				all_records=NTAX
	
			g,a,b,e,f="no. records:\n", "\nts (true):\n","\nte (true):\n","\nts (obs):\n","\nte (obs):\n"
			for i in range(len(TS)):
				g += "%s\t" % (N[i])	
				a += "%s\t" % (TS[i])	
				b += "%s\t" % (TE[i])	
				e += "%s\t" % (FAbeta[i])	
				f += "%s\t" % (LObeta[i])
			data="\ndata (beta=%s)\n%s"  % (beta_par, all_records)		
			
			#for i in range(len(all_records)):
			#	if len(all_records[i])==0: no_data+=1.
	
	
			a1= "\n\nTotal species: %s - %s observed (%s); %s extant; %s no data)" % \
			(len(TE), int(len(all_records)-no_data), round((len(all_records)-no_data)/len(TE),2),len(TE[TE==0]), no_data)

			a1+="\ntotal length: %s (observed: %s, beta: %s, minFO: %s, q: %s, rho: %s)\n" % \
			(sum(TS-TE), sum(FAbeta-LObeta), beta_par, minFO, q, rho)
	
			print "\n", a1
	
			o=''.join([a1,g,a,b,e,f]) # a0,
	
			taxon_sampling = round((len(all_records)-no_data)/float(len(TE)),2)
			return [all_records, o,taxon_sampling]






		# RUN SIMULATION
		sim_data = resample_simulation(TS,TE,q=q_rate)
		all_records = sim_data[0]

		# final taxon sampling 
		rho = sim_data[2]
		# simulatiom name
		sim_output = "sim_%s" % (sim_no)
		# names log files
		output_name = "sim%s_%s_%s_%s_%s_%s_%s" % (sim_no, N,E,rho,q_rate,W_shape,W_scale)



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
