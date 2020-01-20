# python3
import glob, os, argparse
import numpy as np
from numpy import *
import scipy.special
np.set_printoptions(suppress= 1) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit

p = argparse.ArgumentParser()  # description='<input file>')
p.add_argument("-r", type=str, help="Path ro ruNNer (download at: github.com/dsilvestro/ruNNer)", default=None, required=True)
p.add_argument("-d", type=str, help="Path to input data files", default="", metavar="", required=True)
p.add_argument("-m", type=str, help="Full path of the pre-trained ADE-NN model", default="", metavar="", required=True)
p.add_argument("-q_samples", type=int, help="Number of replicates (preservation rate samples)", default=10, metavar=10)
p.add_argument("-tax_bias", type=float, help="Fraction of simulated taxonomic bias", default=0, metavar=0)
p.add_argument('-time_slice', type=float, help='min/max ages of time slice of interest', default= [0,0], metavar= [0,0], nargs=2)

args = p.parse_args()

ruNNer = os.path.join(args.r,"ruNNer.py")
wd = args.d
nn_model = args.m
n_rnd_q_estimates = args.q_samples
run_empirical_taxonbias = args.tax_bias
time_slice = args.time_slice

def data_simulator(q=0,N=1000,min_data_size=50,max_data_size=1000,K=5,HbinsShape = [],fixed_shape=0, fixed_scale=0):
	# MODEL SETTINGS
	if len(HbinsShape)==0: 
		HbinsSh = np.array([0.2,0.6, 0.8,1.2,1.4,2])
	EpochBinSize = np.random.uniform(2,10,N)	
	print("\nSimulating data...")
	if len(HbinsShape)==0:
		HbinsShape = np.linspace(0,2,K+1)
	#nFsampled = nFsampled[0:(N*avg_data_size)].reshape(N,avg_data_size)
	if q==0:
		q_rates = np.random.uniform(0.25,1,N)
	else:
		q_rates = np.ones(N)*q
	hSize = len(Hbins)-1+maxNbinsPerSpecies+2
	h = np.zeros((N,hSize))
	n_species = np.random.randint(min_data_size,max_data_size,N)
	rnd_indx = np.random.randint(0,3,N)
	if fixed_shape==0:
		#rnd_shapes = np.exp(np.random.uniform(np.log(0.5),np.log(2),N))
		# balanced categories
		categ_vec = np.random.randint(0,K,N)
		rnd_shapes = np.random.uniform(HbinsSh[categ_vec],HbinsSh[categ_vec+1])
		
	else:
		rnd_shapes = np.ones(N)*fixed_shape
	label_prob = np.zeros((N,K))
	labels = np.zeros(N)
	if fixed_scale==0:
		mean_longevity = np.random.uniform(1.5,10,N)
		rnd_scales = mean_longevity/scipy.special.gamma(1+1/rnd_shapes)
		#rnd_scales = np.random.uniform(2.5,6,N)
	else: rnd_scales = np.ones(N)*fixed_scale
	for i in np.arange(N):
		EpochBins = np.linspace(0,1000, 1+int(1000/EpochBinSize[i])) 
		if i % 1000 == 0: print("simulation %s/%s" % (i,N))
		rW = np.random.weibull(rnd_shapes[i], n_species[i])*rnd_scales[i]
		rW[rW>990] = 990 # avoid extreme lifespans falling outside of the allowed time window
		nF = np.random.poisson(q_rates[i]*rW)
		# actual occs| hist of how many sp extend over 1, 2, ... time bins
		rnd_spec_time = np.random.uniform(0,10,n_species[i])
		hOccs = np.zeros(maxNbinsPerSpecies)
		brL=0
		for j in np.arange(n_species[i]):
			if nF[j]>0:
				oF = rnd_spec_time[j] + np.random.uniform(0,rW[j],nF[j])
				ndigi = np.digitize(oF,EpochBins)
				nBins = min(maxNbinsPerSpecies,len(np.unique(ndigi)))-1
				hOccs[nBins] = hOccs[nBins]+ 1
				coarse_occs = np.random.uniform(EpochBins[ndigi-1],EpochBins[ndigi])
				br_temp = max(coarse_occs)-min(coarse_occs)
				brL += br_temp 
			else: pass # species with no records are not obs data
		if brL==0:
			print(q_rates[i])
			q_estimate = np.random.uniform(0.25,1)
		else:
			q_estimate = sum(nF[nF>1] - 1)/brL
		#print(q_rates[i],q_estimate)
		hOccs = hOccs/np.sum(hOccs)	
		nF[nF>1000] = 1000	
		nFsampled = nF[nF>0]	
		if maxNbinsPerSpecies>0:
			h_temp = np.append(np.histogram(nFsampled,bins=Hbins,density=True)[0], hOccs)
			h_temp = np.append(h_temp, EpochBinSize[i])
			h[i,:] = np.append(h_temp, q_estimate)
		else:
			h_temp = np.histogram(nFsampled,bins=Hbins,density=True)[0]
			h[i,:] = np.append(h_temp, EpochBinSize[i])
		# determine shape class
		hShape = np.digitize(rnd_shapes[i],HbinsShape)-1
		label_prob[i,hShape] = 1.
		labels[i] = hShape
	return h, label_prob, labels

def estimate_q_from_range_data(tbl):
	minAges = np.min(tbl[1:,2:4].astype(float),axis=1)
	maxAges = np.max(tbl[1:,2:4].astype(float),axis=1)
	# assign species indexes
	sp_indx = [0]
	for i in range(2,np.shape(tbl)[0]):
		sp_name = tbl[i,0]
		if sp_name== tbl[i-1,0]:
			sp_indx.append(sp_indx[-1])
		else:
			sp_indx.append(sp_indx[-1]+1)
	sp_indx = np.array(sp_indx)
	num_occs_per_species = np.unique(sp_indx,return_counts=1)[1]
	rndBr = []
	for i in np.unique(sp_indx):
		minAges_temp = minAges[sp_indx==i]
		maxAges_temp = maxAges[sp_indx==i]
		xN = num_occs_per_species[i]
		ages = np.random.uniform(minAges_temp,maxAges_temp, xN) # rows: occs, cols: species
		rndBr.append(np.max(ages)-np.min(ages))
	
	rndBr = np.array(rndBr)
	# MLE of q rate
	qA= np.sum(num_occs_per_species[num_occs_per_species>1]-1)/np.sum(rndBr[num_occs_per_species>1]+0.00001)
	return qA
	
def get_mean_bin_size(tbl):
	tbl_sp = tbl[ 1:, 2:4 ].astype(float)
	oF = np.max(tbl_sp,1) - np.min(tbl_sp,1)
	return np.mean(oF)
	
def filter_species_time_slice(tbl):
	max_occs_ages = np.amax(tbl[ 1:, 2:4 ].astype(float),1)
	min_occs_ages = np.amin(tbl[ 1:, 2:4 ].astype(float),1)
	sp_indx = [0]
	for i in range(2,np.shape(tbl)[0]):
		sp_name = tbl[i,0]
		if sp_name== tbl[i-1,0]:
			sp_indx.append(sp_indx[-1])
		else:
			sp_indx.append(sp_indx[-1]+1)
	sp_indx = np.array(sp_indx)	
	sp_include = []
	n_species_included = []
	for i in np.unique(sp_indx):
		max_occs_ages_sp = max_occs_ages[sp_indx==i]
		min_occs_ages_sp = min_occs_ages[sp_indx==i]
		if np.max(max_occs_ages_sp)<=np.max(time_slice) and np.min(min_occs_ages_sp)>=np.min(time_slice):
			sp_include = sp_include + list(np.where(sp_indx==i)[0]) 
			n_species_included.append(i)
	sp_include_indx_in_tbl = np.array(sp_include) +1
	
	row_indx = np.array( [0] + list(sp_include_indx_in_tbl) )
	tbl_new = tbl[row_indx]
	return(len(np.unique(n_species_included)), tbl_new)
	
def get_data_array(f):
	f_name = os.path.splitext(os.path.basename(f))[0]
	rnd_data = []
	for i in range(n_rnd_q_estimates):	
		tbl = np.genfromtxt(f,dtype="str")
		
		if np.sum(time_slice)>0:
			n_remaining_sp, tbl = filter_species_time_slice(tbl)
			if n_remaining_sp < min_n_taxa: 
				print("Not enough species in time slice (%s)" % n_remaining_sp)
				return [0, 0, 0, 0, 0]
			elif i == 0:
				print(n_remaining_sp, "species remaining after filtering time slice") 
		
		sp_label = np.unique(tbl[1:,0])
		num_occs_per_species = np.unique(tbl[1:,0],return_counts=1)[1]
		hist = np.histogram(num_occs_per_species,bins=Hbins,density=True)[0]

		bin_size = get_mean_bin_size(tbl)
		EpochBins = np.linspace(0,1000, 1+int(1000/bin_size)) 

		# INTRODUCE TAXONOMIC BIAS
		num_occs_per_species_pr = 1/num_occs_per_species
		pr_synonym = num_occs_per_species_pr/np.sum(num_occs_per_species_pr)

		# synomize
		n_synonyms = int(run_empirical_taxonbias*len(sp_label))
		syn_sp_lab = np.random.choice(sp_label,n_synonyms,p=pr_synonym,replace=False) # species to be turned into new label
		other_sp_lab = np.setdiff1d(sp_label,syn_sp_lab)
		new_names = np.random.choice(other_sp_lab,n_synonyms,replace=True)
		tax_i=0
		for tax in syn_sp_lab: 
			tbl[ tbl[:,0]==tax, 0] = new_names[tax_i]
	
		sp_label = np.unique(tbl[1:,0])
		num_occs_per_species = np.unique(tbl[1:,0],return_counts=1)[1]
		hist = np.histogram(num_occs_per_species,bins=Hbins,density=True)[0]
	
		q_est = estimate_q_from_range_data(tbl)
		#print("new",hist)		


		# count boundary crossers
		hOccs = np.zeros(maxNbinsPerSpecies)
		for tax in sp_label:
			tbl_sp = tbl[ tbl[:,0]==tax, 2:4 ].astype(float)
			oF = np.random.uniform(np.min(tbl_sp,1),np.max(tbl_sp,1)   )
			nBins = min(maxNbinsPerSpecies,len(np.unique(np.digitize(oF,EpochBins))))-1
			#nBins = min(len( np.unique(np.max(tbl_sp,1)) ) ,  len(np.unique(np.min(tbl_sp,1)))) -1
			if nBins>maxNbinsPerSpecies: nBins=maxNbinsPerSpecies
			hOccs[nBins] = hOccs[nBins]+ 1

		hOccs = hOccs/np.sum(hOccs)
		#print(hOccs)
		hist = np.append(hist, hOccs)
		hist_temp = np.append(hist, bin_size)
	
		hist = np.append(hist_temp, q_est)
		rnd_data.append(hist)
	
	rnd_data = np.array(rnd_data)

	np.save(file=os.path.join(wd,f_name+".npy"),arr=rnd_data)
	print("input file saved as:", os.path.join(wd,f_name+".npy"))
	[binSize, Q_est] = np.mean(rnd_data,axis=0)[-2:]
	return os.path.join(wd,f_name+".npy"), len(sp_label), np.sum(num_occs_per_species), binSize, Q_est, 


outpath = os.path.join(wd, "results-ADE-NN")
direct="%s/*.txt" % (wd)
files=glob.glob(direct)
files=sort(files)
n_Hbins = 100                                     # bins of histogram occs-per-species
Hbins = np.linspace(0.5,(n_Hbins+0.5),n_Hbins+1)
Hbins = np.array(list(Hbins)+[1000.5])
maxNbinsPerSpecies = 20                           # bins of histogram timebins-per-species
min_n_taxa = 20

run_empirical = 1
if run_empirical_taxonbias>0:
	outtag = "-outname taxbias_%s" % run_empirical_taxonbias
else: outtag = ""
if np.sum(time_slice) > 0:
	outtag = outtag + "T-%s-%s" % (np.min(time_slice), np.max(time_slice))

outlab = "strongD mildD Const mildI strongI"
if run_empirical:
	for indx in range(len(files)):
		f= files[indx]
		print("analyzing file:", f)
		par_est = []
		infile, n_sp, n_occ, binSize, Q_est = get_data_array(f)
		if infile==0: 
			continue
		print("n_species:", n_sp, "n_occs:", n_occ, "avg_bin_size:", binSize, "est_sampling_rate:", Q_est)
		cmd = "python3 %s -mode predict -e %s -loadNN %s -outpath %s -rescale_data 0 -outlabels %s %s" \
		% (ruNNer, infile, nn_model, outpath, outlab, outtag)
		os.system(cmd)


run_simulations = 0
n_training_data_set = 10000
if run_simulations:
	f1 = os.path.join(wd, "sim_features.npy")
	f3 = os.path.join(wd, "sim_labels.npy")
	weipoi_training,weipoi_trainLabelsPr,weipoi_trainLabels = data_simulator(N=n_training_data_set)		
	np.save(file=f1,arr=weipoi_training)
	np.save(file=f3,arr=weipoi_trainLabels)
