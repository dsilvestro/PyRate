# python3
import keras
import numpy as np
from numpy import *
import scipy.special
np.set_printoptions(suppress= 1) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit
from keras.models import Sequential
from keras.layers import Dense
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow import set_random_seed
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

import argparse, sys

p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('-layers',    type=int, help='', default= 0, metavar= 0)
p.add_argument('-path',      type=str, help='', default= "", metavar= 0)
p.add_argument('-batch',     type=int, help='', default= 100, metavar= 100)
p.add_argument('-epochs',    type=int, help='', default= 100, metavar= 100)
p.add_argument('-verbose',   type=int, help='', default= 0, metavar= 0)
p.add_argument('-loadNN',    type=str, help='', default= '', metavar= '')
p.add_argument('-data_dir',  type=str, help='', default= 0, metavar= 0)
p.add_argument('-seed',      type=int, help='', default= 0, metavar= 0)
p.add_argument('-tax_bias',  type=float, help='', default= 0, metavar= 0)
p.add_argument('-run_empirical',           type=int,   help='', default= 1, metavar= 1)
p.add_argument('-time_slice',              type=float, help='', default= [0,0], metavar= [0,0], nargs=2)


p.add_argument('-run_simulations',         type=int,   help='', default= 0, metavar= 0)
p.add_argument('-run_model',               type=int,   help='', default= 0, metavar= 0)
p.add_argument('-run_validation_accuracy', type=int,   help='', default= 0, metavar= 0)
p.add_argument('-run_test_accuracy',       type=int,   help='', default= 0, metavar= 0)
p.add_argument('-run_tests',               type=int,   help='', default= 0, metavar= 0)
p.add_argument('-run_pyrate_sim_data',     type=int,   help='', default= 0, metavar= 0)
args = p.parse_args()


plot_curves               = 1    # plot acc/loss


run_simulations           = args.run_simulations            # 
run_model                 = args.run_model                  # train NN
run_validation_accuracy   = args.run_validation_accuracy    # run validation       dataset and estimate accuracy
run_test_accuracy         = args.run_test_accuracy          # run test dataset and estimate accuracy
run_tests                 = args.run_tests                  # and estimate thresholds
run_pyrate_sim_data       = args.run_pyrate_sim_data      
run_empirical             = args.run_empirical          
time_slice = args.time_slice

# NN SETTINGS
n_hidden_layers = args.layers # number of extra hidden layers
use_q_est = 1
max_epochs = args.epochs
n_training_data_set = 120000
n_test_data_set = 10000
batch_size_fit = args.batch # batch size
units_multiplier = 1 # number of nodes per input 

# MODEL SETTINGS
fix_q = 0                                         # set to 0 to sample it from uniform
n_Hbins = 100                                     # bins of histogram occs-per-species
maxNbinsPerSpecies = 20                           # bins of histogram timebins-per-species
nCat = 5                                          # discretization of shape parameter
HbinsSh = np.array([0.2,0.6, 0.8,1.2,1.4,2])

Hbins = np.linspace(0.5,(n_Hbins+0.5),n_Hbins+1)
Hbins = np.array(list(Hbins)+[1000.5])
thresh=0.15

if fix_q: Qest=""
else: Qest="Qest"
if max_epochs != 200: epo = "%sEpochs" % max_epochs
else: epo =""

bat = "%sbatch" % batch_size_fit
if units_multiplier != 1: bat = bat + "%sUnit" % units_multiplier

# SET SEEDS
if args.seed==0: rseed = np.random.randint(1000,9999)
else: rseed = args.seed
np.random.seed(rseed)
random.seed(rseed)
set_random_seed(rseed)


activation_function = "relu" # "tanh"

# EMPIRICAL
n_rnd_q_estimates = 100
min_n_taxa = 25


path = args.path

if args.loadNN == "":
	model_name = "%sNNmodel_%slayers%s%s_%s" % (path, n_hidden_layers,epo,bat,rseed)
else:
	model_name = args.loadNN




# DEF SIZE OF THE FEATURES
hSize = len(Hbins)-1+maxNbinsPerSpecies+1
if use_q_est: 
	hSize = len(Hbins)-1+maxNbinsPerSpecies+1+1
# FEATURES
"""
1) hist occs
2) hist NbinsPerSpecies
3) EpochBinSize
4) Qest

"""

### GENERATE DATA
def get_hist(q=0,N=1000,min_data_size=50,max_data_size=1000,K=7,HbinsShape = [],fixed_shape=0, fixed_scale=0):
	EpochBinSize = np.random.uniform(2,10,N)	
	print("\nSimulating data...")
	if len(HbinsShape)==0:
		HbinsShape = np.linspace(0,2,K+1)
	#nFsampled = nFsampled[0:(N*avg_data_size)].reshape(N,avg_data_size)
	if q==0:
		q_rates = np.random.uniform(0.25,1,N)
	else:
		q_rates = np.ones(N)*q
	
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
		if maxNbinsPerSpecies>0:
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
			if use_q_est:
				h_temp = np.append(np.histogram(nFsampled,bins=Hbins,density=True)[0], hOccs)
				h_temp = np.append(h_temp, EpochBinSize[i])
				h[i,:] = np.append(h_temp, q_estimate)
			else:
				h_temp = np.append(np.histogram(nFsampled,bins=Hbins,density=True)[0], hOccs)
				h[i,:] = np.append(h_temp, EpochBinSize[i])
		else:
			h_temp = np.histogram(nFsampled,bins=Hbins,density=True)[0]
			h[i,:] = np.append(h_temp, EpochBinSize[i])
		# determine shape class
		hShape = np.digitize(rnd_shapes[i],HbinsShape)-1
		label_prob[i,hShape] = 1.
		labels[i] = hShape
	return h, label_prob, labels

if run_model:	
	f1 = "%sweipoi_training.npy" % (path)
	f2 = "%sweipoi_trainLabelsPr.npy" % (path)
	f3 = "%sweipoi_trainLabels.npy" % (path)
	
	if run_simulations:
		weipoi_training,weipoi_trainLabelsPr,weipoi_trainLabels = get_hist(N=n_training_data_set,K=nCat,HbinsShape=HbinsSh,q=fix_q)		
		np.save(file=f1,arr=weipoi_training)
		np.save(file=f2,arr=weipoi_trainLabelsPr)
		np.save(file=f3,arr=weipoi_trainLabels)
	else:
		weipoi_training      = np.load(f1)
		weipoi_trainLabelsPr = np.load(f2)
		weipoi_trainLabels   = np.load(f3)
	modelFirstRun=Sequential() # init neural network

	### DEFINE from INPUT HIDDEN LAYER
	# use_bias=True | adds "bias node" intercept in relu function
	# units =8 : number of nodes in hidden layer
	# input_shape=4 : number of features in the data
	# kernel_initializer : init weights

	inShape = np.shape(weipoi_training)[1]

	modelFirstRun.add(Dense(input_shape=(inShape,),units=int(units_multiplier*inShape),activation=activation_function,kernel_initializer="glorot_normal",use_bias=True))
	### ADD HIDDEN LAYER
	# glorot_normal, glorot_uniform, 
	# relu, tanh, sigmoid, 
	for jj  in range(n_hidden_layers):
		modelFirstRun.add(Dense(units=int(units_multiplier*inShape),activation=activation_function,kernel_initializer="glorot_normal",use_bias=True))
	### DEFINE FROM HIDDEN LAYER TO OUTPUT
	# units=3 : number of nodes in the output
	# "softmax" activation functions to convert real numbers into probabilities
	modelFirstRun.add(Dense(units=nCat,activation="softmax",kernel_initializer="glorot_normal",use_bias=True))

	modelFirstRun.summary()


	# loss: cost function 
	# optimizer="adam" : gradient descent algorithm with dynamic learning parameter (eg autotuning)
	modelFirstRun.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

	# Epoch full iteration from input to output across entire training set
	# BAtch size: every 5 datapoints (entries in the table) update weights
	# validation_split=0.2 : 20% data used for validation
	print("Running model.fit") 
	history=modelFirstRun.fit(weipoi_training,weipoi_trainLabelsPr,epochs=max_epochs,batch_size=batch_size_fit,validation_split=0.2,verbose=args.verbose)

	if plot_curves:
		fig = plt.figure(figsize=(20, 8))
		fig.add_subplot(121)
		#plt.figure(figsize=[8,6])
		plt.plot(history.history['loss'],'r',linewidth=3.0)
		plt.plot(history.history['val_loss'],'b',linewidth=3.0)
		plt.legend(['Training loss', 'Validation Loss'],fontsize=12)
		plt.xlabel('Epochs',fontsize=12)
		plt.ylabel('Loss',fontsize=12)
		plt.title('Loss Curves',fontsize=12)
 
		# Accuracy Curves
		fig.add_subplot(122)
		#plt.figure(figsize=[8,6])
		plt.plot(history.history['acc'],'r',linewidth=3.0)
		plt.plot(history.history['val_acc'],'b',linewidth=3.0)
		plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=12)
		plt.xlabel('Epochs',fontsize=12)
		plt.ylabel('Accuracy',fontsize=12)
		plt.title('Accuracy Curves',fontsize=12)
		#plt.show()
		
		file_name = "%s_res.pdf" % (model_name)
		pdf = matplotlib.backends.backend_pdf.PdfPages(file_name)

		pdf.savefig( fig )
		pdf.close()
		
		


	#predictions=np.argmax(model.predict(weipoi_test),axis=1)
	#confusion_matrix(weipoi_testLabels,predictions)
      #
      #
	#scores=model.evaluate(weipoi_test,weipoi_testLabelsPr,verbose=0)
	#print("Test accuracy rate:: %.2f%%"%(scores[1]*100))
	#print("Test error rate: %.2f%%"%(100-scores[1]*100))
	#print('Test cross-entropy loss:',round(scores[0],2))


	# OPTIM OVER VALIDATION AND THEN TEST ON TEST DATASET (THAT'S THE FINAL ACCURACY)
	optimal_number_of_epochs = np.argmin(history.history['val_loss'])
	print("optimal number of epochs:", optimal_number_of_epochs+1)
	history.history['val_acc'][optimal_number_of_epochs]

	model=Sequential() # init neural network
	model.add(Dense(input_shape=(inShape,),units=int(units_multiplier*inShape),activation=activation_function,kernel_initializer="glorot_normal",use_bias=True))
	for jj  in range(n_hidden_layers):
		model.add(Dense(units=int(units_multiplier*inShape),activation=activation_function,kernel_initializer="glorot_normal",use_bias=True))
	model.add(Dense(units=nCat,activation="softmax",kernel_initializer="glorot_normal",use_bias=True))
	model.summary()
	model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
	history=model.fit(weipoi_training,weipoi_trainLabelsPr,epochs=optimal_number_of_epochs+1,batch_size=batch_size_fit,validation_split=0.2, verbose=args.verbose)
	
	model.save_weights(model_name)
	print("\nModel saved as:", model_name, optimal_number_of_epochs+1)

else:
	print("Load weights...")
	model=Sequential() # init neural network
	model.add(Dense(input_shape=(hSize,),units=int(units_multiplier*hSize),activation=activation_function,kernel_initializer="glorot_normal",use_bias=True))
	for jj  in range(n_hidden_layers):
		model.add(Dense(units=int(units_multiplier*hSize),activation=activation_function,kernel_initializer="glorot_normal",use_bias=True))
	model.add(Dense(units=nCat,activation="softmax",kernel_initializer="glorot_normal",use_bias=True))
	model.summary()
	model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
	model.load_weights(model_name, by_name=False)
	print("done.")

if run_validation_accuracy:
	f1 = "%sweipoi_training.npy" % (path)
	f2 = "%sweipoi_trainLabelsPr.npy" % (path)
	f3 = "%sweipoi_trainLabels.npy" % (path)
	weipoi_test         = np.load(f1)
	weipoi_testLabelsPr = np.load(f2)
	weipoi_testLabels   = np.load(f3)
	# remove first 80% used for training
	weipoi_test         = weipoi_test[96000:,:]        
	weipoi_testLabelsPr = weipoi_testLabelsPr[96000:,:] 
	weipoi_testLabels   = weipoi_testLabels[96000:]    
	
	predictions=np.argmax(model.predict(weipoi_test),axis=1)
	confusion_matrix(weipoi_testLabels,predictions)
	scores=model.evaluate(weipoi_test,weipoi_testLabelsPr,verbose=0)
	print("\nValidation accuracy rate:\t%.3f%%" %(scores[1]*100))
	print("Validation error rate:\t%.3f%%" %(100-scores[1]*100))
	print('Validation cross-entropy loss:\t %s' % round(scores[0],3))


if run_test_accuracy:
	f1 = "%s/weipoi_test.npy" % (path)
	f2 = "%s/weipoi_testLabelsPr.npy" % (path)
	f3 = "%s/weipoi_testLabels.npy" % (path)	
	if run_simulations:
		weipoi_test,weipoi_testLabelsPr,weipoi_testLabels = get_hist(N=n_test_data_set,K=nCat,HbinsShape=HbinsSh,q=fix_q)
		np.save(file=f1,arr=weipoi_test)
		np.save(file=f2,arr=weipoi_testLabelsPr)
		np.save(file=f3,arr=weipoi_testLabels)
	else:
		weipoi_test         = np.load(f1)
		weipoi_testLabelsPr = np.load(f2)
		weipoi_testLabels   = np.load(f3)
	
	predictions=np.argmax(model.predict(weipoi_test),axis=1)
	confusion_matrix(weipoi_testLabels,predictions)
	scores=model.evaluate(weipoi_test,weipoi_testLabelsPr,verbose=0)
	print("\nTest accuracy rate: %.3f%%"%(scores[1]*100))
	print("Test error rate: %.3f%%"%(100-scores[1]*100))
	print('Test cross-entropy loss:',round(scores[0],3))


if run_tests:
	## TEST FOR FALSE POSITIVE (shape = 1)
	print("Tests for shape = 1")
	constExt_test,constExt_testLabelsPr,constExt_testLabels = get_hist(N=10000,K=nCat,HbinsShape=HbinsSh,fixed_shape=1)
	estimate_par = model.predict(constExt_test)
	predictions=np.argmax(estimate_par,axis=1)
	print(confusion_matrix(constExt_testLabels,predictions))
	sADE = estimate_par[:,0]
	print("Frequency of significant strong under-estimation:",len(sADE[sADE>0.5])/len(sADE))
	sADE = estimate_par[:,1]
	print("Frequency of significant moderate under-estimation:",len(sADE[sADE>0.5])/len(sADE))
	lADE = estimate_par[:,3]
	print("Frequency of significant moderate over-estimation:",len(lADE[lADE>0.5])/len(lADE))
	lADE = estimate_par[:,4]
	print("Frequency of significant strong over-estimation:",len(lADE[lADE>0.5])/len(lADE))
	lADE = estimate_par[:,2]
	print("Frequency of correct estimation (null):",len(lADE[lADE>0.5])/len(lADE))
	### add code to find threshold that brings false positive rate < 0.05
	Perr = 1
	thresh = 0.8
	while Perr >= 0.05:
		Perr = len(estimate_par[estimate_par[:,2]<thresh,3])/len(estimate_par[:,0])
		thresh -= 0.01
	print("Frequency of wrong null rejection:",len(lADE[lADE<thresh])/len(lADE), "threshold:", round(thresh,3))
	Perr = 1
	thresh = 0.8
	while Perr >= 0.01:
		Perr = len(estimate_par[estimate_par[:,2]<thresh,3])/len(estimate_par[:,0])
		thresh -= 0.01
	print("Frequency of wrong null rejection:",len(lADE[lADE<thresh])/len(lADE), "threshold:", round(thresh,3))


	## TEST FOR FALSE POSITIVE (shape = 1)
	print("\n\nTests for shape = 1 (fixed scale and q)")
	constExt_test,constExt_testLabelsPr,constExt_testLabels = get_hist(N=1000,K=nCat,HbinsShape=HbinsSh,fixed_shape=1,fixed_scale=2.5,q=fix_q)
	estimate_par = model.predict(constExt_test)
	predictions=np.argmax(estimate_par,axis=1)
	print(confusion_matrix(constExt_testLabels,predictions))
	sADE = estimate_par[:,0]
	print("Frequency of significant strong under-estimation:",len(sADE[sADE>0.5])/len(sADE))
	sADE = estimate_par[:,1]
	print("Frequency of significant moderate under-estimation:",len(sADE[sADE>0.5])/len(sADE))
	lADE = estimate_par[:,3]
	print("Frequency of significant moderate over-estimation:",len(lADE[lADE>0.5])/len(lADE))
	lADE = estimate_par[:,4]
	print("Frequency of significant strong over-estimation:",len(lADE[lADE>0.5])/len(lADE))
	lADE = estimate_par[:,2]
	print("Frequency of correct estimation (null):",len(lADE[lADE>0.5])/len(lADE))
	### add code to find threshold that brings false positive rate < 0.05
	Perr = 1
	thresh1 = 0.8
	while Perr >= 0.05:
		Perr = len(estimate_par[estimate_par[:,2]<thresh1,3])/len(estimate_par[:,0])
		thresh1 -= 0.01
	print("Frequency of wrong null rejection:",len(lADE[lADE<thresh1])/len(lADE), "threshold:", round(thresh1,3))
	
	
	## TEST FOR FALSE NEGATIVE (shape = 0.6)
	print("Tests for shape = 0.6")
	ADE1Ext_test,ADE1Ext_testLabelsPr,ADE1Ext_testLabels = get_hist(N=1000,K=nCat,HbinsShape=HbinsSh,fixed_shape=0.6)
	estimate_par = model.predict(ADE1Ext_test)
	predictions=np.argmax(estimate_par,axis=1)
	print(confusion_matrix(ADE1Ext_testLabels,predictions))
	print("Frequency of significant moderate over-estimation (shape >> 1):",len(estimate_par[estimate_par[:,3]>0.05,3])/len(estimate_par[:,0]))
	lADE = estimate_par[:,2]
	print("Frequency of false negative (shape ~ 1):", len(lADE[lADE>thresh])/len(lADE))

	## TEST FOR FALSE NEGATIVE (shape = 1.7)
	print("Tests for shape = 1.7")
	ADE1Ext_test,ADE1Ext_testLabelsPr,ADE1Ext_testLabels = get_hist(N=1000,K=nCat,HbinsShape=HbinsSh,fixed_shape=1.7) # fixed_scale=2.5
	estimate_par = model.predict(ADE1Ext_test)
	predictions=np.argmax(estimate_par,axis=1)
	print(confusion_matrix(ADE1Ext_testLabels,predictions))
	sADE = estimate_par[:,0:2]
	print("Frequency of significant strong under-estimation (shape << 1):", np.size(sADE[sADE>0.05])/len(sADE))
	lADE = estimate_par[:,2]
	print("Frequency of false negative (shape ~ 1):", len(lADE[lADE>thresh])/len(lADE))


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
		rndBr.append(max(ages)-min(ages))
	
	rndBr = np.array(rndBr)
	# MLE of q rate
	qA= sum(num_occs_per_species[num_occs_per_species>1]-1)/sum(rndBr[num_occs_per_species>1]+0.00001)
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
	


######## TEST EMPIRICAL DATA SETS
if run_empirical and args.tax_bias==0:
	import glob, os
	pt = args.data_dir
	direct="%s/*.txt" % (pt)
	files=glob.glob(direct)
	files=sort(files)
	#print("Found following files:", files)

	indx = 0
	outfile_name = "%s/resNN" % (os.path.dirname(pt))
	if np.sum(time_slice)>0:
		outfile_name = "_%s_%s_%s" % (outfile_name, time_slice[0], time_slice[1] )
	outfile = open(outfile_name+".txt" , "w") 
	
	out_str = "file_name\tN_sp\tN_occ\tbinSize\tQ_est\tmin_age\tmax_age\t0-0.6\t\t0.6-0.8\t\t0.8-1.2\t\t1.2-1.4\t\t1.4-2.0"
	outfile.writelines(out_str)
	for indx in np.arange(len(files)):
		f= files[indx]
		print("analyzing file:", f)
		f_name = os.path.splitext(os.path.basename(f))[0]
		tbl = np.genfromtxt(f,dtype="str")
		
		if np.sum(time_slice)>0:
			n_remaining_sp, tbl = filter_species_time_slice(tbl)
			if n_remaining_sp < min_n_taxa: 
				out_str = "%s\t%s" % (f_name, n_remaining_sp)
				out_str = out_str+"\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA"
				outfile.writelines("\n"+out_str)
				continue
		
		sp_indx = [0]
		for i in range(2,np.shape(tbl)[0]):
			sp_name = tbl[i,0]
			if sp_name== tbl[i-1,0]:
				sp_indx.append(sp_indx[-1])
			else:
				sp_indx.append(sp_indx[-1]+1)
	
		sp_indx = np.array(sp_indx)
		
		if len(np.unique(sp_indx)) < min_n_taxa: 
			out_str = "%s\t%s" % (f_name, len(np.unique(sp_indx)))
			out_str = out_str+"\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA"
			outfile.writelines("\n"+out_str)
			continue
		
		q_est = np.array([estimate_q_from_range_data(tbl) for i in range(n_rnd_q_estimates)])
		
		num_occs_per_species = np.unique(sp_indx,return_counts=1)[1]
		hist = np.histogram(num_occs_per_species,bins=Hbins,density=True)[0]
		
		bin_size = get_mean_bin_size(tbl)
		EpochBins = np.linspace(0,1000, 1+int(1000/bin_size)) 

		# count boundary crossers
		if maxNbinsPerSpecies>0:
			hOccs = np.zeros(maxNbinsPerSpecies)
			for tax in np.unique(tbl[1:,0]):
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
		
		par_est = []
		for i in range(n_rnd_q_estimates):
			hist = np.append(hist_temp, q_est[i])
			estimate_par = model.predict(hist.reshape(1,len(hist)))
			par_est.append(estimate_par[0])
		
		par_est = np.array(par_est)
		par_est_avg = np.mean(par_est,axis=0)
		par_est_min = np.min(par_est,axis=0)
		par_est_max = np.max(par_est,axis=0)
		
		max_occs_age = np.amax(tbl[ 1:, 2:4 ].astype(float))
		min_occs_age = np.amin(tbl[ 1:, 2:4 ].astype(float))
		
		
		out_str = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % (f_name,len(np.unique(sp_indx)),np.shape(tbl)[0]-1, round(bin_size,3), round(np.mean(q_est),3),min_occs_age, max_occs_age)
		
		for j in range(len(par_est_avg)):
			out_str += "%s\t(%s-%s)\t" % (round(par_est_avg[j],3),round(par_est_min[j],3),round(par_est_max[j],3))
		#print(out_str)
		outfile.writelines("\n"+out_str)
		outfile.flush()
	outfile.close()
	print("\nResults saved in:", outfile_name, "\n")
		


run_empirical_taxonbias = args.tax_bias
if run_empirical_taxonbias:
	import glob, os
	pt = args.data_dir
	direct="%s/*.txt" % (pt)
	files=glob.glob(direct)
	files=sort(files)

	indx = 0
	outfile_name = "%s/resNN-%s" % (os.path.dirname(pt), run_empirical_taxonbias)
	if np.sum(time_slice)>0:
		outfile_name = "_%s_%s_%s" % (outfile_name, time_slice[0], time_slice[1] )
	outfile = open(outfile_name+".txt" , "w") 
	out_str = "file_name\tN_sp\tN_occ\tbinSize\tQ_est\t0-0.6\t\t0.6-0.8\t\t0.8-1.2\t\t1.2-1.4\t\t1.4-2.0"
	outfile.writelines(out_str)
	for indx in np.arange(len(files)):
		f= files[indx]
		print("analyzing file:", f)
		f_name = os.path.splitext(os.path.basename(f))[0]
		par_est = []
		for i in range(n_rnd_q_estimates):
			
			tbl = np.genfromtxt(f,dtype="str")

			sp_label = np.unique(tbl[1:,0])
			num_occs_per_species = np.unique(tbl[1:,0],return_counts=1)[1]
			hist = np.histogram(num_occs_per_species,bins=Hbins,density=True)[0]
	
			bin_size = get_mean_bin_size(tbl)
			EpochBins = np.linspace(0,1000, 1+int(1000/bin_size)) 
			#print("old",hist)		
		
			# INTRODUCE TAXONOMY BIAS
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
			if maxNbinsPerSpecies>0:
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
			estimate_par = model.predict(hist.reshape(1,len(hist)))
			par_est.append(estimate_par[0])
		
		par_est = np.array(par_est)
		#print(par_est)
		par_est_avg = np.mean(par_est,axis=0)
		par_est_min, par_est_max = [], []
		for j in range(len(par_est_avg)):
			x_temp = np.sort(par_est[:,j])
			par_est_min.append( x_temp[int(0.025*n_rnd_q_estimates)] )
			par_est_max.append( x_temp[int(0.975*n_rnd_q_estimates)] )
		
		out_str = "%s\t%s\t%s\t%s\t%s\t" % (f_name,len(np.unique(sp_label)),np.shape(tbl)[0]-1, round(bin_size,3), round(np.mean(q_est),3))
		
		for j in range(len(par_est_avg)):
			out_str += "%s\t(%s-%s)\t" % (round(par_est_avg[j],3),round(par_est_min[j],3),round(par_est_max[j],3))
		outfile.writelines("\n"+out_str)
	outfile.close()
	print("\nResults saved in:", outfile_name, "\n")
		

