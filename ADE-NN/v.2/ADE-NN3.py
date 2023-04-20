# python3
import glob, os, argparse, sys
import numpy as np
import pandas as pd
import scipy.special
import scipy.ndimage as nd 
np.set_printoptions(suppress= True, precision=3) # prints floats, no scientific notation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
import matplotlib.pyplot as plt


# p = argparse.ArgumentParser()  # description='<input file>')
# p.add_argument("-r", type=str, help="Path ro ruNNer (download at: github.com/dsilvestro/ruNNer)", default=None, required=True)
# p.add_argument("-d", type=str, help="Path to input data files", default="", metavar="", required=True)
# p.add_argument("-m", type=str, help="Full path of the pre-trained ADE-NN model", default="", metavar="", required=True)
# p.add_argument("-q_samples", type=int, help="Number of replicates (preservation rate samples)", default=10, metavar=10)
# p.add_argument("-tax_bias", type=float, help="Fraction of simulated taxonomic bias", default=0, metavar=0)
# p.add_argument('-time_slice', type=float, help='min/max ages of time slice of interest', default= [0,0], metavar= [0,0], nargs=2)
#
# args = p.parse_args()
#
# ruNNer = os.path.join(args.r,"ruNNer.py")
# wd = args.d
# nn_model = args.m
# n_rnd_q_estimates = args.q_samples
# run_empirical_taxonbias = args.tax_bias
# time_slice = args.time_slice


"""
TODO list:
1. check distributions for rnd params (bin size, q, Epoch size)
2. check filter taxa function
3. add q rate heterogeneity (gamma distributed) <- check the effect of this
4. save model to file
5. check taxonomic bias option (currently commented out)


data_simulator_mixture(None, 1, plot=True, fixed_shape=np.array([2, 4]), fixed_scale=np.array([1.5, 10]))                     
data_simulator_mixture(None, 10, plot=False)                                                                                                             

"""

def print_update(s):
    sys.stdout.write('\r')
    sys.stdout.write(s)
    sys.stdout.flush()

def data_simulator(q=None, # if None: rnd drawn
                   N=1000,
                   min_data_size=50,
                   max_data_size=1000,
                   fixed_shape=0, 
                   fixed_scale=0,
                   magnitude = 2,
                   n_Hbins = 100,  # bins of histogram occs-per-species
                   maxNbinsPerSpecies = 20, # bins of histogram timebins-per-species
                   min_n_taxa = 20,
                   gamma_model=True,
                   alpha = 1, # rate heterogeneity across species (smaller alpha = greater variation)
                   verbose=0):
    """
    N: number of datasets
    
    
    """
    Hbins = np.linspace(0.5,(n_Hbins+0.5),n_Hbins+1)
    Hbins = np.array(list(Hbins)+[1000.5])
    EpochBinSize = np.random.uniform(2, 10, N)
                   
                   
    # MODEL SETTINGS
    print("\n")
    
    # preservation 
    if q is None:
        q_rates = np.random.uniform(0.25,1,N)
    else:
        q_rates = np.ones(N)*q
        
        
    
    hSize = len(Hbins)-1+maxNbinsPerSpecies+2
    features = np.zeros((N,hSize))
    n_species = np.random.randint(min_data_size,max_data_size,N)
    
    if fixed_shape==0: 
        # draw rnd shape  
        rnd_shapes = np.exp(np.random.uniform(np.log(1 / magnitude), np.log(magnitude),N))   
    else:
        rnd_shapes = np.ones(N) * fixed_shape
    
    if fixed_scale==0:
        # draw rnd scale  
        mean_longevity = np.random.uniform(1.5,10,N)
        rnd_scales = mean_longevity / scipy.special.gamma(1 + 1 / rnd_shapes)
    else: 
        rnd_scales = np.ones(N)*fixed_scale
    
    
    for i in np.arange(N):
        #loop over datasets
        EpochBins = np.linspace(0,1000, 1+int(1000/EpochBinSize[i])) 
        if i % 10 == 0: 
            print_update("Simulating data...%s/%s" % (i,N))
        
        # simulate species longevities
        rW = np.random.weibull(rnd_shapes[i], n_species[i])*rnd_scales[i]    
        rW[rW>990] = 990 # avoid extreme lifespans falling outside of the allowed time window
        
        
        # simulate fossil record (n. occs)
        q_dataset = q_rates[i]
        if gamma_model:
            q_dataset = np.random.gamma(alpha, 1/alpha, len(rW))
        
        nF = np.random.poisson(q_rates[i]*rW)
        
        # actual occs| hist of how many sp extend over 1, 2, ... time bins
        rnd_spec_time = np.random.uniform(0,10,n_species[i])
        hOccs = np.zeros(maxNbinsPerSpecies)
        brL=0
        for j in range(n_species[i]):
            if nF[j]>0:
                # species with no records are not obs data
                oF = rnd_spec_time[j] + np.random.uniform(0,rW[j],nF[j])
                ndigi = np.digitize(oF,EpochBins)
                nBins = np.min([maxNbinsPerSpecies,len(np.unique(ndigi))])-1
                hOccs[nBins] = hOccs[nBins]+ 1
                coarse_occs = np.random.uniform(EpochBins[ndigi-1],EpochBins[ndigi])
                br_temp = np.max(coarse_occs)-np.min(coarse_occs)
                brL += br_temp 
            
        if brL==0:
            print(q_rates[i])
            q_estimate = np.random.uniform(0.25,1)
            sys.exit("???")
        else:
            q_estimate = np.sum(nF[nF>1] - 1)/brL
        if verbose:
            print(q_rates[i],q_estimate)
        
        # BUILDING FEATURE SET
        # fraction of species with occurrences in 1, 2, 3, ... N time bins
        hOccs = hOccs / np.sum(hOccs)    
        
        nF[nF>1000] = 1000    
        nFsampled = nF[nF>0]    
        if maxNbinsPerSpecies>0:
            # append fraction of species with 1, 2, 3 ... N occurrences
            h_temp = np.append(np.histogram(nFsampled,bins=Hbins,density=True)[0], hOccs)
            # append avg bin size
            h_temp = np.append(h_temp, EpochBinSize[i])
            # append approximate preservation rate
            features[i,:] = np.append(h_temp, q_estimate)
        else:
            h_temp = np.histogram(nFsampled,bins=Hbins,density=True)[0]
            features[i,:] = np.append(h_temp, EpochBinSize[i])
        
    # BUILD LABELS
    labels = np.vstack((rnd_shapes, mean_longevity)).T
    print("\nDone.")
    return features, labels

def data_simulator_mixture(q=None, # if None: rnd drawn
                           N=1000,
                           min_data_size=50,
                           max_data_size=1000,
                           fixed_shape=None, # if not None -> array of 2 values
                           fixed_scale=None, # if not None -> array of 2 values
                           magnitude = 2, # variation in shapes: 1/magnitude -> magnitude
                           n_Hbins = 100,  # bins of histogram occs-per-species
                           maxNbinsPerSpecies = 20, # bins of histogram timebins-per-species
                           min_n_taxa = 20,
                           gamma_model=True,
                           alpha = 1, # rate heterogeneity across species (smaller alpha = greater variation)
                           verbose=0,
                           plot=False):
    """
    N: number of datasets
    
    
    """
    Hbins = np.linspace(0.5,(n_Hbins+0.5),n_Hbins+1)
    Hbins = np.array(list(Hbins)+[1000.5])
    EpochBinSize = np.random.uniform(2, 10, N)
                   
                   
    # MODEL SETTINGS
    print("\n")
    
    # preservation 
    if q is None:
        q_rates = np.random.uniform(0.25,1,N)
    else:
        q_rates = np.ones(N)*q
        
        
    
    hSize = len(Hbins)-1+maxNbinsPerSpecies+2
    features = np.zeros((N,hSize))
    n_species = np.random.randint(min_data_size,max_data_size,N)
    
    if fixed_shape is None: 
        # draw rnd shape  
        rnd_shapes = np.exp(np.random.uniform(np.log(1 / magnitude), np.log(magnitude),(N, 2)))   
        rnd_shapes = np.sort(rnd_shapes, 1)    
    else:
        rnd_shapes = np.ones((N, 2)) * fixed_shape
    
    if fixed_scale is None:
        # draw rnd scale  
        mean_longevity = np.random.uniform(2,30,(N, 2))
        # mean_longevity = np.sort(mean_longevity, 1)
        rnd_scales = mean_longevity / scipy.special.gamma(1 + 1 / rnd_shapes)
    else: 
        rnd_scales = np.ones((N, 2)) * fixed_scale
        mean_longevity = rnd_scales * scipy.special.gamma(1 + 1/rnd_shapes)
    
    
    for i in np.arange(N):
        #loop over datasets
        EpochBins = np.linspace(0,1000, 1+int(1000/EpochBinSize[i])) 
        if i % 10 == 0: 
            print_update("Simulating data...%s/%s" % (i,N))
        
        # simulate species longevities
        compound_index = np.random.binomial(1, 0.5, n_species[i]) # assign species to one of the two Weibulls
        rW = np.random.weibull(rnd_shapes[i][compound_index], n_species[i]) * rnd_scales[i][compound_index]
        rW[rW>990] = 990 # avoid extreme lifespans falling outside of the allowed time window
        if plot:
            plt.hist(rW)
            plt.show()
        
        
        # simulate fossil record (n. occs)
        q_dataset = q_rates[i]
        if gamma_model:
            q_dataset = np.random.gamma(alpha, 1/alpha, len(rW))
        
        nF = np.random.poisson(q_rates[i]*rW)
        
        # actual occs| hist of how many sp extend over 1, 2, ... time bins
        rnd_spec_time = np.random.uniform(0,10,n_species[i])
        hOccs = np.zeros(maxNbinsPerSpecies)
        brL=0
        for j in range(n_species[i]):
            if nF[j]>0:
                # species with no records are not obs data
                oF = rnd_spec_time[j] + np.random.uniform(0,rW[j],nF[j])
                ndigi = np.digitize(oF,EpochBins)
                nBins = np.min([maxNbinsPerSpecies,len(np.unique(ndigi))])-1
                hOccs[nBins] = hOccs[nBins]+ 1
                coarse_occs = np.random.uniform(EpochBins[ndigi-1],EpochBins[ndigi])
                br_temp = np.max(coarse_occs)-np.min(coarse_occs)
                brL += br_temp 
            
        if brL==0:
            print(q_rates[i])
            q_estimate = np.random.uniform(0.25,1)
            sys.exit("???")
        else:
            q_estimate = np.sum(nF[nF>1] - 1)/brL
        if verbose:
            print(q_rates[i],q_estimate)
        
        # BUILDING FEATURE SET
        # fraction of species with occurrences in 1, 2, 3, ... N time bins
        hOccs = hOccs / np.sum(hOccs)    
        
        nF[nF>1000] = 1000    
        nFsampled = nF[nF>0]    
        if maxNbinsPerSpecies>0:
            # append fraction of species with 1, 2, 3 ... N occurrences
            h_temp = np.append(np.histogram(nFsampled,bins=Hbins,density=True)[0], hOccs)
            # append avg bin size
            h_temp = np.append(h_temp, EpochBinSize[i])
            # append approximate preservation rate
            features[i,:] = np.append(h_temp, q_estimate)
        else:
            h_temp = np.histogram(nFsampled,bins=Hbins,density=True)[0]
            features[i,:] = np.append(h_temp, EpochBinSize[i])
        
    # BUILD LABELS
    labels = np.hstack((rnd_shapes, mean_longevity))
    print("\nDone.")
    return features, labels


def calcHPD(data, level):
    assert (0 < level < 1)
    d = list(data)
    d.sort()
    nData = len(data)
    nIn = int(round(level * nData))
    if nIn < 2 :
        sys.exit('\n\nToo little data to calculate marginal parameters.')
    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)):
            rk = d[k+nIn-1] - d[k]
            if rk < r :
                r = rk
                i = k
    assert 0 <= i <= i+nIn-1 < len(d)
    return (d[i], d[i+nIn-1])


def estimate_q_from_range_data(tbl):
    minAges = np.min(tbl[:,2:4].astype(float),axis=1)
    maxAges = np.max(tbl[:,2:4].astype(float),axis=1)
    # assign species indexes
    sp_id = tbl[:,0]
    sp_id_num = pd.factorize(sp_id)[0]     
    
    # count occs per species 
    num_occs_per_species = np.unique(sp_id_num,return_counts=1)[1]
    
    # get branch length
    ages = np.random.uniform(minAges,maxAges) 
    m1 = nd.maximum(ages, sp_id_num, np.unique(sp_id_num))
    m2 = nd.minimum(ages, sp_id_num, np.unique(sp_id_num))
    rndBr = m1 - m2
    qA= np.sum(num_occs_per_species[num_occs_per_species>1]-1)/np.sum(rndBr[num_occs_per_species>1]+0.00001)   
    # MLE of q rate
    qA= np.sum(num_occs_per_species[num_occs_per_species>1]-1)/np.sum(rndBr[num_occs_per_species>1]+0.00001)
    return qA
      
def get_mean_bin_size(tbl):
    tbl_sp = tbl[ 1:, 2:4 ].astype(float)
    oF = np.max(tbl_sp,1) - np.min(tbl_sp,1)
    return np.mean(oF)
    
def filter_species_time_slice(tbl, time_slice):
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
    
def get_data_array(f, time_slice=None, n_rnd_q_estimates=1, n_Hbins=100, maxNbinsPerSpecies=20):
    
    Hbins = np.linspace(0.5,(n_Hbins+0.5),n_Hbins+1)
    Hbins = np.array(list(Hbins)+[1000.5])
    
    
    f_name = os.path.splitext(os.path.basename(f))[0]
    rnd_data = []
    for i in range(n_rnd_q_estimates):    
        tbl = pd.read_csv(f, sep="\t").to_numpy()
        
        if time_slice is not None:
            # TODO: fix this function
            n_remaining_sp, tbl = filter_species_time_slice(tbl.to_numpy(), time_slice)
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
        if False:
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
        # count boundary crossers
        hOccs = np.zeros(maxNbinsPerSpecies)
        for tax in sp_label:
            tbl_sp = tbl[ tbl[:,0]==tax, 2:4 ].astype(float)
            # randomize fossil age
            oF = np.random.uniform(np.min(tbl_sp,1),np.max(tbl_sp,1)   )
            nBins = np.min([maxNbinsPerSpecies,len(np.unique(np.digitize(oF,EpochBins)))])-1
            hOccs[nBins] = hOccs[nBins]+ 1

        hOccs = hOccs/np.sum(hOccs)
        #print(hOccs)
        hist = np.append(hist, hOccs)
        hist_temp = np.append(hist, bin_size)
    
        hist = np.append(hist_temp, q_est)
        rnd_data.append(hist)
    
    rnd_data = np.array(rnd_data)
    [binSize, Q_est] = np.mean(rnd_data,axis=0)[-2:]
    output = {'features': rnd_data, 
                'n_species': len(sp_label), 
                'n_occs': np.sum(num_occs_per_species), 
                'bin_size': binSize, 
                'estimated_q': Q_est
            }
    return output

def build_nn(dense_nodes=None,
             dense_act_f='relu',
             output_nodes=2, # shape, mean longevity
             output_act_f='softplus',
             loss_f='mse', # mean squared error
             dropout_rate=0,
             verbose=1):
    if dense_nodes is None:
        dense_nodes = [32]
    
    input_layer = [tf.keras.layers.Flatten(input_shape=[features.shape[1]])]
    model = keras.Sequential(input_layer)

    for i in range(len(dense_nodes)):
        model.add(layers.Dense(dense_nodes[i],
                               activation=dense_act_f))
    
    if dropout_rate:
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(output_nodes,
                           activation=output_act_f))

    if verbose:
        print(model.summary())

    model.compile(loss=loss_f,
                  optimizer="adam",
                  metrics=['mse'])
    return model


def fit_rnn(Xt, Yt, model,
            criterion="val_loss",
            patience=10,
            verbose=1,
            batch_size=1000,
            max_epochs=1000,
            validation_split=0.2
            ):
    early_stop = keras.callbacks.EarlyStopping(monitor=criterion,
                                               patience=patience,
                                               restore_best_weights=True)
    history = model.fit(Xt, Yt,
                        epochs=max_epochs,
                        validation_split=validation_split,
                        verbose=verbose,
                        callbacks=[early_stop],
                        batch_size=batch_size
                        )
    return history



def save_nn_model(wd, history, model, filename=""):
    # save training history
    with open(os.path.join(wd, "rnn_history" + filename + ".pkl"), 'wb') as output:  # Overwrites any existing file.
        pkl.dump(history.history, output, pkl.HIGHEST_PROTOCOL)
    # save model
    tf.keras.models.save_model(model, os.path.join(wd, 'rnn_model' + filename))



if __name__ == '__main__':
    # simulate training set
    # features, labels = data_simulator(N=10000, gamma_model=False)
    
    features, labels = data_simulator_mixture(N=10000, 
                                              gamma_model=False,
                                              min_n_taxa = 100,
                                              magnitude=4)

    # save to files
    wd = "/Users/dsilvestro/Desktop/ade-nn3-tests"
    f1 = os.path.join(wd, "sim_features_mixture.npy")
    f2 = os.path.join(wd, "sim_labels.npy")
    np.save(file=f1,arr=features)
    np.save(file=f2,arr=labels)

    longevity_rescaler = 10
    labels[:,2:] = labels[:,2:] / longevity_rescaler

    # build NN model   
    model = build_nn(dense_nodes=[64,8], output_nodes=labels.shape[1], dropout_rate=0.05)
    # train NN
    history = fit_rnn(features, labels, model, batch_size=labels.shape[0], max_epochs=1000)
    
    # save model
    save_nn_model(wd, history, model, filename="")

    # simulate test set
    features_test, labels_test = data_simulator_mixture(N=1000, gamma_model=False, min_n_taxa = 100,
                                              magnitude=4)

    n_predictions = 10 #
    y = np.array([model(features_test, training=True) for _ in range(n_predictions)])
    # y.shape = (n_predictions, n_instances, 2)
    mean_y = np.mean(y, axis=0) # average over dropout replicates
    
    confidence_intervals = {
        'shape.1': [calcHPD(y[i,0]) for i in range(y.shape[0])]
        'shape.2': [calcHPD(y[i,1]) for i in range(y.shape[0])]
        'scale.1': [calcHPD(y[i,2]) for i in range(y.shape[0])]
        'scale.2': [calcHPD(y[i,3]) for i in range(y.shape[0])]
    }

    # plot shapes
    fig = plt.figure(figsize = (14,7))
    fig.add_subplot(1,2,1)
    plt.scatter(labels_test[:,0], mean_y[:,0])
    plt.scatter(labels_test[:,1], mean_y[:,1])
    plt.axline((0, 0), (1, 1), linewidth=2, linestyle='dashed', alpha=0.5, color="k")
    plt.xlabel('True shape')
    plt.ylabel('Estimated shape')

    # plot longevities
    fig.add_subplot(1,2,2)
    plt.scatter(labels_test[:,2], mean_y[:,2] * longevity_rescaler)
    plt.scatter(labels_test[:,3], mean_y[:,3] * longevity_rescaler)
    plt.axline((0, 0), (1, 1), linewidth=2, linestyle='dashed', alpha=0.5, color="k")
    plt.xlabel('True longevity')
    plt.ylabel('Estimated longevity')

    fig.show()

    # empirical data
    f = "Example_data/Athyridida5_lz1.txt"
    out = get_data_array(f)
    y = np.array([model(out['features'], training=True) for _ in range(n_predictions)])


