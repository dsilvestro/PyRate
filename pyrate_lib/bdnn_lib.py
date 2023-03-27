import os, platform, glob, sys
import itertools
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import copy as copy_lib
import warnings
import multiprocessing.pool
from tqdm import tqdm

from scipy.spatial import ConvexHull
from matplotlib.path import Path
from math import comb
from scipy import stats

import pyrate_lib.lib_utilities as util
from PyRate import check_burnin
from PyRate import load_pkl
from PyRate import get_rate_BDNN
from PyRate import get_DT
from PyRate import get_binned_div_traj
from PyRate import get_sp_in_frame_br_length

import fastshap
from fastshap.plotting import get_variable_interactions


from scipy.special import bernoulli, binom
from itertools import chain, combinations
import random


def summarize_rate(r, n_rates):
    r_sum = np.zeros((n_rates, 3))
    r_sum[:, 0] = np.nanmean(r, axis = 0)
    for i in range(n_rates):
        r_i = r[:, i]
        r_i = r_i[~np.isnan(r_i)]
        nData = len(r_i)
        nIn = int(round(0.95 * nData))
        if nIn > 3:
            r_sum[i, 1:] = util.calcHPD(r_i, 0.95)
        else:
            r_sum[i, 1:] = np.nan
    r_sum = np.repeat(r_sum, repeats = 2, axis = 0)
    return r_sum


def get_bdnn_rtt(f, burn):
    f_sp = f + "_sp_rates.log"
    f_ex = f + "_ex_rates.log"
    r = np.loadtxt(f_sp)
    num_it = r.shape[0]
    n_rates = int((r.shape[1] - 1) / 2 + 1)
    time_vec = r[0, n_rates:]
    a = np.abs(np.mean(np.diff(time_vec)))
    time_vec = np.concatenate((np.array([time_vec[0] + a]), time_vec, np.zeros(1)))
    time_vec = np.repeat(time_vec, repeats = 2)
    time_vec = time_vec + np.tile(np.array([0.00001, 0.0]), int(len(time_vec)/2))
    time_vec = time_vec[1:]
    time_vec = np.delete(time_vec, -2)
    burnin = check_burnin(burn, num_it)
    r_sp = np.loadtxt(f_sp, skiprows = max(0, int(burnin)))
    r_ex = np.loadtxt(f_ex, skiprows = max(0, int(burnin)))
    r_sp = r_sp[:, :n_rates]
    r_ex = r_ex[:, :n_rates]
    r_div = r_sp - r_ex
    longevity = 1. / r_ex
    r_sp_sum = summarize_rate(r_sp, n_rates)
    r_ex_sum = summarize_rate(r_ex, n_rates)
    r_div_sum = summarize_rate(r_div, n_rates)
    long_sum = summarize_rate(longevity, n_rates)
    return r_sp_sum, r_ex_sum, r_div_sum, long_sum, time_vec


def plot_bdnn_rtt(f, r_sp_sum, r_ex_sum, r_div_sum, long_sum, time_vec):
    output_wd = os.path.dirname(f)
    name_file = os.path.basename(f)
    out = "%s/%s_RTT.r" % (output_wd, name_file)
    newfile = open(out, "w")
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        wd_forward = os.path.abspath(output_wd).replace('\\', '/')
        r_script = "pdf(file='%s/%s_RTT.pdf', width = 9, height = 7, useDingbats = FALSE)\n" % (wd_forward, name_file)
    else:
        r_script = "pdf(file='%s/%s_RTT.pdf', width = 9, height = 7, useDingbats = FALSE)\n" % (output_wd, name_file)
    r_script += util.print_R_vec('\ntime_vec', time_vec)
    r_script += util.print_R_vec('\nsp_mean', r_sp_sum[:, 0])
    r_script += util.print_R_vec('\nsp_lwr', r_sp_sum[:, 1])
    r_script += util.print_R_vec('\nsp_upr', r_sp_sum[:, 2])
    r_script += util.print_R_vec('\nex_mean', r_ex_sum[:, 0])
    r_script += util.print_R_vec('\nex_lwr', r_ex_sum[:, 1])
    r_script += util.print_R_vec('\nex_upr', r_ex_sum[:, 2])
    r_script += util.print_R_vec('\ndiv_mean', r_div_sum[:, 0])
    r_script += util.print_R_vec('\ndiv_lwr', r_div_sum[:, 1])
    r_script += util.print_R_vec('\ndiv_upr', r_div_sum[:, 2])
    r_script += util.print_R_vec('\nlong_mean', long_sum[:, 0])
    r_script += util.print_R_vec('\nlong_lwr', long_sum[:, 1])
    r_script += util.print_R_vec('\nlong_upr', long_sum[:, 2])
    r_script += "\nlayout(matrix(1:4, ncol = 2, nrow = 2, byrow = TRUE))"
    r_script += "\npar(las = 1, mar = c(4.5, 4.5, 0.5, 0.5))"
    r_script += "\nxlim = c(%s, %s)" % (np.max(time_vec), np.min(time_vec))
    r_script += "\nylim = c(%s, %s)" % (np.nanmin(r_sp_sum), np.nanmax(r_sp_sum))
    r_script += "\nnot_NA = !is.na(sp_mean)"
    r_script += "\nplot(time_vec[not_NA], sp_mean[not_NA], type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Speciation rate')"
    r_script += "\npolygon(c(time_vec[not_NA], rev(time_vec[not_NA])), c(sp_lwr[not_NA], rev(sp_upr[not_NA])), col = adjustcolor('#4c4cec', alpha = 0.5), border = NA)"
    r_script += "\nlines(time_vec[not_NA], sp_mean[not_NA], col = '#4c4cec', lwd = 2)"
    r_script += "\nylim = c(%s, %s)" % (np.nanmin(r_ex_sum), np.nanmax(r_ex_sum))
    r_script += "\nnot_NA = !is.na(ex_mean)"
    r_script += "\nplot(time_vec[not_NA], ex_mean[not_NA], type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Extinction rate')"
    r_script += "\npolygon(c(time_vec[not_NA], rev(time_vec[not_NA])), c(ex_lwr[not_NA], rev(ex_upr[not_NA])), col = adjustcolor('#e34a33', alpha = 0.5), border = NA)"
    r_script += "\nlines(time_vec[not_NA], ex_mean[not_NA], col = '#e34a33', lwd = 2)"
    r_script += "\nylim = c(%s, %s)" % (np.nanmin(r_div_sum), np.nanmax(r_div_sum))
    r_script += "\nnot_NA = !is.na(div_mean)"
    r_script += "\nplot(time_vec[not_NA], div_mean[not_NA], type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Net diversification rate')"
    r_script += "\npolygon(c(time_vec[not_NA], rev(time_vec[not_NA])), c(div_lwr[not_NA], rev(div_upr[not_NA])), col = adjustcolor('black', alpha = 0.3), border = NA)"
    r_script += "\nlines(time_vec[not_NA], div_mean[not_NA], col = 'black', lwd = 2)"
    r_script += "\nabline(h = 0, col = 'red', lty = 2)"
    r_script += "\nylim = c(%s, %s)" % (np.nanmin(long_sum), np.nanmax(long_sum))
    r_script += "\nnot_NA = !is.na(long_mean)"
    r_script += "\nplot(time_vec[not_NA], long_mean[not_NA], type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Longevity (Myr)')"
    r_script += "\npolygon(c(time_vec[not_NA], rev(time_vec[not_NA])), c(long_lwr[not_NA], rev(long_upr[not_NA])), col = adjustcolor('black', alpha = 0.3), border = NA)"
    r_script += "\nlines(time_vec[not_NA], long_mean[not_NA], col = 'black', lwd = 2)"
    r_script += "\ndev.off()"
    newfile.writelines(r_script)
    newfile.close()

    if platform.system() == "Windows" or platform.system() == "Microsoft":
        cmd = "cd %s & Rscript %s_RTT.r" % (output_wd, name_file)
    else:
        cmd = "cd %s; Rscript %s_RTT.r" % (output_wd, name_file)
        print("cmd", cmd)
        os.system(cmd)


def apply_thin(w, thin):
    n_samples = w.shape[0]
#    thin_idx = np.arange(0, n_samples, thin)
    if thin > n_samples:
        thin = n_samples
        print("resample set to the number of mcmc samples:", n_samples)
    thin_idx = np.linspace(1, n_samples, num = int(thin), dtype = int) - 1
    return w[thin_idx,:]


def bdnn_read_mcmc_file(mcmc_file, burn, thin):
    m = pd.read_csv(mcmc_file, delimiter = '\t')
    w_sp_indx = np.array([i for i in range(len(m.columns)) if 'w_lam_' in m.columns[i]])
    w_ex_indx = np.array([i for i in range(len(m.columns)) if 'w_mu_' in m.columns[i]])
    ts_indx = np.array([i for i in range(len(m.columns)) if '_TS' in m.columns[i]])
    te_indx = np.array([i for i in range(len(m.columns)) if '_TE' in m.columns[i]])
    np_m = m.to_numpy()
    num_it = np_m.shape[0]
    burnin = check_burnin(burn, num_it)
    w_sp = np_m[burnin:, w_sp_indx]
    w_ex = np_m[burnin:, w_ex_indx]
    ts = np_m[burnin:, ts_indx]
    te = np_m[burnin:, te_indx]
    if thin > 0:
        w_sp = apply_thin(w_sp, thin)
        w_ex = apply_thin(w_ex, thin)
        ts = apply_thin(ts, thin)
        te = apply_thin(te, thin)
    return w_sp, w_ex, ts, te


def bdnn_reshape_w(posterior_w, bdnn_obj):
    w_list = []
    for w in posterior_w:
        c = 0
        w_sample = []
        for i in range(len(bdnn_obj.bdnn_settings['layers_sizes'])):
            w_layer = w[c:c+bdnn_obj.bdnn_settings['layers_sizes'][i]].reshape(bdnn_obj.bdnn_settings['layers_shapes'][i])
            c += bdnn_obj.bdnn_settings['layers_sizes'][i]
            w_sample.append(w_layer)
            # print(w_layer.shape)        
        w_list.append(w_sample)
    return w_list


def bdnn_parse_results(mcmc_file, pkl_file, burn = 0.1, thin = 0):
    ob = load_pkl(pkl_file)
    w_sp, w_ex, post_ts, post_te = bdnn_read_mcmc_file(mcmc_file, burn, thin)
    post_w_sp = bdnn_reshape_w(w_sp, ob)
    post_w_ex = bdnn_reshape_w(w_ex, ob)
    sp_fad_lad = ob.sp_fad_lad
    return ob, post_w_sp, post_w_ex, sp_fad_lad, post_ts, post_te


def bdnn_time_rescaler(x, bdnn_obj):  
    return x * bdnn_obj.bdnn_settings['time_rescaler']


def get_names_features(bdnn_obj):
    na = copy_lib.deepcopy(bdnn_obj.bdnn_settings['names_features'])
    return na


def is_time_trait(bdnn_obj):
    return bdnn_obj.bdnn_settings['use_time_as_trait'] != 0.0


def backscale_bdnn_time(x, bdnn_obj):
    denom = 1
    if is_time_trait(bdnn_obj):
        denom = bdnn_obj.bdnn_settings['time_rescaler']
    return x / denom


def backscale_time_cond_trait_tbl(cond_trait_tbl, bdnn_obj):
    if is_time_trait(bdnn_obj):
        cond_trait_tbl[:, -7] = backscale_bdnn_time(cond_trait_tbl[:, -7], bdnn_obj)
    return cond_trait_tbl


def backscale_bdnn_features(file_transf_features, bdnn_obj, cond_trait_tbl_sp, cond_trait_tbl_ex):
    cond_trait_tbl_sp = backscale_time_cond_trait_tbl(cond_trait_tbl_sp, bdnn_obj)
    cond_trait_tbl_ex = backscale_time_cond_trait_tbl(cond_trait_tbl_ex, bdnn_obj)
    backscale_par = None
    if file_transf_features != "":
        names_feat = get_names_features(bdnn_obj)
        backscale_par = read_backscale_file(file_transf_features)
        cond_trait_tbl_sp = backscale_tbl(bdnn_obj, backscale_par, names_feat, cond_trait_tbl_sp)
        cond_trait_tbl_ex = backscale_tbl(bdnn_obj, backscale_par, names_feat, cond_trait_tbl_ex)
    return cond_trait_tbl_sp, cond_trait_tbl_ex, backscale_par


def get_trt_tbl(bdnn_obj, rate_type):
    trait_tbl = bdnn_obj.trait_tbls[0]
    if rate_type == 'extinction':
        trait_tbl = bdnn_obj.trait_tbls[1]
    return trait_tbl


def get_bdnn_time(bdnn_obj, ts):
    bdnn_time = np.concatenate((np.array([np.max(ts)+ 0.001]),
                                bdnn_obj.bdnn_settings['fixed_times_of_shift_bdnn'],
                                np.zeros(1)))
    return bdnn_time


def is_binary_feature(trait_tbl):
    n_features = trait_tbl.shape[-1]
    b = np.zeros(n_features, dtype = int)
    freq_state = np.zeros(n_features, dtype = int)
    for i in range(n_features):
        if len(trait_tbl.shape) == 3:
            values, counts = np.unique(trait_tbl[:, :, i], return_counts = True)
        else:
            values, counts = np.unique(trait_tbl[:, i], return_counts=True)
        n_values = len(values)
        b[i] = np.all(np.isin(values, np.arange(n_values)))
        freq_state[i] = values[np.argmax(counts)]
    b = b.astype(bool)
    return b, freq_state


def is_time_variable_feature(trait_tbl):
    n_features = trait_tbl.shape[-1]
    n_time_bins = trait_tbl.shape[0]
    time_bins = np.arange(0, n_time_bins)
    time_var_feat = np.zeros(n_features, dtype = int)
    if trait_tbl.ndim == 3:
        for i in range(n_features):
            tr = trait_tbl[time_bins, :, i]
            std_tr = np.std(tr, axis = 0)
            time_var_feat[i] = np.any(std_tr > 1e-10)
    time_var_feat = time_var_feat.astype(bool)
    return time_var_feat


def get_idx_feature_without_variance(trt_tbl):
    if trt_tbl.ndim == 3:
        sd_species = np.std(trt_tbl[0,:,:], axis = 0)
        sd_time = np.zeros(trt_tbl.shape[2])
        for i in range(trt_tbl.shape[2]):
            for j in range(trt_tbl.shape[1]):
                s = np.std(trt_tbl[:, j, i])
                if s > sd_time[i]:
                    sd_time[i] = s
        idx_feature = np.where((sd_species + sd_time) == 0)[0]
    else:
        sd_species = np.std(trt_tbl, axis = 0)
        idx_feature = np.where(sd_species == 0)[0]
    return idx_feature


def expand_grid(v1, v2):
    mg = np.array(np.meshgrid(v1, v2))
    g = mg.reshape(2, len(v1) * len(v2)).T
    return g


def get_fossil_age(bdnn_obj, fad_lad, rate_type):
    z = 0
    # Also extinction should be from first occurrence on b/c from there on a species can go extinct
    if rate_type == 'extinction':
        z = 1
    fa = fad_lad[:,z]
    fa = bdnn_time_rescaler(fa, bdnn_obj)
    return fa


def get_bin_from_fossil_age(bdnn_obj, fad_lad, rate_type):
    bdnn_time = bdnn_obj.bdnn_settings['fixed_times_of_shift_bdnn']
    bins = np.concatenate((bdnn_time, np.zeros(1)))
    # time_trait_tbl = backscale_bdnn_time(trait_tbl[:, 0, -1], bdnn_obj)
    # time_trait_tbl = time_trait_tbl[::-1]
    # bins = np.zeros(len(time_trait_tbl))
    # for i in range(1, len(time_trait_tbl)):
    #     bins[i] = 2 * time_trait_tbl[i - 1] - bins[i - 1]
    # bins = bins[::-1]
    z = 0
    # Also extinction should be from first occurrence on b/c from there on a species can go extinct
    if rate_type == 'extinction':
        z = 1
    bin_of_species = np.digitize(fad_lad[:,z], bins)
    return bin_of_species


def get_mean_div_traj(post_ts, post_te):
    time_bdnn_div = np.arange(np.max(post_ts), 0, -0.001)
    bdnn_div = np.zeros((len(time_bdnn_div), post_ts.shape[0]))
    for i in range(post_ts.shape[0]):
        bdnn_div[:, i] = get_DT(time_bdnn_div, post_ts[i, :], post_te[i, :])
    mean_div = np.mean(bdnn_div, axis = 1)
    return time_bdnn_div, mean_div


def get_minmaxmean_features(trait_tbl, most_freq, binary, conc_feat, len_cont = 100):
    n_features = trait_tbl.shape[-1]
    m = np.zeros((4, n_features))
    for i in range(n_features):
        if len(trait_tbl.shape) == 3:
            m[0, i] = np.nanmin(trait_tbl[:, :, i])
            m[1, i] = np.nanmax(trait_tbl[:, :, i])
            m[2, i] = np.nanmean(trait_tbl[:, :, i])
        else:
            m[0, i] = np.nanmin(trait_tbl[:, i])
            m[1, i] = np.nanmax(trait_tbl[:, i])
            m[2, i] = np.nanmean(trait_tbl[:, i])
    m[2, binary] = most_freq[binary]
    m[3, :] = len_cont
    m[3, binary] = (m[1, binary] - m[0, binary]) + 1
    is_in_feat_group = np.isin(np.arange(n_features), conc_feat)
    m[3, is_in_feat_group] = 1.0
    return m


def get_names_feature_group(fg):
   return list(fg.keys())


def replace_names_by_feature_group(names, idx, fg):
    names_fg = get_names_feature_group(fg)
    for i in range(len(idx)):
        for j in idx[i]:
            names[j] = names_fg[i]
    return names


def exit_on_missing_feature(names, fg):
    for sublist in fg:
        for item in sublist:
            if (item in names) == False:
                sys.exit(print("Item(s) of the feature group(s) not part of trait file"))


def get_idx_comb_feat(names, fg):
    idx_comb_feat = []
    if fg != "":
        fg = list(fg.values())
        exit_on_missing_feature(names, fg)
        for i in range(len(fg)):
            cf = fg[i]
            icf = np.zeros(len(cf), dtype = int)
            for j in range(len(cf)):
                for k in range(len(names)):
                    if names[k] == cf[j]:
                        icf[j] = k
            idx_comb_feat.append(icf)
    return idx_comb_feat


def get_nrows_conditional_trait_tbl(p, m):
    nrows = 0
    for i in range(p.shape[0]):
        li = m[3, int(p[i, 0])]
        lj = 1
        j = p[i, 1]
        if ~np.isnan(j):
            lj = m[3, int(j)]
        nrows += li * lj
    return int(nrows)


def get_feature_type(k, m, b, conc_group):
    k = int(k)
    # Type of k-th feature
    tk = 4 # continuous
    if b[k]:
        if np.isin(k, conc_group):
            tk = 2 # one-hot encoded discrete
        elif m[3, k] > 2:
            tk = 3 # ordinal 0, 1, 2 etc
        else:
            tk = 1 # binary feature
    return tk


def get_plot_type_ij(i, m, b, j = np.nan, group_features = None):
    conc_group = np.array([])
    if group_features:
        conc_group = np.concatenate(group_features)
    ti = get_feature_type(i, m, b, conc_group)
    pt = ti
    if ~np.isnan(j):
        tj = get_feature_type(j, m, b, conc_group)
        if ti == 1 and tj == 1:
            pt = 5 # binary x binary
        if (ti == 1 and tj == 4) or (ti == 4 and tj == 1):
            pt = 6 # binary x continuous
        if ti == 4 and tj == 4:
            pt = 7 # continuous x continuous
        if (ti == 2 and tj == 1) or (ti == 1 and tj == 2):
            pt = 8 # one-hot discrete x binary
        if (ti == 3 and tj == 1) or (ti == 1 and tj == 3):
            pt = 9  # ordinal x binary
        if ti == 2 and tj == 2:
            pt = 10 # one-hot x one-hot
        if (ti == 2 and tj == 3) or (ti == 3 and tj == 2):
            pt = 11 # one-hot x ordinal
        if ti == 3 and tj == 3:
            pt = 12 # ordinal x ordinal
        if (ti == 2 and tj == 4) or (ti == 4 and tj == 2):
            pt = 13 # one-hot x continuous
        if (ti == 3 and tj == 4) or (ti == 4 and tj == 3):
            pt = 14 # ordinal x continuous
    return pt


def get_plot_type(m, b, group_features):
    n = m.shape[1]
    main = np.arange(n, dtype = float)
    inter = np.array(list(combinations(range(0, n), 2)))
    ij = np.zeros((n + inter.shape[0], 3))
    ij[:] = np.nan
    ij[:len(main), 0] = main
    if len(inter) > 0:
        ij[len(main):, :2] = inter
    for z in range(ij.shape[0]):
        ij[z, 2] = get_plot_type_ij(ij[z, 0], m, b, ij[z, 1], group_features)
    remove = ~(ij[:,0] == ij[:,1])
    ij = ij[remove, :]
    return ij


def which_feature_group(k, group_features):
    r = -1
    for i in range(len(group_features)):
        if np.isin(k, group_features[i]):
            r = i + 0
    return r


def get_plot_idx(p, group_features):
    conc_group = np.array([])
    if group_features:
        conc_group = np.concatenate(group_features)
    pidx = np.zeros(p.shape[0])
    pidx[:] = np.nan # NA when we have an interaction between two features of the same feature group
    if group_features is None:
        # No feature-groups
        pidx = np.arange(p.shape[0])
    else:
        counter = 0
        for k in range(p.shape[0]):
            if np.isnan(p[k, 1]):
                # No interaction
                if ~np.isin(p[k, 0], conc_group):
                    # i is not part of a feature-group
                    pidx[k] = counter
                    counter += 1
                else:
                    # i is part of a feature-group
                    if np.isnan(pidx[k]):
                        # First time to encounter this feature group
                        g = which_feature_group(p[k, 0], group_features)
                        g_idx = group_features[g]
                        pidx[np.logical_and(np.isin(p[:, 0], g_idx), np.isnan(p[:, 1]))] = counter
                        counter += 1
            else:
                # Interaction
                if ~np.isin(p[k, 0], conc_group) and ~np.isin(p[k, 1], conc_group):
                    # Neither i nor j are part of a feature-group
                    pidx[k] = counter
                    counter += 1
                else:
                    if np.isnan(pidx[k]):
                        g1 = which_feature_group(p[k, 0], group_features)
                        g2 = which_feature_group(p[k, 1], group_features)
                        if (g1 == g2) is False:
                            # i and j are not part of the same feature-group
                            if g1 > -1 and g2 == -1:
                                # i part of a feature-group but j not
                                g1_idx = group_features[g1]
                                pidx[np.logical_and(np.isin(p[:, 0], g1_idx), p[:, 1] == p[k, 1])] = counter
                                counter += 1
                            elif g1 == -1 and g2 > -1:
                                # i not part of a feature-group but j
                                g2_idx = group_features[g2]
                                pidx[np.logical_and(p[:, 0] == p[k, 0], np.isin(p[:, 1], g2_idx))] = counter
                                counter += 1
                            else:
                                # i part of one feature-group and j part of a different feature-group
                                g1_idx = group_features[g1]
                                g2_idx = group_features[g2]
                                pidx[np.logical_and(np.isin(p[:, 0], g1_idx), np.isin(p[:, 1], g2_idx))] = counter
                                counter += 1
    return pidx


def modify_feature_group(cond_trait_tbl, i, j, idx_comb_feat, counter, lv):
    group_i = which_feature_group(i, idx_comb_feat)
    group_j = which_feature_group(j, idx_comb_feat)
    if group_i != -1:
        features_group_i = idx_comb_feat[group_i]
        cond_trait_tbl[counter:(counter + lv), features_group_i] = 0.0
        cond_trait_tbl[counter:(counter + lv), i] = 1.0
    if group_j != -1:
        features_group_j = idx_comb_feat[group_j]
        cond_trait_tbl[counter:(counter + lv), features_group_j] = 0.0
        cond_trait_tbl[counter:(counter + lv), j] = 1.0
    return cond_trait_tbl


def get_plot_idx_freq(x):
    _, uinv, ucounts = np.unique(x, return_inverse = True, return_counts = True)
    freq = ucounts[uinv]
    return freq


def trait_combination_exists(w, trait_tbl, i, j, feature_is_time_variable, bdnn_obj, fad_lad, rate_type, pt):
    lw = len(w)
    comb_exists = np.zeros(lw)
    use_time_as_trait = is_time_trait(bdnn_obj)
    time_dd_temp = False # No time as trait, diversity-dependence, or time-variable environment
    if len(trait_tbl.shape) == 3:
        time_dd_temp = True
    if np.isin(pt, np.array([1., 2., 3., 4])):
        # Main effects
        comb_exists = np.ones(lw)
    else:
        # Interactions
        i_bin, j_bin = is_binary_feature(trait_tbl)[0][[i, j]]
        i_time_var = feature_is_time_variable[i]
        j_time_var = feature_is_time_variable[j]
        if i_time_var or j_time_var:
            fa = get_fossil_age(bdnn_obj, fad_lad, rate_type)
            bin_species = get_bin_from_fossil_age(bdnn_obj, fad_lad, rate_type)
            trait_at_fad_or_lad = np.zeros((len(bin_species), trait_tbl.shape[2]))
            for k in range(len(bin_species)):
                trait_at_fad_or_lad[k, :] = trait_tbl[bin_species[k], k, :]
        if np.isin(pt, np.array([5., 8., 9., 10., 11., 12.])):
            # 5:  binary x binary
            # 8:  one-hot-encoded discrete x binary
            # 9:  ordinal discrete x binary
            # 10: one-hot-encoded discrete x one-hot-encoded discrete
            # 11: one-hot-encoded discrete x ordinal discrete
            # 12: ordinal discrete x ordinal discrete
            if time_dd_temp:
                t = trait_tbl[0, :, :] + 0.0
            else:
                t = trait_tbl + 0.0
            if i_time_var:
                t[:, i] = trait_at_fad_or_lad[:, i]
            if j_time_var:
                t[:, j] = trait_at_fad_or_lad[:, j]
            t = t[:, [i, j]]
            observed_comb = np.unique(t, axis = 0)
            for k in range(w.shape[0]):
                 if np.any(np.all(w[k, :] == observed_comb, axis = 1)):
#                if np.isin(w[k, 0], observed_comb[:, 0]) and np.isin(w[k, 1], observed_comb[:, 1]):
                    comb_exists[k] = 1.0
        elif np.isin(pt, np.array([6., 13., 14.])):
            # 6:  binary x continuous
            # 13: one-hot-encoded discrete x continuous
            # 14: ordinal discrete x continuous
            bin_idx = i  # Which is binary?
            cont_idx = j
            bin_idx_w = 0
            cont_idx_w = 1
            if j_bin:
                bin_idx = j
                cont_idx = i
                bin_idx_w = 1
                cont_idx_w = 0
            if time_dd_temp:
                t = trait_tbl[0, :, :] + 0.0
            else:
                t = trait_tbl + 0.0
            if i_time_var:
                t[:, i] = trait_at_fad_or_lad[:, i]
            if j_time_var:
                if (j == (t.shape[1] - 1)) and use_time_as_trait:
                    t[:, j] = fa
                else:
                    t[:, j] = trait_at_fad_or_lad[:, j]
            states = np.unique(t[:, bin_idx]).astype(int)
            wc = w[:, cont_idx_w]
            for s in states:
                # State-dependent continuous trait
                ts = t[t[:, bin_idx] == s, cont_idx]
                in_range_cont = np.logical_and(wc >= np.min(ts), wc <= np.max(ts))
                is_state = w[:, bin_idx_w] == s
                exists_idx = np.logical_and(in_range_cont, is_state)
                comb_exists[exists_idx] = 1.0
        else:
            # 7: continuous x continuous
            if time_dd_temp:
                t = trait_tbl[0, :, :] + 0.0
            else:
                t = trait_tbl + 0.0
            if i_time_var or j_time_var:
                if ((j == (t.shape[1] - 1)) and use_time_as_trait) and ((int(i_time_var) + int(j_time_var)) == 1):
                    # Time as trait but no other time variable feature
                    t[:, j] = fa
                else:
                    # Both features vary through time
                    t = trait_at_fad_or_lad + 0.0
            # Check if all w are inside a convex hull defined by the trait_tbl
            hull = ConvexHull(t[:, [i, j]])  # , qhull_options = 'QJ'
            vertices = t[hull.vertices, :][:, [i, j]]
            path_p = Path(vertices)
            inside_poly = path_p.contains_points(w, radius=0.1)
            comb_exists[inside_poly] = 1.0
    return comb_exists


def build_conditional_trait_tbl(bdnn_obj,
                                sp_fad_lad,
                                ts_post, te_post,
                                len_cont = 100,
                                rate_type = "speciation",
                                combine_discr_features = ""):
    trait_tbl = get_trt_tbl(bdnn_obj, rate_type)
    names_features = get_names_features(bdnn_obj)
    # diversity-dependence
    if "diversity" in names_features:
        div_time, div_traj = get_mean_div_traj(ts_post, te_post)
        bdnn_time = get_bdnn_time(bdnn_obj, ts_post)
        div_traj_binned = get_binned_div_traj(bdnn_time, div_time, div_traj)[:-1]
        div_traj_binned = div_traj_binned / bdnn_obj.bdnn_settings["div_rescaler"]
        div_traj_binned = np.repeat(div_traj_binned, trait_tbl.shape[1]).reshape((trait_tbl.shape[0], trait_tbl.shape[1]))
        div_idx_trt_tbl = -1
        if is_time_trait(bdnn_obj):
            div_idx_trt_tbl = -2
        trait_tbl[ :, :, div_idx_trt_tbl] = div_traj_binned
    n_features = trait_tbl.shape[-1]
    idx_comb_feat = get_idx_comb_feat(names_features, combine_discr_features)
    conc_comb_feat = np.array([])
    if idx_comb_feat:
        names_features = replace_names_by_feature_group(names_features, idx_comb_feat, combine_discr_features)
        conc_comb_feat = np.concatenate(idx_comb_feat)
    names_features = np.array(names_features)
    binary_feature, most_frequent_state = is_binary_feature(trait_tbl)
    minmaxmean_features = get_minmaxmean_features(trait_tbl,
                                                  most_frequent_state,
                                                  binary_feature,
                                                  conc_comb_feat,
                                                  len_cont)
    plot_type = get_plot_type(minmaxmean_features, binary_feature, idx_comb_feat)
    plot_idx = get_plot_idx(plot_type, idx_comb_feat)
    plot_type = np.hstack((plot_type, plot_idx.reshape((len(plot_idx), 1))))
    plot_type = plot_type[~np.isnan(plot_type[:, 3]), :]
    plot_idx_freq = get_plot_idx_freq(plot_type[:, 3])
    feature_without_variance = get_idx_feature_without_variance(trait_tbl)
    feature_is_time_variable = is_time_variable_feature(trait_tbl)
    nr = get_nrows_conditional_trait_tbl(plot_type, minmaxmean_features)
    cond_trait_tbl = np.zeros((nr, n_features + 6))
    cond_trait_tbl[:, :n_features] = minmaxmean_features[2, :] # mean/modal values
    cond_trait_tbl[:, -1] = 1 # combination is observed
    counter = 0
    fad_lad = sp_fad_lad[["FAD", "LAD"]].to_numpy()
    for k in range(plot_type.shape[0]):
        i = int(plot_type[k, 0])
        v1 = np.linspace(minmaxmean_features[0, i], minmaxmean_features[1, i], int(minmaxmean_features[3, i]))
        v1 = v1.reshape((len(v1), 1))
        j = plot_type[k, 1]
        feat_idx = [i]
        if ~np.isnan(j):
            j = int(j)
            feat_idx.append(j)
            v2 = np.linspace(minmaxmean_features[0, j], minmaxmean_features[1, j], int(minmaxmean_features[3, j]))
            v1 = expand_grid(v1.flatten(), v2)
        lv = len(v1)
        cond_trait_tbl[counter:(counter + lv), feat_idx] = v1
        cond_trait_tbl[counter:(counter + lv), -6] = i
        cond_trait_tbl[counter:(counter + lv), -5] = j
        cond_trait_tbl[counter:(counter + lv), -4] = plot_type[k, 2]
        cond_trait_tbl[counter:(counter + lv), -3] = plot_type[k, 3]
        if plot_idx_freq[k] > 1:
            # plot involves features of a feature-group
            cond_trait_tbl = modify_feature_group(cond_trait_tbl, i, j, idx_comb_feat, counter, lv)
        cond_trait_tbl[counter:(counter + lv), -2] = k
        comparison_incl_feature_without_variance = np.any(np.isin([i, j], feature_without_variance))
        cond_trait_tbl[counter:(counter + lv), -1] = trait_combination_exists(cond_trait_tbl[counter:(counter + lv), feat_idx],
                                                                              trait_tbl,
                                                                              i, j,
                                                                              feature_is_time_variable,
                                                                              bdnn_obj,
                                                                              fad_lad,
                                                                              rate_type,
                                                                              plot_type[k, 2])
        if comparison_incl_feature_without_variance:
            cond_trait_tbl[counter:(counter + lv), -1] = np.nan
        counter = counter + lv
    # Remove comparisons when there is no variance of the features
    cond_trait_tbl = cond_trait_tbl[~np.isnan(cond_trait_tbl[:, -1]), ]
    return cond_trait_tbl, names_features


def backscale_bdnn_cont(x, mean_x, sd_x):
    x_back = (x * sd_x) + mean_x
    return x_back


def read_backscale_file(backscale_file):
    backscale_par = []
    if backscale_file:
        backscale_par = pd.read_csv(backscale_file, delimiter = '\t')
    return backscale_par


def backscale_tbl(bdnn_obj, backscale_par, names_feat, tbl):
    if isinstance(backscale_par, pd.DataFrame):
        backscale_names = list(backscale_par.columns)
        for i in range(len(names_feat)):
            na = names_feat[i]
            if na in backscale_names:
                mean_feat = backscale_par[na][0]
                sd_feat = backscale_par[na][1]
                tbl[:, i] = backscale_bdnn_cont(tbl[:, i], mean_feat, sd_feat)
    if 'diversity' in names_feat:
        div_idx = np.where(np.array(["diversity"]) == names_feat)[0]
        tbl[:, div_idx] = tbl[:, div_idx] * bdnn_obj.bdnn_settings['div_rescaler']
    return tbl


def get_conditional_rates(bdnn_obj, cond_trait_tbl, post_w):
    num_it = len(post_w)
    obs = cond_trait_tbl[:, -1] == 1
    rate_cond = np.zeros((np.sum(obs), num_it))
    for i in range(num_it):
        rate_cond[:, i] = get_rate_BDNN(1, # constant baseline
                                        cond_trait_tbl[obs, :-6],
                                        post_w[i], # list of arrays
                                        bdnn_obj.bdnn_settings['hidden_act_f'],
                                        bdnn_obj.bdnn_settings['out_act_f'])
    rate_cond2 = np.zeros((len(cond_trait_tbl), num_it))
    rate_cond2[:] = np.nan
    rate_cond2[obs, :] = rate_cond
    return rate_cond2


def get_rates_summary(cond_rates):
    nrows_cond_rates = cond_rates.shape[0]
    rate_sum = np.zeros((nrows_cond_rates, 3))
    rate_sum[:] = np.nan
    rate_sum[:, 0] = np.mean(cond_rates, axis = 1)
    for i in range(nrows_cond_rates):
        all_nan = np.all(np.isnan(cond_rates[i, :]))
        if not all_nan:
            rate_sum[i, 1:] = util.calcHPD(cond_rates[i, :], .95)
    return rate_sum


def remove_conditional_features(t):
    incl = np.concatenate((t[:,-6].flatten(), t[:,-5].flatten()))
    incl = incl[~np.isnan(incl)]
    incl = np.unique(incl)
    incl = incl[~np.isnan(incl)].astype(int)
    t = t[:, incl]
    return t, incl


def get_observed(bdnn_obj, feature_idx, feature_is_time_variable, fossil_age, fossil_bin, rate_type):
    trait_tbl = get_trt_tbl(bdnn_obj, rate_type)
    if len(trait_tbl.shape) == 3:
        obs_cont = trait_tbl[0, :, feature_idx]
        obs_cont = obs_cont.transpose()
    else:
        obs_cont = trait_tbl[:, feature_idx]
    if np.any(feature_is_time_variable[feature_idx]):
        for z in range(len(fossil_bin)):
            obs_cont[z, :] = trait_tbl[fossil_bin[z], z, feature_idx]
    if np.any(feature_idx + 1 == trait_tbl.shape[-1]) and is_time_trait(bdnn_obj):
        obs_cont[:, obs_cont.shape[1] - 1] = fossil_age # Time is always the last
    return obs_cont


def plot_bdnn_discr(rs, r, tr, r_script, names, names_states, rate_type):
    # binary two states or ordinal
    n_states = tr.shape[0]
    rate_max = np.nanmax(rs[:, 2])
    rate_min = np.nanmin(rs[:, 1])
    rate_max += 0.2 * rate_max
    rate_min -= 0.2 * np.abs(rate_min)
    r_script += "\nylim = c(%s, %s)" % (rate_min, rate_max)
    r_script += "\nxlim = c(-0.5, %s + 0.5)" % (n_states - 1)
    r_script += "\nplot(1, 0, type = 'n', xlim = xlim, ylim = ylim, xlab = '', ylab = '%s', xaxt = 'n')" % rate_type
    if rate_type == 'speciation':
        r_script += "\ncol = colorRampPalette(c('lightblue1', rgb(0, 52, 94, maxColorValue = 255)))(%s)" % n_states
    elif rate_type == 'extinction':
        r_script += "\ncol = colorRampPalette(c(rgb(255, 143, 118, maxColorValue = 255), 'darkred'))(%s)" % n_states
    else:
        r_script += "\ncol = colorRampPalette(c('grey75', 'grey40'))(%s)" % n_states
    for i in range(n_states):
        r_script += "\naxis(side = 1, at = %s, labels = '%s')" % (i, str(names_states[i]))
        # r_script += "\nlines(rep(%s, 2), c(%s, %s), col = col[%s])" % (i, rs[i, 1], rs[i, 2], i + 1)
        # r_script += "\npoints(%s, %s, pch = 19, col = col[%s])" % (i, rs[i, 0], i + 1)
        r_tmp = r[i,:]
        r_tmp =  r_tmp[r_tmp < rate_max]
        r_tmp = r_tmp[r_tmp > rate_min]
        r_script += util.print_R_vec("\nvio_data", r_tmp)
        r_script += "\nvioplot(vio_data, at = %s, add = TRUE, wex = 0.5, rectCol = NA, lineCol = NA, colMed = NA, col = col[%s])" % (i, i + 1)
        r_script += "\nlines(rep(%s, 2), c(%s, %s), lwd = 1.5)" % (i, rs[i, 1], rs[i, 2])
        r_script += "\npoints(%s, %s, pch = 19)" % (i, rs[i, 0])
    r_script += "\ntitle(main = '%s')" % names
    return r_script


def plot_bdnn_cont(rs, tr, r_script, names, plot_time, obs, rate_type):
    # Continuous
    r_script += "\nylim = c(%s, %s)" % (np.nanmin(rs[:, 1]), np.nanmax(rs[:, 2]))
    r_script += util.print_R_vec("\nxlim", np.array([np.nanmin(tr[:, 0]), np.nanmax(tr[:, 0])]))
    if plot_time:
        r_script += "\nxlim = xlim[2:1]"
    if rate_type == 'speciation':
        col = '#6092AF'
    elif rate_type == 'extinction':
        col = '#C5483B'
    else:
        col = 'grey50'
    r_script += util.print_R_vec("\ntr", tr[:, 0])
    r_script += util.print_R_vec("\nr", rs[:, 0])
    r_script += util.print_R_vec("\nr_lwr", rs[:, 1])
    r_script += util.print_R_vec("\nr_upr", rs[:, 2])
    r_script += "\nplot(0, 0, type = 'n', xlim = xlim, ylim = ylim, xlab = '%s', ylab = '%s')" % (names, rate_type)
    r_script += util.print_R_vec("\nobs_x", obs.flatten())
    r_script += "\nd = diff(par('usr')[3:4]) * 0.025"
    r_script += "\nh = c(par('usr')[3], par('usr')[3] + d)"
    r_script += "\nfor (i in 1:length(obs_x)) {"
    r_script += "\n    lines(rep(obs_x[i], 2), h, col = '%s')" % col
    r_script += "\n}"
    r_script += "\npolygon(c(tr, rev(tr)), c(r_lwr, rev(r_upr)), col = adjustcolor('%s', alpha = 0.2), border = NA)" % col
    r_script += "\nlines(tr, r, col = '%s', lwd = 1.5)" % col
    # r_script += "\ntitle(main = '%s')" % names
    return r_script


def convert_onehot_to_numeric(tr_onehot):
    tr_states = np.zeros(tr_onehot.shape[0])
    for i in range(tr_onehot.shape[0]):
        tr_states[i] = np.where(tr_onehot[i,:] == 1.0)[0]
    return tr_states


def plot_bdnn_inter_discr_cont(rs, tr, r_script, names, names_states, plot_time, obs, rate_type):
    # binary/ordinal/discrete x continuous
    keep = ~np.isnan(rs[:, 0])
    rs = rs[keep, :]
    tr = tr[keep, :]
    r_script += "\nylim = c(%s, %s)" % (np.min(rs[:, 1]), np.max(rs[:, 2]))
    r_script += util.print_R_vec("\nxlim", np.array([np.min(tr[:, 0]), np.max(tr[:, 0])]))
    if plot_time:
        r_script += "\nxlim = xlim[2:1]"
    r_script += "\nplot(0, 0, type = 'n', xlim = xlim, ylim = ylim, xlab = '%s', ylab = '%s')" % (names[0], rate_type)
    if tr.shape[1] == 2:
        # binary/ordinal
        states = np.unique(tr[:, 1])
        tr_states = tr[:, 1]
        obs_states = obs[:, 1]
    else:
        tr_states = convert_onehot_to_numeric(tr[:,1:])
        states = np.unique(tr_states)
        obs_states = np.zeros(obs.shape[0])
        obs_states[:] = np.nan
        for i in range(obs.shape[0]):
            w = np.where(obs[i, 1:] == 1.0)[0]
            if w.size > 0:
                obs_states[i] = w[0]
    n_states = len(states)
    if rate_type == 'speciation':
        r_script += "\ncol = colorRampPalette(c('lightblue1', rgb(0, 52, 94, maxColorValue = 255)))(%s)" % n_states
    elif rate_type == 'extinction':
        r_script += "\ncol = colorRampPalette(c(rgb(255, 143, 118, maxColorValue = 255), 'darkred'))(%s)" % n_states
    else:
        r_script += "\ncol = colorRampPalette(c('grey75', 'grey40'))(%s)" % n_states
    for i in range(n_states):
        idx = obs_states == states[i]
        r_script += util.print_R_vec("\nobs_x", obs[idx, 0])
        r_script += "\nd = diff(par('usr')[3:4]) * 0.025"
        r_script += "\nh = c(par('usr')[3], par('usr')[3] + d)"
        r_script += "\nfor (i in 1:length(obs_x)) {"
        r_script += "\n    lines(rep(obs_x[i], 2), h, col = col[%s])" % (i + 1)
        r_script += "\n}"
        idx = tr_states == states[i]
        r_script += util.print_R_vec("\ntr", tr[idx, 0])
        r_script += util.print_R_vec("\nr_lwr", rs[idx, 1])
        r_script += util.print_R_vec("\nr_upr", rs[idx, 2])
        r_script += "\npolygon(c(tr, rev(tr)), c(r_lwr, rev(r_upr)), col = adjustcolor(col[%s], alpha = 0.2), border = NA)" % (i + 1)
    for i in range(n_states):
        idx = tr_states == states[i]
        r_script += util.print_R_vec("\ntr", tr[idx, 0])
        r_script += util.print_R_vec("\nr", rs[idx, 0])
        r_script += "\nlines(tr, r, col = col[%s], lwd = 1.5)" % (i + 1)
    r_script += "\nleg = c()"
    for i in range(n_states):
        r_script += "\nleg = c(leg, '%s')" % names_states[i]
    r_script += "\nlegend('topleft', bty = 'n', legend = leg, title = '%s', pch = rep(22, %s), pt.bg = adjustcolor(col, alpha = 0.2), pt.cex = 2, lty = 1, seg.len = 1, col = col)" % (names[1], n_states)
    return r_script


def plot_bdnn_inter_cont_cont(rs, tr, r_script, names, plot_time, obs, rate_type):
    # Interaction of two continuous features
    r_script += "\npar(mar = c(4, 4, 1.5, 5.1))"
    nr = np.sqrt(rs.shape[0])
    r_script += "\nzreord <- 1:%s" % nr
    r_script += "\nbyrow <- FALSE"
    if plot_time:
        r_script += "\nbyrow <- TRUE"
        names = [names[1], names[0]]
        obs = obs[:,[1, 0]]
        obs[:, 0] = -1 * obs[:, 0]
        tr = tr[:, [1, 0]]
        tr[:, 0] = -1 * tr[:, 0]
        r_script += "\nzreord <- rev(zreord)"
    if rate_type == 'speciation':
        r_script += "\ncol = colorRampPalette(c('lightblue1', rgb(0, 52, 94, maxColorValue = 255)))(%s)" % nr
    elif rate_type == 'extinction':
        r_script += "\ncol = colorRampPalette(c(rgb(255, 143, 118, maxColorValue = 255), 'darkred'))(%s)" % nr
    else:
        r_script += "\ncol = colorRampPalette(c('grey85', 'grey15'))(%s)" % nr
    r_script += util.print_R_vec("\nx", tr[:, 0])
    r_script += util.print_R_vec("\ny", tr[:, 1])
    r_script += util.print_R_vec("\nr", rs[:, 0])
    r_script += "\nxyr <- cbind(x, y, r)"
    r_script += "\nxaxis <- sort(unique(xyr[, 1]))"
    r_script += "\nyaxis <- sort(unique(xyr[, 2]))"
    r_script += "\nz <- matrix(xyr[, 3], length(xaxis), length(yaxis), byrow)"
    r_script += "\npadx <- abs(diff(xaxis))[1]"
    r_script += "\npady <- abs(diff(yaxis))[1]"
    r_script += "\nplot(mean(xaxis), mean(yaxis), type='n', xlim = c(min(xaxis) - padx, max(xaxis) + padx), ylim = c(min(yaxis) - pady, max(yaxis) + pady), xlab = '%s', ylab = '%s', xaxt = 'n', xaxs = 'i', yaxs= 'i')" % (names[0], names[1])
    r_script += "\nxtk <- pretty(xaxis, n = 10)"
    r_script += "\naxis(side = 1, at = xtk, labels = abs(xtk))"
    r_script += "\nimage.plot(xaxis, yaxis, z[zreord, ], add = TRUE, col = col)"
    r_script += "\ncontour(xaxis, yaxis, z[zreord, ], col = 'grey50', add = TRUE)"
    r_script += util.print_R_vec("\nobs_x", obs[:, 0])
    r_script += util.print_R_vec("\nobs_y", obs[:, 1])
    r_script += "\npoints(obs_x, obs_y, cex = 0.5, pch = 19, col = 'grey50')"
    r_script += "\npar(las = 1, mar = c(4, 4, 1.5, 0.5))"
    return r_script


def plot_bdnn_inter_discr_discr(rs, r, tr, r_script, feat_idx_1, feat_idx_2, names, names_states_feat_1, names_states_feat_2, rate_type):
    if len(feat_idx_1) > 1:
        # One-hot encoded discrete feature
        states_feat_1 = convert_onehot_to_numeric(tr[:, feat_idx_1])
    else:
        # Binary or ordinal features
        states_feat_1 = tr[:, feat_idx_1].flatten()
    n_states_feat_1 = len(np.unique(states_feat_1))
    if len(feat_idx_2) > 1:
        # One-hot encoded discrete feature
        states_feat_2 = convert_onehot_to_numeric(tr[:, feat_idx_2])
    else:
        # Binary or ordinal features
        states_feat_2 = tr[:, feat_idx_2].flatten()
    n_states_feat_2 = len(np.unique(states_feat_2))
    rate_max = np.nanmax(rs[:, 2])
    rate_min = np.nanmin(rs[:, 1])
    rate_max += 0.2 * rate_max
    rate_min -= 0.2 * np.abs(rate_min)
    r_script += "\npar(las = 2, mar = c(9, 4, 1.5, 0.5))"
    r_script += "\nylim = c(%s, %s)" % (rate_min, rate_max)
    r_script += "\nxlim = c(-0.5, %s + 0.5)" % (n_states_feat_1 * n_states_feat_2 - 1)
    r_script += "\nplot(0, 0, type = 'n', xlim = xlim, ylim = ylim, xlab = '', ylab = '%s', xaxt = 'n')" % (rate_type)
    if rate_type == 'speciation':
        r_script += "\ncol = colorRampPalette(c('lightblue1', rgb(0, 52, 94, maxColorValue = 255)))(%s)" % n_states_feat_2
    elif rate_type == 'extinction':
        r_script += "\ncol = colorRampPalette(c(rgb(255, 143, 118, maxColorValue = 255), 'darkred'))(%s)" % n_states_feat_2
    else:
        r_script += "\ncol = colorRampPalette(c('grey75', 'grey25'))(%s)" % n_states_feat_2
    counter = 0
    for i in range(n_states_feat_1):
        for j in range(n_states_feat_2):
            idx = np.logical_and(states_feat_1 == i, states_feat_2 == j)
            r_tmp = r[idx, :]
            r_tmp = r_tmp[r_tmp < rate_max]
            r_tmp = r_tmp[r_tmp > rate_min]
            if len(r_tmp) > 0:
                r_script += util.print_R_vec("\nvio_data", r_tmp)
                r_script += "\nvioplot(vio_data, at = %s, add = TRUE, wex = 0.5, rectCol = NA, lineCol = NA, colMed = NA, col = col[%s])" % (counter, j + 1)
                r_script += "\nlines(rep(%s, 2), c(%s, %s), lwd = 1.5)" % (counter, float(rs[idx, 1]), float(rs[idx, 2]))
                r_script += "\npoints(%s, %s, pch = 19)" % (counter, float(rs[idx, 0]))
                # r_script += "\nlines(rep(%s, 2), c(%s, %s), col = col[%s])" % (counter, float(rs[idx, 1]), float(rs[idx, 2]), j + 1)
                # r_script += "\npoints(%s, %s, pch = 19, col = col[%s])" % (counter, float(rs[idx, 0]), j + 1)
                # , padj = 0.5
            r_script += "\naxis(side = 1, at = %s, labels = paste0('%s', ': ', '%s', '\n', '%s', ': ', '%s'))" % (counter, names[0], names_states_feat_1[i], names[1], names_states_feat_2[j])
            counter += 1
    r_script += "\ntitle(main = paste0('%s', ' x ', '%s'))" % (names[0], names[1])
    r_script += "\npar(las = 1, mar = c(4, 4, 1.5, 0.5))"
    return r_script


def get_feat_idx(names_features, names, incl_feat):
    b = np.arange(len(incl_feat))
    n = names_features[incl_feat]
    f1 = np.where(n == names[0])[0]
    f2 = np.where(n == names[1])[0]
    f1_idx = b[f1]
    f2_idx = b[f2]
    return f1_idx, f2_idx


def create_R_files_effects(cond_trait_tbl, cond_rates, bdnn_obj, sp_fad_lad, r_script, names_features, backscale_par,
                           rate_type = 'speciation'):
    r_script += "\npkgs = c('fields', 'vioplot')"
    r_script += "\nnew_pkgs = pkgs[!(pkgs %in% installed.packages()[,'Package'])]"
    r_script += "\nif (length(new_pkgs) > 0) {"
    r_script += "\n    install.packages(new_pkgs, repos = 'https://cran.rstudio.com/')"
    r_script += "\n}"
    r_script += "\nsuppressPackageStartupMessages(library(fields))"
    r_script += "\nsuppressPackageStartupMessages(library(vioplot))"
    r_script += "\npar(las = 1, mar = c(4, 4, 1.5, 0.5))"
    plot_idx = np.unique(cond_trait_tbl[:, -3])
    n_plots = len(plot_idx)
    rates_summary = get_rates_summary(cond_rates)
    # set summary to NA when we have not observed the combination/range of features
    not_obs = cond_trait_tbl[:, -1] == 0
    rates_summary[not_obs,:] = np.nan
    cond_rates[not_obs,:] = np.nan
    time_idx = np.nanmax(cond_trait_tbl[:, -3]) + 10.0
    if is_time_trait(bdnn_obj):
        time_idx = np.max(cond_trait_tbl[:, -6])
    rate_type2 = rate_type
    if rate_type == 'net diversification':
        rate_type2 = 'extinction'
    trait_tbl = get_trt_tbl(bdnn_obj, rate_type2)
    feature_is_time_variable = is_time_variable_feature(trait_tbl)
    fad_lad = sp_fad_lad[['FAD', 'LAD']].to_numpy()
    fossil_age = get_fossil_age(bdnn_obj, fad_lad, rate_type2)
    fossil_age = backscale_bdnn_time(fossil_age, bdnn_obj)
    fossil_bin = get_bin_from_fossil_age(bdnn_obj, fad_lad, rate_type2)
    names_features_original = np.array(get_names_features(bdnn_obj))
    binary_feature = is_binary_feature(trait_tbl)[0]
    for i in range(n_plots):
        idx = cond_trait_tbl[:, -3] == plot_idx[i]
        trait_tbl_plt = cond_trait_tbl[idx, :]
        pt = trait_tbl_plt[0, -4]
        trait_tbl_plt, incl_features = remove_conditional_features(trait_tbl_plt)
        names = names_features[incl_features]
        rates_sum_plt = rates_summary[idx, :]
        cond_rates_plt = cond_rates[idx,:]
        obs = get_observed(bdnn_obj, incl_features, feature_is_time_variable, fossil_age, fossil_bin, rate_type2)
        plot_time = np.isin(time_idx, incl_features)
        if np.isin(pt, np.array([1.0, 2.0, 3.0])):
            names = names_features[incl_features[0]]
            names_states = trait_tbl_plt[:, 0].tolist()
            if pt == 2.0:
                names_states = names_features_original[incl_features].tolist()
            r_script = plot_bdnn_discr(rates_sum_plt, cond_rates_plt, trait_tbl_plt, r_script, names, names_states, rate_type)
        elif pt == 4.0:
            names = names_features[incl_features[0]]
            obs = backscale_tbl(bdnn_obj, backscale_par, [names], obs)
            r_script = plot_bdnn_cont(rates_sum_plt, trait_tbl_plt, r_script, names, plot_time, obs, rate_type)
        elif np.isin(pt, np.array([6.0, 13.0, 14.0])):
            b = binary_feature[incl_features]
            names_states = np.unique(trait_tbl_plt[:, b]).tolist()
            if pt == 13.0:
                names_states = names_features_original[incl_features][b]
            names = names[np.argsort(b)]
            trait_tbl_plt = trait_tbl_plt[:, np.argsort(b)] # Continuous feature always in column 0
            obs = obs[:, np.argsort(b)]
            obs[:,0] = backscale_tbl(bdnn_obj, backscale_par, [names[0]], obs[:,0].reshape((obs.shape[0],1))).flatten()
            r_script = plot_bdnn_inter_discr_cont(rates_sum_plt, trait_tbl_plt, r_script, names, names_states, plot_time, obs, rate_type)
        elif pt == 7.0:
            obs = backscale_tbl(bdnn_obj, backscale_par, names.tolist(), obs)
            r_script = plot_bdnn_inter_cont_cont(rates_sum_plt, trait_tbl_plt, r_script, names, plot_time, obs, rate_type)
        elif np.isin(pt, np.array([5.0, 8.0, 9.0, 10.0, 11.00, 12.0])):
            if np.isin(pt, np.array([5.0, 9.0, 12.0])):
                names = names_features[incl_features]
                feat_1 = np.array([0])
                feat_2 = np.array([1])
                names_states_feat_1 = np.unique(trait_tbl_plt[:, feat_1]).tolist()
                names_states_feat_2 = np.unique(trait_tbl_plt[:, feat_2]).tolist()
            if np.isin(pt, np.array([8.0, 10.0, 11.0])):
                names = np.unique(names_features[incl_features])
                feat_1, feat_2 = get_feat_idx(names_features, names, incl_features)
                if len(feat_1) > 1:
                    names_states_feat_1 = names_features_original[incl_features][feat_1]
                else:
                    names_states_feat_1 = np.unique(trait_tbl_plt[:, feat_1]).tolist()
                if len(feat_2) > 1:
                    names_states_feat_2 = names_features_original[incl_features][feat_2]
                else:
                    names_states_feat_2 = np.unique(trait_tbl_plt[:, feat_2]).tolist()
            r_script = plot_bdnn_inter_discr_discr(rates_sum_plt, cond_rates_plt, trait_tbl_plt, r_script, feat_1, feat_2, names, names_states_feat_1, names_states_feat_2, rate_type)
    return r_script


def get_feat_to_keep_for_netdiv(cond_trait_tbl_sp, cond_trait_tbl_ex):
    feat_sp = np.unique(cond_trait_tbl_sp[ :, -6])
    feat_ex = np.unique(cond_trait_tbl_ex[:, -6])
    feat_in_any = np.intersect1d(feat_sp, feat_ex).astype(int)
    l = len(feat_in_any)
    all_values_equal = np.zeros(l, dtype = bool)
    if l > 0:
        for i in range(l):
            ii = feat_in_any[i]
            # Check if values for feature i are the same in both cond_trait_tbls
            idx_feat_i_sp = np.logical_and(cond_trait_tbl_sp[ :, -6] == ii, np.isnan(cond_trait_tbl_sp[ :, -5]))
            idx_feat_i_ex = np.logical_and(cond_trait_tbl_ex[:, -6] == ii, np.isnan(cond_trait_tbl_ex[:, -5]))
            if np.sum(idx_feat_i_sp) == np.sum(idx_feat_i_ex):
                all_values_equal[i] = np.all(cond_trait_tbl_sp[idx_feat_i_sp, ii] == cond_trait_tbl_ex[idx_feat_i_ex, ii])
    feat_to_keep = feat_in_any[all_values_equal]
    return feat_to_keep


def get_rates_cond_trait_tbl_for_netdiv(feat_to_keep, sp_rate_cond, ex_rate_cond, cond_trait_tbl_ex, cond_trait_tbl_sp):
    keep_feat1 = np.isin(cond_trait_tbl_ex[ :, -6], feat_to_keep)
    keep_feat2 = np.isin(cond_trait_tbl_ex[ :, -5], feat_to_keep)
    keep_feat2[np.isnan(cond_trait_tbl_ex[ :, -5])] = True
    keep = np.logical_and(keep_feat1, keep_feat2)
    cond_trait_tbl_ex2 = cond_trait_tbl_ex[keep, :]
    ex_rate_cond2 = ex_rate_cond[keep, :]
    keep_feat1 = np.isin(cond_trait_tbl_sp[ :, -6], feat_to_keep)
    keep_feat2 = np.isin(cond_trait_tbl_sp[ :, -5], feat_to_keep)
    keep_feat2[np.isnan(cond_trait_tbl_sp[ :, -5])] = True
    keep = np.logical_and(keep_feat1, keep_feat2)
    cond_trait_tbl_sp2 = cond_trait_tbl_sp[keep, :]
    sp_rate_cond2 = sp_rate_cond[keep, :]
    netdiv = sp_rate_cond2 - ex_rate_cond2
    # Restrict range of values that should be plotted
    range_to_plot = cond_trait_tbl_sp2[ :, -1] + cond_trait_tbl_ex2[ :, -1]
    cond_trait_tbl_ex2[range_to_plot == 0, -1] = 0
    cond_trait_tbl_ex2[range_to_plot > 0, -1] = 1
    return netdiv, cond_trait_tbl_ex2


def get_effect_objects(mcmc_file, pkl_file, burnin, thin, combine_discr_features = "", file_transf_features = "", num_processes = 1, show_progressbar = False):
    bdnn_obj, post_w_sp, post_w_ex, sp_fad_lad, post_ts, post_te = bdnn_parse_results(mcmc_file, pkl_file, burnin, thin)
    cond_trait_tbl_sp, names_features_sp = build_conditional_trait_tbl(bdnn_obj, sp_fad_lad,
                                                                       post_ts, post_te,
                                                                       len_cont = 100,
                                                                       rate_type = "speciation",
                                                                       combine_discr_features = combine_discr_features)
    cond_trait_tbl_ex, names_features_ex = build_conditional_trait_tbl(bdnn_obj, sp_fad_lad,
                                                                       post_ts, post_te,
                                                                       len_cont = 100,
                                                                       rate_type = "extinction",
                                                                       combine_discr_features = combine_discr_features)
    #sp_rate_cond = bdnn_lib.get_conditional_rates(bdnn_obj, cond_trait_tbl_sp, post_w_sp)
    #ex_rate_cond = bdnn_lib.get_conditional_rates(bdnn_obj, cond_trait_tbl_ex, post_w_ex)
    bdnn_time = get_bdnn_time(bdnn_obj, np.mean(post_ts, axis = 0))
    print("Getting partial dependence rates for speciation")
    sp_rate_cond = get_partial_dependence_rates(bdnn_obj, bdnn_time, cond_trait_tbl_sp, post_w_sp, post_ts,
                                                rate_type = 'speciation', num_processes = num_processes, show_progressbar = show_progressbar)
    print("Getting partial dependence rates for extinction")
    ex_rate_cond = get_partial_dependence_rates(bdnn_obj, bdnn_time, cond_trait_tbl_ex, post_w_ex, post_te,
                                                rate_type = 'extinction', num_processes = num_processes, show_progressbar = show_progressbar)
    cond_trait_tbl_sp, cond_trait_tbl_ex, backscale_par = backscale_bdnn_features(file_transf_features,
                                                                                  bdnn_obj,
                                                                                  cond_trait_tbl_sp,
                                                                                  cond_trait_tbl_ex)
    return bdnn_obj, cond_trait_tbl_sp, cond_trait_tbl_ex, names_features_sp, names_features_ex, sp_rate_cond, ex_rate_cond, sp_fad_lad, backscale_par


def plot_effects(f,
                 cond_trait_tbl_sp,
                 cond_trait_tbl_ex,
                 sp_rate_cond,
                 ex_rate_cond,
                 bdnn_obj,
                 sp_fad_lad,
                 backscale_par,
                 names_features_sp,
                 names_features_ex,
                 suffix_pdf = "effects"):
    # Plot feature-rate relationship
    output_wd = os.path.dirname(f)
    name_file = os.path.basename(f)
    out = "%s/%s_%s.r" % (output_wd, name_file, suffix_pdf)
    newfile = open(out, "w")
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        wd_forward = os.path.abspath(output_wd).replace('\\', '/')
        r_script = "pdf(file='%s/%s_%s.pdf', width = 7, height = 6, useDingbats = FALSE)\n" % (wd_forward, name_file, suffix_pdf)
    else:
        r_script = "pdf(file='%s/%s_%s.pdf', width = 7, height = 6, useDingbats = FALSE)\n" % (output_wd, name_file, suffix_pdf)
    r_script = create_R_files_effects(cond_trait_tbl_sp, sp_rate_cond + 0.0, bdnn_obj, sp_fad_lad, r_script, names_features_sp,
                                      backscale_par, rate_type = 'speciation')
    r_script = create_R_files_effects(cond_trait_tbl_ex, ex_rate_cond + 0.0, bdnn_obj, sp_fad_lad, r_script, names_features_ex,
                                      backscale_par, rate_type = 'extinction')
    feat_to_keep = get_feat_to_keep_for_netdiv(cond_trait_tbl_sp, cond_trait_tbl_ex)
    netdiv_rate_cond, cond_trait_tbl_netdiv = get_rates_cond_trait_tbl_for_netdiv(feat_to_keep,
                                                                                  sp_rate_cond,
                                                                                  ex_rate_cond,
                                                                                  cond_trait_tbl_ex,
                                                                                  cond_trait_tbl_sp)
    r_script = create_R_files_effects(cond_trait_tbl_netdiv, netdiv_rate_cond, bdnn_obj, sp_fad_lad, r_script, names_features_ex,
                                      backscale_par, rate_type = 'net diversification')
    r_script += "\ndev.off()"
    newfile.writelines(r_script)
    newfile.close()
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        cmd = "cd %s & Rscript %s_%s.r" % (output_wd, name_file, suffix_pdf)
    else:
        cmd = "cd %s; Rscript %s_%s.r" % (output_wd, name_file, suffix_pdf)
        print("cmd", cmd)
        os.system(cmd)


def get_prob_1_bin_trait(cond_rates_eff):
    d1 = cond_rates_eff[0, :]
    d2 = cond_rates_eff[1, :]
    prob = get_prob(d1, d2, len(d1))
    mag = d1 / d2
    mean_mag = np.mean(mag)
    mag_HPD = util.calcHPD(mag, .95)
    return np.array([prob, mean_mag, mag_HPD[0], mag_HPD[1]])


def get_prob_1_con_trait(cond_rates_eff):
    mean_rate = np.mean(cond_rates_eff, axis = 1)
    d1 = cond_rates_eff[np.argmax(mean_rate), :]
    d2 = cond_rates_eff[np.argmin(mean_rate), :]
    d = d1 - d2
    d = d.flatten()
    n = np.sum(d > 0.0)
    prob = n / len(d)
    mag = d1 / d2
    mean_mag = np.mean(mag)
    mag_HPD = util.calcHPD(mag, .95)
    return np.array([prob, mean_mag, mag_HPD[0], mag_HPD[1]])


def get_prob_inter_bin_con_trait(rates_eff, state0):
    # interaction binary feature with continuous
    cond_rates_state0 = rates_eff[state0, :]
    cond_rates_state1 = rates_eff[~state0, :]
    state0_cont_obs = ~np.isnan(cond_rates_state0[:, 0])
    state1_cont_obs = ~np.isnan(cond_rates_state1[:, 0])
    overlap_across_cont = np.logical_and(state0_cont_obs, state1_cont_obs)
    prob = np.nan
    mean_mag = np.nan
    mag_HPD = np.array([np.nan, np.nan])
    # This will only work if ranges of the continuous trait are not overlapping for the two states
    if np.any(overlap_across_cont):
        cond_rates_state0 = cond_rates_state0[overlap_across_cont, :]
        cond_rates_state1 = cond_rates_state1[overlap_across_cont, :]
        diff_state = cond_rates_state0 - cond_rates_state1
        mean_diff_state = np.mean(diff_state, axis = 1)
        d1 = diff_state[np.argmax(mean_diff_state), :]
        d2 = diff_state[np.argmin(mean_diff_state), :]
        prob = get_prob(d1, d2, len(d1))
        mag = cond_rates_state0[np.argmax(mean_diff_state), :] / cond_rates_state1[np.argmin(mean_diff_state), :]
        mean_mag = np.mean(mag)
        mag_HPD = util.calcHPD(mag, .95)
    return np.array([prob, mean_mag, mag_HPD[0], mag_HPD[1]])


def get_prob_discr_ord(cond_rates_eff, names, names_states):
    # discrete and ordinal features
    n = len(names_states) #cond_rates_eff.shape[0]
    all_combinations = list(combinations(np.arange(n), 2))
    n_comb = len(all_combinations)
    p = np.zeros((n_comb, 8))
    p[:] = np.nan
    p_df = pd.DataFrame(p)
    for j in range(n_comb):
        l = list(all_combinations[j])
        p_df.loc[j, 0] = names
        p_df.loc[j, 1] = 'none'
        p_df.loc[j, 2] = str(names_states[l[0]]) + '_' + str(names_states[l[1]])
        p_df.loc[j, 3] = 'none'
        p_df.loc[j, 4:] = get_prob_1_bin_trait(cond_rates_eff[l,:])
    p_df.columns = ['0', '1', '2', '3', '4', '5', '6', '7']
    return p_df


def get_prob_inter_discr_discr(cond_rates_eff, tr, feat_idx_1, feat_idx_2,
                               names, names_states_feat_1, names_states_feat_2):
    p_df = pd.DataFrame()
    tr = tr.astype(int)
    len_feat_idx_1 = len(feat_idx_1)
    len_feat_idx_2 = len(feat_idx_2)
    if len_feat_idx_1 > 1:
        # One-hot encoded discrete feature
        states_feat_1 = np.unique(convert_onehot_to_numeric(tr[:, feat_idx_1]))
    else:
        # Binary or ordinal features
        states_feat_1 = np.unique(tr[:, feat_idx_1])
    if len_feat_idx_2 > 1:
        # One-hot encoded discrete feature
        states_feat_2 = np.unique(convert_onehot_to_numeric(tr[:, feat_idx_2]))
    else:
        # Binary or ordinal features
        states_feat_2 = np.unique(tr[:, feat_idx_2])
    comb_states_feat_1 = np.array(list(combinations(states_feat_1, 2))).astype(int)
    comb_states_feat_2 = np.array(list(combinations(states_feat_2, 2))).astype(int)
    num_comb_feat_1 = len(comb_states_feat_1)
    num_comb_feat_2 = len(comb_states_feat_2)
    for j in range(num_comb_feat_1):
        for k in range(num_comb_feat_2):
            if len_feat_idx_1 == 1:
                idx1 = tr[:, feat_idx_1] == comb_states_feat_1[j, 0]
            else:
                idx1 = tr[:, feat_idx_1[comb_states_feat_1[j, 0]]] > 0
            if len_feat_idx_2 == 1:
                idx2 = np.isin(tr[:, feat_idx_2], comb_states_feat_2[k, :])
            else:
                idx2 = np.sum(tr[:, feat_idx_2[comb_states_feat_2[k, :]]], axis = 1) > 0
            idx1 = idx1.flatten()
            idx2 = idx2.flatten()
            idx = np.logical_and(idx1, idx2)
            rate_diff_1 = np.diff(cond_rates_eff[idx, :], axis = 0)
            if len_feat_idx_1 == 1:
                idx1 = tr[:, feat_idx_1] == comb_states_feat_1[j, 1]
            else:
                idx1 = tr[:, feat_idx_1[comb_states_feat_1[j, 1]]] > 0
            idx1 = idx1.flatten()
            idx = np.logical_and(idx1, idx2)
            rate_diff_2 = np.diff(cond_rates_eff[idx, :], axis = 0)
            prob = get_prob_1_bin_trait(np.vstack([rate_diff_1, rate_diff_2]))
            pjk = pd.DataFrame({'0': names[0],
                                '1': names[1],
                                '2': str(names_states_feat_1[comb_states_feat_1[j, 0]]) + '_' + str(names_states_feat_1[comb_states_feat_1[j, 1]]),
                                '3': str(names_states_feat_2[comb_states_feat_2[k, 0]]) + '_' + str(names_states_feat_2[comb_states_feat_2[k, 1]]),
                                '4': prob[0], '5': prob[1], '6': prob[2], '7': prob[3]},
                                index = [0])
            p_df = pd.concat([p_df, pjk], ignore_index = True)
    return p_df





def get_prob_inter_cont_discr_ord(cond_rates_eff, trait_tbl_eff, names_cont, names_discr_ord, names_states):
    # interaction between discr/ordinal features with a continuous feature
    n = len(names_states)
    all_combinations = list(combinations(np.arange(n), 2))
    n_comb = len(all_combinations)
    p = np.zeros((n_comb, 8))
    p[:] = np.nan
    p_df = pd.DataFrame(p)
    for j in range(n_comb):
        l = list(all_combinations[j])
        if trait_tbl_eff.shape[1] > 2:
            # one-hot encoding
            state0 = trait_tbl_eff[:,l[0] + 1] == 1
            state1 = trait_tbl_eff[:, l[1] + 1] == 1
        else:
            # ordinal
            state0 = trait_tbl_eff[:, 1] == l[0]
            state1 = trait_tbl_eff[:, 1] == l[1]
        focal_states = np.logical_or(state0, state1)
        cond_rates_eff_j = cond_rates_eff[focal_states,:]
        trait_tbl_j = trait_tbl_eff[focal_states,:]
        if trait_tbl_eff.shape[1] > 2:
            state0 = trait_tbl_j[:, l[0] + 1] == 1
        else:
            state0 = trait_tbl_j[:, 1] == l[0]
        p_df.loc[j, 0] = names_cont
        p_df.loc[j, 1] = names_discr_ord
        p_df.loc[j, 2] = 'none'
        p_df.loc[j, 3] = str(names_states[l[0]]) + '_' + str(names_states[l[1]])
        p_df.loc[j, 4:] = get_prob_inter_bin_con_trait(cond_rates_eff_j, state0)
    p_df.columns = ['0', '1', '2', '3', '4', '5', '6', '7']
    return p_df


def get_prob(d1, d2, niter_mcmc):
    prob = np.nan
    d = d1 - d2
    if np.any(np.isnan(d) == False):
        d = d.flatten()
        n = max([np.sum(d < 0.0), np.sum(d > 0.0)])
        prob = n / niter_mcmc
    return prob


def get_prob_inter_2_con_trait(cond_rates_eff, trait_tbl_eff, incl_features, cond_rates, cond_trait_tbl):
    # Individual effects
    feat0_idx = np.logical_and(cond_trait_tbl[:, -6] == incl_features[0], np.isnan(cond_trait_tbl[:, -5]))
    rates_feat0 = cond_rates[feat0_idx, :]
    feat0 = cond_trait_tbl[feat0_idx, incl_features[0]]
    feat1_idx = np.logical_and(cond_trait_tbl[:, -6] == incl_features[1], np.isnan(cond_trait_tbl[:, -5]))
    rates_feat1 = cond_rates[feat1_idx, :]
    feat1 = cond_trait_tbl[feat1_idx, incl_features[1]]
    # Interaction
    niter_mcmc = cond_rates_eff.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category = RuntimeWarning)
        mean_rate = np.nanmean(cond_rates_eff, axis = 1)
    idx_min_rate = np.nanargmin(mean_rate)
    idx_max_rate = np.nanargmax(mean_rate)
    feat_at_min_rate = trait_tbl_eff[idx_min_rate, :]
    feat_at_max_rate = trait_tbl_eff[idx_max_rate, :]
    # Get main effects at the feature values
    feat0_idx = [int(np.where(feat_at_min_rate[0] == feat0)[0]), int(np.where(feat_at_max_rate[0] == feat0)[0])]
    feat1_idx = [int(np.where(feat_at_min_rate[1] == feat1)[0]), int(np.where(feat_at_max_rate[1] == feat1)[0])]
    rates_feat0 = rates_feat0[feat0_idx, :]
    rates_feat1 = rates_feat1[feat1_idx, :]
    prob_feat0 = 0.0
    prob_feat1 = 0.0
    if rates_feat0.shape[0] > 1:
        prob_feat0 = get_prob(rates_feat0[0, :], rates_feat0[1, :], niter_mcmc)
    if rates_feat1.shape[0] > 1:
        prob_feat1 = get_prob(rates_feat1[0, :], rates_feat1[1, :], niter_mcmc)
    rates_single_feat = rates_feat0
    if prob_feat1 > prob_feat0:
        rates_single_feat = rates_feat1
    mag_single_feat = rates_single_feat[0, :] / rates_single_feat[1, :]
    d1 = cond_rates_eff[idx_min_rate, :]
    d2 = cond_rates_eff[idx_max_rate, :]
    #prob = get_prob(d1, d2, niter_mcmc)
    mag = d1 / d2
    mean_mag = np.mean(mag)
    mag_HPD = util.calcHPD(mag, .95)
    # Magnitude interaction greater than for the more important feature of both?
    if np.mean(mag) > 1:
        prob = np.sum(mag > mag_single_feat) / niter_mcmc
    else:
        prob = np.sum(mag < mag_single_feat) / niter_mcmc
    return np.array([prob, mean_mag, mag_HPD[0], mag_HPD[1]])


def get_prob_effects(cond_trait_tbl, cond_rates, bdnn_obj, names_features, rate_type = 'speciation'):
    names_features_original = np.array(get_names_features(bdnn_obj))
    not_obs = cond_trait_tbl[:, -1] == 0
    cond_rates[not_obs, :] = np.nan
    prob_effects = pd.DataFrame()
    trait_tbl = get_trt_tbl(bdnn_obj, rate_type)
    binary_feature = is_binary_feature(trait_tbl)[0]
    plot_idx = np.unique(cond_trait_tbl[:, -3])
    n_plots = len(plot_idx)
    for i in range(n_plots):
        idx = cond_trait_tbl[:, -3] == plot_idx[i]
        trait_tbl_eff = cond_trait_tbl[idx, :]
        cond_rates_eff = cond_rates[idx, :]
        pt = trait_tbl_eff[0, -4]
        trait_tbl_eff, incl_features = remove_conditional_features(trait_tbl_eff)
        names = names_features[incl_features]
        if pt == 1.0:
            names_states = trait_tbl_eff[:, 0].astype(int).tolist()
            prob = get_prob_1_bin_trait(cond_rates_eff)
            prob = pd.DataFrame({'0': names[0],
                                 '1': 'none',
                                 '2': str(names_states[0]) + '_' + str(names_states[1]),
                                 '3': 'none',
                                 '4': prob[0], '5': prob[1], '6': prob[2], '7': prob[3]},
                                index = [0])
            prob_effects = pd.concat([prob_effects, prob], ignore_index = True)
        elif np.isin(pt, np.array([2.0, 3.0])):
            names = names_features[incl_features[0]]
            names_states = trait_tbl_eff[:, 0].astype(int).tolist()
            if pt == 2.0:
                names_states = names_features_original[incl_features].tolist()
            prob = get_prob_discr_ord(cond_rates_eff, names, names_states)
            prob_effects = pd.concat([prob_effects, prob], ignore_index = True)
        elif pt == 4.0:
            prob = get_prob_1_con_trait(cond_rates_eff)
            prob = pd.DataFrame({'0': names[0], '1': 'none', '2': 'none', '3': 'none',
                                 '4': prob[0], '5': prob[1], '6': prob[2], '7': prob[3]},
                                index=[0])
            prob_effects = pd.concat([prob_effects, prob], ignore_index = True)
        elif pt == 6.0:
            b = is_binary_feature(np.array([trait_tbl_eff]))[0]
            state0 = trait_tbl_eff[:, b] == 0
            state0 = state0.flatten()
            prob = get_prob_inter_bin_con_trait(cond_rates_eff, state0)
            prob = pd.DataFrame({'0': names[0], '1': names[1], '2': 'none', '3': 'none',
                                 '4': prob[0], '5': prob[1], '6': prob[2], '7': prob[3]},
                                index=[0])
            prob_effects = pd.concat([prob_effects, prob], ignore_index = True)
        elif pt == 7.0:
            prob = get_prob_inter_2_con_trait(cond_rates_eff, trait_tbl_eff, incl_features, cond_rates, cond_trait_tbl)
            prob = pd.DataFrame({'0': names[0], '1': names[1], '2': 'none', '3': 'none',
                                 '4': prob[0], '5': prob[1], '6': prob[2], '7': prob[3]},
                                index=[0])
            prob_effects = pd.concat([prob_effects, prob], ignore_index = True)
        elif np.isin(pt, np.array([5.0, 9.0, 12.0])):
            feat_1 = np.array([0])
            feat_2 = np.array([1])
            names_states_feat_1 = np.unique(trait_tbl_eff[:, feat_1]).tolist()
            names_states_feat_2 = np.unique(trait_tbl_eff[:, feat_2]).tolist()
            prob = get_prob_inter_discr_discr(cond_rates_eff, trait_tbl_eff, feat_1, feat_2,
                                              names, names_states_feat_1, names_states_feat_2)
            prob_effects = pd.concat([prob_effects, prob], ignore_index = True)
        elif np.isin(pt, np.array([8.0, 10.0, 11.0])):
            names = np.unique(names_features[incl_features])
            feat_1, feat_2 = get_feat_idx(names_features, names, incl_features)
            if len(feat_1) > 1:
                names_states_feat_1 = names_features_original[incl_features][feat_1]
            else:
                names_states_feat_1 = np.unique(trait_tbl_eff[:, feat_1]).tolist()
            if len(feat_2) > 1:
                names_states_feat_2 = names_features_original[incl_features][feat_2]
            else:
                names_states_feat_2 = np.unique(trait_tbl_eff[:, feat_2]).tolist()
            prob = get_prob_inter_discr_discr(cond_rates_eff, trait_tbl_eff, feat_1, feat_2,
                                              names, names_states_feat_1, names_states_feat_2)
            prob_effects = pd.concat([prob_effects, prob], ignore_index = True)
        elif np.isin(pt, np.array([13.0, 14.0])):
            b = binary_feature[incl_features]
            names_states = np.unique(trait_tbl_eff[:, b]).tolist()
            if pt == 13.0:
                names_states = names_features_original[incl_features][b]
            names = names[np.argsort(b)]
            trait_tbl_eff = trait_tbl_eff[:, np.argsort(b)]  # Continuous feature always in column 0
            prob = get_prob_inter_cont_discr_ord(cond_rates_eff, trait_tbl_eff, names[0], names[1], names_states)
            prob_effects = pd.concat([prob_effects, prob], ignore_index = True)
    prob_effects.columns = ['feature1', 'feature2', 'feature1_state', 'feature2_state',
                            'posterior_probability',
                            'magnitude_effect', 'magnitude_lwr_CI', 'magnitude_upr_CI']
    return prob_effects


def get_shap_trt_tbl(tse, times, trt_tbl):
    if trt_tbl.ndim == 2:
        shap_trt_tbl = trt_tbl + 0.0
    else:
        s = trt_tbl.shape
        n_species = s[1]
        n_features = s[2]
        digitized_tse = np.digitize(tse, times) - 1
        digitized_tse[digitized_tse < 0] = 0 # Should not do any harm for speciation
        shap_trt_tbl = np.zeros((n_species, n_features))
        for i in range(n_species):
            shap_trt_tbl[i,:] = trt_tbl[digitized_tse[i], i, :]
    return shap_trt_tbl


def take_traits_from_trt_tbl(trait_tbl, cond_trait_tbl, j):
    trait_tbl_tmp = trait_tbl + 0.0
    idx_feat1 = int(cond_trait_tbl[j, -6])
    idx_feat2 = cond_trait_tbl[j, -5]
    trait_tbl_tmp[: , idx_feat1] = cond_trait_tbl[j, idx_feat1]
    if not np.isnan(idx_feat2):
        idx_feat2 = int(idx_feat2)
        trait_tbl_tmp[:, idx_feat2] = cond_trait_tbl[j, idx_feat2]
    return trait_tbl_tmp


def get_pdp_rate_it_i(arg):
    [bdnn_obj, post_w_i, trait_tbl, cond_trait_tbl] = arg
    nrows_cond_trait_tbl = len(cond_trait_tbl)
    rate_it_i = np.zeros(nrows_cond_trait_tbl)
    rate_it_i[:] = np.nan
    obs = cond_trait_tbl[:, -1] == 1
    for j in range(nrows_cond_trait_tbl):
        if obs[j]:
            trait_tbl_tmp = take_traits_from_trt_tbl(trait_tbl, cond_trait_tbl, j)
            rate_BDNN = get_rate_BDNN(1,  # constant baseline
                                      trait_tbl_tmp,
                                      post_w_i,  # list of arrays
                                      bdnn_obj.bdnn_settings['hidden_act_f'],
                                      bdnn_obj.bdnn_settings['out_act_f'])
            rate_it_i[j] = np.mean(rate_BDNN)
    return rate_it_i


def get_partial_dependence_rates(bdnn_obj, bdnn_time, cond_trait_tbl, post_w, post_tse,
                                 rate_type = 'speciation', num_processes = 1, show_progressbar = False):
    num_it = len(post_w)
    trait_tbl = get_trt_tbl(bdnn_obj, rate_type)
    args = []
    for i in range(num_it):
        trait_tbl_a = trait_tbl + 0.0
        trait_tbl_a = get_shap_trt_tbl(post_tse[i, :], bdnn_time, trait_tbl_a)
        a = [bdnn_obj, post_w[i], trait_tbl_a, cond_trait_tbl]
        args.append(a)
    unixos = is_unix()
    if unixos and num_processes > 1:
        pool_perm = multiprocessing.Pool(num_processes)
        rate_pdp = list(tqdm(pool_perm.imap_unordered(get_pdp_rate_it_i, args),
                             total = num_it, disable = show_progressbar == False))
        pool_perm.close()
    else:
        rate_pdp = []
        for i in tqdm(range(num_it), disable = show_progressbar == False):
            rate_pdp.append(get_pdp_rate_it_i(args[i]))
    rate_pdp = np.stack(rate_pdp, axis = 1)
    return rate_pdp


def get_greenwells_feature_importance(cond_trait_tbl, pdp_rates, bdnn_obj, names_features, rate_type = 'speciation'):
    names_features_original = np.array(get_names_features(bdnn_obj))
    not_obs = cond_trait_tbl[:, -1] == 0
    pdp_rates[not_obs, :] = np.nan
    green_imp = pd.DataFrame()
    trait_tbl = get_trt_tbl(bdnn_obj, rate_type)
    binary_feature = is_binary_feature(trait_tbl)[0]
    plot_idx = np.unique(cond_trait_tbl[:, -3])
    n_plots = len(plot_idx)
    for i in range(n_plots):
        idx = cond_trait_tbl[:, -3] == plot_idx[i]
        trait_tbl_eff = cond_trait_tbl[idx, :]
        pdp_rates_eff = pdp_rates[idx, :]
        pt = trait_tbl_eff[0, -4]
        trait_tbl_eff, incl_features = remove_conditional_features(trait_tbl_eff)
        names = names_features[incl_features]
        if pt == 1.0:
            # one binary feature
            names_states = trait_tbl_eff[:, 0].astype(int).tolist()
            imp_sc = np.ptp(pdp_rates_eff, axis = 0) / 4.0
            gii = np.zeros(3)
            gii[0] = np.mean(imp_sc)
            gii[1:] = util.calcHPD(imp_sc, 0.95)
            gii = pd.DataFrame({'0': names[0],
                                '1': 'none',
                                '2': str(names_states[0]) + '_' + str(names_states[1]),
                                '3': 'none',
                                '4': gii[0], '5': gii[1], '6': gii[2]},
                               index=[0])
            green_imp = pd.concat([green_imp, gii], ignore_index = True)
        # elif np.isin(pt, np.array([2.0, 3.0])):
        #     names = names_features[incl_features[0]]
        #     names_states = trait_tbl_eff[:, 0].astype(int).tolist()
        #     if pt == 2.0:
        #         names_states = names_features_original[incl_features].tolist()
        #     prob = get_prob_discr_ord(pdp_rates_eff, names, names_states)
        #     prob_effects = pd.concat([prob_effects, prob], ignore_index=True)
        elif pt == 4.0:
            # one continuous feature
            imp_sc = np.nanstd(pdp_rates_eff, axis = 0)
            gii = np.zeros(3)
            gii[0] = np.mean(imp_sc)
            gii[1:] = util.calcHPD(imp_sc, 0.95)
            gii = pd.DataFrame({'0': names[0], '1': 'none', '2': 'none', '3': 'none',
                                '4': gii[0], '5': gii[1], '6': gii[2]},
                               index=[0])
            green_imp = pd.concat([green_imp, gii], ignore_index=True)
        elif pt == 6.0:
            # binary x continuous feature
            gii = get_greenwells_interaction_importance(pdp_rates_eff, trait_tbl_eff)
            gii = pd.DataFrame({'0': names[0], '1': names[1], '2': 'none', '3': 'none',
                                '4': gii[0], '5': gii[1], '6': gii[2]},
                               index=[0])
            green_imp = pd.concat([green_imp, gii], ignore_index = True)
        elif pt == 7.0:
            # continuous x continuous feature
            gii = get_greenwells_interaction_importance(pdp_rates_eff, trait_tbl_eff)
            gii = pd.DataFrame({'0': names[0], '1': names[1], '2': 'none', '3': 'none',
                                '4': gii[0], '5': gii[1], '6': gii[2],},
                                index=[0])
            green_imp = pd.concat([green_imp, gii], ignore_index = True)
        # elif np.isin(pt, np.array([5.0, 9.0, 12.0])):
        #     feat_1 = np.array([0])
        #     feat_2 = np.array([1])
        #     names_states_feat_1 = np.unique(trait_tbl_eff[:, feat_1]).tolist()
        #     names_states_feat_2 = np.unique(trait_tbl_eff[:, feat_2]).tolist()
        #     prob = get_prob_inter_discr_discr(pdp_rates_eff, trait_tbl_eff, feat_1, feat_2,
        #                                       names, names_states_feat_1, names_states_feat_2)
        #     prob_effects = pd.concat([prob_effects, prob], ignore_index=True)
        # elif np.isin(pt, np.array([8.0, 10.0, 11.0])):
        #     names = np.unique(names_features[incl_features])
        #     feat_1, feat_2 = get_feat_idx(names_features, names, incl_features)
        #     if len(feat_1) > 1:
        #         names_states_feat_1 = names_features_original[incl_features][feat_1]
        #     else:
        #         names_states_feat_1 = np.unique(trait_tbl_eff[:, feat_1]).tolist()
        #     if len(feat_2) > 1:
        #         names_states_feat_2 = names_features_original[incl_features][feat_2]
        #     else:
        #         names_states_feat_2 = np.unique(trait_tbl_eff[:, feat_2]).tolist()
        #     prob = get_prob_inter_discr_discr(pdp_rates_eff, trait_tbl_eff, feat_1, feat_2,
        #                                       names, names_states_feat_1, names_states_feat_2)
        #     prob_effects = pd.concat([prob_effects, prob], ignore_index=True)
        # elif np.isin(pt, np.array([13.0, 14.0])):
        #     b = binary_feature[incl_features]
        #     names_states = np.unique(trait_tbl_eff[:, b]).tolist()
        #     if pt == 13.0:
        #         names_states = names_features_original[incl_features][b]
        #     names = names[np.argsort(b)]
        #     trait_tbl_eff = trait_tbl_eff[:, np.argsort(b)]  # Continuous feature always in column 0
        #     prob = get_prob_inter_cont_discr_ord(pdp_rates_eff, trait_tbl_eff, names[0], names[1], names_states)
        #     prob_effects = pd.concat([prob_effects, prob], ignore_index=True)
    green_imp.columns = ['feature1', 'feature2', 'feature1_state', 'feature2_state',
                         'mean', 'lwr_CI', 'upr_CI']
    return green_imp


def get_importance_score(rates, feat_value):
    num_it = rates.shape[1]
    u = np.unique(feat_value)
    lenu = len(u)
    # standard deviation of rates for each unique feature value across all iterations
    stdu = np.zeros((lenu, num_it))
    for j in range(lenu):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category = RuntimeWarning)
            stdu[j, :] = np.nanstd(rates[feat_value == u[j], :], axis = 0)
    imp_sc = np.nanstd(stdu, axis = 0)
    return imp_sc


def get_greenwells_interaction_importance(rates, feat):
    imp_sc0 = get_importance_score(rates, feat[:, 0])
    imp_sc1 = get_importance_score(rates, feat[:, 1])
    gi = np.mean(np.stack((imp_sc0, imp_sc1), axis = 0), axis = 0)
    gi_summary = np.zeros(3)
    gi_summary[0] = np.mean(gi)
    gi_summary[1:] = util.calcHPD(gi, 0.95)
    return gi_summary


# Feature permutation
#####################
def get_bdnn_time(bdnn_obj, ts):
    bdnn_time = np.concatenate((np.array([np.max(ts)+ 0.001]), # (np.array([sp_fad_lad['FAD'].max()])
                                bdnn_obj.bdnn_settings['fixed_times_of_shift_bdnn'],
                                np.zeros(1)))
    return bdnn_time


# trait_tbl_NN, hidden_act_f, out_act_f added
def BDNN_partial_lik(arg):
    [ts, te, up, lo, rate, par, nn_prm, indx, trait_tbl_NN, hidden_act_f, out_act_f] = arg
    if par == "l":
        i_events = np.intersect1d((ts <= up).nonzero()[0], (ts > lo).nonzero()[0])
    else:
        i_events = np.intersect1d((te <= up).nonzero()[0], (te > lo).nonzero()[0])
    n_all_inframe, n_S = get_sp_in_frame_br_length(ts, te, up, lo)
    if np.isfinite(indx):
        if par == "l":
            r = get_rate_BDNN(rate, trait_tbl_NN[0][indx], nn_prm, hidden_act_f, out_act_f)
        else:
            r = get_rate_BDNN(rate, trait_tbl_NN[1][indx], nn_prm, hidden_act_f, out_act_f)
    else:
        if par == "l":
            r = get_rate_BDNN(rate, trait_tbl_NN[0], nn_prm, hidden_act_f, out_act_f)
        else:
            r = get_rate_BDNN(rate, trait_tbl_NN[1], nn_prm, hidden_act_f, out_act_f)
    lik = np.sum(np.log(r[i_events])) + np.sum(-r[n_all_inframe] * n_S)
    return lik


def get_bdnn_lik(bdnn_obj, bdnn_time, ts, te, w, trait_tbl_NN, rate_type):
    time_var = trait_tbl_NN[0].ndim > 2
    hidden_act_f = bdnn_obj.bdnn_settings['hidden_act_f']
    out_act_f  = bdnn_obj.bdnn_settings['out_act_f']
    if time_var:
        likBDtemp = np.zeros(len(bdnn_time) - 1)
        for temp_l in range(len(bdnn_time) - 1):
            up, lo = bdnn_time[temp_l], bdnn_time[temp_l + 1]
            l = 1.0#L[temp_l]
            args = [ts, te, up, lo, l, rate_type, w, temp_l, trait_tbl_NN, hidden_act_f, out_act_f]
            likBDtemp[temp_l] = BDNN_partial_lik(args)
        bdnn_lik = np.sum(likBDtemp)
    else:
        up, lo = np.max(bdnn_time), np.zeros(1)
        l = 1.0
        args = [ts, te, up, lo, l, rate_type, w, np.inf, trait_tbl_NN, hidden_act_f, out_act_f]
        bdnn_lik = BDNN_partial_lik(args)
    return bdnn_lik


def create_perm_comb(bdnn_obj, combine_discr_features = None):
    names_features = get_names_features(bdnn_obj)
    idx_comb_feat = get_idx_comb_feat(names_features, combine_discr_features)
    n_feature_groups = len(idx_comb_feat)
    idx_not_comb_feat = np.arange(len(names_features))
    if idx_comb_feat:
        names_features = replace_names_by_feature_group(names_features, idx_comb_feat, combine_discr_features)
        conc_comb_feat = np.concatenate(idx_comb_feat)
        idx_not_comb_feat = idx_not_comb_feat[~np.isin(idx_not_comb_feat, conc_comb_feat)]
    perm_names = list()
    perm_feat_idx = list()
    # features expressed in single columns of the trait table
    for i in idx_not_comb_feat:
        perm_names.append([names_features[i], 'none'])
        perm_feat_idx.append([np.array([i]), None])
    # feature groups e.g. one-hot encoded features
    for i in range(n_feature_groups):
        perm_names.append([names_features[idx_comb_feat[i][0]], 'none'])
        perm_feat_idx.append([ idx_comb_feat[i], None ])
    # interaction between features expressed as single columns
    interaction_not_comb_feat = list(combinations(idx_not_comb_feat, 2))
    for i in range(len(interaction_not_comb_feat)):
        inter = interaction_not_comb_feat[i]
        name_inter = [names_features[inter[0]], names_features[inter[1]]]
        perm_names.append(name_inter)
        perm_feat_idx.append([ np.array([inter[0]]), np.array([inter[1]]) ])
    # interaction between feature groups
    interaction_comb_feat = list(combinations(np.arange(n_feature_groups), 2))
    for i in range(len(interaction_comb_feat)):
        inter = interaction_comb_feat[i]
        fg0 = idx_comb_feat[inter[0]]
        fg1 = idx_comb_feat[inter[1]]
        name_inter = [names_features[fg0[0]], names_features[fg1[0]]]
        perm_names.append(name_inter)
        perm_feat_idx.append([fg0, fg1])
    # interaction between single-column feature with feature group
    interaction_sc_fg = expand_grid(idx_not_comb_feat, np.arange(n_feature_groups))
    for i in range(len(interaction_sc_fg)):
        inter = interaction_sc_fg[i]
        sc = inter[0]
        fg = idx_comb_feat[inter[1]]
        name_inter = [names_features[sc], names_features[fg[0]]]
        perm_names.append(name_inter)
        perm_feat_idx.append([ np.array([sc]), fg ])
    return perm_names, perm_feat_idx


def permute_trt_tbl(trt_tbl, feat_idx, feature_is_time_variable, bdnn_obj, post_ts_i, seed = None):
    # # 4 bins, 5 species, 3 trait
    # a = np.arange(60).reshape((4, 5, 3))
    # print(a)
    # # Flip trait 1 in all bins
    # b = a + 0
    # b[:, :, 1] = a[:, [4, 3, 2, 1, 0], 1]
    # print(b)
    # # Flip trait 3 (e.g. time) across bins
    # d = a + 0
    # d[:, :, 2] = a[[3, 2, 1, 0], :, 2]
    # print(d)
    rng = np.random.default_rng(seed)
    if trt_tbl[0].ndim == 3:
        if np.any(feature_is_time_variable[feat_idx]):
            ## time-variable, one-hot encoded features will not work (hopefully nobody will use these type of features)
            ## Swapping time for all species together among bins
#            n_bins = trt_tbl[0].shape[0]
#            bins_perm_idx = rng.permuted(np.arange(n_bins))
#            trt_tbl[0][:, :, feat_idx] = trt_tbl[0][bins_perm_idx, :, :][:, :, feat_idx]
#            trt_tbl[1][:, :, feat_idx] = trt_tbl[1][bins_perm_idx, :, :][:, :, feat_idx]
            ## Free permutation
#            feat_sp = trt_tbl[0][:, :, feat_idx]
#            feat_ex = trt_tbl[1][:, :, feat_idx]
#            n = len(feat_sp)
#            rng = np.random.default_rng(seed)
#            perm_idx = rng.permuted(np.arange(n))
#            trt_tbl[0][:, :, feat_idx] = feat_sp[perm_idx]
#            trt_tbl[1][:, :, feat_idx] = feat_ex[perm_idx]
            ## Permutation according to relative length of each time-bin (what if there is no time e.g. only div-dep?)
            n_species = trt_tbl[0].shape[1]
            n_bins = trt_tbl[0].shape[0]
            bdnn_time = get_bdnn_time(bdnn_obj, post_ts_i)
            rel_time = np.abs(np.diff(bdnn_time)) / np.max(bdnn_time)
            rng = np.random.default_rng(seed)
            perm_idx = rng.choice(n_bins, n_bins * n_species, p = rel_time).reshape((n_bins, n_species))
            trt_tbl_org = copy_lib.deepcopy(trt_tbl)
            for s in range(n_species):
                trt_tbl[0][:, s, feat_idx] = trt_tbl_org[0][:, s, feat_idx][perm_idx[:, s], :]
                trt_tbl[1][:, s, feat_idx] = trt_tbl_org[1][:, s, feat_idx][perm_idx[:, s], :]
        else:
            n_species = trt_tbl[0].shape[1]
            species_perm_idx = rng.permuted(np.arange(n_species))
            trt_tbl[0][:, :, feat_idx] = trt_tbl[0][:, species_perm_idx, :][:, :, feat_idx]
            trt_tbl[1][:, :, feat_idx] = trt_tbl[1][:, species_perm_idx, :][:, :, feat_idx]
    else:
        n_species = trt_tbl[0].shape[0]
        species_perm_idx = rng.permuted(np.arange(n_species))
        trt_tbl[0][:, feat_idx] = trt_tbl[0][species_perm_idx, :][:, feat_idx]
        trt_tbl[1][:, feat_idx] = trt_tbl[1][species_perm_idx, :][:, feat_idx]
    return trt_tbl


def perm_mcmc_sample_i(arg):
    [bdnn_obj, post_ts_i, post_te_i, post_w_sp_i, post_w_ex_i, trt_tbls, n_perm, n_perm_traits, n_features, feature_is_time_variable, perm_feature_idx] = arg
    bdnn_time = get_bdnn_time(bdnn_obj, post_ts_i)
    # Original bd liks
    orig_birth_lik = get_bdnn_lik(bdnn_obj, bdnn_time, post_ts_i, post_te_i, post_w_sp_i,
                                  trt_tbls, rate_type='l')
    orig_death_lik = get_bdnn_lik(bdnn_obj, bdnn_time, post_ts_i, post_te_i, post_w_ex_i,
                                  trt_tbls, rate_type='m')
    sp_lik_j = np.zeros((n_perm, n_perm_traits))
    ex_lik_j = np.zeros((n_perm, n_perm_traits))
    rngint = np.random.default_rng()
    seeds = rngint.integers(low=0, high=1e10, size=n_features)
    for j in range(n_perm_traits):
        perm_feature_idx_j = perm_feature_idx[j]
        for k in range(n_perm):
            trt_tbls_perm = copy_lib.deepcopy(trt_tbls)
            for l in range(len(perm_feature_idx_j)):
                feat_idx = perm_feature_idx_j[l]
                if feat_idx is not None:
                    seed = seeds[feat_idx] + k
                    if feat_idx.size > 1:
                        seed = seed[0]
                    trt_tbls_perm = permute_trt_tbl(trt_tbls_perm, feat_idx, feature_is_time_variable, bdnn_obj, post_ts_i, seed)
            sp_lik_j[k, j] = get_bdnn_lik(bdnn_obj, bdnn_time, post_ts_i, post_te_i, post_w_sp_i,
                                          trt_tbls_perm, rate_type='l')
            ex_lik_j[k, j] = get_bdnn_lik(bdnn_obj, bdnn_time, post_ts_i, post_te_i, post_w_ex_i,
                                          trt_tbls_perm, rate_type='m')
        species_sp_delta_lik = sp_lik_j - orig_birth_lik
        species_ex_delta_lik = ex_lik_j - orig_death_lik
    return np.hstack((species_sp_delta_lik, species_ex_delta_lik))


def is_unix():
    unixos = True
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        unixos = False
    return unixos


def get_idx_maineffect(m, p, k, old_idx):
    out = old_idx
    if len(m) == len(p):
        if np.all(m == p):
            out = k
    return out


def remove_invariant_feature_from_featperm_results(bdnn_obj, res, trt_tbl, combine_discr_features = ""):
    features_without_variance = get_idx_feature_without_variance(trt_tbl)
    names_features = get_names_features(bdnn_obj)
    idx_comb_feat = get_idx_comb_feat(names_features, combine_discr_features)
    if idx_comb_feat:
        names_features = replace_names_by_feature_group(names_features, idx_comb_feat, combine_discr_features)
    for i in features_without_variance:
        res = res[res['feature1'] != names_features[i]]
        res = res[res['feature2'] != names_features[i]]
    return res


def feature_permutation(mcmc_file, pkl_file, burnin, thin, n_perm = 10, num_processes = 1, combine_discr_features = "", show_progressbar = False):
    bdnn_obj, post_w_sp, post_w_ex, sp_fad_lad, post_ts, post_te = bdnn_parse_results(mcmc_file, pkl_file, burnin, thin)
    n_mcmc = post_ts.shape[0]
    trt_tbls = bdnn_obj.trait_tbls
    n_features = trt_tbls[0].shape[-1]
    sp_feature_is_time_variable = is_time_variable_feature(trt_tbls[0])
    ex_feature_is_time_variable = is_time_variable_feature(trt_tbls[1])
    feature_is_time_variable = sp_feature_is_time_variable + ex_feature_is_time_variable
    perm_traits, perm_feature_idx = create_perm_comb(bdnn_obj, combine_discr_features)
    n_perm_traits = len(perm_traits)
    args = []
    for i in range(n_mcmc):
        a = [bdnn_obj, post_ts[i, :], post_te[i, :], post_w_sp[i], post_w_ex[i], trt_tbls,
             n_perm, n_perm_traits, n_features, feature_is_time_variable, perm_feature_idx]
        args.append(a)
    unixos = is_unix()
    if unixos and num_processes > 1:
        pool_perm = multiprocessing.Pool(num_processes)
        #delta_lik = pool_perm.map(perm_mcmc_sample_i, args) # Ordered execution
        delta_lik = list(tqdm(pool_perm.imap_unordered(perm_mcmc_sample_i, args),
                              total = n_mcmc, disable = show_progressbar == False))
        pool_perm.close()
    else:
        delta_lik = []
        for i in tqdm(range(n_mcmc), disable = show_progressbar == False):
            delta_lik.append(perm_mcmc_sample_i(args[i]))
    delta_lik = np.vstack(delta_lik)
    sp_delta_lik = delta_lik[:, :n_perm_traits] + 0.0
    ex_delta_lik = delta_lik[:, n_perm_traits:] + 0.0
    for j in range(n_perm_traits):
        if perm_feature_idx[j][1] is None:
            n_main = j + 1
    for j in range(n_main, n_perm_traits):
        perm_feature_idx_j = perm_feature_idx[j]
        idx_main1 = np.inf
        idx_main2 = np.inf
        for k in range(n_main):
            m = perm_feature_idx[k][0]
            idx_main1 = get_idx_maineffect(m, perm_feature_idx_j[0], k, idx_main1)
            idx_main2 = get_idx_maineffect(m, perm_feature_idx_j[1], k, idx_main2)
        sp_delta_lik[:, j] = (sp_delta_lik[:, idx_main1] + sp_delta_lik[:, idx_main2]) - sp_delta_lik[:, j]
        ex_delta_lik[:, j] = (ex_delta_lik[:, idx_main1] + ex_delta_lik[:, idx_main2]) - ex_delta_lik[:, j]
    names_df = pd.DataFrame(perm_traits, columns = ['feature1', 'feature2'])
    sp_delta_lik_summary = get_rates_summary(sp_delta_lik.T)
    ex_delta_lik_summary = get_rates_summary(ex_delta_lik.T)
    sp_delta_lik_df = pd.DataFrame(sp_delta_lik_summary, columns = ['delta_lik', 'lwr_delta_lik', 'upr_delta_lik'])
    ex_delta_lik_df = pd.DataFrame(ex_delta_lik_summary, columns = ['delta_lik', 'lwr_delta_lik', 'upr_delta_lik'])
    sp_delta_lik_df = pd.concat([names_df, sp_delta_lik_df], axis = 1)
    ex_delta_lik_df = pd.concat([names_df, ex_delta_lik_df], axis = 1)
    sp_delta_lik_df = remove_invariant_feature_from_featperm_results(bdnn_obj, sp_delta_lik_df, trt_tbls[0],
                                                                     combine_discr_features)
    ex_delta_lik_df = remove_invariant_feature_from_featperm_results(bdnn_obj, ex_delta_lik_df, trt_tbls[1],
                                                                     combine_discr_features)
    return sp_delta_lik_df, ex_delta_lik_df


# Shap
#######
def get_num_interaction_bins(trt_tbl):
    num_rows = trt_tbl.shape[0]
    bin_credibility_threshold = 50
    max_auto_bins = 10
    ib = int(num_rows / bin_credibility_threshold)
    ib = np.min([max_auto_bins, ib])
    return ib


def get_interaction_R2(trt_tbl, shap_main):
    s = trt_tbl.shape[1]
    inter_r2 = np.zeros((s, s))
    num_interaction_bins = get_num_interaction_bins(trt_tbl)
    for i in range(s):
        num_cat = len(np.unique(trt_tbl.iloc[:, i]))
        nib = num_interaction_bins
        if num_cat < num_interaction_bins:
            nib = num_cat
        try:
            r2, r2_idx = get_variable_interactions(shap_main, trt_tbl, variable = i, interaction_bins = nib)
        except:
            r2, r2_idx = get_variable_interactions(shap_main, trt_tbl, variable = i, interaction_bins = 2)
        inter_r2[i, r2_idx] = r2
    inter_r2 = (np.triu(inter_r2) + np.triu(inter_r2.T)) / 2
    inter_r2[np.tril_indices(s)] = inter_r2[np.triu_indices(s)]
    return inter_r2


def kernel_explainer(trt_tbl, cov_par, hidden_act_f, out_act_f, idx_comb_feat):
    ke = fastshap.KernelExplainer(
        model = lambda X: get_rate_BDNN(1, X, cov_par, hidden_act_f, out_act_f),
        background_data = trt_tbl
    )
    shap_main = ke.calculate_shap_values(trt_tbl, verbose = False)
    shap_main = shap_main[:,:-1] # remove expected value
    shap_interaction = get_interaction_R2(trt_tbl, shap_main)
    shap_main = main_shap_for_onehot_features(idx_comb_feat, shap_main)
    shap_main = np.mean(np.abs(shap_main), axis = 0)
    shap_interaction = interaction_R2_for_onehot_features(idx_comb_feat, shap_interaction)
    iu1 = np.triu_indices(shap_interaction.shape[0], 1)
    shap_interaction = shap_interaction[iu1]
    return np.concatenate((shap_main, shap_interaction))


def kernel_shap_i(arg):
    [bdnn_obj, post_ts_i, post_te_i, post_w_sp_i, post_w_ex_i, hidden_act_f, out_act_f, trt_tbls, bdnn_dd, idx_comb_feat_sp, idx_comb_feat_ex, binary_feature_sp, binary_feature_ex] = arg
    bdnn_time = get_bdnn_time(bdnn_obj, post_ts_i)
    if bdnn_dd:
        n_taxa = trt_tbls[0].shape[1]
        bdnn_rescale_div = bdnn_obj.bdnn_settings['div_rescaler']
        bdnn_time_div = np.arange(np.max(post_ts_i), 0.0, -0.001)
        bdnn_div = get_DT(bdnn_time_div, post_ts_i, post_te_i)
        bdnn_binned_div = get_binned_div_traj(bdnn_time, bdnn_time_div, bdnn_div)[:-1] / bdnn_rescale_div
        bdnn_binned_div = np.repeat(bdnn_binned_div, n_taxa).reshape((len(bdnn_binned_div), n_taxa))
        trt_tbls[0][:, :, div_idx_trt_tbl] = bdnn_binned_div
        trt_tbls[1][:, :, div_idx_trt_tbl] = bdnn_binned_div
    shap_trt_tbl_sp = get_shap_trt_tbl(post_ts_i, bdnn_time, trt_tbls[0])
    shap_trt_tbl_ex = get_shap_trt_tbl(post_te_i, bdnn_time, trt_tbls[1])
    shap_trt_tbl_sp = pd.DataFrame(shap_trt_tbl_sp)
    shap_trt_tbl_ex = pd.DataFrame(shap_trt_tbl_ex)
    shap_trt_tbl_sp[binary_feature_sp] = shap_trt_tbl_sp[binary_feature_sp].astype('int')
    shap_trt_tbl_ex[binary_feature_ex] = shap_trt_tbl_ex[binary_feature_ex].astype('int')
    lam_ke = kernel_explainer(shap_trt_tbl_sp, post_w_sp_i, hidden_act_f, out_act_f, idx_comb_feat_sp)
    mu_ke = kernel_explainer(shap_trt_tbl_ex, post_w_ex_i, hidden_act_f, out_act_f, idx_comb_feat_ex)
    return np.concatenate((lam_ke, mu_ke))



def kernel_shap(mcmc_file, pkl_file, burnin, thin, num_processes = 1, combine_discr_features = "", show_progressbar = False):
    bdnn_obj, post_w_sp, post_w_ex, sp_fad_lad, post_ts, post_te = bdnn_parse_results(mcmc_file, pkl_file, burnin, thin)
    mcmc_samples = post_ts.shape[0]
    trt_tbls = bdnn_obj.trait_tbls
    binary_feature_sp = is_binary_feature(trt_tbls[0])[0]
    binary_feature_ex = is_binary_feature(trt_tbls[1])[0]
    binary_feature_sp = np.arange(trt_tbls[0].shape[-1])[binary_feature_sp].tolist()
    binary_feature_ex = np.arange(trt_tbls[1].shape[-1])[binary_feature_ex].tolist()
    names_features_sp = get_names_features(bdnn_obj)
    names_features_ex = copy_lib.deepcopy(names_features_sp)
    bdnn_dd = 'diversity' in names_features_sp
    hidden_act_f = bdnn_obj.bdnn_settings['hidden_act_f']
    out_act_f = bdnn_obj.bdnn_settings['out_act_f']
    idx_comb_feat_sp = get_idx_comb_feat(names_features_sp, combine_discr_features)
    idx_comb_feat_ex = get_idx_comb_feat(names_features_ex, combine_discr_features)
    shap_names_sp = make_shap_names(names_features_sp, idx_comb_feat_sp, combine_discr_features)
    shap_names_ex = make_shap_names(names_features_ex, idx_comb_feat_ex, combine_discr_features)
    n_effects_sp = shap_names_sp.shape[0]
    n_effects_ex = shap_names_ex.shape[0]
    args = []
    for i in range(mcmc_samples):
        a = [bdnn_obj, post_ts[i, :], post_te[i, :], post_w_sp[i], post_w_ex[i], hidden_act_f, out_act_f,
             trt_tbls, bdnn_dd, idx_comb_feat_sp, idx_comb_feat_ex, binary_feature_sp, binary_feature_ex]
        args.append(a)
    unixos = is_unix()
    if unixos and num_processes > 1:
        pool_perm = multiprocessing.Pool(num_processes)
        shap_values = list(tqdm(pool_perm.imap_unordered(kernel_shap_i, args),
                                total = mcmc_samples, disable = show_progressbar == False))
        pool_perm.close()
    else:
        shap_values = []
        for i in tqdm(range(mcmc_samples), disable = show_progressbar == False):
            shap_values.append(kernel_shap_i(args[i]))
    shap_values = np.vstack(shap_values)
    mean_shap = np.mean(shap_values, axis = 0)
    mean_shap_sp = mean_shap[:n_effects_sp]
    mean_shap_ex = mean_shap[n_effects_sp:(n_effects_sp + n_effects_ex)]
    feature_without_variance_sp = get_idx_feature_without_variance(trt_tbls[0])
    feature_without_variance_ex = get_idx_feature_without_variance(trt_tbls[1])
    remove_sp = []
    for i in feature_without_variance_sp:
        remove_sp.append(np.where(shap_names_sp[:, 0] == names_features_sp[i])[0])
        remove_sp.append(np.where(shap_names_sp[:, 1] == names_features_sp[i])[0])
    remove_ex = []
    for i in feature_without_variance_ex:
        remove_ex.append(np.where(shap_names_ex[:, 0] == names_features_ex[i])[0])
        remove_ex.append(np.where(shap_names_ex[:, 1] == names_features_ex[i])[0])
    remove_sp = np.array(list(pd.core.common.flatten(remove_sp))).astype(int)
    remove_ex = np.array(list(pd.core.common.flatten(remove_ex))).astype(int)
    mean_shap_sp = np.delete(mean_shap_sp, remove_sp[remove_sp < len(mean_shap_sp)])
    mean_shap_ex = np.delete(mean_shap_ex, remove_ex[remove_ex < len(mean_shap_ex)])
    shap_names_sp = np.delete(shap_names_sp, remove_sp, axis = 0)
    shap_names_ex = np.delete(shap_names_ex, remove_ex, axis = 0)
    shap_values_sp = pd.DataFrame(mean_shap_sp, columns=['shap'])
    shap_values_ex = pd.DataFrame(mean_shap_ex, columns=['shap'])
    shap_names_sp = pd.DataFrame(shap_names_sp, columns = ['feature1', 'feature2'])
    shap_names_ex = pd.DataFrame(shap_names_ex, columns = ['feature1', 'feature2'])
    shap_lam = pd.concat([shap_names_sp, shap_values_sp], axis = 1)
    shap_ex = pd.concat([shap_names_ex, shap_values_ex], axis = 1)
    return shap_lam, shap_ex


def main_shap_for_onehot_features(idx_comb_feat, sm):
    if len(idx_comb_feat) > 0:
        drop = np.array([], dtype = int)
        for i in range(len(idx_comb_feat)):
            sm[:, idx_comb_feat[i][0]] = np.sum(sm[:, idx_comb_feat[i]], axis = 1)
            drop = np.concatenate((drop, idx_comb_feat[i][1:]))
        sm = np.delete(sm, drop, axis = 1)
    return sm


def interaction_R2_for_onehot_features(idx_comb_feat, si):
    if len(idx_comb_feat) > 0:
        drop = np.array([], dtype=int)
        conc_comb_feat = np.concatenate(idx_comb_feat)
        n = si.shape[0]
        J = np.arange(n)
        J = np.delete(J, conc_comb_feat)
        # Cases of no interaction within another one-hot encoded feature
        if len(J) > 0:
            for i in range(len(idx_comb_feat)):
                for j in J:
                    inter_value = np.mean(si[j, idx_comb_feat[i]])
                    si[j, idx_comb_feat[i][0]] = inter_value
                    si[idx_comb_feat[i][0], j] = inter_value
                drop = np.concatenate((drop, idx_comb_feat[i][1:]))
        # Cases of interaction between two one-hot encoded features
        if len(idx_comb_feat) > 1:
            for i in range(len(idx_comb_feat)):
                for k in range(1, len(idx_comb_feat)):
                    inter_value = np.mean(si[idx_comb_feat[k], idx_comb_feat[i]])
                    i1 = idx_comb_feat[k][0]
                    i2 = idx_comb_feat[i][0]
                    si[i1, i2] = inter_value
                    si[i2, i1] = inter_value
        si = np.delete(si, drop, axis = 0)
        si = np.delete(si, drop, axis = 1)
    return si


def make_shap_names(names_features, idx_comb_feat, combine_discr_features):
    if idx_comb_feat:
        names_features = replace_names_by_feature_group(names_features, idx_comb_feat, combine_discr_features)
    names_features = np.array(names_features)
    names_main_features = copy_lib.deepcopy(names_features)
    u, ind = np.unique(names_main_features, return_index = True)
    names_main_features = u[np.argsort(ind)]
    l = len(names_features)
    names_inter_features = np.repeat(names_features, l).reshape((l,l))
    iu1 = np.triu_indices(l, 1)
    names_inter_features = np.stack((names_inter_features[iu1], names_inter_features.T[iu1]), axis = 1)
    names_inter_features_df = pd.DataFrame(names_inter_features)
    keep = names_inter_features_df.duplicated() == False
    names_inter_features = names_inter_features_df.loc[keep, :].to_numpy()
    keep = names_inter_features[:, 0] != names_inter_features[:, 1]
    names_inter_features = names_inter_features[keep, :]
    names_main_features = np.stack((names_main_features, np.repeat('none', len(names_main_features))), axis = 1)
    nf = np.vstack((names_main_features, names_inter_features))
    return nf


# k-additive Choque SHAP
def nParam_kAdd(kAdd, nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr, ii + 1)
    return aux_numb


def powerset(iterable, k_add):
    '''Return the powerset (for coalitions until k_add players) of a set of m attributes
    powerset([1,2,..., m],m) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m) ... (1, ..., m)
    powerset([1,2,..., m],2) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(k_add + 1))


def tr_shap2game(nAttr, k_add):
    '''Return the transformation matrix from Shapley interaction indices, given a k_additive model, to game'''
    nBern = bernoulli(k_add)  # Números de Bernoulli
    k_add_numb = nParam_kAdd(k_add, nAttr)

    coalit = np.zeros((k_add_numb, nAttr))

    for i, s in enumerate(powerset(range(nAttr), k_add)):
        s = list(s)
        coalit[i, s] = 1

    matrix_shap2game = np.zeros((k_add_numb, k_add_numb))
    for i in range(coalit.shape[0]):
        for i2 in range(k_add_numb):
            aux2 = int(sum(coalit[i2, :]))
            aux3 = int(sum(coalit[i, :] * coalit[i2, :]))
            aux4 = 0
            for i3 in range(int(aux3 + 1)):
                aux4 += comb(aux3, i3) * nBern[aux2 - i3]
            matrix_shap2game[i, i2] = aux4
    return matrix_shap2game


def coalition_shap_kadd(k_add, nAttr):
    ''' Return the matrix whose rows represent coalitions of players for cardinality at most k_add '''
    k_add_numb = nParam_kAdd(k_add, nAttr)
    coal_shap = np.zeros((k_add_numb, nAttr))

    for i, s in enumerate(powerset(range(nAttr), k_add)):
        s = list(s)
        coal_shap[i, s] = 1
    return coal_shap


def vector_shap2game(x, k_add, nAttr, coal_shap):
    '''Return the transformation vector, associated with the coalition represented by x, from Shapley interaction indices, given a k_additive model, to game'''
    nBern = bernoulli(k_add)  # Números de Bernoulli
    k_add_numb = nParam_kAdd(k_add, nAttr)
    vector_shap2game = np.zeros((k_add_numb,))

    for ii in range(coal_shap.shape[0]):
        aux1 = int(sum(coal_shap[ii, :]))
        aux2 = int(sum(x * coal_shap[ii, :]))
        aux3 = 0
        for jj in range(int(aux2 + 1)):
            aux3 += comb(aux2, jj) * nBern[aux1 - jj]
        vector_shap2game[ii] = aux3
    return vector_shap2game


def opt_Xbinary_wrand_allMethods(nEval, nAttr, k_add, coal_shap, X_train):
    ''' Return the matrix of coalitions used in Kernel SHAP (X), the transformation matrix used in the proposal (opt_data) '''
    # Select at random, but with probability distributions based on the SHAP weights
    weights_shap = np.zeros((nEval))
    k_add_numb = nParam_kAdd(k_add, nAttr)
    aux = []
    aux2 = np.ones((nAttr,))
    for ii in range(1, nAttr):
        aux = np.append(aux, comb(nAttr, ii) * shapley_kernel(nAttr, ii))
        aux2[ii] = aux2[ii - 1] + comb(nAttr, ii)

    selec_data_aux = np.zeros((nEval,))
    p_aux = aux / sum(aux)
    for ii in range(nEval):
        p_aux = aux / np.sum(aux)
        selec_data_aux[ii] = np.random.choice(np.arange(nAttr - 1) + 1, size=1, replace=False, p=p_aux)
        aux[int(selec_data_aux[ii] - 1)] -= shapley_kernel(nAttr, int(selec_data_aux[ii]))
        aux = np.maximum(aux, np.zeros((len(aux),)))
        p_aux = aux / np.sum(aux)

    unique, counts = np.unique(selec_data_aux, return_counts=True)
    X = np.zeros((nEval, nAttr))
    opt_data = np.zeros((nEval, k_add_numb))
    counter = 0
    for ii in range(len(unique)):
        aux = np.sort(random.sample(range(nAttr), int(unique[ii]))).reshape(1, -1)
        while np.unique(aux, axis=0).shape[0] < counts[ii]:
            aux = np.append(aux, np.sort(random.sample(range(nAttr), int(unique[ii]))).reshape(1, -1), axis=0)
        aux = np.unique(aux, axis=0)

        for jj in range(aux.shape[0]):
            X[counter, aux[jj, :]] = 1
            opt_data[counter, :] = vector_shap2game(X[counter, :], k_add, nAttr, coal_shap)

            weights_shap[counter] = shapley_kernel(nAttr, sum(X[counter, :]))

            counter += 1

    X = np.concatenate((np.concatenate((np.zeros((1, nAttr)), X), axis=0), np.ones((1, nAttr))), axis=0)
    X = np.concatenate((X, np.ones((nEval + 2, 1))), axis=1)
    opt_data = np.concatenate((vector_shap2game(np.zeros((nAttr,)), k_add, nAttr, coal_shap).reshape(1, -1), opt_data),
                              axis=0)
    opt_data = np.concatenate((opt_data, vector_shap2game(np.ones((nAttr,)), k_add, nAttr, coal_shap).reshape(1, -1)),
                              axis=0)
    weights_shap = np.append(10 ** 6, weights_shap)
    weights_shap = np.append(weights_shap, 10 ** 6)

    return X, opt_data, weights_shap


def shapley_kernel(M, s):
    ''' Return the Kernel SHAP weight '''
    if s == 0 or s == M:
        return 100000
    return (M - 1) / (binom(M, s) * s * (M - s))


def k_add_kernel_explainer(trt_tbl, cov_par, hidden_act_f, out_act_f):
    k_add = 3
    n_species, nAttr = trt_tbl.shape  # Number of instances and attributes
    # " Basic elements"
    k_add_numb = 1
    for ii in range(k_add):
        k_add_numb += comb(nAttr, ii + 1)
    # " Providing local explanations "
    coal_shap = coalition_shap_kadd(k_add, nAttr)
    nEval_old = coal_shap.shape[0]
    for ii in range(1, 6):
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                nEval = ii * coal_shap.shape[0]
                # " By selecting weighted random samples (without replacement) "
                X, opt_data, weights_shap = opt_Xbinary_wrand_allMethods(nEval - 2, nAttr, k_add, coal_shap, trt_tbl)
            except:
                nEval = nEval_old
    weights = np.eye(nEval)
    weights[0, 0], weights[-1, -1] = 10 ** 6, 10 ** 6
    shap_main = np.zeros((n_species, nAttr + 1))
    shap_inter = np.zeros((n_species, nAttr, nAttr))
    for i in range(n_species):
        # For all samples
        exp_payoffs_ci = np.zeros((nEval))
        weights_shap_ci = np.zeros(nEval)
        for ll in range(nEval):
            trt_tbl_aux = trt_tbl + 0.0
            trt_tbl_aux[:, np.where(X[ll, 0:-1] == 1)] = trt_tbl[i, np.where(X[ll, 0:-1] == 1)]
            rate_aux = get_rate_BDNN(1, trt_tbl_aux, cov_par, hidden_act_f, out_act_f)
            exp_payoffs_ci[ll] = np.mean(rate_aux)
            weights_shap_ci[ll] = shapley_kernel(nAttr, np.sum(X[ll, 0:-1]))
        exp_payoffs_shap = exp_payoffs_ci
        exp_payoffs_ci = exp_payoffs_ci - exp_payoffs_ci[0]
        # For weighted random samples
        explain_matrix = np.linalg.inv(opt_data.T @ weights @ opt_data) @ opt_data.T @ weights
        inter_val = explain_matrix @ exp_payoffs_ci
        shapley_ci = inter_val[1:]
        shapley_val_ci_shap = np.linalg.inv(X.T @ np.diag(weights_shap) @ X) @ X.T @ np.diag(weights_shap) @ exp_payoffs_shap
        shap_main[i, :] = shapley_val_ci_shap#[0:-1]
        # Interaction indices
        count = 0
        indices = np.zeros((nAttr, nAttr))
        for ii in range(nAttr - 1):
            for jj in range(ii + 1, nAttr):
                indices[ii, jj] = shapley_ci[nAttr + count]
                count += 1
        indices = indices + indices.T
        shap_inter[i, :, :] = indices
    return shap_main, shap_inter


def inter_shap_for_onehot_features(idx_comb_feat, si):
    if len(idx_comb_feat) > 0:
        drop = np.array([], dtype=int)
        conc_comb_feat = np.concatenate(idx_comb_feat)
        n = si.shape[1]
        J = np.arange(n)
        J = np.delete(J, conc_comb_feat)
        # Cases of no interaction within another one-hot encoded feature
        if len(J) > 0:
            for i in range(len(idx_comb_feat)):
                for j in J:
                    inter_value = np.sum(si[:, :, idx_comb_feat[i]][:, j, :], axis = 1)
                    si[:, j, idx_comb_feat[i][0]] = inter_value
                    si[:, idx_comb_feat[i][0], j] = inter_value
                drop = np.concatenate((drop, idx_comb_feat[i][1:]))
        # Cases of interaction between two one-hot encoded features
        if len(idx_comb_feat) > 1:
            for i in range(len(idx_comb_feat)):
                for k in range(1, len(idx_comb_feat)):
                    inter_value = np.sum(si[:, :, idx_comb_feat[i]][:, idx_comb_feat[k], :], axis = (1, 2))
                    i1 = idx_comb_feat[k][0]
                    i2 = idx_comb_feat[i][0]
                    si[:, i1, i2] = inter_value
                    si[:, i2, i1] = inter_value
        si = np.delete(si, drop, axis = 1)
        si = np.delete(si, drop, axis = 2)
    return si


def interaction_R2_for_onehot_features(idx_comb_feat, si):
    if len(idx_comb_feat) > 0:
        drop = np.array([], dtype=int)
        conc_comb_feat = np.concatenate(idx_comb_feat)
        n = si.shape[0]
        J = np.arange(n)
        J = np.delete(J, conc_comb_feat)
        # Cases of no interaction within another one-hot encoded feature
        if len(J) > 0:
            for i in range(len(idx_comb_feat)):
                for j in J:
                    inter_value = np.mean(si[j, idx_comb_feat[i]])
                    si[j, idx_comb_feat[i][0]] = inter_value
                    si[idx_comb_feat[i][0], j] = inter_value
                drop = np.concatenate((drop, idx_comb_feat[i][1:]))
        # Cases of interaction between two one-hot encoded features
        if len(idx_comb_feat) > 1:
            for i in range(len(idx_comb_feat)):
                for k in range(1, len(idx_comb_feat)):
                    inter_value = np.mean(si[idx_comb_feat[k], idx_comb_feat[i]])
                    i1 = idx_comb_feat[k][0]
                    i2 = idx_comb_feat[i][0]
                    si[i1, i2] = inter_value
                    si[i2, i1] = inter_value
        si = np.delete(si, drop, axis = 0)
        si = np.delete(si, drop, axis = 1)
    return si


def make_shap_names(names_features, idx_comb_feat, combine_discr_features):
    if idx_comb_feat:
        names_features = replace_names_by_feature_group(names_features, idx_comb_feat, combine_discr_features)
    names_features = np.array(names_features)
    names_main_features = copy_lib.deepcopy(names_features)
    u, ind = np.unique(names_main_features, return_index = True)
    names_main_features = u[np.argsort(ind)]
    l = len(names_features)
    names_inter_features = np.repeat(names_features, l).reshape((l,l))
    iu1 = np.triu_indices(l, 1)
    names_inter_features = np.stack((names_inter_features[iu1], names_inter_features.T[iu1]), axis = 1)
    names_inter_features_df = pd.DataFrame(names_inter_features)
    keep = names_inter_features_df.duplicated() == False
    names_inter_features = names_inter_features_df.loc[keep, :].to_numpy()
    keep = names_inter_features[:, 0] != names_inter_features[:, 1]
    names_inter_features = names_inter_features[keep, :]
    names_main_features = np.stack((names_main_features, np.repeat('none', len(names_main_features))), axis = 1)
    nf = np.vstack((names_main_features, names_inter_features))
    return nf


def make_taxa_names_shap(taxa_names, n_species, shap_names):
    sn = shap_names.to_numpy()
    shap_names_main = sn[sn[:, 1] == 'none', 0]
    taxa_names_shap = ['baseline']
    for i in range(n_species):
        for j in range(len(shap_names_main)):
            taxa_names_shap.append(taxa_names[i] + '_' + shap_names_main[j])
    return taxa_names_shap


def combine_shap_featuregroup(shap_main_instances, shap_interaction_instances, idx_comb_feat):
    baseline = np.array([shap_main_instances[1, -1]])
    shap_main_instances = main_shap_for_onehot_features(idx_comb_feat, shap_main_instances[:, 0:-1])
    shap_interaction_instances = inter_shap_for_onehot_features(idx_comb_feat, shap_interaction_instances)
    shap_main = np.mean(np.abs(shap_main_instances), axis = 0)
    shap_interaction = np.mean(np.abs(shap_interaction_instances), axis = 0)
    iu1 = np.triu_indices(shap_interaction.shape[0], 1)
    shap_interaction = shap_interaction[iu1]
    return np.concatenate((shap_main, shap_interaction, baseline, shap_main_instances.flatten()))


def k_add_kernel_shap_i(arg):
    [bdnn_obj, post_ts_i, post_te_i, post_w_sp_i, post_w_ex_i, hidden_act_f, out_act_f, trt_tbls, bdnn_dd, idx_comb_feat_sp, idx_comb_feat_ex] = arg
    bdnn_time = get_bdnn_time(bdnn_obj, post_ts_i)
    if bdnn_dd:
        n_taxa = trt_tbls[0].shape[1]
        bdnn_rescale_div = bdnn_obj.bdnn_settings['div_rescaler']
        bdnn_time_div = np.arange(np.max(post_ts_i), 0.0, -0.001)
        bdnn_div = get_DT(bdnn_time_div, post_ts_i, post_te_i)
        bdnn_binned_div = get_binned_div_traj(bdnn_time, bdnn_time_div, bdnn_div)[:-1] / bdnn_rescale_div
        bdnn_binned_div = np.repeat(bdnn_binned_div, n_taxa).reshape((len(bdnn_binned_div), n_taxa))
        trt_tbls[0][:, :, div_idx_trt_tbl] = bdnn_binned_div
        trt_tbls[1][:, :, div_idx_trt_tbl] = bdnn_binned_div
    shap_trt_tbl_sp = get_shap_trt_tbl(post_ts_i, bdnn_time, trt_tbls[0])
    shap_trt_tbl_ex = get_shap_trt_tbl(post_te_i, bdnn_time, trt_tbls[1])
    shap_main_sp, shap_interaction_sp = k_add_kernel_explainer(shap_trt_tbl_sp, post_w_sp_i, hidden_act_f, out_act_f)
    shap_main_ex, shap_interaction_ex = k_add_kernel_explainer(shap_trt_tbl_ex, post_w_ex_i, hidden_act_f, out_act_f)
    lam_ke = combine_shap_featuregroup(shap_main_sp, shap_interaction_sp, idx_comb_feat_sp)
    mu_ke = combine_shap_featuregroup(shap_main_ex, shap_interaction_ex, idx_comb_feat_ex)
    return np.concatenate((lam_ke, mu_ke))


def delete_invariantfeat_from_taxa_shap(feature_without_variance, names_features, shap_names, taxa_shap):
    ''' Delete features without variance from all taxa specific shap values '''
    n_shap = taxa_shap.shape[0] - 1
    shap_names_main = shap_names[shap_names[:, 1] == 'none', 0]
    idx_del = np.array([])
    for i in feature_without_variance:
        w = np.where(shap_names_main == names_features[i])[0]
        r = np.arange(w, n_shap, len(shap_names_main))
        idx_del = np.concatenate((idx_del, r))
    idx_del = 1 + idx_del
    taxa_shap = np.delete(taxa_shap, idx_del.astype(int), axis = 0)
    return taxa_shap


def k_add_kernel_shap(mcmc_file, pkl_file, burnin, thin, num_processes = 1, combine_discr_features = "", show_progressbar = False):
    bdnn_obj, post_w_sp, post_w_ex, sp_fad_lad, post_ts, post_te = bdnn_parse_results(mcmc_file, pkl_file, burnin, thin)
    mcmc_samples = post_ts.shape[0]
    trt_tbls = bdnn_obj.trait_tbls
    n_species = trt_tbls[0].shape[-2]
    names_features_sp = get_names_features(bdnn_obj)
    names_features_ex = copy_lib.deepcopy(names_features_sp)
    bdnn_dd = 'diversity' in names_features_sp
    hidden_act_f = bdnn_obj.bdnn_settings['hidden_act_f']
    out_act_f = bdnn_obj.bdnn_settings['out_act_f']
    idx_comb_feat_sp = get_idx_comb_feat(names_features_sp, combine_discr_features)
    idx_comb_feat_ex = get_idx_comb_feat(names_features_ex, combine_discr_features)
    shap_names_sp = make_shap_names(names_features_sp, idx_comb_feat_sp, combine_discr_features)
    shap_names_ex = make_shap_names(names_features_ex, idx_comb_feat_ex, combine_discr_features)
    n_main_eff_sp = np.sum(shap_names_sp[:,1] == 'none')
    n_main_eff_ex = np.sum(shap_names_ex[:, 1] == 'none')
    n_effects_sp = shap_names_sp.shape[0] + 1 + n_species * n_main_eff_sp
    n_effects_ex = shap_names_ex.shape[0] + 1 + n_species * n_main_eff_ex
    args = []
    for i in range(mcmc_samples):
        a = [bdnn_obj, post_ts[i, :], post_te[i, :], post_w_sp[i], post_w_ex[i],
             hidden_act_f, out_act_f, trt_tbls, bdnn_dd, idx_comb_feat_sp, idx_comb_feat_ex]
        args.append(a)
    unixos = is_unix()
    if unixos and num_processes > 1:
        pool_perm = multiprocessing.Pool(num_processes)
        shap_values = list(tqdm(pool_perm.imap_unordered(k_add_kernel_shap_i, args),
                                total = mcmc_samples, disable = show_progressbar == False))
        pool_perm.close()
    else:
        shap_values = []
        for i in tqdm(range(mcmc_samples), disable = show_progressbar == False):
            shap_values.append(k_add_kernel_shap_i(args[i]))
    shap_values = np.vstack(shap_values)
    shap_summary = get_rates_summary(shap_values.T)
    mean_shap_sp = shap_summary[:shap_names_sp.shape[0], :]
    mean_shap_ex = shap_summary[n_effects_sp:(n_effects_sp + shap_names_ex.shape[0]), :]
    taxa_shap_sp = shap_summary[shap_names_sp.shape[0]:n_effects_sp, :] # First row is baseline
    taxa_shap_ex = shap_summary[(n_effects_sp + shap_names_ex.shape[0]):, :]
    feature_without_variance_sp = get_idx_feature_without_variance(trt_tbls[0])
    feature_without_variance_ex = get_idx_feature_without_variance(trt_tbls[1])
    remove_sp = []
    for i in feature_without_variance_sp:
        remove_sp.append(np.where(shap_names_sp[:, 0] == names_features_sp[i])[0])
        remove_sp.append(np.where(shap_names_sp[:, 1] == names_features_sp[i])[0])
    remove_ex = []
    for i in feature_without_variance_ex:
        remove_ex.append(np.where(shap_names_ex[:, 0] == names_features_ex[i])[0])
        remove_ex.append(np.where(shap_names_ex[:, 1] == names_features_ex[i])[0])
    remove_sp = np.array(list(pd.core.common.flatten(remove_sp))).astype(int)
    remove_ex = np.array(list(pd.core.common.flatten(remove_ex))).astype(int)
    mean_shap_sp = np.delete(mean_shap_sp, remove_sp[remove_sp < len(mean_shap_sp)], axis = 0)
    mean_shap_ex = np.delete(mean_shap_ex, remove_ex[remove_ex < len(mean_shap_ex)], axis = 0)
    shap_names_sp_del = np.delete(shap_names_sp, remove_sp, axis = 0)
    shap_names_ex_del = np.delete(shap_names_ex, remove_ex, axis = 0)
    shap_values_sp = pd.DataFrame(mean_shap_sp, columns = ['shap', 'lwr_shap', 'upr_shap'])
    shap_values_ex = pd.DataFrame(mean_shap_ex, columns = ['shap', 'lwr_shap', 'upr_shap'])
    shap_names_sp_del = pd.DataFrame(shap_names_sp_del, columns = ['feature1', 'feature2'])
    shap_names_ex_del = pd.DataFrame(shap_names_ex_del, columns = ['feature1', 'feature2'])
    shap_lam = pd.concat([shap_names_sp_del, shap_values_sp], axis = 1)
    shap_ex = pd.concat([shap_names_ex_del, shap_values_ex], axis = 1)
    taxa_names = sp_fad_lad["Taxon"]
    taxa_names_shap_sp = make_taxa_names_shap(taxa_names, n_species, shap_names_sp_del)
    taxa_names_shap_ex = make_taxa_names_shap(taxa_names, n_species, shap_names_ex_del)
    taxa_shap_sp = delete_invariantfeat_from_taxa_shap(feature_without_variance_sp, names_features_sp,
                                                       shap_names_sp, taxa_shap_sp)
    taxa_shap_ex = delete_invariantfeat_from_taxa_shap(feature_without_variance_ex, names_features_ex,
                                                       shap_names_ex, taxa_shap_ex)
    taxa_shap_sp = pd.DataFrame(taxa_shap_sp, columns = ['shap', 'lwr_shap', 'upr_shap'])
    taxa_shap_ex = pd.DataFrame(taxa_shap_ex, columns = ['shap', 'lwr_shap', 'upr_shap'])
    taxa_names_shap_sp = pd.DataFrame(taxa_names_shap_sp, columns = ['feature'])
    taxa_names_shap_ex = pd.DataFrame(taxa_names_shap_ex, columns = ['feature'])
    taxa_shap_sp = pd.concat([taxa_names_shap_sp, taxa_shap_sp], axis = 1)
    taxa_shap_ex = pd.concat([taxa_names_shap_ex, taxa_shap_ex], axis = 1)
    return shap_lam, shap_ex, taxa_shap_sp, taxa_shap_ex


# rank aggregation
##################
def scorematrix(r):
    l = len(r)
    sm = np.zeros((l, l))
    a = np.arange(l)
    for i in range(l):
        ind = np.setdiff1d(a, i)
        diffs = np.sign(r[i] - r[ind])
        sm[i, ind] = diffs
    idn = np.isnan(sm)
    sm = ((sm <= 0) * 2 - 1) - np.eye(l)
    sm[idn] = 0
    return sm


def combinpmatr(rm):
    nrow, ncol = rm.shape
    ci = np.zeros((ncol, ncol))
    for i in range(nrow):
        ci += scorematrix(rm[i, :])
    return ci


def findconsensusBB(cij):
    N = cij.shape[1]
    x = np.ones(N)
    indici = list(combinations(range(N), 2))
    for j in range(len(indici)):
        indj0 = indici[j][0]
        indj1 = indici[j][1]
        a = np.sign(cij[indj0, indj1])
        b = np.sign(cij[indj1, indj0])
        if a == 1 and b == -1:
            x[indj0] = x[indj0] + 1
        elif a == -1 and b == 1:
            x[indj1] = x[indj1] + 1
        elif a == -1 and b == -1:
            x[indj0] = np.nan
        elif a == 1 and b == 1:
            x[indj0] = x[indj0] + 1
            x[indj1] = x[indj1] + 1
    x = N - x
    return x


def reorderingBB(rr):
    rr = rr + 1
    r = rr + 0
    k = len(r)
    neword = np.argsort(r)
    # indexing = np.zeros(k - 1)
    # I do not get this right using a loop
    # for j in range((k - 1), 0, -1):
    #     indexing[j] = r[neword[j + 1]] - r[neword[j]]
    j = np.arange(k - 2, -1, -1)
    indexing = r[neword[j + 1]] - r[neword[j]]
    if np.sum(indexing == 0) > 0:
        J = 0
        while J <= (k - 2):
            if indexing[J] == 0:
                r[neword[J + 1]] = r[neword[J]]
                J = J + 1
            elif indexing[J] > 0:
                r[neword[J + 1]] = r[neword[J]] + 2
                J = J + 1
    else:
        J = 0
        while J <= (k - 2):
            r[neword[J + 1]] = r[neword[J]] + 2
            J = J + 1
    return r


def reordering(X):
    s = X.shape
    G = np.zeros(s)
    for j in range(s[0]):
        OX = np.argsort(X[j, :])
        SX = np.sort(X[j, :])
        SX = SX - np.min(SX) + 1
        DC = np.concatenate((np.zeros(1), np.diff(SX)))
        for i in range(s[1] - 1):
            if DC[i + 1] >= 1:
                SX[i + 1] = SX[i] + 1
            elif DC[i + 1] == 0:
                SX[i + 1] = SX[i]
        G[j, OX] = SX - 1
    return G


def penaltyBB2(cij, candidate, ord):
    l = len(ord) - 1
    ds = np.zeros(l)
    addpenalty = np.zeros(l)
    last_ord = ord[-1]
    for k in range(l):
        k_ord = ord[k]
        a = cij[last_ord, k_ord]
        b = cij[k_ord, last_ord]
        sign_a = np.sign(a)
        sign_b = np.sign(b)
        ds[k] = np.sign(candidate[last_ord] - candidate[k_ord])
        if ds[k] == 1:
            if sign_a == 1 and sign_b == -1:
                addpenalty[k] = a - b
            elif (sign_a == 1 and sign_b == 1) or (sign_a == 0 and sign_b == 0) or (sign_a == a and sign_b == 0) or (sign_a == 1 and sign_b == 1):
                addpenalty[k] = a
            elif sign_a == -1 and sign_b == 1:
                addpenalty[k] = 0
        elif ds[k] == -1:
            if sign_a == 1 and sign_b == -1:
                addpenalty[k] = 0
            elif (sign_a == 1 and sign_b == 1) or (sign_a == 0 and sign_b == 0) or (sign_a == a and sign_b == 0) or (sign_a == 1 and sign_b == 1):
                addpenalty[k] = b
            elif sign_a == -1 and sign_b == 1:
                addpenalty[k] = b - a
        elif ds[k] == 0:
            if sign_a == 1 and sign_b == -1:
                addpenalty[k] = -b
            elif (sign_a == 1 and sign_b == 1) or (sign_a == 0 and sign_b == 0) or (sign_a == a and sign_b == 0) or (sign_a == 1 and sign_b == 1):
                addpenalty[k] = 0
            elif sign_a == -1 and sign_b == 1:
                addpenalty[k] = -a
    return np.sum(addpenalty)


def BBconsensus(rr, cij):
    cr = rr + 0
    sij = scorematrix(rr)
    po = np.sum(np.abs(cij)) - np.sum(cij * sij)
    a = np.sort(rr)[::-1]
    la = len(a)
    ord = np.argsort(rr)[::-1]
    r = rr + 0
    addpenalty = np.zeros(la)
    for k in range(1, la):
        b = np.arange(k + 1)
        r = reorderingBB(r)
        kr = r[ord[b]]
        kr = kr[:-1]
        mo = np.max(kr)
        mi = np.min(kr)
        aa = 0
        ko = 1
        kr = np.append(kr, mo + 1)
        r[ord[b]] = kr
        candidate = np.zeros(rr.shape)
        pb = np.zeros(1)
        while ko == 1:
            candidate = np.vstack((candidate, r))
            if aa == 0:
                candidate = np.delete(candidate, 0, axis = 0)
            Sij = scorematrix(candidate[aa, :])
            pb = np.append(pb, np.sum(np.abs(cij)) - np.sum(cij * Sij))
            if aa == 0:
                pb = np.delete(pb, 0)
            if pb[aa] == 0:
                cr = r + 0
                po = np.zeros(1, dtype = int)
                pc = np.zeros(1, dtype = int)
                break
            pc = 1
            r[ord[b[-1]]] = r[ord[b[-1]]] - 1
            if (mi - r[ord[b[-1]]]) > 1:
                ko = 0
            aa += 1
        if pc == 0:
            break
        minp = np.min(pb)
        posp = np.argmin(pb)
        if minp <= po:
            po = minp + 0
            cr = candidate[posp, :].flatten()
        r = cr + 0
        addpenalty[k] = penaltyBB2(cij, r, ord[b])
        candidate = np.zeros(r.shape)
        pb = np.zeros(1)
    if pc == 0:
        #po = np.zeros(1)
        addpenalty = np.zeros(1)
    # else:
    #     poo = np.sum(addpenalty) # ??? poo not used
    # SIJ = scorematrix(cr)
    po = np.sum(addpenalty)
    return [cr, po]


def quickcons(X):
    M, N = X.shape
    if M == 1:
        consensus = X
        taux = 1
    else:
        cij = combinpmatr(X)
        R = findconsensusBB(cij)
        R1 = (N - 1) - R
        consensusA = BBconsensus(R, cij)[0]
        consensusB = BBconsensus(consensusA, cij)[0]
        consensusC = BBconsensus(R1, cij)[0]
        consensusD = BBconsensus(consensusC, cij)[0]
        all_consensus = np.vstack((consensusA, consensusB, consensusC, consensusD))
        consensus = np.unique(reordering(all_consensus), axis = 0)
        howcons = consensus.shape[0]
        taux = np.zeros(howcons)
        for k in range(howcons):
            Sij = scorematrix(consensus[k, :])
            taux[k] = np.sum(cij * Sij) / (M * (N * (N - 1)))
        if howcons > 1:
            nco = np.where(taux == np.max(taux))[0]
            consensus = consensus[nco, :]
            taux = taux[nco]
    return [consensus, taux]


def highest_pvalue_from_interaction(p):
    unique_features = p[['feature1', 'feature2']].drop_duplicates(ignore_index = True)
    num_unique_features = len(unique_features)
    columns = ['posterior_probability', 'magnitude_effect', 'magnitude_lwr_CI', 'magnitude_upr_CI']
    highest_p = pd.DataFrame(columns = columns)
    for i in range(num_unique_features):
        p_tmp = p[(p['feature1'] == unique_features.loc[i, 'feature1']) &
                  (p['feature2'] == unique_features.loc[i, 'feature2'])]
        p_tmp = p_tmp.reset_index()
        h = p_tmp['posterior_probability'].argmax()
        has_probs = h != -1
        if has_probs:
            p_tmp = p_tmp.loc[h, columns].to_frame().T
        else:
            a = np.zeros(4)
            a[:] = np.nan
            p_tmp = pd.DataFrame(a.reshape((1, 4)), columns = columns)
        highest_p = pd.concat([highest_p, p_tmp], axis = 0, ignore_index = True)
    highest_p = pd.concat([unique_features, highest_p], axis = 1)
    return highest_p


def get_same_order(pv, sh, fp):
    pv_reord = highest_pvalue_from_interaction(pv)
    nrows = len(pv_reord)
    sh_reord = pd.DataFrame(columns = ['feature1', 'feature2', 'shap', 'lwr_shap', 'upr_shap'])
    fp_reord = pd.DataFrame(columns = ['feature1', 'feature2', 'delta_lik', 'lwr_delta_lik', 'upr_delta_lik'])
    for i in range(nrows):
        sh_tmp = sh[(sh['feature1'] == pv_reord.loc[i, 'feature1']) &
                    (sh['feature2'] == pv_reord.loc[i, 'feature2']) |
                    (sh['feature1'] == pv_reord.loc[i, 'feature2']) &
                    (sh['feature2'] == pv_reord.loc[i, 'feature1'])]
        fp_tmp = fp[(fp['feature1'] == pv_reord.loc[i, 'feature1']) &
                    (fp['feature2'] == pv_reord.loc[i, 'feature2']) |
                    (fp['feature1'] == pv_reord.loc[i, 'feature2']) &
                    (fp['feature2'] == pv_reord.loc[i, 'feature1'])]
        sh_reord = pd.concat([sh_reord, sh_tmp], axis = 0, ignore_index = True)
        fp_reord = pd.concat([fp_reord, fp_tmp], axis = 0, ignore_index = True)
    return pv_reord, sh_reord, fp_reord


def stack_importance(pv_reord, sh_reord, fp_reord):
    stacked_importance = np.stack((1 - pv_reord['posterior_probability'].to_numpy().astype(float),
                                  -1 * sh_reord['shap'].to_numpy().astype(float),
                                  fp_reord['delta_lik'].to_numpy().astype(float)), axis = 0)
    return stacked_importance


def rank_features(pv_reord, sh_reord, fp_reord):
    feat_importance = stack_importance(pv_reord, sh_reord, fp_reord)
    idx_main = pv_reord['feature2'] == 'none'
    feat_importance_main = feat_importance[:, idx_main]
    feat_importance_inter = feat_importance[:, idx_main == False]
#    # Needs scipy >= 1.10 Too modern?
#    ranked_feat_import_main = stats.rankdata(feat_importance_main, axis = 1, method = 'min', nan_policy = 'omit')
#    ranked_feat_import_inter = stats.rankdata(feat_importance_inter, axis = 1, method = 'min', nan_policy = 'omit')
    nan_idx_main = np.isnan(feat_importance_main)
    nan_idx_inter = np.isnan(feat_importance_inter)
    feat_importance_main[nan_idx_main] = np.nanmax(feat_importance_main) + 1
    feat_importance_inter[nan_idx_inter] = np.nanmax(feat_importance_inter) + 1
    ranked_feat_import_main = stats.rankdata(feat_importance_main, axis = 1, method = 'min').astype(float)
    ranked_feat_import_inter = stats.rankdata(feat_importance_inter, axis = 1, method = 'min').astype(float)
    ranked_feat_import_main[nan_idx_main] = np.nan
    ranked_feat_import_inter[nan_idx_inter] = np.nan
    return ranked_feat_import_main, ranked_feat_import_inter


def merge_results_feat_import(pv, sh, fp, rr):
    nrows = len(pv)
    sh_merge = pd.DataFrame(columns = ['shap', 'lwr_shap', 'upr_shap'])
    fp_merge = pd.DataFrame(columns = ['delta_lik', 'lwr_delta_lik', 'upr_delta_lik'])
    rr_merge = pd.DataFrame(columns = ['rank'])
    for i in range(nrows):
        pv_feat1 = pv.loc[i, 'feature1']
        pv_feat2 = pv.loc[i, 'feature2']
        sh_tmp = sh[(sh['feature1'] == pv_feat1) &
                    (sh['feature2'] == pv_feat2) |
                    (sh['feature1'] == pv_feat2) &
                    (sh['feature2'] == pv_feat1)]
        fp_tmp = fp[(fp['feature1'] == pv_feat1) &
                    (fp['feature2'] == pv_feat2) |
                    (fp['feature1'] == pv_feat2) &
                    (fp['feature2'] == pv_feat1)]
        rr_tmp = rr[(rr['feature1'] == pv_feat1) &
                    (rr['feature2'] == pv_feat2) |
                    (rr['feature1'] == pv_feat2) &
                    (rr['feature2'] == pv_feat1)]
        sh_merge = pd.concat([sh_merge, sh_tmp[['shap', 'lwr_shap', 'upr_shap']]], axis = 0, ignore_index = True)
        fp_merge = pd.concat([fp_merge, fp_tmp[['delta_lik', 'lwr_delta_lik', 'upr_delta_lik']]], axis = 0, ignore_index = True)
        rr_merge = pd.concat([rr_merge, rr_tmp[['rank']]], axis = 0, ignore_index = True)
    merged = pd.concat([pv, sh_merge, fp_merge, rr_merge], axis = 1)
    return merged


def get_consensus_ranking(pv, sh, fp):
    pv_reordered, sh_reordered, fp_reordered = get_same_order(pv, sh, fp)
    feat_main_ranked, feat_inter_ranked = rank_features(pv_reordered, sh_reordered, fp_reordered)
    main_consranks = quickcons(feat_main_ranked)
    main_consrank = np.mean(main_consranks[0], axis = 0).flatten()
    inter_consranks = quickcons(feat_inter_ranked)
    inter_consrank = np.mean(inter_consranks[0], axis = 0).flatten()
    rank_df = pd.DataFrame(np.concatenate((main_consrank, inter_consrank)) + 1.0, columns = ['rank'])
    r = pd.concat([pv_reordered[['feature1', 'feature2']], rank_df], axis = 1)
    feat_merged = merge_results_feat_import(pv, sh, fp, r)
    return feat_merged
