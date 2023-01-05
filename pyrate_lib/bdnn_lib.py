import os, platform, glob, sys
from math import comb
import itertools
import numpy as np
import pandas as pd
import copy as copy_lib

from scipy.spatial import ConvexHull
from matplotlib.path import Path

import pyrate_lib.lib_utilities as util
from PyRate import check_burnin
from PyRate import load_pkl
from PyRate import get_rate_BDNN



def summarize_rate(r, n_rates):
    r_sum = np.zeros((n_rates, 3))
    r_sum[:, 0] = np.mean(r, axis = 0)
    for i in range(n_rates):
        r_sum[i, 1:] = util.calcHPD(r[:, i], 0.95)
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
    r_script += "\nylim = c(%s, %s)" % (np.min(r_sp_sum), np.max(r_sp_sum))
    r_script += "\nplot(time_vec, sp_mean, type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Speciation rate')"
    r_script += "\npolygon(c(time_vec, rev(time_vec)), c(sp_lwr, rev(sp_upr)), col = adjustcolor('#4c4cec', alpha = 0.5), border = NA)"
    r_script += "\nlines(time_vec, sp_mean, col = '#4c4cec', lwd = 2)"
    r_script += "\nylim = c(%s, %s)" % (np.min(r_ex_sum), np.max(r_ex_sum))
    r_script += "\nplot(time_vec, ex_mean, type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Extinction rate')"
    r_script += "\npolygon(c(time_vec, rev(time_vec)), c(ex_lwr, rev(ex_upr)), col = adjustcolor('#e34a33', alpha = 0.5), border = NA)"
    r_script += "\nlines(time_vec, ex_mean, col = '#e34a33', lwd = 2)"
    r_script += "\nylim = c(%s, %s)" % (np.min(r_div_sum), np.max(r_div_sum))
    r_script += "\nplot(time_vec, div_mean, type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Net diversification rate')"
    r_script += "\npolygon(c(time_vec, rev(time_vec)), c(div_lwr, rev(div_upr)), col = adjustcolor('black', alpha = 0.3), border = NA)"
    r_script += "\nlines(time_vec, div_mean, col = 'black', lwd = 2)"
    r_script += "\nabline(h = 0, col = 'red', lty = 2)"
    r_script += "\nylim = c(%s, %s)" % (np.min(long_sum), np.max(long_sum))
    r_script += "\nplot(time_vec, div_mean, type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Longevity (Myr)')"
    r_script += "\npolygon(c(time_vec, rev(time_vec)), c(long_lwr, rev(long_upr)), col = adjustcolor('black', alpha = 0.3), border = NA)"
    r_script += "\nlines(time_vec, long_mean, col = 'black', lwd = 2)"
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
        cond_trait_tbl_sp = backscale_tbl(backscale_par, names_feat, cond_trait_tbl_sp)
        cond_trait_tbl_ex = backscale_tbl(backscale_par, names_feat, cond_trait_tbl_ex)
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
    if len(trait_tbl.shape) == 3:
        for i in range(n_features):
            tr = trait_tbl[time_bins, :, i]
            std_tr = np.std(tr, axis = 0)
            time_var_feat[i] = np.any(std_tr > 1e-10)
    time_var_feat = time_var_feat.astype(bool)
    return time_var_feat


def get_idx_feature_without_variance(m):
    return np.where((m[0, :] - m[1, :]) == 0.0)[0]


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
    if fg !="":
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
    inter = np.array(list(itertools.combinations(range(0, n), 2)))
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
                if np.isin(w[k, 0], observed_comb[:, 0]) and np.isin(w[k, 1], observed_comb[:, 1]):
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
                elif (int(i_time_var) + int(j_time_var)) == 1:
                    # Only one feature varies through time
                    t = trait_at_fad_or_lad + 0.0
                else:
                    # Both features vary through time (There must be something better than allowing all combinations)
                    t = np.zeros((4, trait_tbl.shape[2]))
                    min_i = np.min(trait_tbl[:, :, i])
                    max_i = np.max(trait_tbl[:, :, i])
                    min_j = np.min(trait_tbl[:, :, j])
                    max_j = np.max(trait_tbl[:, :, j])
                    t[0, i] = min_i
                    t[0, j] = min_j
                    t[1, i] = max_i
                    t[1, j] = min_j
                    t[2, i] = max_i
                    t[2, j] = max_j
                    t[3, i] = min_i
                    t[3, j] = max_j
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
    feature_without_variance = get_idx_feature_without_variance(minmaxmean_features)
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


def backscale_tbl(backscale_par, names_feat, tbl):
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
    rate_cond = np.zeros((len(cond_trait_tbl), num_it))
    counter = 0
    for i in range(num_it):
        rate_cond[:, i] = get_rate_BDNN(1, # constant baseline
                                        cond_trait_tbl[:, :-6],
                                        post_w[i], # list of arrays
                                        bdnn_obj.bdnn_settings['hidden_act_f'],
                                        bdnn_obj.bdnn_settings['out_act_f'])
        counter += 1
    return rate_cond


def get_rates_summary(cond_rates):
    nrows_cond_rates = cond_rates.shape[0]
    rate_sum = np.zeros((nrows_cond_rates, 3))
    rate_sum[:, 0] = np.mean(cond_rates, axis = 1)
    for i in range(nrows_cond_rates):
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
    rate_max = np.nanmax(rs[:, 2]) * 1.2
    rate_min = np.nanmin(rs[:, 1]) * 0.8
    r_script += "\nylim = c(%s, %s)" % (rate_min, rate_max)
    r_script += "\nxlim = c(-0.5, %s + 0.5)" % (n_states - 1)
    r_script += "\nplot(1, 0, type = 'n', xlim = xlim, ylim = ylim, xlab = '', ylab = '%s', xaxt = 'n')" % rate_type
    if rate_type == 'speciation':
        r_script += "\ncol = colorRampPalette(c('lightblue1', rgb(0, 52, 94, maxColorValue = 255)))(%s)" % n_states
    else:
        r_script += "\ncol = colorRampPalette(c(rgb(255, 143, 118, maxColorValue = 255), 'darkred'))(%s)" % n_states
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
    col = '#C5483B'
    if rate_type == 'speciation':
        col = '#6092AF'
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
    else:
        r_script += "\ncol = colorRampPalette(c(rgb(255, 143, 118, maxColorValue = 255), 'darkred'))(%s)" % n_states
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
    r_script += "\nylim = c(%s, %s)" % (float(np.nanmin(tr[:, 1])), float(np.nanmax(tr[:, 1])))
    if plot_time:
        r_script += "\nylim = ylim[2:1]"
    nr = np.sqrt(rs.shape[0])
    if rate_type == 'speciation':
        r_script += "\ncol = colorRampPalette(c('lightblue1', rgb(0, 52, 94, maxColorValue = 255)))(%s)" % nr
    else:
        r_script += "\ncol = colorRampPalette(c(rgb(255, 143, 118, maxColorValue = 255), 'darkred'))(%s)" % nr
    r_script += util.print_R_vec("\nx", tr[:, 0])
    r_script += util.print_R_vec("\ny", tr[:, 1])
    r_script += util.print_R_vec("\nr", rs[:, 0])
    r_script += "\nxyr <- cbind(x, y, r)"
    r_script += "\nxaxis <- sort(unique(xyr[, 1]))"
    r_script += "\nyaxis <- sort(unique(xyr[, 2]))"
    r_script += "\nz <- matrix(xyr[, 3], length(xaxis), length(yaxis))"
    r_script += "\nimage.plot(xaxis, yaxis, z, ylim = ylim, col = col, xlab = '%s', ylab = '%s')" % (names[0], names[1])
    r_script += "\ncontour(xaxis, yaxis, z, col = 'grey50', add = TRUE)"
    r_script += util.print_R_vec("\nobs_x", obs[:, 0])
    r_script += util.print_R_vec("\nobs_y", obs[:, 1])
    r_script += "\npoints(obs_x, obs_y, cex = 0.5, pch = 19, col = 'grey50')"
    r_script += "\nbox()"
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
    rate_max = np.nanmax(rs[:, 2]) * 1.2
    rate_min = np.nanmin(rs[:, 1]) * 0.8
    r_script += "\npar(las = 2, mar = c(9, 4, 1.5, 0.5))"
    r_script += "\nylim = c(%s, %s)" % (rate_min, rate_max)
    r_script += "\nxlim = c(-0.5, %s + 0.5)" % (n_states_feat_1 * n_states_feat_2 - 1)
    r_script += "\nplot(0, 0, type = 'n', xlim = xlim, ylim = ylim, xlab = '', ylab = '%s', xaxt = 'n')" % (rate_type)
    if rate_type == 'speciation':
        r_script += "\ncol = colorRampPalette(c('lightblue1', rgb(0, 52, 94, maxColorValue = 255)))(%s)" % n_states_feat_2
    else:
        r_script += "\ncol = colorRampPalette(c(rgb(255, 143, 118, maxColorValue = 255), 'darkred'))(%s)" % n_states_feat_2
    counter = 0
    for i in range(n_states_feat_1):
        for j in range(n_states_feat_2):
            idx = np.logical_and(states_feat_1 == i, states_feat_2 == j)
            r_tmp = r[idx, :]
            r_tmp = r_tmp[r_tmp < rate_max]
            r_tmp = r_tmp[r_tmp > rate_min]
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
    r_script += "\n    install.packages(new_pkgs)"
    r_script += "\n}"
    r_script += "\nsuppressPackageStartupMessages(library(fields))"
    r_script += "\nsuppressPackageStartupMessages(library(vioplot))"
    r_script += "\npar(las = 1, mar = c(4, 4, 1.5, 0.5))"
    plot_idx = np.unique(cond_trait_tbl[:, -3])
    n_plots = len(plot_idx)
    rates_summary = get_rates_summary(cond_rates)
    # set summary to NA when we have not observed the combination of features
    not_obs = cond_trait_tbl[:, -1] == 0
    rates_summary[not_obs,:] = np.nan
    cond_rates[not_obs,:] = np.nan
    time_idx = np.nanmax(cond_trait_tbl[:, -3]) + 10.0
    if is_time_trait(bdnn_obj):
        time_idx = np.max(cond_trait_tbl[:, -6])
    trait_tbl = get_trt_tbl(bdnn_obj, rate_type)
    feature_is_time_variable = is_time_variable_feature(trait_tbl)
    fad_lad = sp_fad_lad[['FAD', 'LAD']].to_numpy()
    fossil_age = get_fossil_age(bdnn_obj, fad_lad, rate_type)
    fossil_age = backscale_bdnn_time(fossil_age, bdnn_obj)
    fossil_bin = get_bin_from_fossil_age(bdnn_obj, fad_lad, rate_type)
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
        obs = get_observed(bdnn_obj, incl_features, feature_is_time_variable, fossil_age, fossil_bin, rate_type)
        plot_time = np.isin(time_idx, incl_features)
        if np.isin(pt, np.array([1.0, 2.0, 3.0])):
            names = names_features[incl_features[0]]
            names_states = trait_tbl_plt[:, 0].tolist()
            if pt == 2.0:
                names_states = names_features_original[incl_features].tolist()
            r_script = plot_bdnn_discr(rates_sum_plt, cond_rates_plt, trait_tbl_plt, r_script, names, names_states, rate_type)
        elif pt == 4.0:
            names = names_features[incl_features[0]]
            obs = backscale_tbl(backscale_par, [names], obs)
            r_script = plot_bdnn_cont(rates_sum_plt, trait_tbl_plt, r_script, names, plot_time, obs, rate_type)
        elif np.isin(pt, np.array([6.0, 13.0, 14.0])):
            b = binary_feature[incl_features]
            names_states = np.unique(trait_tbl_plt[:, b]).tolist()
            if pt == 13.0:
                names_states = names_features_original[incl_features][b]
            names = names[np.argsort(b)]
            trait_tbl_plt = trait_tbl_plt[:, np.argsort(b)] # Continuous feature always in column 0
            obs = obs[:, np.argsort(b)]
            obs[:,0] = backscale_tbl(backscale_par, [names[0]], obs[:,0].reshape((obs.shape[0],1))).flatten()
            r_script = plot_bdnn_inter_discr_cont(rates_sum_plt, trait_tbl_plt, r_script, names, names_states, plot_time, obs, rate_type)
        elif pt == 7.0:
            obs = backscale_tbl(backscale_par, names.tolist(), obs)
            r_script = plot_bdnn_inter_cont_cont(rates_sum_plt, trait_tbl_plt, r_script, names, plot_time, obs, rate_type)
        elif np.isin(pt, np.array([5.0, 8.0, 9.0, 10.0, 11.00, 12.0])):
            names = np.unique(names_features[incl_features])
            if np.isin(pt, np.array([5.0, 9.0, 12.0])):
                feat_1 = np.array([0])
                feat_2 = np.array([1])
                names_states_feat_1 = np.unique(trait_tbl_plt[:, feat_1]).tolist()
                names_states_feat_2 = np.unique(trait_tbl_plt[:, feat_2]).tolist()
            if np.isin(pt, np.array([8.0, 10.0, 11.0])):
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


def plot_effects(f,
                 cond_trait_tbl_sp,
                 cond_trait_tbl_ex,
                 sp_rate_cond,
                 ex_rate_cond,
                 bdnn_obj,
                 sp_fad_lad,
                 backscale_par,
                 names_features_sp,
                 names_features_ex):
    # Plot feature-rate relationship
    output_wd = os.path.dirname(f)
    name_file = os.path.basename(f)
    out = "%s/%s_effects.r" % (output_wd, name_file)
    newfile = open(out, "w")
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        wd_forward = os.path.abspath(output_wd).replace('\\', '/')
        r_script = "pdf(file='%s/%s_effects.pdf', width = 7, height = 6, useDingbats = FALSE)\n" % (wd_forward, name_file)
    else:
        r_script = "pdf(file='%s/%s_effects.pdf', width = 7, height = 6, useDingbats = FALSE)\n" % (output_wd, name_file)
    r_script = create_R_files_effects(cond_trait_tbl_sp, sp_rate_cond, bdnn_obj, sp_fad_lad, r_script, names_features_sp,
                                      backscale_par, rate_type = 'speciation')
    r_script = create_R_files_effects(cond_trait_tbl_ex, ex_rate_cond, bdnn_obj, sp_fad_lad, r_script, names_features_ex,
                                      backscale_par, rate_type = 'extinction')
    r_script += "\ndev.off()"
    newfile.writelines(r_script)
    newfile.close()
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        cmd = "cd %s & Rscript %s_effects.r" % (output_wd, name_file)
    else:
        cmd = "cd %s; Rscript %s_effects.r" % (output_wd, name_file)
        print("cmd", cmd)
        os.system(cmd)


