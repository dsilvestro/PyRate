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
from collections.abc import Iterable

import pyrate_lib.lib_utilities as util
from PyRate import check_burnin
from PyRate import load_pkl
from PyRate import write_pkl
from PyRate import get_rate_BDNN
from PyRate import get_DT
from PyRate import get_binned_div_traj
from PyRate import get_sp_in_frame_br_length
from PyRate import bdnn
from PyRate import MatrixMultiplication
from PyRate import MatrixMultiplication3D
from PyRate import init_weight_prm
from PyRate import update_parameter_normal_vec
from PyRate import update_parameter
from PyRate import update_multiplier_proposal
from PyRate import update_q_multiplier
from PyRate import get_events_ns
from PyRate import BDNN_fast_partial_lik
from PyRate import HPP_NN_lik
from PyRate import HOMPP_NN_lik
from PyRate import get_sp_indx_in_timeframe
from PyRate import get_time_in_q_bins
from PyRate import get_occs_sp
from PyRate import get_fossil_features_q_shifts
from PyRate import make_singleton_mask
from PyRate import get_qbin_ts_te
from PyRate import get_q_rate_BDNN
from PyRate import harmonic_mean_q_per_sp
from PyRate import prior_gamma
from PyRate import add_taxon_age

from scipy.special import bernoulli, binom
from itertools import chain, combinations, product
import random

#import fastshap
#from fastshap.plotting import get_variable_interactions
# fastshap stuff
from pandas import DataFrame as pd_DataFrame
from pandas import Series as pd_Series
from pandas import concat
from sklearn.linear_model import LinearRegression

small_number= 1e-50


def load_trait_tbl(path):
    loaded_trait_tbls = []
    sp_pred_names_tbls = sorted(glob.glob(os.path.join(path[0], "*")))
    sp_pred_tbls = []
    print('\nOrder taxon-time specific speciation tables:')
    if len(sp_pred_names_tbls) > 1:
        for t in sp_pred_names_tbls:
            print(os.path.basename(t))
            sp_tbl = np.loadtxt(t, skiprows = 1)
            sp_pred_tbls.append(sp_tbl)
        sp_pred_tbls = np.array(sp_pred_tbls)
    else:
        print(os.path.basename(sp_pred_names_tbls[0]))
        sp_pred_tbls = np.loadtxt(sp_pred_names_tbls[0], skiprows = 1)
    loaded_trait_tbls.append(sp_pred_tbls)
    ex_pred_names_tbls = sorted(glob.glob(os.path.join(path[len(path) - 1], "*")))
    ex_pred_tbls = []
    print('\nOrder taxon-time specific extinction tables:')
    if len(ex_pred_names_tbls) > 1:
        for t in ex_pred_names_tbls:
            print(os.path.basename(t))
            ex_tbl = np.loadtxt(t, skiprows = 1)
            ex_pred_tbls.append(ex_tbl)
        ex_pred_tbls = np.array(ex_pred_tbls)
    else:
        print(ex_pred_names_tbls[0])
        ex_pred_tbls = np.loadtxt(ex_pred_names_tbls[0], skiprows = 1)
    loaded_trait_tbls.append(ex_pred_tbls)
    colnames = np.loadtxt(ex_pred_names_tbls[0], max_rows = 1, dtype = str).tolist()
    sp_time_variable_pred = is_time_variable_feature(sp_pred_tbls)[0,:]
    ex_time_variable_pred = is_time_variable_feature(ex_pred_tbls)[0,:]
    time_variable_pred = np.any(np.concatenate((sp_time_variable_pred, ex_time_variable_pred), axis = None))
    invariant_pred = [get_idx_feature_without_variance(sp_pred_tbls), get_idx_feature_without_variance(ex_pred_tbls)]
    # Should we check if all colnames are in the same order?
    return loaded_trait_tbls, colnames, time_variable_pred, invariant_pred


def export_trait_tbl(trait_tbls, names_features, output_wd):
    path_predictors = os.path.join(output_wd, 'BDNN_predictors')
    os.makedirs(path_predictors, exist_ok = True)
    trait_tbls[0] = trait_tbls[0][::-1, :, :]
    num_tbls = len(trait_tbls[0])
    digits_file_name = "{:0%sd}" % len(str(num_tbls))
    for i in range(num_tbls):
        tbl = trait_tbls[0][i]
        if 'time' in names_features:
            tbl = tbl[:, :-1]
        tbl_df = pd.DataFrame(tbl, columns = names_features[0:tbl.shape[1]])
        file_name = str(digits_file_name.format(i + 1)) + ".txt"
        tbl_df_file = os.path.join(path_predictors, file_name)
        tbl_df.to_csv(tbl_df_file, index = False, sep ='\t')
    return path_predictors


def combine_pkl(path_to_files, tag):
    infile = path_to_files
    sys.path.append(infile)
    direct_pkl = "%s/*%s*.pkl" % (infile, tag)
    print('direct_pkl', direct_pkl)
    files_pkl = glob.glob(direct_pkl)
    files_pkl = np.sort(files_pkl)
    pkl_list = []
    if len(files_pkl) > 0:
        pkl_list = [load_pkl(fp) for fp in files_pkl]
        
        bd = False
        if 'layers_shapes' in pkl_list[0].bdnn_settings.keys():
            bd = True
        q = False
        if 'layers_shapes_q' in pkl_list[0].bdnn_settings.keys():
            q = True
            time_var_q = False
            if 'q_time_frames' in pkl_list[0].bdnn_settings.keys():
                time_var_q = True

        bdnn_dict = {
            'hidden_act_f': pkl_list[0].bdnn_settings['hidden_act_f']
        }
        if 'prior_t_reg' in pkl_list[0].bdnn_settings.keys():
            bdnn_dict.update({'prior_t_reg': pkl_list[0].bdnn_settings['prior_t_reg']})
        bdnn_dict.update({'prior_cov': pkl_list[0].bdnn_settings['prior_cov']})

        num_replicates = len(pkl_list)
        if bd:
            bdnn_rescale_div = np.zeros(num_replicates)
            time_rescaler = np.zeros(num_replicates)
            n_bins = np.zeros(num_replicates)
            for i in range(num_replicates):
                bdnn_rescale_div[i] = pkl_list[i].bdnn_settings['div_rescaler']
                time_rescaler[i] = pkl_list[i].bdnn_settings['time_rescaler']
                if pkl_list[i].trait_tbls[0].ndim == 3:
                    n_bins[i] = len(pkl_list[i].bdnn_settings['fixed_times_of_shift_bdnn'])
            pkl_most_bins = np.argmax(n_bins)

            bdnn_dict.update({
                'layers_shapes': pkl_list[0].bdnn_settings['layers_shapes'],
                'layers_sizes': pkl_list[0].bdnn_settings['layers_sizes'],
                'out_act_f': pkl_list[0].bdnn_settings['out_act_f'],
                'mask_lam': pkl_list[0].bdnn_settings['mask_lam'],
                'mask_mu': pkl_list[0].bdnn_settings['mask_mu'],
                'fixed_times_of_shift_bdnn': pkl_list[pkl_most_bins].bdnn_settings['fixed_times_of_shift_bdnn'],
                'use_time_as_trait': pkl_list[0].bdnn_settings['use_time_as_trait'],
                'time_rescaler': np.mean(time_rescaler),
                'bdnn_const_baseline': pkl_list[0].bdnn_settings['bdnn_const_baseline'],
                'out_act_f': pkl_list[0].bdnn_settings['out_act_f'],
                'hidden_act_f': pkl_list[0].bdnn_settings['hidden_act_f'],
                'block_nn_model': pkl_list[0].bdnn_settings['block_nn_model'],
                'names_features': pkl_list[0].bdnn_settings['names_features'],
                'div_rescaler': np.mean(bdnn_rescale_div)
            })
        if q:
            bdnn_dict.update({
                'layers_shapes_q': pkl_list[0].bdnn_settings['layers_shapes_q'],
                'layers_sizes_q': pkl_list[0].bdnn_settings['layers_sizes_q'],
                'out_act_f_q': pkl_list[0].bdnn_settings['out_act_f_q'],
                'names_features_q': pkl_list[0].bdnn_settings['names_features_q'],
                'log_factorial_occs': pkl_list[0].bdnn_settings['log_factorial_occs'],
                'pert_prior': pkl_list[0].bdnn_settings['pert_prior']
            })
            if not time_var_q:
                bdnn_dict.update({
                    'occs_sp': pkl_list[0].bdnn_settings['occs_sp']
                })
            else:
                n_bins = np.zeros(num_replicates)
                for i in range(num_replicates):
                    n_bins[i] = pkl_list[i].bdnn_settings['duration_q_bins'].shape[1]
                pkl_most_bins = np.argmax(n_bins)
                bdnn_dict.update({
                    'occs_sp': pkl_list[pkl_most_bins].bdnn_settings['occs_sp'],
                    'q_time_frames': pkl_list[pkl_most_bins].bdnn_settings['q_time_frames'],
                    'duration_q_bins': pkl_list[pkl_most_bins].bdnn_settings['duration_q_bins'],
                    'occs_single_bin': pkl_list[0].bdnn_settings['occs_single_bin']
                })
        
        obj = bdnn(bdnn_settings=bdnn_dict,
                   weights=pkl_list[0].weights,
                   trait_tbls=pkl_list[pkl_most_bins].trait_tbls,
                   sp_fad_lad=pkl_list[0].sp_fad_lad,
                   occ_data=pkl_list[0].occ_data)
        combined_pkl_file = "%s/combined_%s%s.pkl" % (infile, num_replicates, tag)
        write_pkl(obj, combined_pkl_file)


def get_bdnn_model(pkl_file):
    ob = load_pkl(pkl_file)
    bdnn = 'layers_shapes' in ob.bdnn_settings.keys()
    qnn = 'layers_shapes_q' in ob.bdnn_settings.keys()
    if bdnn and not qnn:
        bdnn_model = 1
    if not bdnn and qnn:
        bdnn_model = 2
    if bdnn and qnn:
        bdnn_model = 3
    return bdnn_model


def read_rtt(rtt_file, burnin=0):
    with open(rtt_file, "r") as f:
        rtt = [np.array(x.strip().split(), dtype=float) for x in f.readlines()]
    del rtt[:burnin]
    return rtt


def summarize_rate(r, n_rates):
    r_sum = np.zeros((n_rates, 3))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category = RuntimeWarning)
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
    use_mean = np.isnan(r_sum[:, 1]) & ~np.isnan(r_sum[:, 0])
    if np.any(use_mean):
        r_sum[use_mean, 1:] = r_sum[use_mean, 0]
    r_sum = np.repeat(r_sum, repeats = 2, axis = 0)
    return r_sum


def make_t_vec(r_list):
    """Create time vector for rates through time plot"""
    # Get time of rate shifts. Take it from the iteration with most shifts
    time_vec = []
    num_it = len(r_list)
    for i in range(num_it):
        r_i = r_list[i]
        s1 = len(r_i)
        n_rates = int((s1 + 1) / 2)
        if (n_rates - 1) > len(time_vec):
            time_vec = r_i[n_rates:]
    return time_vec
    

def format_t_vec(t_vec, FA):
    """Format time vector for rates through time plot"""
#    a = np.abs(np.mean(np.diff(t_vec)))
#    t_vec = np.concatenate((np.array([t_vec[0] + a]), t_vec, np.zeros(1)))
    t_vec = np.concatenate((np.array([FA]), t_vec, np.zeros(1)))
    t_vec = np.repeat(t_vec, repeats = 2)
    t_vec = t_vec + np.tile(np.array([0.00001, 0.0]), int(len(t_vec)/2))
    t_vec = t_vec[1:]
    t_vec = np.delete(t_vec, -2)
    return t_vec


def get_qtt(f_q, burn):
    r = read_rtt(f_q)
    num_it = len(r)
    burnin = check_burnin(burn, num_it)
    r_q_list = read_rtt(f_q, burnin)
    time_vec_q = make_t_vec(r_q_list)
    num_it = len(r_q_list)
    r_q = np.zeros((num_it, len(time_vec_q) + 1))
    r_q[:] = np.nan
    for i in range(num_it):
        r_q_i = r_q_list[i]
        s1 = len(r_q_i)
        n_rates = int((s1 + 1) / 2)
        r_q[i, :n_rates] = r_q_i[:n_rates][::-1]
    r_q = r_q[:, ::-1]
    return r_q, time_vec_q


def get_bdnn_rtt(f, burn):
    _, _, _, post_ts, _, _, _, _, _, _, _, _ = bdnn_read_mcmc_file(f, burn, thin=0)
    FA = np.max(np.mean(post_ts, axis=0))
    f = f.replace("_mcmc.log", "")
    f_sp = f + "_sp_rates.log"
    f_ex = f + "_ex_rates.log"
    f_q = f + "_q_rates.log"
    
    try:
        r = read_rtt(f_sp)
        num_it = len(r)
        burnin = check_burnin(burn, num_it)
        r_sp_list = read_rtt(f_sp, burnin)
        r_ex_list = read_rtt(f_ex, burnin)
        time_vec = make_t_vec(r_sp_list)
        # Construct from list of rates a 2D array. Time bins earlier than the oldest fossil will contain nan
        num_it = len(r_sp_list)
        r_sp = np.zeros((num_it, len(time_vec) + 1))
        r_sp[:] = np.nan
        r_ex = np.zeros((num_it, len(time_vec) + 1))
        r_ex[:] = np.nan
        for i in range(num_it):
            r_sp_i = r_sp_list[i]
            r_ex_i = r_ex_list[i]
            s1 = len(r_sp_i)
            n_rates = int((s1 + 1) / 2)
            r_sp[i, :n_rates] = r_sp_i[:n_rates][::-1]
            r_ex[i, :n_rates] = r_ex_i[:n_rates][::-1]
        r_sp = r_sp[:, ::-1]
        r_ex = r_ex[:, ::-1]
        n_rates = r_sp.shape[1]

        time_vec = format_t_vec(time_vec, FA)
        r_div = r_sp - r_ex
        longevity = 1. / r_ex
        r_sp_sum = summarize_rate(r_sp, n_rates)
        r_ex_sum = summarize_rate(r_ex, n_rates)
        r_div_sum = summarize_rate(r_div, n_rates)
        long_sum = summarize_rate(longevity, n_rates)
    except:
        r_sp_sum = None
        r_ex_sum = None
        r_div_sum = None
        long_sum = None
        time_vec = None
    
    try:
        r_q, time_vec_q = get_qtt(f_q, burn)
        n_rates = r_q.shape[1]
        time_vec_q = format_t_vec(time_vec_q, FA)
        r_q_sum = summarize_rate(r_q, n_rates)
    except:
        r_q_sum = None
        time_vec_q = None

    output_wd = os.path.dirname(f)
    name_file = os.path.basename(f)
    name_file = name_file.replace("_mcmc.log", "")
    r_file = "%s_RTT.r" % name_file
    pdf_file = "%s_RTT.pdf" % name_file

    return output_wd, r_file, pdf_file, r_sp_sum, r_ex_sum, r_div_sum, long_sum, time_vec, r_q_sum, time_vec_q



def plot_bdnn_rtt(output_wd, r_file, pdf_file, r_sp_sum, r_ex_sum, r_div_sum, long_sum, time_vec, r_q_sum, time_vec_q):
    out = "%s/%s" % (output_wd, r_file)
    newfile = open(out, "w")
    n_rows = 0
    n_elements = 0
    if not r_sp_sum is None:
        n_rows += 2
        n_elements += 4
    if not r_q_sum is None:
        n_rows += 1
        n_elements += 1
        if not r_sp_sum is None:
            n_elements += 1
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        wd_forward = os.path.abspath(output_wd).replace('\\', '/')
        r_script = "pdf(file='%s/%s', width = 9, height = %s, useDingbats = FALSE)\n" % (wd_forward, pdf_file, 3 * n_rows)
    else:
        r_script = "pdf(file='%s/%s', width = 9, height = %s, useDingbats = FALSE)\n" % (output_wd, pdf_file, 3 * n_rows)
    r_script += "\nlayout(matrix(1:%s, ncol = 2, nrow = %s, byrow = TRUE))" % (n_elements, n_rows)
    r_script += "\npar(las = 1, mar = c(4.5, 4.5, 0.5, 0.5))"
    if not r_sp_sum is None:
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
    if not r_q_sum is None:
        r_script += util.print_R_vec('\ntime_vec_q', time_vec_q)
        r_script += util.print_R_vec('\nq_mean', r_q_sum[:, 0])
        r_script += util.print_R_vec('\nq_lwr', r_q_sum[:, 1])
        r_script += util.print_R_vec('\nq_upr', r_q_sum[:, 2])
        r_script += "\nxlim = c(%s, %s)" % (np.max(time_vec_q), np.min(time_vec_q))
        r_script += "\nylim = c(%s, %s)" % (np.nanmin(r_q_sum), np.nanmax(r_q_sum))
        r_script += "\nnot_NA = !is.na(q_mean)"
        r_script += "\nplot(time_vec_q[not_NA], q_mean[not_NA], type = 'n', xlim = xlim, ylim = ylim, xlab = 'Time (Ma)', ylab = 'Sampling rate')"
        r_script += "\npolygon(c(time_vec_q[not_NA], rev(time_vec_q[not_NA])), c(q_lwr[not_NA], rev(q_upr[not_NA])), col = adjustcolor('#EEC591', alpha = 0.5), border = NA)"
        r_script += "\nlines(time_vec_q[not_NA], q_mean[not_NA], col = '#EEC591', lwd = 2)"
    r_script += "\ndev.off()"
    newfile.writelines(r_script)
    newfile.close()

    if platform.system() == "Windows" or platform.system() == "Microsoft":
        cmd = "cd %s & Rscript %s" % (output_wd, r_file)
    else:
        cmd = "cd %s; Rscript %s" % (output_wd, r_file)
    print("cmd", cmd)
    os.system(cmd)


def plot_bdnn_rtt_groups(path_dir_log_files, groups_path, burn):
    """Make RTT plots for the groups of species defined in the tab-delimited group_path file"""
    mcmc_file = path_dir_log_files
    path_dir_log_files = path_dir_log_files.replace("_mcmc.log", "")
    pkl_file = path_dir_log_files + ".pkl" 
    
    output_wd = os.path.dirname(mcmc_file)
    name_file = os.path.basename(path_dir_log_files)
    name_file = name_file.replace("_mcmc.log", "")

    bdnn_obj, w_sp, w_ex, _, sp_fad_lad, ts, te, t_reg_lam, t_reg_mu, _, reg_denom_lam, reg_denom_mu, _, _ = bdnn_parse_results(mcmc_file, pkl_file, burn)
    species_names = bdnn_obj.sp_fad_lad["Taxon"].to_numpy()
    group_file = pd.read_csv(groups_path, delimiter = '\t')
    group_names = group_file.columns.tolist()
    group_species_idx = []
    for gn in group_names:
        species_in_group = group_file[gn].dropna().to_numpy()
        group_species_idx.append(np.where(np.in1d(species_names, species_in_group))[0])
    
    try:
        # Diversification
        times_of_shift = get_bdnn_time(bdnn_obj, ts)
        num_bins = len(times_of_shift) - 1
        hidden_act_f = bdnn_obj.bdnn_settings['hidden_act_f']
        out_act_f = bdnn_obj.bdnn_settings['out_act_f']
        trait_tbl = bdnn_obj.trait_tbls
        names_features = get_names_features(bdnn_obj, rate_type='speciation')
        bdnn_dd = 'diversity' in names_features
        bdnn_rescale_div = bdnn_obj.bdnn_settings['div_rescaler']
        div_idx_trt_tbl = -1
        if is_time_trait(bdnn_obj):
            div_idx_trt_tbl = -2
        n_taxa = trait_tbl[0].shape[1]
        num_it = ts.shape[0]
        lam_it = np.zeros((num_it, n_taxa, num_bins))
        mu_it = lam_it + 0.0
        # Get speciation and extinction rates (the same as we obtained them during the BDNN inference)
        for i in range(num_it):
            if bdnn_dd:
                bdnn_time_div = np.arange(np.max(ts[i, :]), 0.0, -0.001)
                bdnn_div = get_DT(bdnn_time_div, ts[i, :], te[i, :])
                bdnn_div = get_DT(bdnn_time_div, post_ts_i, post_te_i)
                bdnn_binned_div = get_binned_div_traj(bdnn_time, bdnn_time_div, bdnn_div)[:-1] / bdnn_rescale_div
                bdnn_binned_div = np.repeat(bdnn_binned_div, n_taxa).reshape((len(bdnn_binned_div), n_taxa))
                trait_tbl[0][:, :, div_idx_trt_tbl] = bdnn_binned_div
                trait_tbl[1][:, :, div_idx_trt_tbl] = bdnn_binned_div
            lam = get_rate_BDNN_3D_noreg(trait_tbl[0], w_sp[i], hidden_act_f, out_act_f)
            lam_it[i, :, :] = lam ** t_reg_lam[i] / reg_denom_lam[i]
            mu = get_rate_BDNN_3D_noreg(trait_tbl[1], w_ex[i], hidden_act_f, out_act_f)
            mu_it[i, :, :] = mu ** t_reg_mu[i] / reg_denom_mu[i]

        # Get marginal rates through time for the specified group of taxa
        FA = np.max(np.mean(ts, axis=0))
        time_vec = format_t_vec(times_of_shift[1:-1], FA)
        for g in range(len(group_names)):
            gs = group_species_idx[g]
            r_sp = np.zeros((num_it, num_bins))
            r_ex = np.zeros((num_it, num_bins))
            for i in range(num_it):
                for j in range(num_bins):
                    lam_tmp = lam_it[i, gs, j]
                    mu_tmp = mu_it[i, gs, j]
                    indx = get_sp_indx_in_timeframe(ts[i, gs], te[i, gs], up = times_of_shift[j], lo = times_of_shift[j + 1])
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category = RuntimeWarning)
                        r_sp[i, j] = 1 / np.mean(1 / lam_tmp[indx])
                        r_ex[i, j] = 1 / np.mean(1 / mu_tmp[indx])
            
            r_div = r_sp - r_ex
            longevity = 1. / r_ex
            sptt = summarize_rate(r_sp, num_bins)
            extt = summarize_rate(r_ex, num_bins)
            divtt = summarize_rate(r_div, num_bins)
            longtt = summarize_rate(longevity, num_bins)
            qtt = None
            time_vec_q = None
            r_file = "%s_%s_RTT.r" % (name_file, group_names[g])
            pdf_file = "%s_%s_RTT.pdf" % (name_file, group_names[g])
            plot_bdnn_rtt(output_wd, r_file, pdf_file, sptt, extt, divtt, longtt, time_vec, qtt, time_vec_q)
    except:
        pass
    # Add sampling through time


def apply_thin(w, thin):
    n_samples = w.shape[0]
    if thin > n_samples or thin == 0:
        thin = n_samples
        print("resample set to the number of mcmc samples:", n_samples)
    thin_idx = np.linspace(1, n_samples, num = int(thin), dtype = int) - 1
    return w[thin_idx,:]


def bdnn_read_mcmc_file(mcmc_file, burn, thin):
    m = pd.read_csv(mcmc_file, delimiter = '\t')
    w_sp_indx = [i for i in range(len(m.columns)) if 'w_lam_' in m.columns[i]]
    w_ex_indx = [i for i in range(len(m.columns)) if 'w_mu_' in m.columns[i]]
    w_q_indx = [i for i in range(len(m.columns)) if 'w_q_' in m.columns[i]]
    ts_indx = [i for i in range(len(m.columns)) if '_TS' in m.columns[i]]
    te_indx = [i for i in range(len(m.columns)) if '_TE' in m.columns[i]]
    np_m = m.to_numpy().astype(float) # Stupid pandas does not understand nan
    num_it = np_m.shape[0]
    burnin = check_burnin(burn, num_it)
    w_sp = np_m[burnin:, w_sp_indx]
    w_ex = np_m[burnin:, w_ex_indx]
    w_q = np_m[burnin:, w_q_indx]
    ts = np_m[burnin:, ts_indx]
    te = np_m[burnin:, te_indx]
    num_it = ts.shape[0]
    if 't_reg_lam' in m.columns:
        reg_lam = m['t_reg_lam'].to_numpy()[burnin:].reshape((num_it, 1))
        reg_mu = m['t_reg_mu'].to_numpy()[burnin:].reshape((num_it, 1))
        denom_lam = m['reg_denom_lam'].to_numpy()[burnin:].reshape((num_it, 1))
        denom_mu = m['reg_denom_mu'].to_numpy()[burnin:].reshape((num_it, 1))
    else:
        reg_lam = np.ones((num_it, 1))
        reg_mu = np.ones((num_it, 1))
        denom_lam = np.ones((num_it, 1))
        denom_mu = np.ones((num_it, 1))
    if 't_reg_q' in m.columns:
        reg_q = m['t_reg_q'].to_numpy()[burnin:].reshape((num_it, 1))
        denom_q = m['reg_denom_q'].to_numpy()[burnin:].reshape((num_it, 1))
        norm_fac_q = m['normalize_q'].to_numpy()[burnin:].reshape((num_it, 1))
    else:
        reg_q = np.ones((num_it, 1))
        denom_q = np.ones((num_it, 1))
        norm_fac_q = np.ones((num_it, 1))
    if thin > 0:
        ts = apply_thin(ts, thin)
        te = apply_thin(te, thin)
        if w_sp.shape[1] > 0:
            w_sp = apply_thin(w_sp, thin)
            w_ex = apply_thin(w_ex, thin)
            reg_lam = apply_thin(reg_lam, thin).flatten()
            reg_mu = apply_thin(reg_mu, thin).flatten()
            denom_lam = apply_thin(denom_lam, thin).flatten()
            denom_mu = apply_thin(denom_mu, thin).flatten()
        if w_q.shape[1] > 0:
            w_q = apply_thin(w_q, thin)
            reg_q = apply_thin(reg_q, thin).flatten()
            denom_q = apply_thin(denom_q, thin).flatten()
            norm_fac_q = apply_thin(norm_fac_q, thin).flatten()
    return w_sp, w_ex, w_q, ts, te, reg_lam, reg_mu, reg_q, denom_lam, denom_mu, denom_q, norm_fac_q


def bdnn_reshape_w(posterior_w, bdnn_obj, rate_type):
    w_list = []
    if rate_type == 'diversification':
        layer_sizes = bdnn_obj.bdnn_settings['layers_sizes']
        layer_shapes = bdnn_obj.bdnn_settings['layers_shapes']
    else:
        layer_sizes = bdnn_obj.bdnn_settings['layers_sizes_q']
        layer_shapes = bdnn_obj.bdnn_settings['layers_shapes_q']
    for w in posterior_w:
        c = 0
        w_sample = []
        for i in range(len(layer_sizes)):
            w_layer = w[c:c+layer_sizes[i]].reshape(layer_shapes[i])
            c += layer_sizes[i]
            w_sample.append(w_layer)
        w_list.append(w_sample)
    return w_list


def bdnn_parse_results(mcmc_file, pkl_file, burn = 0.1, thin = 0):
    ob = load_pkl(pkl_file)
    w_sp, w_ex, w_q, post_ts, post_te, post_t_reg_lam, post_t_reg_mu, post_t_reg_q, post_reg_denom_lam, post_reg_denom_mu, post_reg_denom_q, post_norm_q = bdnn_read_mcmc_file(mcmc_file, burn, thin)
    post_w_sp = None
    post_w_ex = None
    post_w_q = None
    if w_sp.shape[1] > 0:
        post_w_sp = bdnn_reshape_w(w_sp, ob, rate_type = 'diversification')
        post_w_ex = bdnn_reshape_w(w_ex, ob, rate_type = 'diversification')
    if w_q.shape[1] > 0:
        post_w_q = bdnn_reshape_w(w_q, ob, rate_type = 'sampling')
    sp_fad_lad = ob.sp_fad_lad
    return ob, post_w_sp, post_w_ex, post_w_q, sp_fad_lad, post_ts, post_te, post_t_reg_lam, post_t_reg_mu, post_t_reg_q, post_reg_denom_lam, post_reg_denom_mu, post_reg_denom_q, post_norm_q


def bdnn_time_rescaler(x, bdnn_obj):
    if 'use_time_as_trait' in bdnn_obj.bdnn_settings.keys():
        x = x * bdnn_obj.bdnn_settings['time_rescaler']
    return x


def get_names_features(bdnn_obj, rate_type):
    if rate_type == "sampling":
        na = copy_lib.deepcopy(bdnn_obj.bdnn_settings['names_features_q'])
    else:
        na = copy_lib.deepcopy(bdnn_obj.bdnn_settings['names_features'])
    return na


def is_time_trait(bdnn_obj):
    t = False
    if 'use_time_as_trait' in bdnn_obj.bdnn_settings.keys():
        t = bdnn_obj.bdnn_settings['use_time_as_trait'] != 0.0
    return t


def backscale_bdnn_diversity(tbl, bdnn_obj, names_feat):
    if 'diversity' in names_feat:
        div_idx = np.where(np.array(["diversity"]) == names_feat)[0]
        tbl[:, div_idx] = tbl[:, div_idx] * bdnn_obj.bdnn_settings['div_rescaler']
    return tbl


def backscale_bdnn_time(x, bdnn_obj):
    denom = 1
    if is_time_trait(bdnn_obj):
        denom = bdnn_obj.bdnn_settings['time_rescaler']
    return x / denom


def backscale_time_cond_trait_tbl(cond_trait_tbl, bdnn_obj):
    if is_time_trait(bdnn_obj):
        cond_trait_tbl[:, -7] = backscale_bdnn_time(cond_trait_tbl[:, -7], bdnn_obj)
    return cond_trait_tbl


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
    return tbl


def backscale_bdnn_features(file_transf_features, bdnn_obj, cond_trait_tbl_sp, cond_trait_tbl_ex, cond_trait_tbl_q):
    backscale_par = None
    if not cond_trait_tbl_sp is None:
        cond_trait_tbl_sp = backscale_time_cond_trait_tbl(cond_trait_tbl_sp, bdnn_obj)
        cond_trait_tbl_ex = backscale_time_cond_trait_tbl(cond_trait_tbl_ex, bdnn_obj)
        names_feat = get_names_features(bdnn_obj, rate_type="speciation")
        cond_trait_tbl_sp = backscale_bdnn_diversity(cond_trait_tbl_sp, bdnn_obj, names_feat)
        cond_trait_tbl_ex = backscale_bdnn_diversity(cond_trait_tbl_ex, bdnn_obj, names_feat)
        if file_transf_features != "":
            backscale_par = read_backscale_file(file_transf_features)
            cond_trait_tbl_sp = backscale_tbl(bdnn_obj, backscale_par, names_feat, cond_trait_tbl_sp)
            cond_trait_tbl_ex = backscale_tbl(bdnn_obj, backscale_par, names_feat, cond_trait_tbl_ex)
    if not cond_trait_tbl_q is None:
        names_feat = get_names_features(bdnn_obj, rate_type="sampling")
        if file_transf_features != "":
            backscale_par = read_backscale_file(file_transf_features)
            cond_trait_tbl_q = backscale_tbl(bdnn_obj, backscale_par, names_feat, cond_trait_tbl_q)
    return cond_trait_tbl_sp, cond_trait_tbl_ex, cond_trait_tbl_q, backscale_par


def get_trt_tbl(bdnn_obj, rate_type):
    match rate_type:
        case "speciation":
            trait_tbl = bdnn_obj.trait_tbls[0]
        case 'extinction':
            trait_tbl = bdnn_obj.trait_tbls[1]
        case 'sampling':
            trait_tbl = bdnn_obj.trait_tbls[2]
    return trait_tbl


def get_bdnn_time(bdnn_obj, ts):
    max_age = np.max(ts) + 0.001
    if isinstance(bdnn_obj, np.ndarray):
        shift_times = bdnn_obj
    else:
        shift_times = bdnn_obj.bdnn_settings['fixed_times_of_shift_bdnn']
    if len(shift_times) > 0:
        shift_times = shift_times[shift_times <= max_age]
    bdnn_time = np.concatenate((np.array([max_age]), shift_times, np.zeros(1)))
    return bdnn_time


def is_binary_feature(trait_tbl):
    n_features = trait_tbl.shape[-1]
    b = np.zeros(n_features, dtype = int)
    freq_state = np.zeros(n_features, dtype = int)
    for i in range(n_features):
        if len(trait_tbl.shape) == 3:
            values, counts = np.unique(trait_tbl[:, :, i], return_counts=True)
        else:
            values, counts = np.unique(trait_tbl[:, i], return_counts=True)
        values_range = np.arange(np.min(values), np.max(values) + 1)
        b[i] = np.all(np.isin(values, values_range))
        freq_state[i] = values[np.argmax(counts)]
    b = b.astype(bool)
    return b, freq_state


def is_time_variable_feature(trait_tbl):
    n_features = trait_tbl.shape[-1]
    n_time_bins = trait_tbl.shape[0]
    time_bins = np.arange(0, n_time_bins)
    var_feat = np.zeros((4, n_features), dtype = int)
    if trait_tbl.ndim == 3:
        for i in range(n_features):
            tr = trait_tbl[time_bins, :, i]
            std_tr = np.std(tr, axis = 0)
            varies_through_time = np.any(std_tr > 1e-10)
            var_feat[0, i] = varies_through_time
            if varies_through_time:
                v = int(np.std(std_tr) > 1e-10)
                var_feat[1 + v, i] = 1 # varies only through time (v = 0, 2nd row) or through species and time (v = 1, 3rd row)
            else:
                std_species = np.std(tr, axis = 1)
                var_feat[3, i] = np.any(std_species > 1e-10) # varies among species
    var_feat = var_feat.astype(bool)
    return var_feat


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


def get_bin_from_fossil_age(bdnn_obj, fad_lad, rate_type, reverse_time=False):
    bin_of_species = np.zeros(fad_lad.shape[0])
    z = 0
    # Also extinction should be from first occurrence on b/c from there on a species can go extinct
    if rate_type == 'extinction':
        z = 1
    if 'fixed_times_of_shift_bdnn' in bdnn_obj.bdnn_settings.keys(): 
        bdnn_time = bdnn_obj.bdnn_settings['fixed_times_of_shift_bdnn']
        bins = np.concatenate((bdnn_time, np.zeros(1)))
        if reverse_time:
            # In case of combined replicates where the times can differ among replicates we need to order from present to past.
            bins = bins[::-1]
            bin_of_species = np.digitize(fad_lad[:,z], bins) - 1
        else:
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
    elif np.isin(k, conc_group):
        tk = 15 # continuous feature part of a feature group
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
        if ti == 15 or tj == 15:
            pt = 16 # any interaction between a feature group containing several continuous features and any other feature (i.e. evetnually bypassing this type of interaction)
    return pt


def get_plot_type(m, b, group_features, do_inter_imp = True):
    """
    :return:
        2D numpy.ndarray with 3 columns: 1st feature, 2nd feature (can be None), and what type of plot this will be (violine, line, surface, interaction etc.)
    """
    n = m.shape[1]
    main = np.arange(n, dtype = float)
    inter = np.array([])
    if do_inter_imp:
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
                                # When features of group are not in sequential columns of the trait table,
                                # then there are two possibilities where we have to assign their interaction plot.
                                possibility1 = np.logical_and(np.isin(p[:, 0], g1_idx), p[:, 1] == p[k, 1])
                                possibility2 = np.logical_and(np.isin(p[:, 1], g1_idx), p[:, 0] == p[k, 1])
                                pidx[np.logical_or(possibility1, possibility2)] = counter
                                counter += 1
                            elif g1 == -1 and g2 > -1:
                                # i not part of a feature-group but j
                                g2_idx = group_features[g2]
                                possibility1 = np.logical_and(p[:, 0] == p[k, 0], np.isin(p[:, 1], g2_idx))
                                possibility2 = np.logical_and(p[:, 1] == p[k, 0], np.isin(p[:, 0], g2_idx))
                                pidx[np.logical_or(possibility1, possibility2)] = counter
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


def trait_combination_exists(w, trait_tbl, i, j, feature_is_time_variable, bdnn_obj, tste, rate_type, pt, names_features):
    lw = len(w)
    comb_exists = np.zeros(lw)
    use_time_as_trait = False
    if rate_type != 'sampling':
        use_time_as_trait = is_time_trait(bdnn_obj) # , rate_type
    time_dd_temp = False # No time as trait, diversity-dependence, or time-variable environment
    if trait_tbl.ndim == 3:
        time_dd_temp = True
    if np.isin(pt, np.array([1., 2., 3., 4.])):
        # Main effects
        comb_exists = np.ones(lw)
        if feature_is_time_variable[i]:
            if use_time_as_trait and i == (trait_tbl.shape[2] - 1) and rate_type != 'sampling':
                fa = get_fossil_age(bdnn_obj, tste, rate_type)
                t = fa
            elif rate_type != 'sampling':
                bin_species = get_bin_from_fossil_age(bdnn_obj, tste, rate_type, reverse_time=True)
                trait_at_ts_or_te = np.zeros((len(bin_species), trait_tbl.shape[2]))
                for k in range(len(bin_species)):
                    trait_at_ts_or_te[k, :] = trait_tbl[bin_species[k], k, :]
                t = trait_at_ts_or_te[:, i]
            else:
                bin_species = bdnn_obj.bdnn_settings['occs_sp']
                bin_species = bin_species[:, ::-1]
                trait_at_occurrence = np.zeros((len(bin_species), trait_tbl.shape[2]))
                for k in range(len(bin_species)):
                    trait_at_occurrence[k, :] = np.mean(trait_tbl[bin_species[k, :] > 0, k, :], axis=0)
                t = trait_at_occurrence[:, i]
            if not names_features[i] == 'taxon_age':
                w = w.flatten()
                delta_w = np.abs(np.diff(w[:2]))
                max_t = np.max(t) + delta_w
                min_t = np.min(t) - delta_w
                outside_obs = np.logical_or(w > max_t, w < min_t)
                comb_exists[outside_obs] = 0.0
    elif pt < 15:
        # Interactions
        i_bin, j_bin = is_binary_feature(trait_tbl)[0][[i, j]]
        i_time_var = feature_is_time_variable[i]
        j_time_var = feature_is_time_variable[j]
        if i_time_var or j_time_var:
            if rate_type != 'sampling':
                fa = get_fossil_age(bdnn_obj, tste, rate_type)
                bin_species = get_bin_from_fossil_age(bdnn_obj, tste, rate_type, reverse_time=True)
                trait_at_ts_or_te = np.zeros((len(bin_species), trait_tbl.shape[2]))
                for k in range(len(bin_species)):
                    trait_at_ts_or_te[k, :] = trait_tbl[bin_species[k], k, :]
            else:
                bin_species = bdnn_obj.bdnn_settings['occs_sp']
                bin_species = bin_species[:, ::-1]
                trait_at_ts_or_te = np.zeros((len(bin_species), trait_tbl.shape[2]))
                for k in range(len(bin_species)):
                    trait_at_ts_or_te[k, :] = np.mean(trait_tbl[bin_species[k, :] > 0, k, :], axis=0)
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
                t[:, i] = trait_at_ts_or_te[:, i]
            if j_time_var:
                t[:, j] = trait_at_ts_or_te[:, j]
            t = t[:, [i, j]]
            observed_comb = np.unique(t, axis = 0)
            for k in range(w.shape[0]):
                 if np.any(np.all(w[k, :] == observed_comb, axis = 1)):
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
                t[:, i] = trait_at_ts_or_te[:, i]
            if j_time_var:
                if (j == (t.shape[1] - 1)) and use_time_as_trait:
                    t[:, j] = fa
                else:
                    t[:, j] = trait_at_ts_or_te[:, j]
            states = np.unique(t[:, bin_idx]).astype(int)
            wc = w[:, cont_idx_w]
            if not 'taxon_age' in names_features[[i, j]]:
                for s in states:
                    # State-dependent continuous trait
                    ts = t[t[:, bin_idx] == s, cont_idx]
                    in_range_cont = np.logical_and(wc >= np.min(ts), wc <= np.max(ts))
                    is_state = w[:, bin_idx_w] == s
                    exists_idx = np.logical_and(in_range_cont, is_state)
                    comb_exists[exists_idx] = 1.0
            else:
                comb_exists = np.ones(lw)
        else:
            # 7: continuous x continuous
            if time_dd_temp:
                t = trait_tbl[0, :, :] + 0.0
            else:
                t = trait_tbl + 0.0
            if i_time_var or j_time_var:
                t = trait_at_ts_or_te + 0.0
                if ((j == (t.shape[1] - 1)) and use_time_as_trait):
                    # Time as trait but no other time variable feature
                    # Are we ever here with sampling?
                    t[:, j] = fa + 0.0
            if not 'taxon_age' in names_features[[i, j]]:
                try:
                    hull = ConvexHull(t[:, [i, j]])
                except:
                    hull = ConvexHull(t[:, [i, j]], qhull_options = 'QJ')
                vertices = t[hull.vertices, :][:, [i, j]]
                path_p = Path(vertices)
                inside_poly = path_p.contains_points(w, radius=0.1)
                comb_exists[inside_poly] = 1.0
#                tmin = np.min(t[:, [i, j]], axis = 0)
#                tmax = np.max(t[:, [i, j]], axis = 0)
            else:
                comb_exists = np.ones(lw)
    else:
        comb_exists = 1.0 # Multiple continuous features combined in a feature group. Do not plot this but keep for predictor importance.
    return comb_exists


def build_conditional_trait_tbl(bdnn_obj,
                                tste,
                                ts_post, te_post,
                                len_cont = 100,
                                rate_type = "speciation",
                                combine_discr_features = "",
                                do_inter_imp = True):
    trait_tbl = get_trt_tbl(bdnn_obj, rate_type)
    names_features = get_names_features(bdnn_obj, rate_type=rate_type)
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
    if rate_type == "sampling" and "taxon_age" in names_features:
        s0, s1, _ = trait_tbl.shape
        # Use random values to avoid being detected as categorical feature
        trait_tbl[ :, :, -1] = np.random.uniform(0.0, 1.0, size=s0 * s1).reshape((s0, s1))
        trait_tbl[ :, 0, -1] = np.linspace(1.0, 0.0, s0)
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
    feature_variation = is_time_variable_feature(trait_tbl)
    feature_is_time_variable = feature_variation[0,:]
    
    if trait_tbl.ndim == 3:
        # In case of combined replicates where the times can differ among replicates we need to order from present to past.
        trait_tbl = trait_tbl[::-1, :, :]
    
    if np.any(feature_is_time_variable) and rate_type != 'sampling':
        fossil_age = get_fossil_age(bdnn_obj, tste, 'speciation')
        fossil_age = backscale_bdnn_time(fossil_age, bdnn_obj)
        fossil_bin_ts = get_bin_from_fossil_age(bdnn_obj, tste, 'speciation', reverse_time=True)
        fossil_age = get_fossil_age(bdnn_obj, tste, 'extinction')
        fossil_age = backscale_bdnn_time(fossil_age, bdnn_obj)
        fossil_bin_te = get_bin_from_fossil_age(bdnn_obj, tste, 'extinction', reverse_time=True)
        n_taxa = len(fossil_bin_te)
        trait_at_ts = np.zeros((n_taxa, trait_tbl.shape[2]))
        trait_at_te = np.zeros((n_taxa, trait_tbl.shape[2]))
        for k in range(n_taxa):
            trait_at_ts[k, :] = trait_tbl[fossil_bin_ts[k], k, :]
            trait_at_te[k, :] = trait_tbl[fossil_bin_te[k], k, :]
        trait_at_ts_or_te = np.vstack((trait_at_ts, trait_at_te))
        minmaxmean_features[0, feature_is_time_variable] = np.min(trait_at_ts_or_te[:, feature_is_time_variable], axis = 0)
        minmaxmean_features[1, feature_is_time_variable] = np.max(trait_at_ts_or_te[:, feature_is_time_variable], axis = 0)
        if is_time_trait(bdnn_obj):
            tste_rescaled = bdnn_time_rescaler(tste, bdnn_obj)
            min_tste = np.min(tste_rescaled)
            if min_tste < minmaxmean_features[0, -1]:
                minmaxmean_features[1, -1] = min_tste * 0.98
            max_tste = np.max(tste_rescaled)
            if max_tste > minmaxmean_features[1, -1]:
                minmaxmean_features[1, -1] = max_tste * 1.02
    plot_type = get_plot_type(minmaxmean_features, binary_feature, idx_comb_feat, do_inter_imp = do_inter_imp)
    plot_idx = get_plot_idx(plot_type, idx_comb_feat)
    plot_type = np.hstack((plot_type, plot_idx.reshape((len(plot_idx), 1))))
    plot_type = plot_type[~np.isnan(plot_type[:, 3]), :]
    plot_idx_freq = get_plot_idx_freq(plot_type[:, 3])
    feature_without_variance = get_idx_feature_without_variance(trait_tbl)
    nr = get_nrows_conditional_trait_tbl(plot_type, minmaxmean_features)
    cond_trait_tbl = np.zeros((nr, n_features + 6))
    cond_trait_tbl[:, :n_features] = minmaxmean_features[2, :] # mean/modal values
    cond_trait_tbl[:, -1] = 1 # combination is observed
    counter = 0
    for k in range(plot_type.shape[0]):
        i = int(plot_type[k, 0])
        v2_constant = False
        v1 = np.linspace(minmaxmean_features[0, i], minmaxmean_features[1, i], int(minmaxmean_features[3, i]))
        v1_constant = np.ptp(v1) == 0.0 and minmaxmean_features[3, i] > 1.0
        v1 = v1.reshape((len(v1), 1))
        j = plot_type[k, 1]
        feat_idx = [i]
        if ~np.isnan(j):
            j = int(j)
            feat_idx.append(j)
            v2 = np.linspace(minmaxmean_features[0, j], minmaxmean_features[1, j], int(minmaxmean_features[3, j]))
            v2_constant = np.ptp(v2) == 0.0 and minmaxmean_features[3, j] > 1.0
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
        comparison_incl_feature_without_variance = np.any(np.isin([i, j], feature_without_variance)) or v1_constant or v2_constant
        cond_trait_tbl[counter:(counter + lv), -1] = trait_combination_exists(cond_trait_tbl[counter:(counter + lv), feat_idx],
                                                                              trait_tbl,
                                                                              i, j,
                                                                              feature_is_time_variable,
                                                                              bdnn_obj,
                                                                              tste,
                                                                              rate_type,
                                                                              plot_type[k, 2],
                                                                              names_features)
        if comparison_incl_feature_without_variance:
            cond_trait_tbl[counter:(counter + lv), -1] = np.nan
        counter = counter + lv
    # Remove comparisons when there is no variance of the features
    cond_trait_tbl = cond_trait_tbl[~np.isnan(cond_trait_tbl[:, -1]),:]
    return cond_trait_tbl, names_features


def get_conditional_rates(bdnn_obj, cond_trait_tbl, post_w, post_t_reg, post_denom):
    num_it = len(post_w)
    obs = cond_trait_tbl[:, -1] == 1
    rate_cond = np.zeros((np.sum(obs), num_it))
    for i in range(num_it):
        rate_cond[:, i] = get_rate_BDNN_noreg(cond_trait_tbl[obs, :-6],
                                              post_w[i], # list of arrays
                                              bdnn_obj.bdnn_settings['hidden_act_f'],
                                              bdnn_obj.bdnn_settings['out_act_f'])
        rate_cond[:, i] = rate_cond[:, i] ** post_t_reg[i] / post_denom[i]
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
        rate_isnan = np.isnan(cond_rates[i, :])
        all_nan = np.all(rate_isnan)
        n_rates = len(cond_rates[i, ~rate_isnan])
        if not all_nan and n_rates > 1:
            rate_sum[i, 1:] = util.calcHPD(cond_rates[i, :], .95)
    needs_median = rate_sum[:, 0] > rate_sum[:, 2]
    rate_sum[needs_median, 0] = np.median(cond_rates[needs_median, :], axis = 1)
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
        trait_tbl = trait_tbl[::-1, :, :]
        obs_cont = trait_tbl[0, :, feature_idx]
        obs_cont = obs_cont.transpose()
    else:
        obs_cont = trait_tbl[:, feature_idx]
    if np.any(feature_is_time_variable[feature_idx]) and rate_type != 'sampling':
        for z in range(len(fossil_bin)):
            obs_cont[z, :] = trait_tbl[fossil_bin[z], z, feature_idx]
    if np.any(feature_idx + 1 == trait_tbl.shape[-1]) and is_time_trait(bdnn_obj) and rate_type != 'sampling':
        obs_cont[:, obs_cont.shape[1] - 1] = fossil_age # Time is always the last
    if np.any(feature_is_time_variable[feature_idx]) and rate_type == 'sampling':
        bin_species = bdnn_obj.bdnn_settings['occs_sp']
        bin_species = bin_species[:, ::-1]
        obs_cont = np.zeros((len(bin_species), len(feature_idx)))
        for k in range(len(bin_species)):
            obs_cont[k, :] = np.mean(trait_tbl[bin_species[k, :] > 0, k, :][:, feature_idx], axis=0)
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
    match rate_type:
        case "speciation":
            r_script += "\ncol = colorRampPalette(c('lightblue1', rgb(0, 52, 94, maxColorValue = 255)))(%s)" % n_states
        case 'extinction':
            r_script += "\ncol = colorRampPalette(c(rgb(255, 143, 118, maxColorValue = 255), 'darkred'))(%s)" % n_states
        case 'net diversification':
            r_script += "\ncol = colorRampPalette(c('grey75', 'grey40'))(%s)" % n_states
        case 'sampling':
            r_script += "\ncol = colorRampPalette(c('burlywood1', 'burlywood4'))(%s)" % n_states
    for i in range(n_states):
        r_script += "\naxis(side = 1, at = %s, labels = '%s')" % (i, str(names_states[i]))
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
    match rate_type:
        case "speciation":
            col = '#6092AF'
        case 'extinction':
            col = '#C5483B'
        case 'net diversification':
            col = 'grey50'
        case 'sampling':
            col = 'burlywood2'
    not_na = np.isnan(rs[:, 0]) == False
    r_script += util.print_R_vec("\ntr", tr[not_na, 0])
    r_script += util.print_R_vec("\nr", rs[not_na, 0])
    r_script += util.print_R_vec("\nr_lwr", rs[not_na, 1])
    r_script += util.print_R_vec("\nr_upr", rs[not_na, 2])
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
    match rate_type:
        case "speciation":
            r_script += "\ncol = colorRampPalette(c('lightblue1', rgb(0, 52, 94, maxColorValue = 255)))(%s)" % n_states
        case 'extinction':
            r_script += "\ncol = colorRampPalette(c(rgb(255, 143, 118, maxColorValue = 255), 'darkred'))(%s)" % n_states
        case 'net diversification':
            r_script += "\ncol = colorRampPalette(c('grey75', 'grey40'))(%s)" % n_states
        case 'sampling':
            r_script += "\ncol = colorRampPalette(c('burlywood1', 'burlywood4'))(%s)" % n_states
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
    match rate_type:
        case "speciation":
            r_script += "\ncol = colorRampPalette(c('lightblue1', rgb(0, 52, 94, maxColorValue = 255)))(%s)" % nr
        case 'extinction':
            r_script += "\ncol = colorRampPalette(c(rgb(255, 143, 118, maxColorValue = 255), 'darkred'))(%s)" % nr
        case 'net diversification':
            r_script += "\ncol = colorRampPalette(c('grey75', 'grey40'))(%s)" % nr
        case 'sampling':
            r_script += "\ncol = colorRampPalette(c('burlywood1', 'burlywood4'))(%s)" % nr
    r_script += util.print_R_vec("\nx", tr[:, 0])
    r_script += util.print_R_vec("\ny", tr[:, 1])
    r_script += util.print_R_vec("\nr", rs[:, 0])
    r_script += "\nxyr <- cbind(x, y, r)"
    r_script += "\nxaxis <- sort(unique(xyr[, 1]))"
    r_script += "\nyaxis <- sort(unique(xyr[, 2]))"
    r_script += "\nz <- matrix(xyr[, 3], length(xaxis), length(yaxis), byrow)"
    r_script += "\npadx <- abs(diff(xaxis))[1]"
    r_script += "\npady <- abs(diff(yaxis))[1]"
#    if plot_time:
#        r_script += "\nxaxis <- xaxis - padx/4"
    r_script += "\nplot(mean(xaxis), mean(yaxis), type='n', xlim = c(min(xaxis) - padx, max(xaxis) + padx), ylim = c(min(yaxis) - pady, max(yaxis) + pady), xlab = '%s', ylab = '%s', xaxt = 'n')" % (names[0], names[1]) # , xaxs = 'i', yaxs = 'i'
    r_script += "\nxtk <- pretty(xaxis, n = 10)"
    r_script += "\nxtk_lbl <- xtk"
    if plot_time:
        r_script += "\nxtk_lbl <- abs(xtk_lbl)"
    r_script += "\naxis(side = 1, at = xtk, labels = xtk_lbl)"
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
    unique_states_feat_1 = np.unique(states_feat_1)
    n_states_feat_1 = len(unique_states_feat_1)
    if len(feat_idx_2) > 1:
        # One-hot encoded discrete feature
        states_feat_2 = convert_onehot_to_numeric(tr[:, feat_idx_2])
    else:
        # Binary or ordinal features
        states_feat_2 = tr[:, feat_idx_2].flatten()
    unique_states_feat_2 = np.unique(states_feat_2)
    n_states_feat_2 = len(unique_states_feat_2)
    rate_max = np.nanmax(rs[:, 2])
    rate_min = np.nanmin(rs[:, 1])
    rate_max += 0.2 * rate_max
    rate_min -= 0.2 * np.abs(rate_min)
    r_script += "\npar(las = 2, mar = c(9, 4, 1.5, 0.5))"
    r_script += "\nylim = c(%s, %s)" % (rate_min, rate_max)
    r_script += "\nxlim = c(-0.5, %s + 0.5)" % (n_states_feat_1 * n_states_feat_2 - 1)
    r_script += "\nplot(0, 0, type = 'n', xlim = xlim, ylim = ylim, xlab = '', ylab = '%s', xaxt = 'n')" % (rate_type)
    match rate_type:
        case "speciation":
            r_script += "\ncol = colorRampPalette(c('lightblue1', rgb(0, 52, 94, maxColorValue = 255)))(%s)" % n_states_feat_2
        case 'extinction':
            r_script += "\ncol = colorRampPalette(c(rgb(255, 143, 118, maxColorValue = 255), 'darkred'))(%s)" % n_states_feat_2
        case 'net diversification':
            r_script += "\ncol = colorRampPalette(c('grey75', 'grey40'))(%s)" % n_states_feat_2
        case 'sampling':
            r_script += "\ncol = colorRampPalette(c('burlywood1', 'burlywood4'))(%s)" % n_states_feat_2
    cex_labels = 1.8 / expand_grid(unique_states_feat_1, unique_states_feat_2).shape[0]**(1.0/2.5)
    counter = 0
    for i in range(n_states_feat_1):
        for j in range(n_states_feat_2):
            idx = np.logical_and(states_feat_1 == unique_states_feat_1[i], states_feat_2 == unique_states_feat_2[j])
            r_tmp = r[idx, :]
            r_tmp = r_tmp[r_tmp < rate_max]
            r_tmp = r_tmp[r_tmp > rate_min]
            if len(r_tmp) > 0:
                r_script += util.print_R_vec("\nvio_data", r_tmp)
                r_script += "\nvioplot(vio_data, at = %s, add = TRUE, wex = 0.5, rectCol = NA, lineCol = NA, colMed = NA, col = col[%s])" % (counter, j + 1)
                r_script += "\nlines(rep(%s, 2), c(%s, %s), lwd = 1.5)" % (counter, float(rs[idx, 1]), float(rs[idx, 2]))
                r_script += "\npoints(%s, %s, pch = 19)" % (counter, float(rs[idx, 0]))
            r_script += "\naxis(side = 1, at = %s, labels = paste0('%s', ': ', '%s', '\n', '%s', ': ', '%s'), cex.axis = %s)" % (counter, names[0], names_states_feat_1[i], names[1], names_states_feat_2[j], cex_labels)
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


def create_R_files_effects(cond_trait_tbl, cond_rates, bdnn_obj, tste, r_script, names_features, backscale_par,
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
    feature_is_time_variable = is_time_variable_feature(trait_tbl)[0, :]
    fossil_age = get_fossil_age(bdnn_obj, tste, rate_type2)
    fossil_age = backscale_bdnn_time(fossil_age, bdnn_obj)
    fossil_bin = get_bin_from_fossil_age(bdnn_obj, tste, rate_type2, reverse_time=True)
    names_features_original = np.array(get_names_features(bdnn_obj, rate_type=rate_type))
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
            obs = backscale_bdnn_diversity(obs, bdnn_obj, [names])
            r_script = plot_bdnn_cont(rates_sum_plt, trait_tbl_plt, r_script, names, plot_time, obs, rate_type)
        elif np.isin(pt, np.array([6.0, 13.0, 14.0])):
            b = binary_feature[incl_features]
            names_states = np.unique(trait_tbl_plt[:, b]).tolist()
            if pt == 13.0:
                names_states = names_features_original[incl_features][b]
            names = names[np.argsort(b)]
            trait_tbl_plt = trait_tbl_plt[:, np.argsort(b)] # Continuous feature always in column 0
            obs = obs[:, np.argsort(b)]
            obs[:, 0] = backscale_tbl(bdnn_obj, backscale_par, [names[0]], obs[:, 0].reshape((obs.shape[0], 1))).flatten()
            obs[:, 1] = backscale_tbl(bdnn_obj, backscale_par, [names[1]], obs[:, 1].reshape((obs.shape[0], 1))).flatten()
            obs = backscale_bdnn_diversity(obs, bdnn_obj, names)
            r_script = plot_bdnn_inter_discr_cont(rates_sum_plt, trait_tbl_plt, r_script, names, names_states, plot_time, obs, rate_type)
        elif pt == 7.0:
            obs = backscale_tbl(bdnn_obj, backscale_par, names.tolist(), obs)
            obs = backscale_bdnn_diversity(obs, bdnn_obj, names)
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
#                print('plot index', i)
#                print('plot type', pt)
#                print(cond_trait_tbl[idx, :])
#                print('names_features', names_features)
#                print('names', names)
#                print('incl_features', incl_features)
                feat_1, feat_2 = get_feat_idx(names_features, names, incl_features)
#                print('feat_1', feat_1)
#                print('feat_2', feat_2)
#                print('trait_tbl_plt', trait_tbl_plt)
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


def get_mean_inferred_tste(post_ts, post_te):
    mean_tste = np.zeros((post_ts.shape[1], 2))
    mean_tste[:, 0] = np.mean(post_ts, axis = 0)
    mean_tste[:, 1] = np.mean(post_te, axis = 0)
    return mean_tste




def get_baseline_q2(mcmc_file, burn, thin, mean_across_shifts=True):
    m = pd.read_csv(mcmc_file, delimiter = '\t')
    num_it = m.shape[0]
    burn_idx = check_burnin(burn, num_it)
    root_age, _ = get_root_age(mcmc_file, burn_idx)
    f = mcmc_file.replace("_mcmc.log", "")
    f_q = f + "_q_rates.log"
    qtt_list = read_rtt(f_q, burn_idx)
    time_vec = make_t_vec(qtt_list)
    # harmonic mean through time
    q = get_mean_rates(qtt_list, root_age)
    q = q.reshape((len(q), 1))
    q = apply_thin(q, thin)
    return q


def downsample_q_times(q_time_frames, highres_q_repeats):
    u, c = np.unique(highres_q_repeats, return_counts=True)
    q_time_frame_idx = np.concatenate((np.zeros(1), np.cumsum(c)), axis=None).astype(int)
    q_times_downsampled = q_time_frames[q_time_frame_idx]
    return q_times_downsampled


def get_baseline_q(mcmc_file, burn, thin, mean_across_shifts=True):
    m = pd.read_csv(mcmc_file, delimiter = '\t')
    q_indx = np.array([i for i in range(len(m.columns)) if m.columns[i].startswith('q_')])
    alpha_idx = [i for i in range(len(m.columns)) if m.columns[i] == "alpha"]
    q_indx = q_indx[q_indx < alpha_idx]
    np_m = m.to_numpy().astype(float)
    num_it = np_m.shape[0]
    burnin = check_burnin(burn, num_it)
    q = np_m[burnin:, q_indx]
    if q.shape[1] == 1:
        q = q.reshape((len(q), 1))
    if q.shape[1] > 1 and mean_across_shifts:
        pkl_file = mcmc_file.replace("_mcmc.log", "") + ".pkl"
        bdnn_obj = load_pkl(pkl_file)
        weights = np.ones(q.shape[1]) / q.shape[1]
        if 'q_time_frames' in bdnn_obj.bdnn_settings.keys():
            q_time_frames = bdnn_obj.bdnn_settings['q_time_frames']
            if 'highres_q_repeats' in bdnn_obj.bdnn_settings.keys():
                q_time_frames = downsample_q_times(q_time_frames, bdnn_obj.bdnn_settings['highres_q_repeats'])
            weights = np.diff(q_time_frames)
            weights = weights / np.sum(weights)
#        mask = q==0.0 # np.isnan(q)
        mask = np.isnan(q)
        qma = np.ma.MaskedArray(q, mask=mask)
        q = np.ma.average(qma, weights=weights, axis=1, keepdims=True)
    q = apply_thin(q, thin)
    return q


def get_effect_objects(mcmc_file, pkl_file, burnin, thin, combine_discr_features = "", file_transf_features = "", num_processes = 1, show_progressbar = False, do_inter_imp = True):
    bdnn_obj, post_w_sp, post_w_ex, post_w_q, sp_fad_lad, post_ts, post_te, post_t_reg_lam, post_t_reg_mu, post_t_reg_q, post_reg_denom_lam, post_reg_denom_mu, post_reg_denom_q, post_norm_q = bdnn_parse_results(mcmc_file, pkl_file, burnin, thin)
    mean_tste = get_mean_inferred_tste(post_ts, post_te)
    cond_trait_tbl_sp = None
    names_features_sp = None
    sp_rate_cond = None
    cond_trait_tbl_ex = None
    names_features_ex = None
    ex_rate_cond = None
    cond_trait_tbl_q = None
    names_features_q = None
    q_rate_cond = None
    if not post_w_sp is None:
        cond_trait_tbl_sp, names_features_sp = build_conditional_trait_tbl(bdnn_obj, mean_tste,
                                                                           post_ts, post_te,
                                                                           len_cont = 100,
                                                                           rate_type = "speciation",
                                                                           combine_discr_features = combine_discr_features,
                                                                           do_inter_imp = do_inter_imp)
        cond_trait_tbl_ex, names_features_ex = build_conditional_trait_tbl(bdnn_obj, mean_tste,
                                                                           post_ts, post_te,
                                                                           len_cont = 100,
                                                                           rate_type = "extinction",
                                                                           combine_discr_features = combine_discr_features,
                                                                           do_inter_imp = do_inter_imp)
        print("\nGetting partial dependence rates for speciation")
        sp_rate_cond = get_partial_dependence_rates(bdnn_obj, cond_trait_tbl_sp,
                                                    post_w_sp, post_t_reg_lam, post_reg_denom_lam,
                                                    post_ts, post_te, combine_discr_features = combine_discr_features,
                                                    rate_type = 'speciation',
                                                    num_processes = num_processes, show_progressbar = show_progressbar)
        print("Getting partial dependence rates for extinction")
        ex_rate_cond = get_partial_dependence_rates(bdnn_obj, cond_trait_tbl_ex,
                                                    post_w_ex, post_t_reg_mu, post_reg_denom_mu,
                                                    post_ts, post_te, combine_discr_features = combine_discr_features,
                                                    rate_type = 'extinction',
                                                    num_processes = num_processes, show_progressbar = show_progressbar)
    if not post_w_q is None:
        cond_trait_tbl_q, names_features_q = build_conditional_trait_tbl(bdnn_obj, mean_tste,
                                                                         post_ts, post_te,
                                                                         len_cont = 100,
                                                                         rate_type = "sampling",
                                                                         combine_discr_features = combine_discr_features,
                                                                         do_inter_imp = do_inter_imp)
        baseline_q = get_baseline_q(mcmc_file, burnin, thin)
        print("Getting partial dependence rates for sampling")
#        bdnn_time = np.zeros(1) # Placeholder, not doing anything for sampling
        q_rate_cond = get_partial_dependence_rates(bdnn_obj, cond_trait_tbl_q,
                                                   post_w_q, post_t_reg_q, post_reg_denom_q,
                                                   post_ts, post_te, combine_discr_features = combine_discr_features,
                                                   rate_type = 'sampling',
                                                   num_processes=num_processes, show_progressbar=show_progressbar,
                                                   baseline=baseline_q, norm=post_norm_q)
    cond_trait_tbl_sp, cond_trait_tbl_ex, cond_trait_tbl_q, backscale_par = backscale_bdnn_features(file_transf_features,
                                                                                                    bdnn_obj,
                                                                                                    cond_trait_tbl_sp,
                                                                                                    cond_trait_tbl_ex,
                                                                                                    cond_trait_tbl_q)
    return bdnn_obj, cond_trait_tbl_sp, cond_trait_tbl_ex, cond_trait_tbl_q, names_features_sp, names_features_ex, names_features_q, sp_rate_cond, ex_rate_cond, q_rate_cond, mean_tste, backscale_par


def plot_effects(f,
                 cond_trait_tbl_sp,
                 cond_trait_tbl_ex,
                 cond_trait_tbl_q,
                 sp_rate_cond,
                 ex_rate_cond,
                 q_rate_cond,
                 bdnn_obj,
                 tste,
                 backscale_par,
                 names_features_sp,
                 names_features_ex,
                 names_features_q,
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
    if not cond_trait_tbl_sp is None:
        r_script = create_R_files_effects(cond_trait_tbl_sp, sp_rate_cond + 0.0, bdnn_obj, tste, r_script, names_features_sp,
                                          backscale_par, rate_type = 'speciation')
        r_script = create_R_files_effects(cond_trait_tbl_ex, ex_rate_cond + 0.0, bdnn_obj, tste, r_script, names_features_ex,
                                          backscale_par, rate_type = 'extinction')
        feat_to_keep = get_feat_to_keep_for_netdiv(cond_trait_tbl_sp, cond_trait_tbl_ex)
        netdiv_rate_cond, cond_trait_tbl_netdiv = get_rates_cond_trait_tbl_for_netdiv(feat_to_keep,
                                                                                      sp_rate_cond,
                                                                                      ex_rate_cond,
                                                                                      cond_trait_tbl_ex,
                                                                                      cond_trait_tbl_sp)
        r_script = create_R_files_effects(cond_trait_tbl_netdiv, netdiv_rate_cond, bdnn_obj, tste, r_script, names_features_ex,
                                          backscale_par, rate_type = 'net diversification')
    if not cond_trait_tbl_q is None:
        r_script = create_R_files_effects(cond_trait_tbl_q, q_rate_cond + 0.0, bdnn_obj, tste, r_script, names_features_q,
                                          backscale_par, rate_type = 'sampling')
    r_script += "\nn <- dev.off()"
    newfile.writelines(r_script)
    newfile.close()
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        cmd = "cd %s & Rscript %s_%s.r" % (output_wd, name_file, suffix_pdf)
    else:
        cmd = "cd %s; Rscript %s_%s.r" % (output_wd, name_file, suffix_pdf)
    # print("cmd", cmd)
    print("\nThe plot file %s_%s.r was saved in %s \n" % (name_file, suffix_pdf, output_wd))
    os.system(cmd)


# Coefficient of rate variation
###############################
def get_cv(x):
    x_mean = np.mean(x, axis=0)
    cv = np.std(x_mean) / np.mean(x_mean)
    return cv


def get_weighted_harmonic_mean(x, w):
    x_nan = np.isnan(x)
    w_no_nan = w[~x_nan]
    x_no_nan = x[~x_nan]
    hmr = np.sum(w_no_nan) / np.sum(w_no_nan / x_no_nan)
    hmr *= (np.sum(w_no_nan) / np.sum(w))
    return hmr


def get_mean_rates(rtt, root):
    n_iter = len(rtt)
    mean_rtt = np.zeros(n_iter)
    for i in range(n_iter):
        rtt_i = rtt[i]
        s1 = len(rtt_i)
        n_rates = int((s1 + 1) / 2)
        bins = rtt_i[n_rates:]
        bins = np.concatenate((np.array(root), bins, np.zeros(1)), axis=None)
        duration_bins = np.abs(np.diff(bins))
        rtt_i = rtt_i[:n_rates]
        mean_rtt[i] = get_weighted_harmonic_mean(rtt_i, duration_bins)
    return mean_rtt


def get_mean_rate_through_time(rtt, root):
    mean_rtt = get_mean_rates(rtt, root)
    summary_rtt = np.zeros(3)
    summary_rtt[0] = np.mean(mean_rtt)
    summary_rtt[1:] = util.calcHPD(mean_rtt, .95)
    return summary_rtt


def get_root_age(mcmc_file, burnin):
    m = pd.read_csv(mcmc_file, delimiter = '\t', na_values='nan', keep_default_na=False)
    root = m['root_age'].to_numpy().astype(float)[burnin:]
    root_CI = util.calcHPD(root, .95).tolist()
    return np.mean(root), root_CI


def get_num_traits_and_cat_levels(bdnn_obj, rate_type, combine_discr_features=""):
    # Number of continuous and categorical features
    trait_tbl = get_trt_tbl(bdnn_obj, rate_type)
    names_features = get_names_features(bdnn_obj, rate_type)
    idx_comb_feat = get_idx_comb_feat(names_features, combine_discr_features)
    conc_comb_feat = np.array([])
    if idx_comb_feat:
        conc_comb_feat = np.concatenate(idx_comb_feat)
    binary_feature, most_frequent_state = is_binary_feature(trait_tbl)
    minmaxmean_features = get_minmaxmean_features(trait_tbl,
                                                  most_frequent_state,
                                                  binary_feature,
                                                  conc_comb_feat,
                                                  100)
    levels_cat_trait = minmaxmean_features[3, binary_feature].flatten()
    num_traits = minmaxmean_features.shape[1]
    return num_traits, levels_cat_trait


def get_CV_from_sim_bdnn(bdnn_obj, num_taxa, sp_rates, ex_rates, lam_tt, mu_tt, root_age, combine_discr_features="", num_sim=5, num_processes=1, show_progressbar=False):
    cv_rates = np.zeros((2, 3))
    cv_rates[0, 1] = get_cv(sp_rates)
    cv_rates[1, 1] = get_cv(ex_rates)
    rangeSP = [num_taxa * 0.7, num_taxa * 1.3]
    rangeL = lam_tt[1:].tolist()
    rangeM = mu_tt[1:].tolist()

    maxL = np.max(rangeL)
    minM = np.min(rangeM)
#    if maxL < minM:
#        rangeL[1] = minM
#        rangeM[0] = maxL

    # Number of continuous and categorical features
    num_traits, levels_cat_trait = get_num_traits_and_cat_levels(bdnn_obj, 'extinction',
                                                                 combine_discr_features=combine_discr_features)

    # Network architecture and priors
    layer_shapes = bdnn_obj.bdnn_settings['layers_shapes']
    n_nodes = []
    for i in range(len(layer_shapes) - 1):
        n_nodes.append(layer_shapes[i][0])
    bdnn_update_f = np.arange(1, len(layer_shapes) + 1)
    bdnn_update_f = bdnn_update_f / (2.0 * np.max(bdnn_update_f))
    act_f = bdnn_obj.bdnn_settings['hidden_act_f']
    out_act_f = bdnn_obj.bdnn_settings['out_act_f']
    prior_t_reg = [-1.0, -1.0] # In case of very old log files witout regularization
    if 'prior_t_reg' in bdnn_obj.bdnn_settings:
        prior_t_reg = bdnn_obj.bdnn_settings['prior_t_reg']
        independ_reg = False
        # Keep compatibility with pre-independent regularization
        if 'independent_t_reg' in bdnn_obj.bdnn_settings:
            independ_reg = bdnn_obj.bdnn_settings['independent_t_reg']
        else:
            prior_t_reg = [prior_t_reg, prior_t_reg]
    prior_cov = 1.0
    if 'prior_cov' in bdnn_obj.bdnn_settings:
        prior_cov = bdnn_obj.bdnn_settings['prior_cov']

    args = []
    for i in range(num_sim):
        a = [i,
             rangeSP, rangeL, rangeM, root_age,
             num_traits, levels_cat_trait,
             n_nodes, bdnn_update_f,
             act_f, out_act_f,
             prior_t_reg, independ_reg, prior_cov]
        args.append(a)
    unixos = is_unix()
    if unixos and num_processes > 1:
        pool_perm = multiprocessing.Pool(num_processes)
        cv_sim = list(tqdm(pool_perm.imap_unordered(get_CV_from_sim_i, args),
                           total = num_sim, disable = show_progressbar == False))
        pool_perm.close()
    else:
        cv_sim = []
        for i in tqdm(range(num_sim), disable = show_progressbar == False):
            cv_sim.append(get_CV_from_sim_i(args[i]))
    cv_rate = np.array(cv_sim)
    cv_rates[:, 2] = np.quantile(cv_sim, q=0.95, axis=0)
    return cv_rates


def get_coefficient_rate_variation(path_dir_log_files, burn, combine_discr_features="", num_sim=1000, num_processes=1, show_progressbar=False):
    pkl_file = path_dir_log_files + ".pkl"
    mcmc_file = path_dir_log_files + "_mcmc.log"
    lam_tt_file = path_dir_log_files + "_sp_rates.log"
    mu_tt_file = path_dir_log_files + "_ex_rates.log"
    rates_mcmc_file = path_dir_log_files + "_per_species_rates.log"
    
    bdnn_obj = load_pkl(pkl_file)
    species_rates = np.loadtxt(rates_mcmc_file, skiprows = 1)
    s = species_rates.shape
    num_taxa = int((s[1] - 1) / 2)
    num_it = s[0]
    burnin = check_burnin(burn, num_it)
    species_rates = species_rates[burnin:, :]
    sp_rates = species_rates[:, 1:(num_taxa + 1)]
    ex_rates = species_rates[:, (num_taxa + 1):]
    
    root_age, root_age_CI = get_root_age(mcmc_file, burnin)
    lam_tt = read_rtt(lam_tt_file, burnin)
    mu_tt = read_rtt(mu_tt_file, burnin)
    lam_tt_CI = get_mean_rate_through_time(lam_tt, root_age)
    mu_tt_CI = get_mean_rate_through_time(mu_tt, root_age)

    cv_rates = get_CV_from_sim_bdnn(bdnn_obj, num_taxa, sp_rates, ex_rates, lam_tt_CI, mu_tt_CI, root_age_CI,
                                    combine_discr_features=combine_discr_features,
                                    num_sim=num_sim,
                                    num_processes=num_processes,
                                    show_progressbar=show_progressbar)
    
    print('Coefficient of rate variation')
    print('    Speciation:', f'{float(cv_rates[0, 1]):.2f}', 'Expected:' + f'{float(cv_rates[0, 2]):.2f}')
    print('    Extinction:', f'{float(cv_rates[1, 1]):.2f}', 'Expected:' + f'{float(cv_rates[1, 2]):.2f}')
    cv_rates = pd.DataFrame(cv_rates, columns = ['rate', 'cv_empirical', 'cv_expected'])
    cv_rates['rate'] = ['speciation', 'extinction']
    output_wd = os.path.dirname(path_dir_log_files)
    name_file = os.path.basename(path_dir_log_files)
    cv_rates_file = os.path.join(output_wd, name_file + '_coefficient_of_rate_variation.csv')
    cv_rates.to_csv(cv_rates_file, na_rep = 'NA', index = False)


def get_rnd_gen(seed=None):
    return np.random.default_rng(seed)


class BdSimulator():
    def __init__(self,
                 s_species=1,  # number of starting species (can be a range)
                 rangeSP=[100, 1000],  # min/max size data set
                 minEX_SP=0,  # minimum number of extinct lineages allowed
                 minEXTANT_SP=1,
                 maxEXTANT_SP=np.inf,
                 root_r=[30.0, 100.0],  # range root ages
                 rangeL=[0.2, 0.5],
                 rangeM=[0.2, 0.5],
                 scale=100.0,
                 seed=None):
        self.s_species = s_species
        self.rangeSP = rangeSP
        self.minSP = np.min(rangeSP)
        self.maxSP = np.max(rangeSP)
        self.minEX_SP = minEX_SP
        self.minEXTANT_SP = minEXTANT_SP
        self.maxEXTANT_SP = maxEXTANT_SP
        self.root_r = root_r
        self.rangeL = rangeL
        self.rangeM = rangeM
        self.scale = scale
        self._rs = get_rnd_gen(seed)

    def reset_seed(self, seed):
        self._rs = get_rnd_gen(seed)

    def simulate(self, L, M, timesL, timesM, root, dd_model=False, verbose=False):
        ts = list()
        te = list()
        L, M, root = L / self.scale, M / self.scale, int(root * self.scale)

        if isinstance(self.s_species, Iterable):
            s_species = self._rs.integers(self.s_species[0], self.s_species[1])
            ts = list(np.zeros(s_species) + root)
            te = list(np.zeros(s_species))
        else:
            for i in range(self.s_species):
                ts.append(root)
                te.append(0)

        for t in range(root, 0):  # time
            for j in range(len(timesL) - 1):
                if -t / self.scale <= timesL[j] and -t / self.scale > timesL[j + 1]:
                    l = L[j]
            for j in range(len(timesM) - 1):
                if -t / self.scale <= timesM[j] and -t / self.scale > timesM[j + 1]:
                    m = M[j]

            TE = len(te)
            if TE > self.maxSP:
                break
            ran_vec = self._rs.random(TE)
            te_extant = np.where(np.array(te) == 0)[0]

            no = self._rs.random(2)  # draw a random number
            no_extant_lineages = len(te_extant)  # the number of currently extant species

            te = np.array(te)
            ext_species = np.where((ran_vec > l) & (ran_vec < (l + m)) & (te == 0))[0]
            te[ext_species] = t
            rr = ran_vec[te_extant]
            n_new_species = len(rr[rr < l])
            te = list(te) + list(np.zeros(n_new_species))
            ts = ts + list(np.zeros(n_new_species) + t)

        return -np.array(ts) / self.scale, -np.array(te) / self.scale


    def get_random_settings(self, root, num_trial, max_trial):
        root = np.abs(root)
        timesL = np.array([root, 0.])
        timesM = np.array([root, 0.])
        L = self._rs.uniform(np.min(self.rangeL), np.max(self.rangeL), 1)
        M = self._rs.uniform(np.min(self.rangeM), np.max(self.rangeM), 1)
        # Speciation should be higher than extinction
#        LM = np.sort(np.array([L, M]))
#        L = LM[1]
#        M = LM[0]
        if L < M and num_trial > max_trial/2:
            L = M * 1.1
        return timesL, timesM, L, M


    def run_simulation(self, print_res=False):
        LOtrue = [0]
        n_extinct = -0
        n_extant = -0
        min_extant = self.minEXTANT_SP
        max_extant = self.maxEXTANT_SP

        max_trial = 1e3
        num_trial = 0
        while (len(LOtrue) < self.minSP or len(LOtrue) > self.maxSP or n_extinct < self.minEX_SP or n_extant < min_extant or n_extant > max_extant) and num_trial < max_trial:
            if isinstance(self.root_r, Iterable):
                root = -self._rs.uniform(np.min(self.root_r), np.max(self.root_r))
            else:
                root = -self.root_r
            timesL, timesM, L, M = self.get_random_settings(root, num_trial, max_trial)
            FAtrue, LOtrue = self.simulate(L, M, timesL, timesM, root, verbose=print_res)
            n_extinct = len(LOtrue[LOtrue > 0])
            n_extant = len(LOtrue[LOtrue == 0])
            num_trial += 1

        if num_trial < max_trial:
            ts_te = np.array([FAtrue, LOtrue])
            if print_res:
                print("L", L, "M", M, "tL", timesL, "tM", timesM)
                print("N. species", len(LOtrue))
                max_standin_div = np.max([len(FAtrue[FAtrue > i]) - len(LOtrue[LOtrue > i]) for i in range(int(max(FAtrue)))]) / 80
                ltt = ""
                for i in range(int(max(FAtrue))):
                    n = len(FAtrue[FAtrue > i]) - len(LOtrue[LOtrue > i])
                    ltt += "\n%s\t%s\t%s" % (i, n, "*" * int(n / max_standin_div))
                print(ltt)
            sim_res = ts_te.T
        else:
            sim_res = None

        return sim_res

    def reset_s_species(self, s):
        self.s_species = s


class BdnnTester():
    def __init__(self,
                 sp_longevities,
                 extant_species,
                 n_traits=10,
                 levels_cat_trait=5,
                 n_nodes=[16, 8],
                 out_act_f=np.tanh,
                 act_f=np.tanh,
                 bdnn_update_f=[0.1, 0.2, 0.4],
                 prior_t_reg=-1.0,
                 independ_reg=False,
                 prior_cov=1.0,
                 mcmc_iterations=25000,
                 burnin=5000,
                 seed=None,
                 verbose=True
                 ):
        # Random number generator
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        # model settings
        self.n_nodes = n_nodes
        self.out_act_f = out_act_f
        self.act_f = act_f
        # data settings
        self.sp_longevities = sp_longevities
        self.extant_species = extant_species
        self.n_species = len(sp_longevities)
        self.n_traits = n_traits
        self.levels_cat_trait = levels_cat_trait + 1
        simulate_traits = self.simulate_traits()
        # mcmc settings
        self.bdnn_update_f = bdnn_update_f
        self.prior_t_reg = prior_t_reg
        self.independ_reg = independ_reg
        self.prior_cov = prior_cov
        self.mcmc_iterations = mcmc_iterations
        self.burnin = burnin
        self.verbose = verbose
        
    
    def reset_sp_longevities(self, sp_longevities, extant_species):
        self.sp_longevities = sp_longevities
        self.extant_species = extant_species
        self.n_species = len(sp_longevities)
        if self.traits.shape[0] != self.n_species:
            self.simulate_traits()
        
        
    def simulate_traits(self):
        # make up traits
        self.traits = self._rng.normal(0, 1, (self.n_species, self.n_traits))
        num_cat_traits = len(self.levels_cat_trait)
        if num_cat_traits > 0:
            for i in range(num_cat_traits):
                self.traits[:, i] = self._rng.choice(self.levels_cat_trait, size=self.n_species, replace=True)


    def get_bd_lik(self, w_lam, w_mu, t_reg):
        lam, _ = get_rate_BDNN(t_reg[0], self.traits, w_lam, act_f=self.act_f, out_act_f=self.out_act_f)
        mu, _ = get_rate_BDNN(t_reg[1], self.traits, w_mu, act_f=self.act_f, out_act_f=self.out_act_f)
        bd_lik = np.sum(np.log(lam)) + np.sum(self.extant_species * np.log(mu)) - np.sum((lam + mu) * self.sp_longevities)
        return bd_lik


    def get_prior(self, w_lam, w_mu, t_reg):
        prior = np.sum([np.sum(stats.norm.logpdf(i, loc=0, scale=self.prior_cov)) for i in w_lam])
        prior += np.sum([np.sum(stats.norm.logpdf(i, loc=0, scale=self.prior_cov)) for i in w_mu])
        if self.prior_t_reg[0] > 0.0:
            prior += np.log(self.prior_t_reg[0]) - self.prior_t_reg[0] * t_reg[0]
        if self.prior_t_reg[1] > 0.0 and self.independ_reg:
            prior += np.log(self.prior_t_reg[1]) - self.prior_t_reg[1] * t_reg[1]
        return prior

#    def get_prior(self, w_lam, w_mu):
#        prior = np.sum([np.sum(-(i**2) / 2) for i in w_lam])
#        prior += np.sum([np.sum(-(i**2) / 2)  for i in w_mu])
#        return prior
        
    def print_update(self, s):
        sys.stdout.write('\r')
        sys.stdout.write(s)
        sys.stdout.flush()

    def get_cv(self):
        # init model
        w_lamA = init_weight_prm(self.n_nodes, self.n_traits, size_output=1, init_std=0.1, bias_node=1)
        w_muA = init_weight_prm(self.n_nodes, self.n_traits, size_output=1, init_std=0.1, bias_node=1)
        t_regA = np.ones(2)
        t_reg = np.ones(2)
        if self.prior_t_reg[0] > 0.0:
            t_regA[0] = 0.5
            t_regA[1] = 0.5
        postA = self.get_bd_lik(w_lamA, w_muA, t_regA) + self.get_prior(w_lamA, w_muA, t_regA)
        
        # run mcmc
        lam_acc = []
        mu_acc = []
        for iteration in range(self.mcmc_iterations):
            w_lam, w_mu = copy_lib.deepcopy(w_lamA), copy_lib.deepcopy(w_muA)
            rnd_layer = self._rng.integers(0, len(w_lamA))
            # update layers B rate
            w_lam[rnd_layer] = update_parameter_normal_vec(w_lamA[rnd_layer],
                                                            d=0.05,
                                                            f=self.bdnn_update_f[rnd_layer] )
            # update layers D rate
            w_mu[rnd_layer] = update_parameter_normal_vec(w_muA[rnd_layer],
                                                            d=0.05,
                                                            f=self.bdnn_update_f[rnd_layer] )
            # update temp regularization
            if self.prior_t_reg[0] > 0.0:
                t_reg[0] = update_parameter(t_regA[0], 0, 1, d=0.05, f=1)
                if not self.independ_reg:
                    t_reg[1] = t_reg[0]
            if self.prior_t_reg[1] > 0.0 and self.independ_reg:
                t_reg[1] = update_parameter(t_regA[1], 0, 1, d=0.05, f=1)

            post = self.get_bd_lik(w_lam, w_mu, t_reg) + self.get_prior(w_lam, w_mu, t_reg)

            if iteration % 1000 == 0 and self.verbose:
                self.print_update("%s %s %s" % (iteration, post, postA))

            if (post - postA) > np.log(self._rng.random()):
                postA = post
                w_lamA, w_muA = copy_lib.deepcopy(w_lam), copy_lib.deepcopy(w_mu)
                t_regA = t_reg
    
            if iteration % 100 == 0 and iteration > self.burnin:
                lam, _ = get_rate_BDNN(t_reg[0], self.traits, w_lam, act_f=self.act_f, out_act_f=self.out_act_f)
                mu, _ = get_rate_BDNN(t_reg[1], self.traits, w_mu, act_f=self.act_f, out_act_f=self.out_act_f)
                lam_acc.append(lam)
                mu_acc.append(mu)

        # summarize results
        lam_acc = np.array(lam_acc)
        mu_acc = np.array(mu_acc)
        post_lam = np.mean(lam_acc, 0)
        post_mu = np.mean(mu_acc, 0)
        
        cv = np.zeros(2)
        cv[0] = np.std(post_lam) / np.mean(post_lam)
        cv[1] = np.std(post_mu) / np.mean(post_mu)
        if self.verbose:
            print('\nlambda', post_lam, "\n")
            print('\nmu', post_mu, "\n")
            print("\n\nCV lambda:", cv[0])
            print("CV mu:", cv[1], "\n")

        return cv


class BdnnTesterSampling():
    def __init__(self,
                 ts,
                 te,
                 occs_sp,
                 log_factorial_occs,
                 singleton_mask,
                 duration_q_bins=None,
                 occs_single_bin=None,
                 time_frames=None,
                 use_HPP_NN_lik=False,
                 TPP_model = 0,
                 n_traits=10,
                 levels_cat_trait=5,
                 n_nodes=[16, 8],
                 out_act_f=np.tanh,
                 act_f=np.tanh,
                 bdnn_update_f=[0.1, 0.2, 0.4],
                 prior_t_reg=-1.0,
                 prior_cov=1.0,
                 pert_prior=[1.5, 1.1],
                 mcmc_iterations=25000,
                 burnin=5000,
                 seed=None,
                 verbose=True
                 ):
        # Random number generator
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        # model settings
        self.use_HPP_NN_lik = use_HPP_NN_lik
        self.TPP_model = TPP_model
        self.n_nodes = n_nodes
        self.out_act_f = out_act_f
        self.act_f = act_f
        # data settings
        self.ts = ts
        self.te = te
        self.occs_sp = occs_sp
        self.log_factorial_occs = log_factorial_occs
        self.singleton_mask = singleton_mask
        self.duration_q_bins = duration_q_bins
        self.occs_single_bin = occs_single_bin
        self.time_frames = time_frames
        self.n_species = len(log_factorial_occs)
        self.n_traits = n_traits
        self.levels_cat_trait = levels_cat_trait + 1
        simulate_traits = self.simulate_traits()
        # mcmc settings
        self.bdnn_update_f = bdnn_update_f
        self.prior_t_reg = prior_t_reg
        self.prior_cov = prior_cov
        self.mcmc_iterations = mcmc_iterations
        self.burnin = burnin
        self.d2 = [1.2, 1.2]
        self.fQ = 0.5
        self.pert_prior = pert_prior
        self.verbose = verbose
        if self.use_HPP_NN_lik:
            self.time_in_q_bins = get_time_in_q_bins(self.ts, self.te, self.time_frames, self.duration_q_bins, self.occs_single_bin)


    def simulate_traits(self):
        # make up traits
        self.traits = self._rng.normal(0, 1, (self.n_species, self.n_traits))
        num_cat_traits = len(self.levels_cat_trait)
        if num_cat_traits > 0:
            for i in range(num_cat_traits):
                self.traits[:, i] = self._rng.choice(self.levels_cat_trait, size=self.n_species, replace=True)


    def get_fossil_lik(self, bdnn_q_rates):
        if self.use_HPP_NN_lik:
            # Could be even faster because this term does not change: get_time_in_q_bins(ts, te, time_frames, duration_q_bins, single_bin)
            fossil_lik = HPP_NN_lik([self.ts, self.te, bdnn_q_rates,
                                     self.occs_sp, self.log_factorial_occs,
                                     self.time_frames, self.duration_q_bins, self.occs_single_bin])
        else:
            fossil_lik = HOMPP_NN_lik([self.ts, self.te, bdnn_q_rates, self.occs_sp, self.log_factorial_occs])
        fossil_lik = np.sum(fossil_lik)
        return fossil_lik


    def get_prior(self, q_rates, w_q, t_reg):
        prior = np.sum([np.sum(stats.norm.logpdf(i, loc=0, scale=self.prior_cov)) for i in w_q])
        if self.prior_t_reg > 0.0:
            prior += np.log(self.prior_t_reg) - self.prior_t_reg * t_reg
        if self.TPP_model:
            if self.pert_prior[1] > 0.0:
                prior += np.sum(prior_gamma(q_rates, self.pert_prior[0], self.pert_prior[1]))
            else:
                hpGammaQ_shape = 1.01 # hyperprior is essentially flat
                hpGammaQ_rate = 0.1
                post_rate_prm_Gq = np.random.gamma(shape=hpGammaQ_shape + self.pert_prior[0] * len(q_rates), scale=1. / (hpGammaQ_rate + np.sum(q_rates)))
                prior += np.sum(prior_gamma(q_rates, self.pert_prior[0], post_rate_prm_Gq))
        else:
            prior += prior_gamma(q_rates[1], self.pert_prior[0], self.pert_prior[1])
        return prior


    def print_update(self, s):
        sys.stdout.write('\r')
        sys.stdout.write(s)
        sys.stdout.flush()

    def get_cv(self):
        # init model
        q_ratesA = np.array([np.random.uniform(.5, 1), np.random.uniform(0.25, 1)])
        if self.TPP_model == 1:
            q_ratesA = np.zeros(len(self.time_frames) - 1) + q_ratesA[1]
        else:
            # There is no non-homogeneous sampling for now
            q_ratesA[0] = 1.0

        w_qA = init_weight_prm(self.n_nodes, self.n_traits, size_output=1, init_std=0.1, bias_node=1)
        t_regA = 1.0
        t_reg = 1.0
        if self.prior_t_reg > 0.0:
            t_regA = 0.5
        bdnn_q_ratesA, _, _ = get_q_rate_BDNN(q_ratesA, t_regA, self.traits, w_qA, self.act_f, self.out_act_f, self.singleton_mask)
        postA = self.get_fossil_lik(bdnn_q_ratesA) + self.get_prior(q_ratesA, w_qA, t_regA)
        
        # run mcmc
        q_acc = []
        for iteration in range(self.mcmc_iterations):
            # update q_rates
            q_rates = np.zeros(len(q_ratesA)) + q_ratesA
            if self.TPP_model == 1:
                q_rates, hasting = update_q_multiplier(q_ratesA, d=self.d2[1], f=self.fQ)
            else:
                q_rates[1], hasting = update_multiplier_proposal(q_ratesA[1], self.d2[1])
            w_q = copy_lib.deepcopy(w_qA)
            rnd_layer = self._rng.integers(0, len(w_qA))
            # update layers sampling rate
            w_q[rnd_layer] = update_parameter_normal_vec(w_qA[rnd_layer],
                                                         d=0.05,
                                                         f=self.bdnn_update_f[rnd_layer])

            # update temp regularization
            if self.prior_t_reg > 0.0:
                t_reg = update_parameter(t_regA, 0, 1, d=0.05, f=1)

            bdnn_q_rates, _, _ = get_q_rate_BDNN(q_rates, t_reg, self.traits, w_q, self.act_f, self.out_act_f, self.singleton_mask)
            post = self.get_fossil_lik(bdnn_q_rates) + self.get_prior(q_rates, w_q, t_reg)

            if iteration % 1000 == 0 and self.verbose:
                self.print_update("%s %s %s" % (iteration, post, postA))

            if (post - postA + hasting) > np.log(self._rng.random()):
                q_ratesA = q_rates
                bdnn_q_ratesA = bdnn_q_rates
                postA = post
                w_qA = copy_lib.deepcopy(w_q)
                t_regA = t_reg
    
            if iteration % 100 == 0 and iteration > self.burnin:
                if self.use_HPP_NN_lik:
                    q_per_sp = harmonic_mean_q_per_sp(bdnn_q_ratesA, self.time_in_q_bins)
                else:
                    q_per_sp = bdnn_q_ratesA.reshape(-1)
                q_acc.append(q_per_sp)

        # summarize results
        q_acc = np.array(q_acc)
        post_q = np.mean(q_acc, axis=0)
        
        cv = np.std(post_q) / np.mean(post_q)
        if self.verbose:
            print('\nsampling', post_q, "\n")
            print("\n\nCV sampling:", cv)

        return cv


def get_CV_from_sim_i(arg):
    [rep, rangeSP, rangeL, rangeM, root_age, num_traits, levels_cat_trait, n_nodes, bdnn_update_f, act_f, out_act_f, prior_t_reg, independ_reg, prior_cov] = arg

#    # Random seed
#    rs = np.random.default_rng(None)
    
    sim_bd = BdSimulator(s_species=1,
                         rangeSP=rangeSP,
                         rangeL=rangeL,
                         rangeM=rangeM,
                         root_r=root_age,
                         seed=rep)
    sp_x = sim_bd.run_simulation(print_res=False)
    cv = np.full(2, np.nan)
    
    if isinstance(sp_x, np.ndarray):
        sp_longevities = sp_x[:,0] - sp_x[:,1]
        extant_species = (sp_x[:,1] == 0).astype(int)
        n_species = len(sp_longevities)

        bdnn_sim = BdnnTester(sp_longevities=sp_longevities,
                              extant_species=extant_species,
                              n_traits=num_traits,
                              levels_cat_trait=levels_cat_trait,
                              out_act_f=out_act_f,
                              act_f=act_f,
                              n_nodes=n_nodes,
                              bdnn_update_f=bdnn_update_f,
                              prior_t_reg=prior_t_reg,
                              independ_reg=independ_reg,
                              prior_cov=prior_cov,
                              verbose=False)
        cv = bdnn_sim.get_cv()
    return cv


def get_coefficient_sampling_variation(path_dir_log_files, burn, combine_discr_features="", num_sim=1000, num_processes=1, show_progressbar=False):
    pkl_file = path_dir_log_files + ".pkl"
    mcmc_file = path_dir_log_files + "_mcmc.log"
    q_tt_file = path_dir_log_files + "_q_rates.log"
    rates_mcmc_file = path_dir_log_files + "_per_species_q_rates.log"
    
    bdnn_obj = load_pkl(pkl_file)
    species_rates = np.loadtxt(rates_mcmc_file, skiprows = 1)
    s = species_rates.shape
    num_taxa = s[1] - 1
    num_it = s[0]
    burnin = check_burnin(burn, num_it)
    species_rates = species_rates[burnin:, 1:]
    
    ts_mean, te_mean = get_ts_te(mcmc_file, burnin)
    shift_time_q = np.array([0.0, np.max(ts_mean)])
    axis_q_mean = None
    use_HPP_NN_lik = False
    TPP_model = 0
    if 'q_time_frames' in bdnn_obj.bdnn_settings.keys():
        shift_time_q = np.copy(bdnn_obj.bdnn_settings['q_time_frames'])
        max_ts = np.max(ts_mean)
        if shift_time_q[0] < max_ts:
            # Sometime earliest speciation is prior to the lower boundary of the q-shifts
            shift_time_q[0] = max_ts
        axis_q_mean = 0
        use_HPP_NN_lik = True
        TPP_model = 1
    qtt, time_vec_q = get_qtt(q_tt_file, burn)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category = RuntimeWarning)
        qtt_mean = np.nanmean(qtt, axis = axis_q_mean)
    if isinstance(qtt_mean, np.floating):
        # np.mean is not returning an array for the overall mean (i.e. when there are no shifts through time)
        qtt_mean = np.array([qtt_mean])

    cv_rates = np.zeros((1, 3))
    cv_rates[0, 1] = get_cv(species_rates)

    num_traits, levels_cat_trait = get_num_traits_and_cat_levels(bdnn_obj, 'sampling', combine_discr_features=combine_discr_features)

    # Network architecture and priors
    layer_shapes = bdnn_obj.bdnn_settings['layers_shapes_q']
    n_nodes = []
    for i in range(len(layer_shapes) - 1):
        n_nodes.append(layer_shapes[i][0])
    bdnn_update_f = np.arange(1, len(layer_shapes) + 1)
    bdnn_update_f = bdnn_update_f / (2.0 * np.max(bdnn_update_f))
    act_f = bdnn_obj.bdnn_settings['hidden_act_f']
    out_act_f = bdnn_obj.bdnn_settings['out_act_f_q']
    prior_t_reg = -1.0
    if 'prior_t_reg' in bdnn_obj.bdnn_settings:
        prior_t_reg = bdnn_obj.bdnn_settings['prior_t_reg']
    prior_cov = 1.0
    if 'prior_cov' in bdnn_obj.bdnn_settings:
        prior_cov = bdnn_obj.bdnn_settings['prior_cov']

    args = []
    for i in range(num_sim):
        a = [i,
             ts_mean, te_mean, qtt_mean, shift_time_q,
             num_traits, levels_cat_trait,
             n_nodes, bdnn_update_f,
             act_f, out_act_f,
             prior_t_reg, prior_cov, bdnn_obj.bdnn_settings['pert_prior'],
             TPP_model, use_HPP_NN_lik]
        args.append(a)
    unixos = is_unix()
    if unixos and num_processes > 1:
        pool_perm = multiprocessing.Pool(num_processes)
        cv_sim = list(tqdm(pool_perm.imap_unordered(get_sampling_CV_from_sim_i, args),
                           total = num_sim, disable = show_progressbar == False))
        pool_perm.close()
    else:
        cv_sim = []
        for i in tqdm(range(num_sim), disable = show_progressbar == False):
            cv_sim.append(get_sampling_CV_from_sim_i(args[i]))

    cv_rates[0, 2] = np.nanquantile(np.array(cv_sim).reshape(-1), q=0.95)
    print('Coefficient of rate variation')
    print('    Sampling:', f'{float(cv_rates[0, 1]):.2f}', 'Expected:' + f'{float(cv_rates[0, 2]):.2f}')
    cv_rates = pd.DataFrame(cv_rates, columns = ['rate', 'cv_empirical', 'cv_expected'])
    cv_rates['rate'] = ['sampling']
    output_wd = os.path.dirname(path_dir_log_files)
    name_file = os.path.basename(path_dir_log_files)
    cv_rates_file = os.path.join(output_wd, name_file + '_coefficient_of_q_rate_variation.csv')
    cv_rates.to_csv(cv_rates_file, na_rep = 'NA', index = False)


def get_sampling_CV_from_sim_i(arg):
    [rep, ts, te, qtt_mean, shift_time_q, num_traits, levels_cat_trait, n_nodes, bdnn_update_f, act_f, out_act_f, prior_t_reg, prior_cov, pert_prior, TPP_model, use_HPP_NN_lik] = arg
    
    # Get fossils and their features for fast lik calculation
    fossils = sim_fossil_occurrences(ts, te, qtt_mean, shift_time_q)
    n_taxa = len(fossils)
    log_factorial_occs = None
    duration_q_bins = None
    occs_single_bin = None
    occs_sp = np.zeros(n_taxa)
    log_factorial_occs = np.zeros(n_taxa)
    for i in range(n_taxa):
        occs_sp[i] = np.count_nonzero(fossils[i])
        log_factorial_occs[i] = np.sum(np.log(np.arange(1, occs_sp[i] + 1)))
    if use_HPP_NN_lik:
        occs_sp = get_occs_sp(fossils, shift_time_q)
        log_factorial_occs, duration_q_bins, occs_single_bin = get_fossil_features_q_shifts(fossils, shift_time_q[::-1], occs_sp, te)
    singleton_mask = make_singleton_mask(occs_sp)
    
    
    bdnn_sim = BdnnTesterSampling(ts=ts,
                                  te=te,
                                  occs_sp=occs_sp,
                                  log_factorial_occs=log_factorial_occs,
                                  singleton_mask=singleton_mask,
                                  duration_q_bins=duration_q_bins,
                                  occs_single_bin=occs_single_bin,
                                  time_frames=shift_time_q,
                                  use_HPP_NN_lik=use_HPP_NN_lik,
                                  TPP_model = TPP_model,
                                  n_traits=num_traits,
                                  levels_cat_trait=levels_cat_trait,
                                  n_nodes=n_nodes,
                                  out_act_f=out_act_f,
                                  act_f=act_f,
                                  bdnn_update_f=bdnn_update_f,
                                  prior_t_reg=prior_t_reg,
                                  prior_cov=prior_cov,
                                  pert_prior=pert_prior,
                                  mcmc_iterations=25000,
                                  burnin=5000,
                                  seed=rep,
                                  verbose=False)
    cv = bdnn_sim.get_cv()
    return cv



def get_ts_te(mcmc_file, burnin):
    m = pd.read_csv(mcmc_file, delimiter = '\t')
    ts_indx = [i for i in range(len(m.columns)) if '_TS' in m.columns[i]]
    te_indx = [i for i in range(len(m.columns)) if '_TE' in m.columns[i]]
    np_m = m.to_numpy().astype(float)
    ts = np_m[burnin:, ts_indx]
    te = np_m[burnin:, te_indx]
    ts = np.mean(ts, axis=0)
    te = np.mean(te, axis=0)
    return ts, te


def get_duration(ts, te, upper, lower):
    ts2 = np.copy(ts)
    te2 = np.copy(te)
    ts2[ts2 > upper] = upper
    te2[te2 < lower] = lower
    d = ts2 - te2
    d[d < 0.0] = 0.0
    return d, ts2, te2


def sim_fossil_occurrences(ts, te, q, shift_time_q):
    is_alive = te == 0.0
    n_taxa = len(ts)
    occ = [np.array([])] * n_taxa
    len_q = len(q)
    for i in range(len_q):
        dur, ts2, te2 = get_duration(ts, te, shift_time_q[i], shift_time_q[i + 1])
        poi_rate_occ = q[i] * dur
        exp_occ = np.random.poisson(poi_rate_occ)
        exp_occ = np.round(exp_occ)
        for y in range(n_taxa):
            occ_y = np.random.uniform(ts2[y], te2[y], exp_occ[y])
            present = np.array([])
            if is_alive[y] and i == (len_q - 1): # Alive and most recent sampling strata
                present = np.zeros(1, dtype='float')
            occ[y] = np.sort(np.concatenate((occ[y], occ_y, present)))[::-1]
    for i in range(n_taxa):
        O = occ[i]
        O = O[O != 0.0] # Do not count single occurrence at the present
        if len(O) == 0:
            add_singleton = np.random.uniform(te[i], ts[i], size=1)
            occ[i] = np.concatenate((add_singleton, occ[i]), axis=None)
    return occ


# Credible differences
######################
def get_prob_1_bin_trait(cond_rates_eff):
    d1 = cond_rates_eff[0, :]
    d2 = cond_rates_eff[1, :]
    prob = get_prob(d1, d2, len(d1))
    mag = d1 / d2
    mean_mag = np.mean(mag)
    mag_HPD = np.array([np.nan, np.nan])
    if np.sum(~np.isnan(mag)) > 2:
        mag_HPD = util.calcHPD(mag[~np.isnan(mag)], .95)
    return np.array([prob, mean_mag, mag_HPD[0], mag_HPD[1]])


def get_prob_1_con_trait(cond_rates_eff):
    mean_rate = np.mean(cond_rates_eff, axis = 1)
    d1 = cond_rates_eff[np.nanargmax(mean_rate), :]
    d2 = cond_rates_eff[np.nanargmin(mean_rate), :]
    d = d1 - d2
    d = d.flatten()
    n = np.sum(d > 0.0)
    prob = n / len(d)
    mag = d1 / d2
    mean_mag = np.mean(mag)
    mag_HPD = np.array([np.nan, np.nan])
    if np.sum(~np.isnan(mag)) > 2:
        mag_HPD = util.calcHPD(mag[~np.isnan(mag)], .95)
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
        idx_largest_diff = np.argmax(mean_diff_state)
        idx_smallest_diff = np.argmin(mean_diff_state)
        d1 = diff_state[idx_largest_diff, :]
        d2 = diff_state[idx_smallest_diff, :]
        prob = get_prob(d1, d2, len(d1))
#        m1 = np.abs(cond_rates_state0[idx_largest_diff, :] - cond_rates_state0[idx_smallest_diff, :])
#        m2 = np.abs(cond_rates_state1[idx_largest_diff, :] - cond_rates_state1[idx_smallest_diff, :])
#        mag = m1 / m2
        mag = np.abs(diff_state[idx_largest_diff, :]) / np.abs(diff_state[idx_smallest_diff, :])
        mean_mag = np.nanmedian(mag)
        if np.sum(~np.isnan(mag)) > 2:
            mag_HPD = util.calcHPD(mag[~np.isnan(mag)], .95)
    return np.array([prob, mean_mag, mag_HPD[0], mag_HPD[1]])


def get_prob_discr_ord(cond_rates_eff, names, names_states):
    # discrete and ordinal features
    n = len(names_states)
    all_combinations = list(combinations(np.arange(n), 2))
    n_comb = len(all_combinations)
    col0 = []
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    col7 = []
    for j in range(n_comb):
        l = list(all_combinations[j])
        col0.append(names)
        col1.append('none')
        col2.append(str(names_states[l[0]]) + '_' + str(names_states[l[1]]))
        col3.append('none')
        p = get_prob_1_bin_trait(cond_rates_eff[l,:])
        col4.append(p[0])
        col5.append(p[1])
        col6.append(p[2])
        col7.append(p[3])
    p_df = pd.DataFrame({'0': col0, '1': col1, '2': col2, '3': col3, '4': col4, '5': col5, '6': col6, '7': col7})
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
    min_states_feat_1 = np.min(comb_states_feat_1)
    min_states_feat_2 = np.min(comb_states_feat_2)
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
                                '2': str(names_states_feat_1[comb_states_feat_1[j, 0] - min_states_feat_1]) + '_' + str(names_states_feat_1[comb_states_feat_1[j, 1] - min_states_feat_1]),
                                '3': str(names_states_feat_2[comb_states_feat_2[k, 0] - min_states_feat_2]) + '_' + str(names_states_feat_2[comb_states_feat_2[k, 1] - min_states_feat_2]),
                                '4': prob[0], '5': prob[1], '6': prob[2], '7': prob[3]},
                                index = [0])
            p_df = pd.concat([p_df, pjk], ignore_index = True)
    return p_df


def get_prob_inter_cont_discr_ord(cond_rates_eff, trait_tbl_eff, names_cont, names_discr_ord, names_states):
    # interaction between discr/ordinal features with a continuous feature
    n = len(names_states)
    all_combinations = list(combinations(np.arange(n), 2))
    n_comb = len(all_combinations)
    col0 = []
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    col7 = []
    for j in range(n_comb):
        l = list(all_combinations[j])
        if trait_tbl_eff.shape[1] > 2:
            # one-hot encoding
            state0 = trait_tbl_eff[:,l[0] + 1] == 1
            state1 = trait_tbl_eff[:, l[1] + 1] == 1
        else:
            # ordinal
            try:
                min_state = int(names_states[0])
                if min_state > np.min(trait_tbl_eff[:, 1]):
                    min_state = 0.0
            except:
                min_state = 0.0
            state0 = trait_tbl_eff[:, 1] == l[0] + min_state
            state1 = trait_tbl_eff[:, 1] == l[1] + min_state
        focal_states = np.logical_or(state0, state1)
        cond_rates_eff_j = cond_rates_eff[focal_states,:]
        trait_tbl_j = trait_tbl_eff[focal_states,:]
        if trait_tbl_eff.shape[1] > 2:
            # one-hot encoding
            state0 = trait_tbl_j[:, l[0] + 1] == 1
        else:
            # ordinal
            state0 = trait_tbl_j[:, 1] == l[0] + min_state
        col0.append(names_cont)
        col1.append(names_discr_ord)
        col2.append('none')
        col3.append(str(names_states[l[0]]) + '_' + str(names_states[l[1]]))
        p = get_prob_inter_bin_con_trait(cond_rates_eff_j, state0)
        col4.append(p[0])
        col5.append(p[1])
        col6.append(p[2])
        col7.append(p[3])
    p_df = pd.DataFrame({'0': col0, '1': col1, '2': col2, '3': col3, '4': col4, '5': col5, '6': col6, '7': col7})
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
    mag_HPD = np.array([np.nan, np.nan])
    if np.sum(~np.isnan(mag)) > 2:
        mag_HPD = util.calcHPD(mag[~np.isnan(mag)], .95)
    # Magnitude interaction greater than for the more important feature of both?
    if np.mean(mag) > 1:
        prob = np.sum(mag > mag_single_feat) / niter_mcmc
    else:
        prob = np.sum(mag < mag_single_feat) / niter_mcmc
    return np.array([prob, mean_mag, mag_HPD[0], mag_HPD[1]])


def get_prob_effects(cond_trait_tbl, cond_rates, bdnn_obj, names_features, rate_type = 'speciation', ):
    names_features_original = np.array(get_names_features(bdnn_obj, rate_type=rate_type))
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
        elif pt == 15.0: # Multiple continuous features combined in a feature group
            names = np.unique(names_features[incl_features])
            prob = pd.DataFrame({'0': names[0], '1': 'none', '2': 'none', '3': 'none',
                                 '4': -1.0, '5': np.nan, '6': np.nan, '7': np.nan},
                                index=[0])
            prob_effects = pd.concat([prob_effects, prob], ignore_index = True)
        elif pt == 16.0: # Interaction of multiple continuous features with any other feature
            names = np.sort(np.unique(names_features[incl_features]))
            prob = pd.DataFrame({'0': names[0], '1': names[1], '2': 'none', '3': 'none',
                                 '4': -1.0, '5': np.nan, '6': np.nan, '7': np.nan},
                                index=[0])
            prob_effects = pd.concat([prob_effects, prob], ignore_index = True)
    prob_effects.columns = ['feature1', 'feature2', 'feature1_state', 'feature2_state',
                            'posterior_probability',
                            'magnitude_effect', 'magnitude_lwr_CI', 'magnitude_upr_CI']
    prob_effects = prob_effects[~prob_effects.duplicated(['feature1', 'feature2', 'feature1_state', 'feature2_state'])] # Feature groups with multiple continuous features are appearing several times. MAybe only when they are not in adjecent columns? Remove this.
    return prob_effects


def get_shap_trt_tbl(tse, times, trt_tbl):
    if trt_tbl.ndim == 2:
        shap_trt_tbl = trt_tbl + 0.0
    else:
        # In case of combined replicates where the times can differ among replicates we need to order from present to past 
        # so that tse = 0 is always the first bin in the trt_tbl
        trt_tbl = trt_tbl[::-1, :, :]
        times = times[::-1]
        s = trt_tbl.shape
        n_species = s[1]
        n_features = s[2]
        digitized_tse = np.digitize(tse, times) - 1
        digitized_tse[digitized_tse < 0] = 0 # Should not do any harm for speciation
        shap_trt_tbl = np.zeros((n_species, n_features))
        for i in range(n_species):
            shap_trt_tbl[i,:] = trt_tbl[digitized_tse[i], i, :]
    return shap_trt_tbl


def get_shap_trt_tbl_sampling(bin_species, trt_tbl):
    if trt_tbl.ndim == 2:
        shap_trt_tbl = trt_tbl + 0.0
    else:
        s = trt_tbl.shape
        n_species = s[1]
        n_features = s[2]
        shap_trt_tbl = np.zeros((n_species, n_features))
        for i in range(n_species):
            shap_trt_tbl[i,:] = np.mean(trt_tbl[bin_species[i, :] > 0, i, :], axis=0)
    return shap_trt_tbl


def insert_onehot(trait_tbl_tmp, idx_comb_feat, idx_feat):
    len_comb = len(idx_comb_feat)
    if len_comb > 0:
        if np.isin(idx_feat, np.concatenate(idx_comb_feat)):
            for l in range(len_comb):
                idx_comb_feat_l = idx_comb_feat[l]
                if np.isin(idx_feat, idx_comb_feat_l):
                    not_idx_feat = idx_comb_feat_l[idx_feat != idx_comb_feat_l]
                    trait_tbl_tmp[:, not_idx_feat] = 0.0
    return trait_tbl_tmp



def take_traits_from_trt_tbl(trait_tbl, cond_trait_tbl, j, idx_comb_feat):
    trait_tbl_tmp = trait_tbl + 0.0
    idx_feat1 = int(cond_trait_tbl[j, -6])
    idx_feat2 = cond_trait_tbl[j, -5]
    trait_tbl_tmp[: , idx_feat1] = cond_trait_tbl[j, idx_feat1]
    trait_tbl_tmp = insert_onehot(trait_tbl_tmp, idx_comb_feat, idx_feat1)
    if not np.isnan(idx_feat2):
        idx_feat2 = int(idx_feat2)
        trait_tbl_tmp[:, idx_feat2] = cond_trait_tbl[j, idx_feat2]
        trait_tbl_tmp = insert_onehot(trait_tbl_tmp, idx_comb_feat, idx_feat2)
    return trait_tbl_tmp


def get_pdp_rate_it_i(arg):
    [bdnn_obj, post_w_i, post_t_reg_i, post_denom_i, baseline, norm, trait_tbl, cond_trait_tbl, idx_comb_feat, out_act_f] = arg
    nrows_cond_trait_tbl = len(cond_trait_tbl)
    rate_it_i = np.zeros(nrows_cond_trait_tbl)
    rate_it_i[:] = np.nan
    obs = cond_trait_tbl[:, -1] == 1
    for j in range(nrows_cond_trait_tbl):
        if obs[j]:
            trait_tbl_tmp = take_traits_from_trt_tbl(trait_tbl, cond_trait_tbl, j, idx_comb_feat)
            rate_BDNN = get_rate_BDNN_noreg(trait_tbl_tmp,
                                            post_w_i,  # list of arrays
                                            bdnn_obj.bdnn_settings['hidden_act_f'],
                                            out_act_f)
            rate_BDNN = (rate_BDNN * norm) ** post_t_reg_i / post_denom_i
            rate_BDNN = baseline * rate_BDNN
            rate_it_i[j] = 1.0 / np.mean(1.0 / rate_BDNN) #np.mean(rate_BDNN)
    return rate_it_i


def get_partial_dependence_rates(bdnn_obj, cond_trait_tbl, post_w, post_t_reg, post_denom,
                                 post_ts, post_te,
                                 combine_discr_features = '', rate_type = 'speciation',
                                 num_processes = 1, show_progressbar = False, baseline=np.ones(1), norm=np.ones(1)):
    num_it = len(post_w)
    trait_tbl = get_trt_tbl(bdnn_obj, rate_type)
    names_features = get_names_features(bdnn_obj, rate_type=rate_type)
    idx_comb_feat = get_idx_comb_feat(names_features, combine_discr_features)
    args = []
    if rate_type == 'sampling':
        out_act_f = bdnn_obj.bdnn_settings['out_act_f_q']
    else:
        out_act_f = bdnn_obj.bdnn_settings['out_act_f']
    for i in range(num_it):
        trait_tbl_a = trait_tbl + 0.0
        if rate_type == 'sampling':
            if 'taxon_age' in names_features:
                trait_tbl_a = add_taxon_age(post_ts[i, :], post_te[i, :], bdnn_obj.bdnn_settings['q_time_frames'], trait_tbl_a)
            trait_tbl_a = get_shap_trt_tbl_sampling(bdnn_obj.bdnn_settings['occs_sp'], trait_tbl_a)
        elif rate_type == 'speciation':
            bdnn_time = get_bdnn_time(bdnn_obj, post_ts[i, :])
            trait_tbl_a = get_shap_trt_tbl(post_ts[i, :], bdnn_time, trait_tbl_a)
        else:
            bdnn_time = get_bdnn_time(bdnn_obj, post_ts[i, :])
            trait_tbl_a = get_shap_trt_tbl(post_te[i, :], bdnn_time, trait_tbl_a)
        b=baseline
        n=norm
        if len(baseline) > 1:
            b = baseline[i, :]
            n = norm[i]
        a = [bdnn_obj, post_w[i], post_t_reg[i], post_denom[i], b, n, trait_tbl_a, cond_trait_tbl, idx_comb_feat, out_act_f]
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


def match_names_comb_with_features(names_comb, names_features):
    keys_names_comb = get_names_feature_group(names_comb)
    m = []
    for i in range(len(names_comb)):
        names_comb_i = names_comb[keys_names_comb[i]]
        J = len(names_comb_i)
        idx = np.zeros(J)
        for j in range(J):
            idx[j] = np.where(names_comb_i[j] == names_features)[0]
        m.append(idx)
    return m


def get_all_combination(names_comb_idx, minmaxmean_features):
    l = []
    for i in range(len(names_comb_idx)):
        idx_i = names_comb_idx[i]
        for j in range(len(idx_i)):
            jj = int(idx_i[j])
            v1 = np.linspace(minmaxmean_features[0, jj], minmaxmean_features[1, jj], int(minmaxmean_features[3, jj]))
            l.append(v1)
    all_comb = build_all_combinations(l)
    # Remove rows where we have more than one state of a one-hot encoded feature
    names_comb_idx_conc = np.concatenate(names_comb_idx).astype(int)
    for i in range(len(names_comb_idx)):
        idx_i = names_comb_idx[i]
        if len(idx_i) > 1:
            col_idx = np.arange(len(names_comb_idx_conc))[np.isin(names_comb_idx_conc, idx_i)]
            keep = np.sum(all_comb[:, col_idx], axis = 1) == 1
            all_comb = all_comb[keep, :]
    return all_comb


def build_all_combinations(list):
    all_comb = [p for p in product(*list)]
    all_comb_np = np.array(all_comb)
    return all_comb_np


def get_pdp_rate_it_i_free_combination(arg):
    [bdnn_obj, post_w_i, t_reg_i, denom_reg_i, trait_tbl, all_comb_tbl, names_comb_idx_conc, out_act_f] = arg
    nrows_all_comb_tbl = len(all_comb_tbl)
    rate_it_i = np.zeros(nrows_all_comb_tbl)
    rate_it_i[:] = np.nan
    for j in range(nrows_all_comb_tbl):
        trait_tbl_tmp = trait_tbl + 0.0
        trait_tbl_tmp[:, names_comb_idx_conc] = all_comb_tbl[j, :]
        rate_BDNN = get_rate_BDNN_noreg(trait_tbl_tmp,
                                        post_w_i,  # list of arrays
                                        bdnn_obj.bdnn_settings['hidden_act_f'],
                                        out_act_f)
        rate_BDNN = rate_BDNN ** t_reg_i / denom_reg_i
        rate_it_i[j] = 1.0 / np.mean(1.0/rate_BDNN)
    return rate_it_i


def get_pdp_rate_free_combination(bdnn_obj,
                                  sp_fad_lad,
                                  ts_post, te_post, w_post, t_reg_post, denom_reg_post,
                                  names_comb,
                                  backscale_par,
                                  len_cont=100,
                                  rate_type="speciation",
                                  fix_observed=False,
                                  num_processes=1,
                                  show_progressbar=False):
    trait_tbl = get_trt_tbl(bdnn_obj, rate_type)
    names_features = get_names_features(bdnn_obj, rate_type=rate_type)
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
    conc_comb_feat = np.array([])
    names_features = np.array(names_features)
    names_comb_idx = match_names_comb_with_features(names_comb, names_features)
    names_comb_idx_conc = np.concatenate(names_comb_idx).astype(int)
    binary_feature, most_frequent_state = is_binary_feature(trait_tbl)
    minmaxmean_features = get_minmaxmean_features(trait_tbl,
                                                  most_frequent_state,
                                                  binary_feature,
                                                  conc_comb_feat,
                                                  len_cont)
    feature_is_time_variable = is_time_variable_feature(trait_tbl)[0, :]
    if trait_tbl.ndim == 3:
        # In case of combined replicates where the times can differ among replicates we need to order from present to past.
        trait_tbl = trait_tbl[::-1, :, :]
    tste = get_mean_inferred_tste(ts_post, te_post)
    if np.any(feature_is_time_variable):
        fossil_bin_ts = get_bin_from_fossil_age(bdnn_obj, tste, 'speciation', reverse_time=True)
        fossil_bin_te = get_bin_from_fossil_age(bdnn_obj, tste, 'extinction', reverse_time=True)
        n_taxa = len(fossil_bin_te)
        trait_at_ts = np.zeros((n_taxa, trait_tbl.shape[2]))
        trait_at_te = np.zeros((n_taxa, trait_tbl.shape[2]))
        for k in range(n_taxa):
            trait_at_ts[k, :] = trait_tbl[fossil_bin_ts[k], k, :]
            trait_at_te[k, :] = trait_tbl[fossil_bin_te[k], k, :]
        trait_at_ts_or_te = np.vstack((trait_at_ts, trait_at_te))
        minmaxmean_features[0, feature_is_time_variable] = np.min(trait_at_ts_or_te[:, feature_is_time_variable], axis = 0)
        minmaxmean_features[1, feature_is_time_variable] = np.max(trait_at_ts_or_te[:, feature_is_time_variable], axis = 0)
        if is_time_trait(bdnn_obj):
            tste_rescaled = bdnn_time_rescaler(tste, bdnn_obj)
            min_tste = np.min(tste_rescaled)
            if min_tste < minmaxmean_features[0, -1]:
                minmaxmean_features[1, -1] = min_tste * 0.98
            max_tste = np.max(tste_rescaled)
            if max_tste > minmaxmean_features[1, -1]:
                minmaxmean_features[1, -1] = max_tste * 1.02
    if fix_observed is False:
        all_comb_tbl = get_all_combination(names_comb_idx, minmaxmean_features)
    else:
        all_comb_tbl = trait_tbl[0]
        if rate_type == "speciation":
            all_comb_tbl[:, feature_is_time_variable] = trait_at_ts[:, feature_is_time_variable]
        else:
            all_comb_tbl[:, feature_is_time_variable] = trait_at_te[:, feature_is_time_variable]
        all_comb_tbl = all_comb_tbl[:, names_comb_idx_conc]
    bdnn_time = get_bdnn_time(bdnn_obj, np.max(ts_post))
    bdnn_time = bdnn_time[::-1]
    if rate_type == 'sampling':
        out_act_f = bdnn_obj.bdnn_settings['out_act_f_q']
    else:
        out_act_f = bdnn_obj.bdnn_settings['out_act_f']
    num_it = len(w_post)
    trait_tbl_for_mean = np.zeros((num_it, trait_tbl.shape[-2], len(names_comb_idx_conc)))
    args = []
    for i in range(num_it):
        trait_tbl_a = trait_tbl + 0.0
        if rate_type == "speciation":
            trait_tbl_a = get_shap_trt_tbl(ts_post[i, :], bdnn_time, trait_tbl_a)
        else:
            trait_tbl_a = get_shap_trt_tbl(te_post[i, :], bdnn_time, trait_tbl_a)
        trait_tbl_for_mean[i, :, :] = trait_tbl_a[:, names_comb_idx_conc]
        a = [bdnn_obj, w_post[i], t_reg_post[i], denom_reg_post[i], trait_tbl_a, all_comb_tbl, names_comb_idx_conc, out_act_f]
        args.append(a)
    trait_tbl_mean = np.mean(trait_tbl_for_mean, axis = 0)
    b = binary_feature[names_comb_idx_conc]
    trait_tbl_mean[:, b] = stats.mode(trait_tbl_for_mean[:, :, b], axis = 0)[0]
    if fix_observed:
        trait_tbl_mean = all_comb_tbl + 0.0
    unixos = is_unix()
    if unixos and num_processes > 1:
        pool_perm = multiprocessing.Pool(num_processes)
        rate_pdp = list(tqdm(pool_perm.imap_unordered(get_pdp_rate_it_i_free_combination, args),
                             total = num_it, disable = show_progressbar == False))
        pool_perm.close()
    else:
        rate_pdp = []
        for i in tqdm(range(num_it), disable = show_progressbar == False):
            rate_pdp.append(get_pdp_rate_it_i_free_combination(args[i]))
    rate_pdp = np.stack(rate_pdp, axis = 1)
#    np.savetxt("/home/torsten/Work/BDNN/Proboscideans/PyRateAnalyses40Ma/Humans_Island_SpTemp_Grass_Feb_2024/NMDS2_Humans/NMDS2_Humans_" + rate_type + "_full.txt", rate_pdp, delimiter="\t")
#    np.savetxt("/home/torsten/Work/BDNN/Proboscideans/PyRateAnalyses40Ma/Humans_Island_SpTemp_Grass_Feb_2024/Geography/Geography_" + rate_type + "_pdp.txt", rate_pdp, delimiter="\t")
    rate_pdp_sum = get_rates_summary(rate_pdp)
    rate_pdp_sum_df = pd.DataFrame(rate_pdp_sum, columns = ['mean', 'lwr', 'upr'])
    names_features = names_features[names_comb_idx_conc]
    if is_time_trait(bdnn_obj):
        time_idx = np.where('time' == names_features)[0]
        all_comb_tbl[:, time_idx] = backscale_bdnn_time(all_comb_tbl[:, time_idx], bdnn_obj)
        trait_tbl_mean[:, time_idx] = backscale_bdnn_time(trait_tbl_mean[:, time_idx], bdnn_obj)
    all_comb_tbl = backscale_bdnn_diversity(all_comb_tbl, bdnn_obj, names_features)
    trait_tbl_mean = backscale_bdnn_diversity(trait_tbl_mean, bdnn_obj, names_features)
    all_comb_tbl = backscale_tbl(bdnn_obj, backscale_par, names_features, all_comb_tbl)
    trait_tbl_mean = backscale_tbl(bdnn_obj, backscale_par, names_features, trait_tbl_mean)
    all_comb_df = pd.DataFrame(all_comb_tbl, columns = names_features)
    rate_out = pd.concat([all_comb_df, rate_pdp_sum_df], axis = 1)
    trt_df = pd.DataFrame(trait_tbl_mean, columns = names_features, index = sp_fad_lad['Taxon'])
    return rate_out, trt_df, names_features.tolist()


def get_greenwells_feature_importance(cond_trait_tbl, pdp_rates, bdnn_obj, names_features, rate_type = 'speciation'):
    names_features_original = np.array(get_names_features(bdnn_obj, rate_type=rate_type))
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

def get_rate_BDNN_noreg(x, w, act_f, out_act_f):
    tmp = x + 0
    for i in range(len(w)-1):
        tmp = act_f(MatrixMultiplication(tmp, w[i]))
    tmp = MatrixMultiplication(tmp, w[i + 1])
    rates = out_act_f(tmp).flatten() + small_number
    return rates


def get_rate_BDNN_3D_noreg(x, w, act_f, out_act_f, sampling=False, singleton_mask=None, qbin_ts_te=None):
    tmp = x+0
    for i in range(len(w)-1):
        tmp = act_f(MatrixMultiplication3D(tmp, w[i]))
    
    tmp = MatrixMultiplication3D(tmp, w[i+1])
    tmp = np.squeeze(tmp).T
    # output
    rates = out_act_f(tmp)
    
    if sampling:
        if rates.ndim == 2:
            # set rate to NA before ts and after te
            for i in range(rates.shape[0]):
                rates[i, :qbin_ts_te[i, 0]] = np.nan
                rates[i, qbin_ts_te[i, 1]:] = np.nan
            rates_not_nan = np.isnan(rates) == False
        else:
            rates_not_nan = np.repeat(True, rates.shape[-1])
        rates[np.logical_and(singleton_mask, rates_not_nan)] = 1.0

    rates += small_number
    return rates


def get_bdnn_lik(bdnn_obj, bdnn_time, i_events_sp, i_events_ex, n_S, w, t_reg, reg_denom, trait_tbl_NN, rate_type):
    hidden_act_f = bdnn_obj.bdnn_settings['hidden_act_f']
    out_act_f = bdnn_obj.bdnn_settings['out_act_f']
    trait_tbl_idx = 0
    if rate_type == 'm':
        trait_tbl_idx = 1
    r = get_rate_BDNN_3D_noreg(trait_tbl_NN[trait_tbl_idx], w, hidden_act_f, out_act_f)
    r = r ** t_reg / reg_denom
    if rate_type == 'l':
        args = [ i_events_sp, n_S, r ]
    else:
        args = [ i_events_ex, n_S, r ]
    bdnn_lik = BDNN_fast_partial_lik(args)
    return bdnn_lik


def create_perm_comb(bdnn_obj, do_inter_imp = True, combine_discr_features = None, rate_type='speciation'):
    names_features = get_names_features(bdnn_obj, rate_type)
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
    if do_inter_imp:
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


def permute_trt_tbl(feat_idx, feature_is_time_variable, post_ts_i, trt_tbl_lowres, trt_tbl=None, seed=None):
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
    use_high_res = False
    if trt_tbl_lowres[0].ndim == 3:
        if np.any(feature_is_time_variable[0, feat_idx]):
            use_high_res = True
#            print(feat_idx)
            if np.any(feature_is_time_variable[1, feat_idx]):
                ## Swapping time for all species together among bins
#                print('Varies through time but not species', feat_idx)
                n_bins = trt_tbl[0].shape[0]
                bins_perm_idx = rng.permuted(np.arange(n_bins))
                trt_tbl[0][:, :, feat_idx] = trt_tbl[0][bins_perm_idx, :, :][:, :, feat_idx]
                trt_tbl[1][:, :, feat_idx] = trt_tbl[1][bins_perm_idx, :, :][:, :, feat_idx]
            if np.any(feature_is_time_variable[2, feat_idx]):
#                print('Varies through time and species', feat_idx)
                ## Free permutation
                feat_sp = trt_tbl[0][:, :, feat_idx]
                feat_ex = trt_tbl[1][:, :, feat_idx]
                n = len(feat_sp)
                perm_idx = rng.permuted(np.arange(n))
                trt_tbl[0][:, :, feat_idx] = feat_sp[perm_idx]
                trt_tbl[1][:, :, feat_idx] = feat_ex[perm_idx]
                ## Permutation (sampling with replacement) according to relative length of each time-bin (what if there is no time e.g. only div-dep?)
#                n_species = trt_tbl[0].shape[1]
#                n_bins = trt_tbl[0].shape[0]
#                bdnn_time = get_bdnn_time(bdnn_obj, post_ts_i)
#                rel_time = np.abs(np.diff(bdnn_time)) / np.max(bdnn_time)
#                perm_idx = rng.choice(n_bins, n_bins * n_species, p = rel_time).reshape((n_bins, n_species))
#                trt_tbl_org = copy_lib.deepcopy(trt_tbl)
#                for s in range(n_species):
#                   trt_tbl[0][:, s, feat_idx] = trt_tbl_org[0][:, s, feat_idx][perm_idx[:, s], :]
#                   trt_tbl[1][:, s, feat_idx] = trt_tbl_org[1][:, s, feat_idx][perm_idx[:, s], :]
        else:
            n_species = trt_tbl[0].shape[1]
            species_perm_idx = rng.permuted(np.arange(n_species))
            trt_tbl_lowres[0][:, :, feat_idx] = trt_tbl_lowres[0][:, species_perm_idx, :][:, :, feat_idx]
            trt_tbl_lowres[1][:, :, feat_idx] = trt_tbl_lowres[1][:, species_perm_idx, :][:, :, feat_idx]
            trt_tbl = trt_tbl_lowres
    else:
        n_species = trt_tbl_lowres[0].shape[0]
        species_perm_idx = rng.permuted(np.arange(n_species))
        trt_tbl_lowres[0][:, feat_idx] = trt_tbl_lowres[0][species_perm_idx, :][:, feat_idx]
        trt_tbl_lowres[1][:, feat_idx] = trt_tbl_lowres[1][species_perm_idx, :][:, feat_idx]
        trt_tbl = trt_tbl_lowres
    return trt_tbl, use_high_res


def perm_mcmc_sample_i(arg):
    [bdnn_obj, post_ts_i, post_te_i,
     bdnn_time, i_events_sp, i_events_ex, n_S,
     bdnn_time_highres, i_events_sp_highres, i_events_ex_highres, n_S_highres,
     post_w_sp_i, post_w_ex_i, t_reg_lam_i, t_reg_mu_i, reg_denom_lam_i, reg_denom_mu_i,
     trt_tbls, trt_tbls_highres, n_perm, n_perm_traits, n_features, feature_is_time_variable, bdnn_dd, div_idx_trt_tbl, perm_feature_idx] = arg
    if bdnn_dd:
        n_taxa = trt_tbls[0].shape[1]
        bdnn_rescale_div = bdnn_obj.bdnn_settings['div_rescaler']
        bdnn_time_div = np.arange(np.max(post_ts_i), 0.0, -0.001)
        bdnn_div = get_DT(bdnn_time_div, post_ts_i, post_te_i)
        bdnn_binned_div_highres = get_binned_div_traj(bdnn_time_highres, bdnn_time_div, bdnn_div)[:-1] / bdnn_rescale_div
        bdnn_binned_div_highres = np.repeat(bdnn_binned_div_highres, n_taxa).reshape((len(bdnn_binned_div_highres), n_taxa))
        trt_tbls_highres[0][:, :, div_idx_trt_tbl] = bdnn_binned_div_highres
        trt_tbls_highres[1][:, :, div_idx_trt_tbl] = bdnn_binned_div_highres
        bdnn_binned_div = get_binned_div_traj(bdnn_time, bdnn_time_div, bdnn_div)[:-1] / bdnn_rescale_div
        bdnn_binned_div = np.repeat(bdnn_binned_div, n_taxa).reshape((len(bdnn_binned_div), n_taxa))
        trt_tbls[0][:, :, div_idx_trt_tbl] = bdnn_binned_div
        trt_tbls[1][:, :, div_idx_trt_tbl] = bdnn_binned_div
    # Original bd liks
    orig_birth_lik = get_bdnn_lik(bdnn_obj,
                                  bdnn_time, i_events_sp, i_events_ex, n_S,
                                  post_w_sp_i, t_reg_lam_i, reg_denom_lam_i,
                                  trt_tbls, rate_type='l')
    orig_death_lik = get_bdnn_lik(bdnn_obj,
                                  bdnn_time, i_events_sp, i_events_ex, n_S,
                                  post_w_ex_i, t_reg_mu_i, reg_denom_mu_i,
                                  trt_tbls, rate_type='m')
    sp_lik_j = np.zeros((n_perm, n_perm_traits))
    ex_lik_j = np.zeros((n_perm, n_perm_traits))
    rngint = np.random.default_rng()
    seeds = rngint.integers(low=0, high=1e10, size=n_features)
#    print(perm_feature_idx)
    for j in range(n_perm_traits):
        perm_feature_idx_j = perm_feature_idx[j]
        for k in range(n_perm):
            trt_tbls_perm = copy_lib.deepcopy(trt_tbls)
            trt_tbls_perm_highres = copy_lib.deepcopy(trt_tbls_highres)
            for l in range(len(perm_feature_idx_j)):
                feat_idx = perm_feature_idx_j[l]
                if feat_idx is not None:
                    seed = seeds[feat_idx] + k
                    if feat_idx.size > 1:
                        seed = seed[0]
                    trt_tbls_perm, use_high_res = permute_trt_tbl(feat_idx, feature_is_time_variable, post_ts_i, trt_tbls_perm, trt_tbls_perm_highres, seed=seed)
            if use_high_res:
                # Use high temporal resolution (obtained with set_temporal_resolution) for time-variable features but not for traits.
                # This makes the calculation faster.
                sp_lik_j[k, j] = get_bdnn_lik(bdnn_obj,
                                              bdnn_time_highres, i_events_sp_highres, i_events_ex_highres, n_S_highres,
                                              post_w_sp_i, t_reg_lam_i, reg_denom_lam_i,
                                              trt_tbls_perm, rate_type='l')
                ex_lik_j[k, j] = get_bdnn_lik(bdnn_obj,
                                              bdnn_time_highres, i_events_sp_highres, i_events_ex_highres, n_S_highres,
                                              post_w_ex_i, t_reg_mu_i, reg_denom_mu_i,
                                              trt_tbls_perm, rate_type='m')
            else:
                sp_lik_j[k, j] = get_bdnn_lik(bdnn_obj,
                                              bdnn_time, i_events_sp, i_events_ex, n_S,
                                              post_w_sp_i, t_reg_lam_i, reg_denom_lam_i,
                                              trt_tbls_perm, rate_type='l')
                ex_lik_j[k, j] = get_bdnn_lik(bdnn_obj,
                                              bdnn_time, i_events_sp, i_events_ex, n_S,
                                              post_w_ex_i, t_reg_mu_i, reg_denom_mu_i,
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


def remove_invariant_feature_from_featperm_results(bdnn_obj, res, trt_tbl, combine_discr_features = "", rate_type='speciation'):
    features_without_variance = get_idx_feature_without_variance(trt_tbl)
    names_features = get_names_features(bdnn_obj, rate_type=rate_type)
    idx_comb_feat = get_idx_comb_feat(names_features, combine_discr_features)
    if idx_comb_feat:
        names_features = replace_names_by_feature_group(names_features, idx_comb_feat, combine_discr_features)
    for i in features_without_variance:
        res = res[res['feature1'] != names_features[i]]
        res = res[res['feature2'] != names_features[i]]
    return res


def set_temporal_resolution(bdnn_obj, min_bs, rate_type='speciation', ts=None):
    trt_tbls = copy_lib.deepcopy(bdnn_obj.trait_tbls)
    fixed_shifts = None
    n_bins_highres = 1.0
    if rate_type != 'sampling':
        if trt_tbls[0].ndim == 3:
            fixed_shifts = copy_lib.deepcopy(bdnn_obj.bdnn_settings['fixed_times_of_shift_bdnn'])
            fixed_shifts2 = get_bdnn_time(bdnn_obj, ts)[::-1]
            bin_size = np.diff(fixed_shifts2)
            if ~np.all(bin_size[:-1] == np.mean(bin_size[:-1])) and min_bs < 0.0:
                new_bs = np.min(bin_size)
                print(("\nDifferent bin sizes detected due to using -fixShift.\nTime windows resampled to a resolution of %s." ) % new_bs)
                print("Window size can be set with -BDNN_pred_importance_window_size")
            else:
                new_bs = min_bs
            if new_bs > 0.0:
#                bin_size = np.concatenate((bin_size, bin_size[-1]), axis = None) # Should be okay
                n_bins_lowres = trt_tbls[0].shape[0]
                trt_tbl_lam = trt_tbls[0][::-1]
                trt_tbl_mu = trt_tbls[1][::-1]
                n_bins_highres = np.floor(bin_size / new_bs).astype(int)
                if np.all(n_bins_highres == 0):
                    sys.exit("\nError: Decreasing temporal resolution instead of increasing\n")
                n_bins_highres[n_bins_highres == 0] = 1
                bin_idx_lowres = np.repeat(np.arange(n_bins_lowres), repeats = n_bins_highres)
                trt_tbl_lam_highres = trt_tbl_lam[bin_idx_lowres, :, :]
                trt_tbl_mu_highres = trt_tbl_mu[bin_idx_lowres, :, :]
                trt_tbls[0] = trt_tbl_lam_highres[::-1]
                trt_tbls[1] = trt_tbl_mu_highres[::-1]
                fixed_shifts = np.zeros(trt_tbls[0].shape[0] - 1)
                # Maybe we need to remove the oldest age of fixed_shifts2
                for i in range(len(bin_size)):
                    if n_bins_highres[i] == 1:
                         if i < len(bin_size):
                             add_shifts = fixed_shifts2[i + 1]
                         else:
                             add_shifts = fixed_shifts2[i]
                         fixed_shifts[np.sum(n_bins_highres[:i])] = add_shifts
                    else:
                         add_shifts = (fixed_shifts2[i] + new_bs) + np.linspace(0.0, bin_size[i] - new_bs, n_bins_highres[i])
                         idx = np.arange(np.sum(n_bins_highres[:i]), np.sum(n_bins_highres[:(i + 1)]), 1, dtype = int)
                         if i == (len(bin_size) - 1):
                             idx = idx[:-1]
                             add_shifts = add_shifts[:-1]
                         fixed_shifts[idx] = add_shifts
                fixed_shifts = fixed_shifts[::-1]
    else:
        if trt_tbls[2].ndim == 3:
            fixed_shifts = copy_lib.deepcopy(bdnn_obj.bdnn_settings['q_time_frames'])[::-1]
            fixed_shifts2 = np.copy(fixed_shifts)
            bin_size = np.diff(fixed_shifts2)
            if ~np.all(bin_size == np.mean(bin_size)) and min_bs < 0.0:
                new_bs = np.min(bin_size)
                print(("\nDifferent bin sizes detected due to using -qShift.\nTime windows resampled to a resolution of %s." ) % new_bs)
                print("Window size can be set with -BDNN_pred_importance_window_size")
            else:
                new_bs = min_bs
            if new_bs > 0.0:
                n_bins_lowres = trt_tbls[2].shape[0]
                trt_tbl_q = trt_tbls[2][::-1]
                n_bins_highres = np.floor(bin_size / new_bs).astype(int)
                if np.all(n_bins_highres == 0):
                    sys.exit("\nError: Decreasing temporal resolution instead of increasing\n")
                n_bins_highres[n_bins_highres == 0] = 1
                bin_idx_lowres = np.repeat(np.arange(n_bins_lowres), repeats = n_bins_highres)
                trt_tbl_q_highres = trt_tbl_q[bin_idx_lowres, :, :]
                trt_tbls[2] = trt_tbl_q_highres[::-1]
                fixed_shifts = np.zeros(trt_tbls[2].shape[0] - 1)
                for i in range(len(bin_size)):
                    if n_bins_highres[i] == 1:
                         if i < len(bin_size):
                             add_shifts = fixed_shifts2[i + 1]
                         else:
                             add_shifts = fixed_shifts2[i]
                         add_shift_idx = np.sum(n_bins_highres[:i])
                         if add_shift_idx < len(fixed_shifts):
                            fixed_shifts[add_shift_idx] = add_shifts
                    else:
                         add_shifts = (fixed_shifts2[i] + new_bs) + np.linspace(0.0, bin_size[i] - new_bs, n_bins_highres[i])
                         idx = np.arange(np.sum(n_bins_highres[:i]), np.sum(n_bins_highres[:(i + 1)]), 1, dtype = int)
                         if i == (len(bin_size) - 1):
                             idx = idx[:-1]
                             add_shifts = add_shifts[:-1]
                         fixed_shifts[idx] = add_shifts
                fixed_shifts = np.concatenate((np.max(fixed_shifts2), fixed_shifts[::-1], np.zeros(1)), axis=None)
                n_bins_highres = n_bins_highres[::-1]
                fixed_shifts = fixed_shifts[::-1]
    return trt_tbls, fixed_shifts, n_bins_highres


def trim_trt_tbls_to_match_event(trt_tbls, events):
    ncol_events = events.shape[1]
    nbins_trt_tbl = trt_tbls[0].shape[0]
    trim = nbins_trt_tbl - ncol_events
    trimmed_trt_tbls = []
    trimmed_trt_tbls.append(trt_tbls[0][trim:, :, :])
    trimmed_trt_tbls.append(trt_tbls[1][trim:, :, :])
    return trimmed_trt_tbls


def feature_permutation(mcmc_file, pkl_file, burnin, thin, min_bs, n_perm = 10, num_processes = 1, combine_discr_features = "", show_progressbar = False, do_inter_imp = True):
    bdnn_obj, post_w_sp, post_w_ex, _, sp_fad_lad, post_ts, post_te, post_t_reg_lam, post_t_reg_mu, _, post_reg_denom_lam, post_reg_denom_mu, _, _ = bdnn_parse_results(mcmc_file, pkl_file, burnin, thin)
    n_mcmc, n_taxa = post_ts.shape
    fixed_times_of_shift = copy_lib.deepcopy(bdnn_obj.bdnn_settings['fixed_times_of_shift_bdnn']) # We need this for the diversity dependence
    trt_tbls = bdnn_obj.trait_tbls[:2]
#    trt_tbls_highres, bdnn_obj.bdnn_settings['fixed_times_of_shift_bdnn'], _ = set_temporal_resolution(bdnn_obj, min_bs) # We need the fixed_times setting for the diversity-dependence
    trt_tbls_highres, fixed_times_of_shift_highres, _ = set_temporal_resolution(bdnn_obj, min_bs, ts=post_ts)
    n_features = trt_tbls[0].shape[-1]
    names_features_sp = get_names_features(bdnn_obj, rate_type='speciation')
    names_features_ex = copy_lib.deepcopy(names_features_sp)
    bdnn_dd = 'diversity' in names_features_sp
    div_idx_trt_tbl = -1
    if is_time_trait(bdnn_obj) and bdnn_dd:
            div_idx_trt_tbl = -2
    if bdnn_dd:
        trt_tbls_highres[0][0, :, div_idx_trt_tbl] = 1.0
        trt_tbls_highres[1][0, :, div_idx_trt_tbl] = 1.0
        trt_tbls[0][0, :, div_idx_trt_tbl] = 1.0
        trt_tbls[1][0, :, div_idx_trt_tbl] = 1.0
    sp_feature_is_time_variable = is_time_variable_feature(trt_tbls[0])
    ex_feature_is_time_variable = is_time_variable_feature(trt_tbls[1])
    feature_is_time_variable = (sp_feature_is_time_variable + ex_feature_is_time_variable) > 0
    perm_traits, perm_feature_idx = create_perm_comb(bdnn_obj, do_inter_imp, combine_discr_features)
    n_perm_traits = len(perm_traits)
    args = []
    bdnn_time = np.array([np.max(post_ts), 0.0])
    bdnn_time_highres = None
    i_events_sp_highres, i_events_ex_highres, n_S_highres = None, None, None
    trt_tbls_highres_trimmed = None
    for i in range(n_mcmc):
        if trt_tbls[0].ndim == 3:
            bdnn_time = get_bdnn_time(fixed_times_of_shift, post_ts[i, :])
            bdnn_time_highres = get_bdnn_time(fixed_times_of_shift_highres, post_ts[i, :])
            bin_size = np.tile(np.abs(np.diff(bdnn_time_highres)), n_taxa).reshape((n_taxa, len(bdnn_time_highres) - 1))
            i_events_sp_highres, i_events_ex_highres, n_S_highres = get_events_ns(post_ts[i, :], post_te[i, :], bdnn_time_highres, bin_size)
            trt_tbls_highres_trimmed = trim_trt_tbls_to_match_event(trt_tbls_highres, i_events_sp_highres)
        bin_size = np.tile(np.abs(np.diff(bdnn_time)), n_taxa).reshape((n_taxa, len(bdnn_time) - 1))
        i_events_sp, i_events_ex, n_S = get_events_ns(post_ts[i, :], post_te[i, :], bdnn_time, bin_size)
        a = [bdnn_obj, post_ts[i, :], post_te[i, :],
             bdnn_time, i_events_sp, i_events_ex, n_S,
             bdnn_time_highres, i_events_sp_highres, i_events_ex_highres, n_S_highres,
             post_w_sp[i], post_w_ex[i], post_t_reg_lam[i], post_t_reg_mu[i], post_reg_denom_lam[i], post_reg_denom_mu[i],
             trt_tbls, trt_tbls_highres_trimmed,
             n_perm, n_perm_traits, n_features, feature_is_time_variable, bdnn_dd, div_idx_trt_tbl, perm_feature_idx]
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


def get_q_rate_BDNN_noreg(x, w, hidden_act_f, out_act_f, t_reg, reg_denom, q, n, singleton_mask, qbin_ts_te=None):
    r = get_rate_BDNN_3D_noreg(x, w, hidden_act_f, out_act_f, True, singleton_mask, qbin_ts_te)
    r = (r * n) ** t_reg / reg_denom
    if len(q) > 1:
        # Several q rates
        if r.ndim == 1:
            # No time-varying trait table
            r = r[:, np.newaxis] * q
        else:
            # Time-varingy trait table
            r = r * q[np.newaxis, :]
    else:
        r = r * q
    return r


def perm_mcmc_sample_q_i(arg):
    if len(arg) == 22:
        use_HPP_NN_lik = True
        [q, n, w_q_i, t_reg_q_i, reg_denom_q_i, trt_tbl, hidden_act_f, out_act_f, n_perm, n_perm_traits, n_features, feature_is_time_variable, perm_feature_idx, ts_i, te_i, occs_sp, log_factorial_occs, sm, qb_se, q_time_frames, duration_q_bins, occs_single_bin] = arg
    else:
        use_HPP_NN_lik = False
        [q, n, w_q_i, t_reg_q_i, reg_denom_q_i, trt_tbl, hidden_act_f, out_act_f, n_perm, n_perm_traits, n_features, feature_is_time_variable, perm_feature_idx, ts_i, te_i, occs_sp, log_factorial_occs, sm, qb_se] = arg

    bdnn_q_rates = get_q_rate_BDNN_noreg(trt_tbl, w_q_i, hidden_act_f, out_act_f, t_reg_q_i, reg_denom_q_i, q, n, sm, qb_se)
    if use_HPP_NN_lik:
        orig_fossil_lik = HPP_NN_lik([ts_i, te_i, bdnn_q_rates, occs_sp, log_factorial_occs, q_time_frames, duration_q_bins, occs_single_bin])
    else:
        orig_fossil_lik = HOMPP_NN_lik([ts_i, te_i, bdnn_q_rates, occs_sp, log_factorial_occs])
    orig_fossil_lik = np.sum(orig_fossil_lik)

    q_lik_j = np.zeros((n_perm, n_perm_traits))
    rngint = np.random.default_rng()
    seeds = rngint.integers(low=0, high=1e10, size=n_features)
    trt_tbls = [trt_tbl, trt_tbl]
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
                    trt_tbls_perm, _ = permute_trt_tbl(feat_idx, feature_is_time_variable, ts_i, trt_tbl_lowres=trt_tbls_perm, trt_tbl=trt_tbls_perm, seed=seed)
            bdnn_q_rates = get_q_rate_BDNN_noreg(trt_tbls_perm[0], w_q_i, hidden_act_f, out_act_f, t_reg_q_i, reg_denom_q_i, q, n, sm, qb_se)
            if use_HPP_NN_lik:
                perm_fossil_lik = HPP_NN_lik([ts_i, te_i, bdnn_q_rates, occs_sp, log_factorial_occs, q_time_frames, duration_q_bins, occs_single_bin])
            else:
                perm_fossil_lik = HOMPP_NN_lik([ts_i, te_i, bdnn_q_rates, occs_sp, log_factorial_occs])
            q_lik_j[k, j] = np.sum(perm_fossil_lik)
    q_delta_lik = q_lik_j - orig_fossil_lik
    return q_delta_lik


def feature_permutation_sampling(mcmc_file, pkl_file, burnin, thin, min_bs, n_perm=10, num_processes=1, combine_discr_features="", show_progressbar=False, do_inter_imp=True):
    bdnn_obj, _, _, w_q, _, ts, te, _, _, t_reg_q, reg_denom_lam, _, reg_denom_q, norm_q = bdnn_parse_results(mcmc_file, pkl_file, burnin, thin)
    n_mcmc = ts.shape[0]
    trt_tbl = get_trt_tbl(bdnn_obj, rate_type='sampling')
    feature_is_time_variable = is_time_variable_feature(trt_tbl)
    occs_sp = np.copy(bdnn_obj.bdnn_settings['occs_sp'])
    log_factorial_occs = np.copy(bdnn_obj.bdnn_settings['log_factorial_occs'])
    q = get_baseline_q(mcmc_file, burnin, thin, mean_across_shifts=False)
    age_dependent_sampling = 'highres_q_repeats' in bdnn_obj.bdnn_settings.keys()
    if 'q_time_frames' in bdnn_obj.bdnn_settings.keys():
        duration_q_bins = np.copy(bdnn_obj.bdnn_settings['duration_q_bins'])
        occs_single_bin = np.copy(bdnn_obj.bdnn_settings['occs_single_bin'])
        q_bins = np.copy(bdnn_obj.bdnn_settings['q_time_frames'])
        if np.any(feature_is_time_variable) or age_dependent_sampling:
            # Time-variable feature
            if age_dependent_sampling:
                # No upsampling with set_temporal_resolution() if this has been already done during model inference
                min_bs = 0.0
            trt_tbls, q_bins, num_repeats_q = set_temporal_resolution(bdnn_obj, min_bs, rate_type='sampling')
            trt_tbl = trt_tbls[2]
            if q.shape[1] > 1 and not age_dependent_sampling:
                q_idx_lowres = np.repeat(np.arange(q.shape[1]), repeats = num_repeats_q)
                q = q[:, q_idx_lowres]
            if age_dependent_sampling:
                q = q[:, bdnn_obj.bdnn_settings['highres_q_repeats'].astype(int)]
            # get occs_sp, log_factorial_occs, occs_single_bin, and duration_q_bins for the new bins
            occs_sp = get_occs_sp(bdnn_obj.occ_data, q_bins[::-1])
            log_factorial_occs, duration_q_bins, occs_single_bin = get_fossil_features_q_shifts(bdnn_obj.occ_data, q_bins, occs_sp, te[0, :])
            duration_q_bins = duration_q_bins[:, ::-1]
            q_bins = q_bins[::-1]
    n_features = trt_tbl.shape[-1]
    names_features = get_names_features(bdnn_obj, rate_type='sampling')
    perm_traits, perm_feature_idx = create_perm_comb(bdnn_obj, do_inter_imp, combine_discr_features, rate_type='sampling')
    n_perm_traits = len(perm_traits)
    out_act_f = bdnn_obj.bdnn_settings['out_act_f_q']
    
    sm_mask = ''
    if np.any(feature_is_time_variable) or age_dependent_sampling:
        sm_mask = 'make_3D'
    singleton_mask = make_singleton_mask(occs_sp, sm_mask)
    
    args = []
    for i in range(n_mcmc):
        constant_baseline_q = q.shape[1] == 1
        not_na = ~np.isnan(q[i, :]) # remove empty q bins resulting from combing replicates with different number of bins
        q_i = q[i, not_na]
        trt_tbl_a = np.copy(trt_tbl)
        if "taxon_age" in names_features:
            trt_tbl_a = add_taxon_age(ts[i, :], te[i, :], q_bins, trt_tbl_a)
        occ_sp_for_a = occs_sp
        if not constant_baseline_q:
            occ_sp_for_a = occs_sp[:, not_na]

        qbins_ts_te = None
        if trt_tbl_a.ndim == 3:
            qbins_ts_te = get_qbin_ts_te(ts[i, :], te[i, :], q_bins)

        a = [q_i, norm_q[i], w_q[i], t_reg_q[i], reg_denom_q[i], trt_tbl_a,
             bdnn_obj.bdnn_settings['hidden_act_f'], out_act_f,
             n_perm, n_perm_traits, n_features, feature_is_time_variable, perm_feature_idx,
             ts[i, :], te[i, :], occ_sp_for_a, log_factorial_occs, singleton_mask, qbins_ts_te]
        if 'q_time_frames' in bdnn_obj.bdnn_settings.keys():
            if constant_baseline_q:
                a += [q_bins, duration_q_bins, occs_single_bin]
            else:
                not_na_q_bins = np.concatenate((not_na[::-1], np.array([True])), axis=None)
                a += [q_bins[not_na_q_bins], duration_q_bins[:, not_na[::-1]], occs_single_bin]
        args.append(a)
    unixos = is_unix()
    if unixos and num_processes > 1:
        pool_perm = multiprocessing.Pool(num_processes)
        delta_lik = list(tqdm(pool_perm.imap_unordered(perm_mcmc_sample_q_i, args),
                              total = n_mcmc, disable = show_progressbar == False))
        pool_perm.close()
    else:
        delta_lik = []
        for i in tqdm(range(n_mcmc), disable = show_progressbar == False):
            delta_lik.append(perm_mcmc_sample_q_i(args[i]))
    delta_lik = np.vstack(delta_lik)
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
        delta_lik[:, j] = (delta_lik[:, idx_main1] + delta_lik[:, idx_main2]) - delta_lik[:, j]
    names_df = pd.DataFrame(perm_traits, columns = ['feature1', 'feature2'])
    delta_lik_summary = get_rates_summary(delta_lik.T)
    delta_lik_df = pd.DataFrame(delta_lik_summary, columns = ['delta_lik', 'lwr_delta_lik', 'upr_delta_lik'])
    delta_lik_df = pd.concat([names_df, delta_lik_df], axis = 1)
    delta_lik_df = remove_invariant_feature_from_featperm_results(bdnn_obj, delta_lik_df, trt_tbl_a, combine_discr_features, 'sampling')
    return delta_lik_df


# Fastshap
##########
# forked from https://github.com/AnotherSamWilson/fastshap/tree/master/fastshap
# change from arithmetic mean for baseline (i.e. background_preds.mean(0) and data_preds.mean(0)) to harmonic mean
def _assign_pd(table, location, values):
        table.iloc[location] = values


def _view_pd(table, location):
    return table.iloc[location]


def _concat_pd(pd_list, axis):
    return concat(pd_list, axis=axis)


def _assign_np(array, location, values):
    array[location] = values


def _view_np(array, location):
    return array[location]


def _concat_np(np_list, axis):
    if axis == 0:
        return np.vstack(np_list)
    else:
        return np.hstack(np_list)


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, pd_DataFrame):
        return x.to_numpy()
    elif isinstance(x, pd_Series):
        if x.dtype.name == "category":
            return x.cat.codes.to_numpy()
        elif x.dtype.name == "object":
            return x.fillna("").to_numpy()
        else:
            return x.to_numpy()
    else:
        raise ValueError("Unknown datatype")


def _repeat(x, repeats, axis=0):
    if isinstance(x, pd_DataFrame) or isinstance(x, pd_Series):
        newind = np.arange(x.shape[0]).repeat(repeats)
        return x.iloc[newind].reset_index(drop=True)
    else:
        return np.repeat(x, repeats, axis=axis)


def _tile(x, reps):
    if isinstance(x, pd_DataFrame) or isinstance(x, pd_Series):
        new_ind = np.tile(np.arange(x.shape[0]), reps[0])
        return x.iloc[new_ind].reset_index(drop=True)
    else:
        return np.tile(x, reps)

def consecutive_slices(data):
    runs = np.split(data, np.where(np.diff(data) != 1)[0] + 1)
    return [slice(min(run), max(run) + 1) for run in runs]


def stratified_continuous_folds(y, nfold):
    """
    Create primitive stratified folds for continuous data.
    """
    elements = len(y)
    assert elements >= nfold, "more splits then elements."
    sorted = np.argsort(y)
    val = [sorted[range(i, len(y), nfold)] for i in range(nfold)]
    return val


def _get_variable_name_index(variable, data):
    """
    Returns the variable name as a string, and the variable index as an int,
    whether a string or int is passed to a pd.DataFrame or numpy.ndarray

    :param variable: str, int
        The variable we want to return the name and index for.
    :param data:
        The data in which the variable resides.

    :return:
        variable_name (str), variable_index (int)
    """
    if isinstance(data, pd_DataFrame):
        if isinstance(variable, str):
            variable_name = variable
            variable_index = data.columns.tolist().index(variable)
        else:
            variable_name = str(data.columns.tolist()[variable])
            variable_index = variable

    elif isinstance(data, np.ndarray):
        assert isinstance(
            variable, int
        ), "data was numpy array, variable must be an integer"
        variable_name = str(variable)
        variable_index = variable
    else:
        raise ValueError("data not recognized. Must be numpy array or pd.DataFrame")

    return variable_name, variable_index


def _safe_isnan(x):
    if isinstance(x, pd_DataFrame) or isinstance(x, pd_Series):
        return x.isnull().values
    elif isinstance(x, np.ndarray):
        return np.isnan(x)
    else:
        raise ValueError("x not recognized")


def _ensure_2d_array(a):
    if a.ndim == 1:
        return a.reshape(-1, 1)
    else:
        return a


def _fill_missing_cat(x, s):
    assert isinstance(
        x, pd_Series
    ), "Can only fill cat on pandas object or categorical series"
    x_nan = x.isnull()
    if x_nan.sum() > 0:
        if x.dtype.name == "category":
            x = x.cat.add_categories(s).fillna(s)
        elif x.dtype.name == "object":
            x = x.fillna(s)
        else:
            raise ValueError("Series datatype must be object or category.")
    return x


def _keep_top_n_cats_unique(x, n, s, m, codes=False):
    """
    Groups least popular categories together.
    Can return the category codes.

    :param x: pd.Series
        The series to be grouped
    :param n: int
        The number of categories to leave unique (including nans)
    :param s:
        The value to impute non-popular categories as
    :param m:
        The value to impute missing values as.
    :return:
    """

    return_type = "category" if codes else x.dtype.name
    x = _fill_missing_cat(x, m)
    c = x.value_counts().sort_values(ascending=False).index.to_numpy()[:n]
    d = pd_Series(np.where(x.isin(c), x, s), dtype=return_type)
    if codes:
        d = d.cat.codes
    return d


def ampute_data(
    data,
    variables=None,
    perc=0.1,
    random_seed=None,
):
    """
    Ampute Data

    Returns a copy of data with specified variables amputed.

    Parameters
    ----------
     data : Pandas DataFrame
        The data to ampute
     variables : None or list
        If None, are variables are amputed.
     perc : double
        The percentage of the data to ampute.
    random_state: None, int, or np.random.RandomState

    Returns
    -------
    pandas DataFrame
        The amputed data
    """
    amputed_data = data.copy()
    data_shape = amputed_data.shape
    amp_rows = int(perc * data_shape[0])
    random_state = np.random.RandomState(random_seed)
    if len(data_shape) > 1:
        if variables is None:
            variables = [i for i in range(amputed_data.shape[1])]
        elif isinstance(variables, list):
            if isinstance(variables[0], str):
                variables = [data.columns.tolist().index(i) for i in variables]
        if isinstance(amputed_data, pd_DataFrame):
            for v in variables:
                na_ind = random_state.choice(
                    np.arange(data_shape[0]), replace=False, size=amp_rows
                )
                amputed_data.iloc[na_ind, v] = np.NaN
        if isinstance(amputed_data, np.ndarray):
            amputed_data = amputed_data.astype("float64")
            for v in variables:
                na_ind = random_state.choice(
                    np.arange(data_shape[0]), replace=False, size=amp_rows
                )
                amputed_data[na_ind, v] = np.NaN
    else:
        na_ind = random_state.choice(
            np.arange(data_shape[0]), replace=False, size=amp_rows
        )
        amputed_data[na_ind] = np.NaN
    return amputed_data


class Logger:
    def __init__(self, verbose):
        self.verbose = verbose

    def log(self, message):
        if self.verbose:
            print(message)


class KernelExplainer:
    def __init__(self, model, background_data):
        """
        The KernelExplainer is capable of calculating shap values for any arbitrary function.

        Parameters
        ----------

        model: Callable.
            Some function which takes the background_data and return
            a numpy array of shape (n, ).

        background_data: pandas.DataFrame or np.ndarray
            The background set which will be used to act as the "missing"
            data when calculating shap values. Smaller background sets
            will make the process fun faster, but may cause shap values
            to drift from their "true" values.

            It is possible to stratify this background set by running
            .stratify_background_set()

        """
        self.model = model
        self.background_data = [background_data]
        self.num_columns = background_data.shape[1]
        self.n_splits = 0
        background_preds = model(background_data)
        assert isinstance(background_preds, np.ndarray)
        assert background_preds.ndim <= 2, "Maximum 2 dimensional outputs supported"
        self.output_dim = 1 if background_preds.ndim == 1 else background_preds.shape[1]
        self.return_type = background_preds.dtype

        if isinstance(background_data, pd_DataFrame):
            self.col_names = background_data.columns.tolist()
            self.dtypes = {col: background_data[col].dtype for col in self.col_names}

            #from .compat import _assign_pd, _view_pd, _concat_pd

            self._assign = _assign_pd
            self._view = _view_pd
            self._concat = _concat_pd

        if isinstance(background_data, np.ndarray):
            #from .compat import _assign_np, _view_np, _concat_np

            self._assign = _assign_np
            self._view = _view_np
            self._concat = _concat_np

    def calculate_shap_values(
        self,
        data,
        outer_batch_size=None,
        inner_batch_size=None,
        n_coalition_sizes=3,
        background_fold_to_use=None,
        linear_model=None,
        verbose=False,
    ):
        """
        Calculates approximate shap values for data.


        Parameters
        ----------

        data: pandas.DataFrame or np.ndarray
            The data to calculate the shap values for

        outer_batch_size: int
            Shap values are calculated all at once in the outer batch.
            The outer batch requires the creation of the Linear Targets,
            which is an array of size(`Total Coalitions`, `outer_batch_size`)

            To determine an appropriate outer_batch_size, play around with
            the .get_theoretical_array_expansion_sizes() function.

        inner_batch_size: int
            To get the Linear Targets, an array of the following size must
            be evaluated by the model: (`inner_batch_size`, `# background samples`)
            and then aggregated.

            To determine an appropriate inner_batch_size, play around with
            the .get_theoretical_array_expansion_sizes() function.

        n_coalition_sizes: int
            The coalition sizes, starting at 1, and their complements which will
            be used to calculate the shap values.

            Not all possible column combinations can be evaluated to calculate
            the shap values. The shap kernel puts more weight on lower
            coalition sizes (and their complements). These also tend to have
            fewer possible combinations.

            For example, if our dataset has 10 columns, and we set
            n_coalition_sizes = 3, then the process will calculate the shap
            values by integrating over all column combinations of
            size 1, 9, 2, 8, 3, and 7.

        background_fold_to_use: None or int
            If the background dataset has been stratified, select one of
            them to use to calculate the shap values.

        linear_model: sklearn.linear_model
            The linear model used to obtain the shap values.
            Must have a fit method. Intercept should not be fit.
            See github page for examples.

        verbose: bool
            Should progress be printed?


        Returns
        -------

        If the output is multiple dimensions (multiclass problems):
            Returns a numpy array of shape (# data rows, # columns + 1, output dimension).
            So, for example, if you would like to access the shap values for the second
            class in a multi-class output, you can use the slice shap_values[:,:,1],
            which will return an array of shape (# data rows, # columns + 1). The final
            column is the expected value for that class.

        If the output is a single dimension (binary, regression problems):
            Returns a numpy array of shape (# data rows, # columns + 1). The final
            column is the expected value for that class.


        """

        logger = Logger(verbose)

        if background_fold_to_use is not None:
            assert (
                background_fold_to_use < self.n_splits
            ), f"There are only {self.n_splits} splits of the background dataset"
        else:
            background_fold_to_use = 0

        if linear_model is None:
            linear_model = LinearRegression(fit_intercept=False)
        else:
            assert hasattr(linear_model, "fit")

        working_background_data = self.background_data[background_fold_to_use]
        background_preds = _ensure_2d_array(self.model(working_background_data))
        background_pred_mean = np.array([ 1.0 / np.mean(1.0 / background_preds) ]) #background_preds.mean(0)

        # Do cursory glances at the background and new data
        if isinstance(data, pd_DataFrame):
            assert set(data.columns) == set(self.col_names), "Columns don't match"
        else:
            assert data.shape[1] == self.num_columns, "Different number of columns"

        num_new_samples = data.shape[0]
        col_array = np.arange(self.num_columns)
        n_background_rows = working_background_data.shape[0]
        outer_batch_size, inner_batch_size = self._configure_batch_sizes(
            outer_batch_size=outer_batch_size,
            inner_batch_size=inner_batch_size,
            data=data,
        )
        index = np.arange(num_new_samples)
        outer_batches = [
            index[i : np.min([i + outer_batch_size, index[-1] + 1])]
            for i in range(0, num_new_samples, outer_batch_size)
        ]

        data_preds = _ensure_2d_array(self.model(data))
        shap_values = np.empty(
            shape=(data.shape[0], data.shape[1] + 1, self.output_dim)
        ).astype(
            self.return_type
        )  # +1 for expected value

        # Determine how many coalition sizes in the symmetric kernel are paired.
        # There may be one unpaired weight if the number of columns is even.
        # This is because we calculate k and it's complement for each coalition size
        # i.e. If we have 10 columns, when we evaluate the size 1 subsets, we also evaluate
        # the complement, which consists of all the size 9 subsets. However, we shouldn't
        # evaluate the complement of the size 5 subsets, because we would double count.
        n_choose_k_midpoint = (self.num_columns - 1) / 2.0
        coalition_sizes_to_combinate = np.min(
            [np.ceil(n_choose_k_midpoint), n_coalition_sizes]
        ).astype("int32")
        symmetric_sizes_to_combinate = np.min(
            [np.floor(n_choose_k_midpoint), n_coalition_sizes]
        ).astype("int32")
        coalition_pared_ind = [
            (cs in range(symmetric_sizes_to_combinate))
            for cs in range(coalition_sizes_to_combinate)
        ]

        # Number of coalition combinations (excluding complement) per coalition size.
        coalitions_per_coalition_size = [
            binom(self.num_columns, cs).astype("int32")
            for cs in range(1, coalition_sizes_to_combinate + 1)
        ]
        # Number of coalition combinations (including complement) per coalition size.
        total_coalitions_per_coalition_size = [
            coalitions_per_coalition_size[i] * (2 if coalition_pared_ind[i] else 1)
            for i in range(coalition_sizes_to_combinate)
        ]
        cc_cs = np.cumsum(coalitions_per_coalition_size)
        num_total_coalitions_to_run = np.sum(total_coalitions_per_coalition_size)
        logger.log(
            f"Number of coalitions to run per sample: {str(num_total_coalitions_to_run)}"
        )

        # Theoretical weights if we use all possible coalition sizes (before scaling)
        coalition_size_weights = np.array(
            [
                (self.num_columns - 1.0) / (i * (self.num_columns - i))
                for i in range(1, self.num_columns)
            ]
        )
        # Weights are symmetric, so we can
        selected_coalition_size_weights = np.concatenate(
            [
                coalition_size_weights[:coalition_sizes_to_combinate],
                coalition_size_weights[-symmetric_sizes_to_combinate:],
            ]
        )
        selected_coalition_size_weights /= selected_coalition_size_weights.sum()

        for outer_batch in outer_batches:
            # outer_batch = outer_batches[0]
            logger.log(f"Starting Samples {outer_batch[0]} - {outer_batch[-1]}")
            outer_batch_length = len(outer_batch)
            masked_coalition_avg = np.empty(
                shape=(num_total_coalitions_to_run, outer_batch_length, self.output_dim)
            ).astype(self.return_type)
            mask_matrix = np.zeros(
                shape=(num_total_coalitions_to_run, self.num_columns)
            ).astype("int8")
            coalition_weights = np.empty(num_total_coalitions_to_run)

            inner_batches_relative = [
                slice(i, i + inner_batch_size)
                for i in range(0, outer_batch_length, inner_batch_size)
            ]
            inner_batches_absolute = [
                slice(outer_batch[0] + i, outer_batch[0] + i + inner_batch_size)
                for i in range(0, outer_batch_length, inner_batch_size)
            ]
            inner_batch_count = len(inner_batches_absolute)

            for coalition_size in range(1, coalition_sizes_to_combinate + 1):
                # coalition_size = 1
                has_complement = coalition_size <= symmetric_sizes_to_combinate
                choose_count = binom(self.num_columns, coalition_size).astype("int32")
                model_evals = inner_batch_count * (choose_count * 2 + inner_batch_count)
                logger.log(
                    f"Coalition Size: {str(coalition_size)} - Model Evaluations: {model_evals}"
                )
                inds = combinations(np.arange(self.num_columns), coalition_size)
                listinds = [list(i) for i in inds]
                coalition_weight = (
                    selected_coalition_size_weights[coalition_size - 1] / choose_count
                )

                # Get information about where these coalitions are stored in the arrays
                start = (cc_cs - coalitions_per_coalition_size)[coalition_size - 1]
                end = cc_cs[coalition_size - 1]
                coalition_loc = np.arange(start, end)
                mask_matrix[coalition_loc.reshape(-1, 1), listinds] = 1
                coalition_weights[coalition_loc] = coalition_weight

                if has_complement:
                    end_c = num_total_coalitions_to_run - start
                    start_c = num_total_coalitions_to_run - end
                    coalition_c_loc = np.arange(start_c, end_c)
                    mask_matrix[coalition_c_loc] = 1 - mask_matrix[coalition_loc]
                    coalition_weights[coalition_c_loc] = coalition_weight

                for inner_batch_i in range(inner_batch_count):
                    # Inner loop is where things get expanded
                    # inner_batch_i = 0
                    slice_absolute = inner_batches_absolute[inner_batch_i]
                    slice_relative = inner_batches_relative[inner_batch_i]
                    inner_batch_size = len(
                        range(*slice_relative.indices(masked_coalition_avg.shape[1]))
                    )

                    repeated_batch_data = _repeat(
                        self._view(data, (slice_absolute, slice(None))),
                        repeats=n_background_rows,
                        axis=0,
                    )

                    # For each mask (and complement, if it is paired)
                    for coalition_i in range(choose_count):
                        masked_data = _tile(
                            working_background_data,
                            (inner_batch_size, 1),
                        )

                        if has_complement:
                            masked_data_complement = masked_data.copy()
                        else:
                            masked_data_complement = None

                        mask = listinds[coalition_i]
                        mask_c = np.setdiff1d(col_array, mask)

                        mask_slices = consecutive_slices(mask)
                        mask_c_slices = consecutive_slices(mask_c)

                        # Overwrite masked data with real batch data.
                        # Order of masked_data is mask, background, batch
                        # Broken up into possible slices for faster insertion
                        for ms in mask_slices:
                            self._assign(
                                masked_data,
                                (slice(None), ms),
                                self._view(repeated_batch_data, (slice(None), ms)),
                            )

                        if has_complement:
                            for msc in mask_c_slices:
                                self._assign(
                                    masked_data_complement,
                                    (slice(None), msc),
                                    self._view(repeated_batch_data, (slice(None), msc)),
                                )

                        masked_coalition_avg[
                            coalition_loc[coalition_i], slice_relative
                        ] = (
                            self.model(masked_data)
                            .reshape(
                                inner_batch_size, n_background_rows, self.output_dim
                            )
                            .mean(axis=1)
                        )
                        if has_complement:
                            masked_coalition_avg[
                                coalition_c_loc[coalition_i], slice_relative
                            ] = (
                                self.model(masked_data_complement)
                                .reshape(
                                    inner_batch_size, n_background_rows, self.output_dim
                                )
                                .mean(axis=1)
                            )

                # Clean up inner batch
                del repeated_batch_data
                del masked_data
                if has_complement:
                    del masked_data_complement

            # Back to outer batch
            mean_model_output = np.array([ 1.0 / np.mean(1.0 / data_preds) ])#data_preds.mean(0)
            linear_features = mask_matrix[:, :-1] - mask_matrix[:, -1].reshape(-1, 1)

            for outer_batch_sample in range(outer_batch_length):
                for output_dimension in range(self.output_dim):
                    linear_target = (
                        masked_coalition_avg[:, outer_batch_sample, output_dimension]
                        - mean_model_output[output_dimension]
                        - (
                            mask_matrix[:, -1]
                            * (
                                data_preds[
                                    outer_batch[outer_batch_sample], output_dimension
                                ]
                                - background_pred_mean[output_dimension]
                            )
                        )
                    )
                    linear_model.fit(
                        X=linear_features,
                        sample_weight=coalition_weights,
                        y=linear_target,
                    )
                    shap_values[
                        outer_batch[outer_batch_sample], :-2, output_dimension
                    ] = linear_model.coef_

        shap_values[:, -2, :] = data_preds - (
            shap_values[:, :-2, :].sum(1) + background_pred_mean
        )
        shap_values[:, -1, :] = background_pred_mean

        if self.output_dim == 1:
            shap_values.resize(data.shape[0], data.shape[1] + 1)

        return shap_values

    def _configure_batch_sizes(self, outer_batch_size, inner_batch_size, data):
        n_rows = data.shape[0]
        outer_batch_size = n_rows if outer_batch_size is None else outer_batch_size
        outer_batch_size = n_rows if outer_batch_size > n_rows else outer_batch_size
        inner_batch_size = (
            outer_batch_size if inner_batch_size is None else inner_batch_size
        )
        assert (
            inner_batch_size <= outer_batch_size
        ), "outer batch size < inner batch size"
        return outer_batch_size, inner_batch_size

    def stratify_background_set(self, n_splits=10, output_dim_to_stratify=0):
        """
        Helper function that breaks up the background
        set into folds stratified by the model output
        on the background set. The larger n_splits,
        the smaller each background set is.

        Parameters
        ----------

        n_splits: int
            The number split datasets created. Raise
            this number to calculate shap values
            faster, at the expense of integrating
            over a smaller dataset.

        output_dim_to_stratify: int
            If the model has multiple outputs, which
            one should be used to stratify the
            background set?

        """

        self.background_data = self._concat(self.background_data, axis=0)
        background_preds = _ensure_2d_array(self.model(self.background_data))[
            :, output_dim_to_stratify
        ]
        folds = stratified_continuous_folds(background_preds, n_splits)
        self.background_data = [self._view(self.background_data, f) for f in folds]
        self.n_splits = n_splits



# k-additive Choque SHAP
########################
def main_shap_for_onehot_features(idx_comb_feat, sm, use_mean=False):
    if len(idx_comb_feat) > 0:
        if use_mean:
            sm = np.abs(sm)
        drop = np.array([], dtype = int)
        for i in range(len(idx_comb_feat)):
            if use_mean:
                group_value = np.mean(sm[:, idx_comb_feat[i]], axis = 1)
            else:
                group_value = np.sum(sm[:, idx_comb_feat[i]], axis = 1)
            sm[:, idx_comb_feat[i][0]] = group_value
            drop = np.concatenate((drop, idx_comb_feat[i][1:]))
        sm = np.delete(sm, drop, axis = 1)
    return sm


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


def opt_Xbinary_wrand_allMethods(nEval, nAttr, k_add, coal_shap):
    ''' Return the matrix of coalitions used in Kernel SHAP (X), the transformation matrix used in the proposal (opt_data) '''
    # Select at random, but with probability distributions based on the SHAP weights
    weights_shap = np.zeros((nEval))
    k_add_numb = nParam_kAdd(k_add, nAttr)
    aux = []
    aux2 = np.ones((nAttr,))
    for ii in range(1, nAttr):
        aux = np.append(aux, comb(nAttr, ii) * shapley_kernel(nAttr, ii))
        aux2[ii] = aux2[ii - 1] + comb(nAttr, ii)

    selec_data_aux = np.zeros(nEval)
    p_aux = aux / sum(aux)
    for ii in range(nEval):
        p_aux = aux / np.sum(aux)
        selec_data_aux[ii] = np.random.choice(np.arange(nAttr - 1) + 1, size=1, replace=False, p=p_aux)[0]
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
    opt_data = np.concatenate((vector_shap2game(np.zeros(nAttr), k_add, nAttr, coal_shap).reshape(1, -1), opt_data),
                              axis=0)
    opt_data = np.concatenate((opt_data, vector_shap2game(np.ones(nAttr), k_add, nAttr, coal_shap).reshape(1, -1)),
                              axis=0)
    weights_shap = np.append(10 ** 6, weights_shap)
    weights_shap = np.append(weights_shap, 10 ** 6)

    return X, opt_data, weights_shap


def shapley_kernel(M, s):
    ''' Return the Kernel SHAP weight '''
    if s == 0 or s == M:
        return 100000
    return (M - 1) / (binom(M, s) * s * (M - s))


def get_shap_species_i(i, nEval, trt_tbl, X, cov_par, t_reg, reg_denom, hidden_act_f, out_act_f, explain_matrix, XX_w, baseline, norm):
    n_species, nAttr = trt_tbl.shape
    trt_tbl_aux2 = np.zeros(nEval * n_species * nAttr).reshape(nEval, n_species, nAttr)
    for ll in range(nEval):
        trt_tbl_aux = trt_tbl + 0.0
        idx = np.where(X[ll, 0:-1] == 1)
        trt_tbl_aux[:, idx] = trt_tbl[i, idx]
        trt_tbl_aux2[ll, :, :] = trt_tbl_aux
    trt_tbl_aux = trt_tbl_aux2.reshape(nEval * n_species, nAttr)
    rate_aux = get_rate_BDNN_noreg(trt_tbl_aux, cov_par, hidden_act_f, out_act_f)
    rate_aux = (rate_aux * norm) ** t_reg / reg_denom
    rate_aux = rate_aux * baseline
    rate_aux = rate_aux.reshape(nEval, n_species)
    exp_payoffs_ci = 1.0 / np.mean(1.0 / rate_aux, axis = 1)
    exp_payoffs_shap = exp_payoffs_ci + 0.0
    exp_payoffs_ci = exp_payoffs_ci - exp_payoffs_ci[0]
    # For weighted random samples
    inter_val = explain_matrix @ exp_payoffs_ci
    shapley_ci = inter_val[1:]
    shapley_val_ci_shap = XX_w @ exp_payoffs_shap
    # Interaction indices
    count = 0
    indices = np.zeros((nAttr, nAttr))
    for ii in range(nAttr - 1):
        for jj in range(ii + 1, nAttr):
            indices[ii, jj] = shapley_ci[nAttr + count]
            count += 1
    indices = indices + indices.T
    return shapley_val_ci_shap, indices


def k_add_kernel_explainer(trt_tbl, cov_par, t_reg, reg_denom, hidden_act_f, out_act_f, baseline=1.0, norm=1.0):
    n_species, nAttr = trt_tbl.shape  # Number of instances and attributes
    k_add = 3
    k_add_not_ok = True
    while k_add_not_ok:
        coal_shap = coalition_shap_kadd(k_add, nAttr)
        nEval_old = coal_shap.shape[0]
        for ii in range(1, 6):
            try:
                nEval = ii * coal_shap.shape[0]
                # " By selecting weighted random samples (without replacement) "
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category = RuntimeWarning)
                    X, opt_data, weights_shap = opt_Xbinary_wrand_allMethods(nEval - 2, nAttr, k_add, coal_shap)
            except:
                nEval = nEval_old
        weights = np.eye(nEval)
        weights[0, 0], weights[-1, -1] = 10 ** 6, 10 ** 6
        try:
            # Pre-computing
            m = opt_data.T @ weights
            explain_matrix = np.linalg.inv(m @ opt_data) @ m
            X_w = X.T @ np.diag(weights_shap)  # This can be pre-computed once for all species
            XX_w = np.linalg.inv(X_w @ X) @ X_w
            _, _ = get_shap_species_i(0, nEval, trt_tbl, X, cov_par, t_reg, reg_denom, hidden_act_f, out_act_f, explain_matrix, XX_w, baseline, norm)
            k_add_not_ok = False
        except:
            k_add_not_ok = True
            k_add += 1
    weights = np.eye(nEval)
    weights[0, 0], weights[-1, -1] = 10 ** 6, 10 ** 6
    shap_main = np.zeros((n_species, nAttr + 1))
    shap_inter = np.zeros((n_species, nAttr, nAttr))
    for i in range(n_species):
        # For all samples
        shapley_val_ci_shap, indices = get_shap_species_i(i, nEval, trt_tbl, X, cov_par, t_reg, reg_denom, hidden_act_f, out_act_f, explain_matrix, XX_w, baseline, norm)
        shap_main[i, :] = shapley_val_ci_shap
        shap_inter[i, :, :] = indices
    return shap_main, shap_inter


def fastshap_kernel_explainer(trt_tbl, cov_par, t_reg, reg_denom, hidden_act_f, out_act_f, baseline=1.0, norm=1.0):
    ke = KernelExplainer(
        model = lambda X: ((get_rate_BDNN_noreg(X, cov_par, hidden_act_f, out_act_f) * norm) ** t_reg / reg_denom) * baseline,
        background_data = trt_tbl
    )
#    shap_main = ke.calculate_shap_values(trt_tbl, verbose = False)
    strata = np.ceil(trt_tbl.shape[0] / 100.0).astype(int)
    ke.stratify_background_set(strata)
    shap_main = ke.calculate_shap_values(trt_tbl, verbose = False, background_fold_to_use = 0)
    return shap_main


def inter_shap_for_onehot_features(idx_comb_feat, si, use_mean=False):
    if len(idx_comb_feat) > 0:
        if use_mean:
            si = np.abs(si)
        conc_comb_feat = np.concatenate(idx_comb_feat)
        n = si.shape[1]
        J = np.arange(n)
        J = np.delete(J, conc_comb_feat)
        # Cases of no interaction within another one-hot encoded feature
        if len(J) > 0:
            for i in range(len(idx_comb_feat)):
                for j in J:
                    if use_mean:
                        group_value = np.mean(si[:, :, idx_comb_feat[i]][:, j, :], axis = 1)
                    else:
                        group_value = np.sum(si[:, :, idx_comb_feat[i]][:, j, :], axis = 1)
                    inter_value = group_value
                    si[:, j, idx_comb_feat[i][0]] = inter_value
                    si[:, idx_comb_feat[i][0], j] = inter_value
        # Cases of interaction between two one-hot encoded features
        if len(idx_comb_feat) > 1:
            for i in range(len(idx_comb_feat)):
                for k in range(1, len(idx_comb_feat)):
                    inter_value = np.sum(si[:, :, idx_comb_feat[i]][:, idx_comb_feat[k], :], axis = (1, 2))
                    i1 = idx_comb_feat[k][0]
                    i2 = idx_comb_feat[i][0]
                    si[:, i1, i2] = inter_value
                    si[:, i2, i1] = inter_value
        drop = np.array([], dtype=int)
        for i in range(len(idx_comb_feat)):
            drop = np.concatenate((drop, idx_comb_feat[i][1:]))
        si = np.delete(si, drop, axis = 1)
        si = np.delete(si, drop, axis = 2)
    return si



def make_shap_names(names_features, idx_comb_feat, combine_discr_features, do_inter_imp = True):
    if idx_comb_feat:
        names_features = replace_names_by_feature_group(names_features, idx_comb_feat, combine_discr_features)
    names_features = np.array(names_features)
    names_main_features = copy_lib.deepcopy(names_features)
    u, ind = np.unique(names_main_features, return_index = True)
    names_main_features = u[np.argsort(ind)]
    names_main_features = np.stack((names_main_features, np.repeat('none', len(names_main_features))), axis = 1)
    nf = names_main_features
    if do_inter_imp:
        l = len(names_features)
        names_inter_features = np.repeat(names_features, l).reshape((l,l))
        iu1 = np.triu_indices(l, 1)
        names_inter_features = np.stack((names_inter_features[iu1], names_inter_features.T[iu1]), axis = 1)
        names_inter_features = np.sort(names_inter_features, axis=1)
        names_inter_features_df = pd.DataFrame(names_inter_features)
        keep = names_inter_features_df.duplicated() == False
        names_inter_features = names_inter_features_df.loc[keep, :].to_numpy()
        keep = names_inter_features[:, 0] != names_inter_features[:, 1]
        names_inter_features = names_inter_features[keep, :]
        nf = np.vstack((names_main_features, names_inter_features))
    return nf


def make_taxa_names_shap(taxa_names, n_species, shap_names):
    sn = shap_names.to_numpy()
    shap_names_main = sn[sn[:, 1] == 'none', 0]
    taxa_names_shap = ['baseline']
    for i in range(n_species):
        for j in range(len(shap_names_main)):
            taxa_names_shap.append(taxa_names[i] + '__' + shap_names_main[j])
    return taxa_names_shap


def combine_shap_featuregroup(shap_main_instances, shap_interaction_instances, idx_comb_feat, use_mean=False):
    '''
    Combines instance-specific shap values of a feature group, e.g. one-hot encoded multistate traits
    use_mean switches between (a) taking first the sum of all instance-specific shaps and then the absolute of this sum to get the global importance, or (b) using the mean absolute values across all instance values
    '''
    baseline = np.array([shap_main_instances[0, -1]])
    shap_main_instances_global_importance = main_shap_for_onehot_features(idx_comb_feat, shap_main_instances[:, 0:-1], use_mean)
    shap_main = np.mean(np.abs(shap_main_instances_global_importance), axis = 0)
    shap_main_instances = main_shap_for_onehot_features(idx_comb_feat, shap_main_instances[:, 0:-1])

    shap_interaction = np.array([])
    if np.any(shap_interaction_instances):
        shap_interaction_instances = inter_shap_for_onehot_features(idx_comb_feat, shap_interaction_instances, use_mean)
        shap_interaction = np.mean(np.abs(shap_interaction_instances), axis = 0)
        iu1 = np.triu_indices(shap_interaction.shape[0], 1)
        shap_interaction = shap_interaction[iu1]
    combined_shap = np.concatenate((shap_main, shap_interaction, baseline, shap_main_instances.flatten()))
    return combined_shap


def k_add_kernel_shap_i(arg):
    [bdnn_obj, post_ts_i, post_te_i, post_w_sp_i, post_w_ex_i, t_reg_lam_i, t_reg_mu_i, reg_denom_lam_i, reg_denom_mu_i, hidden_act_f, out_act_f, trt_tbls, bdnn_dd, div_idx_trt_tbl, idx_comb_feat_sp, idx_comb_feat_ex, do_inter_imp, use_mean] = arg
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
    if do_inter_imp:
        shap_main_sp, shap_interaction_sp = k_add_kernel_explainer(shap_trt_tbl_sp, post_w_sp_i, t_reg_lam_i, reg_denom_lam_i, hidden_act_f, out_act_f)
        shap_main_ex, shap_interaction_ex = k_add_kernel_explainer(shap_trt_tbl_ex, post_w_ex_i, t_reg_mu_i, reg_denom_mu_i, hidden_act_f, out_act_f)
    else:
        shap_main_sp = fastshap_kernel_explainer(shap_trt_tbl_sp, post_w_sp_i, t_reg_lam_i, reg_denom_lam_i, hidden_act_f, out_act_f)
        shap_main_ex = fastshap_kernel_explainer(shap_trt_tbl_ex, post_w_ex_i, t_reg_mu_i, reg_denom_mu_i, hidden_act_f, out_act_f)
        shap_interaction_sp = np.array([])
        shap_interaction_ex = np.array([])
    lam_ke = combine_shap_featuregroup(shap_main_sp, shap_interaction_sp, idx_comb_feat_sp, use_mean)
    mu_ke = combine_shap_featuregroup(shap_main_ex, shap_interaction_ex, idx_comb_feat_ex, use_mean)
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


def get_species_rates_from_shap(shap, n_species, n_main_eff, mcmc_samples):
    baseline = shap[:, 0]
    sr = np.zeros((n_species, mcmc_samples))
    for i in range(mcmc_samples):
        shap_species = shap[i, 1:].reshape((n_species, n_main_eff))
        sr[:, i] = baseline[i] + np.sum(shap_species, axis = 1)
    sr_summary = get_rates_summary(sr)
    return sr_summary


def merge_taxa_shap_and_species_rates(taxa_shap, taxa_names_shap, rates_from_shap, n_species):
    n = taxa_shap.shape[0]
    r = np.zeros((n, 3))
    r[:] = np.nan
    idx = np.arange(1, n, int((n - 1) / n_species))
    r[idx, :] = rates_from_shap
    merged = np.hstack((taxa_shap, r))
    merged_df = pd.DataFrame(merged, columns = ['shap', 'lwr_shap', 'upr_shap', 'rate', 'rate_lwr', 'rate_upr'])
    taxa_names_shap_df = pd.DataFrame(taxa_names_shap, columns = ['feature'])
    out_df = pd.concat([taxa_names_shap_df, merged_df], axis = 1)
    return out_df


def make_shap_result_for_single_feature(names_features_sp, names_features_ex, combine_discr_features):
    one_feature_name_sp = names_features_sp[0]
    one_feature_name_ex = names_features_ex[0]
    if len(combine_discr_features) > 0:
        first_feature_group = list(combine_discr_features.keys())[0]
        one_feature_name_sp = first_feature_group
        one_feature_name_ex = first_feature_group
    shap_lam = pd.DataFrame({'feature1': one_feature_name_sp, 'feature2': 'none',
                             'shap': np.nan, 'lwr_shap': np.nan, 'upr_shap': np.nan},
                             index = [0])
    shap_ex = pd.DataFrame({'feature1': one_feature_name_ex, 'feature2': 'none',
                             'shap': np.nan, 'lwr_shap': np.nan, 'upr_shap': np.nan},
                             index = [0])
    taxa_shap_sp = pd.DataFrame(columns = ['shap', 'lwr_shap', 'upr_shap', 'rate', 'rate_lwr', 'rate_upr'])
    taxa_shap_ex = pd.DataFrame(columns = ['shap', 'lwr_shap', 'upr_shap', 'rate', 'rate_lwr', 'rate_upr'])
    return shap_lam, shap_ex, taxa_shap_sp, taxa_shap_ex


def make_shap_result_for_single_feature_sampling(names_features_q, combine_discr_features):
    one_feature_name_q = names_features_q[0]
    if len(combine_discr_features) > 0:
        first_feature_group = list(combine_discr_features.keys())[0]
        one_feature_name_q = first_feature_group
    shap_q = pd.DataFrame({'feature1': one_feature_name_q, 'feature2': 'none',
                           'shap': np.nan, 'lwr_shap': np.nan, 'upr_shap': np.nan},
                           index = [0])
    taxa_shap_q = pd.DataFrame(columns = ['shap', 'lwr_shap', 'upr_shap', 'rate', 'rate_lwr', 'rate_upr'])
    return shap_q, shap_q


def k_add_kernel_shap(mcmc_file, pkl_file, burnin, thin, num_processes=1, combine_discr_features={}, show_progressbar=False, do_inter_imp=True, use_mean=False):
#    if do_inter_imp == False:
#        from fastshap import KernelExplainer
    bdnn_obj, post_w_sp, post_w_ex, _, sp_fad_lad, post_ts, post_te, post_t_reg_lam, post_t_reg_mu, _, post_reg_denom_lam, post_reg_denom_mu, _, _ = bdnn_parse_results(mcmc_file, pkl_file, burnin, thin)
    mcmc_samples = post_ts.shape[0]
    trt_tbls = bdnn_obj.trait_tbls[:2]
    n_species = trt_tbls[0].shape[-2]
    n_features = trt_tbls[0].shape[-1]
    names_features_sp = get_names_features(bdnn_obj, rate_type='speciation')
    names_features_ex = copy_lib.deepcopy(names_features_sp)
    n_states = 1
    if len(combine_discr_features) > 0:
        n_states = len(combine_discr_features[list(combine_discr_features.keys())[0]])
#    if n_features == 1 or (n_states == n_features):
#        return make_shap_result_for_single_feature(names_features_sp, names_features_ex, combine_discr_features)
    if n_features == 1:
        if n_states > n_features:
            do_inter_imp = False
        else:
            return make_shap_result_for_single_feature(names_features_sp, names_features_ex, combine_discr_features)
    bdnn_dd = 'diversity' in names_features_sp
    div_idx_trt_tbl = -1
    if is_time_trait(bdnn_obj) and bdnn_dd:
            div_idx_trt_tbl = -2
    hidden_act_f = bdnn_obj.bdnn_settings['hidden_act_f']
    out_act_f = bdnn_obj.bdnn_settings['out_act_f']
    idx_comb_feat_sp = get_idx_comb_feat(names_features_sp, combine_discr_features)
    idx_comb_feat_ex = get_idx_comb_feat(names_features_ex, combine_discr_features)
    shap_names_sp = make_shap_names(names_features_sp, idx_comb_feat_sp, combine_discr_features, do_inter_imp = do_inter_imp)
    shap_names_ex = make_shap_names(names_features_ex, idx_comb_feat_ex, combine_discr_features, do_inter_imp = do_inter_imp)
    n_main_eff_sp = np.sum(shap_names_sp[:, 1] == 'none')
    n_main_eff_ex = np.sum(shap_names_ex[:, 1] == 'none')
    n_inter_eff_sp = int(n_main_eff_sp * (n_main_eff_sp - 1) / 2)
    n_inter_eff_ex = int(n_main_eff_sp * (n_main_eff_sp - 1) / 2)
    if do_inter_imp is False:
        n_inter_eff_sp = 0
        n_inter_eff_ex = 0
    n_effects_sp = n_main_eff_sp + n_inter_eff_sp + 1 + n_species * n_main_eff_sp # np.concatenate((shap_main, shap_interaction, baseline, shap_main_instances.flatten()))
    n_effects_ex = n_main_eff_ex + n_inter_eff_ex + 1 + n_species * n_main_eff_ex
    args = []
    for i in range(mcmc_samples):
        a = [bdnn_obj, post_ts[i, :], post_te[i, :],
             post_w_sp[i], post_w_ex[i], post_t_reg_lam[i], post_t_reg_mu[i], post_reg_denom_lam[i], post_reg_denom_mu[i],
             hidden_act_f, out_act_f, trt_tbls, bdnn_dd, div_idx_trt_tbl, idx_comb_feat_sp, idx_comb_feat_ex, do_inter_imp, use_mean]
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
    mean_shap_sp = shap_summary[:(n_main_eff_sp + n_inter_eff_sp), :]
    mean_shap_ex = shap_summary[n_effects_sp:(n_effects_sp + n_main_eff_ex + n_inter_eff_ex), :]
    taxa_shap_sp = shap_summary[(n_main_eff_sp + n_inter_eff_sp):n_effects_sp, :] # First row is baseline
    taxa_shap_ex = shap_summary[(n_effects_sp + n_main_eff_ex + n_inter_eff_ex):, :]
    if bdnn_dd:
        trt_tbls[0][0, :, div_idx_trt_tbl] = 1.0
        trt_tbls[1][0, :, div_idx_trt_tbl] = 1.0
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
    sp_from_shap = get_species_rates_from_shap(shap_values[:, (n_main_eff_sp + n_inter_eff_sp):n_effects_sp],
                                               n_species, n_main_eff_sp, mcmc_samples)
    ex_from_shap = get_species_rates_from_shap(shap_values[:, (n_effects_sp + n_main_eff_ex + n_inter_eff_ex):],
                                               n_species, n_main_eff_ex, mcmc_samples)
    taxa_shap_sp = merge_taxa_shap_and_species_rates(taxa_shap_sp, taxa_names_shap_sp, sp_from_shap, n_species)
    taxa_shap_ex = merge_taxa_shap_and_species_rates(taxa_shap_ex, taxa_names_shap_ex, ex_from_shap, n_species)
    return shap_lam, shap_ex, taxa_shap_sp, taxa_shap_ex


def k_add_kernel_shap_sampling_i(arg):
    [post_w_q_i, t_reg_q_i, reg_denom_q_i, q, n, hidden_act_f, out_act_f, shap_trt_tbl, idx_comb_feat, do_inter_imp] = arg
    if do_inter_imp:
        shap_main, shap_interaction = k_add_kernel_explainer(shap_trt_tbl, post_w_q_i, t_reg_q_i, reg_denom_q_i, hidden_act_f, out_act_f, q, n)
    else:
        shap_main = fastshap_kernel_explainer(shap_trt_tbl, post_w_q_i, t_reg_q_i, reg_denom_q_i, hidden_act_f, out_act_f, q, n) # check this later
        shap_interaction = np.array([])
    ke = combine_shap_featuregroup(shap_main, shap_interaction, idx_comb_feat)
    return ke


def k_add_kernel_shap_sampling(mcmc_file, pkl_file, burnin, thin, num_processes = 1, combine_discr_features = {}, show_progressbar = False, do_inter_imp = True):
    bdnn_obj, _, _, post_w_q, sp_fad_lad, post_ts, post_te, _, _, post_t_reg_q, _, _, post_reg_denom_q, post_norm_q = bdnn_parse_results(mcmc_file, pkl_file, burnin, thin)
    mcmc_samples = post_ts.shape[0]
    trt_tbl = get_trt_tbl(bdnn_obj, 'sampling')
    n_species = trt_tbl.shape[-2]
    n_features = trt_tbl.shape[-1]
    names_features = get_names_features(bdnn_obj, rate_type='sampling')
    n_states = 1
    if len(combine_discr_features) > 0:
        n_states = len(combine_discr_features[list(combine_discr_features.keys())[0]])
    if n_features == 1:
        if n_states > n_features:
            do_inter_imp = False
        else:
            return make_shap_result_for_single_feature_sampling(names_features, combine_discr_features)
    hidden_act_f = bdnn_obj.bdnn_settings['hidden_act_f']
    out_act_f = bdnn_obj.bdnn_settings['out_act_f_q']
    idx_comb_feat = get_idx_comb_feat(names_features, combine_discr_features)
    shap_names = make_shap_names(names_features, idx_comb_feat, combine_discr_features, do_inter_imp = do_inter_imp)
    n_main_eff = np.sum(shap_names[:, 1] == 'none')
    n_inter_eff = int(n_main_eff * (n_main_eff - 1) / 2)
    if do_inter_imp is False:
        n_inter_eff = 0
    n_effects = n_main_eff + n_inter_eff + 1 + n_species * n_main_eff
    q = get_baseline_q(mcmc_file, burnin, thin, mean_across_shifts=True)
    args = []
    for i in range(mcmc_samples):
        if 'taxon_age' in names_features:
            trt_tbl = add_taxon_age(post_ts[i, :], post_te[i, :], bdnn_obj.bdnn_settings['q_time_frames'], trt_tbl)
        shap_trt_tbl = get_shap_trt_tbl_sampling(bdnn_obj.bdnn_settings['occs_sp'], trt_tbl)
        a = [post_w_q[i], post_t_reg_q[i], post_reg_denom_q[i], q[i, :], post_norm_q[i],
             hidden_act_f, out_act_f,
             shap_trt_tbl, idx_comb_feat, do_inter_imp]
        args.append(a)
    unixos = is_unix()
    if unixos and num_processes > 1:
        pool_perm = multiprocessing.Pool(num_processes)
        shap_values = list(tqdm(pool_perm.imap_unordered(k_add_kernel_shap_sampling_i, args),
                                total = mcmc_samples, disable = show_progressbar == False))
        pool_perm.close()
    else:
        shap_values = []
        for i in tqdm(range(mcmc_samples), disable = show_progressbar == False):
            shap_values.append(k_add_kernel_shap_sampling_i(args[i]))
    shap_values = np.vstack(shap_values)
    shap_summary = get_rates_summary(shap_values.T)
    mean_shap = shap_summary[:(n_main_eff + n_inter_eff), :]
    taxa_shap = shap_summary[(n_main_eff + n_inter_eff):, :] # First row is baseline
    feature_without_variance = get_idx_feature_without_variance(trt_tbl)
    remove_q = []
    for i in feature_without_variance:
        remove_q.append(np.where(shap_names[:, 0] == names_features[i])[0])
        remove_q.append(np.where(shap_names[:, 1] == names_features[i])[0])
    remove_q = np.array(list(pd.core.common.flatten(remove_q))).astype(int)
    mean_shap = np.delete(mean_shap, remove_q[remove_q < len(mean_shap)], axis = 0)
    shap_names_del = np.delete(shap_names, remove_q, axis = 0)
    shap_values_q = pd.DataFrame(mean_shap, columns = ['shap', 'lwr_shap', 'upr_shap'])
    shap_names_del = pd.DataFrame(shap_names_del, columns = ['feature1', 'feature2'])
    shap_q = pd.concat([shap_names_del, shap_values_q], axis = 1)
    taxa_names = sp_fad_lad["Taxon"]
    taxa_names_shap = make_taxa_names_shap(taxa_names, n_species, shap_names_del)
    taxa_shap = delete_invariantfeat_from_taxa_shap(feature_without_variance, names_features,
                                                    shap_names, taxa_shap)
    q_from_shap = get_species_rates_from_shap(shap_values[:, (n_main_eff + n_inter_eff):],
                                              n_species, n_main_eff, mcmc_samples)
    taxa_shap_q = merge_taxa_shap_and_species_rates(taxa_shap, taxa_names_shap, q_from_shap, n_species)
    return shap_q, taxa_shap_q


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
#    if pc == 0:
#        #po = np.zeros(1)
#        addpenalty = np.zeros(1)
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
        has_probs = not p_tmp['posterior_probability'].isnull().all()
#        pp = p_tmp['posterior_probability'].to_numpy()
#        has_probs = ~np.all(np.isnan(pp))
#        print('p\n', pp)
#        print('has probs', has_probs)
#        print('has_probs 2\n', np.__bool__(has_probs))
#        h = p_tmp['posterior_probability'].argmax()
#        has_probs = h != -1
        if has_probs:
            h = p_tmp['posterior_probability'].argmax()
            p_tmp = p_tmp.loc[h, columns].to_frame().T
        else:
            a = np.zeros(4)
            a[:] = np.nan
            p_tmp = pd.DataFrame(a.reshape((1, 4)), columns = columns)
        highest_p = pd.concat([highest_p, p_tmp], axis = 0, ignore_index = True)
    highest_p = pd.concat([unique_features, highest_p], axis = 1)
    return highest_p


def get_same_order(pv, sh, fp):
    pv = pv.reset_index(drop=True)
    pv_reord = highest_pvalue_from_interaction(pv)
    nrows = len(pv_reord)
    for i in range(nrows):
        pv_feat1 = pv_reord.loc[i, 'feature1']
        pv_feat2 = pv_reord.loc[i, 'feature2']
        sh_tmp = sh[(sh['feature1'] == pv_feat1) &
                    (sh['feature2'] == pv_feat2) |
                    (sh['feature1'] == pv_feat2) &
                    (sh['feature2'] == pv_feat1)].reset_index(drop=True)
        sh_tmp.loc[0, 'feature1'] = pv_feat1
        sh_tmp.loc[0, 'feature2'] = pv_feat2
        fp_tmp = fp[(fp['feature1'] == pv_feat1) &
                    (fp['feature2'] == pv_feat2) |
                    (fp['feature1'] == pv_feat2) &
                    (fp['feature2'] == pv_feat1)].reset_index(drop=True)
        fp_tmp.loc[0, 'feature1'] = pv_feat1
        fp_tmp.loc[0, 'feature2'] = pv_feat2
        if i == 0:
            sh_reord = sh_tmp
            fp_reord = fp_tmp
        else:
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
    feat_importance_main[nan_idx_main] = np.nanmax(feat_importance_main) + 1
    ranked_feat_import_main = stats.rankdata(feat_importance_main, axis = 1, method = 'min').astype(float)
    ranked_feat_import_main[nan_idx_main] = np.nan
    n_inter = np.sum(idx_main == False)
    ranked_feat_import_inter = np.array([])
    if n_inter == 1:
        ranked_feat_import_inter = np.zeros((1, 3))
    elif n_inter > 1:
        nan_idx_inter = np.isnan(feat_importance_inter)
        feat_importance_inter[nan_idx_inter] = np.nanmax(feat_importance_inter) + 1
        ranked_feat_import_inter = stats.rankdata(feat_importance_inter, axis = 1, method = 'min').astype(float)
        ranked_feat_import_inter[nan_idx_inter] = np.nan
    return ranked_feat_import_main, ranked_feat_import_inter


def merge_results_feat_import(pv, sh, fp, rr):
    pv = pv.reset_index(drop=True)
    nrows = len(pv)
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
        if i == 0:
            sh_merge = sh_tmp[['shap', 'lwr_shap', 'upr_shap']]
            fp_merge = fp_tmp[['delta_lik', 'lwr_delta_lik', 'upr_delta_lik']]
            rr_merge = rr_tmp[['rank']]
        else:
            sh_merge = pd.concat([sh_merge, sh_tmp[['shap', 'lwr_shap', 'upr_shap']]], axis = 0, ignore_index = True)
            fp_merge = pd.concat([fp_merge, fp_tmp[['delta_lik', 'lwr_delta_lik', 'upr_delta_lik']]], axis = 0, ignore_index = True)
            rr_merge = pd.concat([rr_merge, rr_tmp[['rank']]], axis = 0, ignore_index = True)
    merged = pd.concat([pv, sh_merge, fp_merge, rr_merge], axis = 1)
    return merged


def get_consensus_ranking(pv, sh, fp):
    pv_reordered, sh_reordered, fp_reordered = get_same_order(pv, sh, fp)
    feat_main_ranked, feat_inter_ranked = rank_features(pv_reordered, sh_reordered, fp_reordered)
    main_consrank = np.zeros(1)
    if feat_main_ranked.shape[1] > 1:
        main_consranks = quickcons(feat_main_ranked)
        main_consrank = stats.mode(main_consranks[0], axis = 0)[0].flatten()
    inter_consrank = np.array([])
    if feat_inter_ranked.shape[0] > 0:
        inter_consrank = np.zeros(1)
        if feat_inter_ranked.shape[1] > 1:
            inter_consranks = quickcons(feat_inter_ranked)
            inter_consrank = stats.mode(inter_consranks[0], axis = 0)[0].flatten()
    rank_df = pd.DataFrame(np.concatenate((main_consrank, inter_consrank)) + 1.0, columns = ['rank'])
    r = pd.concat([pv_reordered[['feature1', 'feature2']], rank_df], axis = 1)
    feat_merged = merge_results_feat_import(pv, sh, fp, r)
#    feat_merged = merge_results_feat_import(pv_reordered, sh_reordered, fp_reordered, r)
    return feat_merged, main_consrank


# Plot SHAP
###########
def plot_species_shap(pkl_file, output_wd, name_file, sp_taxa_shap, ex_taxa_shap, sp_consrank, ex_consrank):
    ob = load_pkl(pkl_file)
    species_names = ob.sp_fad_lad["Taxon"]
    suffix_pdf = "contribution_per_species_rates"
    out = "%s/%s_%s.r" % (output_wd, name_file, suffix_pdf)
    newfile = open(out, "w")
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        wd_forward = os.path.abspath(output_wd).replace('\\', '/')
        r_script = "pdf(file='%s/%s_%s.pdf', width = 7, height = 6, useDingbats = FALSE, pointsize = 7)\n" % (wd_forward, name_file, suffix_pdf)
    else:
        r_script = "pdf(file='%s/%s_%s.pdf', width = 7, height = 6, useDingbats = FALSE, pointsize = 7)\n" % (output_wd, name_file, suffix_pdf)
    r_script += "\nord_by_importance = function(s, consrank, rate_mean) {"
    r_script += "\n  imp = consrank**2"
    r_script += "\n  s = scale(s)"
    r_script += "\n  imp = imp / sum(imp)"
    r_script += "\n  #imp = imp / max(imp)"
    r_script += "\n  s = t(t(s) * imp)"
    r_script += "\n  p = prcomp(cbind(s, scale(rate_mean)))"
    r_script += "\n  ord <- order(p$x[, 1])"
    r_script += "\n  if (cor(1:length(rate_mean), rate_mean[ord]) < 0) {"
    r_script += "\n     ord = order(p$x[, 1], decreasing = TRUE)"
    r_script += "\n  }"
    r_script += "\n  return(ord)"
    r_script += "\n}"
    r_script += "\nshap_heatmap <- function(shap, baseline, rates, species_names, feat_names, rate_type = 'speciation') {"
    r_script += "\n  nspecies = nrow(shap)"
    r_script += "\n  nfeat = ncol(shap)"
    r_script += "\n  shap_pos = shap > 0"
    r_script += "\n  shap_neg = shap < 0"
    r_script += "\n  shap_sqrt = shap"
    r_script += "\n  shap_sqrt[shap == 0] = 0"
    r_script += "\n  offset_sqrt = 0.0"
    r_script += "\n  shap_sqrt[shap_pos] = sqrt(shap[shap_pos] + offset_sqrt)"
    r_script += "\n  shap_sqrt[shap_neg] = sqrt(-1 * shap[shap_neg] + offset_sqrt)"
    r_script += "\n  max_pos = max(shap_sqrt[shap_pos])"
    r_script += "\n  max_neg = max(shap_sqrt[shap_neg])"
    r_script += "\n  steps_pos = round(max_pos / (max_pos + max_neg), 2) * 200"
    r_script += "\n  steps_neg = round(max_neg / (max_pos + max_neg), 2) * 200"
    r_script += "\n  # BrBG diverging colors"
    r_script += "\n  total_steps = steps_pos + steps_neg"
    r_script += "\n  max_steps = max(c(steps_pos, steps_neg))"
    r_script += "\n  col_pos = colorRampPalette(c('#F5F5F5', '#543005'))(max_steps)"
    r_script += "\n  col_neg = colorRampPalette(c('#F5F5F5', '#003C30'))(max_steps)"
    r_script += "\n  col_pos = col_pos[1:steps_pos]"
    r_script += "\n  col_neg = col_neg[1:steps_neg]"
    r_script += "\n  shap_col = shap"
    r_script += "\n  shap_col[shap == 0] = '#F5F5F5'"
    r_script += "\n  col_idx = findInterval(shap_sqrt[shap_pos], seq(sqrt(offset_sqrt), max_pos, length.out = steps_pos), all.inside = TRUE)"
    r_script += "\n  shap_col[shap_pos] = col_pos[col_idx]"
    r_script += "\n  col_idx = findInterval(shap_sqrt[shap_neg], seq(sqrt(offset_sqrt), max_neg, length.out = steps_neg), all.inside = TRUE)"
    r_script += "\n  shap_col[shap_neg] = col_neg[col_idx]"
    r_script += "\n  species_names = gsub('_', ' ', species_names)"
    r_script += "\n  # Plot"
    r_script += "\n  heights = c(0.1, 0.9)"
    r_script += "\n  if (nfeat < 9) {"
    r_script += "\n    h = seq(0.5, 0.8, length.out = 8)"
    r_script += "\n    h = h[nfeat]"
    r_script += "\n    heights = c(1 - h, h)"
    r_script += "\n  }"
    r_script += "\n  rate_name = 'Speciation rate'"
    r_script += "\n  rate_col = 'dodgerblue'"
    r_script += "\n  if (rate_type == 'extinction') {"
    r_script += "\n    rate_name = 'Extinction rate'"
    r_script += "\n    rate_col = 'red'"
    r_script += "\n  }"
    r_script += "\n  layout(matrix(1:4, nrow = 2, ncol = 2), heights = heights, widths = c(0.9, 0.1))"
    r_script += "\n  # Rates per species"
    r_script += "\n  par(las = 1, mar = c(0.1, 6, 0.5, 0.1), mgp = c(3, 1, 0))"
    r_script += "\n  y_tck = pretty(range(rates, na.rm = TRUE), n = 5)"
    r_script += "\n  plot(0, 0, type = 'n', xaxs = 'i', yaxs = 'i',"
    r_script += "\n       xlim = c(0, nspecies), ylim = range(y_tck),"
    r_script += "\n       axes = FALSE, ylab = rate_name)"
    r_script += "\n  abline(h = baseline, lty = 2, col = 'grey')"
    r_script += "\n  x = c(1:nspecies) - 0.5"
    r_script += "\n  polygon(c(x, rev(x)), c(rates[, 2], rev(rates[, 3])), col = adjustcolor(rate_col, alpha = 0.25), border = NA)"
    r_script += "\n  lines(x, rates[, 1], col = rate_col, lwd = 1.5)"
    r_script += "\n  axis(side = 2, at = y_tck)"
    r_script += "\n  # Shape values"
    r_script += "\n  par(mar = c(4, 6, 0.5, 0.1), mgp = c(3, 1, 0))"
    r_script += "\n  plot(0, 0, type = 'n', xlim = c(0, nspecies), ylim = c(0, nfeat),"
    r_script += "\n       xaxs = 'i', yaxs = 'i', xlab = '', ylab = '', axes = FALSE)"
    r_script += "\n  par(ljoin = 1)"
    r_script += "\n  for (i in 1:nspecies) {"
    r_script += "\n    for (j in 1:nfeat) {"
    r_script += "\n      rect(i - 1, j - 1, i, j, border = shap_col[i, j], col = shap_col[i, j])"
    r_script += "\n    }"
    r_script += "\n  }"
    r_script += "\n  par(las = 2, mgp = c(3, 0.5, 0), ljoin = 0)"
    r_script += "\n  cex_axis = 2 / sqrt(length(feat_names))"
    r_script += "\n  cex_axis = ifelse(cex_axis > 1, 1, cex_axis)"
    r_script += "\n  axis(side = 2, at = c(1:nfeat) - 0.5, lwd = -1, lwd.ticks = -1,"
    r_script += "\n       labels = feat_names, cex.axis = cex_axis)"
    r_script += "\n  par(mgp = c(3, 0.1, 0))"
    r_script += "\n  text(x = c(1:nspecies) - 0.5, y = par('usr')[3] - diff(par('usr')[3:4]) * 0.01,"
    r_script += "\n       labels = species_names, xpd = NA, srt = 35, adj = 0.965,"
    r_script += "\n       cex = 2 / sqrt(length(species_names)), font = 3)"
    r_script += "\n  # Empty plot"
    r_script += "\n  plot(0, 0, type = 'n', axes = FALSE, xlab = '', ylab = '')"
    r_script += "\n  # Legend"
    r_script += "\n  col = c(rev(col_neg), col_pos)"
    r_script += "\n  n_lead_digit = nchar(as.character(max(round(abs(shap)))))"
    r_script += "\n  par(mar = c(4, 0.5, 0.5, 0.5), mgp = c(3, 1, 0))"
    r_script += "\n  plot(0, 0, type = 'n', xlab = '', ylab = '', axes = FALSE,"
    r_script += "\n       xlim = c(0, 2), ylim = c(0, 1.05 * total_steps))"
    r_script += "\n  for (j in 1:total_steps) {"
    r_script += "\n    rect(0, j - 1, 0.7, j, border = col[j], col = col[j])"
    r_script += "\n  }"
    r_script += "\n  rect(0, 0, 0.7, total_steps, border = 'black')"
    r_script += "\n  lines(x = c(0.7, 1.0), y = rep(steps_neg, 2))"
    r_script += "\n  text(x = 0, y = 1.05 * total_steps, labels = 'Rate change', adj = c(0, 0.5))"
    r_script += "\n  a = c(1, 0.5) # right align"
    r_script += "\n  text(x = 2, y = steps_neg, labels = sprintf(paste0('%.', 3 - n_lead_digit, 'f'), 0), adj = a)"
    r_script += "\n  n_tck = 4"
    r_script += "\n  s = seq(sqrt(offset_sqrt), max(c(max_pos, max_neg)), length.out = n_tck)"
    r_script += "\n  pos = (s / s[n_tck]) * max_steps"
    r_script += "\n  pos = pos[-1]"
    r_script += "\n  s = s[-1]"
    r_script += "\n  for (i in 1:length(pos)) {"
    r_script += "\n    p = steps_neg + pos[i]"
    r_script += "\n    if (p <= total_steps) {"
    r_script += "\n      lines(x = c(0.7, 1.0), y = rep(p, 2))"
    r_script += "\n      text(x = 2, y = p, adj = a, labels = sprintf(paste0('%.', 3 - n_lead_digit, 'f'), s[i]**2))"
    r_script += "\n    }"
    r_script += "\n    p = steps_neg - pos[i]"
    r_script += "\n    if (p >= 0) {"
    r_script += "\n      lines(x = c(0.7, 1.0), y = rep(p, 2))"
    r_script += "\n      text(x = 2, y = p, adj = a, labels = sprintf(paste0('%.', 3 - n_lead_digit, 'f'), -(s[i]**2)))"
    r_script += "\n    }"
    r_script += "\n  }"
    r_script += "\n}"
    # Speciation
    r_script = get_rscript_species_shap(r_script, species_names, sp_taxa_shap, sp_consrank, rate_type = 'speciation')
    r_script = get_rscript_species_shap(r_script, species_names, ex_taxa_shap, ex_consrank, rate_type = 'extinction')
    r_script += "\ndev.off()"
    newfile.writelines(r_script)
    newfile.close()
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        cmd = "cd %s & Rscript %s_%s.r" % (output_wd, name_file, suffix_pdf)
    else:
        cmd = "cd %s; Rscript %s_%s.r" % (output_wd, name_file, suffix_pdf)
    print("cmd", cmd)
    os.system(cmd)


def get_rscript_species_shap(r_script, species_names, taxa_shap, consrank, rate_type = 'speciation'):
    nspecies = len(species_names)
    baseline = taxa_shap.iloc[0, 1]
    shap = taxa_shap.iloc[1:, 1].to_numpy()
    nfeat = int(len(shap) / nspecies)
    shap = shap.reshape((nspecies, nfeat))
    r_script += util.print_R_vec("\nconsrank", consrank)
    r_script += "\nconsrank = order(consrank, decreasing = TRUE)"
    r_script += "\nshap_list = list()"
    for i in range(nfeat):
        r_script += util.print_R_vec("\nshap_list[[%s]]", np.round(shap[:, i], 1)) % (i + 1)
    r_script += "\nshap = do.call('cbind', shap_list)"
    r_script += "\nbaseline = %s" % baseline
    r = taxa_shap.loc[1:, 'rate']
    r_lwr = taxa_shap.loc[1:, 'rate_lwr']
    r_upr = taxa_shap.loc[1:, 'rate_upr']
    is_rate = np.isnan(r) == False
    r_script += util.print_R_vec("\nrate", r[is_rate].to_numpy().flatten())
    r_script += util.print_R_vec("\nrate_lwr", r_lwr[is_rate].to_numpy().flatten())
    r_script += util.print_R_vec("\nrate_upr", r_upr[is_rate].to_numpy().flatten())
    r_script += "\nrates = cbind(rate, rate_lwr, rate_upr)"
    r_script += "\nspecies_names = c()"
    for i in range(nspecies):
        r_script += "\nspecies_names = c(species_names, '%s')" % species_names[i]
    feat_names = taxa_shap.iloc[1:(nfeat+1), 0].to_list()
    r_script += "\nfeat_names = c()"
    for i in range(nfeat):
        r_script += "\nfeat_names = c(feat_names, '%s')" % feat_names[i].split('__')[-1]
    r_script += "\nord = ord_by_importance(shap, consrank, rate)"
    r_script += "\nshap_ord = shap[ord, consrank, drop = FALSE]"
    r_script += "\nrates_ord = rates[ord, ]"
    r_script += "\nspecies_names_ord = species_names[ord]"
    r_script += "\nfeat_names_ord = feat_names[consrank]"
    r_script += "\nshap_heatmap(shap_ord, baseline, rates_ord, species_names_ord, feat_names_ord, rate_type = '%s')" % rate_type
    return r_script


def get_features_for_shap_plot(mcmc_file, pkl_file, burnin, thin, rate_type, combine_discr_features, file_transf_features):
    bdnn_obj, post_w_sp, post_w_ex, post_w_q, sp_fad_lad, ts_post, te_post, post_t_reg_lam, post_t_reg_mu, post_t_reg_q, post_reg_denom_lam, post_reg_denom_mu, post_reg_denom_q, post_norm_q = bdnn_parse_results(mcmc_file, pkl_file, burnin, thin)
    trait_tbl = get_trt_tbl(bdnn_obj, rate_type)
    names_features = get_names_features(bdnn_obj, rate_type)
    names_features_orig = np.array(names_features)
    # diversity-dependence
    if "diversity" in names_features and rate_type != 'sampling':
        div_time, div_traj = get_mean_div_traj(ts_post, te_post)
        bdnn_time = get_bdnn_time(bdnn_obj, ts_post)
        div_traj_binned = get_binned_div_traj(bdnn_time, div_time, div_traj)[:-1]
        div_traj_binned = np.repeat(div_traj_binned, trait_tbl.shape[1]).reshape((trait_tbl.shape[0], trait_tbl.shape[1]))
        div_idx_trt_tbl = -1
        if is_time_trait(bdnn_obj):
            div_idx_trt_tbl = -2
        trait_tbl[ :, :, div_idx_trt_tbl] = div_traj_binned
    idx_comb_feat = get_idx_comb_feat(names_features, combine_discr_features)
    if idx_comb_feat:
        names_features = replace_names_by_feature_group(names_features, idx_comb_feat, combine_discr_features)
    names_features = np.array(names_features)
    if rate_type == 'sampling':
        if "taxon_age" in names_features:
            ts = np.mean(ts_post, axis=0)
            te = np.mean(te_post, axis=0)
            trait_tbl = add_taxon_age(ts, te, bdnn_obj.bdnn_settings['q_time_frames'], trait_tbl)
        shap_trt_tbl = get_shap_trt_tbl_sampling(bdnn_obj.bdnn_settings['occs_sp'], trait_tbl)
        if file_transf_features != '':
            names_feat = get_names_features(bdnn_obj, rate_type='sampling') # Needed?
            backscale_par = read_backscale_file(file_transf_features)
            shap_trt_tbl = backscale_tbl(bdnn_obj, backscale_par, names_feat, shap_trt_tbl)
    else:
        bdnn_time = get_bdnn_time(bdnn_obj, np.max(ts_post, axis=0))
        if rate_type == 'speciation':
            tse = np.mean(ts_post, axis = 0)
        elif rate_type == 'extinction':
            tse = np.mean(te_post, axis = 0)
        shap_trt_tbl = get_shap_trt_tbl(tse, bdnn_time, trait_tbl)
        # Backscale features
        if is_time_trait(bdnn_obj):
            shap_trt_tbl[:, -1] = backscale_bdnn_time(shap_trt_tbl[:, -1], bdnn_obj)
        if file_transf_features != '':
            names_feat = get_names_features(bdnn_obj, rate_type='speciation')
            backscale_par = read_backscale_file(file_transf_features)
            shap_trt_tbl = backscale_tbl(bdnn_obj, backscale_par, names_feat, shap_trt_tbl)
    # Remove invariant features
    feature_without_variance = get_idx_feature_without_variance(shap_trt_tbl)
    shap_trt_tbl = np.delete(shap_trt_tbl, feature_without_variance, axis = 1)
    names_features = np.delete(names_features, feature_without_variance)
    names_features_orig = np.delete(names_features_orig, feature_without_variance)
    return shap_trt_tbl, names_features, names_features_orig


def get_dotplot_rscript_species_shap(r_script, species_names, taxa_shap, consrank, shap_trt_tbl, names_features, names_features_orig, rate_type):
    nspecies = len(species_names)
    baseline = taxa_shap.iloc[0, 1]
    shap = taxa_shap.iloc[1:, 1].to_numpy()
    nfeat = int(len(shap) / nspecies)
    shap = shap.reshape((nspecies, nfeat))
    r_script += util.print_R_vec("\nconsrank", consrank)
    r_script += "\nconsrank2 = order(consrank, decreasing = FALSE)"
    r_script += "\nconsrank = order(consrank, decreasing = TRUE)"
    r_script += "\nshap_list = list()"
    for i in range(nfeat):
        r_script += util.print_R_vec("\nshap_list[[%s]]", shap[:, i]) % (i + 1)
    r_script += "\nshap = do.call('cbind', shap_list)"
    r_script += "\nbaseline = %s" % baseline
    r = taxa_shap.loc[1:, 'rate']
    r_lwr = taxa_shap.loc[1:, 'rate_lwr']
    r_upr = taxa_shap.loc[1:, 'rate_upr']
    is_rate = np.isnan(r) == False
    r_script += util.print_R_vec("\nrate", r[is_rate].to_numpy().flatten())
    r_script += util.print_R_vec("\nrate_lwr", r_lwr[is_rate].to_numpy().flatten())
    r_script += util.print_R_vec("\nrate_upr", r_upr[is_rate].to_numpy().flatten())
    r_script += "\nrates = cbind(rate, rate_lwr, rate_upr)"
    r_script += "\nspecies_names = c()"
    for i in range(nspecies):
        r_script += "\nspecies_names = c(species_names, '%s')" % species_names[i]
    feat_names = taxa_shap.iloc[1:(nfeat+1), 0].to_list()
    feat_names = [feat_names[i].split('__')[-1] for i in range(len(feat_names))]
    r_script += "\nfeat_names = c()"
    for i in range(nfeat):
        r_script += "\nfeat_names = c(feat_names, '%s')" % feat_names[i]
    r_script += "\nord = ord_by_importance(shap, consrank, rate)"
    r_script += "\nshap_ord = shap[ord, consrank2, drop = FALSE]"
    r_script += "\nrates_ord = rates[ord, ]"
    r_script += "\nspecies_names_ord = species_names[ord]"
    r_script += "\nfeat_names_ord = feat_names[consrank2]"
    r_script += "\nfeat = vector(mode = 'list', length = %s)" % nfeat
    r_script += "\nfeat_states = vector(mode = 'list', length = %s)" % nfeat
    for i in range(nfeat):
        idx = np.where(names_features == feat_names[i])[0]
        len_featgroup = len(idx)
        no_featgroup = len_featgroup == 1
        if no_featgroup:
            r_script += util.print_R_vec("\ntmp", shap_trt_tbl[:, idx].flatten())
            r_script += "\nfeat[[%s]] = tmp[ord]" % (i + 1)
            r_script += "\nfeat_states[[%s]] = NA" % (i + 1)
        else:
            r_script += "\ntmp = vector(mode = 'list', length = %s)" % len_featgroup
            r_script += "\ntmp_states = c()"
            for j in idx:
                r_script += util.print_R_vec("\ntmp[[%s]]", shap_trt_tbl[:, j].flatten()) % (j + 1)
                r_script += "\ntmp_states = c(tmp_states, '%s')" % str(names_features_orig[j])
            r_script += "\ntmp = do.call('rbind', tmp)"
            r_script += "\nfeat[[%s]] = tmp[, ord]" % (i + 1)
            r_script += "\nfeat_states[[%s]] = tmp_states" % (i + 1)
    r_script += "\nfeat_ord = feat[consrank2]"
    r_script += "\nfeat_states_ord = feat_states[consrank2]"
    r_script += "\nshap_heatmap(shap_ord, baseline, rates_ord, species_names_ord, feat_ord,"
    r_script += "\n             feat_names_ord, feat_states_ord, rate_type = '%s', n_individual_pred = 3)" % rate_type
    r_script += "\n"
    return r_script


def dotplot_species_shap(mcmc_file, pkl_file, burnin, thin, output_wd, name_file,
                         sp_taxa_shap, ex_taxa_shap, q_taxa_shap,
                         sp_consrank, ex_consrank, q_consrank,
                         combine_discr_features = '', file_transf_features = ''):
    ob = load_pkl(pkl_file)
    species_names = ob.sp_fad_lad["Taxon"]
    suffix_pdf = "contribution_per_species_rates"
    out = "%s/%s_%s.r" % (output_wd, name_file, suffix_pdf)
    newfile = open(out, "w")
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        wd_forward = os.path.abspath(output_wd).replace('\\', '/')
        r_script = "pdf(file='%s/%s_%s.pdf', width = 7, height = 6, useDingbats = FALSE, pointsize = 7)\n" % (wd_forward, name_file, suffix_pdf)
    else:
        r_script = "pdf(file='%s/%s_%s.pdf', width = 7, height = 6, useDingbats = FALSE, pointsize = 7)\n" % (output_wd, name_file, suffix_pdf)
    r_script += "\nord_by_importance = function(s, consrank, rate_mean) {"
    r_script += "\n  imp = consrank**2"
    r_script += "\n  s = scale(s)"
    r_script += "\n  imp = imp / sum(imp)"
    r_script += "\n  #imp = imp / max(imp)"
    r_script += "\n  s = t(t(s) * imp)"
    r_script += "\n  p = prcomp(cbind(s, scale(rate_mean)))"
    r_script += "\n  ord <- order(p$x[, 1])"
    r_script += "\n  if (cor(1:length(rate_mean), rate_mean[ord]) < 0) {"
    r_script += "\n     ord = order(p$x[, 1], decreasing = TRUE)"
    r_script += "\n  }"
    r_script += "\n  return(ord)"
    r_script += "\n}"
    r_script += "\n"
    r_script += "\nis_discrete <- function(x) {"
    r_script += "\n  s = min(x):max(x)"
    r_script += "\n  any(x == 0) && length(s) > 1 && all(s %in% unique(x))"
    r_script += "\n}"
    r_script += "\n"
    r_script += "\ncombine_predictors <- function(shap, feat, feat_names, n_comb) {"
    r_script += "\n  n_comb1 = n_comb + 1"
    r_script += "\n  idx_comb = n_comb1:ncol(shap)"
    r_script += "\n  shap[, n_comb1] = rowSums(shap[, idx_comb])"
    r_script += "\n  shap = shap[, 1:n_comb1]"
    r_script += "\n  comb_pred = t(do.call('rbind', feat[idx_comb]))"
    r_script += "\n  pc1 = prcomp(comb_pred)$x[, 1]"
    r_script += "\n  feat[[n_comb1]] = pc1"
    r_script += "\n  feat[(n_comb + 2):(length(feat))] = NULL"
    r_script += "\n  feat_names = feat_names[1:n_comb1]"
    r_script += "\n  feat_names[n_comb1] = 'PC1 other predictors'"
    r_script += "\n  out = vector(mode = 'list', length = 3)"
    r_script += "\n  out[[1]] = shap"
    r_script += "\n  out[[2]] = feat"
    r_script += "\n  out[[3]] = feat_names"
    r_script += "\n  return(out)"
    r_script += "\n}"
    r_script += "\n"
    r_script += "\nshap_heatmap <- function(shap, baseline, rates, species_names, feat,"
    r_script += "\n                         feat_names, feat_states, rate_type = 'speciation', n_individual_pred = NULL) {"
    r_script += "\n  nfeat = ncol(shap)"
    r_script += "\n  if (!is.null(n_individual_pred)) {"
    r_script += "\n    if ((n_individual_pred + 1) < nfeat) {"
    r_script += "\n      comb_p = combine_predictors(shap, feat, feat_names, n_individual_pred)"
    r_script += "\n      shap = comb_p[[1]]"
    r_script += "\n      feat = comb_p[[2]]"
    r_script += "\n      feat_names = comb_p[[3]]"
    r_script += "\n      nfeat = ncol(shap)"
    r_script += "\n    }"
    r_script += "\n  }"
    r_script += "\n  nspecies = nrow(shap)"
    r_script += "\n  shap_pos = shap > 0"
    r_script += "\n  shap_neg = shap < 0"
    r_script += "\n  shap_sqrt = shap"
    r_script += "\n  shap_sqrt[shap == 0] = 0"
    r_script += "\n  offset_sqrt = 0.0"
    r_script += "\n  shap_sqrt[shap_pos] = sqrt(shap[shap_pos] + offset_sqrt)"
    r_script += "\n  shap_sqrt[shap_neg] = sqrt(-1 * shap[shap_neg] + offset_sqrt)"
    r_script += "\n  max_pos = max(shap_sqrt[shap_pos])"
    r_script += "\n  max_neg = max(shap_sqrt[shap_neg])"
    r_script += "\n  steps_pos = round(max_pos / (max_pos + max_neg), 2) * 200"
    r_script += "\n  steps_neg = round(max_neg / (max_pos + max_neg), 2) * 200"
    r_script += "\n  # BrBG diverging colors"
    r_script += "\n  total_steps = steps_pos + steps_neg"
    r_script += "\n  max_steps = max(c(steps_pos, steps_neg))"
    r_script += "\n  col_pos = colorRampPalette(c('#F5F5F5', '#543005'))(max_steps)"
    r_script += "\n  col_neg = colorRampPalette(c('#F5F5F5', '#003C30'))(max_steps)"
    r_script += "\n  col_pos = col_pos[1:steps_pos]"
    r_script += "\n  col_neg = col_neg[1:steps_neg]"
    r_script += "\n  shap_col = shap"
    r_script += "\n  shap_col[shap == 0] = '#F5F5F5'"
    r_script += "\n  col_idx = findInterval(shap_sqrt[shap_pos], seq(sqrt(offset_sqrt), max_pos, length.out = steps_pos), all.inside = TRUE)"
    r_script += "\n  shap_col[shap_pos] = col_pos[col_idx]"
    r_script += "\n  col_idx = findInterval(shap_sqrt[shap_neg], seq(sqrt(offset_sqrt), max_neg, length.out = steps_neg), all.inside = TRUE)"
    r_script += "\n  shap_col[shap_neg] = col_neg[col_idx]"
    r_script += "\n  species_names = gsub('_', ' ', species_names)"
    r_script += "\n  # Plot"
    r_script += "\n  h = 0.15"
    r_script += "\n  if (nfeat < 10) {"
    r_script += "\n    h = seq(0.4, 0.1, length.out = 9)"
    r_script += "\n    h = h[nfeat]"
    r_script += "\n  }"
    r_script += "\n  heights = c(h, rep((1 - 1.25 * h)/nfeat, nfeat), h/4)"
    r_script += "\n  rate_name = 'Speciation rate'"
    r_script += "\n  rate_col = 'dodgerblue'"
    r_script += "\n  if (rate_type == 'extinction') {"
    r_script += "\n    rate_name = 'Extinction rate'"
    r_script += "\n    rate_col = 'red'"
    r_script += "\n  }"
    r_script += "\n  if (rate_type == 'sampling') {"
    r_script += "\n    rate_name = 'Sampling rate'"
    r_script += "\n    rate_col = 'burlywood2'"
    r_script += "\n  }"
    r_script += "\n  layout(cbind(c(1, 2:(nfeat+2)), c(nfeat + 3, rep(nfeat + 4, nfeat), nfeat + 5)),"
    r_script += "\n         heights = heights, widths = c(0.9, 0.1))"
    r_script += "\n  # Rates per species"
    r_script += "\n  par(las = 1, mar = c(0.1, 6, 0.5, 0.1), mgp = c(5, 1, 0))"
    r_script += "\n  cex_lab = 2 / sqrt(length(feat_names))"
    r_script += "\n  cex_lab = ifelse(cex_lab > 1, 1, cex_lab)"
    r_script += "\n  y_tck = pretty(range(rates, na.rm = TRUE), n = 5)"
    r_script += "\n  plot(0, 0, type = 'n', xaxs = 'i', yaxs = 'i',"
    r_script += "\n       xlim = c(0, nspecies), ylim = range(y_tck),"
    r_script += "\n       axes = FALSE, ylab = rate_name, cex.lab = cex_lab)"
    r_script += "\n  abline(h = baseline, lty = 2, col = 'grey')"
    r_script += "\n  x = c(1:nspecies) - 0.5"
    r_script += "\n  polygon(c(x, rev(x)), c(rates[, 2], rev(rates[, 3])), col = adjustcolor(rate_col, alpha = 0.25), border = NA)"
    r_script += "\n  lines(x, rates[, 1], col = rate_col, lwd = 1.5)"
    r_script += "\n  axis(side = 2, at = y_tck, cex.axis = cex_lab)"
    r_script += "\n  # Shape values"
    r_script += "\n  species_x = c(1:nspecies) - 0.5"
    r_script += "\n  for (i in 1:nfeat) {"
    r_script += "\n    f = feat[[i]]"
    r_script += "\n    if (is.null(dim(f))) {"
    r_script += "\n      d = is_discrete(f)"
    r_script += "\n      y_tck = pretty(range(f), n = 5)"
    r_script += "\n      ylim = range(y_tck)"
    r_script += "\n      if (d) {"
    r_script += "\n        ylim = c(min(f) - 0.5, max(f) + 0.5)"
    r_script += "\n      }"
    r_script += "\n      plot(species_x, f, type = 'n', ylim = ylim,"
    r_script += "\n           xlim = c(0, nspecies), xaxs = 'i', axes = FALSE,"
    r_script += "\n           xlab = '', ylab = feat_names[i], cex.lab = cex_lab)"
    r_script += "\n      if (!d) {"
    r_script += "\n        lines(species_x, f, col = 'grey')"
    r_script += "\n        axis(side = 2, at = y_tck, cex.axis = cex_lab)"
    r_script += "\n      }"
    r_script += "\n      else {"
    r_script += "\n        axis(side = 2, at = min(f):max(f), cex.axis = cex_lab)"
    r_script += "\n      }"
    r_script += "\n      points(species_x, f, pch = 19, col = shap_col[, i])"
    r_script += "\n    }"
    r_script += "\n    else {"
    r_script += "\n      nr = nrow(f)"
    r_script += "\n      plot(0, 0, type = 'n', ylim = c(1 - 0.5, nr + 0.5),"
    r_script += "\n           xlim = c(0, nspecies), xaxs = 'i', axes = FALSE,"
    r_script += "\n           xlab = '', ylab = feat_names[i], cex.lab = cex_lab)"
    r_script += "\n      axis(side = 2, at = 1:nr, labels = feat_states[[i]], cex.axis = cex_lab)"
    r_script += "\n      for (j in 1:nr) {"
    r_script += "\n        fj = f[j, ]"
    r_script += "\n        idx = fj == 1"
    r_script += "\n        points(species_x[idx], rep(j, sum(idx)), pch = 19, col = shap_col[idx, i])"
    r_script += "\n      }"
    r_script += "\n    }"
    r_script += "\n  }"
    r_script += "\n  # Species names"
    r_script += "\n  par(mar = c(0.1, 6, 0.5, 0.1), mgp = c(3, 0.1, 0))"
    r_script += "\n  plot(0, 0, type = 'n', xlim = c(0, nspecies), ylim = c(0, 3),"
    r_script += "\n       xaxs = 'i', axes = FALSE, xlab = '', ylab = '')"
    r_script += "\n  text(x = species_x, y = 3, labels = species_names,"
    r_script += "\n       xpd = NA, srt = 35, adj = 0.965,"
    r_script += "\n       cex = 4 / sqrt(length(species_names)), font = 3)"
    r_script += "\n  # Empty plot"
    r_script += "\n  plot(0, 0, type = 'n', axes = FALSE, xlab = '', ylab = '')"
    r_script += "\n  # Legend"
    r_script += "\n  col = c(rev(col_neg), col_pos)"
    r_script += "\n  n_lead_digit = nchar(as.character(max(round(abs(shap)))))"
    r_script += "\n  par(mar = c(4, 0.5, 0.5, 0.5), mgp = c(3, 1, 0))"
    r_script += "\n  plot(0, 0, type = 'n', xlab = '', ylab = '', axes = FALSE,"
    r_script += "\n       xlim = c(0, 2), ylim = c(0, 1.05 * total_steps))"
    r_script += "\n  for (j in 1:total_steps) {"
    r_script += "\n    rect(0, j - 1, 0.7, j, border = col[j], col = col[j])"
    r_script += "\n  }"
    r_script += "\n  rect(0, 0, 0.7, total_steps, border = 'black')"
    r_script += "\n  lines(x = c(0.7, 1.0), y = rep(steps_neg, 2))"
    r_script += "\n  text(x = 0, y = 1.05 * total_steps, labels = 'Rate change', adj = c(0, 0.5))"
    r_script += "\n  a = c(1, 0.5) # right align"
    r_script += "\n  text(x = 2, y = steps_neg, labels = sprintf(paste0('%.', 3 - n_lead_digit, 'f'), 0), adj = a)"
    r_script += "\n  n_tck = 4"
    r_script += "\n  s = seq(sqrt(offset_sqrt), max(c(max_pos, max_neg)), length.out = n_tck)"
    r_script += "\n  pos = (s / s[n_tck]) * max_steps"
    r_script += "\n  pos = pos[-1]"
    r_script += "\n  s = s[-1]"
    r_script += "\n  for (i in 1:length(pos)) {"
    r_script += "\n    p = steps_neg + pos[i]"
    r_script += "\n    if (p <= total_steps) {"
    r_script += "\n      lines(x = c(0.7, 1.0), y = rep(p, 2))"
    r_script += "\n      text(x = 2, y = p, adj = a, labels = sprintf(paste0('%.', 3 - n_lead_digit, 'f'), s[i]**2))"
    r_script += "\n    }"
    r_script += "\n    p = steps_neg - pos[i]"
    r_script += "\n    if (p >= 0) {"
    r_script += "\n      lines(x = c(0.7, 1.0), y = rep(p, 2))"
    r_script += "\n      text(x = 2, y = p, adj = a, labels = sprintf(paste0('%.', 3 - n_lead_digit, 'f'), -(s[i]**2)))"
    r_script += "\n    }"
    r_script += "\n  }"
    r_script += "\n}"
    r_script += "\n"
    if not sp_taxa_shap is None:
        if len(sp_taxa_shap) > 1:
            sp_shap_trt_tbl, sp_names_features, sp_names_features_orig = get_features_for_shap_plot(mcmc_file, pkl_file,
                                                                                                    burnin, thin, 'speciation',
                                                                                                    combine_discr_features,
                                                                                                    file_transf_features)
            r_script = get_dotplot_rscript_species_shap(r_script, species_names, sp_taxa_shap, sp_consrank, sp_shap_trt_tbl,
                                                        sp_names_features, sp_names_features_orig, rate_type = 'speciation')
        else:
            r_script += "\nplot(1:5, 1:5, type = 'n', main = 'No shap values available when there is only one predictor')"
    if not ex_taxa_shap is None:
        if len(ex_taxa_shap) > 1:
            ex_shap_trt_tbl, ex_names_features, ex_names_features_orig = get_features_for_shap_plot(mcmc_file, pkl_file,
                                                                                                    burnin, thin, 'extinction',
                                                                                                    combine_discr_features,
                                                                                                    file_transf_features)
            r_script = get_dotplot_rscript_species_shap(r_script, species_names, ex_taxa_shap, ex_consrank, ex_shap_trt_tbl,
                                                        ex_names_features, ex_names_features_orig, rate_type = 'extinction')
        else:
            r_script += "\nplot(1:5, 1:5, type = 'n', main = 'No shap values available when there is only one predictor')"
    if not q_taxa_shap is None:
        if len(q_taxa_shap) > 1:
            q_shap_trt_tbl, q_names_features, q_names_features_orig = get_features_for_shap_plot(mcmc_file, pkl_file,
                                                                                                 burnin, thin, 'sampling',
                                                                                                 combine_discr_features,
                                                                                                 file_transf_features)
            r_script = get_dotplot_rscript_species_shap(r_script, species_names, q_taxa_shap, q_consrank, q_shap_trt_tbl,
                                                        q_names_features, q_names_features_orig, rate_type = 'sampling')
        else:
            r_script += "\nplot(1:5, 1:5, type = 'n', main = 'No shap values available when there is only one predictor')"
    r_script += "\ndev.off()"
    newfile.writelines(r_script)
    newfile.close()
    if platform.system() == "Windows" or platform.system() == "Microsoft":
        cmd = "cd %s & Rscript %s_%s.r" % (output_wd, name_file, suffix_pdf)
    else:
        cmd = "cd %s; Rscript %s_%s.r" % (output_wd, name_file, suffix_pdf)
    print("cmd", cmd)
    os.system(cmd)
