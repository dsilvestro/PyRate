import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=3)
from scipy import special

class ADE_slice:
    def __init__(self,
                 nbins=1000,
                 xLim = 50,
                 min_dt=0.0000001
                 ):
        # integration settings //--> Add to command list
        self._nbins = nbins
        self._xLim = xLim
        self._x_bins = np.linspace(min_dt, xLim, nbins)
        self._x_bin_size = self._x_bins[1] - self._x_bins[0]

    # ADE model
    def avg_longevity_to_scale(self, shape, long):
        return long / special.gamma(1 + 1 / shape)

    def prob_no_fossils(self, q, x):
        return np.exp(-q * x)

    def prob_fossils(self, q, x):
        return 1 - self.prob_no_fossils(q, x)

    def cdf_WR(self, w_shape, w_scale, x):
        return (x / w_scale) ** (w_shape)

    def log_wr(self, t, w_shape, w_scale):  # return np.log extinction rate at time t based on ADE model
        return np.log(w_shape / w_scale) + (w_shape - 1) * np.log(t / w_scale)

    def log_wei_pdf(self, x, w_shape, w_scale):  # np.log pdf Weibull
        return np.log(w_shape / w_scale) + (w_shape - 1) * np.log(x / w_scale) - (x / w_scale) ** w_shape

    def wei_pdf(self, x, w_shape, w_scale):  # pdf Weibull
        return w_shape / w_scale * (x / w_scale) ** (w_shape - 1) * np.exp(-(x / w_scale) ** w_shape)

    def pdf_w_poi(self, w_shape, w_scale, q, x):  # np.exp np.log Weibull + Q_function
        return np.exp(self.log_wei_pdf(x, w_shape, w_scale) + np.log(self.prob_fossils(q, x)))

    def pdf_w_poi_nolog(self, w_shape, w_scale, q, x):  # Weibull + Q_function
        return self.wei_pdf(x, w_shape, w_scale) * (self.prob_fossils(q, x))

    def cdf_weibull(self, x, w_shape, w_scale):  # Weibull cdf
        return 1 - np.exp(-(x / w_scale) ** w_shape)

    def integrate_pdf(self, P, v, d, upper_lim):
        if upper_lim == 0:
            return 0
        else:
            return np.sum(P[v < upper_lim]) * d

    def get_mu_ade(self, sigma, w_shape, w_scale):
        return w_shape / w_scale * (sigma / w_scale) ** (w_shape - 1)

    # ASSUMING SAMPLED OCCURRENCES IN BIN
    def get_const(self, w_shape, w_scale):
        # partial integral (xLim => Inf) via cdf_weibull
        return 1 - self.cdf_weibull(self._xLim, w_shape, w_scale)

    # 1. lik of fully included species
    def get_lik_full_species(self, sigma, w_shape, w_scale, q):
        lik1 = self.log_wei_pdf(sigma, w_shape=w_shape, w_scale=w_scale) + np.log(self.prob_fossils(q, sigma))
        # numerical integration + analytical for right tail
        p = self.pdf_w_poi(w_shape, w_scale, q, self._x_bins)  # partial integral (0 => xLim) via numerical integration
        lik2 = np.log(np.sum(p) * self._x_bin_size + self.get_const(w_shape, w_scale))
        return np.sum(lik1 - lik2)

    # 2. lik of a species that originates in bin but doesn't go extinct (in bin)
    def get_lik_right_trunc_species(self, sigma, w_shape, w_scale, q):
        p = self.pdf_w_poi(w_shape, w_scale, q, self._x_bins)  # partial integral (0 => xLim) via numerical integration
        lik2 = np.log(np.sum(p) * self._x_bin_size + self.get_const(w_shape, w_scale))
        c = self.get_const(w_shape, w_scale)
        lik_extant = np.array(
            [np.log(np.sum(p[self._x_bins > s]) * self._x_bin_size + c) for s in sigma])  # P(x > ts | w_shape, w_scale, q)
        return np.sum(lik_extant - lik2)
        # lik_extant = np.array([np.log(np.sum(p[x_bins < s]) * x_bin_size + const_int) for s in sigma]) # P(x > ts | w_shape, w_scale, q)
        # return np.sum(np.log(1 - np.exp(lik_extant - lik2)))

    # 3. lik of left-truncated species: death likelihood, assuming sampled lineage
    def get_lik_left_trunc_species(self, age_at_bin_start, age_at_extinction, w_shape, w_scale, te_lt):
        # ext rate at extinction event (if lineage goes extinct)
        lik1 = np.log(self.get_mu_ade(age_at_extinction, w_shape, w_scale)) * (te_lt > 0)
        mu_t = self.get_mu_ade(self._x_bins, w_shape, w_scale)
        lik2_1 = np.array([(np.sum(mu_t[self._x_bins <= s]) * self._x_bin_size) for s in age_at_extinction])
        lik2_0 = np.array([(np.sum(mu_t[self._x_bins <= s]) * self._x_bin_size) for s in age_at_bin_start])
        # ext rate integrated along waiting time within bin
        lik2 = lik2_1 - lik2_0
        return np.sum(lik1 - lik2)

    def get_corrected_q(self, mean_sigma, q_ztp):
        lam_ZTP = q_ztp * mean_sigma  # expected n. of fossils per species
        lam = np.linspace(lam_ZTP * 0.05, lam_ZTP, 1000)
        lam_ZTP_vec = lam / (1 - np.exp(-lam))
        indx = np.argmin(abs(lam_ZTP_vec - lam_ZTP))
        return lam[indx] / mean_sigma


    def get_full_likelihood(self,
                            dat,
                            w_shape=None, w_scale=None, q=None,
                            prms=None):
        if prms is not None:
            w_shape = prms[0]
            w_scale = self.avg_longevity_to_scale(prms[0], prms[1]) # sample avg longevity
            q = prms[2]

        mean_sigma = np.mean(dat['mean_length_in_bin'])
        corrected_q = self.get_corrected_q(mean_sigma, q)

        l1 = self.get_lik_full_species(
            dat['sampled_sigmas'][dat['in_bin']], w_shape, w_scale, corrected_q)
        l2 = self.get_lik_right_trunc_species(
            dat['sampled_sigmas'][dat['rg_trunc']], w_shape, w_scale, corrected_q)
        l3 = self.get_lik_left_trunc_species(
            dat['sampled_age_at_bin_start'][dat['lf_trunc']],
            dat['sampled_sigmas'][dat['lf_trunc']],
            w_shape, w_scale,
            dat['sampled_te'][dat['lf_trunc']])
        l4 = np.log(q) * len(dat['foss_age_in_bin']) - q * np.sum(dat['length_in_bin'])

        return l1 + l2 + l3 + l4


class ADE_simulator:
    def __init__(self, seed=None):
        self.rg = np.random.default_rng(seed)

    def simulate(self,
                 root=12,
                 q = .25,
                 w_shape = .75,
                 w_scale = 10,
                 t0 = 8.,
                 t1 = 0,
                 n_taxa = 1000,
                 min_age_ts=0.
                 ):
        # 1. longevities
        sigmas = self.rg.weibull(w_shape, n_taxa) * w_scale
        ts = np.sort(self.rg.uniform(min_age_ts, root, len(sigmas)))[::-1]
        te = np.maximum(ts - sigmas, t1)
        # drop things prior to t0
        ind = np.where(te < t0)[0]
        sigmas = sigmas[ind]
        ts = ts[ind]
        te = te[ind]

        # truncate living species duration from initial weibull
        sigmas_t = np.minimum(sigmas, ts - te)

        # age of left truncated species
        age_at_bin_start = np.maximum(0, ts - t0)

        # 2. records
        n_foss = self.rg.poisson(q * sigmas_t)
        foss_age_list = []
        for i in range(len(n_foss)):
            if n_foss[i] > 0:
                foss_ages = ts[i] - self.rg.random(int(n_foss[i])) * sigmas_t[i]
                foss_age_list = foss_age_list + list(foss_ages)

        foss_age_list = np.array(foss_age_list)

        # 3. drop unsampled species
        sampled_age_at_bin_start = age_at_bin_start[n_foss > 0]
        sampled_sigmas = sigmas_t[n_foss > 0]
        sampled_n_foss = n_foss[n_foss > 0]
        sampled_ts = ts[n_foss > 0]
        sampled_te = te[n_foss > 0]

        # fully in bin
        in_bin = np.where(((sampled_age_at_bin_start == 0) * (sampled_te > 0)) > 0)[0]
        # right truncated (ts in bin and te = 0)
        rg_trunc = np.where(((sampled_te == 0) * (sampled_age_at_bin_start == 0)) > 0)[0]
        # np.where(((sampled_te == 0) * (sampled_age_at_bin_start == 0)) > 0)[0]

        # left trunc (ts before bin start, te >= 0)
        lf_trunc = np.where(((sampled_age_at_bin_start > 0)) > 0)[0]

        sim_res = {'in_bin': in_bin,
                   'rg_trunc': rg_trunc,
                   'lf_trunc': lf_trunc,
                   'sampled_age_at_bin_start': sampled_age_at_bin_start,
                   'sampled_sigmas': sampled_sigmas,
                   'sampled_ts': sampled_ts,
                   'sampled_te': sampled_te,
                   'sampled_n_foss': sampled_n_foss,
                   'foss_age_list': foss_age_list,
                   'foss_age_in_bin': foss_age_list[foss_age_list <= t0],
                   'length_in_bin': sampled_sigmas - sampled_age_at_bin_start,
                   'true_sigmas': sigmas_t,
                   'mean_sampled_sigmas': np.mean(sampled_sigmas),
                   'mean_length_in_bin': np.mean(sampled_sigmas - sampled_age_at_bin_start)
                   }


        return sim_res


class MCMC():
    def __init__(self, seed,
                 iterations=10000,
                 sampling_f=100,
                 print_f=100,
                 likelihood_f=None,
                 prior_f=None,
                 proposal_window=None
                 ):
        self.rg = np.random.default_rng(seed)
        self.iterations = iterations
        self.sampling_f = sampling_f
        self.print_f = print_f
        self.likelihood_f = likelihood_f
        self.prior_f = prior_f
        self.proposal_window = proposal_window


    def multiplier_proposal_vector(self, q, f=1):
        if self.proposal_window is not None:
            d = self.proposal_window
        else:
            d = 1.05
        ff = self.rg.binomial(1, f, q.shape)
        u = self.rg.random(q.shape)
        l = 2 * np.log(d)
        m = np.exp(l * (u - .5))
        m[ff == 0] = 1.
        new_q = q * m
        U = np.sum(np.log(m))
        return new_q, U

    def run(self, data, prm=None):

        posterior_samples = []
        prior = self.prior_f(prm)
        lik = self.likelihood_f(data, prms=prm)
        posterior = prior + lik
        print('posterior:', posterior, "lik:", lik, "prior:", prior)

        for it in range(self.iterations):
            prm_prime, u = self.multiplier_proposal_vector(prm)
            prior_prime = self.prior_f(prm_prime)
            lik_prime = self.likelihood_f(data, prms=prm_prime)

            if np.log(self.rg.random()) <= (prior_prime + lik_prime) - posterior + u:
                prm = prm_prime + 0
                prior = prior_prime + 0
                lik = lik_prime + 0
                posterior = prior_prime + lik_prime

            if it % self.sampling_f == 0:
                p = [it, posterior, lik, prior] + list(prm)
                posterior_samples.append(p)
            if it % self.print_f == 0:
                print(it, np.array([posterior, lik, prior]), "prms:", prm)

        return posterior_samples



def exp_prior_f(prm):
    return np.sum(np.log(0.1) - prm * 0.1)

def ade_prior_f(prm):
    return exp_prior_f(prm[1:]) + scipy.stats.norm.logpdf(np.log(prm[0]),0, 2)


if __name__ == '__main__':
    seed = 1234
    ade_sim = ADE_simulator(seed=1234)
    ade_model = ADE_slice()

    root = 12
    min_age_ts = 0
    q = .1
    w_shape = 1.75
    # w_scale = 2.5
    # w_scale * scipy.special.gamma(1 / (1 + w_shape))
    avg_longevity = 3
    w_scale = avg_longevity / scipy.special.gamma(1 + 1 / w_shape)


    t0 = 8.
    t1 = 0
    n_taxa = 1000
    # ---
    ade_data = ade_sim.simulate(min_age_ts=min_age_ts,
                                root=root,
                                w_shape=w_shape,
                                w_scale=w_scale,
                                q=q,
                                t0=t0,
                                t1=t1,
                                n_taxa=n_taxa)
    range_w_shapes = [0.2, 0.5, 0.75, 1., 1.5, 2., 3., 5.]
    ll = np.array([ade_model.get_full_likelihood(ade_data, w_shape=i, w_scale=10, q=0.25) for i in range_w_shapes])
    print(np.vstack((range_w_shapes, ll)).T)
    # ---

    init_prm = np.array([1., np.mean(ade_data['mean_length_in_bin']), q])
                         # np.sum(ade_data['sampled_n_foss']) / np.sum(ade_data['sampled_sigmas'])])

    mcmc = MCMC(seed=1234,
                likelihood_f=ade_model.get_full_likelihood,
                prior_f=ade_prior_f,
                proposal_window=np.array([1.05, 1.05, 1.05]),
                iterations=20000,
                sampling_f=100,
                print_f=1000)
    post = mcmc.run(ade_data, prm=init_prm)


    post_pd = pd.DataFrame(post)
    post_pd.columns = ["iteration", "posterior", "lik", "prior", "w_shape", "longevity", "q_ztp"]
    post_pd["w_scale"] = ade_model.avg_longevity_to_scale(post_pd["w_shape"], post_pd["longevity"])
    corrected_q_post = [ade_model.get_corrected_q(ade_data['mean_sampled_sigmas'], q_ztp) for q_ztp in post_pd['q_ztp']]
    post_pd["corrected_q"] = corrected_q_post

    wd = "./ADE_sliced"
    os.makedirs(wd, exist_ok=True)
    post_pd.to_csv(os.path.join(wd, 'posterior.log'), index=False, sep='\t')
