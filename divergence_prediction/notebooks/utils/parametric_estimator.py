import os
import pandas as pd
import numpy as np

import jax.numpy as jnp
from scipy.stats import norm, laplace, genextreme

class ParametricFDCEstimator:
    def __init__(self, ctx, data, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.ctx = ctx
        self.data = data
        self.target_stn = self.data.target_stn
        # self.data = data
        self.predicted_param_dict = self.ctx.predicted_param_dict
        self.predicted_param_df = pd.DataFrame(self.predicted_param_dict).T
        self.include_random_test = True

    
    def _compute_lognorm_pmf(self, mu, sigma):

        mu = max(mu, self.data.zero_equiv_log_uar_actual)  # ensure mu is at least the zero-flow threshold
        # create a function for the CDF
        F = lambda y: norm(loc=mu, scale=sigma).cdf(y)
                
        # initialize a pmf array of zeros
        pmf = np.zeros(len(self.data.log_edges_extended) - 1, dtype=float)

        i0 = self.data.zero_bin_index # zero flow index on log_edges_extended
        # get the cdf value at the zero-equivalent flow bin (0-->threshold)
        pmf[0] = max(F(self.data.pos_edges[0]) - F(self.data.log_edges_extended[0]), 0.0)

        # positive bin probabilities (threshold --> max)
        pmf[i0+1:] = F(self.data.pos_edges[1:]) - F(self.data.pos_edges[:-1])
        pmf /= pmf.sum()  # normalize raw PMF
        return pmf
    

    def _compute_GEV_pmf(self, xi, mu, sigma):
        # assert values are within the valid range for GEV
        xi = max(xi, -0.5 + 1e-12)  # clip xi to avoid numerical issues
        sigma = max(sigma, 1e-12)  # ensure sigma is positive
        # pdf = genextreme.pdf(self.data.log_x, xi, loc=mu, scale=sigma)
        cdf_base = np.zeros_like(self.data.log_edges_extended)
        # assign probability for the zero-equivalent bin
        # equal to the cdf value at the threshold
        cdf_base[0] = genextreme.cdf(self.zero_flow_log_uar, xi, loc=mu, scale=sigma)
        cdf_base[self.zero_flow_idx:] = genextreme.cdf(self.data.log_edges_extended[self.zero_flow_idx:], loc=mu, scale=sigma)
        # pdf /= jnp.trapezoid(pdf, x=self.data.log_edges_extended)
        # pmf = pdf * self.data.log_w
        pmf = np.diff(cdf_base)
        pmf /= pmf.sum()  # normalize raw PMF
        return pmf


    def _estimate_from_mle(self):
        log_mu = self.predicted_param_dict[self.target_stn]['log_uar_mean_actual']
        log_sigma = self.predicted_param_dict[self.target_stn]['log_uar_std_actual']
        return self._compute_lognorm_pmf(log_mu, log_sigma)
    

    def _estimate_from_predicted_log_params(self):
        mu = self.predicted_param_dict[self.target_stn]['log_uar_mean_predicted']
        sigma = self.predicted_param_dict[self.target_stn]['log_uar_std_predicted']
        return self._compute_lognorm_pmf(mu, sigma)
        
    
    def _estimate_from_predicted_linear_mom(self):
        mean_x = max(self.predicted_param_dict[self.target_stn]['uar_mean_predicted'], 1e-4)
        sd_x = self.predicted_param_dict[self.target_stn]['uar_std_predicted']
        v = np.log(1 + (sd_x / mean_x) ** 2)
        assert mean_x > 0, f'Mean must be positive for lognormal distribution, got {mean_x}'
        mu = np.log(mean_x) - 0.5 * v
        return self._compute_lognorm_pmf(mu, np.sqrt(v))
    

    def _estimate_LN_from_randomly_drawn_params(self):
        # randomly draw from the predicted parameters
        random_idx = np.random.choice(len(self.predicted_param_df))
        random_stn_idx = self.predicted_param_df.index[random_idx]
        mu_random =self.predicted_param_dict[random_stn_idx]['log_uar_mean_predicted']
        sigma_random = self.predicted_param_dict[random_stn_idx]['log_uar_std_predicted']
        return self._compute_lognorm_pmf(mu_random, sigma_random)


    def run_estimators(self):

        results = {}
        fns = [
            self._estimate_from_mle, 
            self._estimate_from_predicted_log_params,
            self._estimate_from_predicted_linear_mom, 
            self._estimate_LN_from_randomly_drawn_params,
            # self._estimate_from_observed_lmoments_gev,
            # self._estimate_from_predicted_lmoments_gev, 
            # self._estimate_LMOM_gev_from_randomly_drawn_params
            ]
        
        labels = ['MLE', 'PredictedLog', 'PredictedMOM', 
                  #'ObsLMomentsGEV', 'PredictedLMomentsGEV', 'LMomentsGEVRandomDraw',
                  ]
        if self.include_random_test:
            labels += ['RandomDraw']
        for fn, label in zip(fns, labels):
            
            pmf = fn()
            
            if 'Moments' in label:
                # assert no nan values in the pmf
                assert not np.any(np.isnan(pmf)), f'PMF contains NaN values for {label}: {pmf[:10]}'

            # add a small amount of uncertainty to avoid zero probabilities
            pmf_prior_adjusted = self.data._compute_adjusted_distribution_with_mixed_uniform(pmf)

            results[label] = {'prior_adjusted_pmf': pmf_prior_adjusted.tolist(), 'pmf': pmf.tolist()}
            # set the minimum measurable uar to the left edge of the zero-equivalent bin
            # so that it reflects the discrete binning used in the PMF
            estimation_metrics = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(pmf_prior_adjusted, self.data.baseline_pmf, min_measurable_log_uar=self.data.min_measurable_log_uar)
            results[label]['eval'] = estimation_metrics

            # compute the bias
            final_bias_metrics = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(pmf, pmf_prior_adjusted, min_measurable_log_uar=self.data.min_measurable_log_uar)
            results[label]['bias'] = final_bias_metrics
                
        # compute the bias from the eps
        return results