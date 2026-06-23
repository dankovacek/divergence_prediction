from collections import defaultdict
from scipy.stats import wasserstein_distance
import numpy as np


class EvaluationMetrics:
    def __init__(self, bitrate, log_x=None, data=None, min_measurable_log_uar=None):
        """
        Initialize the EvaluationMetrics class with station data and configuration.
        
        Parameters
        ----------
        stn : str
            Station identifier.
        df : pd.DataFrame
            DataFrame containing observed discharge data.
        baseline_log_x : np.ndarray
            Log-transformed grid of flow values.
        """
        if data is not None:
            # inherit the attributes from StationData
            for k, v in data.__dict__.items():
                setattr(self, k, v)
            # adjust the log_x and lin_x according to station-specific zero flow equivalent UAR
            # to avoid evaluating FDC residuals on values we have defined as indistinguishable from zero
            assert hasattr(self, 'log_x'), "data must have attribute 'log_x'"
            assert hasattr(self, 'zero_equiv_uar'), "data must have attribute 'zero_equiv_uar'"
        else:
            self.log_x = log_x
            self.lin_x = np.exp(self.log_x)
            assert min_measurable_log_uar is not None, "min_measurable_log_uar must be provided (with log_x) if data is None"
            self.min_measurable_log_uar = min_measurable_log_uar
            self.min_measurable_uar = np.exp(min_measurable_log_uar)

        self.bitrate = bitrate

        self.metric_limits = {
            'kld': 0.001,
            'emd': 0.05,  # this is L/s/km^2, 0.05 is very small.
            'nse': 1 - 0.001, # flipped because 1.0 is perfect
            'kge': 1 - 0.001, # flipped because 1.0 is perfect
            'mean_error': 0.01,
            'pct_vol_bias': 0.01,
            "mean_abs_rel_error": 0.01,
            'rmse': 0.01,
        }

    def _compute_kl_divergence(self, p, q):
        """Compute the KL divergence between two probability distributions."""
        p = p.astype(float)
        q = q.astype(float)
        mask = (p > 0) & (q > 0)

        with np.errstate(divide='ignore', invalid='ignore'):
            log_term = np.zeros_like(p)
            log_term[mask] = np.log2(p[mask] / q[mask])
            terms = np.zeros_like(p)
            terms[mask] = p[mask] * log_term[mask]

        bad_idx = ~np.isfinite(terms)
        if np.any(bad_idx):
            print("[DEBUG] Invalid values in KL divergence computation:")
            for i in np.flatnonzero(bad_idx):
                print(
                    f"  i={i}, p={p[i]:.3e}, q={q[i]:.3e}, "
                    f"p/q={p[i]/q[i]:.3e}, log(p/q)={log_term[i]:.3e}, "
                    f"term={terms[i]:.3e}"
                )

        return np.sum(terms[mask])  # Only use valid terms


    def _get_log_bin_widths(self):
        """Return bin widths aligned with self.log_x for PMF/PDF conversion in log space."""
        log_x = np.asarray(self.log_x, dtype=float)
        assert log_x.ndim == 1, "log_x must be one-dimensional"
        assert log_x.size > 1, "log_x must contain at least two points"
        log_w = np.diff(np.sort(log_x))
        return np.append(log_w, log_w[-1])
    

    def _compute_emd(self, p, q):
        assert np.all(q >= 0), f'min q_i < 0: {np.min(q)}'

        emd = wasserstein_distance(self.lin_x, self.lin_x, p, q)
        return float(round(emd, 4)) #, {'bias': None, 'unsupported_mass': None, 'pct_of_signal': None}
    

    def _compute_volumetric_pct_bias_from_fdc(self, p, q):
        """
        Compute volumetric bias between two PMFs p (observed) and q (simulated),
        defined over discrete support (flow) x.
        
        Parameters:
        - p: observed PMF (sum to 1)
        - q: simulated PMF (sum to 1)
        - x: bin centers corresponding to flow magnitudes
        
        Returns:
        - Relative bias in expected flow volume
        """
        # if p is a cdf over 1 to 99 percentiles, then compute the expected volume of p
        E_p = np.sum(p * 0.01)
        E_q = np.sum(q * 0.01)
        return (E_p - E_q) / E_p, E_p
    

    def _compute_volumetric_pct_bias_from_pmfs(self, baseline_pmf, pmf_est):
        """
        Compute volumetric bias between two PMFs p (observed) and q (simulated),
        defined over discrete support (flow) x.
        
        Parameters:
        - obs: observed flow values
        - sim: simulated flow values

        Returns:
        - Relative bias in expected flow volume
        """
        assert np.isclose(np.sum(baseline_pmf), 1), f"baseline_pmf does not sum to 1: {np.sum(baseline_pmf)}"
        assert np.isclose(np.sum(pmf_est), 1), f"pmf_est does not sum to 1: {np.sum(pmf_est)}"
        E_p = np.sum(self.lin_x * baseline_pmf)
        E_q = np.sum(self.lin_x * pmf_est)

        return (E_p - E_q) / E_p, E_p


    def _compute_nse(self, obs, sim):
        """Compute the Nash-Sutcliffe Efficiency (NSE) between observed and simulated values."""
        assert not np.isnan(obs).any(), f'NaN values in obs: {obs}'
        assert not np.isnan(sim).any(), f'NaN values in sim: {sim}'
        # Compute the NSE
        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - obs.mean()) ** 2)
        nse = 1 - (numerator / denominator)
        return nse


    def _compute_RMSE(self, obs, sim):
        """Compute the Root Mean Square Error (RMSE) between observed and simulated values."""
        assert not np.isnan(obs).any(), f'NaN values in obs: {obs}'
        assert not np.isnan(sim).any(), f'NaN values in sim: {sim}'
        # Compute the RMSE
        return np.sqrt(np.mean((obs - sim) ** 2))


    def _compute_KGE(self, obs, sim):
        """Compute the Kling-Gupta Efficiency (KGE) between observed and simulated values."""
        assert not np.isnan(obs).any(), f"NaN values in obs: {obs}"
        assert not np.isnan(sim).any(), f"NaN values in sim: {sim}"
        if obs.size == 0:
            return np.nan
        # Compute the KGE
        r = np.corrcoef(obs, sim)[0, 1]
        alpha = sim.mean() / obs.mean()
        beta = sim.std() / obs.std()
        return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        

    def _compute_pinball_loss(self, p, q, quantile=0.5):
        """Compute the Pinball Loss for a given quantile between observed and simulated values."""
        assert not np.isnan(p).any(), f"NaN values in obs"
        assert not np.isnan(q).any(), f"NaN values in sim"
        errors = p - q
        loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
        return np.nanmean(loss) 
    

    def _evaluate_fdc_metrics_from_pmf(self, pmf_est, baseline_obs_pmf, min_measurable_log_uar=None, epsilon=1e-2):
        """
        Evaluate RMSE, relative error, NSE, and KGE between two FDCs represented by discrete PMFs.
        Note these are evaluated over the log_x which is set in the context.

        Parameters
        ----------
        pmf_est : np.ndarray
            Discrete PMF representing the estimated FDC, over `log_x`.
        log_x : np.ndarray
            Grid of log-transformed flow values corresponding to PMF bins.

        Returns
        -------
        dict
            Dictionary of RMSE, RelativeError, NSE, and KGE computed over p=1,...,99 quantiles.
        """
        assert (
            len(baseline_obs_pmf) == len(pmf_est) == len(self.log_x)
        ), f"Array length mismatch, {len(baseline_obs_pmf)}, {len(pmf_est)}, {len(self.log_x)}"

        pmf_est /= np.sum(pmf_est)
        baseline_obs_pmf /= np.sum(baseline_obs_pmf)

        assert np.isclose(np.sum(pmf_est), 1, atol=1e-3), f'sum Estimated P = {np.sum(pmf_est)}'
        assert np.isclose(np.sum(baseline_obs_pmf), 1, atol=1e-3), f'sum Baseline P = {np.sum(baseline_obs_pmf)}'

        # Compute CDFs
        cdf_true = np.cumsum(baseline_obs_pmf)
        cdf_true /= cdf_true[-1]
        cdf_est = np.cumsum(pmf_est)
        cdf_est /= cdf_est[-1]
        
        assert np.isfinite(cdf_true).all(), "Non-finite values in cdf_true"
        assert np.diff(cdf_true).sum() > 0, f"cdf_true has no spread {list(baseline_obs_pmf)}"

        # Equal-spaced probabilities for quantile function interpolation
        probs = np.linspace(epsilon, 1 - epsilon, 2**self.bitrate)

        # Interpolate inverse CDF (log-flow values at given probabilities)
        # set the left value to the minimum measurable uar (zero-equivalent)
        # to avoid basing metrics on flows we assume cannot be validated by measurement
        # this is important for metrics based on log-space flows
        if min_measurable_log_uar is not None:
            min_uar, min_log_uar = np.exp(min_measurable_log_uar), min_measurable_log_uar
        else:
            assert hasattr(self, 'min_measurable_uar'), "min_measurable_uar must be set in the instance or provided as argument"
            min_uar, min_log_uar = self.min_measurable_uar, self.min_measurable_log_uar

        log_q_true = np.interp(
            probs, cdf_true, self.log_x, left=min_log_uar, right=self.log_x[-1]
        )
        log_q_est = np.interp(
            probs, cdf_est, self.log_x, left=min_log_uar, right=self.log_x[-1]
        )
        linear_q_true = np.interp(
            probs, cdf_true, self.lin_x, left=min_uar, right=np.exp(self.log_x[-1])
        )
        linear_q_est = np.interp(
            probs, cdf_est, self.lin_x, left=min_uar, right=np.exp(self.log_x[-1])
        )
        # # set a small minimum flow value to avoid issues with zero value division
        linear_q_true = np.clip(linear_q_true, a_min=min_uar, a_max=None)
        linear_q_est = np.clip(linear_q_est, a_min=min_uar, a_max=None)

        if np.any(linear_q_est <= 0):
            print("Min:", np.nanmin(linear_q_est))
            print("Any negative?", np.any(linear_q_est < 0))
            print("Any NaN?", np.any(np.isnan(linear_q_est)))
            print("dtype:", linear_q_est.dtype)

        assert np.all(linear_q_true >= 0), "Zero or negative values in q_true — invalid for relative error"
        assert np.all(linear_q_est >= 0),  "Zero or negative values in q_est — unexpected for flow"

        # Metrics
        residuals = linear_q_est - linear_q_true
        abs_err = np.abs(residuals)

        # linear
        pct_vol_bias = 100.0 * np.sum(residuals) / np.sum(linear_q_true) # PBIAS
        norm_abs_err = np.sum(abs_err) / np.sum(linear_q_true) # NAE
        ve = 1 - norm_abs_err # volume efficiency
        mean_abs_pct_error = 100.0 * np.mean(abs_err / linear_q_true) # MAPE

        # square (but log-space)
        rmse = np.sqrt(np.mean((log_q_true - log_q_est) ** 2))
        nse = self._compute_nse(log_q_true, log_q_est)
        kge = self._compute_KGE(log_q_true, log_q_est)

        # other, combined
        pinball_loss_50 = self._compute_pinball_loss(linear_q_true, linear_q_est, quantile=0.5)
        kld = self._compute_kl_divergence(baseline_obs_pmf, pmf_est)
        emd = self._compute_emd(baseline_obs_pmf, pmf_est)

        return {
            "pct_vol_bias": float(pct_vol_bias), # this is PBIAS (labeled RB in some notebooks)
            "mean_abs_pct_error": float(mean_abs_pct_error), # this is MAPE
            "norm_abs_error": float(np.mean(norm_abs_err)), 
            "rmse": float(rmse), 
            "nse": float(nse), 
            "kge": float(kge), 
            "ve": float(ve), 
            "pb_50": float(pinball_loss_50), 
            "kld": float(kld), 
            "emd": float(emd), 
        }