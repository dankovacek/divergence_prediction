from dataclasses import dataclass
# from collections import defaultdict
# from scipy.stats import laplace
# import xarray as xr

import numpy as np
import pandas as pd
from utils.kde_estimator import KDEEstimator
# import jax.numpy as jnp
# from scipy.stats import wasserstein_distance

@dataclass
class StationData:
    def __init__(self, context, stn):
        self.ctx = context
        self.target_stn = stn
        self.attr_gdf = context.attr_gdf
        self.predicted_param_dict = context.LN_param_dict[stn]
        # self.n_grid_points = context.n_grid_points
        # self.min_flow = context.min_flow # don't allow flows below this value
        # self.divergence_measures = context.divergence_measures
        # self.met_forcings_folder = context.LSTM_forcings_folder
        self.LSTM_ensemble_result_folder = context.LSTM_ensemble_result_folder

        self.target_da = float(self.attr_gdf[self.attr_gdf['official_id'] == stn]['drainage_area_km2'].values[0])
        self.complete_year_dict = self.ctx.complete_year_stats
        
        self._load_baseline_distribution_dfs()

        self.zero_equivalent_threshold = self.ctx.zero_flow_threshold

        self.zero_equiv_uar = float(1000.0 * self.zero_equivalent_threshold / self.target_da)
        self.zero_equiv_log_uar_actual = np.log(self.zero_equiv_uar)
        self._set_zero_flow_edges()

        self._initialize_station(stn)
        
        # set the measures to check for excessive influence
        self.error_check_metrics = ['mean_error', 'pct_vol_bias', 'kld', 'rmse']

        # set the maximum allowable perturbance of Q
        self.delta = context.delta

        # set the site-specific zero-flow equivalent UAR
        self.zero_equiv_uar = 1000.0 * context.zero_flow_threshold / self.target_da
        self.zero_equiv_log_uar = np.log(self.zero_equiv_uar)

        
    def _load_baseline_distribution_dfs(self):
        # load the baseline observed PMF for the target station
        self.baseline_pmf_df = self.ctx.baseline_obs_pmf_df.copy()
        # self.baseline_pdf_df = self.ctx.baseline_obs_pdf_df.copy()
        if self.ctx.regularization_type == 'kde':
            self.baseline_pmf_df = self.ctx.baseline_kde_pmf_df.copy()
            # self.baseline_pdf_df = self.ctx.baseline_kde_pdf_df.copy()

        self.log_x = self.baseline_pmf_df.index.values # index is same between kde and discrete
        self.lin_x = np.exp(self.log_x)
        self.log_edges = np.linspace(np.log(self.ctx.global_min_uar), np.log(self.ctx.global_max_uar), 2**self.ctx.bitrate)
        self.left_log_edge = self.log_edges[0] - (self.log_edges[1] - self.log_edges[0]) 
        self.log_edges_extended = np.array([self.left_log_edge] + self.log_edges.tolist())
        self.log_x_extended = 0.5 * (self.log_edges_extended[:-1] + self.log_edges_extended[1:])
        self.linear_edges_extended = np.exp(self.log_edges_extended)
        self.log_w = np.diff(self.log_edges_extended)  # bin widths in log space
    
    
    def load_baseline_distributions(self):
        """
        Set the baseline distribution for the target station.
        This is used to compare the simulated distributions against the observed distribution.
        """
        assert np.isclose(np.sum(self.baseline_pmf), 1), f'Baseline PMF for {self.target_stn} does not sum to 1: {np.sum(self.baseline_pmf)}'

        self.baseline_pmf = self.baseline_pmf_df[self.target_stn].values
        # self.baseline_pdf = self.baseline_pdf_df[self.target_stn].values
        # assert all widths are positive and approx equal
        assert np.all(self.log_w > 0), f'Baseline PMF for {self.target_stn} has non-positive bin widths'
        
        # compute the PDF from the PMF given the linear grid (index of the pdf_df)
        # pdf_area = np.trapezoid(self.baseline_pdf, x=self.log_w)
        # self.baseline_pdf /= pdf_area  # normalize the PDF to sum to 1
        # msg = f'Baseline PDF for {self.target_stn} does not sum to 1: {pdf_area}'
        # assert np.isclose(np.trapezoid(self.baseline_pdf, x=self.log_w), 1), msg


    def retrieve_and_preprocess_timeseries_discharge(self, stn):
        """
        Retrieve and preprocess timeseries discharge data for a given station.
        The zero flow values reflect a threshold below which discharge 
        is considered indistinguishable from zero.  
        We want to flag the zero flows such that we can handle them separately.
        
        Parameters:

        stn (str): Official ID of the station.
        zero_flow_threshold (float): Threshold below which discharge is considered zero flow.

        Returns:
            pd.DataFrame: Preprocessed discharge timeseries for the station.
        """
        hit = self.ctx._cache_get(stn)
        if hit is None:
            wid = self.ctx.official_id_dict[stn]
            da  = float(self.ctx.da_dict[stn])
            # read one column, downcast to float32
            x = self.ctx.ds["discharge"].sel(watershed=str(wid)).astype("float32").values
            m = np.isfinite(x)
            x = x[m]
            t = self.ctx._time_index[m]
            # set information related to this catchment's zero-flow threshold
            log_zf = np.log(1000.0 * self.ctx.zero_flow_threshold / da)
            zero_bin_index = np.digitize(log_zf, self.log_edges_extended, right=True)
            min_measurable_uar = np.exp(self.log_x_extended[zero_bin_index])
            # print(f'Station {stn} zero bin index: {zero_bin_index}, log_zf: {log_zf:.3f}  min_measurable_uar: {min_measurable_uar:.6f}')
            zf = float(self.ctx.zero_flow_threshold)
            zflag = bool((x <= zf).any())
            # store raw column + metadata (no DataFrame in cache)
            self.ctx._cache_put(stn, (x, zflag, da, t, zero_bin_index, min_measurable_uar))
            discharge, zflag, da, t, zero_bin_index, min_measurable_uar = x, zflag, da, t, zero_bin_index, min_measurable_uar
        else:
            discharge, zflag, da, t, zero_bin_index, min_measurable_uar = hit

        # Build DF only for the caller
        df = pd.DataFrame({"discharge": discharge}, index=t)
        df[f"{stn}_uar"] = 1000.0 * discharge / da
        # clip to the "minimum measurable UAR" if this call is retrieving
        # data for a donor station in a KNN context,
        # to prevent the model from incorporating information that
        # is not practically measurable
        df[f'{stn}_uar_clipped'] = df[f'{stn}_uar'].clip(lower=min_measurable_uar)
        return df, zflag
    
    
    def filter_complete_hydrological_years(self, df, da, min_days=20):
        s = df.copy().sort_index()  # daily discharge series

        if not s.index.is_unique:
        # drop dup timestamps once, at the source
            s = s[~s.index.duplicated(keep="first")]

        # count months
        m_count = s["discharge"].resample("MS").count()        

        cal_good = (
        m_count.ge(min_days)                       # month ok?
            .groupby(pd.Grouper(freq="YE-DEC")) # to calendar year
            .sum()
            .eq(12)                             # 12 good months
        )
        cal_good = cal_good[cal_good].index.to_period("Y-DEC")  # good calendar years

        # 3) hydrological-year completeness (A-SEP)
        hyd_good = (
            m_count.ge(min_days)
                .groupby(pd.Grouper(freq="YE-SEP"))  # year ends in Sep
                .sum()
                .eq(12)
        )
        hyd_good = hyd_good[hyd_good].index.to_period("Y-SEP")  # good hydro years

        # 4) tag daily rows with both periods
        daily_cal = s.index.to_period("Y-DEC")
        daily_hyd = s.index.to_period("Y-SEP")

        cal_df = s[daily_cal.isin(cal_good)].copy()
        hyd_df = s[daily_hyd.isin(hyd_good)].copy()

        cal_df['uar'] = 1000.0 * cal_df['discharge'] / da
        hyd_df['uar'] = 1000.0 * hyd_df['discharge'] / da
        return cal_df, hyd_df


    def _preprocess_flow_data(self):
        # use the left edge of the positive bins for the threshold 
        # to filter by the quantization 
        zf_lin = float(np.exp(self.pos_edges[0]))
        self.zero_equivalent_flow_counts_hyd = (self.hyd_df['uar'] < zf_lin).sum()
        self.zero_equivalent_flow_counts_cal = (self.cal_df['uar'] < zf_lin).sum()
        # now filter the cal and hyd_df to only include non-zero-equivalent flow days
        self.hyd_df = self.hyd_df[self.hyd_df['uar'] >= zf_lin]
        self.cal_df = self.cal_df[self.cal_df['uar'] >= zf_lin]
        # now compute the log_uar columns
        self.hyd_df['log_uar'] = np.log(self.hyd_df['uar'])
        self.cal_df['log_uar'] = np.log(self.cal_df['uar'])
        

    def _initialize_station(self, stn):
        self.stn = stn
        self.da = self.ctx.da_dict[stn]
        self.df, self.zero_flow_flag = self.retrieve_and_preprocess_timeseries_discharge(stn)
        self.cal_df, self.hyd_df = self.filter_complete_hydrological_years(self.df, self.da)
        self._preprocess_flow_data()
       

    def _set_zero_flow_edges(self):
        # threshold in log-UAR
        y_zf = float(np.log(1000.0 * self.ctx.zero_flow_threshold / self.target_da))
        # pos_edges, zero_bin_index = set_pos_edges_and_zero_bin(edges_extended, y_zf)
        i_right = np.digitize(y_zf, self.log_edges_extended, right=True)
        # print(self.target_da, self.target_stn, self.log_edges_extended, i_right, y_zf)
        # left edge idx of containing zero-equivalent flow bin
        self.zero_bin_index  = max(i_right - 1, 0)  
        self.pos_edges = self.log_edges_extended[self.zero_bin_index + 1:]
        # set the minimum measurable UAR to the left edge of the zero-equivalent bin
        # this is used for quantile-based evaluation metrics
        # setting it one bin left of the threshold sets a maximum "penalty" for 
        # flows that are below the threshold to one bin width (i.e. 4% for 8 bits)
        self.min_measurable_log_uar = self.log_x_extended[self.zero_bin_index]
        self.min_measurable_uar = np.exp(self.min_measurable_log_uar)
        
        assert self.pos_edges[0] > y_zf, "pos_edges must start above threshold."


    def digitize_uar_series(self, df, minimum_log_uar_threshold):
        # digitize the uar series
        
        lin_edges_extended = np.exp(self.log_edges_extended)    
        df['uar_bin'] = np.digitize(df['uar'], lin_edges_extended, right=False) - 1 # hydrologic year data
        zero_bin_index = np.digitize(np.array([minimum_log_uar_threshold]), self.log_edges_extended, right=False)[0] - 1

        # map the quantized bin values back to the series
        lin_x_extended = np.exp(0.5 *(self.log_edges_extended[1:] + self.log_edges_extended[:-1]))
        df['uar_discrete'] = lin_x_extended[df['uar_bin'].clip(0, np.inf)]
        # clip the bin indices to valid range
        assert df['uar_bin'].max() < len(lin_x_extended), f"uar_bin index out of range. {df['uar_bin'].max()} >= {len(lin_x_extended)}"

        # handle bin values below the minimum measurable threshold
        df['uar_bin_adjusted'] = df['uar_bin'].copy()
        # handle values below the minimum measurable threshold
        df['uar_zero_adjusted'] = df['uar'].copy()
        if df['uar_bin'].min() < zero_bin_index and zero_bin_index > 0:
            # get the minimum log value
            # the discrete x value to the left of the bin containing the "minimum measurable value"
            min_uar = lin_x_extended[zero_bin_index - 1]
            # min_uar2 = lin_x_extended[zero_bin_index]
            # check that the assigned min_uar corresponds to the correct
            # # bin x value (midpoint in log space)
            # print(f"Values found below minimum measurable threshold: {minimum_uar_threshold:.5f}")
            # print(min_uar, minimum_uar_threshold, min_uar2)
            # print(lin_x_extended[zero_bin_index - 2:zero_bin_index + 3])
            # print(lin_edges_extended[zero_bin_index - 2:zero_bin_index + 3])
            df.loc[df['uar_bin'] < 0, 'uar_discrete'] = np.float32(min_uar)
            # df.loc[df['uar_bin'] < 0, 'uar'] = np.float32(min_uar)

            # adjust the uar bin where the bin index is smaller than the zero bin index
            df.loc[df['uar_bin_adjusted'] < zero_bin_index, 'uar_bin_adjusted'] = 0
            # adjust the uar values below the minimum measurable threshold
            df.loc[df['uar_bin_adjusted'] < zero_bin_index, 'uar_zero_adjusted'] = np.float32(min_uar)
            # print(self.hyd_df[self.hyd_df['uar_bin'] == self.hyd_df['uar_bin'].min()].copy(), self.zero_bin_index)
            # raise ValueError(f"uar_bin index negative: {self.hyd_df['uar_bin'].min()} < 0")
        # else:
        #     print(f"No values below minimum measurable threshold: {minimum_uar_threshold:.5f}")
        #     print(self.hyd_df[self.hyd_df['uar_bin'] == self.hyd_df['uar_bin'].min()].copy(), self.zero_bin_index)

        return df
    
    
    def build_pmf_from_timeseries(self, uar_values, min_measurable_log_uar, drainage_area_km2):
        """
        Build the discretized and KDE smoothed PMFs for the daily uar timeseries.
        the input uar threshold and drainage area should be as follows:
            -for ensemble average timeseries models: the target catchment's threshold and drainage area
            -for donor station timeseries models: the donor catchment's threshold and drainage area
        This is to ensure that the PMF support reflects what can be validated at both the
        donor station for model input (avoid contributing values / information that cannot be measured)
        and at the target catchment (avoid contributing values / information that cannot be measured).
        """
        df = pd.DataFrame({'uar': uar_values})
        df = self.digitize_uar_series(df, min_measurable_log_uar) - 1

        # counts = np.histogram(log_uar, bins=self.pos_edges, density=False)[0].astype(np.int64, copy=False)
        unique_bin_idxs, bin_counts = np.unique(df['uar_bin_adjusted'].values, return_counts=True)

        if self.ctx.regularization_type == 'discrete':
            # initialize the PMF 
            pmf = np.zeros(len(self.log_x_extended))

            # assign the observation counts to the pmf by bin index
            pmf[unique_bin_idxs] = bin_counts.astype(int)

            # assert the counts match
            assert pmf.sum() == len(df), f"PMF counts {pmf.sum()} do not match number of observations {len(df)}"

            # normalize to PMF
            pmf /= pmf.sum()

        elif self.ctx.regularization_type == 'kde':
            # KDE on the same grid given the observed uar values
            # with the smallest value adjusted to the minimum measurable threshold
            positive_values = df[df['uar'].values >= np.exp(min_measurable_log_uar)].copy()['uar'].values
            N_p = len(positive_values)
            N_n = len(df) - N_p
            assert N_n != len(df), "All values are below the minimum measurable threshold."
            pmf_kde_raw, _ = self.kde_estimator.compute(
                positive_values,
                drainage_area_km2
                )  # returns pmf over pos_edges intervals
            
            kde_counts = (pmf_kde_raw * N_p)
            
            if N_n > 0:
                pmf = np.zeros_like(kde_counts)
                if self.zero_bin_index == 0: 
                    # the minimum measurable threshold is below the support, N_n are zero flows.
                    # all zero flows go to the first bin and there is no lower bin mass to consider
                    low_probability_mass = N_n 
                else:
                    # compute the low probability mass from the KDE below the zero bin index
                    low_probability_mass = N_n + kde_counts[:self.zero_bin_index].sum()
                
                # if zero_bin_index == 0, the zero index bin will be reassigned in the second step
                pmf[self.zero_bin_index:] = kde_counts[self.zero_bin_index:]
                pmf[0] = low_probability_mass
                # print(N_n, low_probability_mass, pmf_kde[0], self.zero_bin_index)
                
            else:
                assert N_p == len(df)
                pmf = (pmf_kde_raw * len(df))

            count_match = int(pmf.sum() - len(df))

            assert np.isclose(count_match, 0), f"PMF counts {pmf.sum()} do not match number of observations {len(df)} after zero bin adjustment"
            pmf /= pmf.sum()  # renormalize to PMF
        else:
            raise ValueError(f"Unknown regularization type set in context: {self.ctx.regularization_type}.  Should be 'discrete' or 'kde'.")
        
        assert np.isclose(pmf.sum(), 1.0), f"Discrete PMF does not sum to 1: {pmf.sum()}"
        
        return pmf


    def build_pmf(
        self,
        log_uar: np.ndarray, 
        ):
        """
        Returns (pmf_obs_col, pmf_kde_col), both length = len(edges_extended) - 1
        aligned on the same grid with a dedicated zero bin at the interval that contains y_zf.
        Assumes baseline._initialize_station(stn, left_edge) filters data to >= threshold.
        """

        zero_equivalent_flow_counts = (log_uar <= self.pos_edges[0]).sum()
        positive_uar = log_uar[log_uar > self.pos_edges[0]]

        # histogram positives on pos_edges
        counts = np.histogram(positive_uar, bins=self.pos_edges, density=False)[0].astype(np.int64, copy=False)

        # zero bin is [edges[i0], edges[i0+1]); first positive edge is edges[i0+1]
        assert np.allclose(self.pos_edges[0], self.log_edges_extended[self.zero_bin_index + 1]), f"Positive edges do not align for {self.stn}."

        # histogram length matches positive bins
        assert len(self.pos_edges) - 1 == counts.size, f"Counts size {counts.size} != pos bins {len(self.pos_edges)-1} for {self.stn}."

        # allocate columns
        n_bins = self.log_edges_extended.size - 1
        pmf_obs_col = np.zeros(n_bins, dtype=np.float32)
        pmf_kde_col = np.zeros(n_bins, dtype=np.float32)

        # set start/end for positive bins
        # make sure it accounts for the bin edges
        # vs discrete bin counts offset (the latter is one less)
        start = self.zero_bin_index + 1
        end   = start + counts.size

        if self.ctx.regularization_type == 'discrete':
            if end > pmf_obs_col.size:
                raise ValueError(f"Counts overflow: {counts.size} bins, available {pmf_obs_col.size - start}")
            pmf_obs_col[self.zero_bin_index] = int(zero_equivalent_flow_counts)
            pmf_obs_col[start:end]      = counts

            # normalize empirical to PMF
            s_obs = pmf_obs_col.sum()
            pmf_obs_col /= s_obs  # renormalize to PMF
            pmf = pmf_obs_col
        elif self.ctx.regularization_type == 'kde':
            # KDE on the exact same grid
            kde = KDEEstimator(self.pos_edges)
            pmf_kde_pos, _ = kde.compute(np.exp(positive_uar), self.target_da)  # returns pmf over pos_edges intervals
            kde_counts = (pmf_kde_pos * len(positive_uar)).astype(np.float32, copy=False)

            pmf_kde_col[self.zero_bin_index] = float(zero_equivalent_flow_counts)
            pmf_kde_col[start:end]      = kde_counts
            s_kde = pmf_kde_col.sum()
            pmf_kde_col /= s_kde  # renormalize to PMF
            pmf = pmf_kde_col
        else:
            raise ValueError(f"Unknown regularization type set in context: {self.ctx.regularization_type}.  Should be 'discrete' or 'kde'.")

        return pmf
        
    
    def D_bits_Q_to_Qlam(self, Q, lam):
        """Exact D_bits(Q || Q_lam) for Q_lam = (1-lam)Q + lam U."""
        Q = np.asarray(Q, dtype=float)
        Q = np.clip(Q / Q.sum(), 1e-300, 1.0)
        N = Q.size
        U = 1.0 / N
        Qlam = (1.0 - lam) * Q + lam * U
        return np.sum(Q * (np.log2(Q) - np.log2(Qlam)))
    
    
    def compute_optimal_delta_limited_lambda(self, Q, maxit=100, tol=1e-6):
        """
        Largest λ ∈ [0,1] such that D_bits(Q || Q_λ) ≤ δ (exact, monotone bisection).
        """
        lo, hi = 0.0, 1.0
        if self.D_bits_Q_to_Qlam(Q, hi) <= self.delta:
            return hi
        for _ in range(maxit):
            mid = 0.5 * (lo + hi)
            if self.D_bits_Q_to_Qlam(Q, mid) <= self.delta:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return lo
    

    def mix_with_uniform(self, Q: np.ndarray, lam: float) -> np.ndarray:
        """Shrink Q toward uniform with weight λ.  
        Applies a Dirichlet (uniform) prior to the distribution.
        lam = lambda = strength of belief in the prior
        """
        U = np.ones_like(Q) / len(Q)
        return (1.0 - lam) * Q + lam * U
    

    def _compute_adjusted_distribution_with_mixed_uniform(self, pmf):
        """
        Compute a mixture Q_mixed = (1 - alpha) * Q_kde + alpha * Q_uniform
        The mixture is limited by delta, the maximum allowed perturbation between
        Q_mixture and the original Q.
        Given delta, we can compute the largest allowable lambda,
        which represents the most noise added without overly influencing Q.
        """
        lam_exact = self.compute_optimal_delta_limited_lambda(pmf)
        pmf_mixed = self.mix_with_uniform(pmf, lam_exact)
        # pdf_mixed = pmf_mixed / self.log_w
        assert np.isclose(np.sum(pmf_mixed), 1.0), f'Mixed PMF does not sum to 1: {np.sum(pmf_mixed)}'
        pmf_mixed /= pmf_mixed.sum()

        # pdf_check = np.trapezoid(pdf_mixed, x=self.log_x)
        # pdf_mixed /= pdf_check

        return pmf_mixed
