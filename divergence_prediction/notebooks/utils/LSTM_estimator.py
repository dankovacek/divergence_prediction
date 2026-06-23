# import data_processing_functions as dpf
# from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import pandas as pd
from time import time

class LSTMFDCEstimator:
    def __init__(self, ctx, data, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.ctx = ctx
        self.data = data
        self.target_stn = self.data.target_stn
        # self.data = data
        # self.LSTM_forcings_folder = self.ctx.LSTM_forcings_folder
        self.LSTM_ensemble_result_folder = self.ctx.LSTM_ensemble_result_folder
        self.df = self._load_ensemble_result()
        self.sim_cols = sorted([c for c in self.df.columns if c.startswith('streamflow_sim_')])
        # self.df = self._filter_for_complete_years()
        self._filter_complete_hydrological_years()
        

    def _load_ensemble_result(self):
        fpath = os.path.join(self.LSTM_ensemble_result_folder, f'{self.target_stn}_ensemble.csv')
        df = pd.read_csv(fpath)
        # rename 'Unnamed: 0' to 'time' and set to index
        df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df
    
    
    def _filter_complete_hydrological_years(self):
        s = self.df.copy()

        # Calendar years (A-DEC)
        ok_cal = (s.resample('MS').count().ge(20)
                    .groupby(pd.Grouper(freq='YE-DEC')).sum().eq(12))
        ok_cal.index = ok_cal.index.to_period('Y-DEC')      # <- PeriodIndex

        per_cal = s.index.to_period('Y-DEC')
        mask_cal = ok_cal.reindex(per_cal, fill_value=False).to_numpy()
        self.cal_df = s[mask_cal].copy()  # daily values in complete calendar years
        # add the discharge label
        # self.cal_df[self.sim_cols] = 1000 * self.cal_df[self.sim_cols] / self.data.target_da

        # Hydrologic years, e.g. Oct–Sep -> A-SEP
        hyd_ms = 'SEP'
        ok_hyd = (s.resample('MS').count().ge(20)
                    .groupby(pd.Grouper(freq=f'YE-{hyd_ms}')).sum().eq(12))
        
        ok_hyd.index = ok_hyd.index.to_period(f'Y-{hyd_ms}')  # <- PeriodIndex

        per_hyd = s.index.to_period(f'Y-{hyd_ms}')
        mask_hyd = ok_hyd.reindex(per_hyd, fill_value=False).to_numpy()
        self.hyd_df = s[mask_hyd].copy()
        # self.hyd_df[self.sim_cols] = 1000 * self.hyd_df[self.sim_cols] / self.data.target_da
        
    
    def _load_LSTM_forcing_file(self):
        # retrieve LSTM forcing data
        # read the forcing data from the LSTM forcing file
        # and return a dataframe with the same index as the LSTM results
        ldf = pd.read_csv(os.path.join(self.met_forcings_folder, f'{self.target_stn}_forcing.csv'))
        ldf.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
        ldf['time'] = pd.to_datetime(ldf['time'])
        ldf.set_index('time', inplace=True)
        ldf = ldf.loc[self.stn_df.index]
        # convert to unit area runoff (L/s/km2)
        ldf['uar'] = 1000 * ldf['discharge'] / self.target_da
        return ldf

    
    
    def _compute_time_ensemble_pmf(self):
        """
        For the temporal ensemble, we do not constrain the model by what can be 
        validated based on the assumed minimum measurable UAR of the catchment.
        The resulting temporal ensemble timeseries is however processed for
        zero-flow thresholding based on the target catchment's threshold,
        i.e. based on what could be validated by observations.
        """
        data = self.hyd_df[self.sim_cols].copy()
        temporal_ensemble_log = data.mean(axis=1) # this is still in log space
        self.temporal_ensemble = np.exp(temporal_ensemble_log.dropna().values) # convert to linear space for KDE
        pmf = self.data.build_pmf(temporal_ensemble_log.dropna().to_numpy(copy=False, dtype=np.float32))
        return pmf


    def _compute_ensemble_pmf_by_bincount(self, sim_cols, df, bitrate):
        """
        Individual timeseries are processed for
        zero-flow thresholding based on the target catchment's threshold,
        i.e. based on what could be validated by observations.
        Then the ensemble PMF is computed by averaging the individual PMFs.
        """
        n_bins = 2 ** bitrate
        pmf_arr = np.zeros((n_bins, len(sim_cols)), dtype=np.float32)
        for j, sim in enumerate(sim_cols):
            data_log_uar = df[sim].dropna().to_numpy(copy=False, dtype=np.float32)
            # Pull once, as ndarray
            pmf_arr[:, j] = self.data.build_pmf(data_log_uar)

        # compute the ensemble average pmf
        ensemble_pmf = pmf_arr.mean(axis=1)
        ensemble_pmf /= ensemble_pmf.sum()
        return ensemble_pmf


    def _compute_ensemble_distribution_estimate(self, ensemble_type):
        if ensemble_type == 'time':
            pmf = self._compute_time_ensemble_pmf()
        elif ensemble_type == 'frequency':
            # pmf = self._compute_density_mixture_average_pmf()
            pmf = self._compute_ensemble_pmf_by_bincount(self.sim_cols, self.hyd_df, self.ctx.bitrate)
        else:
            raise ValueError(f'Unknown ensemble type: {ensemble_type}')
        
        # normalize the estimated PMF
        pmf = pmf / pmf.sum()
        prior_adjusted_pmf = self.data._compute_adjusted_distribution_with_mixed_uniform(pmf)

        # log_zf = np.log(1000.0 * self.ctx.zero_flow_threshold / self.data.target_da)
        # zero_bin_index = int(np.searchsorted(self.data.log_edges_extended, float(log_zf), side="right")) - 1
        # min_measurable_log_uar = self.data.log_x_extended[zero_bin_index]
        min_measurable_log_uar = self.data.min_measurable_log_uar
        
        result = {}
        result['pmf'] = pmf.tolist()
        result['prior_adjusted_pmf'] = prior_adjusted_pmf.tolist()
        result['eval'] = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(prior_adjusted_pmf, self.data.baseline_pmf, min_measurable_log_uar=min_measurable_log_uar)
        result['bias'] = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(prior_adjusted_pmf, pmf, min_measurable_log_uar=min_measurable_log_uar)
        return result


    def run_estimators(self):
        results = {}
        for ensemble_type in ['time', 'frequency']:
            result = self._compute_ensemble_distribution_estimate(ensemble_type)
            results[ensemble_type] = result
        return results
    

