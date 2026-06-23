# knn_estimator.py
import numpy as np
import pandas as pd
from collections import defaultdict
# from concurrent.futures import ThreadPoolExecutor
from jax import numpy as jnp
from time import time


class kNNEstimator:
    def __init__(self, ctx, data, *args, **kwargs):
        self.ctx = ctx
        self.data = data
        self.target_stn = self.data.target_stn
        self.k_nearest = data.k_nearest
        # self.max_to_check_start = data.max_to_check
        # self.max_to_check = data.max_to_check
        self.weight_schemes = [1, 2] #inverse distance and inverse square distance
        self.knn_simulation_data = {}
        self.knn_pdfs = pd.DataFrame()
        self.knn_pmfs = pd.DataFrame()


    def _find_k_nearest_neighbors(self, tree_type, max_to_check):
        # Query the k+1 nearest neighbors because the first neighbor is the target point itself
        target_idx = self.ctx.id_to_idx[self.target_stn]
        if tree_type == 'spatial_dist':
            distances, indices = self.ctx.spatial_tree.query(self.ctx.coords[target_idx], k=max_to_check, workers=-1)
            distances *= 1 / 1000
        elif tree_type == 'attribute_dist':
            # Example query: Find the nearest neighbors for the first point
            distances, indices = self.ctx.attribute_tree.query(self.ctx.normalized_attr_values[target_idx], k=max_to_check, workers=-1)
        else:
            raise Exception('tree type not identified, must be one of spatial_dist, or attribute_dist.')
        
        # Remove target (self) from the results
        self_index = target_idx
        keep = indices != self_index
        indices = indices[keep]
        distances = distances[keep]

        return indices, np.round(distances, 3)
    

    def _compute_effective_k(self, df, max_k=None):
        arr = df.to_numpy()
        T, K = arr.shape
        max_k = max_k or K

        nan_mask = np.isnan(arr)
        sorted_idx = np.argsort(nan_mask, axis=1)
        row_idx = np.arange(T)[:, None]

        ks = np.arange(1, max_k + 1)
        effective_k = []
        mean_furthest = []

        for k in ks:
            idx = sorted_idx[:, :k]
            valid = ~nan_mask[row_idx, idx]
            valid_count = valid.sum(axis=1)

            effective_k.append(valid_count.mean())
            furthest_idx = np.where(valid, idx, -1).max(axis=1)
            mean_furthest.append(furthest_idx[valid_count >= k].mean() if np.any(valid_count >= k) else np.nan)

        return pd.DataFrame({
            'effective_k': np.round(effective_k, 1),
            'mean_furthest_idx': np.round(mean_furthest, 2)
        }, index=ks)
    

    def _query_distance(self, tree, id1, id2):
        """Query distance between two points in a tree using official_id."""
        if id1 not in self.ctx.id_to_idx or id2 not in self.ctx.id_to_idx:
            raise ValueError(f"One or both IDs ({id1}, {id2}) not found.")
    
        # Get indices from ID mapping
        index1, index2 = self.ctx.id_to_idx[id1], self.ctx.id_to_idx[id2]
        # Query the distance
        distance = np.linalg.norm(tree.data[index1] - tree.data[index2])  # Euclidean distance
        return distance
            
    
    def _prewarm_neighbor_cache(self, station_ids, k=10):
        """Primes the cache for the first k neighbors to reduce I/O latency later."""
        for stn in station_ids[:k]:
            _ = self.data.retrieve_and_preprocess_timeseries_discharge(stn)


    def _retrieve_nearest_nbr_data(self, tree_type):
        """
        Iterate through the nearest neighbours until enough good neighbours are found.
        Filter each donor by complete years based on the context settings.
        Use clipped UAR values for the proxy data to avoid introducing bias from 
        low-flow values the donor catchment that cannot be validated.
        Clipping means the proxy data values below the "minimum measurable UAR" threshold
        are replaced with a constant value that is one bin to the left of the smallest
        positive bin in the target catchment's FDC support.  This sets a (relative) upper bound
        on the error that can be introduced from low-flow values in the donor catchment.
        """
        MAX_CHECK = 250
        REQUIRED_GOOD = self.k_nearest
        # Get the index of the target station
        
        # Query once for all potential neighbors
        nbr_idxs, dists = self._find_k_nearest_neighbors(tree_type, MAX_CHECK)
        nbr_ids = [self.ctx.idx_to_id[i] for i in nbr_idxs if self.ctx.idx_to_id[i] != self.target_stn]
        distances = [d for i, d in zip(nbr_idxs, dists) if self.ctx.idx_to_id[i] != self.target_stn]

        good_nbrs = []
        sorted_nbrs = sorted(zip(nbr_ids, distances), key=lambda x: x[1])
        # Pre-warm the cache on the first few neighbors to reduce I/O latency
        self._prewarm_neighbor_cache(nbr_ids)
        t0 = time()
        for (nbr_id, dist) in sorted_nbrs:
            df, _ = self.data.retrieve_and_preprocess_timeseries_discharge(nbr_id)
            self.cal_df, self.hyd_df = self.data.filter_complete_hydrological_years(df[['discharge', f'{nbr_id}_uar_clipped']], self.ctx.da_dict[nbr_id])

            self.cal_df.rename(columns={'uar': f'{nbr_id}_uar'}, inplace=True)
            self.hyd_df.rename(columns={'uar': f'{nbr_id}_uar'}, inplace=True)
            
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue  # Skip bad or empty DataFrames
                
            if self.ctx.year_type == 'hydrological':
                complete_years = self.data.complete_year_dict[nbr_id]['hyd_years']
                proxy_df = self.hyd_df.copy()
            elif self.ctx.year_type == 'calendar':
                complete_years = self.data.complete_year_dict[nbr_id]['cal_years']
                proxy_df = self.cal_df.copy()
            else:
                raise Exception('year type not recognized, must be one of hydrological or calendar.')
            n_years = len(complete_years)
            proxy_df['uar'] = proxy_df[f'{nbr_id}_uar_clipped']
            donor_da = self.ctx.da_dict[nbr_id]
            min_log_uar_donor = np.log(1000.0 * self.data.min_measurable_uar / donor_da)  # L/s/km^2
            digitized = self.data.digitize_uar_series(proxy_df, min_log_uar_donor)
            unique_vals = np.unique(digitized['uar_bin'])
            n_unique_vals = len(unique_vals)
            # print(min_log_uar_donor, self.data.min_measurable_log_uar)
            # max bin must be greater than the target zero bin index to ensure variability
            # max_bin_idx = digitized['uar_bin'].max()
            if n_years >= self.ctx.min_record_length and n_unique_vals > 1:
                proxy_df.drop(columns=[f'{nbr_id}_uar'], inplace=True)
                proxy_df.rename(columns={f'{nbr_id}_uar_clipped': f'{nbr_id}_uar'}, inplace=True)
                good_nbrs.append((nbr_id, dist, n_years, proxy_df[[f'{nbr_id}_uar']]))
                # print(f'   ...added {nbr_id} as good neighbor for {self.target_stn}: distance={dist}, n_years={n_years}, n_unique_vals={n_unique_vals}.')
                if len(good_nbrs) == REQUIRED_GOOD:
                    break

        t1 = time()
        print(f'    ...found {len(good_nbrs)} good neighbors for {self.target_stn} using {tree_type} in {t1 - t0:.1f}s.')
        if len(good_nbrs) < REQUIRED_GOOD:
            raise Exception(f'Not enough good neighbors found for {self.target_stn} using {tree_type}. Found {len(good_nbrs)}, required {REQUIRED_GOOD}.')

        # Concatenate the timeseries
        nbr_df = pd.concat([r[3] for r in good_nbrs], axis=1)
        # Build metadata DataFrame
        complement_type = 'attribute_dist' if tree_type == 'spatial_dist' else 'spatial_dist'
        complement_tree = getattr(self.ctx, f"{complement_type.split('_')[0]}_tree")
        scale = 1 / 1000 if complement_type == 'spatial_dist' else 1

        nbr_data = pd.DataFrame(
            [r[:3] for r in good_nbrs],
            columns=['official_id', 'distance', 'n_years']
        )
        nbr_data[complement_type] = nbr_data['official_id'].apply(
            lambda x: round(scale * self._query_distance(complement_tree, x, self.target_stn), 3)
        )

        return nbr_df, nbr_data


    def _initialize_nearest_neighbour_data(self):
        """
        Generate nearest neighbours for spatial and attribute selected k-nearest neighbours for both concurrent and asynchronous records.
        """
        print(f'    ...initializing nearest neighbours with minimum concurrent record for {self.data.target_stn}.')
        self.nbr_dfs = defaultdict(lambda: defaultdict(dict))
        
        for tree_type in ['spatial_dist', 'attribute_dist']:
            nbr_df, nbr_data = self._retrieve_nearest_nbr_data(tree_type)
            effective_k = self._compute_effective_k(nbr_df, max_k=self.k_nearest)
            self.nbr_dfs[tree_type] = {
                'nbr_df': nbr_df,
                'nbr_data': nbr_data,
                'effective_k': effective_k,
            }
    

    def _compute_weights(self, m, k, distances, epsilon=1e-3):
        """Compute normalized inverse (square) distance weights to a given power."""

        distances = jnp.where(distances == 0, epsilon, distances)

        if k == 1:
            return jnp.array([1])
        else:
            inv_weights = 1 / (jnp.abs(distances) ** m)
            return inv_weights / jnp.sum(inv_weights)
           
    
    def _compute_ensemble_density_mixture(self, distribution, weights):
        """
        This function computes the weighted ensemble density mixture estimates.
        """
        # Normalize distance weights
        if weights is not None:
            weights /= jnp.sum(weights).astype(jnp.float32)
            weights = jnp.array(weights)  # Ensure 1D array
            pmf_est = jnp.asarray(distribution.to_numpy() @ weights)
        else:
            pmf_est = jnp.asarray(distribution.mean(axis=1).to_numpy())

        # Renormalize
        pmf_est /= jnp.sum(pmf_est)

        return pmf_est
    
    
    def _compute_ensemble_distributions(self, wm, nbr_df, nbr_data, distance_type, epsilon=1e-2):
        """
        Compute the donor ensemble density mixture estimates based on k-nearest neighbors.
        The donor PMFs were pre-computed, with the important note that the probability mass
        associated with uar values below the minimum measurable UAR threshold of the target catchment
        are all assigned to the 0-index bin.  
        """
        knn_df_all = nbr_df.iloc[:, :self.k_nearest].copy() # units are linear UAR (L/s/km^2)
        knn_data_all = nbr_data.iloc[:, :self.k_nearest].copy()
        proxy_ids = [c.split('_')[0] for c in knn_df_all.columns.tolist()]
        # load the kde PMFs to serve as the proxies to ensure complete support coverage
        if self.ctx.regularization_type == 'kde':
            density_mixture_pmf = self.ctx.baseline_kde_pmf_df[proxy_ids].copy()
        else:
            density_mixture_pmf = self.ctx.baseline_obs_pmf_df[proxy_ids].copy()

        labels, pmfs = [], []
        all_distances = knn_data_all['distance'].values
        all_ids = knn_data_all['official_id'].values

        for k in range(1, self.k_nearest + 1):
            distances = all_distances[:k]
            nbr_ids = all_ids[:k]
            knn_pmfs = density_mixture_pmf.iloc[:, :k].copy()

            label = f'{self.target_stn}_{k}_NN_{distance_type}_ID{wm}_freqEnsemble'
            weights = self._compute_weights(wm, k, distances)
            pmf_est = self._compute_ensemble_density_mixture(knn_pmfs, weights)
            unique_vals = np.unique(pmf_est)
            assert pmf_est is not None, f'pmf_est is None for {label}'

            # adjust the PMF based on the minimum observable threshold of the target catchment
            pmf_adjusted = np.zeros_like(pmf_est)
            i0 = self.data.zero_bin_index # zero flow index on log_edges_extended

            # get the cdf value at the zero-equivalent flow bin (0-->threshold)
            low_probability_mass = pmf_est[:i0].sum()
            if self.data.zero_bin_index == 0: 
                # the minimum measurable threshold is below the support, N_n are zero flows.
                # all zero flows go to the first bin and there is no lower bin mass to consider
                pmf_adjusted = pmf_est
            else:                
                # compute the low probability mass from the KDE below the zero bin index
                low_probability_mass = pmf_est[:i0].sum()
                pmf_adjusted[0] = low_probability_mass
                pmf_adjusted[i0:] = pmf_est[i0:]
            
            # positive bin probabilities (threshold --> max)
            pmf_adjusted /= pmf_adjusted.sum()  # normalize raw PMF
        
            # compute the mean number of observations (non-nan values) per row
            mean_obs_per_timestep = knn_df_all.iloc[:, :k].notna().sum(axis=1).mean()
            mean_obs_per_proxy = knn_df_all.iloc[:, :k].notna().sum(axis=0).mean()

            # _, pmf_prior_adjusted = self.data._compute_adjusted_distribution_with_laplace_prior(pmf_est)
            pmf_prior_adjusted = self.data._compute_adjusted_distribution_with_mixed_uniform(pmf_adjusted)
            # if distance_type == 'spatial_dist':
            #     print('prior adjusted pmf')
            #     print(list(pmf_prior_adjusted))
            #     print('pmf adjusted')
            #     print(list(pmf_adjusted))
            #     print('self.data.baseline_pmf')
            #     print(list(self.data.baseline_pmf))
            #     print(f'io= {i0}, low_prob_mass={low_probability_mass}')
            eval = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(pmf_prior_adjusted, self.data.baseline_pmf, min_measurable_log_uar=self.data.min_measurable_log_uar, epsilon=epsilon)
            bias = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(pmf_prior_adjusted, pmf_adjusted, min_measurable_log_uar=self.data.min_measurable_log_uar, epsilon=epsilon)

            # compute the frequency-based ensemble pdf estimate
            self.knn_simulation_data[label] = {
                'k': k, 'n_obs': mean_obs_per_proxy,
                'mean_obs_per_timestep': mean_obs_per_timestep,
                'nbrs': ','.join(nbr_ids),
                'eval': eval,
                'bias': bias,
                }
            pmfs.append(np.asarray(pmf_adjusted))
            labels.append(label)

        # create a dataframe of labels(columns) for each pdf
        knn_pmfs = pd.DataFrame(pmfs, index=labels).T
        # Filter out already existing columns to avoid duplication
        new_pmf_cols = knn_pmfs.columns.difference(self.knn_pmfs.columns)
        # Concat only new columns
        self.knn_pmfs = pd.concat([self.knn_pmfs, knn_pmfs[new_pmf_cols]], axis=1)


    def _compare_bootstrap_P_to_ensemble_distributions(self, wm, nbr_df, nbr_data, bootstrap_P, distance_type):
        """
        For asynchronous comparisons, we estimate pdfs for ensemble members, then compute the mean in the time domain
        to represent the FDC simulation.  We do not do temporal averaging in this case.
        """
        knn_df_all = nbr_df.iloc[:, :self.k_nearest].copy()
        knn_data_all = nbr_data.iloc[:, :self.k_nearest].copy()
        proxy_ids = [c.split('_')[0] for c in knn_df_all.columns.tolist()]
        # load the kde PMFs to serve as the proxies to ensure complete support coverage
        # if self.baseline_pmf_type == 'kde':
        #     density_mixture_pdfs = self.ctx.baseline_kde_pmf_df[proxy_ids].copy()
        # else:
        density_mixture_pdfs = self.ctx.baseline_obs_pmf_df[proxy_ids].copy()

        labels, pdfs, pmfs = [], [], []
        all_distances = knn_data_all['distance'].values
        all_ids = knn_data_all['official_id'].values
        eval_dict = {}
        for k in range(1, self.k_nearest + 1):
            distances = all_distances[:k]
            nbr_ids = all_ids[:k]
            knn_pdfs = density_mixture_pdfs.iloc[:, :k].copy()

            label = f'{self.target_stn}_{k}_NN_{distance_type}_ID{wm}_freqEnsemble'
            weights = self._compute_weights(wm, k, distances)
            # pmf_est, pdf_est = self._compute_density_mixture_mean(knn_pdfs, weights)
            pmf_est = self._compute_ensemble_density_mixture(knn_pdfs, weights)
            assert pmf_est is not None, f'pmf_est is None for {label}'
        
            # compute the mean number of observations (non-nan values) per row
            # mean_obs_per_timestep = knn_df_all.iloc[:, :k].notna().sum(axis=1).mean()
            # mean_obs_per_proxy = knn_df_all.iloc[:, :k].notna().sum(axis=0).mean()

            pmf_prior_adjusted = self.data._compute_adjusted_distribution_with_mixed_uniform(pmf_est)

            sample_evals = []
            for bootstrap_pmf in bootstrap_P.T:
                assert bootstrap_pmf.shape == pmf_prior_adjusted.shape, f' shapes not equal {bootstrap_pmf.shape} != {pmf_prior_adjusted.shape} '
                
                eval = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(pmf_prior_adjusted, bootstrap_pmf)
                sample_evals.append(eval)
            eval_dict[k] = sample_evals
        return eval_dict

    
        #     # compute the frequency-based ensemble pdf estimate
        #     self.knn_simulation_data[label] = {
        #         'k': k, 'n_obs': mean_obs_per_proxy,
        #         'mean_obs_per_timestep': mean_obs_per_timestep,
        #         'nbrs': ','.join(nbr_ids),
        #         'eval': eval,
        #         'bias': bias,
        #         }
            
        #     pdfs.append(np.asarray(pdf_est))
        #     pmfs.append(np.asarray(pmf_est))
        #     labels.append(label)

        # # create a dataframe of labels(columns) for each pdf
        # knn_pdfs = pd.DataFrame(pdfs, index=labels).T
        # knn_pmfs = pd.DataFrame(pmfs, index=labels).T
        # # Filter out already existing columns to avoid duplication
        # new_pdf_cols = knn_pdfs.columns.difference(self.knn_pdfs.columns)
        # new_pmf_cols = knn_pmfs.columns.difference(self.knn_pmfs.columns)
        # # Concat only new columns
        # self.knn_pdfs = pd.concat([self.knn_pdfs, knn_pdfs[new_pdf_cols]], axis=1)
        # self.knn_pmfs = pd.concat([self.knn_pmfs, knn_pmfs[new_pmf_cols]], axis=1)
      

    def _weighted_row_mean_ignore_nan(self, df: pd.DataFrame, weights: np.ndarray):
        """
        Computes the weighted mean for each row, accounting for NaNs and ensuring that
        weights are re-normalized based on valid values only. Returns a Series aligned
        with df.index, as well as ensemble stats.
        The inputs come from knn_df, and the units should be UAR [L/s/km^2].
        """
        # drop rows with all NaNs
        df = df.dropna(how='all')
        assert not df.empty, 'DataFrame is empty after dropping all-NaN rows.'
        X = df.to_numpy()
        mask = ~np.isnan(X) # boolean mask for valid values in the matrix

        W = np.broadcast_to(weights, X.shape) # broadcast weights to match the shape of X
        masked_weights = np.where(mask, W, 0.0) # apply mask to weights, setting weight to 0 where X is NaN

        row_weight_sums = masked_weights.sum(axis=1) # sum weights across each row
        row_weight_sums[row_weight_sums == 0] = np.nan # avoid division by zero by setting sums to NaN where they are zero

        normalized_weights = masked_weights / row_weight_sums[:, None]
        estimated = np.nansum(X * normalized_weights, axis=1)

        # Return as Series aligned with index
        estimated_series = pd.Series(estimated, index=df.index)

        # Also compute stats on weights
        mean_valid_per_row = mask.sum(axis=1).mean()
        mean_weight_per_col = np.nanmean(normalized_weights, axis=0)
        effective_k = 1.0 / np.nansum(mean_weight_per_col ** 2)

        return estimated_series, mean_valid_per_row, effective_k


    def _finalize_temporal_ensemble(
            self, k, label, temporal_ensemble_mean, nbrs_used,
            effective_k, mean_valid_per_row
            ):

        # Check for zero runoff 
        # zero values are ok, units should be linear UAR, 
        # it will be addressed in the pmf estimation step
        # assert np.min(temporal_ensemble_mean.values) > 0, f'min value in temporal_ensemble_mean is {np.min(temporal_ensemble_mean.values)} for {label}'
        # zero_bin_index = self.data.zero_bin_index
        # min_measurable_log_uar = self.data.log_x_extended[zero_bin_index]

        # Estimate PDF/PMF using KDE or 
        if len(np.unique(temporal_ensemble_mean.values)) == 1:
            raise Exception(f'    ....only one unique value in temporal ensemble mean {label}, cannot proceed. {temporal_ensemble_mean}')
        else:
            # since we are inputting the temporal mean, this is now a model, so the minimum uar threshold 
            # and drainage area correspond to the target station to specify the support that cannot be 
            # validated given the assumption of the minimum measurable flow in the target catchment.
            est_pmf = self.data.build_pmf_from_timeseries(temporal_ensemble_mean.values, self.data.min_measurable_log_uar, self.data.target_da)

            assert len(np.unique(est_pmf)) > 1, f'estimated pmf has only one unique value for {k}-NN {label}'

        assert est_pmf is not None, f'pmf is None for {label}'

        # _, pmf_prior_adjusted = self.data._compute_adjusted_distribution_with_laplace_prior(est_pmf)
        pmf_prior_adjusted = self.data._compute_adjusted_distribution_with_mixed_uniform(est_pmf)
        estimation_metrics = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(pmf_prior_adjusted, self.data.baseline_pmf, min_measurable_log_uar=self.data.min_measurable_log_uar)
        bias = self.data.eval_metrics._evaluate_fdc_metrics_from_pmf(pmf_prior_adjusted, est_pmf, min_measurable_log_uar=self.data.min_measurable_log_uar)

        # Store simulation outputs and metadata
        self.knn_pmfs[label] = est_pmf
        self.knn_simulation_data[label] = {
            'nbrs': nbrs_used,
            'k': k,
            'n_obs': len(temporal_ensemble_mean),
            'mean_valid_per_row': mean_valid_per_row,
            'mean_nbrs_per_timestep': effective_k,  # rename if clearer
            'effective_k': effective_k,
            'eval': estimation_metrics,
            'bias': bias,
        }


    def _compute_temporal_ensemble_distributions(self, distance_type, wm, nbr_df, nbr_data):
        distances = nbr_data['distance'].values
        weights_10NN = self._compute_weights(wm, 10, distances[:10])
        for k in range(1, self.k_nearest + 1):
            knn_df = nbr_df.iloc[:, :k].copy()
            label = f'{self.target_stn}_{k}_NN_{distance_type}_ID{wm}_timeEnsemble'
            weights = weights_10NN[:k]
            temporal_ensemble_mean, mean_valid_per_row, effective_k = self._weighted_row_mean_ignore_nan(knn_df, weights)
            nbrs_used = [c.split('_')[0] for c in knn_df.columns]
            self._finalize_temporal_ensemble(
                k, label, temporal_ensemble_mean, nbrs_used,
                effective_k, mean_valid_per_row
            )
    
    
    def _compute_distribution_estimates(self, distance_type):

        nbr_df = self.nbr_dfs[distance_type]['nbr_df'].copy()
        nbr_data = self.nbr_dfs[distance_type]['nbr_data'].copy()

        for wm in self.weight_schemes:
            # compute the FDC estimate by temporal ensemble mean
            t0 = time()
            self._compute_temporal_ensemble_distributions(distance_type, wm, nbr_df, nbr_data)
            t1 = time()
            # compute the ensemble average density pdfs
            self._compute_ensemble_distributions(wm, nbr_df, nbr_data, distance_type)
            t2 = time()
            print(f'    ...{distance_type} ID{wm} took {t1 - t0:.2f}s for temporal ensemble, {t2 - t1:.2f}s for density ensemble.')


    def run_estimators(self):              
        self._initialize_nearest_neighbour_data()
        # set the baseline pdf by
        for dist in ['spatial_dist', 'attribute_dist']:        
            self._compute_distribution_estimates(dist)
        return self._format_results()
    
    
    def _make_json_serializable(self, d):
        output = {}
        for k, v in d.items():
            if isinstance(v, (np.ndarray, jnp.ndarray)):
                output[k] = v.tolist()
            elif hasattr(v, "tolist"):
                output[k] = v.tolist()
            else:
                output[k] = v
        return output
    
    
    def _format_results(self):
        pmf_labels, sim_labels = list(self.knn_pmfs.columns), list(self.knn_simulation_data.keys())
        # assert label sets are the same
        # assert set(pmf_labels) == set(pdf_labels), f'pmf_labels {pmf_labels} != pdf_labels {pdf_labels}'
        assert set(pmf_labels) == set(sim_labels), f'pmf_labels {pmf_labels} != sim_labels {sim_labels}'
        results = self.knn_simulation_data
        for label in pmf_labels:
            # add the pmf and pdf in a json serializable format
            results[label]['pmf'] = self.knn_pmfs[label].tolist()
            # results[label]['pdf'] = self.knn_pdfs[label].tolist()
            results[label] = self._make_json_serializable(results[label])
        return results