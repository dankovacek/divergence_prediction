import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point
from collections import OrderedDict

from numba import jit
import jax.numpy as jnp

from pathlib import Path

HYSETS_DIR = Path('/home/danbot/code/common_data/HYSETS')


@jit
def compute_overlap_matrix(mask_jax):
    return jnp.matmul(mask_jax, mask_jax.T)


class FDCEstimationContext:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.baseline_distribution_folder = os.path.join(self.baseline_distribution_folder, f'{self.bitrate:02d}_bits')
        
        self._load_baseline_distributions()
        self._load_catchment_data()
        self._load_and_filter_hysets_data()
        self.LN_param_dict = self.predicted_param_dict
        self._set_tree_idx_mappers()
        self._build_network_trees()
        self._set_attribute_indexers()
        self.prepare_fdc_cache()

    
    def _cache_get(self, stn):
        v = self._fdc_cache.pop(stn, None)
        if v is not None:
            self._fdc_cache[stn] = v
        return v  # v is (col_array, zero_flag, da_km2) or similar

    def _cache_put(self, stn, val):
        if stn in self._fdc_cache:
            self._fdc_cache.pop(stn)
        self._fdc_cache[stn] = val
        while len(self._fdc_cache) > self._fdc_cache_cap:
            self._fdc_cache.popitem(last=False)

    
    def _estimate_cache_capacity(self, target_ram_gb=8.0):
        """
        Choose how many stations to cache based on available RAM.
        target_ram_gb: how much RAM to dedicate to the column cache (adjust).
        """
        # Discover lengths once (after you’ve set self.ctx.ds)
        n_time = int(self.ds.sizes["time"])
        # We store float32 columns (4 bytes per value).
        bytes_per_col = n_time * 4

        # Convert target to bytes and keep a small overhead factor
        target_bytes = int(target_ram_gb * (1024**3))
        overhead = 1.15  # cushion for Python/DF overhead
        cap = max(64, target_bytes // int(bytes_per_col * overhead))
        return cap

    def prepare_fdc_cache(self, target_ram_gb=32.0):
        # call this once after loading ds
        self._time_index = self.ds["time"].to_index()
        self._fdc_cache_cap = self._estimate_cache_capacity(target_ram_gb)
        self._fdc_cache = OrderedDict()
        
    
    def _load_baseline_distributions(self):
        """Load the baseline distributions for the stations."""
        self.baseline_distribution_folder = Path(self.baseline_distribution_folder)
        if not self.baseline_distribution_folder.exists():
            raise FileNotFoundError(f"Baseline distribution folder {self.baseline_distribution_folder} does not exist.")
        
        # Load the baseline distributions
        obs_pmf_fpath = self.baseline_distribution_folder / f'pmf_obs.csv'
        self.baseline_obs_pmf_df = pd.read_csv(obs_pmf_fpath, index_col='log_x_uar')
        kde_pmf_fpath = self.baseline_distribution_folder / f'pmf_kde.csv'
        self.baseline_kde_pmf_df = pd.read_csv(kde_pmf_fpath, index_col='log_x_uar')
        
        # obs_pdf_fpath = self.baseline_distribution_folder / f'pdf_obs.csv'
        # self.baseline_obs_pdf_df = pd.read_csv(obs_pdf_fpath, index_col='log_x_uar')
        # kde_pdf_fpath = self.baseline_distribution_folder / f'pdf_kde.csv'
        # self.baseline_kde_pdf_df = pd.read_csv(kde_pdf_fpath, index_col='log_x_uar')
        
        
    def _load_and_filter_hysets_data(self):
        hs_df = pd.read_csv('data/HYSETS_watershed_properties.txt', sep=';')
        if self.include_pre_1980_data == True:
            self.global_start_date = pd.to_datetime('1950-01-01') # Daymet starts 1980-01-01
        else:            
            self.global_start_date = pd.to_datetime('1980-01-01') # Hysets streamflow starts 1950-01-01
        
        hs_df = hs_df[hs_df['Official_ID'].isin(self.official_ids)]
        self.hs_df = hs_df
        # self.official_ids = hs_df['Official_ID'].values

        # load the updated HYSETS data
        updated_filename = 'HYSETS_2023_update_QC_stations.nc'
        ds = xr.open_dataset(HYSETS_DIR / updated_filename)  

        # Get valid IDs as a NumPy array
        selected_ids = hs_df['Watershed_ID'].values

        # Get boolean index where watershedID in selected_set
        # safely access watershedID as a variable first
        ws_ids = ds['watershedID'].data  
        mask = np.isin(ws_ids, selected_ids)

        # Apply mask to data
        ds = ds.sel(watershed=mask)
        # Step 1: Promote 'watershedID' to a coordinate on the 'watershed' dimension
        ds = ds.assign_coords(watershedID=("watershed", ds["watershedID"].data))

        # filter the timeseries by the global start date
        ds = ds.sel(time=slice(self.global_start_date, None))

        # Step 2: Set 'watershedID' as the index for the 'watershed' dimension
        self.ds = ds.set_index(watershed="watershedID")
    
    
    def _load_catchment_data(self):
        df = gpd.read_file(self.attr_df_fpath)
        df.columns = [c.lower() for c in df.columns]

        df['geometry'] = df.apply(lambda row: Point(row['centroid_lon_deg_e'], row['centroid_lat_deg_n']), axis=1)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')  # Ensure the CRS is WGS84
        df = df.to_crs(epsg=3005)  # Ensure the CRS is BC Albers for distance calculations
    
        # filter for the common stations between BCUB region and LSTM-compatible (i.e. 1980-)
        if self.include_pre_1980_data == True:
            meta_cols = ['log_x', 'lin_x', 'left_log_edges', 'right_log_edges']
            self.official_ids = [c for c in self.baseline_obs_pmf_df.columns if c not in meta_cols]
            print(f'    Using all stations in the catchment data with a baseline PMF (validated): {len(self.official_ids)}')
        else:
            self.official_ids = self.daymet_concurrent_stations
            print(f'   Using only stations with Daymet concurrency: {len(self.official_ids)}')

        df = df[df['official_id'].isin(self.official_ids)]
                
        # create dict structures for easier access of attributes and geometries
        self.da_dict = df[['official_id', 'drainage_area_km2']].set_index('official_id').to_dict()['drainage_area_km2']
        df['tmean'] = (df['tmin'].astype(float) + df['tmax'].astype(float)) / 2.0
        df['log_drainage_area_km2'] = np.log(df['drainage_area_km2'].astype(float))
        self.attr_gdf = df


    def _build_network_trees(self, attribute_cols=['log_drainage_area_km2', 'elevation_m', 'prcp', 'tmean', 'swe', 'srad',
                            'centroid_lon_deg_e', 'centroid_lat_deg_n', 'land_use_forest_frac_2010', 'land_use_snow_ice_frac_2010',
                            #  , 'land_use_wetland_frac_2010', 'land_use_water_frac_2010', 
                            ]):
        self.coords = np.array([[geom.x, geom.y] for geom in self.attr_gdf.geometry.values])
        self.spatial_tree = cKDTree(self.coords)

        # Extract values (excluding 'official_id' since it's categorical)
        attr_values = self.attr_gdf[attribute_cols].to_numpy()
        scaler = StandardScaler()
        self.normalized_attr_values = scaler.fit_transform(attr_values)
        # Construct the KDTree for the attribute space
        self.attribute_tree = cKDTree(self.normalized_attr_values)


    def _set_tree_idx_mappers(self):
        """Set the index mappers for the official_id to index and vice versa.
        This is for the network TREES"""
        self.id_to_idx = {id: i for i, id in enumerate(self.attr_gdf['official_id'].values)}
        self.idx_to_id = {i: id for i, id in enumerate(self.attr_gdf['official_id'].values)}
    

    def _set_attribute_indexers(self):
        
        # Create mapping from official_id to index
        # self.id_to_index = {oid: i for i, oid in enumerate(self.attr_gdf["official_id"])}
        # self.index_to_id = {i: oid for oid, i in self.id_to_index.items()}  # Reverse mapping
        # create a dictionary where the keys are Watershed_ID and the values are Official_ID
        self.watershed_id_dict = {row['Watershed_ID']: row['Official_ID'] for _, row in self.hs_df.iterrows()}
        # and the inverse
        self.official_id_dict = {row['Official_ID']: row['Watershed_ID'] for _, row in self.hs_df.iterrows()}
        # also for drainage areas
        self.da_dict = {row['Official_ID']: row['Drainage_Area_km2'] for _, row in self.hs_df.iterrows()}
        

    def _load_predicted_ln_params(self):
        """Retrieve log-normal parameters for a station."""
        parameter_prediction_results_folder = os.path.join('data', 'results', 'parameter_prediction_results', )
        predicted_params_fpath   = os.path.join(parameter_prediction_results_folder, 'OOS_parameter_predictions.csv')
        rdf = pd.read_csv(predicted_params_fpath, index_col=['official_id'], dtype={'official_id': str})
        rdf.columns = ['_'.join(c.split('_')[:-1]) for c in rdf.columns]
        return rdf.to_dict(orient='index')
    
 
    # def _compute_concurrent_overlap_dict(self, variable='discharge'):
    #     """
    #     Compute the concurrent overlap of monitored watersheds in the dataset.
    #     Threshold years represent the minimum number of days of overlap 
    #     (ignoring continuity) for a watershed to be considered concurrent.
    #     """
    #     overlap_dict_file = os.path.join('data', 'record_overlap_dict.json')
    #     if os.path.exists(overlap_dict_file):
    #         with open(overlap_dict_file, 'r') as f:
    #             overlap_dict = json.load(f)
    #         print(f'    ...overlap dict loaded from {overlap_dict_file}')
    #         return overlap_dict
        
    #     watershed_ids = self.ds['watershed'].values
    #     data = self.ds[variable].load().values  # (N, T)
    #     dates = pd.to_datetime(self.ds['time'].values) # shape (T, )
    #     N, T = data.shape

    #     # Extract year/month info 
    #     years = np.array([d.year for d in dates])
    #     months = np.array([d.month for d in dates])
    #     unique_years = np.unique(years)
    #     year_to_index = {year: i for i, year in enumerate(unique_years)}
    #     Y = len(unique_years)

    #     # Count valid obs per (year, site, month) 
    #     monthly_counts = np.zeros((Y, N, 12), dtype=np.uint16)
    #     valid_mask = ~np.isnan(data)  # shape (N, T)

    #     for t in range(T):
    #         y_idx = year_to_index[years[t]]
    #         m_idx = months[t] - 1  # month index in [0, 11]
    #         monthly_counts[y_idx, :, m_idx] += valid_mask[:, t]

    #     # Create monthly validity mask
    #     monthly_valid = monthly_counts >= self.minimum_days_per_month  # shape: (Y, N, 12)
    #     valid_years = np.all(monthly_valid, axis=-1)  # shape: (Y, N)
    #     complete_years = np.sum(valid_years, axis=0)  # shape: (N,)

    #     # Compute joint validity per pair
    #     joint_valid = np.logical_and(
    #         monthly_valid[:, :, None, :],  # shape: (Y, i, 1, M)
    #         monthly_valid[:, None, :, :]   # shape: (Y, 1, j, M)
    #     )  # shape: (Y, i, j, M)

    #     joint_year_valid = np.all(joint_valid, axis=-1)               # shape: (Y, i, j)
    #     concurrent_years_matrix = np.sum(joint_year_valid, axis=0)    # shape: (i, j)
        
    #     # loop over minimum concurrent proportions
    #     overlap_dict = defaultdict(dict)
    #     for prop in [0, 25, 50, 75, 100]:
    #         # Mask where concurrent years ≥ required proportion of complete years for i
    #         thresholds = (complete_years * prop / 100.0).astype(int)  # shape: (N,)
    #         valid_mask = concurrent_years_matrix >= thresholds[:, None]  # broadcast shape: (N, N)
    #         np.fill_diagonal(valid_mask, False)

    #         # Build overlap_dict
    #         odict = {
    #             int(watershed_ids[i]): list(map(int, watershed_ids[valid_mask[i]]))
    #             for i in range(N)
    #         }
    #         overlap_dict[str(prop)] = odict

    #     with open(overlap_dict_file, 'w') as f:
    #         json.dump(overlap_dict, f)
    #     print(f'    ...saved overlap dict to {overlap_dict_file}')
    #     return overlap_dict
        
        
    # def _generate_12_month_windows(self, index):
    #     months = pd.date_range(index.min(), index.max(), freq='MS')
    #     windows = [(start, start + pd.DateOffset(months=12) - pd.Timedelta(days=1)) for start in months]
    #     return [w for w in windows if w[1] <= index.max()]
    
    
    # def _is_window_valid(self, ts, start, end):
    #     window = ts.loc[start:end]
    #     if window.empty:
    #         return False
    #     grouped = window.groupby(window.index.month).size()
    #     if set(grouped.index) != set(range(1, 13)):
    #         return False
    #     if grouped.min() < 10:
    #         return False
    #     return True
    
    
    # def _compute_station_valid_windows(self, ts, windows):
    #     return [self._is_window_valid(ts, start, end) for (start, end) in windows]
    
    
    # def _count_valid_shared_windows(self, valid_i, valid_j):
    #     return sum(np.logical_and(valid_i, valid_j))
    
 