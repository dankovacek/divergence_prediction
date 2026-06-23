import os
import pandas as pd
import numpy as np
import json
from time import time
import xarray as xr
import geopandas as gpd

from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

import xgboost as xgb
xgb.config_context(verbosity=2)
from scipy.stats import wasserstein_distance
from scipy.stats import norm, laplace, genextreme

import jax
import jax.numpy as jnp
from jax import jit

# from KDEpy import FFTKDE

import data_processing_functions as dpf

from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

from pathlib import Path
BASE_DIR = os.getcwd()

from bokeh.plotting import figure, output_file, save
from bokeh.models import LinearColorMapper, ColorBar, ColumnDataSource, CustomJS

# import xyzservices.providers as xyz
# tiles = xyz['USGS']['USTopo']

# go up one level to the root of the project
HYSETS_DIR = Path(BASE_DIR).parent / 'common_data' / 'HYSETS'

@jit
def compute_overlap_matrix(mask_jax):
    return jnp.matmul(mask_jax, mask_jax.T)

class FDCEstimationContext:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._load_catchment_data()
        self._load_and_filter_hysets_data()
        self.LN_param_dict = self._get_ln_params()
        self.id_to_idx, self.idx_to_id = self._set_tree_idx_mappers()
        self._build_network_trees()
        self._set_attribute_indexers()
        self.overlap_dict = self._compute_concurrent_overlap_dict()
        
    
    def _load_and_filter_hysets_data(self):
        hs_df = pd.read_csv(HYSETS_DIR / 'HYSETS_watershed_properties.txt', sep=';')
        hs_df = hs_df[hs_df['Official_ID'].isin(self.station_ids)]
        self.global_start_date = pd.to_datetime('1950-01-01') # Hysets streamflow starts 1950-01-01
        self.hs_df = hs_df
        self.official_ids = hs_df['Official_ID'].values
        print(f'Use only stations with minimum concurrency with Daymet / LSTM results: (n={len(self.official_ids)})')

        # load the updated HYSETS data

        updated_filename = 'HYSETS_2023_update_QC_stations.nc'
        ds = xr.open_dataset(HYSETS_DIR / updated_filename)

        # Get valid IDs as a NumPy array
        selected_ids = hs_df['Watershed_ID'].values

        # Get boolean index where watershedID in selected_set
        # safely access watershedID as a variable first
        ws_ids = ds['watershedID'].data  # or .values if you prefer
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
        gdf = gpd.read_file(self.attr_gdf_fpath)
        gdf.columns = [c.lower() for c in gdf.columns]

        # LSTM ensemble predictions
        lstm_result_folder = '/home/danbot2/code_5820/neuralhydrology/data/ensemble_results'
        lstm_result_files = os.listdir(lstm_result_folder)
        lstm_result_stns = [e.split('_')[0] for e in lstm_result_files]

        # filter for the common stations between BCUB region and LSTM-compatible (i.e. 1980-)
        self.daymet_concurrent_stations = list(set(station_ids) & set(lstm_result_stns))
        print(f'There are {len(daymet_concurrent_stations)} monitored basins concurrent with LSTM ensemble results.')
        if self.LSTM_concurrent_network == True:
            self.official_ids = self.daymet_concurrent_stations
        else:
            self.official_ids = station_ids
        
        gdf = gdf[gdf['official_id'].isin(self.official_ids)]
        # import the license water extraction points
        dam_gdf = gpd.read_file('data/Dam_Points_20240103.gpkg')
        assert gdf.crs == dam_gdf.crs, "CRS of catchment and dam dataframes do not match"
        joined = gpd.sjoin(gdf, dam_gdf, how="inner", predicate="contains")
        joined = joined[joined['official_id'].isin(self.official_ids)]
        # Create a new boolean column 'contains_dam' in catchment_gdf.
        # If a polygon's index appears in the joined result, it means it contains at least one point.
        regulated = joined['official_id'].values
        N = len(gdf)
        print(f'{len(regulated)}/{N} catchments contain withdrawal licenses')
                
        # create dict structures for easier access of attributes and geometries
        self.da_dict = gdf[['official_id', 'drainage_area_km2']].set_index('official_id').to_dict()['drainage_area_km2']
        self.polygon_dict = gdf[['official_id', 'geometry']].set_index('official_id').to_dict()['geometry']
        
        centroids = gdf.apply(lambda x: self.polygon_dict[x['official_id']].centroid, axis=1)
        attr_gdf = gpd.GeoDataFrame(gdf, geometry=centroids, crs=gdf.crs)
        attr_gdf["contains_dam"] = attr_gdf['official_id'].apply(lambda x: x in regulated)
        attr_gdf.reset_index(inplace=True, drop=True)
        self.attr_gdf = attr_gdf


    def _build_network_trees(self, attribute_cols=['log_drainage_area_km2', 'elevation_m', 'prcp', 'tmean', 'swe', 'srad',
                            'centroid_lon_deg_e', 'centroid_lat_deg_n', 'land_use_forest_frac_2010', 'land_use_snow_ice_frac_2010',
                            #  , 'land_use_wetland_frac_2010', 'land_use_water_frac_2010', 
                            ]):
        self.coords = np.array([[geom.x, geom.y] for geom in self.attr_gdf.geometry.values])
        self.spatial_tree = cKDTree(self.coords)

        # Create mapping from official_id to index
        self.id_to_index = {oid: i for i, oid in enumerate(self.attr_gdf["official_id"])}
        self.index_to_id = {i: oid for oid, i in self.id_to_index.items()}  # Reverse mapping

        # Extract values (excluding 'official_id' since it's categorical)
        self.attr_gdf['tmean'] = (self.attr_gdf['tmin'] + self.attr_gdf['tmax']) / 2.0
        self.attr_gdf['log_drainage_area_km2'] = np.log(self.attr_gdf['drainage_area_km2'])
        attr_values = self.attr_gdf[attribute_cols].to_numpy()
        scaler = StandardScaler()
        self.normalized_attr_values = scaler.fit_transform(attr_values)
        # Construct the KDTree for the attribute space
        self.attribute_tree = cKDTree(self.normalized_attr_values)


    def _set_tree_idx_mappers(self):
        """Set the index mappers for the official_id to index and vice versa.
        This is for the network TREES"""
        id_to_idx = {id: i for i, id in enumerate(self.attr_gdf['official_id'].values)}
        idx_to_id = {i: id for i, id in enumerate(self.attr_gdf['official_id'].values)}

        return id_to_idx, idx_to_id
    

    def _set_attribute_indexers(self):
        # map keys to their 
        # overlap_dict[1].keys()
        # create a dictionary where the keys are Watershed_ID and the values are Official_ID
        self.watershed_id_dict = {row['Watershed_ID']: row['Official_ID'] for _, row in self.hs_df.iterrows()}
        # and the inverse
        self.official_id_dict = {row['Official_ID']: row['Watershed_ID'] for _, row in self.hs_df.iterrows()}
        # also for drainage areas
        self.da_dict = {row['Official_ID']: row['Drainage_Area_km2'] for _, row in self.hs_df.iterrows()}
        

    def _get_ln_params(self):
        """Retrieve log-normal parameters for a station."""

        predicted_param_dict = {}
        for t in self.parametric_target_cols:
            print(t)
            fpath = os.path.join(self.parameter_prediction_results_folder, f'best_out_of_sample_{t}_predictions.csv')
            rdf = pd.read_csv(fpath, index_col='official_id')
            rdf = rdf[[c for c in rdf.columns if not c.startswith('Unnamed:')]].sort_values('official_id')
            predicted_param_dict[t] = rdf.to_dict(orient='index')
        return predicted_param_dict
    

    def _generate_12_month_windows(self, index):
        months = pd.date_range(index.min(), index.max(), freq='MS')
        windows = [(start, start + pd.DateOffset(months=12) - pd.Timedelta(days=1)) for start in months]
        return [w for w in windows if w[1] <= index.max()]
    
    
    def _is_window_valid(self, ts, start, end):
        window = ts.loc[start:end]
        if window.empty:
            return False
        grouped = window.groupby(window.index.month).size()
        if set(grouped.index) != set(range(1, 13)):
            return False
        if grouped.min() < 10:
            return False
        return True
    
    
    def _compute_station_valid_windows(self, ts, windows):
        return [self._is_window_valid(ts, start, end) for (start, end) in windows]
    
    
    def _count_valid_shared_windows(self, valid_i, valid_j):
        return sum(np.logical_and(valid_i, valid_j))
    
    
    def _compute_concurrent_overlap_dict(self, variable='discharge'):
        """
        Compute the concurrent overlap of monitored watersheds in the dataset.
        Threshold years represent the minimum number of days of overlap 
        (ignoring continuity) for a watershed to be considered concurrent.
        """
        overlap_dict_file = os.path.join('data', 'record_overlap_dict.json')
        if os.path.exists(overlap_dict_file):
            with open(overlap_dict_file, 'r') as f:
                overlap_dict = json.load(f)
            print(f'    ...overlap dict loaded from {overlap_dict_file}')
            return overlap_dict
        
        watershed_ids = self.ds['watershed'].values
        data = self.ds[variable].load().values  # (N, T)
        threshold_years = self.minimum_concurrent_years

        # Compute mask on GPU
        M = jnp.asarray(~np.isnan(data), dtype=jnp.uint16) # 
        O = compute_overlap_matrix(M)

        N = M.shape[0] # number of sites
        T = M.shape[1] # number of time steps
        connectivity_factor = np.sum(O) / float(N**2 * T)
        print(f'Connectivity factor: {connectivity_factor:.4f}')

        # Build output
        N = len(watershed_ids)
        print(f'    ...building overlap dict for N={N} monitored watersheds in the network.')
        thresholds_days = 365 * np.array(threshold_years) 
        overlap_dict = {t: {} for t in threshold_years}

        for t_years, t_days in zip(threshold_years, thresholds_days):
            over_thresh = O >= t_days
            over_thresh = over_thresh.at[jnp.diag_indices(N)].set(False)

            over_thresh_np = np.array(over_thresh)
            for i in range(N):
                ws = int(watershed_ids[i])
                overlap_dict[t_years][ws] = list(watershed_ids[over_thresh_np[i]])

        # Save the overlap dictionary to a JSON file
        with open(overlap_dict_file, 'w') as f:
            json.dump(overlap_dict, f)
        print(f'    ...overlap dict saved to {overlap_dict_file}')

        return overlap_dict

np.random.seed(42)

# rev_date = '20250227'
# # from utils import FDCEstimationContext
# attr_gdf_fpath = os.path.join('data', f'BCUB_watershed_attributes_updated_{rev_date}.geojson')
# LSTM_forcings_folder = '/home/danbot2/code_5820/neuralhydrology/data/BCUB_catchment_mean_met_forcings_20250320'
# LSTM_ensemble_result_folder = '/home/danbot/code/neuralhydrology/data/ensemble_results'
# parameter_prediction_results_folder = os.path.join('data', 'parameter_prediction_results')
bcub_stn_file = Path(BASE_DIR) / 'docs' / 'notebooks' / 'data' / 'BCUB_watershed_attributes_updated_20250227.csv'
bcub_df = pd.read_csv(bcub_stn_file)
bcub_gdf = gpd.GeoDataFrame(bcub_df, geometry=gpd.points_from_xy(bcub_df['centroid_lon_deg_e'], bcub_df['centroid_lat_deg_n']), crs='EPSG:4326')
bcub_gdf = bcub_gdf.to_crs('EPSG:3005')


def load_and_filter_hysets_data(official_ids):
    
    hs_df = pd.read_csv(HYSETS_DIR / 'HYSETS_watershed_properties.txt', sep=';')
    hs_df = hs_df[hs_df['Official_ID'].isin(official_ids)]

    updated_filename = 'HYSETS_2023_update_QC_stations.nc'
    hs_data_path = HYSETS_DIR / updated_filename
    assert os.path.exists(hs_data_path), f"HYSETS data file {updated_filename} not found in {HYSETS_DIR}"
    ds = xr.open_dataset(hs_data_path)

    # Get valid IDs as a NumPy array
    selected_ids = hs_df['Watershed_ID'].values

    # Get boolean index where watershedID in selected_set
    # safely access watershedID as a variable first
    ws_ids = ds['watershedID'].data  # or .values if you prefer
    mask = np.isin(ws_ids, selected_ids)

    # Apply mask to data
    ds = ds.sel(watershed=mask)
    # Step 1: Promote 'watershedID' to a coordinate on the 'watershed' dimension
    ds = ds.assign_coords(watershedID=("watershed", ds["watershedID"].data))

    # Step 2: Set 'watershedID' as the index for the 'watershed' dimension
    ds = ds.set_index(watershed="watershedID")
    return ds
    

# Input: discharge DataArray with dims ('watershed', 'time')
discharge = load_and_filter_hysets_data(bcub_gdf['official_id'].values)  # Load HYSETS data

print('Loaded HYSETS data.')

# Convert time to weeklly binning periods
# week = pd.to_datetime(discharge.time.values).to_period("W").to_timestamp()
# discharge.coords["week"] = ("time", week)
# Convert time to monthly binning periods
month = pd.to_datetime(discharge.time.values).to_period("M").to_timestamp()
discharge.coords["month"] = ("time", month)
print('Converted time to monthly periods.')

# Count non-NaNs per (watershed, week)
# weekly_counts = discharge.groupby("month").map(lambda x: np.isfinite(x).sum(dim="time"))
# monthly_counts = discharge.groupby("month").map(lambda x: np.isfinite(x).sum(dim="time"))
monthly_counts = np.isfinite(discharge).resample(time="1MS").sum(dim="time")
monthly_counts["month"] = monthly_counts.time.dt.month
# print('processed weekly counts:')
print('processed monthly counts:')
# # Binary mask: 1 if ≥3 valid values that week
# weekly_mask = (weekly_counts >= 3)
# monthly_mask = (monthly_counts >= 20) # minimum 20 days to be considered a valid month
monthly_mask = (
    np.isfinite(discharge)
    .groupby("time.month")
    .sum(dim="time")  # shape: ('month', 'watershed')
)

# Transpose for plotting (shape: weeks × watersheds)
# img = weekly_mask.transpose("week", "watershed").astype(float).values[::-1, :]
img = monthly_mask.transpose("month", "watershed").astype(float).values[::-1, :]
print('generated monthly mask image with shape:', img.shape)
# Store the image as a 1-item list for Bokeh
source = ColumnDataSource(data=dict(image=[img]))

months = pd.to_datetime(monthly_mask['month'].values)
watersheds = monthly_mask['watershed'].values
n_watersheds = len(watersheds)
print(f'Number of monitored watersheds: {n_watersheds}')

output_file("interactive_data_availability.html")

mapper = LinearColorMapper(palette=["white", "black", "dodgerblue"], low=0, high=2)

p = figure(
    width=1200,
    height=300,
    y_axis_type="datetime",
    y_range=(weeks.min(), weeks.max()),
    x_range=(0, n_watersheds),
    title='Click a sensor column to highlight nearby columns',
    tools="tap"
)

# Plot the image from source
p.image(
    image="image",
    x=0,
    y=weeks.min(),
    dw=n_watersheds,
    dh=weeks.max() - weeks.min(),
    color_mapper=mapper,
    source=source
)

# TapTool callback to highlight ±10 columns around clicked one
callback = CustomJS(args=dict(source=source), code="""
    const img = source.data.image[0];
    const nrows = img.length;
    const ncols = img[0].length;

    const x = cb_obj.x;  // clicked x (sensor ID)
    const center_col = Math.round(x);

    // Create new image with dodgerblue (value = 2) columns highlighted
    const new_img = [];
    for (let r = 0; r < nrows; r++) {
        new_img.push([]);
        for (let c = 0; c < ncols; c++) {
            const base = img[r][c] === 1 ? 1 : 0;
            if (Math.abs(c - center_col) <= 10 && base === 1)
                new_img[r].push(2);  // highlight
            else
                new_img[r].push(base);  // keep original
        }
    }

    source.data.image = [new_img];
    source.change.emit();
""")

# Attach callback
p.js_on_event('tap', callback)

save(p)