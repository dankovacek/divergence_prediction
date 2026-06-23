"""
Comprehensive Configuration for Flow Duration Curve Distribution Estimation

This module consolidates all configuration constants, assumptions, and parameters
used throughout the distribution estimation pipeline. All hardcoded values should
be defined here to maintain a single source of truth.

Usage:
    from config import Config
    
    bitrates = Config.SUPPORTED_BITRATES
    min_uar = Config.GLOBAL_MIN_UAR
"""

# import os
from pathlib import Path
# from typing import Union, Optional
# import numpy as np


REPO_ROOT = Path(__file__).resolve().parent


class Config:
    """
    Centralized configuration for FDC estimation pipeline.
    All hardcoded constants should be defined here.
    """
    
    # ==================== DATA PATHS ====================
    HYSETS_DIR = Path('/home/danbot/code/common_data/HYSETS')
    HYSETS_FILENAME = 'HYSETS_2023_update_QC_stations.nc'
    
    LSTM_ENSEMBLE_DIR = Path('/home/danbot/code/neuralhydrology/data')
    LSTM_ENSEMBLE_RESULTS_DIR = Path('/home/danbot/code/neuralhydrology/data/ensemble_results_20250514')
    GR4J_RESULTS_DIR = Path('/home/danbot/code/2026/raven_model_for_BC/gr4j-cn_output')
    
    DATA_DIR = Path('notebooks/data')
    # TEMP_DIR = DATA_DIR / 'temp'
    BASELINE_DIST_DIR = DATA_DIR / 'baseline_distributions/'
    RESULTS_DIR = DATA_DIR / 'results'
    FDC_ESTIMATION_RESULTS_DIR = RESULTS_DIR / 'fdc_estimation_results/'  
    COMPLETE_YEAR_STATS_FILE = DATA_DIR / 'complete_year_stats.npy'
        
    ATTRIBUTE_FORCINGS_REV_DATE = '20260203'
    CLIMATE_FORCINGS_DIR = Path(
        f'/home/danbot/code/common_data/BC_Monitored_catchment_mean_met_forcings_{ATTRIBUTE_FORCINGS_REV_DATE}'
    )
    
    ATTRIBUTE_FILE_DATE = '20260203'  # Date of catchment attribute file (for versioning)
    WATERSHED_DESCRIPTORS_FILENAME = f'Watershed_descriptors_{ATTRIBUTE_FILE_DATE}_with_stats.csv'
    WATERSHED_DESCRIPTORS_PATH =  DATA_DIR / WATERSHED_DESCRIPTORS_FILENAME
    
    # ==================== DISCRETIZATION ====================
    DEFAULT_BITRATE = 8
    SUPPORTED_BITRATES = [6, 8, 10]#[5, 6, 8, 10, 12]
   
    
    # ==================== TEMPORAL PARAMETERS ====================
    MIN_YEARS_OF_RECORD = 10  # years
    MIN_DAYS_PER_MONTH = 25  # days (for complete month)
        
    
    # ==================== FLOW THRESHOLDS ====================
    ZERO_FLOW_THRESHOLD = 1e-4  # m³/s
    
    # UAR (Unit Area Runoff) bounds
    GLOBAL_MIN_UAR = 5e-5  # L/s/km²
    GLOBAL_MAX_UAR = 1e4   # L/s/km²
    GLOBAL_MAX_FLOW = 2e4  # m³/s (max discharge threshold)

    EXCLUDED_STATIONS = ['08FA009', '08GA037', '08NC003', '12052500', '12090480', '12107950', '12108450', '12119300', 
                    '12119450', '12200684', '12200762', '12203000', '12409500', '15056070', '15081510',
                    '12323760', '12143700', '12143900', '12398000', '12058800', '12137800', '12100000',
                    '15056030']
    
    # Climate column name mappings (raw -> standardized)
    CLIMATE_COLUMN_MAPPER = {
        'srad (w/m2)': 'srad', 'tmin (degrees c)': 'tmin', 'tmax (degrees c)': 'tmax', 
        'prcp (mm/day)': 'prcp', 'swe (kg/m2)': 'swe', 'vp (pa)': 'vp', 
        'pet (mm/day)': 'pet', 'dayl (s)': 'dayl',
        'high_prcp_freq (fraction)': 'high_prcp_freq', 'low_prcp_freq (fraction)': 'low_prcp_freq',
        'high_prcp_duration (days)': 'high_prcp_duration', 'low_prcp_duration (days)': 'low_prcp_duration'
    }
    
    # Catchment attribute column names (CAMELS-style CSV, land use columns carry _2010 suffix)
    DESCRIPTOR_COLS = [
        # climate
        'prcp', 'tmean', 'swe', # 'pet', 'tmax (degrees c)', 'tmin (degrees c)', 'srad',  'vp',
        'high_prcp_freq', 'low_prcp_freq', 'high_prcp_duration', 'low_prcp_duration',
        # terrain
       'log_drainage_area_km2', 'slope_deg', 'elevation_m',# 'aspect_deg', 
       # soil
       # 'logk_ice_x100', 'porosity_x100', 
       # land use
       'land_use_forest_frac_2010', 'land_use_snow_ice_frac_2010', 
    #    'land_use_shrubs_frac_2010', 'land_use_grass_frac_2010',
    #    'land_use_wetland_frac_2010', 'land_use_crops_frac_2010', 
    #    'land_use_urban_frac_2010', 'land_use_water_frac_2010',
    ]

    # Caravan/HydroATLAS variant: same attributes, but land use columns are
    # unversioned (no _2010 suffix) as produced by _CARAVAN_COL_MAP renaming.
    CARAVAN_DESCRIPTOR_COLS = [
        # climate
        'prcp', 'tmean', 'swe',
        'high_prcp_freq', 'low_prcp_freq', 'high_prcp_duration', 'low_prcp_duration',
        # terrain
        'log_drainage_area_km2', 'slope_deg', 'elevation_m',
        # land use
        'land_use_forest_frac', 'land_use_snow_ice_frac',
    ]
    
    # Terrain attributes for analysis
    # TERRAIN_ATTRIBUTES = [
    #     'slope_deg', 'aspect_deg', 'elevation_m', 'log_drainage_area_km2'
    # ]
    
    # ==================== EVALUATION METRICS ====================
    class Metrics:
        """Evaluation metric configuration and thresholds."""
        # Metric tolerance limits (thresholds for "perfect")
        LIMITS = {
            'kld': 0.001,
            'emd': 0.05,  # L/s/km²
            'nse': 1 - 0.001,  # Flipped: 1.0 is perfect
            'kge': 1 - 0.001,  # Flipped: 1.0 is perfect
            'mean_error': 0.01,
            'pct_vol_bias': 0.01,
            'mean_abs_rel_error': 0.01,
            'rmse': 0.01
        }
        
    # KL divergence delta (max uncertainty from uniform mixture)
    KLD_DELTA_MAX = 0.001
    
    # Bootstrap parameters
    N_BOOTSTRAPS = 1000
    BOOTSTRAP_ALPHA = 0.05  # 95% CI

    
    


    