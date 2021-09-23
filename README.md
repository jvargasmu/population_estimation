# population_estimation
Implementation of methods for population estimation


Change the file config_pop.py to define the covariates tif files that you have available on your computer

- Example to preprocess data:

time python preprocessing_pop_data.py world_pop_admin_boundaries.shp raster_world_pop_admin_boundaries.tif raster_humdata_admin_boundaries.tif census_table.csv output_preprocessed_data.pkl tza

- Example to run the building disaggregation baseline:

time python building_disagg_baseline.py preprocessed_data.pkl raster_world_pop_admin_boundaries.tif output_dir/ P_2020 tza

