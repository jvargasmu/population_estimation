# Population Estimation
Implementation of methods for population estimation


Change the file config_pop.py to define the covariates tif files that you have available on your computer

- Example to preprocess data:

time python preprocessing_pop_data.py world_pop_admin_boundaries.shp raster_world_pop_admin_boundaries.tif raster_humdata_admin_boundaries.tif census_table.csv output_preprocessed_data.pkl tza P_2020

- Example to run the building disaggregation baseline:

time python building_disagg_baseline.py preprocessed_data.pkl raster_world_pop_admin_boundaries.tif output_dir/ tza

- Example to run the WorldPop baseline:

time python train_model_with_agg_data.py preprocessed_data.pkl raster_world_pop_admin_boundaries.tif output_dir/ tza

- Example to run Superpixel script:


time python superpixel_disagg_model.py -train tza,uga -train_lvl c,f -test tza,uga -lr 0.0001 -optim adam -wr 0.001 -lstep 400 



## Installation Steps

 - 'git clone' this repo
 - 'virtualenv -p python3 env'
 - 'source HACenv/bin/activate'
 - 'pip install --upgrade pip'
 - 'pip install gdal==3.2.1'
 - 'pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==\`gdal-config --version'
 - 'pip install numpy torch matplotlib h5py wandb tqdm torchvision fiona sklearn'