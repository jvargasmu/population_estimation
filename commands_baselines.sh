#!/bin/bash

# List of Commands

##### TZA #####

#### Preprocessing commands ####
time python preprocessing_pop_data_v2.py ~/data/wpop/OtherBoundaries/TZA/tza_admbnda_adm3_20181019/tza_admbnda_adm3_20181019.shp ~/data/wpop/OtherBoundaries/TZA/tza_adm3_sid.tif ~/data/wpop/OtherMastergrid/TZA/tza_subnational_2000_2020_sid.tif ~/data/wpop/OtherCensusTables/tza_population_2000_2020_sid.csv ~/data/wpop/preproc2/tza_preproc_input.pkl tza P_2020

### Building disaggregation


### WorldPop - trained with all coarse census data
time python train_model_with_agg_data.py -pre ~/data/wpop/preproc2/tza_preproc_input.pkl -adm_rst ~/data/wpop/OtherMastergrid/TZA/tza_subnational_2000_2020_sid.tif -out ~/data/wpop/predictions/wpop_tza_c_all_rs1610/ -data tza -bu True -e5f False -train_lvl c -rs 1610

### WorldPop - trained with coarse census data, using the split 3/1/1
time python train_model_with_agg_data.py -pre ~/data/wpop/preproc2/tza_preproc_input.pkl -adm_rst ~/data/wpop/OtherMastergrid/TZA/tza_subnational_2000_2020_sid.tif -out ~/data/wpop/predictions/wpop_tza_c_311_rs1610/ -data tza -bu True -e5f True -train_lvl c -rs 1610

### WorldPop - trained with fine scale census data, using the split 3/1/1
time python train_model_with_agg_data.py -pre ~/data/wpop/preproc2/tza_preproc_input.pkl -adm_rst ~/data/wpop/OtherMastergrid/TZA/tza_subnational_2000_2020_sid.tif -out ~/data/wpop/predictions/wpop_tza_f_311_rs1610/ -data tza -bu True -e5f True -train_lvl f -rs 1610

### MRF (adj.)

