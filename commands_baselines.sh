#!/bin/bash

# List of Commands

##### TZA #####

#### Preprocessing commands ####
time python preprocessing_pop_data.py ~/data/wpop/OtherBoundaries/TZA/tza_admbnda_adm3_20181019/tza_admbnda_adm3_20181019.shp ~/data/wpop/OtherBoundaries/TZA/tza_adm3_sid.tif ~/data/wpop/OtherMastergrid/TZA/tza_subnational_2000_2020_sid.tif ~/data/wpop/OtherCensusTables/tza_population_2000_2020_sid.csv ~/data/wpop/preproc2/tza_preproc_input.pkl tza P_2020

### Building disaggregation
time python building_disagg_baseline.py --output_dir ~/data/wpop/preproc2/tza_bdisagg/ --train_dataset_name tza

### WorldPop - trained with all coarse census data
time python train_model_with_agg_data.py -pre ~/data/wpop/preproc2/tza_preproc_input.pkl -adm_rst ~/data/wpop/OtherMastergrid/TZA/tza_subnational_2000_2020_sid.tif -out ~/data/wpop/predictions/wpop_tza_c_all_rs42/ -data tza -bu True -e5f False -train_lvl c -rs 42

### WorldPop - trained with coarse census data, using the split 3/1/1
time python train_model_with_agg_data.py -pre ~/data/wpop/preproc2/tza_preproc_input.pkl -adm_rst ~/data/wpop/OtherMastergrid/TZA/tza_subnational_2000_2020_sid.tif -out ~/data/wpop/predictions/wpop_tza_c_311_rs42_rsf1610/ -data tza -bu True -e5f True -train_lvl c -rs 42 -rsf 1610

### WorldPop - trained with fine scale census data, using the split 3/1/1
time python train_model_with_agg_data.py -pre ~/data/wpop/preproc2/tza_preproc_input.pkl -adm_rst ~/data/wpop/OtherMastergrid/TZA/tza_subnational_2000_2020_sid.tif -out ~/data/wpop/predictions/wpop_tza_f_311_rs42_rsf1610/ -data tza -bu True -e5f True -train_lvl f -rs 42 -rsf 1610

### WorldPop - Transfer Task (train and validate with data of one or more countries and obtain the test performance in data of a another country) [-pre_dir is only used if the preprocessing data is in a folder different from the current directory]
time python train_model_with_agg_data_for_transfer_task.py -pre_dir ~/data/wpop/preproc2/ -out ~/data/wpop/preproc2/tza_train_f_test_uga/ -train tza -test uga -bu True -e5s True -tperc 0.8 -train_lvl f -rs 42 -rsf 1610 -pop_target True -log True
# time python train_model_with_agg_data_for_transfer_task.py -pre_dir ~/data/wpop/preproc2/ -out ~/data/wpop/preproc2/tza_train_f_test_uga/ -train rwa,uga,cod,nga,moz -test tza -bu True -e5s True -tperc 0.8 -train_lvl f -rs 42 -rsf 1610 -pop_target True -log True
# time python train_model_with_agg_data_for_transfer_task.py -pre_dir . -out wpop/preproc2/transfer_to_tza/ -train rwa,uga,cod,nga,moz -test tza -bu True -e5s True -tperc 0.8 -train_lvl f -rs 42 -rsf 1610 -pop_target True -log True
# time python train_model_with_agg_data_for_transfer_task.py -pre_dir . -out wpop/preproc2/transfer_to_moz/ -train rwa,uga,cod,nga,tza -test moz -bu True -e5s True -tperc 0.8 -train_lvl f -rs 42 -rsf 1610 -pop_target True -log True
# time python train_model_with_agg_data_for_transfer_task.py -pre_dir . -out wpop/preproc2/transfer_to_rwa/ -train tza,uga,cod,nga,moz -test rwa -bu True -e5s True -tperc 0.8 -train_lvl f -rs 42 -rsf 1610 -pop_target True -log True

### MRF with 3 features (count buildings, average size buildings, night light images)
time python preprocessing_pop_data.py ~/data/wpop/OtherBoundaries/TZA/tza_admbnda_adm3_20181019/tza_admbnda_adm3_20181019.shp ~/data/wpop/OtherBoundaries/TZA/tza_adm3_sid.tif ~/data/wpop/OtherMastergrid/TZA/tza_subnational_2000_2020_sid.tif ~/data/wpop/OtherCensusTables/tza_population_2000_2020_sid.csv ~/data/wpop/preproc2/tza_f3_preproc_input.pkl tza_f3 P_2020
time python compute_graph.py ~/data/wpop/preproc2/tza_f3_preproc_input.pkl ~/data/wpop/OtherMastergrid/TZA/tza_subnational_2000_2020_sid.tif ~/data/wpop/preproc2/tza_f3_mrf/ tza_f3 15 1.0 10
time python train_mrf.py ~/data/wpop/preproc2/tza_f3_preproc_input.pkl ~/data/wpop/OtherMastergrid/TZA/tza_subnational_2000_2020_sid.tif ~/data/wpop/preproc2/tza_f3_mrf/ tza_f3 0.1 3 0.1 ~/data/wpop/preproc2/tza_f3_mrf/graph_ind_k_15_sub_100.npy ~/data/wpop/preproc2/tza_f3_mrf/graph_dist_k_15_sub_100.npy

### MRF with all the 15 features
time python compute_graph.py ~/data/wpop/preproc2/tza_preproc_input.pkl ~/data/wpop/OtherMastergrid/TZA/tza_subnational_2000_2020_sid.tif ~/data/wpop/preproc2/tza_mrf/ tza 15 1.0 10
time python train_mrf.py ~/data/wpop/preproc2/tza_preproc_input.pkl ~/data/wpop/OtherMastergrid/TZA/tza_subnational_2000_2020_sid.tif ~/data/wpop/preproc2/tza_mrf/ tza_f3 0.1 3 0.1 ~/data/wpop/preproc2/tza_mrf/graph_ind_k_15_sub_100.npy ~/data/wpop/preproc2/tza_mrf/graph_dist_k_15_sub_100.npy
