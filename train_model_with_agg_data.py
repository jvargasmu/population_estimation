import os
import argparse
import pickle
import numpy as np
from osgeo import gdal
from utils import read_input_raster_data, compute_performance_metrics, write_geolocated_image, create_map_of_valid_ids, \
    compute_grouped_values, transform_dict_to_array, transform_dict_to_matrix
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region
import config_pop as cfg
from building_disagg_baseline import disaggregate_weighted_by_preds
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from distutils.util import strtobool
from utils import compute_grouped_values


def compute_density(areas, census, id_list):
    density = {}
    for id in id_list:
        if areas[id] == 0:
            density[id] = 0
        else:
            density[id] = census[id] / areas[id]
    return density


def compute_avg_feats(feats_list, features, valid_ids, id_to_cr_id, areas, grouped_area):
    # Initialize feature values
    grouped_features = {}
    for id in valid_ids:
        cr_id = id_to_cr_id[id]
        if cr_id not in grouped_features.keys():
            grouped_features[cr_id] = {elem: 0 for elem in feats_list}
    # Aggregate targets
    for feat in feats_list:
        for id in valid_ids:
            cr_id = id_to_cr_id[id]
            grouped_features[cr_id][feat] += features[id][feat] * (areas[id] / grouped_area[cr_id])

    return grouped_features


def get_all_pixel_features(inputs, feats_list):
    inputs_mat = []
    for feat in feats_list:
        inputs_mat.append(inputs[feat])
    inputs_mat = np.array(inputs_mat)
    height = inputs_mat.shape[1]
    width = inputs_mat.shape[2]

    all_features = inputs_mat.reshape((inputs_mat.shape[0], -1))
    all_features = all_features.transpose()
    print("all_features shape {}".format(all_features.shape))
    
    return all_features, height, width


def perform_prediction_at_pixel_level(inputs, feats_list, model):
    all_features, height, width = get_all_pixel_features(inputs, feats_list)

    predictions = model.predict(all_features)

    return predictions.reshape((height, width))


def compute_performance_metrics_from_dict(preds_dict, gt_dict):
    assert len(preds_dict) == len(gt_dict)

    preds = []
    gt = []
    ids = preds_dict.keys()
    for id in ids:
        preds.append(preds_dict[id])
        gt.append(gt_dict[id])

    preds = np.array(preds).astype(np.float)
    gt = np.array(gt).astype(np.float)

    r2 = r2_score(gt, preds)
    mae = mean_absolute_error(gt, preds)
    mse = mean_squared_error(gt, preds)

    return r2, mae, mse


def select_subset_dict(data_dict, choice_ind, offset=0):
    return {ind+offset:data_dict[ind+offset] for ind in choice_ind}


def get_finest_level_indexes(id_to_cr_id, choice_ind_c):
    set_choice_ind_c = set(choice_ind_c)
    choice_ind_f = []
    for id in range(len(id_to_cr_id)):
        cr_id = id_to_cr_id[id]
        ind_cr_id = cr_id - 1
        if ind_cr_id in set_choice_ind_c:
            choice_ind_f.append(id)
    return np.array(choice_ind_f)    


def perform_rf_parameter_search(train_features, train_labels, val_features, val_labels, log_of_target):
    n_estimators_values = np.arange(20,201,20)
    max_depth_values = list(np.arange(4, 21, 4)) + [None]
    best_accuracy = -999999
    best_n_estimators = None
    best_max_depth = None
    for n_estimators_val in n_estimators_values:
        for max_depth_val in max_depth_values:
            clf = RandomForestRegressor(random_state=42, n_jobs=4, n_estimators=n_estimators_val, max_depth=max_depth_val)
            final_train_labels = train_labels
            
            if log_of_target:
                final_train_labels = np.log(train_labels)
            
            clf.fit(train_features, final_train_labels)
            val_preds = clf.predict(val_features)
            if log_of_target:
                val_preds = np.exp(val_preds)
            
            acc = r2_score(val_labels, val_preds)
            if acc > best_accuracy:
                best_accuracy = acc
                best_n_estimators = n_estimators_val
                best_max_depth = max_depth_val
    print("best_r2 {}".format(best_accuracy))
    return best_n_estimators, best_max_depth


def train_model_with_agg_data(preproc_data_path, rst_wp_regions_path, output_dir, dataset_name, 
                              built_up_area_agg, eval_5fold, train_level, random_seed, population_target, log_of_target):
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("built_up_area_agg {}".format(built_up_area_agg))
    print("eval_5fold {}".format(eval_5fold))
    # Read input data
    input_paths = cfg.input_paths[dataset_name]

    with open(preproc_data_path, 'rb') as handle:
        pdata = pickle.load(handle)

    cr_census_arr = pdata["cr_census_arr"]
    valid_ids = pdata["valid_ids"]
    no_valid_ids = pdata["no_valid_ids"]
    id_to_cr_id = pdata["id_to_cr_id"]
    valid_census = pdata["valid_census"]
    num_coarse_regions = pdata["num_coarse_regions"]
    geo_metadata = pdata["geo_metadata"]
    areas = pdata["areas"]
    if built_up_area_agg:
        areas = pdata["built_up_areas"]
    wp_rst_regions = gdal.Open(rst_wp_regions_path).ReadAsArray().astype(np.uint32)
    wp_ids = list(np.unique(wp_rst_regions))
    num_wp_ids = len(wp_ids)
    inputs = read_input_raster_data(input_paths)
    input_buildings = inputs["buildings"]

    # Binary map representing a pixel belong to a region with valid id
    map_valid_ids = create_map_of_valid_ids(wp_rst_regions, no_valid_ids)

    # Get map of coarse level regions
    cr_regions = compute_map_with_new_labels(wp_rst_regions, id_to_cr_id, map_valid_ids)

    # Compute area of coarse regions
    cr_areas = compute_grouped_values(areas, valid_ids, id_to_cr_id)

    # Compute average features at the coarse level
    feats_list = inputs.keys()
    if not population_target:
        feats_list = [feat for feat in feats_list if feat != "buildings"]
    
    features = pdata["features"]
    if built_up_area_agg:
        features = pdata["features_from_built_up_areas"]
    
    building_counts = {}
    target_norm = areas
    cr_target_norm = cr_areas
    if not population_target:
        
        for id in features.keys():
            building_counts[id] = features[id]["buildings"] * areas[id]
            del features[id]["buildings"]
        
        # Compute the number of buildings per region
        cr_building_counts = compute_grouped_values(building_counts, valid_ids, id_to_cr_id)
        
        target_norm = building_counts
        cr_target_norm = cr_building_counts

    # Create model
    if eval_5fold:
        n_folds = 5
        all_pixel_features, height, width = get_all_pixel_features(inputs, feats_list)
        # Split dataset in folds, using same splits as the ones used for ScaleNet
        #np.random.seed(1610)
        np.random.seed(random_seed)
        trainidxs, validxs, houtidxs = [],[],[]
        n_samples = len(cr_areas)
        n_splits = n_folds
        for spl in range(n_splits):
            orig_indices = np.arange(n_samples)
            np.random.shuffle(orig_indices)
            idx_offset = n_samples
            indices = np.concatenate((orig_indices, orig_indices, orig_indices))

            fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
            fold_sizes[: n_samples % n_splits] += 1
            current = 0
            for fold_size in fold_sizes:
                val_start, val_stop = current, current + fold_size
                hout_start, hout_stop = current - fold_size, current
                train_start, train_stop = current + fold_size, current + fold_size * (n_splits - 2)
                
                trainidxs.append(indices[idx_offset+train_start:idx_offset+train_stop])
                validxs.append(indices[idx_offset+val_start:idx_offset+val_stop])
                houtidxs.append(indices[idx_offset+hout_start:idx_offset+hout_stop])
                
                current = val_stop
        
        final_pred_map = np.zeros((height, width), dtype=np.float32)
        for validation_fold in range(n_folds):
            choice_val_c = validxs[validation_fold]
            choice_hout_c = houtidxs[validation_fold]
            # indices of the train set
            ind_val_hout_c = np.zeros(len(cr_areas), dtype=bool) 
            ind_val_hout_c[choice_val_c] = True 
            ind_val_hout_c[choice_hout_c] = True 
            ind_train_c = ~ind_val_hout_c
            all_cr_indexes = np.arange(len(cr_areas))
            choice_train_c = all_cr_indexes[ind_train_c]

            id_offset = 0
            if train_level == 'c':
                id_offset = 1
                cr_features = compute_avg_feats(feats_list, features, valid_ids, id_to_cr_id, areas, cr_areas)
                density = compute_density(cr_target_norm, cr_census_arr, list(cr_areas.keys()))
                # Obtain features
                features_train = select_subset_dict(cr_features, choice_train_c, offset=id_offset)
                features_val = select_subset_dict(cr_features, choice_val_c, offset=id_offset)
                features_hout = select_subset_dict(cr_features, choice_hout_c, offset=id_offset)
                features_train_arr = transform_dict_to_matrix(features_train)
                features_val_arr = transform_dict_to_matrix(features_val)
                choice_train = choice_train_c
                choice_val = choice_val_c
                choice_hout = choice_hout_c
            else:
                # Obtain finest level regions that correspond to the coarse regions selected 
                choice_train = get_finest_level_indexes(id_to_cr_id, choice_train_c)
                choice_val = get_finest_level_indexes(id_to_cr_id, choice_val_c)
                choice_hout = get_finest_level_indexes(id_to_cr_id, choice_hout_c)
                # Obtain features
                features_train = select_subset_dict(features, choice_train, offset=id_offset)
                features_val = select_subset_dict(features, choice_val, offset=id_offset)
                features_hout = select_subset_dict(features, choice_hout, offset=id_offset)
                features_train_arr = transform_dict_to_matrix(features_train)
                features_val_arr = transform_dict_to_matrix(features_val)
                density = compute_density(target_norm, valid_census, list(valid_census.keys())) #TODO: verify if is correct
            
            # Compute log of density to be used as target for training the model
            density_train = select_subset_dict(density, choice_train, offset=id_offset)
            density_train_arr = transform_dict_to_array(density_train)
            density_val = select_subset_dict(density, choice_val, offset=id_offset)
            density_val_arr = transform_dict_to_array(density_val)
            density_hout = select_subset_dict(density, choice_hout, offset=id_offset)
            density_hout_arr = transform_dict_to_array(density_hout)
            # remove samples with density equal to 0 because when taking the log it does not work that well
            mask_valid_train_samples = density_train_arr > 0
            valid_features_train_arr = features_train_arr[mask_valid_train_samples, :]
            valid_density_train_arr = density_train_arr[mask_valid_train_samples]
            mask_valid_val_samples = density_val_arr > 0
            valid_features_val_arr = features_val_arr[mask_valid_val_samples, :]
            valid_density_val_arr = density_val_arr[mask_valid_val_samples]
            # obtain best RF paramenters
            best_n_estimators, best_max_depth = perform_rf_parameter_search(valid_features_train_arr, valid_density_train_arr, 
                                                                         valid_features_val_arr, valid_density_val_arr, log_of_target)
            
            # train the model in using the current fold training dataset
            model = RandomForestRegressor(random_state=42, n_jobs=4, n_estimators=best_n_estimators, max_depth=best_max_depth)
            final_valid_density_train_arr = valid_density_train_arr
            if log_of_target:
                final_valid_density_train_arr = np.log(valid_density_train_arr)
            model.fit(valid_features_train_arr, final_valid_density_train_arr)
            print("model fold {} feature importance {}".format(validation_fold, model.feature_importances_))
            
            predictions = model.predict(all_pixel_features)
            pred_map = predictions.reshape((height, width))
            if log_of_target:
                pred_map = np.exp(pred_map)
            pred_map = pred_map.astype(np.float32)
            
            for ind_cr_id in choice_hout_c:
                cr_id = ind_cr_id+1
                final_pred_map[cr_regions==cr_id] = pred_map[cr_regions==cr_id]

            
        pred_map = final_pred_map    
        
    else:

        cr_features = compute_avg_feats(feats_list, features, valid_ids, id_to_cr_id, areas, cr_areas)
        cr_features_arr = transform_dict_to_matrix(cr_features)

        # Compute WorldPop target : log of density
        cr_density = compute_density(cr_target_norm, cr_census_arr, list(cr_areas.keys()))
        cr_density_arr = transform_dict_to_array(cr_density)
        final_cr_density_arr = cr_density_arr
        if log_of_target:
            final_cr_density_arr = np.log(cr_density_arr)
        # Fit model
        model = RandomForestRegressor(random_state=42, n_jobs=4)
        model.fit(cr_features_arr, final_cr_density_arr)
        print("feature importance {}".format(model.feature_importances_))

        # Perform prediction per pixel
        pred_map = perform_prediction_at_pixel_level(inputs, feats_list, model)
        if log_of_target:
            pred_map = np.exp(pred_map)        
        pred_map = pred_map.astype(np.float32)

    if not population_target:
        preproc_input_buildings = np.multiply(input_buildings, np.multiply(input_buildings > 0, (input_buildings < 255)))
        pred_map = pred_map * preproc_input_buildings
    
    # Get building maps with values between 0 and 1 (sometimes 255 represent no data values)
    unnorm_weights = pred_map.copy()
    mask = np.multiply(input_buildings > 0, (input_buildings < 255))
    
    # Compute accuracy before disaggregation
    pred_map_masked = pred_map
    if mask is not None:
        final_mask = np.multiply((map_valid_ids == 1).astype(np.float32), mask.astype(np.float32))
        pred_map_masked = np.multiply(pred_map, final_mask)
    orig_agg_preds_arr = compute_accumulated_values_by_region(wp_rst_regions, pred_map_masked, map_valid_ids, num_wp_ids)
    orig_agg_preds = {id: orig_agg_preds_arr[id] for id in valid_ids}
    orig_metrics = compute_performance_metrics(orig_agg_preds, valid_census)
    print("Metrics before disagg r2 {} mae {} mse {} mape {}".format(orig_metrics["r2"], orig_metrics["mae"], orig_metrics["mse"], orig_metrics["mape"]))
    
    # Disaggregate population using pred maps as weights
    disagg_population = disaggregate_weighted_by_preds(cr_census_arr, unnorm_weights,
                                                       map_valid_ids, cr_regions, num_coarse_regions, output_dir,
                                                       mask=mask, save_images=True, geo_metadata=geo_metadata,
                                                       return_global_scale=False)

    # Aggregate pixel level predictions to the finest level region
    agg_preds_arr = compute_accumulated_values_by_region(wp_rst_regions, disagg_population, map_valid_ids, num_wp_ids)
    agg_preds = {id: agg_preds_arr[id] for id in valid_ids}

    preds_and_gt_dict = {}
    for id in valid_census.keys():
        preds_and_gt_dict[id] = {"pred": agg_preds[id], "gt": valid_census[id]}

    # Save predictions
    preds_and_gt_path = "{}preds_and_gt.pkl".format(output_dir)
    with open(preds_and_gt_path, 'wb') as handle:
        pickle.dump(preds_and_gt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Compute metrics
    metrics = compute_performance_metrics(agg_preds, valid_census)
    print("Metrics after disagg r2 {} mae {} mse {} mape {}".format(metrics["r2"], metrics["mae"], metrics["mse"], metrics["mape"]))
    

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--preproc_data_path", "-pre", type=str, default="", help="Preprocessed data of regions (pickle file)")
    parser.add_argument("--rst_wp_regions_path", "-adm_rst", type=str, default="",
                        help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("--output_dir", "-out", type=str, default="", help="Output dir ")
    parser.add_argument("--dataset_name", "-data", type=str, default="", help="Dataset name")
    parser.add_argument("--built_up_area_agg", "-bu", type=lambda x: bool(strtobool(x)), default=True, help="Flag that indicates if we should aggregate features using only the built up area")
    parser.add_argument("--eval_5fold", "-e5f", type=lambda x: bool(strtobool(x)), default=False, help="Perform 5 fold validation")
    parser.add_argument("--train_level", "-train_lvl", type=str, default="f", help="Train census level: c (coarse), f (finest)")
    parser.add_argument("--random_seed", "-rs", type=int, default=1610, help="Random seed used to dataset splitting")
    parser.add_argument("--population_target", "-pop_target", type=lambda x: bool(strtobool(x)), default=True, help="Use population as target")
    parser.add_argument("--log_of_target", "-log", type=lambda x: bool(strtobool(x)), default=True, help="Apply log to the target")
    args = parser.parse_args()

    train_model_with_agg_data(args.preproc_data_path, args.rst_wp_regions_path,
                             args.output_dir, args.dataset_name, args.built_up_area_agg, args.eval_5fold, args.train_level, 
                             args.random_seed, args.population_target, args.log_of_target)


if __name__ == "__main__":
    main()
