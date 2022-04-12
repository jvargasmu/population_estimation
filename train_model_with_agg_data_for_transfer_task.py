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
from superpixel_disagg_model import unroll_arglist
from train_model_with_agg_data import compute_density, compute_avg_feats, get_all_pixel_features, perform_prediction_at_pixel_level, \
    compute_performance_metrics_from_dict, select_subset_dict, get_finest_level_indexes, perform_rf_parameter_search


def get_dataset(dataset_name, preproc_data_dir, built_up_area_agg, population_target):
    
    rst_wp_regions_path = cfg.metadata[dataset_name]["rst_wp_regions_path"]
    preproc_data_path = os.path.join(preproc_data_dir, cfg.metadata[dataset_name]["preproc_data_path"])
    
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
    feats_list = list(inputs.keys())
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
    
    dataset = {
        "features": features,
        "feature_names":feats_list,
        "map_valid_ids": map_valid_ids,
        "id_to_cr_id": id_to_cr_id,
        "cr_regions": cr_regions,
        "cr_areas":  cr_areas,
        "areas": areas,
        "valid_ids": valid_ids,
        "geo_metadata": geo_metadata,
        "target_norm" : target_norm,
        "cr_target_norm" : cr_target_norm,
        "cr_census_arr": cr_census_arr,
        "valid_census": valid_census,
        "num_coarse_regions": num_coarse_regions
    }
    
    return dataset


def compute_prediction_map_metrics(dataset, dataset_name, pred_map, inputs, output_dir):
    
    input_buildings = inputs["buildings"]
    map_valid_ids = dataset["map_valid_ids"]
    valid_ids = dataset["valid_ids"]
    valid_census = dataset["valid_census"]
    cr_regions = dataset["cr_regions"]
    cr_census_arr = dataset["cr_census_arr"]
    num_coarse_regions = dataset["num_coarse_regions"]
    geo_metadata = dataset["geo_metadata"]
    rst_wp_regions_path = cfg.metadata[dataset_name]["rst_wp_regions_path"]
    wp_rst_regions = gdal.Open(rst_wp_regions_path).ReadAsArray().astype(np.uint32)
    wp_ids = list(np.unique(wp_rst_regions))
    num_wp_ids = len(wp_ids)
    
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
    
    # Compute metrics
    metrics = compute_performance_metrics(agg_preds, valid_census)
    print("Metrics after disagg r2 {} mae {} mse {} mape {}".format(metrics["r2"], metrics["mae"], metrics["mse"], metrics["mape"]))


def train_model_with_agg_data_for_transfer_task(preproc_data_dir, output_dir, train_dataset_name, test_dataset_name, 
                              built_up_area_agg, eval_5splits, train_perc, train_level, random_seed, 
                              random_seed_folds, population_target, log_of_target):
    
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    test_but_not_train = list(set(test_dataset_name) - set(train_dataset_name) )
    all_dataset_names = train_dataset_name + test_but_not_train
    
    datasets = {}
    for ds in all_dataset_names:
        datasets[ds] = get_dataset(ds, preproc_data_dir, built_up_area_agg, population_target)
    
    final_features_arr = []
    final_density_arr = []
    for i,ds in enumerate(train_dataset_name): 
        dataset = datasets[ds]
        feats_list = dataset["feature_names"]
        features = dataset["features"]
        valid_ids = dataset["valid_ids"]
        id_to_cr_id = dataset["id_to_cr_id"]
        areas = dataset["areas"]
        cr_areas = dataset["cr_areas"]
        cr_census_arr = dataset["cr_census_arr"]
        cr_target_norm = dataset["cr_target_norm"]
        target_norm = dataset["target_norm"]
        valid_census = dataset["valid_census"]
        
        #id_offset = 0
        if train_level[i] == 'c':
            #id_offset = 1
            cr_features = compute_avg_feats(feats_list, features, valid_ids, id_to_cr_id, areas, cr_areas)
            features_arr = transform_dict_to_matrix(cr_features)
            density = compute_density(cr_target_norm, cr_census_arr, list(cr_areas.keys()))
        else:
            valid_features = {id:features[id] for id in valid_census.keys()}
            features_arr = transform_dict_to_matrix(valid_features)
            density = compute_density(target_norm, valid_census, list(valid_census.keys()))
        
        density_arr = transform_dict_to_array(density)
        
        final_features_arr.append(features_arr)
        final_density_arr.append(density_arr)
    
    final_features_arr = np.concatenate(final_features_arr, axis=0)
    final_density_arr = np.concatenate(final_density_arr, axis=0)
    num_samples = final_features_arr.shape[0]
    
    if eval_5splits:
        num_splits = 5
        np.random.seed(random_seed_folds)
        
        for validation_split in range(num_splits):
            
            orig_indices = np.arange(num_samples)
            np.random.shuffle(orig_indices)
            # Split dataset
            num_train_samples = int(num_samples * train_perc)
            train_idxs = orig_indices[:num_train_samples]
            val_idxs = orig_indices[num_train_samples:]
            
            features_train_arr = final_features_arr[train_idxs, :]
            density_train_arr = final_density_arr[train_idxs]
            features_val_arr = final_features_arr[val_idxs, :]
            density_val_arr = final_density_arr[val_idxs]
            
            # remove samples with density equal to 0 because when taking the log it does not work that well
            mask_valid_train_samples = density_train_arr > 0
            valid_features_train_arr = features_train_arr[mask_valid_train_samples, :]
            valid_density_train_arr = density_train_arr[mask_valid_train_samples]
            mask_valid_val_samples = density_val_arr > 0
            valid_features_val_arr = features_val_arr[mask_valid_val_samples, :]
            valid_density_val_arr = density_val_arr[mask_valid_val_samples]
            
            # obtain best RF paramenters
            best_n_estimators, best_max_depth = perform_rf_parameter_search(valid_features_train_arr, valid_density_train_arr, 
                                                                            valid_features_val_arr, valid_density_val_arr, log_of_target, random_seed)
            
            # train the model
            model = RandomForestRegressor(random_state=random_seed, n_jobs=4, n_estimators=best_n_estimators, max_depth=best_max_depth)
            final_valid_density_train_arr = valid_density_train_arr
            if log_of_target:
                final_valid_density_train_arr = np.log(valid_density_train_arr)
            model.fit(valid_features_train_arr, final_valid_density_train_arr)
            print("model split {} feature importance {}".format(validation_split, model.feature_importances_))
            
            # Perform prediction in each test dataset country
            for i,ds in enumerate(test_dataset_name):
                test_dataset = datasets[ds]
                input_paths = cfg.input_paths[ds]
                inputs = read_input_raster_data(input_paths)
                feats_list = test_dataset["feature_names"]
                all_pixel_features, height, width = get_all_pixel_features(inputs, feats_list)

                predictions = model.predict(all_pixel_features)
                pred_map = predictions.reshape((height, width))
                if log_of_target:
                    pred_map = np.exp(pred_map)
                pred_map = pred_map.astype(np.float32)
                
                compute_prediction_map_metrics(test_dataset, ds, pred_map, inputs, output_dir)
    else:
        
        # remove samples with density equal to 0 because when taking the log it does not work that well
        mask_valid_train_samples = final_density_arr > 0
        valid_features_train_arr = final_features_arr[mask_valid_train_samples, :]
        valid_density_train_arr = final_density_arr[mask_valid_train_samples]
        
        if log_of_target:
            valid_density_train_arr = np.log(valid_density_train_arr)
        # Fit model
        model = RandomForestRegressor(random_state=random_seed, n_jobs=4)
        model.fit(valid_features_train_arr, valid_density_train_arr)
        print("feature importance {}".format(model.feature_importances_))

        for i,ds in enumerate(test_dataset_name):
            test_dataset = datasets[ds]
            input_paths = cfg.input_paths[ds]
            inputs = read_input_raster_data(input_paths)
            input_buildings = inputs["buildings"]
            # Perform prediction per pixel
            pred_map = perform_prediction_at_pixel_level(inputs, feats_list, model)
            if log_of_target:
                pred_map = np.exp(pred_map)        
            pred_map = pred_map.astype(np.float32)

            if not population_target:
                preproc_input_buildings = np.multiply(input_buildings, np.multiply(input_buildings > 0, (input_buildings < 255)))
                pred_map = pred_map * preproc_input_buildings

            compute_prediction_map_metrics(test_dataset, ds, pred_map, inputs, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preproc_data_dir", "-pre_dir", type=str, default="", help="Preprocessed data directory containing pickle files")
    parser.add_argument("--output_dir", "-out", type=str, default="", help="Output dir ")
    parser.add_argument("--train_dataset_name", "-train", type=str, help="Train Dataset name (separated by commas)", required=True)
    parser.add_argument("--test_dataset_name", "-test", type=str, help="Test Dataset name (separated by commas)", required=True)
    parser.add_argument("--built_up_area_agg", "-bu", type=lambda x: bool(strtobool(x)), default=True, help="Flag that indicates if we should aggregate features using only the built up area")
    parser.add_argument("--eval_5splits", "-e5s", type=lambda x: bool(strtobool(x)), default=False, help="Perform 5 evaluations with different splits")
    parser.add_argument("--train_perc", "-tperc", type=float, default=0.8, help="Traininig percentage")
    parser.add_argument("--train_level", "-train_lvl", type=str, default="f", help="Train census level: c (coarse), f (finest)")
    parser.add_argument("--random_seed", "-rs", type=int, default=42, help="Random seed for the RF model")
    parser.add_argument("--random_seed_folds", "-rsf", type=int, default=1610, help="Random seed used to dataset splitting.")
    parser.add_argument("--population_target", "-pop_target", type=lambda x: bool(strtobool(x)), default=True, help="Use population as target")
    parser.add_argument("--log_of_target", "-log", type=lambda x: bool(strtobool(x)), default=True, help="Apply log to the target")
    
    args = parser.parse_args()

    # check arguments and fill with default values
    args.train_dataset_name = unroll_arglist(args.train_dataset_name)
    args.train_level = unroll_arglist(args.train_level, 'c', len(args.train_dataset_name))
    args.test_dataset_name = unroll_arglist(args.test_dataset_name)
    
    train_model_with_agg_data_for_transfer_task(args.preproc_data_dir,
                             args.output_dir, args.train_dataset_name, args.test_dataset_name, args.built_up_area_agg, 
                             args.eval_5splits, args.train_perc, args.train_level, 
                             args.random_seed, args.random_seed_folds, args.population_target, args.log_of_target)


if __name__ == "__main__":
    main()
