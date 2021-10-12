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


def compute_density(areas, census):
    return {id: census[id] / areas[id] for id in areas.keys()}


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


def perform_prediction_at_pixel_level(inputs, feats_list, model):
    inputs_mat = []
    for feat in feats_list:
        inputs_mat.append(inputs[feat])
    inputs_mat = np.array(inputs_mat)
    height = inputs_mat.shape[1]
    width = inputs_mat.shape[2]

    all_features = inputs_mat.reshape((inputs_mat.shape[0], -1))
    all_features = all_features.transpose()
    print("all_features shape {}".format(all_features.shape))

    predictions = model.predict(all_features)

    return predictions.reshape((height, width))


def train_model_with_agg_data(preproc_data_path, rst_wp_regions_path,
                             output_dir, dataset_name):
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
    features = pdata["features"]
    cr_features = compute_avg_feats(feats_list, features, valid_ids, id_to_cr_id, areas, cr_areas)
    cr_features_arr = transform_dict_to_matrix(cr_features)

    # Computet WorldPop target : log of density
    cr_density = compute_density(cr_areas, cr_census_arr)
    cr_density_arr = transform_dict_to_array(cr_density)
    cr_log_density_arr = np.log(cr_density_arr)

    # Create model
    model = RandomForestRegressor(random_state=42, n_jobs=4, n_estimators=1)

    # Fit model
    model.fit(cr_features_arr, cr_log_density_arr)
    print("feature importance {}".format(model.feature_importances_))

    # Perform prediction per pixel
    pred_map = perform_prediction_at_pixel_level(inputs, feats_list, model)
    pred_map = np.exp(pred_map)
    pred_map = pred_map.astype(np.float32)

    # Get building maps with values between 0 and 1 (sometimes 255 represent no data values)
    unnorm_weights = pred_map
    mask = np.multiply(input_buildings > 0, (input_buildings < 255))

    # Disaggregate population using pred maps as weights
    disagg_population = disaggregate_weighted_by_preds(cr_census_arr, unnorm_weights,
                                                       map_valid_ids, cr_regions, num_coarse_regions, output_dir,
                                                       mask=mask, save_images=True, geo_metadata=geo_metadata)

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
    r2, mae, mse = compute_performance_metrics(agg_preds, valid_census)
    print("r2 {} mae {} mse {}".format(r2, mae, mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preproc_data_path", type=str, help="Preprocessed data of regions (pickle file)")
    parser.add_argument("rst_wp_regions_path", type=str,
                        help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("output_dir", type=str, help="Output dir ")
    parser.add_argument("dataset_name", type=str, help="Dataset name")
    args = parser.parse_args()

    train_model_with_agg_data(args.preproc_data_path, args.rst_wp_regions_path,
                             args.output_dir, args.dataset_name)


if __name__ == "__main__":
    main()
