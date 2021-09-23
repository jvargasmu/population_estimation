import argparse
import pickle
import numpy as np
from osgeo import gdal
from utils import read_input_raster_data, compute_performance_metrics, write_geolocated_image, preprocess_census_targets
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region
import config_pop as cfg


def group_targets(census_data, wp_ids, valid_ids_mask, wp_to_ghd):
    # Initialize target values
    grouped_targets = {}
    for id in wp_ids:
        if valid_ids_mask[id] == 1:
            gid = wp_to_ghd[id]
            if gid not in grouped_targets.keys():
                grouped_targets[gid] = 0
    # Aggregate targets
    for id in wp_ids:
        if valid_ids_mask[id] == 1:
            gid = wp_to_ghd[id]
            grouped_targets[gid] += census_data[id]
    return grouped_targets


def get_valid_ids(wp_ids, matches_wp_to_hd, wp_no_data):
    valid_ids = []
    # Remove regions with no data value or with no matches in humdata.org
    ids_with_no_matches = [id for id in matches_wp_to_hd.keys() if matches_wp_to_hd[id] is None]
    for id in wp_ids:
        if (id not in wp_no_data) and (id not in ids_with_no_matches):
            valid_ids.append(id)
    return valid_ids


def disaggregate_weighted_by_preds(input_buildings, g_census_target_arr, pred_map, map_valid_ids,
                                   output_dir, g_regions, num_groups, suffix="", save_images=True, input_paths=None):
    # mask_input_buildings = np.multiply(input_buildings > 0, input_buildings < 255).astype(np.float32)
    # final_mask = np.multiply((map_valid_ids == 1).astype(np.float32), mask_input_buildings)
    # pred_map = np.multiply(pred_map, final_mask)

    # Compute total predictions per region
    pred_map_per_g_region = compute_accumulated_values_by_region(g_regions, pred_map, map_valid_ids, num_groups)

    # Compute normalized weights
    weights = compute_disagg_weights(g_regions, pred_map, pred_map_per_g_region, map_valid_ids)

    # Initialize output matrix
    disagg_population = set_value_for_each_region(g_regions, g_census_target_arr, map_valid_ids)
    disagg_population = np.multiply(disagg_population, weights)

    if save_images and input_paths is not None:
        buildings_path = input_paths["buildings"]
        source = gdal.Open(buildings_path)
        src_geo_transform = source.GetGeoTransform()
        src_projection = source.GetProjection()
        pred_map_path = "{}pred_map_{}.tif".format(output_dir, suffix)
        write_geolocated_image(pred_map.astype(np.float32), pred_map_path, src_geo_transform, src_projection)
        weights_path = "{}weights_map_{}.tif".format(output_dir, suffix)
        write_geolocated_image(weights.astype(np.float32), weights_path, src_geo_transform, src_projection)
        disagg_pop_path = "{}disagg_pop_map{}.tif".format(output_dir, suffix)
        write_geolocated_image(disagg_population.astype(np.float32), disagg_pop_path, src_geo_transform, src_projection)

    return disagg_population


def building_disagg_baseline(preproc_data_path, rst_wp_regions_path,
                             output_dir, target_col, dataset_name):
    # Read input data
    input_paths = cfg.input_paths[dataset_name]

    with open(preproc_data_path, 'rb') as handle:
        preproc_data = pickle.load(handle)

    matches_wp_to_hd = preproc_data["matches_wp_to_hd"]
    hd_parents = preproc_data["hd_parents"]
    wp_no_data = preproc_data["wp_no_data"]
    wp_census_target = preproc_data["wp_census_target"]
    wp_census_target = wp_census_target[target_col]
    wp_census_target = preprocess_census_targets(wp_census_target)
    wp_ids = list(matches_wp_to_hd.keys())
    num_wp_ids = len(wp_ids)
    num_groups = preproc_data["num_groups"]
    wp_rst_regions = gdal.Open(rst_wp_regions_path).ReadAsArray().astype(np.uint32)
    inputs = read_input_raster_data(input_paths)
    input_buildings = inputs["buildings"]

    # Get ids of no data
    list_no_data = list(wp_no_data)
    for id in matches_wp_to_hd.keys():
        if matches_wp_to_hd[id] is None:
            list_no_data.append(id)
    # Binary map representing a pixel belong to a region with valid id
    map_valid_ids = np.ones(wp_rst_regions.shape).astype(np.uint32)
    for nd in list_no_data:
        map_valid_ids[wp_rst_regions == nd] = 0
    # List of valid ids
    valid_ids = get_valid_ids(wp_ids, matches_wp_to_hd, wp_no_data)
    # Binary array indicating if a id (index of the array) is valid
    valid_ids_mask = np.zeros(num_wp_ids)
    for id in valid_ids:
        valid_ids_mask[id] = 1
    # Valid WorldPop census data
    valid_wp_census_target = {}
    for id in wp_census_target.keys():
        if id in valid_ids:
            valid_wp_census_target[id] = wp_census_target[id]
    # WorldPop id to parent region in humdata
    wp_to_ghd = np.zeros(num_wp_ids).astype(np.uint32)
    for id in valid_ids:
        hd_id = matches_wp_to_hd[id]
        gid = hd_parents[hd_id][cfg.col_coarse_level_seq_id]
        wp_to_ghd[id] = gid
    # Aggregate targets
    g_census_target = group_targets(wp_census_target, wp_ids, valid_ids_mask, wp_to_ghd)
    g_census_target_arr = np.zeros(num_groups).astype(np.float32)
    for gid in g_census_target.keys():
        g_census_target_arr[gid] = g_census_target[gid]
    # Get map of coarse level regions
    g_regions = compute_map_with_new_labels(wp_rst_regions, wp_to_ghd, map_valid_ids)

    # Get building maps with values between 0 and 1 (sometimes 255 represent no data values)
    unnorm_weights = np.multiply(input_buildings, (input_buildings < 255).astype(np.float32))

    # Disaggregate population using building maps as weights
    disagg_population = disaggregate_weighted_by_preds(input_buildings, g_census_target_arr, unnorm_weights,
                                                       map_valid_ids,
                                                       output_dir, g_regions, num_groups, input_paths=input_paths)

    # Aggregate pixel level predictions to the finest level region
    agg_preds_arr = compute_accumulated_values_by_region(wp_rst_regions, disagg_population, map_valid_ids, num_wp_ids)
    agg_preds = {id: agg_preds_arr[id] for id in valid_ids}

    preds_and_gt_dict = {}
    for id in valid_wp_census_target.keys():
        preds_and_gt_dict[id] = {"pred": agg_preds[id], "gt": valid_wp_census_target[id]}

    # Save predictions
    preds_and_gt_path = "{}preds_and_gt.pkl".format(output_dir)
    with open(preds_and_gt_path, 'wb') as handle:
        pickle.dump(preds_and_gt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Compute metrics
    r2, mae, mse = compute_performance_metrics(agg_preds, valid_wp_census_target)
    print("r2 {} mae {} mse {}".format(r2, mae, mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preproc_data_path", type=str, help="Preprocessed data of regions (pickle file)")
    parser.add_argument("rst_wp_regions_path", type=str,
                        help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("output_dir", type=str, help="Output dir ")
    parser.add_argument("target_col", type=str, help="Target column")
    parser.add_argument("dataset_name", type=str, help="Dataset name")
    args = parser.parse_args()

    building_disagg_baseline(args.preproc_data_path, args.rst_wp_regions_path,
                             args.output_dir, args.target_col, args.dataset_name)


if __name__ == "__main__":
    main()
