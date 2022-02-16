import argparse
import pickle
import numpy as np
from osgeo import gdal
from utils import read_input_raster_data, compute_performance_metrics, write_geolocated_image, create_map_of_valid_ids
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region
import config_pop as cfg


def disaggregate_weighted_by_preds(cr_census_arr, pred_map, map_valid_ids,
                                   cr_regions, num_cr_regions, output_dir,
                                   mask=None, suffix="", save_images=True, geo_metadata=None):
    # Obtained masked predictions
    pred_map_masked = pred_map
    if mask is not None:
        final_mask = np.multiply((map_valid_ids == 1).astype(np.float32), mask.astype(np.float32))
        pred_map_masked = np.multiply(pred_map, final_mask)

    # Compute total predictions per region
    pred_map_per_cr_region = compute_accumulated_values_by_region(cr_regions, pred_map_masked, map_valid_ids,
                                                                  num_cr_regions)

    # Compute normalized weights
    weights = compute_disagg_weights(cr_regions, pred_map_masked, pred_map_per_cr_region, map_valid_ids)

    # Initialize output matrix
    disagg_population = set_value_for_each_region(cr_regions, cr_census_arr, map_valid_ids)
    disagg_population = np.multiply(disagg_population, weights)

    if save_images and geo_metadata is not None:
        src_geo_transform = geo_metadata["geo_transform"]
        src_projection = geo_metadata["projection"]
        pred_map_path = "{}pred_map_{}.tif".format(output_dir, suffix)
        # write_geolocated_image(pred_map_masked.astype(np.float32), pred_map_path, src_geo_transform, src_projection)
        weights_path = "{}weights_map_{}.tif".format(output_dir, suffix)
        # write_geolocated_image(weights.astype(np.float32), weights_path, src_geo_transform, src_projection)
        disagg_pop_path = "{}disagg_pop_map{}.tif".format(output_dir, suffix)
        # write_geolocated_image(disagg_population.astype(np.float32), disagg_pop_path, src_geo_transform, src_projection)

    return disagg_population


def building_disagg_baseline(preproc_data_path, rst_wp_regions_path,
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
    wp_rst_regions = gdal.Open(rst_wp_regions_path).ReadAsArray().astype(np.uint32)
    wp_ids = list(np.unique(wp_rst_regions))
    num_wp_ids = len(wp_ids)
    print("num_wp_ids {}".format(num_wp_ids))
    inputs = read_input_raster_data(input_paths)
    input_buildings = inputs["buildings"]

    # Binary map representing a pixel belong to a region with valid id
    map_valid_ids = create_map_of_valid_ids(wp_rst_regions, no_valid_ids)

    # Get map of coarse level regions
    cr_regions = compute_map_with_new_labels(wp_rst_regions, id_to_cr_id, map_valid_ids)

    # Get building maps with values between 0 and 1 (sometimes 255 represent no data values)
    unnorm_weights = np.multiply(input_buildings, (input_buildings < 255).astype(np.float32))
    mask = unnorm_weights > 0

    # Disaggregate population using building maps as weights
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

    building_disagg_baseline(args.preproc_data_path, args.rst_wp_regions_path,
                             args.output_dir, args.dataset_name)


if __name__ == "__main__":
    main()
