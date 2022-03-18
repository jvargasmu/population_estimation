import argparse
import pickle
import numpy as np
from osgeo import gdal
from utils import read_input_raster_data, compute_performance_metrics, create_map_of_valid_ids
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region, bool_arr_to_seq_of_indices, cy_fast_ICM_with_pop_target
import config_pop as cfg
from building_disagg_baseline import disaggregate_weighted_by_preds


def train_mrf_model(input_buildings, map_valid_ids, cr_regions,
                    cr_census_arr, num_coarse_regions, graph_ind_path, graph_dist_path,
                    perc_change, max_iter, lambda_val, output_dir):

    # Compute initial population predictions (using building disaggregation)
    ini_pred_map = np.multiply(input_buildings, (input_buildings < 255).astype(np.float32))
    mask = np.multiply(input_buildings > 0, (input_buildings < 255))
    ini_pop_pred = disaggregate_weighted_by_preds(cr_census_arr, ini_pred_map,
                                                       map_valid_ids, cr_regions, num_coarse_regions, output_dir,
                                                       mask=mask, save_images=False, return_global_scale=False)
    ini_target = ini_pop_pred.flatten()
    # Get valid targert
    valid_mask = map_valid_ids.flatten().astype(np.bool)
    valid_mask = np.multiply(valid_mask, mask.flatten())  # For efficiency
    valid_target = ini_target[valid_mask]
    # Load neighbours metadata
    neigh_dist = np.load(graph_dist_path)
    neigh_ind = np.load(graph_ind_path)
    print("Load saved neigh dist, ind numpy arrays")
    # Get data at the coarse level of census
    neigh_dist = neigh_dist.astype(np.float32)
    neigh_ind = neigh_ind.astype(np.uint32)
    valid_target = valid_target.astype(np.float32)
    cr_regions_flat = cr_regions.flatten()
    valid_g_regions = cr_regions_flat[valid_mask].astype(np.uint32)
    cr_census_arr = cr_census_arr.astype(np.float32)
    # Perform MRF regularization
    seq_all = np.arange(valid_mask.shape[0]).astype(np.uint32)
    valid_ind = seq_all[valid_mask]
    pix_ind_to_valid_ind = np.zeros(valid_mask.shape[0]).astype(np.int32) - 1
    num_valid = np.sum(valid_mask)
    pix_ind_to_valid_ind[valid_mask] = np.arange(num_valid)
    valid_output = cy_fast_ICM_with_pop_target(valid_target, neigh_ind, valid_g_regions, cr_census_arr,
                                               num_coarse_regions, perc_change, max_iter, lambda_val)

    output = np.zeros(map_valid_ids.shape[0] * map_valid_ids.shape[1]).astype(np.float32)
    output[valid_mask] = valid_output
    print("output.shape {}".format(output.shape))
    output_map = output.reshape((map_valid_ids.shape[0], map_valid_ids.shape[1]))
    return output_map


def train_mrf(preproc_data_path, rst_wp_regions_path, output_dir, dataset_name,
                    perc_change, max_iter, lambda_val, graph_ind_path, graph_dist_path):
    # Read input data
    input_paths = cfg.input_paths[dataset_name]

    with open(preproc_data_path, 'rb') as handle:
        pdata = pickle.load(handle)

    cr_census_arr = pdata["cr_census_arr"]
    no_valid_ids = pdata["no_valid_ids"]
    id_to_cr_id = pdata["id_to_cr_id"]
    num_coarse_regions = pdata["num_coarse_regions"]
    wp_rst_regions = gdal.Open(rst_wp_regions_path).ReadAsArray().astype(np.uint32)
    inputs = read_input_raster_data(input_paths)
    input_buildings = inputs["buildings"]
    geo_metadata = pdata["geo_metadata"]
    valid_ids = pdata["valid_ids"]
    valid_census = pdata["valid_census"]
    wp_ids = list(np.unique(wp_rst_regions))
    num_wp_ids = len(wp_ids)

    # Binary map representing a pixel belong to a region with valid id
    map_valid_ids = create_map_of_valid_ids(wp_rst_regions, no_valid_ids)

    # Get map of coarse level regions
    cr_regions = compute_map_with_new_labels(wp_rst_regions, id_to_cr_id, map_valid_ids)

    # Train MRF model
    pred_map = train_mrf_model(input_buildings, map_valid_ids, cr_regions,
                    cr_census_arr, num_coarse_regions, graph_ind_path, graph_dist_path,
                    perc_change, max_iter, lambda_val, output_dir)

    mask = pred_map > 0
    
    # Compute accuracy before disaggregation
    final_mask = np.multiply((map_valid_ids == 1).astype(np.float32), mask.astype(np.float32))
    pred_map_masked = np.multiply(pred_map, final_mask)
    orig_agg_preds_arr = compute_accumulated_values_by_region(wp_rst_regions, pred_map_masked, map_valid_ids, num_wp_ids)
    orig_agg_preds = {id: orig_agg_preds_arr[id] for id in valid_ids}
    orig_metrics = compute_performance_metrics(orig_agg_preds, valid_census)
    print("Metrics before disagg r2 {} mae {} mse {} mape {}".format(orig_metrics["r2"], orig_metrics["mae"], orig_metrics["mse"], orig_metrics["mape"]))
    
    # Disaggregate population using building maps as weights
    disagg_population = disaggregate_weighted_by_preds(cr_census_arr, pred_map,
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
    #r2, mae, mse = compute_performance_metrics(agg_preds, valid_census)
    #print("r2 {} mae {} mse {}".format(r2, mae, mse))
    metrics = compute_performance_metrics(agg_preds, valid_census)
    print("Metrics after disagg r2 {} mae {} mse {} mape {}".format(metrics["r2"], metrics["mae"], metrics["mse"], metrics["mape"]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preproc_data_path", type=str, help="Preprocessed data of regions (pickle file)")
    parser.add_argument("rst_wp_regions_path", type=str,
                        help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("output_dir", type=str, help="Output dir ")
    parser.add_argument("dataset_name", type=str, help="Dataset name")
    parser.add_argument("perc_change", type=float, help="Percentage of change")
    parser.add_argument("max_iter", type=int, help="Maximum number of iterations")
    parser.add_argument("lambda_val", type=float, help="Lambda coeficient")
    parser.add_argument("graph_ind_path", type=str, help="Graph neighbour index path")
    parser.add_argument("graph_dist_path", type=str, help="Graph neighbour distance path")
    args = parser.parse_args()

    train_mrf(args.preproc_data_path, args.rst_wp_regions_path, args.output_dir, args.dataset_name,
              args.perc_change, args.max_iter, args.lambda_val, args.graph_ind_path, args.graph_dist_path)


if __name__ == "__main__":
    main()
