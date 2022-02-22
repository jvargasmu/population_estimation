import argparse
import pickle
import numpy as np
from osgeo import gdal
from utils import read_input_raster_data, read_input_raster_data_to_np, compute_performance_metrics, write_geolocated_image, create_map_of_valid_ids
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region
import config_pop as cfg


def disaggregate_weighted_by_preds(cr_census_arr, pred_map, map_valid_ids,
                                   cr_regions, num_cr_regions, output_dir,
                                   mask=None, suffix="", save_images=True, geo_metadata=None, return_global_scale=True):
    # Obtained masked predictions
    pred_map_masked = pred_map
    if mask is not None:
        final_mask = np.multiply((map_valid_ids == 1).astype(np.float32), mask.astype(np.float32))
        pred_map_masked = np.multiply(pred_map, final_mask)

    # Compute total predictions per region
    pred_map_per_cr_region = compute_accumulated_values_by_region(cr_regions.astype(np.uint32), pred_map_masked.astype(np.float32), map_valid_ids.astype(np.uint32),
                                                                  num_cr_regions)

    # Compute normalized weights
    weights = compute_disagg_weights(cr_regions.astype(np.uint32), pred_map_masked.astype(np.float32),
        pred_map_per_cr_region,  map_valid_ids.astype(np.uint32))

    # Initialize output matrix
    disagg_population = set_value_for_each_region(cr_regions.astype(np.uint32), cr_census_arr.astype(np.float32), map_valid_ids.astype(np.uint32))
    disagg_population = np.multiply(disagg_population, weights)

    if return_global_scale:
        return disagg_population, cr_census_arr[1]/pred_map_per_cr_region[1]

    if save_images and geo_metadata is not None:
        src_geo_transform = geo_metadata["geo_transform"]
        src_projection = geo_metadata["projection"]
        pred_map_path = "{}pred_map_{}.tif".format(output_dir, suffix)
        write_geolocated_image(pred_map_masked.astype(np.float32), pred_map_path, src_geo_transform, src_projection)
        weights_path = "{}weights_map_{}.tif".format(output_dir, suffix)
        write_geolocated_image(weights.astype(np.float32), weights_path, src_geo_transform, src_projection)
        disagg_pop_path = "{}disagg_pop_map{}.tif".format(output_dir, suffix)

        write_geolocated_image(disagg_population.astype(np.float32), disagg_pop_path, src_geo_transform, src_projection)
    return disagg_population


def building_disagg_baseline(output_dir, dataset_name, test_dataset_name, global_disag):
    # Read input data

    all_unnorm_weights = np.zeros((0,1))
    all_map_valid_ids =  np.zeros((0,1))
    all_cr_regions =  np.zeros((0,1))
    all_mask =  np.zeros((0,1))
    all_cr_census_arr = 0

    for name in dataset_name:
            
        input_paths = cfg.input_paths[name]
        rst_wp_regions_path = cfg.metadata[name]["rst_wp_regions_path"]
        preproc_data_path = cfg.metadata[name]["preproc_data_path"]


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
        inputs = read_input_raster_data_to_np(input_paths)
        # input_buildings = inputs["buildings_google"]

        feature_names = list(input_paths.keys())
        
        # Merging building inputs from google and maxar if both are available
        merge_with_maxar = False
        if ('buildings_google' in feature_names) and ('buildings_maxar' in feature_names) and merge_with_maxar:
            # Taking the max over both available inputs
            #  max operation for mean building areas
            gidx = np.where([el=='buildings_google' for el in feature_names])
            midx = np.where([el=='buildings_maxar' for el in feature_names])

            maxargs = np.argmax(np.concatenate([inputs[gidx,:,:,None], inputs[midx,:,:,None]], 4), 4).astype(bool).squeeze()
    
            inputs[gidx,maxargs] =  inputs[midx,maxargs]
            feature_names[np.squeeze(gidx)] = 'buildings_merge' 
            bkeepers = np.where([el!='buildings_maxar' for el in feature_names])
            inputs = inputs[bkeepers]
            feature_names.remove('buildings_maxar') 

            if ('buildings_google_mean_area' in feature_names) and ('buildings_maxar_mean_area' in feature_names): 
                gaidx = np.where([el=='buildings_google_mean_area' for el in feature_names])
                maidx = np.where([el=='buildings_maxar_mean_area' for el in feature_names])
                
                inputs[gaidx,maxargs] =  inputs[maidx, maxargs]
                feature_names[np.squeeze(gaidx)] = 'buildings_merge_mean_area'
                bmakeepers = np.where([el!='buildings_maxar_mean_area' for el in feature_names])
                inputs = inputs[bmakeepers]
                feature_names.remove('buildings_maxar_mean_area') 
            input_buildings = inputs[np.where([el=='buildings_merge' for el in feature_names])]
        else:
            input_buildings = inputs[np.where([el=='buildings_google' for el in feature_names])]
            
        # Binary map representing a pixel belong to a region with valid id
        map_valid_ids = create_map_of_valid_ids(wp_rst_regions, no_valid_ids)

        # Get map of coarse level regions
        cr_regions = compute_map_with_new_labels(wp_rst_regions, id_to_cr_id, map_valid_ids)

        # Get building maps with values between 0 and 1 (sometimes 255 represent no data values)
        unnorm_weights = np.multiply(input_buildings, (input_buildings < 255).astype(np.float32))
        mask = unnorm_weights > 0

        # Append to lists
        all_unnorm_weights = np.concatenate([all_unnorm_weights, unnorm_weights.reshape(-1,1)],0)
        all_map_valid_ids = np.concatenate([all_map_valid_ids, map_valid_ids.reshape(-1,1)],0)
        all_cr_regions = np.concatenate([all_cr_regions, cr_regions.reshape(-1,1)],0)
        all_mask = np.concatenate([all_mask, mask.reshape(-1,1)],0) 
        all_cr_census_arr += cr_census_arr.sum()

    # Disaggregate population using building maps as weights
    # global_disag = global_disag
    if global_disag:
        cr_census_arr = np.concatenate([[0], [cr_census_arr.sum()]])
        all_cr_regions[all_cr_regions>=1] = 1
        num_coarse_regions = 2
    
    if test_dataset_name is not None:

        _, scale = disaggregate_weighted_by_preds(np.concatenate([[0], [all_cr_census_arr]]), all_unnorm_weights,
                                                       all_map_valid_ids, all_cr_regions, num_coarse_regions, output_dir,
                                                       mask=all_mask, save_images=True, geo_metadata=geo_metadata, return_global_scale=True)
                                                        
        # evaluate on testdataset
        name = test_dataset_name
            
        input_paths = cfg.input_paths[name]
        rst_wp_regions_path = cfg.metadata[name]["rst_wp_regions_path"]
        preproc_data_path = cfg.metadata[name]["preproc_data_path"]

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
        inputs = read_input_raster_data_to_np(input_paths)
        # input_buildings = inputs["buildings_google"]

        feature_names = list(input_paths.keys())
        
        # Merging building inputs from google and maxar if both are available 
        if ('buildings_google' in feature_names) and ('buildings_maxar' in feature_names) and merge_with_maxar:
            # Taking the max over both available inputs
            #  max operation for mean building areas
            gidx = np.where([el=='buildings_google' for el in feature_names])
            midx = np.where([el=='buildings_maxar' for el in feature_names])

            maxargs = np.argmax(np.concatenate([inputs[gidx,:,:,None], inputs[midx,:,:,None]], 4), 4).astype(bool).squeeze()
    
            inputs[gidx,maxargs] =  inputs[midx,maxargs]
            feature_names[np.squeeze(gidx)] = 'buildings_merge' 
            bkeepers = np.where([el!='buildings_maxar' for el in feature_names])
            inputs = inputs[bkeepers]
            feature_names.remove('buildings_maxar') 

            if ('buildings_google_mean_area' in feature_names) and ('buildings_maxar_mean_area' in feature_names): 
                gaidx = np.where([el=='buildings_google_mean_area' for el in feature_names])
                maidx = np.where([el=='buildings_maxar_mean_area' for el in feature_names])
                
                inputs[gaidx,maxargs] =  inputs[maidx, maxargs]
                feature_names[np.squeeze(gaidx)] = 'buildings_merge_mean_area'
                bmakeepers = np.where([el!='buildings_maxar_mean_area' for el in feature_names])
                inputs = inputs[bmakeepers]
                feature_names.remove('buildings_maxar_mean_area') 
            # input_buildings = inputs["buildings_merge"]
            input_buildings = inputs[np.where([el=='buildings_merge' for el in feature_names])]
        else:
            input_buildings = inputs[np.where([el=='buildings_google' for el in feature_names])]

        # Binary map representing a pixel belong to a region with valid id
        map_valid_ids = create_map_of_valid_ids(wp_rst_regions, no_valid_ids)

        # Get map of coarse level regions
        cr_regions = compute_map_with_new_labels(wp_rst_regions, id_to_cr_id, map_valid_ids)

        # Get building maps with values between 0 and 1 (sometimes 255 represent no data values)
        unnorm_weights = np.multiply(input_buildings, (input_buildings < 255).astype(np.float32))
        mask = unnorm_weights > 0
        unnorm_weights[~mask] = 0

        disagg_population = unnorm_weights[0]*scale
        
    else:
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
    metrics = compute_performance_metrics(agg_preds, valid_census)
    # r2, mae, mse = compute_performance_metrics(agg_preds, valid_census)
    print("r2 {} mae {} mse {} mape {}".format(metrics["r2"], metrics["mae"], metrics["mse"], metrics["mape"]))

def pad_list(arg_list, fill, target_len):
    if fill is not None:
        arg_list.extend([fill]*(target_len- len(arg_list)))
    return arg_list

def unroll_arglist(arg_list, fill=None, target_len=None):
    arg_list = arg_list.split(",")
    return pad_list(arg_list, fill, target_len)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("preproc_data_path", type=str, help="Preprocessed data of regions (pickle file)")
    # parser.add_argument("rst_wp_regions_path", type=str, help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("--output_dir", type=str, help="Output dir ")
    parser.add_argument("--train_dataset_name", type=str, help="Dataset name")
    parser.add_argument("--test_dataset_name", type=str, help="Dataset name")
    parser.add_argument("--global_disag", action="store_true", help="country wide disag")
    args = parser.parse_args()

    args.train_dataset_name = unroll_arglist(args.train_dataset_name)

    building_disagg_baseline(
        # args.preproc_data_path,
        #args.rst_wp_regions_path,
        args.output_dir,
        args.train_dataset_name,
        args.test_dataset_name,
        args.global_disag
    )



if __name__ == "__main__":
    main()
