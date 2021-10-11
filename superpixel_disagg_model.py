import os
os.environ["OMP_PROC_BIND"] = os.environ.get("OMP_PROC_BIND", "true")
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from osgeo import gdal

import config_pop as cfg
from utils import read_input_raster_data, compute_performance_metrics, write_geolocated_image, create_map_of_valid_ids, \
    compute_grouped_values, transform_dict_to_array, transform_dict_to_matrix, calculate_densities, plot_2dmatrix
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region

from pix_transform.pix_transform import PixTransform
from pix_transform_baselines.baselines import bicubic
from pix_transform_utils.utils import downsample,align_images
# from prox_tv import tvgen
from pix_transform_utils.plots import plot_result




def superpixel_with_pix_data(preproc_data_path, rst_wp_regions_path,
                             output_dir, dataset_name):

    ####  define parameters  ########################################################
    params = {#'img_idxs' : [], # idx images to process, if empty then all of them
                
            #'scaling': 8,
            'feature_downsampling': 1,
            'greyscale': False, # Turn image into grey-scale
            'channels': -1,
            
            'spatial_features_input': False,
            'weights_regularizer': [0., 0., 0.], # spatial color head
            # 'weights_regularizer': [0.0001, 0.001, 0.0001], # spatial color head
            'loss': 'l1',
            "predict_log_values": False,
    
            'optim': 'adam',
            'lr': 0.001,
                    
            'batch_size': 32,
            'patch_size': 8,
            #'iteration': 32768*20,
            "epochs": 3,

            'logstep': 1,
            
            'final_TGV' : False, # Total Generalized Variation in post-processing
            'align': False, # Move image around for evaluation in case guide image and target image are not perfectly aligned
            'delta_PBP': 1, # Delta for percentage of bad pixels 
            }

    ####  load dataset  #############################################################
    # Read input data
    input_paths = cfg.input_paths[dataset_name]
    no_data_values = cfg.no_data_values[dataset_name]

    with open(preproc_data_path, 'rb') as handle:
        pdata = pickle.load(handle)

    cr_census_arr = pdata["cr_census_arr"]
    valid_ids = pdata["valid_ids"]
    no_valid_ids = pdata["no_valid_ids"]
    id_to_cr_id = pdata["id_to_cr_id"]
    fine_census = pdata["valid_census"]
    num_coarse_regions = pdata["num_coarse_regions"]
    geo_metadata = pdata["geo_metadata"]
    areas = pdata["areas"] 
    fine_regions = gdal.Open(rst_wp_regions_path).ReadAsArray().astype(np.uint32)
    wp_ids = list(np.unique(fine_regions)) 
    fine_area = dict(zip(wp_ids, areas))
    num_wp_ids = len(wp_ids)
    inputs = read_input_raster_data(input_paths)
    input_buildings = inputs["buildings"]

    # Binary map representing a pixel belong to a region with valid id
    map_valid_ids = create_map_of_valid_ids(fine_regions, no_valid_ids)

    # Get map of coarse level regions
    cr_regions = compute_map_with_new_labels(fine_regions, id_to_cr_id, map_valid_ids)

    # Compute area of coarse regions
    cr_areas = compute_grouped_values(areas, valid_ids, id_to_cr_id)

    cr_census = {}
    for key in cr_areas.keys():
        cr_census[key] = cr_census_arr[key]

    # Reorganize features into one numpy array and handling of no-data mask
    feature_names = list(inputs.keys()) 
    ih, iw = inputs[feature_names[0]].shape
    num_feat = len(inputs.keys())
    features = np.zeros( (len(inputs.keys()), ih, iw))
    valid_data_mask = np.ones( (ih, iw), dtype=np.bool8)
    for i,name in enumerate(feature_names):
        features[i] = inputs[name]
        valid_data_mask *= features[i]!=no_data_values[name] 
    valid_data_mask *= map_valid_ids.astype(bool)
    guide_res = features.shape[1:3]

    # Create dataformat with densities for administrative boundaries of level -1 and -2
    # Fills in the densities per pixel
    fine_density, fine_density_map = calculate_densities(census=fine_census, area=fine_area, map=fine_regions)
    cr_density, cr_density_map = calculate_densities(census=cr_census, area=cr_areas, map=cr_regions)

    #downsample the features to target resolution
    if params['feature_downsampling']!=1:
        valid_data_mask_resamp = downsample(valid_data_mask, params['feature_downsampling'])==1
        features_resamp = downsample(features, params['feature_downsampling'])
        fine_density_map_resamp = downsample(fine_density_map, params['feature_downsampling'])
        cr_density_map_resamp = downsample(cr_density_map, params['feature_downsampling'])
    else:
        valid_data_mask_resamp = valid_data_mask
        features_resamp = features
        fine_density_map_resamp = fine_density_map
        cr_density_map_resamp = cr_density_map

    # target the log of population densities

    if params["predict_log_values"]:
        validation_map = np.log(fine_density_map_resamp)
        source_map = np.log(cr_density_map_resamp)
    else:
        validation_map = fine_density_map_resamp
        source_map = cr_density_map_resamp
    
    # replace -inf with 1e-16 ("-16" on log scale) is close enough to zero.
    replacement = -16
    np.nan_to_num(validation_map, copy=False, neginf=replacement)
    np.nan_to_num(source_map, copy=False, neginf=replacement)
    features_resamp[:,~valid_data_mask_resamp] = replacement
    validation_map[~valid_data_mask_resamp] = replacement
    source_map[~valid_data_mask_resamp] = replacement

    # Guide are the high resolution features, read them here and sort them into the matrix
    # Source is the administrative population density map.

    predicted_target_img = PixTransform(
        guide_img=features_resamp,
        source_img=source_map,
        valid_mask=valid_data_mask_resamp,
        params=params,
        validation_data=(fine_census, validation_map, fine_regions, valid_ids, map_valid_ids),
        orig_guide_res=guide_res
    )

    #TODO: Backsample? the predicted target image to original resolution of the features for comparisons, fill eges
    #TODO: visualizations

    if params['final_TGV'] :
        print("applying TGV...")
        predicted_target_img = tvgen(predicted_target_img,[0.1, 0.1],[1, 2],[1, 1])
        
    if params['align'] :
        print("aligning...")
        target_img,predicted_target_img = align_images(target_img,predicted_target_img)

    # TODO: visualizations
    f, ax = plot_result(guide_img,source_img,predicted_target_img,bicubic_target_img,target_img)
    plt.show()

    if target_img is not None:
        # compute metrics and plot results
        MSE = np.mean((predicted_target_img - target_img) ** 2)
        MAE = np.mean(np.abs(predicted_target_img - target_img))
        PBP = np.mean(np.abs(predicted_target_img - target_img) > params["delta_PBP"])

        print("MSE: {:.3f}  ---  MAE: {:.3f}  ---  PBP: {:.3f}".format(MSE,MAE,PBP))
        print("\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preproc_data_path", type=str, help="Preprocessed data of regions (pickle file)")
    parser.add_argument("rst_wp_regions_path", type=str,
                        help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("output_dir", type=str, help="Output dir ")
    parser.add_argument("dataset_name", type=str, help="Dataset name")
    args = parser.parse_args()

    superpixel_with_pix_data(args.preproc_data_path, args.rst_wp_regions_path,
                             args.output_dir, args.dataset_name)


if __name__ == "__main__":
    main()
