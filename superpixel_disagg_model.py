import os
os.environ["OMP_PROC_BIND"] = os.environ.get("OMP_PROC_BIND", "true")
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt 
from osgeo import gdal
import wandb

import config_pop as cfg
from utils import read_input_raster_data, compute_performance_metrics, write_geolocated_image, create_map_of_valid_ids, \
    compute_grouped_values, transform_dict_to_array, transform_dict_to_matrix, calculate_densities, plot_2dmatrix
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region

from pix_transform.pix_admin_transform import PixAdminTransform
from pix_transform.pix_transform import PixTransform
from pix_transform_utils.utils import downsample,align_images
# from prox_tv import tvgen
from pix_transform_utils.plots import plot_result


def superpixel_with_pix_data(preproc_data_path, rst_wp_regions_path,
                             output_dir, dataset_name):

    ####  define parameters  ########################################################

    params = {'feature_downsampling': 1,
            'spatial_features_input': False,
            'weights_regularizer': 0.001, # spatial color head
            'loss': 'LogL1',
            "predict_log_values": False,

            "admin_augment": True,
            "load_state": None,#None, 'brisk-armadillo-86'
            "Net": 'ScaleNet', # Choose between ScaleNet and PixNet


            'optim': 'adam',
            'lr': 0.0001,
            "epochs": 100,
            'logstep': 1,
            'dataset_name': dataset_name,
            'input_variables': list(cfg.input_paths[dataset_name].keys())
            }

    wandb.init(project="HAC", entity="nandometzger", config=params)

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
    # input_buildings = inputs["buildings"]

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
    features = torch.zeros( (len(inputs.keys()), ih, iw), dtype=torch.float32)
    valid_data_mask = torch.ones( (ih, iw), dtype=torch.bool)
    for i,name in enumerate(feature_names):
        features[i] = torch.from_numpy(inputs[name])
        # features[i] = inputs[name]
        if name=='buildings':
            features[i][features[i]==no_data_values[name] ] = 0

        this_mask = features[i]!=no_data_values[name]
        valid_data_mask *= this_mask

        # Normalize the features, execpt for the buildings layer when the scale Network is used
        if (params['Net'] in ['ScaleNet']) and (name not in ['buildings', 'buildings_j']):
            if dataset_name in cfg.norms.keys():
                # normalize by known mean and std
                features[i] = (features[i] - cfg.norms[dataset_name][name][0]) / cfg.norms[dataset_name][name][1]
            else:
                # calculate mean std your self...
                fmean = features[i][this_mask].mean()
                fstd = features[i][this_mask].std()
                features[i] = (features[i] - fmean) / fstd

    del inputs
    guide_res = features.shape[1:3]

    # also account for the invalid map ids
    valid_data_mask *= map_valid_ids.astype(bool)

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
        # fine_census = {key: np.log(value) for key,value in fine_census.items()}
        cr_census = {key: np.log(value) for key,value in cr_census.items()}
        replacement = -16
    else:
        validation_map = fine_density_map_resamp
        source_map = cr_density_map_resamp
        replacement = 0

    # replace -inf with 1e-16 ("-16" on log scale) is close enough to zero for the log scale, otherwise take 0
    np.nan_to_num(validation_map, copy=False, neginf=replacement)
    np.nan_to_num(source_map, copy=False, neginf=replacement)

    source_map = torch.from_numpy(source_map)
    validation_map = torch.from_numpy(validation_map).float()
    valid_data_mask_resamp = valid_data_mask_resamp.to(torch.bool)
    fine_regions = torch.from_numpy(fine_regions.astype(np.int16))
    map_valid_ids = torch.from_numpy(map_valid_ids.astype(np.bool8))
    # valid_ids = torch.tensor(valid_ids, dtype=torch.bool)

    features_resamp[:,~valid_data_mask_resamp] = replacement
    validation_map[~valid_data_mask_resamp] = replacement
    source_map[~valid_data_mask_resamp] = replacement

    # Guide are the high resolution features, read them here and sort them into the matrix
    # Source is the administrative population density map.

    predicted_target_img = PixAdminTransform(
        guide_img=features_resamp,
        source=(cr_census, cr_regions, source_map),
        valid_mask=valid_data_mask_resamp,
        params=params,
        validation_data=(fine_census, validation_map, fine_regions, valid_ids, map_valid_ids),
        orig_guide_res=guide_res
    )

    
    f, ax = plot_result(source_map.numpy(), predicted_target_img.numpy(), validation_map.numpy())
    plt.show()


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
