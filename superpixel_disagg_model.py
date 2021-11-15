import os
os.environ["OMP_PROC_BIND"] = os.environ.get("OMP_PROC_BIND", "true")
import argparse
import pickle
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from osgeo import gdal
import wandb

import config_pop as cfg
from utils import read_input_raster_data, compute_performance_metrics, write_geolocated_image, create_map_of_valid_ids, \
    compute_grouped_values, transform_dict_to_array, transform_dict_to_matrix, calculate_densities, plot_2dmatrix, save_as_geoTIFF
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region

from pix_transform.pix_admin_transform import PixAdminTransform
from pix_transform.pix_transform import PixTransform
from pix_transform_utils.utils import downsample,align_images
# from prox_tv import tvgen
from pix_transform_utils.plots import plot_result


def get_dataset(dataset_name, params, building_features, related_building_features):

    # configure paths
    rst_wp_regions_path = cfg.metadata[dataset_name]["rst_wp_regions_path"]
    preproc_data_path = cfg.metadata[dataset_name]["preproc_data_path"]

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

    # Merging building features from google and maxar if both are available
    if ('buildings_google' in feature_names) and ('buildings_maxar' in feature_names):
        # Calculate drift/slope to approximately match maxar building to the google buildings
        # Only considering pixels that contain buildings in both patches

        # scale_maxar_to_google = cfg.metadata[dataset_name]['scale_maxar_to_google']
        # if scale_maxar_to_google is None:
        #     valid_build_mask = np.logical_and(inputs['buildings_google']>0, inputs['buildings_maxar']>0)

        #     scale_maxar_to_google = LinearRegression().fit(inputs['buildings_maxar'][valid_build_mask].reshape(-1, 1),
        #                                                 inputs['buildings_google'][valid_build_mask].reshape(-1, 1)).coef_
        # else:
        scale_maxar_to_google = 1.
        
        inputs['buildings_maxar'] *= scale_maxar_to_google

        # Taking the max over both available features
        #  max operation for mean building areas
        maxargs = np.argmax( np.concatenate([inputs['buildings_google'][:,:,None], inputs['buildings_maxar'][:,:,None]],2),2 ).astype(bool)
        inputs['buildings_merge'] = inputs['buildings_google'].copy()
        inputs['buildings_merge'][maxargs] =  inputs['buildings_maxar'][maxargs]
        del inputs['buildings_google'], inputs['buildings_maxar']

        if ('buildings_google' in feature_names) and ('buildings_maxar' in feature_names): 
            inputs['buildings_merge_mean_area'] = inputs['buildings_google_mean_area'].copy()
            inputs['buildings_merge_mean_area'][maxargs] =  inputs['buildings_maxar_mean_area'][maxargs]
            del inputs['buildings_google_mean_area'], inputs['buildings_maxar_mean_area']

        # rearrage the feature names
        feature_names = list(inputs.keys())
        buildingidx = np.where(np.array(list(inputs.keys()))=='buildings_merge')[0][0]
        feature_names[0], feature_names[buildingidx] = feature_names[buildingidx], feature_names[0]

    # Assert that first input is a building variable
    assert(feature_names[0] in building_features)

    ih, iw = inputs[feature_names[0]].shape
    num_feat = len(inputs.keys())
    features = torch.zeros( (len(inputs.keys()), ih, iw), dtype=torch.float32)
    valid_data_mask = torch.ones( (ih, iw), dtype=torch.bool)
    fmean = torch.zeros((len(inputs.keys())))
    fstd = torch.zeros((len(inputs.keys())))
    for i,name in enumerate(feature_names):
        
        # convert dict into an matrix of features
        features[i] = torch.from_numpy(inputs[name]) 
        if name in (building_features + related_building_features):
            features[i][features[i]<0] = 0 
        else:
            this_mask = features[i]!=no_data_values[name]
            if no_data_values[name]>1e30:
                this_mask *= ~torch.from_numpy(np.isclose(features[i],no_data_values[name]))
            valid_data_mask *= this_mask

        # Normalize the features, execpt for the buildings layer when the scale Network is used
        if (params['Net'] in ['ScaleNet']) and (name not in building_features):
            if name in cfg.norms[dataset_name].keys():
                # normalize by known mean and std
                features[i] = (features[i] - cfg.norms[dataset_name][name][0]) / cfg.norms[dataset_name][name][1]
            else:
                raise Exception("Did not find precalculated mean and std")
                pass

                # calculate mean std your self...
                # fmean[i] = features[i][this_mask].mean()
                # fstd[i] = features[i][this_mask].std()
                # features[i] = (features[i] - fmean) / fstd

    del inputs


    # this_mask = features[0]!=no_data_values[name]
    if params["Net"]=='ScaleNet':
        valid_data_mask *= features[0]>0

    guide_res = features.shape[1:3]

    # also account for the invalid map ids
    valid_data_mask *= map_valid_ids.astype(bool)

    fstd = features[:,valid_data_mask].std(1)
    fmean = features[:,valid_data_mask].mean(1)

    # Create dataformat with densities for administrative boundaries of level -1 and -2
    # Fills in the densities per pixel
    #TODO: distribute sourcemap and target map according to the building pixels! To do so, we need to calculate the number of builtup pixels per regions!
    fine_density, fine_density_map = calculate_densities(census=fine_census, area=fine_area, map=fine_regions)
    cr_density, cr_density_map = calculate_densities(census=cr_census, area=cr_areas, map=cr_regions)

    fine_map = fine_density_map
    cr_map = cr_density_map
    replacement = 0

    # replace -inf with 1e-16 ("-16" on log scale) is close enough to zero for the log scale, otherwise take 0
    np.nan_to_num(fine_map, copy=False, neginf=replacement)
    np.nan_to_num(cr_map, copy=False, neginf=replacement)

    cr_map = torch.from_numpy(cr_map)
    fine_map = torch.from_numpy(fine_map).float()
    valid_data_mask = valid_data_mask.to(torch.bool)
    fine_regions = torch.from_numpy(fine_regions.astype(np.int16))
    map_valid_ids = torch.from_numpy(map_valid_ids.astype(np.bool8))
    id_to_cr_id = torch.from_numpy(id_to_cr_id.astype(np.int32))
    cr_regions = torch.from_numpy(cr_regions.astype(np.int32)) 

    features[:,~valid_data_mask] = replacement
    fine_map[~valid_data_mask] = replacement
    cr_map[~valid_data_mask] = replacement
    cr_map[~valid_data_mask] = 1e-10 # TODO: verify this operation!

    dataset = {
        "features": features,
        "feature_names":feature_names,
        "cr_map": cr_map,
        "fine_map": fine_map,
        "valid_data_mask": valid_data_mask,
        "fine_regions": fine_regions,
        "map_valid_ids": map_valid_ids,
        "id_to_cr_id": id_to_cr_id,
        "cr_regions": cr_regions,
        "cr_census": cr_census,
        "fine_census": fine_census,
        "valid_ids": valid_ids,
        "guide_res": guide_res,
        "geo_metadata": geo_metadata,
        "mean_std": (fmean, fstd),
        "num_valid_pix": valid_data_mask.sum()
    }
    
    return dataset


def build_variable_list(dataset: dict, var_list: list) -> list:
    """
    Selects the variables specified in var_list from the datset and returns them as a list of same order as var_list
    """
    outlist = []
    for var in var_list:
        outlist.append(dataset[var])
    return outlist


def superpixel_with_pix_data(
    train_dataset_name,
    train_level,
    test_dataset_name
    ):

    ####  define parameters  ########################################################

    params = {
            'weights_regularizer': 0.,#0.001, # spatial color head
            'weights_regularizer_adamw': 0.001,
            'kernel_size': [1,1,1,1],
            'loss': 'NormL1',

            "admin_augment": True,
            "load_state": None, #, UGA:'fluent-star-258', TZA: 'vague-voice-185' ,'dainty-flower-151',#None, 'brisk-armadillo-86'
            "eval_only": False,
            "Net": 'ScaleNet', # Choose between ScaleNet and PixNet

            'PCA': None,

            'optim': 'adamw',
            'lr': 0.00001,
            "epochs": 100,
            'logstep': 1,
            'train_dataset_name': train_dataset_name,
            'train_level': train_level,
            'test_dataset_name': test_dataset_name,
            'input_variables': list(cfg.input_paths[train_dataset_name[0]].keys())
            }

    building_features = ['buildings', 'buildings_j', 'buildings_google', 'buildings_maxar', 'buildings_merge']
    related_building_features = ['buildings_google_mean_area', 'buildings_maxar_mean_area', 'buildings_merge_mean_area']

    fine_train_source_vars = ["features", "fine_census", "fine_regions", "fine_map", "guide_res", "valid_data_mask"]
    cr_train_source_vars = ["features", "cr_census", "cr_regions", "cr_map", "guide_res", "valid_data_mask"]
    fine_val_data_vars = ["features", "fine_census", "fine_regions", "fine_map", "valid_ids", "map_valid_ids", "guide_res", "valid_data_mask"]
    cr_disaggregation_data_vars = ["id_to_cr_id", "cr_census", "cr_regions"]

    wandb.init(project="HAC", entity="nandometzger", config=params)

    ####  load dataset  #############################################################
    # 

    assert(all(elem=="c" or elem=="f" for elem in train_level))

    train_dataset = {}
    test_dataset = {}
    train_level_dict = {}
    test_level_dict = {}
    for i,ds in enumerate(train_dataset_name):
        train_level_dict[ds] = train_level[i]
        train_dataset[ds] = get_dataset(ds, params, building_features, related_building_features) 

    calculate_norm = True
    if calculate_norm:
        feats = torch.zeros((train_dataset[ds]["features"].shape[0],0))
        for i,ds in enumerate(train_dataset_name):
            feats = torch.cat([ feats, train_dataset[ds]["features"][:,train_dataset[ds]["valid_data_mask"]] ],1)
        print("means", feats.mean(1))
        print("stds", feats.std(1))


    for ds in test_dataset_name:
        if ds in list(train_dataset.keys()):
            test_dataset[ds] = train_dataset[ds]
        else:
            test_dataset[ds] = get_dataset(ds, params, building_features, related_building_features)


    ##################################################

    # build training
    training_source ={}
    for i,ds in enumerate(train_dataset_name):
        train_variables = fine_train_source_vars if train_level[i]=="f" else cr_train_source_vars

        training_source[ds] = build_variable_list(train_dataset[ds], train_variables)

    # build validation
    validation_data ={}
    disaggregation_data={}
    for ds in test_dataset_name:
        validation_data[ds] = build_variable_list(test_dataset[ds], fine_val_data_vars)
        disaggregation_data[ds] = build_variable_list(test_dataset[ds], cr_disaggregation_data_vars)

    res = PixAdminTransform(
        training_source=training_source,
        validation_data=validation_data,
        params=params,
        disaggregation_data=disaggregation_data,
    )

    f, ax = plot_result(
        cr_map.numpy(), predicted_target_img.numpy(),
        predicted_target_img_adj.numpy(), fine_map.numpy() )
    plt.show()

    # save as geoTIFF files
    save_files = True
    if save_files:
        cr_map[~valid_data_mask]= np.nan
        predicted_target_img[~valid_data_mask]= np.nan
        predicted_target_img_adj[~valid_data_mask]= np.nan
        fine_map[~valid_data_mask]= np.nan
        dest_folder = '../../../viz/outputs/{}'.format(wandb.run.name)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        write_geolocated_image( cr_map.numpy(), dest_folder+'/source_map.tiff'.format(wandb.run.name),
            geo_metadata["geo_transform"], geo_metadata["projection"] )
        write_geolocated_image( predicted_target_img.numpy(), dest_folder+'/predicted_target_img.tiff'.format(wandb.run.name),
            geo_metadata["geo_transform"], geo_metadata["projection"] )
        write_geolocated_image( predicted_target_img_adj.numpy(), dest_folder+'/predicted_target_img_adj.tiff'.format(wandb.run.name),
            geo_metadata["geo_transform"], geo_metadata["projection"] )
        write_geolocated_image( fine_map.numpy(), dest_folder+'/validation_map.tiff'.format(wandb.run.name),
            geo_metadata["geo_transform"], geo_metadata["projection"] )
        write_geolocated_image( scales.numpy(), dest_folder+'/scales.tiff'.format(wandb.run.name),
            geo_metadata["geo_transform"], geo_metadata["projection"] )

    return


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("preproc_data_path", type=str, help="Preprocessed data of regions (pickle file)")
    # parser.add_argument("rst_wp_regions_path", type=str,
                        # help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("output_dir", type=str, help="Output dir ")
    parser.add_argument("--train_dataset_name", "-train", nargs='+', help="Train Dataset name (separated by commas)", required=True)
    parser.add_argument("--train_level", "-train_lvl", nargs='+', help="ordered by --train_dataset_name [f:finest, c: coarser level] (separated by commas) ", required=True)
    parser.add_argument("--test_dataset_name", "-test", nargs='+', help="Test Dataset name (separated by commas)", required=True)
    parser.add_argument("--optimizer", "-optim", type=str, default="adamw", help=" ")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.00001, help=" ")
    parser.add_argument("--weights_regularizer", "-wr", type=float, default=0., help=" ")
    parser.add_argument("--weights_regularizer_adamw", "-adamwr", type=float, default=0.001, help=" ")
    # parser.add_argument("--test_level", "-test_lvl", nargs='+', help="ordered by --train_dataset_name [f:finest, c: coarser level] (separated by commas) ", required=True)
    args = parser.parse_args()

    args.train_dataset_name = args.train_dataset_name[0].split(",")
    args.train_level = args.train_level[0].split(",")
    args.test_dataset_name = args.test_dataset_name[0].split(",")

    superpixel_with_pix_data(
        args.train_dataset_name,
        args.train_level,
        args.test_dataset_name,
        args.optimizer,
        args.learning_rate,
        args.weights_regularizer,
        args.weights_regularizer_adamw, 
    )

if __name__ == "__main__":
    main()
