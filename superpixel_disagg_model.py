import os
os.environ["OMP_PROC_BIND"] = os.environ.get("OMP_PROC_BIND", "true")
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt 
from osgeo import gdal
import wandb
from pathlib import Path
import h5py 
from tqdm import tqdm as tqdm
from pathlib import Path
import random

import config_pop as cfg
from utils import read_input_raster_data, read_input_raster_data_to_np, compute_performance_metrics, write_geolocated_image, create_map_of_valid_ids, \
    compute_grouped_values, transform_dict_to_array, transform_dict_to_matrix, calculate_densities, plot_2dmatrix, save_as_geoTIFF, \
    bbox2
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
    print(rst_wp_regions_path)
    fine_regions = gdal.Open(rst_wp_regions_path).ReadAsArray().astype(np.uint32)
    wp_ids = list(np.unique(fine_regions)) 
    fine_area = dict(zip(wp_ids, areas))
    num_wp_ids = len(wp_ids)
    features = read_input_raster_data_to_np(input_paths)

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
    feature_names = list(input_paths.keys())
    # torch_feature_names = torch.tensor(list(input_paths.keys()))

    # Merging building features from google and maxar if both are available
    if ('buildings_google' in feature_names) and ('buildings_maxar' in feature_names):
        # Taking the max over both available features
        #  max operation for mean building areas
        gidx = np.where([el=='buildings_google' for el in feature_names])
        midx = np.where([el=='buildings_maxar' for el in feature_names])

        maxargs = np.argmax(np.concatenate([features[gidx,:,:,None], features[midx,:,:,None]], 4), 4).astype(bool).squeeze()
 
        features[gidx,maxargs] =  features[midx,maxargs]
        feature_names[np.squeeze(gidx)] = 'buildings_merge' 
        bkeepers = np.where([el!='buildings_maxar' for el in feature_names])
        features = features[bkeepers]
        feature_names.remove('buildings_maxar') 

        if ('buildings_google_mean_area' in feature_names) and ('buildings_maxar_mean_area' in feature_names): 
            gaidx = np.where([el=='buildings_google_mean_area' for el in feature_names])
            maidx = np.where([el=='buildings_maxar_mean_area' for el in feature_names])
            
            features[gaidx,maxargs] =  features[maidx, maxargs]
            feature_names[np.squeeze(gaidx)] = 'buildings_merge_mean_area'
            bmakeepers = np.where([el!='buildings_maxar_mean_area' for el in feature_names])
            features = features[bmakeepers]
            feature_names.remove('buildings_maxar_mean_area') 
            
    # Assert that first input is a building variable
    assert(feature_names[0] in building_features)

    num_feat, ih, iw = features.shape
    valid_data_mask = torch.ones( (ih, iw), dtype=torch.bool) 
    for i, name in enumerate(feature_names):
        
        if name in (building_features + related_building_features):
            features[i][features[i]<0] = 0
        else:
            this_mask = features[i]!=no_data_values[name]
            if no_data_values[name]>1e30:
                this_mask *= ~(np.isclose(features[i],no_data_values[name]))
            valid_data_mask *= this_mask

        # Normalize the features, execpt for the buildings layer when the scale Network is used
        if (params['Net'] in ['ScaleNet']) and (name not in building_features):
            if name in list(cfg.norms[dataset_name].keys()):
                # normalize by known mean and std
                features[i] = (features[i] - cfg.norms[dataset_name][name][0]) / cfg.norms[dataset_name][name][1]
            else:
                raise Exception("Did not find precalculated mean and std")
                
    # features = torch.cat(features, 0)
    features = torch.from_numpy(features)

    # this_mask = features[0]!=no_data_values[name]
    if params["Net"]=='ScaleNet':
        valid_data_mask *= features[0]>0

    guide_res = features.shape[1:3]

    # also account for the invalid map ids
    valid_data_mask *= map_valid_ids.astype(bool)

    # Create dataformat with densities for administrative boundaries of level -1 and -2
    # Fills in the densities per pixel
    # TODO: distribute sourcemap and target map according to the building pixels! To do so, we need to calculate the number of builtup pixels per regions!
    fine_density, fine_map = calculate_densities(census=fine_census, area=fine_area, map=fine_regions)
    cr_density, cr_map = calculate_densities(census=cr_census, area=cr_areas, map=cr_regions)

    # fine_map = fine_density_map
    # cr_map = cr_density_map
    replacement = 0

    # replace -inf with 1e-16 ("-16" on log scale) is close enough to zero for the log scale, otherwise take 0
    np.nan_to_num(fine_map, copy=False, neginf=replacement)
    np.nan_to_num(cr_map, copy=False, neginf=replacement)

    cr_map = torch.from_numpy(cr_map)
    fine_map = torch.from_numpy(fine_map).float()
    valid_data_mask =  valid_data_mask.to(torch.bool)
    fine_regions = torch.from_numpy(fine_regions.astype(np.int16))
    map_valid_ids = torch.from_numpy(map_valid_ids.astype(np.bool8))
    id_to_cr_id = torch.from_numpy(id_to_cr_id.astype(np.int32))
    cr_regions = torch.from_numpy(cr_regions.astype(np.int32)) 

    # replacements of invalid values
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
        # "mean_std": (fmean, fstd),
        "num_valid_pix": valid_data_mask.sum(),
        "fine": "fine",
        "coarse": "coarse",
    }
    
    return dataset


def prep_train_hdf5_file(training_source, h5_filename, var_filename, silent_mode=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate throuh the image an cut out examples
    tX,tY,tregid,tMasks,tBBox = [],[],[],[],[]

    tr_features, tr_census, tr_regions, tr_map, tr_guide_res, tr_valid_data_mask, level, feature_names = training_source
    
    tr_regions = tr_regions.to(device)
    tr_valid_data_mask = tr_valid_data_mask.to(device)
    
    for regid in tqdm(tr_census.keys(), disable=silent_mode):
        mask = (regid==tr_regions) * tr_valid_data_mask
        boundingbox = bbox2(mask)
        rmin, rmax, cmin, cmax = boundingbox
        tX.append(tr_features[:,rmin:rmax, cmin:cmax].numpy())
        tY.append(np.asarray(tr_census[regid]))
        tregid.append(np.asarray(regid))
        tMasks.append(mask[rmin:rmax, cmin:cmax].cpu().numpy())
        boundingbox = [rmin.cpu(), rmax.cpu(), cmin.cpu(), cmax.cpu()]
        tBBox.append(boundingbox)
        
    tr_regions = tr_regions.cpu()
    tr_valid_data_mask = tr_valid_data_mask.cpu().numpy()

    dim, h, w = tr_features.shape

    if not os.path.isfile(h5_filename):
        with h5py.File(h5_filename, "w") as f:
            h5_features = f.create_dataset("features", (1, dim, h, w), dtype=np.float32, fillvalue=0, chunks=(1,dim,512,512))
            for i,feat in tqdm(enumerate(tr_features)):
                h5_features[:,i] = feat
    
    # TODO: Add feature_names here!!! and unpack it in the datasetloader
    with open(var_filename, 'wb') as handle:
        pickle.dump([tr_census, tr_regions, tr_valid_data_mask, tY, tregid, tMasks, tBBox, feature_names], handle, protocol=pickle.HIGHEST_PROTOCOL)


def prep_test_hdf5_file(validation_data, this_disaggregation_data, h5_filename,  var_filename, disag_filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_features, val_census, val_regions, val_map, val_valid_ids, val_map_valid_ids, val_guide_res, val_valid_data_mask, geo_metadata, cr_map = validation_data

    dim, h, w = val_features.shape

    if not os.path.isfile(h5_filename):
        with h5py.File(h5_filename, "w") as f:
            h5_features = f.create_dataset("features", (1, dim, h, w), dtype=np.float32, fillvalue=0, chunks=(1,dim,512,512))
            for i,feat in tqdm(enumerate(val_features)):
                h5_features[:,i] = feat
            
    with open(var_filename, 'wb') as handle:
        pickle.dump(
            [val_census, val_regions, val_map, val_valid_ids,\
            val_map_valid_ids, val_guide_res, val_valid_data_mask,
            geo_metadata, cr_map], 
            handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(disag_filename, 'wb') as handle:
        pickle.dump( this_disaggregation_data,  handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    test_dataset_name,
    optimizer,
    learning_rate,
    weights_regularizer,
    weights_regularizer_adamw, 
    memory_mode,
    log_step,
    random_seed,
    validation_split,
    validation_fold,
    weights,
    sampler,
    custom_sampler_weights,
    dropout,
    loss,
    load_state,
    eval_only,
    input_scaling,
    output_scaling,
    silent_mode,
    dataset_dir,
    max_step
    ):

    ####  define parameters  ########################################################

    params = {
            'weights_regularizer': weights_regularizer,#0.001, # spatial color head
            'weights_regularizer_adamw': weights_regularizer_adamw,
            'kernel_size': [1,1,1,1],
            'loss': loss,

            "admin_augment": True,
            "load_state": load_state, #, UGA:'fluent-star-258', TZA: 'vague-voice-185' ,'dainty-flower-151',#None, 'brisk-armadillo-86'
            "eval_only": eval_only,
            "Net": 'ScaleNet', # Choose between ScaleNet and PixNet

            'optim': optimizer,
            'lr': learning_rate,
            "epochs": 100,
            'logstep': log_step,
            'maxstep': max_step,
            'train_dataset_name': train_dataset_name,
            'train_level': train_level,
            'test_dataset_name': test_dataset_name,
            'input_variables': list(cfg.input_paths[train_dataset_name[0]].keys()),
            'memory_mode': memory_mode,
            'random_seed': random_seed,
            'validation_split': validation_split,
            'validation_fold': validation_fold,
            'weights': weights,
            'sampler': sampler,
            'custom_sampler_weights': custom_sampler_weights,
            'dropout': dropout,
            'input_scaling': input_scaling,
            'output_scaling': output_scaling,
            'silent_mode': silent_mode,
            'dataset_dir': dataset_dir
            }

    building_features = ['buildings', 'buildings_j', 'buildings_google', 'buildings_maxar', 'buildings_merge']
    related_building_features = ['buildings_google_mean_area', 'buildings_maxar_mean_area', 'buildings_merge_mean_area']

    fine_train_source_vars = ["features", "fine_census", "fine_regions", "fine_map", "guide_res", "valid_data_mask", "fine", "feature_names"]
    cr_train_source_vars = ["features", "cr_census", "cr_regions", "cr_map", "guide_res", "valid_data_mask", "coarse", "feature_names"]
    fine_val_data_vars = ["features", "fine_census", "fine_regions", "fine_map", "valid_ids", "map_valid_ids", "guide_res",
                            "valid_data_mask", "geo_metadata", "cr_map"]
    cr_disaggregation_data_vars = ["id_to_cr_id", "cr_census", "cr_regions"]

    wandb.init(project="HAC", entity="nandometzger", config=params)

    # Fix all random seeds
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    ####  load dataset  #############################################################

    assert(all(elem=="c" or elem=="f" for elem in train_level))

    datalocations = {} 
    test_but_not_train = list(set(test_dataset_name) - set(train_dataset_name) )
    all_dataset_names = train_dataset_name + test_but_not_train
    train_level = pad_list(train_level, fill='f', target_len=len(all_dataset_names))    
    params["memory_mode"] = pad_list(params["memory_mode"], fill='d', target_len=len(all_dataset_names))    
    params["weights"] = pad_list(params["weights"], fill=1., target_len=len(all_dataset_names))    
    params["custom_sampler_weights"] = pad_list(params["custom_sampler_weights"], fill=1., target_len=len(all_dataset_names))    

    for i,ds in enumerate(all_dataset_names):
        this_level = train_level[i]

        h5_filename = f"{dataset_dir}/{ds}/data.hdf5"
        train_var_filename_c = f"{dataset_dir}/{ds}/additional_train_vars_c.pkl"
        train_var_filename_f = f"{dataset_dir}/{ds}/additional_train_vars_f.pkl"
        eval_var_filename = f"{dataset_dir}/{ds}/additional_test_vars.pkl"
        eval_disag_filename = f"{dataset_dir}/{ds}/disag_vars.pkl"
        parent_dir = f"{dataset_dir}/{ds}/"
        print("h5_filename", h5_filename)

        if not (os.path.isfile(h5_filename) and os.path.isfile(train_var_filename_f) and os.path.isfile(train_var_filename_c) \
            and os.path.isfile(eval_var_filename) and os.path.isfile(eval_disag_filename)):
            Path(parent_dir).mkdir(parents=True, exist_ok=True)

            this_dataset = get_dataset(ds, params, building_features, related_building_features) 
            prep_train_hdf5_file(build_variable_list(this_dataset, fine_train_source_vars), h5_filename, train_var_filename_f, silent_mode=silent_mode)
            prep_train_hdf5_file(build_variable_list(this_dataset, cr_train_source_vars), h5_filename, train_var_filename_c, silent_mode=silent_mode)
            
            # Build testdataset here to avoid dublicate executions later
            this_validation_data = build_variable_list(this_dataset, fine_val_data_vars)
            this_disaggregation_data = build_variable_list(this_dataset, cr_disaggregation_data_vars) 
            prep_test_hdf5_file(this_validation_data, this_disaggregation_data, h5_filename,  eval_var_filename, eval_disag_filename)
            
            # Free up RAM
            del this_disaggregation_data, this_validation_data
            del this_dataset 

        datalocations[ds] = {"features": h5_filename, "train_vars_f": train_var_filename_f, "train_vars_c": train_var_filename_c,
            "eval_vars": eval_var_filename, "disag": eval_disag_filename}

    res, log_dict = PixAdminTransform(
        datalocations=datalocations,
        train_dataset_name=train_dataset_name,
        test_dataset_names=test_dataset_name,
        params=params, 
    )

    # save as geoTIFF files
    save_files = True
    if save_files:
        for name in test_dataset_name:
            with open(datalocations[name]['eval_vars'], "rb") as f:
                _, _, fine_map, _, _, _, valid_data_mask, geo_metadata, cr_map = pickle.load(f) 

            predicted_target_img = res[name+'/predicted_target_img']
            predicted_target_img_adjusted = res[name+'/predicted_target_img_adjusted']
            scales = res[name+'/scales']

            if name+'/variances' in list(res.keys()):
                variances = res[name+'/variances']
                variances[~valid_data_mask]= np.nan

            scale_vars_available = False
            if scales.shape.__len__()==3:
                scale_vars = scales[1]
                scale_vars[~valid_data_mask]= np.nan
                scales = scales[0]
                scale_vars_available = True
                

            cr_map[~valid_data_mask]= np.nan
            predicted_target_img[~valid_data_mask]= np.nan
            predicted_target_img_adjusted[~valid_data_mask]= np.nan
            scales[~valid_data_mask]= np.nan
            fine_map[~valid_data_mask]= np.nan

            #Prepate the output folder
            dest_folder = '../../../viz/outputs/{}'.format(wandb.run.name)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

            write_geolocated_image( cr_map.numpy(), dest_folder+'/{}_cr_map.tiff'.format(name),
                geo_metadata["geo_transform"], geo_metadata["projection"] )
            write_geolocated_image( predicted_target_img.numpy(), dest_folder+'/{}_predicted_target_img.tiff'.format(name),
                geo_metadata["geo_transform"], geo_metadata["projection"] )
            write_geolocated_image( predicted_target_img_adjusted.numpy(), dest_folder+'/{}_predicted_target_img_adjusted.tiff'.format(name),
                geo_metadata["geo_transform"], geo_metadata["projection"] )
            write_geolocated_image( fine_map.numpy(), dest_folder+'/{}_fine_map.tiff'.format(name),
                geo_metadata["geo_transform"], geo_metadata["projection"] )
            write_geolocated_image( scales.numpy(), dest_folder+'/{}_scales.tiff'.format(name),
                geo_metadata["geo_transform"], geo_metadata["projection"] )

            if name+'/variances' in list(res.keys()):
                write_geolocated_image( variances.numpy(), dest_folder+'/{}_variances.tiff'.format(name),
                    geo_metadata["geo_transform"], geo_metadata["projection"] )
            if scale_vars_available:
                write_geolocated_image( scale_vars.numpy(), dest_folder+'/{}_scale_variances.tiff'.format(name),
                    geo_metadata["geo_transform"], geo_metadata["projection"] )

    return


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
    # parser.add_argument("rst_wp_regions_path", type=str,
                        # help="Raster of WorldPop administrative boundaries information") 
    parser.add_argument("--train_dataset_name", "-train", type=str, help="Train Dataset name (separated by commas)", required=True)
    parser.add_argument("--train_level", "-train_lvl", type=str,  default='c', help="ordered by --train_dataset_name [f:finest, c: coarser level] (separated by commas) ")
    parser.add_argument("--test_dataset_name", "-test", type=str, help="Test Dataset name (separated by commas)", required=True)

    parser.add_argument("--sampler", "-sap", type=str, default=None, help="Options: natural (not recommended yet), custom (see --custom_sampler_weights), <blank> (no sampler)")
    parser.add_argument("--custom_sampler_weights", "-csw", type=str,  default='1', help="ordered by --train_dataset_name weight for the sampler (separated by commas) ")

    parser.add_argument("--optimizer", "-optim", type=str, default="adam", help="adam, adamw ")
    parser.add_argument("--loss", "-l", type=str, default="NormL1", help="NormL1, NormL2, gaussNLL, laplaceNLL")
    parser.add_argument("--train_weight", "-train_w", type=str,  default='1', help="ordered by --train_dataset_name weighting of the samples in the datasets (separated by commas) ")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.00001, help=" ")
    parser.add_argument("--weights_regularizer", "-wr", type=float, default=0., help=" ")
    parser.add_argument("--weights_regularizer_adamw", "-adamwr", type=float, default=0.001, help=" ")
    parser.add_argument("--dropout", "-drop", type=float, default=0.0, help="propout probability ")

    parser.add_argument("--memory_mode", "-mm", type=str, default='m', help="Loads the variables into memory to speed up the training process. Obviously: Needs more memory! m:load into memory; d: load from a hdf5 file on disk. (separated by commas)")
    parser.add_argument("--log_step", "-lstep", type=float, default=2000, help="Evealuate the model after 'logstep' batchiterations.")
    parser.add_argument("--max_step", "-mstep", type=float, default=np.inf, help="Evealuate the model after 'logstep' batchiterations.")

    parser.add_argument("--validation_split", "-vs", type=float, default=0.2, help="Evealuate the model after 'logstep' batchiterations.")
    parser.add_argument("--validation_fold", "-fold", type=int, default=None, help="Validation fold. One of [0,1,2,3,4]. When used --validation_split is ignored.")
    parser.add_argument("--random_seed", "-rs", type=int, default=1610, help="Random seed for this run.")
    
    parser.add_argument("--load_state", "-load", type=str, default=None, help="Loading from a specific state. Attention: 5fold evaluation not implmented yet!")
    parser.add_argument("--eval_only", "-eval", type=bool, default=False, help="Just evaluate the model and save results. Attention: 5fold evaluation not implmented yet! ")

    parser.add_argument("--input_scaling", "-is", type=bool, default=False, help="Countrywise input feature scaling.")
    parser.add_argument("--output_scaling", "-os", type=bool, default=False, help="Countrywise output scaling.")

    parser.add_argument("--silent_mode", "-silent", type=bool, default=False, help="Surpresses tqdm output mostly")
    parser.add_argument("--dataset_dir", "-dd", type=str, default='datasets', help="Directory of the hdf5 files")

    args = parser.parse_args()


    # check arguments and fill with default values
    args.train_dataset_name = unroll_arglist(args.train_dataset_name)
    args.train_level = unroll_arglist(args.train_level, 'c', len(args.train_dataset_name))
    args.test_dataset_name = unroll_arglist(args.test_dataset_name)
    args.memory_mode = unroll_arglist(args.memory_mode, 'm', len(args.train_dataset_name))
    
    args.train_weight = unroll_arglist(args.train_weight, '1', len(args.train_dataset_name))
    args.train_weight = [ float(el) for el in args.train_weight ]
    args.train_weight =  [ el/sum(args.train_weight) for el in args.train_weight ]

    args.custom_sampler_weights = unroll_arglist(args.custom_sampler_weights, '1', len(args.train_dataset_name))
    args.custom_sampler_weights = [ float(el) for el in args.custom_sampler_weights ]
    args.custom_sampler_weights =  [ el/sum(args.custom_sampler_weights) for el in args.custom_sampler_weights ]


    import gc
    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, h5py.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass # Was already closed

    superpixel_with_pix_data( 
        args.train_dataset_name,
        args.train_level,
        args.test_dataset_name,
        args.optimizer,
        args.learning_rate,
        args.weights_regularizer,
        args.weights_regularizer_adamw,
        args.memory_mode,
        args.log_step,
        args.random_seed,
        args.validation_split,
        args.validation_fold,
        args.train_weight,
        args.sampler,
        args.custom_sampler_weights,
        args.dropout,
        args.loss,
        args.load_state,
        args.eval_only,
        args.input_scaling,
        args.output_scaling,
        args.silent_mode,
        args.dataset_dir,
        args.max_step
    )


if __name__ == "__main__":
    main()
