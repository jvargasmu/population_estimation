import argparse
import numpy as np
from osgeo import gdal
import config_pop as cfg
import csv
import pickle
from utils import read_shape_layer_data, read_input_raster_data, read_input_raster_data_Sat2Pop, preprocess_census_targets, compute_grouped_values, write_geolocated_image
from preprocessing_pop_data import compute_agg_features_from_raster
from tqdm import tqdm
import torch
import pandas as pd


def read_multiple_targets_from_csv(csv_path):
    targets = {}
    col_to_index = {}
    index_to_col = {}
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for i, row in enumerate(reader):
            if i > 0:
                id = row[1]
                for j in range(len(index_to_col.keys())):
                    targets[index_to_col[j]][id] = row[j]
            else:
                # header
                for j, val in enumerate(row):
                    col_to_index[val] = j
                    index_to_col[j] = val
                    targets[val] = {}
    return targets


def geo_match_to_coarse(geo_match):
    return "_".join(geo_match.split("_")[:-1])


def remap_wp_ids(wp_rst_regions):
    # wp_rst_regions = torch.from_numpy(wp_rst_regions.astype(np.int32))
    wp_rst_regions[wp_rst_regions==8888] = 0
    wp_unique_list = np.unique(wp_rst_regions)
    offset = np.partition(wp_unique_list, 1)[1] - 1
    wp_rst_regions[wp_rst_regions>0] -= offset
    wp_unique_list2 = np.unique(wp_rst_regions)

    palette = np.zeros(wp_unique_list2.max()+1, dtype=np.uint32)
    mapping = {i:value for i,value in enumerate(wp_unique_list2)}
    for newkey,oldkey in mapping.items():
        palette[oldkey] = newkey
        assert(oldkey>=newkey)
    remapped_wp_rst_regions = palette[wp_rst_regions]

    finalmapping = {oldkey:newkey for oldkey,newkey in zip(wp_unique_list,np.unique(remapped_wp_rst_regions))} 

    return finalmapping, remapped_wp_rst_regions


def change_keys_in_dict(data_dict_orig, mapping):
    data_dict = {}
    for k in data_dict_orig.keys():
        data_dict[mapping[k]] = data_dict_orig[k]
    return data_dict


def preprocessing_wp_census_data(wp_regions_path, rst_wp_regions_path, census_data_path, 
                                    target_col, output_path, dataset_name, mode):
    
    # Read input data
    if mode=="Geodata":
        input_paths = cfg.input_paths[dataset_name]
        metadata = cfg.metadata[dataset_name]
        inputs = read_input_raster_data(input_paths)
    else:
        input_paths = cfg.input_paths_sat2pop[dataset_name]
        metadata = cfg.metadata[dataset_name]
        inputs = read_input_raster_data_Sat2Pop(input_paths)

    buildings = inputs["buildings"]
    buildings_mask = buildings > 0
    hd_regions = read_shape_layer_data(wp_regions_path)
    all_census = read_multiple_targets_from_csv(census_data_path)
    print("census_data size {}".format(len(all_census.keys())))
    # hd_rst_regions = gdal.Open(rst_hd_regions_path).ReadAsArray().astype(np.uint32)
    wp_rst_regions = gdal.Open(rst_wp_regions_path).ReadAsArray().astype(np.uint32)

    # Get geo spatial references
    if "buildings_maxar" in input_paths.keys():
        buildings_path = input_paths["buildings_maxar"]
    if "buildings_google" in input_paths.keys():
        buildings_path = input_paths["buildings_google"]
    if "buildings" in input_paths.keys():
        buildings_path = input_paths["buildings"]
    if "BuildingPreds_Own" in input_paths.keys():
        buildings_path = input_paths["BuildingPreds_Own"]

    source = gdal.Open(buildings_path)
    geo_transform = source.GetGeoTransform()
    projection = source.GetProjection()
    geo_metadata = {"geo_transform": geo_transform, "projection": projection}

    # remap wp ids
    finalmapping, wp_rst_regions_sid = remap_wp_ids(wp_rst_regions.copy())

    # save sid subnational for later
    write_geolocated_image( wp_rst_regions_sid, metadata["rst_wp_regions_path"],
                geo_metadata["geo_transform"], geo_metadata["projection"] )
    # Accumulate features
    features, areas, masked_features, built_up_areas = compute_agg_features_from_raster(wp_rst_regions_sid, inputs, no_data_vals=metadata["wp_no_data"], buildings_mask=buildings_mask)
    
    # Get census target
    valid_census = all_census[target_col]
    valid_census = preprocess_census_targets(valid_census)
    valid_census = change_keys_in_dict(valid_census, finalmapping)
    census = valid_census.copy()
    census[0] = 0.0

    df = pd.DataFrame(hd_regions, columns=hd_regions[0].keys())
    cr_names = list(df["adm_name"].unique())
    crmap = {name:i for i,name in enumerate(cr_names)}

    # create dataframe for later operations on the shapefile
    df["cr_id"] = df["adm_name"].apply(lambda x: crmap[x])
    df["fine_id"] = df["adm_id"].apply(lambda x: finalmapping[x])
    df["census"] = df["fine_id"].apply(lambda x: census[x])

    # create coarse census
    crdf = df.groupby(['cr_id']).sum().reset_index()
    cr_census = {id:popcensus for id,popcensus in zip(crdf["cr_id"],crdf["census"])} 
    cr_census_arr = np.zeros(crdf["cr_id"].max()+1)
    for key,value in cr_census.items():
        cr_census_arr[key] = value

    # create mapping between fine and coarse
    fine_to_cr_dict = {fine:cr for cr,fine in zip(df["cr_id"],df["fine_id"])} 
    id_to_cr_id = np.zeros(df["fine_id"].max()+1, dtype=np.uint32)
    for finekey,crkey in fine_to_cr_dict.items():
        id_to_cr_id[finekey] = crkey

    cr_id = np.array(df["cr_id"], dtype=np.uint32)
    fina_cr_ids = cr_id[1:]

    no_valid_ids = list(metadata["wp_no_data"])
    valid_ids = list(valid_census.keys())
    final_num_coarse_regions = len(cr_census_arr)

    # Save metadata
    preproc_data = {
        "features": features, #
        "features_from_built_up_areas": masked_features, #
        "areas": areas, #
        "built_up_areas": built_up_areas, #
        "census": census, #
        "valid_census": valid_census, #
        "cr_census_arr": cr_census_arr, #
        "matches_wp_to_hd": None, #
        "wp_no_data": metadata["wp_no_data"], #
        "valid_ids": valid_ids,  #
        "no_valid_ids": no_valid_ids, #
        "id_to_cr_id": id_to_cr_id, #
        "num_coarse_regions": final_num_coarse_regions, #
        "cr_ids": fina_cr_ids, #
        "geo_metadata": geo_metadata, #
        "id_to_gr_id": id_to_cr_id #
    }

    with open(output_path, 'wb') as handle:
        pickle.dump(preproc_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("wp_regions_path", type=str, help="Shapefile with Worldpop.org administrative regions")
    parser.add_argument("rst_wp_regions_path", type=str,  help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("census_data_path", type=str, help="CSV file containing ")
    parser.add_argument("target_col", type=str, help="Target column")
    parser.add_argument("output_path", type=str, help="Output directory")
    parser.add_argument("dataset_name", type=str, help="Country code")
    parser.add_argument("mode", type=str, help="'Geodata' or 'Sat2Pop'")
    args = parser.parse_args()

    preprocessing_wp_census_data(args.wp_regions_path, args.rst_wp_regions_path, args.census_data_path, 
                                     args.target_col, args.output_path, args.dataset_name, args.mode)

if __name__ == "__main__":
    main()
