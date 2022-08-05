import argparse
from osgeo import gdal
import numpy as np
import csv
import pickle
import config_pop as cfg
from cy_utils import count_matches, compute_area_of_regions, compute_accumulated_values_by_region
from utils import read_input_raster_data_Sat2Pop, read_shape_layer_data, read_input_raster_data, preprocess_census_targets, compute_grouped_values


def get_valid_ids(wp_ids, matches_wp_to_hd, wp_no_data):
    valid_ids = []
    # Remove regions with no data value or with no matches in humdata.org
    ids_with_no_matches = [id for id in matches_wp_to_hd.keys() if matches_wp_to_hd[id] is None]
    for id in wp_ids:
        if (id not in wp_no_data) and (id not in ids_with_no_matches):
            valid_ids.append(id)
    return valid_ids


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


def match_raster_ids(raster1, raster2, raster1_no_data, raster2_no_data, offset_y=0, offset_x=0):
    assert raster1.shape == raster2.shape

    if (offset_y > 0) and (offset_x > 0):
        raster2_new = np.zeros(raster2.shape).astype(np.uint32)
        raster2_new[offset_y:, offset_x:] = raster2[:-offset_y, :-offset_x]
        raster2 = raster2_new

    map1_valid_ids = np.ones(raster1.shape).astype(np.uint32)
    for nd in raster1_no_data:
        map1_valid_ids[raster1 == nd] = 0

    map2_valid_ids = np.ones(raster2.shape).astype(np.uint32)
    for nd in raster2_no_data:
        map2_valid_ids[raster2 == nd] = 0

    final_mask = np.multiply((map1_valid_ids == 1).astype(np.uint32), (map2_valid_ids == 1).astype(np.uint32))

    num_labels_raster1 = int(np.max(raster1) + 1)
    num_labels_raster2 = int(np.max(raster2) + 1)

    threshold = 0.5
    matches = count_matches(raster1, raster2, final_mask, num_labels_raster1, num_labels_raster2)

    id_best_match = {}
    score_best_match = {}
    for id1 in range(num_labels_raster1):
        id_best_match[id1] = None
        score_best_match[id1] = 0
        total_matches = np.sum(matches[id1, :])
        for id2 in range(num_labels_raster2):
            if matches[id1, id2] > score_best_match[id1]:
                score_best_match[id1] = matches[id1, id2]
                id_best_match[id1] = id2
        if total_matches == 0:
            score_best_match[id1] = 0
            id_best_match[id1] = None
            print("no matches for id1 : {}".format(id1))
        else:
            score_best_match[id1] = score_best_match[id1] / float(total_matches)

        if score_best_match[id1] <= threshold:
            print("warning score smaller than threshold: {} for id1 : {}, id2: {}".format(score_best_match[id1], id1,
                                                                                          id_best_match[id1]))

    return id_best_match


# def compute_area_of_regions(regions, map_valid_ids, num_ids):
#     # def compute_area_of_regions(np.ndarray[np.uint32_t, ndim=2] regions, np.ndarray[np.uint32_t, ndim=2] map_valid_ids, int num_labels):

#     if np.isfortran(regions):
#         raise ValueError("The input image is not C-contiguous")

#     h = regions.shape[0]
#     w = regions.shape[1]

#     areas = np.zeros(num_ids, dtype=np.uint32)

#     for i in range(h):
#         for j in range(w):
#             if map_valid_ids[i, j] == 1:
#                 areas[regions[i, j]] = areas[regions[i, j]] + 1

#     for region in regions:
#         asdf

#     return areas

# def compute_area(regions, inputs, no_data_vals=None, buildings_mask=None):
    # feats_list = list(inputs.keys())
    # ids = list(np.unique(regions))
    # num_ids = len(ids)

    # map_valid_ids = np.ones(regions.shape).astype(np.uint32)
    # for nd in no_data_vals:
    #     map_valid_ids[regions == nd] = 0
    
    # print("regions.shape {}".format(regions.shape))

    # return compute_area_of_regions(regions, map_valid_ids, num_ids)

def compute_agg_features_from_raster(regions, inputs, no_data_vals=None, buildings_mask=None):
    feats_list = list(inputs.keys())
    ids = list(np.unique(regions))
    num_ids = len(ids)

    map_valid_ids = np.ones(regions.shape).astype(np.uint32)
    for nd in no_data_vals:
        map_valid_ids[regions == nd] = 0
    
    print("regions.shape {}".format(regions.shape))

    areas = compute_area_of_regions(regions, map_valid_ids, num_ids)
    
    built_up_areas = None
    if buildings_mask is not None:
       masked_map_valid_ids = np.multiply(map_valid_ids, buildings_mask).astype(np.uint32)
       built_up_areas = compute_area_of_regions(regions, masked_map_valid_ids, num_ids)
    
    features_arr = []
    masked_features_arr = []
    # Compute features and areas
    for k, feat in enumerate(feats_list):
        input = inputs[feat]     
        accumulated_features = compute_accumulated_values_by_region(regions, input, map_valid_ids, num_ids)
        features_arr.append(accumulated_features)
        
        if buildings_mask is not None:
            masked_map_valid_ids = np.multiply(map_valid_ids, buildings_mask).astype(np.uint32)
            accumulated_masked_features = compute_accumulated_values_by_region(regions, input, masked_map_valid_ids, num_ids)
            masked_features_arr.append(accumulated_masked_features)    

    features_arr = np.array(features_arr).astype(np.float32)
    features_arr = features_arr.transpose()
    
    if buildings_mask is not None:
        masked_features_arr = np.array(masked_features_arr).astype(np.float32)
        masked_features_arr = masked_features_arr.transpose()

    features = {id: {} for id in range(num_ids)}
    masked_features = {id: {} for id in range(num_ids)}
    for id in range(num_ids):
        for k, feat in enumerate(feats_list):
            features[id][feat] = features_arr[id, k]
            masked_features[id][feat] = masked_features_arr[id, k]

    regions_with_no_buildings = []
    
    for id in ids:
        if areas[id] > 0:
            for feat in feats_list:
                features[id][feat] /= areas[id]
        else:
            print("no buildings found in {}".format(id))
            regions_with_no_buildings.append(id)
        
        if built_up_areas[id] > 0:
            for feat in feats_list:
                masked_features[id][feat] /= built_up_areas[id]

    print("number of regions with no buildings {}".format(len(regions_with_no_buildings)))
    print(regions_with_no_buildings)

    return features, areas, masked_features, built_up_areas


def preprocessing_pop_data(hd_regions_path, rst_hd_regions_path, rst_wp_regions_path,
                           census_data_path, output_path, dataset_name, target_col, mode):
    # Read input data
    if mode=="Geodata":
        input_paths = cfg.input_paths[dataset_name]
        metadata = cfg.metadata[dataset_name]
        inputs = read_input_raster_data(input_paths)
    elif mode=="Sat2Pop":
        input_paths = cfg.input_paths_sat2pop[dataset_name]
        metadata = cfg.metadata[dataset_name]
        inputs = read_input_raster_data_Sat2Pop(input_paths)
    buildings = inputs["buildings"]
    buildings_mask = buildings > 0
    hd_regions = read_shape_layer_data(hd_regions_path)
    all_census = read_multiple_targets_from_csv(census_data_path)
    print("census_data size {}".format(len(all_census.keys())))
    hd_rst_regions = gdal.Open(rst_hd_regions_path).ReadAsArray().astype(np.uint32)
    wp_rst_regions = gdal.Open(rst_wp_regions_path).ReadAsArray().astype(np.uint32)

    # Store information about the administrative region parents of humdata.org
    hd_parents = {}
    cr_ids = []
    for b in hd_regions:
        hd_parents[b[cfg.col_finest_level_seq_id]] = b[cfg.col_coarse_level_seq_id]

        cr_ids.append(b[cfg.col_coarse_level_seq_id])

    cr_ids = np.unique(cr_ids).astype(np.uint32)
    num_coarse_regions = len(cr_ids) + 1 # indices start in 1 in the shp file, the index 0 corresponds to no data value

    # Match Humdata (hd) and WorldPop (wp) administrative regions
    matches_wp_to_hd = match_raster_ids(wp_rst_regions, hd_rst_regions, metadata["wp_no_data"], metadata["hd_no_data"],
                                        offset_y=1, offset_x=0)

    # Verify if wp regions are matched to just one hd region
    acc_matched = []
    duplicated = []
    for id in matches_wp_to_hd.keys():
        val = matches_wp_to_hd[id]
        if val in acc_matched:
            print("duplicated hd_val {}".format(val))
            duplicated.append(val)
        else:
            acc_matched.append(val)
    
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

    # Accumulate features
    features, areas, masked_features, built_up_areas = compute_agg_features_from_raster(wp_rst_regions, inputs, no_data_vals=metadata["wp_no_data"], buildings_mask=buildings_mask)

    # Get census target
    census = all_census[target_col]
    census = preprocess_census_targets(census)

    # Get valid ids
    wp_ids = list(np.unique(wp_rst_regions))
    valid_ids = get_valid_ids(wp_ids, matches_wp_to_hd, metadata["wp_no_data"])

    # Get ids of no data
    no_valid_ids = list(metadata["wp_no_data"])
    for id in matches_wp_to_hd.keys():
        if matches_wp_to_hd[id] is None:
            no_valid_ids.append(id)

    # Get regions parent's id (WorldPop id to parent region in humdata)
    num_wp_ids = len(wp_ids)
    id_to_gr_id = np.zeros(num_wp_ids).astype(np.uint32)
    for id in valid_ids:
        hd_id = matches_wp_to_hd[id]
        gid = hd_parents[hd_id]
        id_to_gr_id[id] = gid
    
    # correct sequential IDs of coarse level regions
    gr_ids_with_no_data = []
    for gr_id in range(1, num_coarse_regions):
        if gr_id not in id_to_gr_id:
            gr_ids_with_no_data.append(gr_id)
    
    if len(gr_ids_with_no_data) == 0:
        id_to_cr_id = id_to_gr_id
        final_num_coarse_regions = num_coarse_regions
        fina_cr_ids = cr_ids
    else:
        id_to_cr_id = np.zeros(num_wp_ids).astype(np.uint32)
        for id in range(num_wp_ids):
            gid = id_to_gr_id[id]
            shift_value = 0
            for gr_id_nodata in gr_ids_with_no_data:
                if gid > gr_id_nodata:
                    shift_value += 1
            id_to_cr_id[id] = gid - shift_value

        final_num_coarse_regions = len(np.unique(id_to_cr_id))
        fina_cr_ids = np.arange(final_num_coarse_regions, dtype=np.uint32)

    # Valid WorldPop census data
    valid_census = {}
    for id in valid_ids:
        valid_census[id] = census[id]

    # Aggregate targets : coarse census
    cr_census = compute_grouped_values(valid_census, valid_ids, id_to_cr_id)
    cr_census_arr = np.zeros(num_coarse_regions).astype(np.float32)
    for gid in cr_census.keys():
        cr_census_arr[gid] = cr_census[gid]

    # Save metadata
    preproc_data = {
        "features": features,
        "features_from_built_up_areas": masked_features,
        "areas": areas,
        "built_up_areas": built_up_areas,
        "census": census,
        "valid_census": valid_census,
        "cr_census_arr": cr_census_arr,
        "matches_wp_to_hd": matches_wp_to_hd,
        "wp_no_data": metadata["wp_no_data"],
        "valid_ids": valid_ids,
        "no_valid_ids": no_valid_ids,
        "id_to_cr_id": id_to_cr_id,
        "num_coarse_regions": final_num_coarse_regions,
        "cr_ids": fina_cr_ids,
        "geo_metadata": geo_metadata,
        "id_to_gr_id": id_to_gr_id
    }

    with open(output_path, 'wb') as handle:
        pickle.dump(preproc_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hd_regions_path", type=str, help="Shapefile with humdata.org administrative regions information")
    parser.add_argument("rst_hd_regions_path", type=str,  help="Raster of humdata.org administrative regions information")
    parser.add_argument("rst_wp_regions_path", type=str,
                        help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("census_data_path", type=str, help="CSV file containing ")
    parser.add_argument("output_path", type=str, help="Output path")
    parser.add_argument("dataset_name", type=str, help="Dataset name")
    parser.add_argument("target_col", type=str, help="Target column")
    parser.add_argument("mode", type=str, help="'Geodata' or 'Sat2Pop'")
    args = parser.parse_args()

    preprocessing_pop_data(args.hd_regions_path, args.rst_hd_regions_path,
                           args.rst_wp_regions_path, args.census_data_path, args.output_path,
                           args.dataset_name, args.target_col, args.mode)


if __name__ == "__main__":
    main()
