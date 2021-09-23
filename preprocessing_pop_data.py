import argparse
from osgeo import gdal
import numpy as np
import csv
import pickle
import config_pop as cfg
from cy_utils import count_matches, compute_area_of_regions, compute_accumulated_values_by_region
from utils import read_shape_layer_data, read_input_raster_data


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
            print("**** no matches for id1 : {}".format(id1))
        else:
            score_best_match[id1] = score_best_match[id1] / float(total_matches)

        if score_best_match[id1] <= threshold:
            print("warning score smaller than threshold: {} for id1 : {}, id2: {}".format(score_best_match[id1], id1,
                                                                                          id_best_match[id1]))

    return id_best_match


def compute_agg_features_from_raster(regions, inputs, no_data_vals=None):
    feats_list = list(inputs.keys())
    ids = list(np.unique(regions))
    num_ids = len(ids)

    map_valid_ids = np.ones(regions.shape).astype(np.uint32)
    for nd in no_data_vals:
        map_valid_ids[regions == nd] = 0

    print("regions.shape {}".format(regions.shape))

    areas = compute_area_of_regions(regions, map_valid_ids, num_ids)
    features_arr = []

    # Compute features and areas
    for k, feat in enumerate(feats_list):
        input = inputs[feat]
        accumulated_features = compute_accumulated_values_by_region(regions, input, map_valid_ids, num_ids)
        features_arr.append(accumulated_features)

    features_arr = np.array(features_arr).astype(np.float32)
    features_arr = features_arr.transpose()

    features = {id: {} for id in range(num_ids)}
    for id in range(num_ids):
        for k, feat in enumerate(feats_list):
            features[id][feat] = features_arr[id, k]

    regions_with_no_buildings = []
    for id in ids:
        if areas[id] > 0:
            for feat in feats_list:
                features[id][feat] /= areas[id]
        else:
            print("no buildings found in {}".format(id))
            regions_with_no_buildings.append(id)

    print("number of regions with no buildings {}".format(len(regions_with_no_buildings)))
    print(regions_with_no_buildings)

    return features, areas


def preprocessing_pop_data(hd_regions_path, rst_hd_regions_path, rst_wp_regions_path,
                           census_data_path, output_path, dataset_name):
    # Read input data
    input_paths = cfg.input_paths[dataset_name]
    metadata = cfg.metadata[dataset_name]
    inputs = read_input_raster_data(input_paths)
    hd_regions = read_shape_layer_data(hd_regions_path)
    census_target = read_multiple_targets_from_csv(census_data_path)
    print("census_data size {}".format(len(census_target.keys())))
    hd_rst_regions = gdal.Open(rst_hd_regions_path).ReadAsArray().astype(np.uint32)
    wp_rst_regions = gdal.Open(rst_wp_regions_path).ReadAsArray().astype(np.uint32)

    # Store information about the administrative region parents of humdata.org
    hd_parents = {}
    gids = []
    for b in hd_regions:
        hd_parents[b[cfg.col_finest_level_seq_id]] = {cfg.col_finest_level_name: b[cfg.col_finest_level_name],
                                                      cfg.col_finest_level_code: b[cfg.col_finest_level_code],
                                                      cfg.col_coarse_level_name: b[cfg.col_coarse_level_name],
                                                      cfg.col_coarse_level_code: b[cfg.col_coarse_level_code],
                                                      cfg.col_coarse_level_seq_id: b[cfg.col_coarse_level_seq_id]
                                                      }
        gids.append(b[cfg.col_coarse_level_seq_id])

    gids = np.unique(gids).astype(np.uint32)
    num_groups = len(gids)

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

    print([id for id in matches_wp_to_hd.keys() if matches_wp_to_hd[id] is None])

    # Accumulate features
    features, areas = compute_agg_features_from_raster(wp_rst_regions, inputs, no_data_vals=metadata["wp_no_data"])

    # Save metadata
    preproc_data = {
        "features": features,
        "areas": areas,
        "wp_census_target": census_target,
        "matches_wp_to_hd": matches_wp_to_hd,
        "wp_no_data": metadata["wp_no_data"],
        "hd_parents": hd_parents,
        "num_groups": num_groups,
        "gids": gids
    }

    with open(output_path, 'wb') as handle:
        pickle.dump(preproc_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hd_regions_path", type=str,
                        help="Shapefile with humdata.org administrative regions information")
    parser.add_argument("rst_hd_regions_path", type=str,
                        help="Raster of humdata.org administrative regions information")
    parser.add_argument("rst_wp_regions_path", type=str,
                        help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("census_data_path", type=str, help="CSV file containing ")
    parser.add_argument("output_path", type=str, help="Output path")
    parser.add_argument("dataset_name", type=str, help="Dataset name")
    args = parser.parse_args()

    preprocessing_pop_data(args.hd_regions_path, args.rst_hd_regions_path,
                           args.rst_wp_regions_path, args.census_data_path, args.output_path, args.dataset_name)


if __name__ == "__main__":
    main()