import argparse
import pickle
import numpy as np
from osgeo import gdal
from utils import read_input_raster_data, create_map_of_valid_ids, compute_features_from_raw_inputs, \
    mostly_non_empty_map
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region, bool_arr_to_seq_of_indices
import config_pop as cfg
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def compute_and_save_graph(inputs, feats_list, map_valid_ids,
                           k_neigh, perc_subsample, output_dir, n_jobs):

    perc_subsample_int_100 = int(perc_subsample * 100)
    nearest_neigh_dist_path = "{}graph_dist_k_{}_sub_{}.npy".format(output_dir, k_neigh, perc_subsample_int_100)
    nearest_neigh_ind_path = "{}graph_ind_k_{}_sub_{}.npy".format(output_dir, k_neigh, perc_subsample_int_100)
    nearest_neigh_spdist_path = "{}graph_spdist_k_{}_sub_{}.npy".format(output_dir, k_neigh, perc_subsample_int_100)
    # Pre-process input to remove very large numbers outside mask
    min_threshold = 0
    max_threshold = 1000.0
    for k in inputs.keys():
        inputs[k][inputs[k] > max_threshold] = 0
        inputs[k][inputs[k] < min_threshold] = 0
    # Scale features
    all_features = compute_features_from_raw_inputs(inputs, feats_list)
    valid_mask = map_valid_ids.flatten().astype(np.bool)
    valid_seq_all = bool_arr_to_seq_of_indices(valid_mask.astype(np.uint32))
    valid_features = all_features[valid_mask, :]
    scaler = StandardScaler().fit(valid_features)
    norm_all_feats = scaler.transform(all_features)
    valid_norm_feats = norm_all_feats[valid_mask]
    # Compute a dataset for KNN search (reducing the number of samples with features 0)
    mostly_non_empty = mostly_non_empty_map(map_valid_ids, feats_list, inputs, threshold=0.99999, min_val=0.001)
    mostly_non_empty_mask = mostly_non_empty.flatten().astype(np.bool)
    np.random.seed(42)
    select_subset_mask = np.random.rand(map_valid_ids.shape[0] * map_valid_ids.shape[1]) <= perc_subsample  # 0.02
    select_tr_mask = np.multiply(valid_mask, mostly_non_empty_mask).astype(np.bool)
    select_tr_mask = np.multiply(select_tr_mask, select_subset_mask)
    seq_all = np.arange(valid_mask.shape[0]).astype(np.uint32)
    select_tr_seq_all = seq_all[select_tr_mask]
    select_tr_features = norm_all_feats[select_tr_mask, :]
    print("sample search size {}".format(select_tr_features.shape))
    # Obtain nearest neighbours
    neigh = NearestNeighbors(n_neighbors=k_neigh, algorithm='kd_tree', n_jobs=n_jobs)
    neigh.fit(select_tr_features)
    neigh_dist_tr, neigh_ind_tr = neigh.kneighbors(valid_norm_feats)
    neigh_ind_tr = neigh_ind_tr.astype(np.uint32)
    neigh_ind_global = select_tr_seq_all[neigh_ind_tr]
    neigh_ind = valid_seq_all[neigh_ind_global]
    # Compute spatial distance of neighbours
    width = map_valid_ids.shape[1]
    y_coords = seq_all // width
    x_coords = seq_all % width
    coords = np.stack([y_coords, x_coords]).transpose()
    valid_coords = coords[valid_mask, :].astype(np.float32)
    neigh_coords = coords[neigh_ind_global].astype(np.float32)
    list_k_neigh_sp_dist = []
    for k in range(k_neigh):
        sp_dist = np.linalg.norm(valid_coords - neigh_coords[:, k, :], axis=1)
        list_k_neigh_sp_dist.append(sp_dist)
    neigh_sp_dist = np.stack(list_k_neigh_sp_dist).transpose().astype(np.float32)
    # Save neighbours metadata
    if output_dir is not None:
        np.save(nearest_neigh_dist_path, neigh_dist_tr)
        np.save(nearest_neigh_ind_path, neigh_ind)
        np.save(nearest_neigh_spdist_path, neigh_sp_dist)
        print("Saved: nearest neighbour dist {} and ind {}".format(nearest_neigh_dist_path, nearest_neigh_ind_path))


def compute_graph(preproc_data_path, rst_wp_regions_path, dataset_name, k_neigh, perc_subsample, output_dir, n_jobs):
    # Read input data
    input_paths = cfg.input_paths[dataset_name]

    with open(preproc_data_path, 'rb') as handle:
        pdata = pickle.load(handle)

    no_valid_ids = pdata["no_valid_ids"]
    wp_rst_regions = gdal.Open(rst_wp_regions_path).ReadAsArray().astype(np.uint32)
    inputs = read_input_raster_data(input_paths)
    feats_list = inputs.keys()

    # Binary map representing a pixel belong to a region with valid id
    map_valid_ids = create_map_of_valid_ids(wp_rst_regions, no_valid_ids)

    # compute and save neighbours
    compute_and_save_graph(inputs, feats_list, map_valid_ids,
                           k_neigh, perc_subsample, output_dir, n_jobs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preproc_data_path", type=str, help="Preprocessed data of regions (pickle file)")
    parser.add_argument("rst_wp_regions_path", type=str,
                        help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("output_dir", type=str, help="Output dir ")
    parser.add_argument("dataset_name", type=str, help="Dataset name")
    parser.add_argument("k_neigh", type=int, help="Number of neighbours")
    parser.add_argument("perc_subsample", type=float, help="Number of neighbours")
    parser.add_argument("n_jobs", type=int, help="Num of processors to be used")
    args = parser.parse_args()

    compute_graph(args.preproc_data_path, args.rst_wp_regions_path,
                  args.dataset_name, args.k_neigh, args.perc_subsample, args.output_dir, args.n_jobs)


if __name__ == "__main__":
    main()
