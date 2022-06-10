import argparse
import numpy as np
from osgeo import gdal
import config_pop as cfg
import csv
import pickle
from utils import read_shape_layer_data, read_input_raster_data, preprocess_census_targets, compute_grouped_values
from preprocessing_pop_data import compute_agg_features_from_raster


def read_multiple_targets_from_zmb_csv(csv_path, id_col, selected_cols, header_row_index = 3):
    data_per_col = {}
    col_to_index = {}
    index_to_col = {}
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for i, row in enumerate(reader):
            if i > header_row_index:
                id = row[col_to_index[id_col]]
                for j in range(len(index_to_col.keys())):
                    data_per_col[index_to_col[j]][id] = row[j]
            else:
                # header
                for j, val in enumerate(row):
                    col_to_index[val] = j
                    index_to_col[j] = val
                    data_per_col[val] = {}
    return data_per_col


def get_census_data_by_year(census_data, geo_match_to_id, year_col):
    census_by_geo_match = census_data[year_col]
    print("census_data size {}".format(len(census_by_geo_match.keys())))
    # Store information about the administrative region parents
    id_to_geo_match = {geo_match_to_id[k] : k for k in geo_match_to_id.keys()}
    cr_geo_match_to_cr_id = {}
    cr_geo_match_list = set()
    for geo_match in census_by_geo_match.keys():
        cr_geo_match = "_".join(geo_match.split("_")[:-1])
        cr_geo_match_list.add(cr_geo_match)
    cr_geo_match_list = list(cr_geo_match_list)
    cr_geo_match_list.sort()
    
    cr_sid = 1
    for cr_geo_match in cr_geo_match_list:
        cr_geo_match_to_cr_id[cr_geo_match] = cr_sid
        cr_sid += 1
    
    census = {geo_match_to_id[geo_match] : census_by_geo_match[geo_match] for geo_match in census_by_geo_match.keys()}
    
    num_coarse_regions = len(cr_geo_match_list) + 1  # indices start in 1 in the shp file, the index 0 corresponds to no data value

    return census, num_coarse_regions, id_to_geo_match, cr_geo_match_to_cr_id


def geo_match_to_coarse(geo_match):
    return "_".join(geo_match.split("_")[:-1])


def preprocessing_zmb_census_data(rst_wp_regions_path, census_data_path, 
                                    target_year, output_path, dataset_name):
    # Get census data from csv file
    numeric_cols = ["BTOTL_{}".format(year) for year in range(2010, 2021)] # data is available from 2010 to 2020 
    selected_cols = ["GEO_MATCH", "GEO_CONCAT", "CNTRY_NAME", "ADM1_NAME", "ADM2_NAME", "ADM3_NAME", "ADM4_NAME", "ADM_LEVEL"] + numeric_cols
    no_valid_ids = [0]
    
    id_col = "GEO_MATCH"
    pdata = read_multiple_targets_from_zmb_csv(census_data_path, id_col, selected_cols)
    pdata = {id:pdata[id] for id in selected_cols}
    for col in numeric_cols:
        for id in pdata[col].keys():
            pdata[col][id] = int(pdata[col][id].replace(",", ""))
    
    # Select samples only from level 4
    ids_level4 = []
    for id in pdata["ADM_LEVEL"].keys():
        if pdata["ADM_LEVEL"][id] == "4":
             ids_level4.append(id)
    
    selected_pdata = {}
    for col in selected_cols:
        selected_pdata[col] = {}
        for id in ids_level4:
            selected_pdata[col][id] = pdata[col][id]
    census_data = selected_pdata
    
    # Create mapping of GEO_MATCH to SID
    geo_match_to_id = {}
    sid = 1
    for col in selected_pdata["GEO_MATCH"]:
        geo_match_to_id[col] = sid
        sid += 1
    
    target_year_col = "BTOTL_{}".format(target_year)
    
    # Read input data
    input_paths = cfg.input_paths[dataset_name]
    metadata = cfg.metadata[dataset_name]
    inputs = read_input_raster_data(input_paths)
    buildings = inputs["buildings"]
    buildings_mask = buildings > 0
    wp_rst_regions_gdal = gdal.Open(rst_wp_regions_path)
    wp_rst_regions = wp_rst_regions_gdal.ReadAsArray().astype(np.uint32)

    geo_transform = wp_rst_regions_gdal.GetGeoTransform()
    projection = wp_rst_regions_gdal.GetProjection()
    geo_metadata = {"geo_transform": geo_transform, "projection": projection}

    # Get census by year
    census, num_coarse_regions, id_to_geo_match, cr_geo_match_to_cr_id = get_census_data_by_year(census_data, geo_match_to_id, target_year_col)
    
    # Get geo spatial references
    if "buildings_maxar" in input_paths.keys():
        buildings_path = input_paths["buildings_maxar"]
    if "buildings_google" in input_paths.keys():
        buildings_path = input_paths["buildings_google"]
    if "buildings" in input_paths.keys():
        buildings_path = input_paths["buildings"]
    
    source = gdal.Open(buildings_path)
    geo_transform = source.GetGeoTransform()
    projection = source.GetProjection()
    geo_metadata = {"geo_transform": geo_transform, "projection": projection}

    # Accumulate features
    features, areas, masked_features, built_up_areas = compute_agg_features_from_raster(wp_rst_regions, inputs, no_data_vals=metadata["wp_no_data"], buildings_mask=buildings_mask)

    # Get valid ids
    max_sid = np.max(np.array([geo_match_to_id[k] for k in geo_match_to_id.keys()]))
    wp_ids = list(np.unique(wp_rst_regions))
    valid_ids = list( set(list(np.arange(max_sid+1))) - set(no_valid_ids) )
    
    # Get regions parent's id (from the CSV columns itself)
    num_wp_ids = len(wp_ids)
    id_to_gr_id = np.zeros(num_wp_ids).astype(np.uint32)
    for id in valid_ids:
        geo_match = id_to_geo_match[id]
        cr_geo_match = geo_match_to_coarse(geo_match)
        cr_id = cr_geo_match_to_cr_id[cr_geo_match]
        id_to_gr_id[id] = cr_id

    # Set final sequential IDs of coarse level regions
    id_to_cr_id = id_to_gr_id
    cr_ids = np.arange(num_coarse_regions)
    
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
        "matches_wp_to_hd": None,
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
    parser.add_argument("rst_wp_regions_path", type=str,
                        help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("census_data_path", type=str, help="CSV file containing ")
    parser.add_argument("target_year", type=int, help="Target column")
    parser.add_argument("output_path", type=str, help="Output directory")
    parser.add_argument("dataset_name", type=str, help="Country code")
    args = parser.parse_args()

    preprocessing_zmb_census_data(args.rst_wp_regions_path, args.census_data_path, 
                                     args.target_year, args.output_path, args.dataset_name)


if __name__ == "__main__":
    main()
