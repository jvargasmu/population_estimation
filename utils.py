import fiona
from osgeo import gdal
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def get_properties_dict(data_dict_orig):
    data_dict = []
    for data_row in data_dict_orig:
        data_dict.append(data_row["properties"])
    return data_dict


def read_input_raster_data(input_paths):
    inputs = {}
    for kinp in input_paths.keys():
        print("read {}".format(input_paths[kinp]))
        inputs[kinp] = gdal.Open(input_paths[kinp]).ReadAsArray().astype(np.float32)
    return inputs


def read_shape_layer_data(shape_layer_path):
    with fiona.open(shape_layer_path) as reader:
        layer_data_orig = [elem for elem in reader]
    layer_data = get_properties_dict(layer_data_orig)
    return layer_data


def compute_performance_metrics(preds_dict, gt_dict):
    assert len(preds_dict) == len(gt_dict)

    preds = []
    gt = []
    ids = preds_dict.keys()
    for id in ids:
        preds.append(preds_dict[id])
        gt.append(gt_dict[id])

    preds = np.array(preds).astype(np.float)
    gt = np.array(gt).astype(np.float)

    r2 = r2_score(gt, preds)
    mae = mean_absolute_error(gt, preds)
    mse = mean_squared_error(gt, preds)

    return r2, mae, mse


def write_geolocated_image(image, output_path, src_geo_transform, src_projection):
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_path, image.shape[1], image.shape[0], 1, gdal.GDT_Float32, options=['COMPRESS=LZW'])
    outdata.SetGeoTransform(src_geo_transform)
    outdata.SetProjection(src_projection)
    outdata.GetRasterBand(1).WriteArray(image)
    outdata.FlushCache()
    outdata = None
    ds = None


def convert_str_to_int_keys(data_dict_orig):
    data_dict = {}
    for k in data_dict_orig.keys():
        data_dict[int(k)] = data_dict_orig[k]
    return data_dict


def convert_dict_vals_str_to_float(data_dict_orig):
    return {k: float(data_dict_orig[k]) for k in data_dict_orig.keys()}


def preprocess_census_targets(data_dict_orig):
    data_dict = convert_str_to_int_keys(data_dict_orig)
    data_dict = convert_dict_vals_str_to_float(data_dict)
    return data_dict


def create_map_of_valid_ids(regions, no_valid_ids):
    map_valid_ids = np.ones(regions.shape).astype(np.uint32)
    for id in no_valid_ids:
        map_valid_ids[regions == id] = 0
    return map_valid_ids


def create_valid_mask_array(num_ids, valid_ids):
    valid_ids_mask = np.zeros(num_ids)
    for id in valid_ids:
        valid_ids_mask[id] = 1
    return valid_ids_mask


def compute_grouped_values(data, valid_ids, id_to_gid):
    # Initialize target values
    grouped_data = {}
    for id in valid_ids:
        gid = id_to_gid[id]
        if gid not in grouped_data.keys():
            grouped_data[gid] = 0
    # Aggregate targets
    for id in valid_ids:
        gid = id_to_gid[id]
        grouped_data[gid] += data[id]
    return grouped_data


def transform_dict_to_array(data_dict):
    return np.array([data_dict[k] for k in data_dict.keys()]).astype(np.float32)


def transform_dict_to_matrix(data_dict):
    assert len(data_dict.keys()) > 0
    # get size of matrix
    keys = list(data_dict.keys())
    num_rows = len(keys)
    first_row = data_dict[keys[0]]
    col_keys = list(first_row.keys())
    num_cols = len(col_keys)
    # fill matrix
    data_array = np.zeros((num_rows, num_cols)).astype(np.float32)
    for i, rk in enumerate(keys):
        for j, ck in enumerate(col_keys):
            data_array[i, j] = data_dict[rk][ck]

    return data_array