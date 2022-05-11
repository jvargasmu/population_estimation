import os
import argparse
import numpy as np
from osgeo import gdal
import pickle
from utils import write_geolocated_image


def visualization_of_fine_pred_and_gt(rst_wp_regions_path, preproc_data_path, pred_map_path, output_dir):
    # Read raster data
    wp_rst_regions_gdal = gdal.Open(rst_wp_regions_path)
    wp_rst_regions = wp_rst_regions_gdal.ReadAsArray().astype(np.uint32)
    geo_transform = wp_rst_regions_gdal.GetGeoTransform()
    projection = wp_rst_regions_gdal.GetProjection()
    
    pred_map = gdal.Open(pred_map_path).ReadAsArray().astype(np.float32)
    
    with open(preproc_data_path, 'rb') as handle:
        pdata = pickle.load(handle)
    
    valid_ids = pdata["valid_ids"]
    valid_census = pdata["valid_census"]
    
    # Group predictions into fine level map
    valid_f_preds = {}
    
    for id in valid_ids:
        valid_f_preds[id] = np.sum(pred_map[(wp_rst_regions == id) & (~np.isnan(pred_map)) ])

    # Create map of fine level predictions and census data
    fine_map_pred = np.zeros(wp_rst_regions.shape, dtype=np.float32)
    fine_map_gt = np.zeros(wp_rst_regions.shape, dtype=np.float32)
    fine_map_mape = np.zeros(wp_rst_regions.shape, dtype=np.float32)
    fine_map_mpe = np.zeros(wp_rst_regions.shape, dtype=np.float32)
    for id in valid_ids:
        fine_map_pred[wp_rst_regions == id] = valid_f_preds[id]
        fine_map_gt[wp_rst_regions == id] = valid_census[id]
        if  valid_census[id] > 0:
            fine_map_mape[wp_rst_regions == id] = abs(valid_f_preds[id] - valid_census[id]) / valid_census[id]
            fine_map_mpe[wp_rst_regions == id] = (valid_f_preds[id] - valid_census[id]) / valid_census[id]
    
    #fine_map_pred[wp_rst_regions <= 2] = np.nan
    #fine_map_gt[wp_rst_regions <= 2] = np.nan
    
    fine_map_pred_path = os.path.join(output_dir, "fine_map_pred.tif")
    fine_map_gt_path = os.path.join(output_dir, "fine_map_gt.tif")
    fine_map_mape_path = os.path.join(output_dir, "fine_map_mape.tif")
    fine_map_mpe_path = os.path.join(output_dir, "fine_map_mpe.tif")
    
    write_geolocated_image(fine_map_pred, fine_map_pred_path, geo_transform, projection)
    write_geolocated_image(fine_map_gt, fine_map_gt_path, geo_transform, projection)
    write_geolocated_image(fine_map_mape, fine_map_mape_path, geo_transform, projection)
    write_geolocated_image(fine_map_mpe, fine_map_mpe_path, geo_transform, projection)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rst_wp_regions_path", type=str, help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("preproc_data_path",  type=str, help="Preprocessed data of regions (pickle file)")
    parser.add_argument("pred_map_path", type=str, help="Population prediction map")
    parser.add_argument("output_dir", type=str, help="Output directory")
    args = parser.parse_args()

    visualization_of_fine_pred_and_gt(args.rst_wp_regions_path, args.preproc_data_path, args.pred_map_path, args.output_dir)


if __name__ == "__main__":
    main()
