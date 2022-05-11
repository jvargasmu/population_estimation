import os
import argparse
import numpy as np
from osgeo import gdal
import pickle
from utils import write_geolocated_image
import config_pop as cfg
from utils import read_input_raster_data


def visualization_of_ocr_pred_and_gt(dataset_name, rst_wp_regions_path, preproc_data_path, pred_map_path, output_dir):
    # Building occupancy rate comparison options
    # Map1-option1: scale_map (per pixel estimations of scale), average of those estimations per ADM [we are computing average of averages]
    # Map1-option2: pred_map is aggregated-sum per ADM and then we divide it by the number of buildings per ADM
    # Map2-option1: divide census data by the number of buildings in each ADM
    # Maps to compare: Map1-option2 vs Map2-option1 (This maps give a sense of how densely populated are different regions)
    # The Percentage error per administrative regions is the sames as comparing population predictions
    # Read raster data
    input_paths = cfg.input_paths[dataset_name]
    inputs = read_input_raster_data(input_paths)
    input_buildings = inputs["buildings"]
    
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
    valid_ocr_census = {}
    valid_ocr_preds = {}
    for id in valid_ids:
        pred_pop_per_region = np.sum(pred_map[(wp_rst_regions == id) & (~np.isnan(pred_map)) ])
        num_buildings_per_region = np.sum(input_buildings[(wp_rst_regions == id) & (~np.isnan(input_buildings)) & (input_buildings>=0) ])
        valid_ocr_census[id] = valid_census[id] / num_buildings_per_region
        valid_ocr_preds[id] = pred_pop_per_region / num_buildings_per_region
    # Create map of fine level predictions and census data
    fine_map_pred = np.zeros(wp_rst_regions.shape, dtype=np.float32)
    fine_map_gt = np.zeros(wp_rst_regions.shape, dtype=np.float32)
    fine_map_mape = np.zeros(wp_rst_regions.shape, dtype=np.float32)
    fine_map_mpe = np.zeros(wp_rst_regions.shape, dtype=np.float32)
    for id in valid_ids:
        fine_map_pred[wp_rst_regions == id] = valid_ocr_preds[id]
        fine_map_gt[wp_rst_regions == id] = valid_ocr_census[id]
        if  valid_ocr_census[id] > 0:
            fine_map_mape[wp_rst_regions == id] = abs(valid_ocr_preds[id] - valid_ocr_census[id]) / valid_ocr_census[id]
            fine_map_mpe[wp_rst_regions == id] = (valid_ocr_preds[id] - valid_ocr_census[id]) / valid_ocr_census[id]
    
    #fine_map_pred[wp_rst_regions <= 2] = np.nan
    #fine_map_gt[wp_rst_regions <= 2] = np.nan
    
    fine_map_pred_path = os.path.join(output_dir, "ocr_fine_map_pred.tif")
    fine_map_gt_path = os.path.join(output_dir, "ocr_fine_map_gt.tif")
    fine_map_mape_path = os.path.join(output_dir, "ocr_fine_map_mape.tif")
    fine_map_mpe_path = os.path.join(output_dir, "ocr_fine_map_mpe.tif")
    
    write_geolocated_image(fine_map_pred, fine_map_pred_path, geo_transform, projection)
    write_geolocated_image(fine_map_gt, fine_map_gt_path, geo_transform, projection)
    write_geolocated_image(fine_map_mape, fine_map_mape_path, geo_transform, projection)
    write_geolocated_image(fine_map_mpe, fine_map_mpe_path, geo_transform, projection)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., tza,uga)")
    parser.add_argument("rst_wp_regions_path", type=str, help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("preproc_data_path",  type=str, help="Preprocessed data of regions (pickle file)")
    parser.add_argument("pred_map_path", type=str, help="Population prediction map")
    parser.add_argument("output_dir", type=str, help="Output directory")
    args = parser.parse_args()

    visualization_of_ocr_pred_and_gt(args.dataset_name, args.rst_wp_regions_path, args.preproc_data_path, args.pred_map_path, args.output_dir)


if __name__ == "__main__":
    main()
