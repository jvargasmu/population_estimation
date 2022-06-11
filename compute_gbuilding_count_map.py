import argparse
from itertools import count
# from osgeo import gdal
import rasterio as rs
import csv
import numpy as np
import math
import os
from tqdm import tqdm
import pandas as pd


def write_geoloc_image(image, output_path, geo_ref_path):
    source = gdal.Open(geo_ref_path)

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_path, image.shape[1], image.shape[0], 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(source.GetGeoTransform())  
    outdata.SetProjection(source.GetProjection())  
    outdata.GetRasterBand(1).WriteArray(image)
    #save to disk
    outdata.FlushCache()  
    outdata = None
    ds = None

def rounder(values):
    def f(x):
        idx = np.argmin(np.abs(values - x))
        return values[idx]
    return np.frompyfunc(f, 1, 1)


def compute_percentage_of_built_up_area(regions_path, csv_dir, output_path, mode="count"):
    # Get csv paths
    csv_paths = [os.path.join(csv_dir, elem) for elem in os.listdir(csv_dir) if elem.endswith(".csv")]
    # Read image
    with rs.open(regions_path) as guide_raster:
        meta = guide_raster.meta.copy()

    # img_obj = gdal.Open(regions_path)
    # Get image metadata
    xpixel, _, xmin, _, ypixel, ymax = meta["transform"][0], meta["transform"][1], meta["transform"][2], meta["transform"][3], meta["transform"][4], meta["transform"][5]
    # ypixel, _, ymin, _, xpixel, xmax = meta["transform"][0], meta["transform"][1], meta["transform"][2], meta["transform"][3], meta["transform"][4], meta["transform"][5]
    # xmin, xpixel, _, ymax, _, ypixel = img_obj.GetGeoTransform()
    width, height = meta["width"], meta["height"]
    # width, height = img_obj.RasterXSize, img_obj.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel
    print("ymin {} ymax {} xmin {}, xmax {}, width {}, height {}, xpixel {} ypixel {}".format(ymin, ymax, xmin, xmax,
                                                                                              width, height, xpixel,
                                                                                              ypixel))
    
    counts = np.zeros((height, width)).astype(np.float32)
    area_map = np.zeros((height, width)).astype(np.float32)
     
    for csv_path in csv_paths:
        with open(csv_path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(tqdm(csv_reader)):
                if i > 0:
                    latitude = float(row[0])
                    longitude = float(row[1])
                    area = float(row[2])
                    confidence = float(row[3])

                    pix_y = int(math.floor((latitude - ymax) / -abs(ypixel)))
                    pix_x = int(math.floor((longitude - xmin) / xpixel))

                    if (pix_y >= 0 and pix_y < width) and (pix_x >= 0 and pix_x < height):
                        counts[pix_y, pix_x] += 1 
                        area_map[pix_y, pix_x] += ((area_map[pix_y, pix_x] * (counts[pix_y, pix_x]-1)) + area) / (counts[pix_y, pix_x])

    if mode=="count":
        output = counts
    elif mode=="area":
        output = area_map

    # write_geoloc_image(output, output_path, regions_path)

    with rs.Env(): 
        meta.update({"driver": "GTiff", "count": 1, 'dtype': 'float32'}) 
        # meta.update({"driver": "GTiff",'dtype': 'float32'}) 
        # with rs.open(output_path, 'w', **meta) as dst: 
        #     dst.write(output,1)  # rasterio bands are 1-indexed 
        with rs.open('/scratch/Nando/HAC2/data/OtherBuildings/ZMB/ZMB_own_google_bcount.tif', 'w', **meta) as dst: 
            dst.write(counts,1)  # rasterio bands are 1-indexed 

        with rs.open('/scratch/Nando/HAC2/data/OtherBuildings/ZMB/ZMB_own_google_barea_v2.tif', 'w', **meta) as dst: 
            dst.write(area_map,1)  # rasterio bands are 1-indexed 
    return output




def compute_building_count_area(guide_tiff, csv_dir, output_path, mode="count"):
    # Get csv paths
    # .index(long,lat)  
    with rs.open(guide_tiff) as gtiff:
        meta = gtiff.meta

    height = meta["height"]
    width = meta["width"]

    long_start = meta["transform"][2]
    long_pixel = meta["transform"][0]
    long_end = long_start + long_pixel* width
    long_range = np.linspace(long_start,long_end, width-1) + long_pixel/2

    lat_start = meta["transform"][2]
    lat_pixel = meta["transform"][0]
    lat_end = long_start + lat_pixel* height
    lat_range = np.linspace(lat_start,lat_end, height-1) + lat_pixel/2

    buildingsdf = pd.read_csv(csv_dir)
    buildingsdf["long_range_round"] = rounder(long_range)(buildingsdf) 
    buildingsdf["lat_range_round"] = rounder(lat_range)(buildingsdf) 


    counts = np.zeros((guide_tiff["height"], guide_tiff["width"])).astype(np.float32)
    area_map = np.zeros((guide_tiff["height"], guide_tiff["width"])).astype(np.float32)
    confs = np.zeros((guide_tiff["height"], guide_tiff["width"])).astype(np.float32)
    if buildingsdf.shape[0]>0:
        with rs.open(guide_tiff) as gtiff:
            buildingsdf["index_coords"] =  buildingsdf.apply(lambda x: gtiff.index(x["long_range_round"],x["lat_range_round"]), axis=1)

        buildingsdf["x"] =  buildingsdf["index_coords"].apply(lambda x: x[0])
        buildingsdf["y"] =  buildingsdf["index_coords"].apply(lambda x: x[1])

        count_df = buildingsdf.groupby(["x","y"]).size().reset_index(name='building_counts')
        area_df = buildingsdf.groupby(["x","y"])["area_in_meters"].mean().reset_index(name='mean_area')
        conf_df = buildingsdf.groupby(["x","y"])["confidence"].mean().reset_index(name='mean_confidence')
        
        # Fill in the output file
        counts[0,count_df["x"],count_df["y"]] = count_df["building_counts"]
        area_map[1,area_df["x"],area_df["y"]] = area_df["mean_area"]
        confs[2,conf_df["x"],conf_df["y"]] = conf_df["mean_confidence"]

    if mode=="count":
        output = counts
    elif mode=="area":
        output = area_map

    with rs.Env():
        # read profile info from first file 
        meta = guide_tiff["reader"].meta.copy()
        # guide_tiff["reader"].close()

        meta.update({"driver": "GTiff",'dtype': 'float32'})
        # meta.update({"driver": "GTiff", "count": 1, 'dtype': 'float32'})

        with rs.open(output_path, 'w', **meta) as dst: 
            dst.write(output)  # rasterio bands are 1-indexed 

    return output  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rst_wp_boundaries_path", type=str,
                        help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("csv_dir", type=str, help="CSV directory")
    parser.add_argument("output_path", type=str, help="Output path (tif file)")
    args = parser.parse_args()


    
    compute_percentage_of_built_up_area(args.rst_wp_boundaries_path, args.csv_dir, args.output_path)
    # compute_building_count_area(args.rst_wp_boundaries_path, args.csv_dir, args.output_path)


if __name__ == "__main__":
    main()

# python ../../../data/ZMB/OtherBuildings/ZMB_v2_0_count.tif ../../../data/ZMB/OtherBuildings/open_buildings_v1_polygons_wb_10m_ZMB.csv ../../../data/ZMB/OtherBuildings/ZMB_own_google_raster_count.tif
# python compute_gbuilding_count_map.py /scratch/Nando/HAC2/data/OtherBuildings/ZMB/ZMB_buildings_v2_0_count.tif /scratch/Nando/HAC2/data/OtherBuildings/ZMB/ /scratch/Nando/HAC2/data/OtherBuildings/ZMB/ZMB_own_google_raster_area.tif