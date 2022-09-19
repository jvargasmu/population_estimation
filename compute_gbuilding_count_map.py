import argparse
from osgeo import gdal
import csv
import numpy as np
import math
import os
from tqdm import tqdm


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


def compute_percentage_of_built_up_area(regions_path, csv_path, output_bcount, output_meanarea):
    # Get csv paths
    csv_paths = [csv_path]
    # Read image
    img_obj = gdal.Open(regions_path)
    # Get image metadata
    xmin, xpixel, _, ymax, _, ypixel = img_obj.GetGeoTransform()
    width, height = img_obj.RasterXSize, img_obj.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel
    print("ymin {} ymax {} xmin {}, xmax {}, width {}, height {}, xpixel {} ypixel {}".format(ymin, ymax, xmin, xmax,
                                                                                              width, height, xpixel,
                                                                                              ypixel))
    
    cnt = np.zeros((height, width)).astype(np.float32)
    bcounts = np.zeros((height, width)).astype(np.float32)
    mean_area = np.zeros((height, width)).astype(np.float32)


    for csv_path in csv_paths:
        with open(csv_path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in tqdm(enumerate(csv_reader)):
                if i > 0:
                    latitude = float(row[0])
                    longitude = float(row[1])
                    area = float(row[2])
                    confidence = float(row[3])

                    pix_y = int(math.floor((latitude - ymax) / -abs(ypixel)))
                    pix_x = int(math.floor((longitude - xmin) / xpixel))

                    if (pix_y >= 0 and pix_y < height) and (pix_x >= 0 and pix_x < width): 
                        all_area = mean_area[pix_y, pix_x]*bcounts[pix_y, pix_x] 
                        bcounts[pix_y, pix_x] += 1 
                        mean_area[pix_y, pix_x] = (all_area+area)/bcounts[pix_y, pix_x]

    write_geoloc_image(bcounts, output_bcount, regions_path)
    write_geoloc_image(mean_area, output_meanarea, regions_path)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rst_wp_boundaries_path", type=str,
                        help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("csv_dir", type=str, help="CSV directory")
    parser.add_argument("output_bcount", type=str, help="Output path (tif file)")
    parser.add_argument("output_meanarea", type=str, help="Output path (tif file)")
    args = parser.parse_args()

    compute_percentage_of_built_up_area(args.rst_wp_boundaries_path, args.csv_dir, args.output_bcount, args.output_meanarea)


if __name__ == "__main__":
    main()
