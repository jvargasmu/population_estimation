import argparse
import csv
from osgeo import gdal
from osgeo import ogr
from osgeo import osr


def convert_csv_to_shapefile(csv_path, output_path):
    spatialref = osr.SpatialReference()  # Set the spatial ref.
    spatialref.SetWellKnownGeogCS('WGS84')  # WGS84 aka ESPG:4326
    driver = ogr.GetDriverByName("ESRI Shapefile") # Shapefile driver
    dstfile = driver.CreateDataSource(output_path) # Your output file

    dstlayer = dstfile.CreateLayer("layer", spatialref, geom_type=ogr.wkbPolygon)

    # Add the other attribute fields needed with the following schema :
    fielddef1 = ogr.FieldDefn("latitude", ogr.OFTReal)
    #fielddef1.SetWidth(16)
    dstlayer.CreateField(fielddef1)

    fielddef2 = ogr.FieldDefn("longitude", ogr.OFTReal)
    #fielddef2.SetWidth(16)
    dstlayer.CreateField(fielddef2)

    fielddef3 = ogr.FieldDefn("area_in_m", ogr.OFTReal)
    #fielddef3.SetWidth(16)
    dstlayer.CreateField(fielddef3)

    fielddef4 = ogr.FieldDefn("confidence", ogr.OFTReal)
    #fielddef4.SetWidth(16)
    dstlayer.CreateField(fielddef4)

    fielddef5 = ogr.FieldDefn("full_code", ogr.OFTString)
    fielddef5.SetWidth(80)
    dstlayer.CreateField(fielddef5)

    fielddef6 = ogr.FieldDefn("map_val", ogr.OFTReal)
    # fielddef6.SetWidth(16)
    dstlayer.CreateField(fielddef6)

    print("FieldDef DONE")

    # Read the features in your csv file:
    with open(csv_path) as file_input:
        reader = csv.reader(file_input)
        next(reader)  # Skip the header
        for nb, row in enumerate(reader):
            poly = ogr.CreateGeometryFromWkt(row[4])
            feature = ogr.Feature(dstlayer.GetLayerDefn())
            feature.SetGeometry(poly)
            feature.SetField("latitude", float(row[0]))
            feature.SetField("longitude", float(row[1]))
            feature.SetField("area_in_m", float(row[2]))
            feature.SetField("confidence", float(row[3]))
            feature.SetField("full_code", row[5])
            feature.SetField("map_val", 255.0)
            dstlayer.CreateFeature(feature)
        feature.Destroy()
        dstfile.Destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="CSV file path")
    parser.add_argument("output_path", type=str, help="Output shapefile path")
    args = parser.parse_args()

    convert_csv_to_shapefile(args.csv_path, args.output_path)


if __name__ == "__main__":
    main()