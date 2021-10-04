
import os


# Data root path
# if os.isdir()
root_paths = ["/home/john.vargas/data/wpop/",
    "/scratch/Nando/HAC2/data/"]
for dir in root_paths:
    if os.path.isdir(dir):
        root_path = dir

# Input files

input_paths = {
    "tza": {
        "buildings": "{}OtherBuildings/TZA/tza_gbuildings.tif".format(root_path),
        # "buildings_g": "/home/pf/pfstaff/projects/Daudt_HAC/ftp.worldpop.org.uk/GIS/Covariates/Building_patterns/Google_Open_Buildings/v1_0/country/TZA/TZA_g_bldg_patterns_100m_v1/TZA_gbp_BCB_v1_count.tif".format(root_path),
        "esaccilc_dst011_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst011_100m_2015.tif".format(
            root_path),
        "esaccilc_dst040_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst040_100m_2015.tif".format(
            root_path),
        "esaccilc_dst130_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst130_100m_2015.tif".format(
            root_path),
        "esaccilc_dst140_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst140_100m_2015.tif".format(
            root_path),
        "esaccilc_dst150_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst150_100m_2015.tif".format(
            root_path),
        "esaccilc_dst160_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst160_100m_2015.tif".format(
            root_path),
        "esaccilc_dst190_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst190_100m_2015.tif".format(
            root_path),
        "esaccilc_dst200_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst190_100m_2015.tif".format(
            root_path),
        "osm_dst_roadintersec_100m_2016": "{}Covariates/TZA/OSM/DST/tza_osm_dst_roadintersec_100m_2016.tif".format(
            root_path),
        "osm_dst_waterway_100m_2016": "{}Covariates/TZA/OSM/DST/tza_osm_dst_waterway_100m_2016.tif".format(
            root_path),
        "osm_dst_road_100m_2016": "{}Covariates/TZA/OSM/DST/tza_osm_dst_road_100m_2016.tif".format(
            root_path)

    }
}

metadata = {
    "tza": {
        "wp_no_data": [0, 1],
        "hd_no_data": [0]
    }
}

# Columns of shapefiles
col_coarse_level_name = "ADM2_EN"
col_coarse_level_code = "ADM2_PCODE"
col_finest_level_name = "ADM3_EN"
col_finest_level_code = "ADM3_PCODE"
col_coarse_level_seq_id = "GR_SID"
col_finest_level_seq_id = "SID"
