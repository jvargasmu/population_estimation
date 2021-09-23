# Data root path
root_path = "/home/john.vargas/data/wpop/"

# Input files

input_paths = {
    "tza": {
        "buildings": "{}OtherBuildings/TZA/tza_gbuildings.tif".format(root_path),
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