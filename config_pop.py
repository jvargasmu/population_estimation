
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
        #"buildings": "{}OtherBuildings/TZA/tza_gbuildings.tif".format(root_path),
        "buildings": "{}OtherBuildings/TZA/TZA_gbp_BCB_v1_count.tif".format(root_path),
        "esaccilc_dst011_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst011_100m_2015.tif".format(
            root_path),
        "esaccilc_dst040_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst040_100m_2015.tif".format(
            root_path),
        "esaccilc_dst130_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst130_100m_2015.tif".format(
            root_path),
        # "esaccilc_dst140_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst140_100m_2015.tif".format(
        #     root_path),
        # "esaccilc_dst150_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst150_100m_2015.tif".format(
        #     root_path),
        # "esaccilc_dst160_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst160_100m_2015.tif".format(
        #     root_path),
        # "esaccilc_dst190_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst190_100m_2015.tif".format(
        #     root_path),
        # "esaccilc_dst200_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst200_100m_2015.tif".format(
        #     root_path), 
        # "tza_tt50k_100m_2000": "{}Covariates/TZA/Accessibility/tza_tt50k_100m_2000.tif".format(
        #     root_path), 
        # "tza_dst_bsgme_100m_2015": "{}Covariates/TZA/BSGM/2015/DTE/tza_dst_bsgme_100m_2015.tif".format(
        #     root_path),
        # "tza_dst_ghslesaccilcgufghsll_100m_2014": "{}Covariates/TZA/BuiltSettlement/2014/DTE/tza_dst_ghslesaccilcgufghsll_100m_2014.tif".format(
        #     root_path),
        # "tza_dst_coastline_100m_2000_2020": "{}Covariates/TZA/Coastline/DST/tza_dst_coastline_100m_2000_2020.tif".format(
        #     root_path),
        # "tza_dmsp_100m_2011": "{}Covariates/TZA/DMSP/tza_dmsp_100m_2011.tif".format(
        #     root_path),
        # "tza_esaccilc_dst_water_100m_2000_2012": "{}Covariates/TZA/ESA_CCI_Water/DST/tza_esaccilc_dst_water_100m_2000_2012.tif".format(
        #     root_path),
        # "tza_osm_dst_roadintersec_100m_2016": "{}Covariates/TZA/OSM/DST/tza_osm_dst_roadintersec_100m_2016.tif".format(
        #     root_path),
        # "tza_osm_dst_waterway_100m_2016": "{}Covariates/TZA/OSM/DST/tza_osm_dst_waterway_100m_2016.tif".format(
        #     root_path),
        # "tza_osm_dst_road_100m_2016": "{}Covariates/TZA/OSM/DST/tza_osm_dst_road_100m_2016.tif".format(
        #     root_path),
        # "tza_srtm_slope_100m": "{}Covariates/TZA/Slope/tza_srtm_slope_100m.tif".format(
        #     root_path),
        # "tza_srtm_topo_100m": "{}Covariates/TZA/Topo/tza_srtm_topo_100m.tif".format(
        #     root_path),
        # "tza_viirs_100m_2015": "{}Covariates/TZA/VIIRS/tza_viirs_100m_2015.tif".format(
        #     root_path),
        # "tza_wdpa_dst_cat1_100m_2015": "{}Covariates/TZA/WDPA/WDPA_1/tza_wdpa_dst_cat1_100m_2015.tif".format(
        #     root_path),

    }
}


no_data_values = {
    "tza": {
        "buildings": None,
        "esaccilc_dst011_100m_2000": -99999,
        "esaccilc_dst040_100m_2000": -99999,
        "esaccilc_dst130_100m_2000": -99999,
        "esaccilc_dst140_100m_2000": -99999,
        "esaccilc_dst150_100m_2000": -99999,
        "esaccilc_dst160_100m_2000": -99999,
        "esaccilc_dst190_100m_2000": -99999,
        "esaccilc_dst200_100m_2000": -99999,
        "tza_tt50k_100m_2000": -99999,
        "tza_dst_bsgme_100m_2015": -99999,
        "tza_dst_ghslesaccilcgufghsll_100m_2014": -99999,
        "tza_dst_coastline_100m_2000_2020": -99999,
        "tza_dmsp_100m_2011": -99999,
        "tza_esaccilc_dst_water_100m_2000_2012":-99999, 
        "tza_osm_dst_roadintersec_100m_2016":-99999, 
        "tza_osm_dst_waterway_100m_2016": -99999,
        "tza_osm_dst_road_100m_2016": -99999,
        "tza_srtm_slope_100m": -99999,
        "tza_srtm_topo_100m": -99999,
        "tza_viirs_100m_2015": -99999,
        "tza_wdpa_dst_cat1_100m_2015": -99999,
    }
}




metadata = {
    "tza": {
        "wp_no_data": [0, 1],
        "wp_covariates_no_data": -9999,
        "hd_no_data": [0],

    }
}

input_paths["tza_f2"] = {
    "buildings": input_paths["tza"]["buildings"],
    "tza_viirs_100m_2016" : "{}Covariates/TZA/VIIRS/tza_viirs_100m_2016.tif".format(root_path)
}

metadata["tza_f2"] = metadata["tza"]

# Columns of shapefiles
col_coarse_level_name = "ADM2_EN"
col_coarse_level_code = "ADM2_PCODE"
col_finest_level_name = "ADM3_EN"
col_finest_level_code = "ADM3_PCODE"
col_coarse_level_seq_id = "GR_SID"
col_finest_level_seq_id = "SID"
