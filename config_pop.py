import os

# Define root path
root_paths = ["/home/john.vargas/data/pomelo_input_data/",
    "/scratch/Nando/HAC2/pomelo_input_data/",
    "/cluster/work/igp_psr/metzgern/HAC/pomelo_input_data/",
    "/scratch2/metzgern/HAC/pomelo_input_data/",
    "/scratch2/metzgern/HAC/data/pomelo_input_data/"]

for dir in root_paths:
    if os.path.isdir(dir):
        root_path = dir

# Input file definition
input_paths = {
    "tza": {
        "buildings_google": f"{root_path}TZA/covariates/TZA_gbp_BCB_v1_count.tif",
        "buildings_maxar": f"{root_path}TZA/covariates/TZA_mbp_BCB_v3_count.tif",
        "buildings_google_mean_area": f"{root_path}TZA/covariates/TZA_gbp_BCB_v1_mean_area.tif",
        "buildings_maxar_mean_area": f"{root_path}TZA/covariates/TZA_mbp_BCB_v3_mean_area.tif",
        "tza_tt50k_100m_2000": f"{root_path}TZA/covariates/tza_tt50k_100m_2000.tif",
        "tza_dst_bsgme_100m_2015": f"{root_path}TZA/covariates/tza_dst_bsgme_100m_2015.tif",
        "tza_dst_ghslesaccilcgufghsll_100m_2014": f"{root_path}TZA/covariates/tza_dst_ghslesaccilcgufghsll_100m_2014.tif",
        "tza_dst_coastline_100m_2000_2020": f"{root_path}TZA/covariates/tza_dst_coastline_100m_2000_2020.tif",
        "tza_dmsp_100m_2011": f"{root_path}TZA/covariates/tza_dmsp_100m_2011.tif",
        "tza_esaccilc_dst_water_100m_2000_2012": f"{root_path}TZA/covariates/tza_esaccilc_dst_water_100m_2000_2012.tif",
        "tza_osm_dst_roadintersec_100m_2016": f"{root_path}TZA/covariates/tza_osm_dst_roadintersec_100m_2016.tif",
        "tza_osm_dst_waterway_100m_2016": f"{root_path}TZA/covariates/tza_osm_dst_waterway_100m_2016.tif",
        "tza_osm_dst_road_100m_2016": f"{root_path}TZA/covariates/tza_osm_dst_road_100m_2016.tif",
        "tza_srtm_slope_100m": f"{root_path}TZA/covariates/tza_srtm_slope_100m.tif",
        "tza_srtm_topo_100m": f"{root_path}TZA/covariates/tza_srtm_topo_100m.tif",
        "tza_viirs_100m_2015": f"{root_path}TZA/covariates/tza_viirs_100m_2015.tif",
        "tza_wdpa_dst_cat1_100m_2015": f"{root_path}TZA/covariates/tza_wdpa_dst_cat1_100m_2015.tif"
    },
    "uga": {
        "buildings_google": f"{root_path}UGA/covariates/UGA_gbp_BCB_v1_count.tif",
        "buildings_maxar": f"{root_path}UGA/covariates/UGA_mbp_BCB_v3_count.tif",
        "buildings_google_mean_area": f"{root_path}UGA/covariates/UGA_gbp_BCB_v1_mean_area.tif",
        "buildings_maxar_mean_area": f"{root_path}UGA/covariates/UGA_mbp_BCB_v3_mean_area.tif",
        "uga_tt50k_100m_2000": f"{root_path}UGA/covariates/uga_tt50k_100m_2000.tif",
        "uga_dst_bsgme_100m_2015": f"{root_path}UGA/covariates/uga_dst_bsgme_100m_2015.tif",
        "uga_dst_ghslesaccilcgufghsll_100m_2014": f"{root_path}UGA/covariates/uga_dst_ghslesaccilcgufghsll_100m_2014.tif",
        "uga_dst_coastline_100m_2000_2020": f"{root_path}UGA/covariates/uga_dst_coastline_100m_2000_2020.tif",
        "uga_dmsp_100m_2011": f"{root_path}UGA/covariates/uga_dmsp_100m_2011.tif",
        "uga_esaccilc_dst_water_100m_2000_2012": f"{root_path}UGA/covariates/uga_esaccilc_dst_water_100m_2000_2012.tif",
        "uga_osm_dst_roadintersec_100m_2016": f"{root_path}UGA/covariates/uga_osm_dst_roadintersec_100m_2016.tif",
        "uga_osm_dst_waterway_100m_2016": f"{root_path}UGA/covariates/uga_osm_dst_waterway_100m_2016.tif",
        "uga_osm_dst_road_100m_2016": f"{root_path}UGA/covariates/uga_osm_dst_road_100m_2016.tif",
        "uga_srtm_slope_100m": f"{root_path}UGA/covariates/uga_srtm_slope_100m.tif",
        "uga_srtm_topo_100m": f"{root_path}UGA/covariates/uga_srtm_topo_100m.tif",
        "uga_viirs_100m_2015": f"{root_path}UGA/covariates/uga_viirs_100m_2015.tif",
        "uga_wdpa_dst_cat1_100m_2015": f"{root_path}UGA/covariates/uga_wdpa_dst_cat1_100m_2015.tif",
    },
    "cod": {
        "buildings_google": f"{root_path}COD/covariates/COD_gbp_BCB_v1_count.tif",
        "buildings_maxar": f"{root_path}COD/covariates/COD_mbp_BCB_v3_count.tif",
        "buildings_google_mean_area": f"{root_path}COD/covariates/COD_gbp_BCB_v1_mean_area.tif",
        "buildings_maxar_mean_area": f"{root_path}COD/covariates/COD_mbp_BCB_v3_mean_area.tif",
        "cod_tt50k_100m_2000": f"{root_path}COD/covariates/cod_tt50k_100m_2000.tif",
        "cod_dst_bsgme_100m_2015": f"{root_path}COD/covariates/cod_dst_bsgme_100m_2015.tif",
        "cod_dst_ghslesaccilcgufghsll_100m_2014": f"{root_path}COD/covariates/cod_dst_ghslesaccilcgufghsll_100m_2014.tif",
        "cod_dst_coastline_100m_2000_2020": f"{root_path}COD/covariates/cod_dst_coastline_100m_2000_2020.tif",
        "cod_dmsp_100m_2011": f"{root_path}COD/covariates/cod_dmsp_100m_2011.tif",
        "cod_esaccilc_dst_water_100m_2000_2012": f"{root_path}COD/covariates/cod_esaccilc_dst_water_100m_2000_2012.tif",
        "cod_osm_dst_roadintersec_100m_2016": f"{root_path}COD/covariates/cod_osm_dst_roadintersec_100m_2016.tif",
        "cod_osm_dst_waterway_100m_2016": f"{root_path}COD/covariates/cod_osm_dst_waterway_100m_2016.tif",
        "cod_osm_dst_road_100m_2016": f"{root_path}COD/covariates/cod_osm_dst_road_100m_2016.tif",
        "cod_srtm_slope_100m": f"{root_path}COD/covariates/cod_srtm_slope_100m.tif",
        "cod_srtm_topo_100m": f"{root_path}COD/covariates/cod_srtm_topo_100m.tif",
        "cod_viirs_100m_2015": f"{root_path}COD/covariates/cod_viirs_100m_2015.tif",
        "cod_wdpa_dst_cat1_100m_2015": f"{root_path}COD/covariates/cod_wdpa_dst_cat1_100m_2015.tif"
    },
    "rwa": {
        "buildings_google": f"{root_path}RWA/covariates/RWA_gbp_BCB_v1_count.tif",
        "buildings_maxar": f"{root_path}RWA/covariates/RWA_mbp_BCB_v3_count.tif",
        "buildings_google_mean_area": f"{root_path}RWA/covariates/RWA_gbp_BCB_v1_mean_area.tif",
        "buildings_maxar_mean_area": f"{root_path}RWA/covariates/RWA_mbp_BCB_v3_mean_area.tif",
        "rwa_tt50k_100m_2000": f"{root_path}RWA/covariates/rwa_tt50k_100m_2000.tif",
        "rwa_dst_bsgme_100m_2015": f"{root_path}RWA/covariates/rwa_dst_bsgme_100m_2015.tif",
        "rwa_dst_ghslesaccilcgufghsll_100m_2014": f"{root_path}RWA/covariates/rwa_dst_ghslesaccilcgufghsll_100m_2014.tif",
        "rwa_dst_coastline_100m_2000_2020": f"{root_path}RWA/covariates/rwa_dst_coastline_100m_2000_2020.tif",
        "rwa_dmsp_100m_2011": f"{root_path}RWA/covariates/rwa_dmsp_100m_2011.tif",
        "rwa_esaccilc_dst_water_100m_2000_2012": f"{root_path}RWA/covariates/rwa_esaccilc_dst_water_100m_2000_2012.tif",
        "rwa_osm_dst_roadintersec_100m_2016": f"{root_path}RWA/covariates/rwa_osm_dst_roadintersec_100m_2016.tif",
        "rwa_osm_dst_waterway_100m_2016": f"{root_path}RWA/covariates/rwa_osm_dst_waterway_100m_2016.tif",
        "rwa_osm_dst_road_100m_2016": f"{root_path}RWA/covariates/rwa_osm_dst_road_100m_2016.tif",
        "rwa_srtm_slope_100m": f"{root_path}RWA/covariates/rwa_srtm_slope_100m.tif",
        "rwa_srtm_topo_100m": f"{root_path}RWA/covariates/rwa_srtm_topo_100m.tif",
        "rwa_viirs_100m_2015": f"{root_path}RWA/covariates/rwa_viirs_100m_2015.tif",
        "rwa_wdpa_dst_cat1_100m_2015": f"{root_path}RWA/covariates/rwa_wdpa_dst_cat1_100m_2015.tif"
    },
    "moz": {
        "buildings_google": f"{root_path}MOZ/covariates/MOZ_gbp_BCB_v1_count.tif",
        "buildings_maxar": f"{root_path}MOZ/covariates/MOZ_mbp_BCB_v3_count.tif",
        "buildings_google_mean_area": f"{root_path}MOZ/covariates/MOZ_gbp_BCB_v1_mean_area.tif",
        "buildings_maxar_mean_area": f"{root_path}MOZ/covariates/MOZ_mbp_BCB_v3_mean_area.tif",
        "moz_tt50k_100m_2000": f"{root_path}MOZ/covariates/moz_tt50k_100m_2000.tif",
        "moz_dst_bsgme_100m_2015": f"{root_path}MOZ/covariates/moz_dst_bsgme_100m_2015.tif",
        "moz_dst_ghslesaccilcgufghsll_100m_2014": f"{root_path}MOZ/covariates/moz_dst_ghslesaccilcgufghsll_100m_2014.tif",
        "moz_dst_coastline_100m_2000_2020": f"{root_path}MOZ/covariates/moz_dst_coastline_100m_2000_2020.tif",
        "moz_dmsp_100m_2011": f"{root_path}MOZ/covariates/moz_dmsp_100m_2011.tif",
        "moz_esaccilc_dst_water_100m_2000_2012": f"{root_path}MOZ/covariates/moz_esaccilc_dst_water_100m_2000_2012.tif",
        "moz_osm_dst_roadintersec_100m_2016": f"{root_path}MOZ/covariates/moz_osm_dst_roadintersec_100m_2016.tif",
        "moz_osm_dst_waterway_100m_2016": f"{root_path}MOZ/covariates/moz_osm_dst_waterway_100m_2016.tif",
        "moz_osm_dst_road_100m_2016": f"{root_path}MOZ/covariates/moz_osm_dst_road_100m_2016.tif",
        "moz_srtm_slope_100m": f"{root_path}MOZ/covariates/moz_srtm_slope_100m.tif",
        "moz_srtm_topo_100m": f"{root_path}MOZ/covariates/moz_srtm_topo_100m.tif",
        "moz_viirs_100m_2015": f"{root_path}MOZ/covariates/moz_viirs_100m_2015.tif",
        "moz_wdpa_dst_cat1_100m_2015": f"{root_path}MOZ/covariates/moz_wdpa_dst_cat1_100m_2015.tif"
    },
    "zmb" : {
        "buildings_google": f"{root_path}ZMB/covariates/ZMB_gbp_BCB_v1_count.tif",
        "buildings_maxar": f"{root_path}ZMB/covariates/ZMB_buildings_v2_0_count.tif",
        "buildings_google_mean_area": f"{root_path}ZMB/covariates/ZMB_gbp_BCB_v1_mean_area.tif",
        "buildings_maxar_mean_area": f"{root_path}ZMB/covariates/ZMB_buildings_v2_0_mean_area.tif",
        "zmb_tt50k_100m_2000": f"{root_path}ZMB/covariates/zmb_tt50k_100m_2000.tif",
        "zmb_dst_bsgme_100m_2015": f"{root_path}ZMB/covariates/zmb_dst_bsgme_100m_2015.tif",
        "zmb_dst_ghslesaccilcgufghsll_100m_2014": f"{root_path}ZMB/covariates/zmb_dst_ghslesaccilcgufghsll_100m_2014.tif",
        "zmb_dst_coastline_100m_2000_2020": f"{root_path}ZMB/covariates/zmb_dst_coastline_100m_2000_2020.tif",
        "zmb_dmsp_100m_2011": f"{root_path}ZMB/covariates/zmb_dmsp_100m_2011.tif",
        "zmb_esaccilc_dst_water_100m_2000_2012": f"{root_path}ZMB/covariates/zmb_esaccilc_dst_water_100m_2000_2012.tif",
        "zmb_osm_dst_roadintersec_100m_2016": f"{root_path}ZMB/covariates/zmb_osm_dst_roadintersec_100m_2016.tif",
        "zmb_osm_dst_waterway_100m_2016": f"{root_path}ZMB/covariates/zmb_osm_dst_waterway_100m_2016.tif",
        "zmb_osm_dst_road_100m_2016": f"{root_path}ZMB/covariates/zmb_osm_dst_road_100m_2016.tif",
        "zmb_srtm_slope_100m": f"{root_path}ZMB/covariates/zmb_srtm_slope_100m.tif",
        "zmb_srtm_topo_100m": f"{root_path}ZMB/covariates/zmb_srtm_topo_100m.tif",
        "zmb_viirs_100m_2015": f"{root_path}ZMB/covariates/zmb_viirs_100m_2015.tif",
        "zmb_wdpa_dst_cat1_100m_2015": f"{root_path}ZMB/covariates/zmb_wdpa_dst_cat1_100m_2015.tif"
    }
}


no_data_values = {
    "tza": {
        "buildings_j": None,
        "buildings_google": -99999,
        "buildings_maxar": -99999, 
        "buildings_google_mean_area": -99999,
        "buildings_merge_mean_area": -99999,
        "buildings_maxar_mean_area": -99999,
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
        "tza_dmsp_100m_2011": 32767.,
        "tza_esaccilc_dst_water_100m_2000_2012":-99999, 
        "tza_osm_dst_roadintersec_100m_2016":-99999, 
        "tza_osm_dst_waterway_100m_2016": -99999,
        "tza_osm_dst_road_100m_2016": -99999,
        "tza_srtm_slope_100m": 255,
        "tza_srtm_topo_100m": 32767.,
        "tza_viirs_100m_2015": 3.4028e+38,
        "tza_wdpa_dst_cat1_100m_2015": -99999,
    },
    "uga":{
        "buildings_j": None,
        "buildings_google": -99999,
        "buildings_maxar": -99999, 
        "buildings_google_mean_area": -99999,
        "buildings_merge_mean_area": -99999,
        "buildings_maxar_mean_area": -99999,
        "esaccilc_dst011_100m_2000": -99999,
        "esaccilc_dst040_100m_2000": -99999,
        "esaccilc_dst130_100m_2000": -99999,
        "esaccilc_dst140_100m_2000": -99999,
        "esaccilc_dst150_100m_2000": -99999,
        "esaccilc_dst160_100m_2000": -99999,
        "esaccilc_dst190_100m_2000": -99999,
        "esaccilc_dst200_100m_2000": -99999,
        "uga_tt50k_100m_2000": -99999,
        "uga_dst_bsgme_100m_2015": -99999,
        "uga_dst_ghslesaccilcgufghsll_100m_2014": -99999,
        "uga_dst_coastline_100m_2000_2020": -99999,
        "uga_dmsp_100m_2011": 32767.,
        "uga_esaccilc_dst_water_100m_2000_2012":-99999, 
        "uga_osm_dst_roadintersec_100m_2016":-99999, 
        "uga_osm_dst_waterway_100m_2016": -99999,
        "uga_osm_dst_road_100m_2016": -99999,
        "uga_srtm_slope_100m": 255.,
        "uga_srtm_topo_100m": 32767.,
        "uga_viirs_100m_2015": 3.4028e+38,
        "uga_wdpa_dst_cat1_100m_2015": -99999,
    },
    "cod":{
        "buildings_j": None,
        "buildings_google": -99999,
        "buildings_maxar": -99999, 
        "buildings_google_mean_area": -99999,
        "buildings_merge_mean_area": -99999,
        "buildings_maxar_mean_area": -99999,
        "esaccilc_dst011_100m_2000": -99999,
        "esaccilc_dst040_100m_2000": -99999,
        "esaccilc_dst130_100m_2000": -99999,
        "esaccilc_dst140_100m_2000": -99999,
        "esaccilc_dst150_100m_2000": -99999,
        "esaccilc_dst160_100m_2000": -99999,
        "esaccilc_dst190_100m_2000": -99999,
        "esaccilc_dst200_100m_2000": -99999,
        "cod_tt50k_100m_2000": -99999,
        "cod_dst_bsgme_100m_2015": -99999,
        "cod_dst_ghslesaccilcgufghsll_100m_2014": -99999,
        "cod_dst_coastline_100m_2000_2020": -99999,
        "cod_dmsp_100m_2011": 32767.,
        "cod_esaccilc_dst_water_100m_2000_2012":-99999, 
        "cod_osm_dst_roadintersec_100m_2016":-99999, 
        "cod_osm_dst_waterway_100m_2016": -99999,
        "cod_osm_dst_road_100m_2016": -99999,
        "cod_srtm_slope_100m": 255.,
        "cod_srtm_topo_100m": 32767.,
        "cod_viirs_100m_2015": 3.4028e+38,
        "cod_wdpa_dst_cat1_100m_2015": -99999,
    },
    "rwa":{
        "buildings_j": None,
        "buildings_google": -99999,
        "buildings_maxar": -99999, 
        "buildings_google_mean_area": -99999,
        "buildings_merge_mean_area": -99999,
        "buildings_maxar_mean_area": -99999,
        "esaccilc_dst011_100m_2000": -99999,
        "esaccilc_dst040_100m_2000": -99999,
        "esaccilc_dst130_100m_2000": -99999,
        "esaccilc_dst140_100m_2000": -99999,
        "esaccilc_dst150_100m_2000": -99999,
        "esaccilc_dst160_100m_2000": -99999,
        "esaccilc_dst190_100m_2000": -99999,
        "esaccilc_dst200_100m_2000": -99999,
        "rwa_tt50k_100m_2000": -99999,
        "rwa_dst_bsgme_100m_2015": -99999,
        "rwa_dst_ghslesaccilcgufghsll_100m_2014": -99999,
        "rwa_dst_coastline_100m_2000_2020": -99999,
        "rwa_dmsp_100m_2011": 32767.,
        "rwa_esaccilc_dst_water_100m_2000_2012":-99999, 
        "rwa_osm_dst_roadintersec_100m_2016":-99999, 
        "rwa_osm_dst_waterway_100m_2016": -99999,
        "rwa_osm_dst_road_100m_2016": -99999,
        "rwa_srtm_slope_100m": 255.,
        "rwa_srtm_topo_100m": 32767.,
        "rwa_viirs_100m_2015": 3.4028e+38,
        "rwa_wdpa_dst_cat1_100m_2015": -99999,
    },
    "moz":{
        "buildings_j": None,
        "buildings_google": -99999,
        "buildings_maxar": -99999, 
        "buildings_google_mean_area": -99999,
        "buildings_merge_mean_area": -99999,
        "buildings_maxar_mean_area": -99999,
        "esaccilc_dst011_100m_2000": -99999,
        "esaccilc_dst040_100m_2000": -99999,
        "esaccilc_dst130_100m_2000": -99999,
        "esaccilc_dst140_100m_2000": -99999,
        "esaccilc_dst150_100m_2000": -99999,
        "esaccilc_dst160_100m_2000": -99999,
        "esaccilc_dst190_100m_2000": -99999,
        "esaccilc_dst200_100m_2000": -99999,
        "moz_tt50k_100m_2000": -99999,
        "moz_dst_bsgme_100m_2015": -99999,
        "moz_dst_ghslesaccilcgufghsll_100m_2014": -99999,
        "moz_dst_coastline_100m_2000_2020": -99999,
        "moz_dmsp_100m_2011": 32767.,
        "moz_esaccilc_dst_water_100m_2000_2012":-99999, 
        "moz_osm_dst_roadintersec_100m_2016":-99999, 
        "moz_osm_dst_waterway_100m_2016": -99999,
        "moz_osm_dst_road_100m_2016": -99999,
        "moz_srtm_slope_100m": 255.,
        "moz_srtm_topo_100m": 32767.,
        "moz_viirs_100m_2015": 3.4028e+38,
        "moz_wdpa_dst_cat1_100m_2015": -99999,
    },
    "zmb":{
        "buildings_j": None,
        "buildings_google": -99999,
        "buildings_maxar": -99999, 
        "buildings_google_mean_area": -99999,
        "buildings_merge_mean_area": -99999,
        "buildings_maxar_mean_area": -99999,
        "esaccilc_dst011_100m_2000": -99999,
        "esaccilc_dst040_100m_2000": -99999,
        "esaccilc_dst130_100m_2000": -99999,
        "esaccilc_dst140_100m_2000": -99999,
        "esaccilc_dst150_100m_2000": -99999,
        "esaccilc_dst160_100m_2000": -99999,
        "esaccilc_dst190_100m_2000": -99999,
        "esaccilc_dst200_100m_2000": -99999,
        "zmb_tt50k_100m_2000": -99999,
        "zmb_dst_bsgme_100m_2015": -99999,
        "zmb_dst_ghslesaccilcgufghsll_100m_2014": -99999,
        "zmb_dst_coastline_100m_2000_2020": -99999,
        "zmb_dmsp_100m_2011": 32767.,
        "zmb_esaccilc_dst_water_100m_2000_2012":-99999, 
        "zmb_osm_dst_roadintersec_100m_2016":-99999, 
        "zmb_osm_dst_waterway_100m_2016": -99999,
        "zmb_osm_dst_road_100m_2016": -99999,
        "zmb_srtm_slope_100m": 255.,
        "zmb_srtm_topo_100m": 32767.,
        "zmb_viirs_100m_2015": 3.4028235e+38,
        "zmb_wdpa_dst_cat1_100m_2015": -99999,
    }
}

norms = {
    "tza": {
        "buildings_j": (0.00089380914, 8.41622997e-03),
        "esaccilc_dst011_100m_2000": (2.81727052, 5.69885715e+00),
        "esaccilc_dst040_100m_2000": (0.44520899, 2.72345595e+00),
        "esaccilc_dst130_100m_2000": (3.09648584, 4.70480562e+00),
        "esaccilc_dst140_100m_2000": (9.18009012, 9.85124076e+00),
        "esaccilc_dst150_100m_2000": (108.52330299, 8.17261502e+01),
        "esaccilc_dst160_100m_2000": (8.65616757, 9.26884634e+00),
        "esaccilc_dst190_100m_2000": (37.38046272, 3.10730075e+01),
        "esaccilc_dst200_100m_2000": (64.73759992, 4.46013049e+01),
        "tza_tt50k_100m_2000": (209.1351, 188.1936),
        "tza_dst_bsgme_100m_2015": (3.2670, 4.2283),
        "tza_dst_ghslesaccilcgufghsll_100m_2014": (3.3381, 4.2542),
        "tza_dst_coastline_100m_2000_2020": (698.8104,  326.6751),
        "tza_dmsp_100m_2011": (71.8836 ,  399.8602 ),
        "tza_esaccilc_dst_water_100m_2000_2012":(18.1815, 14.8920), 
        "tza_osm_dst_roadintersec_100m_2016":(19.8096, 29.5883), 
        "tza_osm_dst_waterway_100m_2016": (15.2655, 15.4519), 
        "tza_osm_dst_road_100m_2016": (3.7082,  5.6608), 
        "tza_srtm_slope_100m": (3.2089 , 3.8157),
        "tza_srtm_topo_100m": (1143.9080 , 402.7680),
        "tza_viirs_100m_2015": (0.2205 , 1.0825),
        "tza_wdpa_dst_cat1_100m_2015": (388.8701 , 221.6834),
        'buildings_merge_mean_area': (26.3673, 48.1988)
    },
    "uga":{
        "buildings_j": (0.00089380914, 8.41622997e-03),
        "esaccilc_dst011_100m_2000": (2.81727052, 5.69885715e+00),
        "esaccilc_dst040_100m_2000": (0.44520899, 2.72345595e+00),
        "esaccilc_dst130_100m_2000": (3.09648584, 4.70480562e+00),
        "esaccilc_dst140_100m_2000": (9.18009012, 9.85124076e+00),
        "esaccilc_dst150_100m_2000": (108.52330299, 8.17261502e+01),
        "esaccilc_dst160_100m_2000": (8.65616757, 9.26884634e+00),
        "esaccilc_dst190_100m_2000": (37.38046272, 3.10730075e+01),
        "esaccilc_dst200_100m_2000": (64.73759992, 4.46013049e+01),
        "uga_tt50k_100m_2000": (209.1351, 188.1936),
        "uga_dst_bsgme_100m_2015": (3.2670, 4.2283),
        "uga_dst_ghslesaccilcgufghsll_100m_2014": (3.3381, 4.2542),
        "uga_dst_coastline_100m_2000_2020": (698.8104,  326.6751),
        "uga_dmsp_100m_2011": (71.8836 ,  399.8602 ),
        "uga_esaccilc_dst_water_100m_2000_2012":(18.1815, 14.8920), 
        "uga_osm_dst_roadintersec_100m_2016":(19.8096, 29.5883), 
        "uga_osm_dst_waterway_100m_2016": (15.2655, 15.4519), 
        "uga_osm_dst_road_100m_2016": (3.7082,  5.6608), 
        "uga_srtm_slope_100m": (3.2089 , 3.8157),
        "uga_srtm_topo_100m": (1143.9080 , 402.7680),
        "uga_viirs_100m_2015": (0.2205 , 1.0825),
        "uga_wdpa_dst_cat1_100m_2015": (388.8701 , 221.6834),
        'buildings_merge_mean_area': (26.3673, 48.1988)
    },
    "cod":{
        "buildings_j": (0.00089380914, 8.41622997e-03),
        "esaccilc_dst011_100m_2000": (2.81727052, 5.69885715e+00),
        "esaccilc_dst040_100m_2000": (0.44520899, 2.72345595e+00),
        "esaccilc_dst130_100m_2000": (3.09648584, 4.70480562e+00),
        "esaccilc_dst140_100m_2000": (9.18009012, 9.85124076e+00),
        "esaccilc_dst150_100m_2000": (108.52330299, 8.17261502e+01),
        "esaccilc_dst160_100m_2000": (8.65616757, 9.26884634e+00),
        "esaccilc_dst190_100m_2000": (37.38046272, 3.10730075e+01),
        "esaccilc_dst200_100m_2000": (64.73759992, 4.46013049e+01),
        "cod_tt50k_100m_2000": (209.1351, 188.1936),
        "cod_dst_bsgme_100m_2015": (3.2670, 4.2283),
        "cod_dst_ghslesaccilcgufghsll_100m_2014": (3.3381, 4.2542),
        "cod_dst_coastline_100m_2000_2020": (698.8104,  326.6751),
        "cod_dmsp_100m_2011": (71.8836 ,  399.8602 ),
        "cod_esaccilc_dst_water_100m_2000_2012":(18.1815, 14.8920), 
        "cod_osm_dst_roadintersec_100m_2016":(19.8096, 29.5883), 
        "cod_osm_dst_waterway_100m_2016": (15.2655, 15.4519), 
        "cod_osm_dst_road_100m_2016": (3.7082,  5.6608), 
        "cod_srtm_slope_100m": (3.2089 , 3.8157),
        "cod_srtm_topo_100m": (1143.9080 , 402.7680),
        "cod_viirs_100m_2015": (0.2205 , 1.0825),
        "cod_wdpa_dst_cat1_100m_2015": (388.8701 , 221.6834),
        'buildings_merge_mean_area': (26.3673, 48.1988)
    },
    "rwa":{
        "buildings_j": (0.00089380914, 8.41622997e-03),
        "esaccilc_dst011_100m_2000": (2.81727052, 5.69885715e+00),
        "esaccilc_dst040_100m_2000": (0.44520899, 2.72345595e+00),
        "esaccilc_dst130_100m_2000": (3.09648584, 4.70480562e+00),
        "esaccilc_dst140_100m_2000": (9.18009012, 9.85124076e+00),
        "esaccilc_dst150_100m_2000": (108.52330299, 8.17261502e+01),
        "esaccilc_dst160_100m_2000": (8.65616757, 9.26884634e+00),
        "esaccilc_dst190_100m_2000": (37.38046272, 3.10730075e+01),
        "esaccilc_dst200_100m_2000": (64.73759992, 4.46013049e+01),
        "rwa_tt50k_100m_2000": (209.1351, 188.1936),
        "rwa_dst_bsgme_100m_2015": (3.2670, 4.2283),
        "rwa_dst_ghslesaccilcgufghsll_100m_2014": (3.3381, 4.2542),
        "rwa_dst_coastline_100m_2000_2020": (698.8104,  326.6751),
        "rwa_dmsp_100m_2011": (71.8836 ,  399.8602 ),
        "rwa_esaccilc_dst_water_100m_2000_2012":(18.1815, 14.8920), 
        "rwa_osm_dst_roadintersec_100m_2016":(19.8096, 29.5883), 
        "rwa_osm_dst_waterway_100m_2016": (15.2655, 15.4519), 
        "rwa_osm_dst_road_100m_2016": (3.7082,  5.6608), 
        "rwa_srtm_slope_100m": (3.2089 , 3.8157),
        "rwa_srtm_topo_100m": (1143.9080 , 402.7680),
        "rwa_viirs_100m_2015": (0.2205 , 1.0825),
        "rwa_wdpa_dst_cat1_100m_2015": (388.8701 , 221.6834),
        'buildings_merge_mean_area': (26.3673, 48.1988)
    },
    "moz":{
        "buildings_j": (0.00089380914, 8.41622997e-03),
        "esaccilc_dst011_100m_2000": (2.81727052, 5.69885715e+00),
        "esaccilc_dst040_100m_2000": (0.44520899, 2.72345595e+00),
        "esaccilc_dst130_100m_2000": (3.09648584, 4.70480562e+00),
        "esaccilc_dst140_100m_2000": (9.18009012, 9.85124076e+00),
        "esaccilc_dst150_100m_2000": (108.52330299, 8.17261502e+01),
        "esaccilc_dst160_100m_2000": (8.65616757, 9.26884634e+00),
        "esaccilc_dst190_100m_2000": (37.38046272, 3.10730075e+01),
        "esaccilc_dst200_100m_2000": (64.73759992, 4.46013049e+01),
        "moz_tt50k_100m_2000": (209.1351, 188.1936),
        "moz_dst_bsgme_100m_2015": (3.2670, 4.2283),
        "moz_dst_ghslesaccilcgufghsll_100m_2014": (3.3381, 4.2542),
        "moz_dst_coastline_100m_2000_2020": (698.8104,  326.6751),
        "moz_dmsp_100m_2011": (71.8836 ,  399.8602 ),
        "moz_esaccilc_dst_water_100m_2000_2012":(18.1815, 14.8920), 
        "moz_osm_dst_roadintersec_100m_2016":(19.8096, 29.5883), 
        "moz_osm_dst_waterway_100m_2016": (15.2655, 15.4519), 
        "moz_osm_dst_road_100m_2016": (3.7082,  5.6608), 
        "moz_srtm_slope_100m": (3.2089 , 3.8157),
        "moz_srtm_topo_100m": (1143.9080 , 402.7680),
        "moz_viirs_100m_2015": (0.2205 , 1.0825),
        "moz_wdpa_dst_cat1_100m_2015": (388.8701 , 221.6834),
        'buildings_merge_mean_area': (26.3673, 48.1988)
    },
    "zmb":{
        "buildings_j": (0.00089380914, 8.41622997e-03),
        "esaccilc_dst011_100m_2000": (2.81727052, 5.69885715e+00),
        "esaccilc_dst040_100m_2000": (0.44520899, 2.72345595e+00),
        "esaccilc_dst130_100m_2000": (3.09648584, 4.70480562e+00),
        "esaccilc_dst140_100m_2000": (9.18009012, 9.85124076e+00),
        "esaccilc_dst150_100m_2000": (108.52330299, 8.17261502e+01),
        "esaccilc_dst160_100m_2000": (8.65616757, 9.26884634e+00),
        "esaccilc_dst190_100m_2000": (37.38046272, 3.10730075e+01),
        "esaccilc_dst200_100m_2000": (64.73759992, 4.46013049e+01),
        "zmb_tt50k_100m_2000": (209.1351, 188.1936),
        "zmb_dst_bsgme_100m_2015": (3.2670, 4.2283),
        "zmb_dst_ghslesaccilcgufghsll_100m_2014": (3.3381, 4.2542),
        "zmb_dst_coastline_100m_2000_2020": (698.8104,  326.6751),
        "zmb_dmsp_100m_2011": (71.8836 ,  399.8602 ),
        "zmb_esaccilc_dst_water_100m_2000_2012":(18.1815, 14.8920), 
        "zmb_osm_dst_roadintersec_100m_2016":(19.8096, 29.5883), 
        "zmb_osm_dst_waterway_100m_2016": (15.2655, 15.4519), 
        "zmb_osm_dst_road_100m_2016": (3.7082,  5.6608), 
        "zmb_srtm_slope_100m": (3.2089 , 3.8157),
        "zmb_srtm_topo_100m": (1143.9080 , 402.7680),
        "zmb_viirs_100m_2015": (0.2205 , 1.0825),
        "zmb_wdpa_dst_cat1_100m_2015": (388.8701 , 221.6834),
        'buildings_merge_mean_area': (26.3673, 48.1988)
    }
}

metadata = {
    "tza": {
        "wp_no_data": [0, 1],
        "wp_covariates_no_data": -9999,
        "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": f'{root_path}TZA/preprocessed_census_data_tza.pkl',
        "rst_wp_regions_path": f'{root_path}TZA/admin_regions/tza_wp_admin_regions.tif'
    },
    "uga":{
        "wp_no_data": [0, 1],
        "wp_covariates_no_data": -9999,
        "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": f'{root_path}UGA/preprocessed_census_data_uga.pkl',
        "rst_wp_regions_path": f'{root_path}UGA/admin_regions/uga_wp_admin_regions.tif'
    },
    "cod":{
        "wp_no_data": [0, 1],
        "wp_covariates_no_data": -9999,
        "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": f'{root_path}COD/preprocessed_census_data_cod.pkl',
        "rst_wp_regions_path": f'{root_path}COD/admin_regions/cod_wp_admin_regions.tif'
    },
    "rwa":{
        "wp_no_data": [0, 1],
        "wp_covariates_no_data": -9999,
        "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": f'{root_path}RWA/preprocessed_census_data_rwa.pkl',
        "rst_wp_regions_path": f'{root_path}RWA/admin_regions/rwa_wp_admin_regions.tif'
    },
    "moz":{
        "wp_no_data": [0, 1],
        "wp_covariates_no_data": -9999,
        "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": f'{root_path}MOZ/preprocessed_census_data_moz.pkl',
        "rst_wp_regions_path": f'{root_path}MOZ/admin_regions/moz_wp_admin_regions.tif'
    },
    "zmb":{
        "wp_no_data": [0],
        "wp_covariates_no_data": -9999,
        "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": f'{root_path}ZMB/preprocessed_census_data_zmb.pkl',
        "rst_wp_regions_path": f'{root_path}ZMB/admin_regions/zmb_wp_admin_regions.tif'
    }
}

input_paths["tza_f3"] = {
    "buildings_google": input_paths["tza"]["buildings_google"],
    "buildings_maxar": input_paths["tza"]["buildings_maxar"],
    "buildings_google_mean_area": input_paths["tza"]["buildings_google_mean_area"],
    "buildings_maxar_mean_area": input_paths["tza"]["buildings_maxar_mean_area"],
    "tza_viirs_100m_2015" : input_paths["tza"]["tza_viirs_100m_2015"]
}

metadata["tza_f3"] = metadata["tza"]

# Columns of shapefiles
col_coarse_level_seq_id = "GR_SID"
col_finest_level_seq_id = "SID"
