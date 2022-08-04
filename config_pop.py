
import os


# Data root path
# if os.isdir()
root_paths = ["/home/john.vargas/data/wpop/",
    "/scratch/Nando/HAC2/data/",
    "/cluster/work/igp_psr/metzgern/HAC/data/",
    "/scratch2/metzgern/HAC/data/",
    "/scratch2/metzgern/HAC/data/codedata/"]

for dir in root_paths:
    if os.path.isdir(dir):
        root_path = dir

# Input files
input_paths_sat2pop = {
    "tza": {
        "buildings_preds": "{}OtherBuildings/TZA/TZA_gbp_BCB_v1_count.tif".format(root_path),
    },
    "rwa": {
        # "buildings_google": "{}OtherBuildings/RWA/RWA_gbp_BCB_v1_count.tif".format(root_path),
        "BuildingPreds_Own": "{}/ResampledBuildDenseCovariates5/BuildingPreds-1-RWAc.tif".format(root_path),
        "DeepFeatures_Own": "{}/ResampledBuildDenseCovariates5/BuildingFeatures-1-RWAc.tif".format(root_path), 
    },
    "zaf": {
        # "buildings_google": "{}OtherBuildings/RWA/RWA_gbp_BCB_v1_count.tif".format(root_path),
        "BuildingPreds_Own": "{}/ResampledBuildDenseCovariates3/BuildingPreds-1-ZAFc.tif".format(root_path),
        "DeepFeatures_Own": "{}/ResampledBuildDenseCovariates3/BuildingFeatures-1-ZAFc.tif".format(root_path), 
    }
}

input_paths = {
    "tza": {
        # "buildings_j": "{}OtherBuildings/TZA/tza_gbuildings.tif".format(root_path),
        "buildings_google": "{}OtherBuildings/TZA/TZA_gbp_BCB_v1_count.tif".format(root_path),
        "buildings_maxar": "{}OtherBuildings/TZA/TZA_mbp_BCB_v3_count.tif".format(root_path),
        "buildings_google_mean_area": "{}OtherBuildings/TZA/TZA_gbp_BCB_v1_mean_area.tif".format(root_path),
        "buildings_maxar_mean_area": "{}OtherBuildings/TZA/TZA_mbp_BCB_v3_mean_area.tif".format(root_path),
        "tza_tt50k_100m_2000": "{}Covariates/TZA/Accessibility/tza_tt50k_100m_2000.tif".format(
            root_path), 
        "tza_dst_bsgme_100m_2015": "{}Covariates/TZA/BSGM/2015/DTE/tza_dst_bsgme_100m_2015.tif".format(
            root_path),
        "tza_dst_ghslesaccilcgufghsll_100m_2014": "{}Covariates/TZA/BuiltSettlement/2014/DTE/tza_dst_ghslesaccilcgufghsll_100m_2014.tif".format(
            root_path),
        "tza_dst_coastline_100m_2000_2020": "{}Covariates/TZA/Coastline/DST/tza_dst_coastline_100m_2000_2020.tif".format(
            root_path),
        "tza_dmsp_100m_2011": "{}Covariates/TZA/DMSP/tza_dmsp_100m_2011.tif".format(
            root_path),
        "tza_esaccilc_dst_water_100m_2000_2012": "{}Covariates/TZA/ESA_CCI_Water/DST/tza_esaccilc_dst_water_100m_2000_2012.tif".format(
            root_path),
        "tza_osm_dst_roadintersec_100m_2016": "{}Covariates/TZA/OSM/DST/tza_osm_dst_roadintersec_100m_2016.tif".format(
            root_path),
        "tza_osm_dst_waterway_100m_2016": "{}Covariates/TZA/OSM/DST/tza_osm_dst_waterway_100m_2016.tif".format(
            root_path),
        "tza_osm_dst_road_100m_2016": "{}Covariates/TZA/OSM/DST/tza_osm_dst_road_100m_2016.tif".format(
            root_path),
        "tza_srtm_slope_100m": "{}Covariates/TZA/Slope/tza_srtm_slope_100m.tif".format(
            root_path),
        "tza_srtm_topo_100m": "{}Covariates/TZA/Topo/tza_srtm_topo_100m.tif".format(
            root_path),
        "tza_viirs_100m_2015": "{}Covariates/TZA/VIIRS/tza_viirs_100m_2015.tif".format(
            root_path),
        "tza_wdpa_dst_cat1_100m_2015": "{}Covariates/TZA/WDPA/WDPA_1/tza_wdpa_dst_cat1_100m_2015.tif".format(
            root_path),
    },
    "uga": {
        "buildings_google": "{}OtherBuildings/UGA/UGA_gbp_BCB_v1_count.tif".format(root_path),
        "buildings_maxar": "{}OtherBuildings/UGA/UGA_mbp_BCB_v3_count.tif".format(root_path),
        "buildings_google_mean_area": "{}OtherBuildings/UGA/UGA_gbp_BCB_v1_mean_area.tif".format(root_path),
        "buildings_maxar_mean_area": "{}OtherBuildings/UGA/UGA_mbp_BCB_v3_mean_area.tif".format(root_path),
        "uga_tt50k_100m_2000": "{}Covariates/UGA/Accessibility/uga_tt50k_100m_2000.tif".format(
            root_path),
        "uga_dst_bsgme_100m_2015": "{}Covariates/UGA/BSGM/2015/DTE/uga_dst_bsgme_100m_2015.tif".format(
            root_path),
        "uga_dst_ghslesaccilcgufghsll_100m_2014": "{}Covariates/UGA/BuiltSettlement/2014/DTE/uga_dst_ghslesaccilcgufghsll_100m_2014.tif".format(
            root_path),
        "uga_dst_coastline_100m_2000_2020": "{}Covariates/UGA/Coastline/DST/uga_dst_coastline_100m_2000_2020.tif".format(
            root_path),
        "uga_dmsp_100m_2011": "{}Covariates/UGA/DMSP/uga_dmsp_100m_2011.tif".format(
            root_path),
        "uga_esaccilc_dst_water_100m_2000_2012": "{}Covariates/UGA/ESA_CCI_Water/DST/uga_esaccilc_dst_water_100m_2000_2012.tif".format(
            root_path),
        "uga_osm_dst_roadintersec_100m_2016": "{}Covariates/UGA/OSM/DST/uga_osm_dst_roadintersec_100m_2016.tif".format(
            root_path),
        "uga_osm_dst_waterway_100m_2016": "{}Covariates/UGA/OSM/DST/uga_osm_dst_waterway_100m_2016.tif".format(
            root_path),
        "uga_osm_dst_road_100m_2016": "{}Covariates/UGA/OSM/DST/uga_osm_dst_road_100m_2016.tif".format(
            root_path),
        "uga_srtm_slope_100m": "{}Covariates/UGA/Slope/uga_srtm_slope_100m.tif".format(
            root_path),
        "uga_srtm_topo_100m": "{}Covariates/UGA/Topo/uga_srtm_topo_100m.tif".format(
            root_path),
        "uga_viirs_100m_2015": "{}Covariates/UGA/VIIRS/uga_viirs_100m_2015.tif".format(
            root_path),
        "uga_wdpa_dst_cat1_100m_2015": "{}Covariates/UGA/WDPA/WDPA_1/uga_wdpa_dst_cat1_100m_2015.tif".format(
            root_path),
    },
    "cod": {
        "buildings_google": "{}OtherBuildings/COD/COD_gbp_BCB_v1_count.tif".format(root_path),
        "buildings_maxar": "{}OtherBuildings/COD/COD_mbp_BCB_v3_count.tif".format(root_path),
        "buildings_google_mean_area": "{}OtherBuildings/COD/COD_gbp_BCB_v1_mean_area.tif".format(root_path),
        "buildings_maxar_mean_area": "{}OtherBuildings/COD/COD_mbp_BCB_v3_mean_area.tif".format(root_path),
        "cod_tt50k_100m_2000": "{}Covariates/COD/Accessibility/cod_tt50k_100m_2000.tif".format(
            root_path),
        "cod_dst_bsgme_100m_2015": "{}Covariates/COD/BSGM/2015/DTE/cod_dst_bsgme_100m_2015.tif".format(
            root_path),
        "cod_dst_ghslesaccilcgufghsll_100m_2014": "{}Covariates/COD/BuiltSettlement/2014/DTE/cod_dst_ghslesaccilcgufghsll_100m_2014.tif".format(
            root_path),
        "cod_dst_coastline_100m_2000_2020": "{}Covariates/COD/Coastline/DST/cod_dst_coastline_100m_2000_2020.tif".format(
            root_path),
        "cod_dmsp_100m_2011": "{}Covariates/COD/DMSP/cod_dmsp_100m_2011.tif".format(
            root_path),
        "cod_esaccilc_dst_water_100m_2000_2012": "{}Covariates/COD/ESA_CCI_Water/DST/cod_esaccilc_dst_water_100m_2000_2012.tif".format(
            root_path),
        "cod_osm_dst_roadintersec_100m_2016": "{}Covariates/COD/OSM/DST/cod_osm_dst_roadintersec_100m_2016.tif".format(
            root_path),
        "cod_osm_dst_waterway_100m_2016": "{}Covariates/COD/OSM/DST/cod_osm_dst_waterway_100m_2016.tif".format(
            root_path),
        "cod_osm_dst_road_100m_2016": "{}Covariates/COD/OSM/DST/cod_osm_dst_road_100m_2016.tif".format(
            root_path),
        "cod_srtm_slope_100m": "{}Covariates/COD/Slope/cod_srtm_slope_100m.tif".format(
            root_path),
        "cod_srtm_topo_100m": "{}Covariates/COD/Topo/cod_srtm_topo_100m.tif".format(
            root_path),
        "cod_viirs_100m_2015": "{}Covariates/COD/VIIRS/cod_viirs_100m_2015.tif".format(
            root_path),
        "cod_wdpa_dst_cat1_100m_2015": "{}Covariates/COD/WDPA/WDPA_1/cod_wdpa_dst_cat1_100m_2015.tif".format(
            root_path),
    },
    "rwa": {
        "buildings_google": "{}OtherBuildings/RWA/RWA_gbp_BCB_v1_count.tif".format(root_path),
        "buildings_maxar": "{}OtherBuildings/RWA/RWA_mbp_BCB_v3_count.tif".format(root_path),
        "buildings_google_mean_area": "{}OtherBuildings/RWA/RWA_gbp_BCB_v1_mean_area.tif".format(root_path),
        "buildings_maxar_mean_area": "{}OtherBuildings/RWA/RWA_mbp_BCB_v3_mean_area.tif".format(root_path),
        "rwa_tt50k_100m_2000": "{}Covariates/RWA/Accessibility/rwa_tt50k_100m_2000.tif".format(
            root_path),
        "rwa_dst_bsgme_100m_2015": "{}Covariates/RWA/BSGM/2015/DTE/rwa_dst_bsgme_100m_2015.tif".format(
            root_path),
        "rwa_dst_ghslesaccilcgufghsll_100m_2014": "{}Covariates/RWA/BuiltSettlement/2014/DTE/rwa_dst_ghslesaccilcgufghsll_100m_2014.tif".format(
            root_path),
        "rwa_dst_coastline_100m_2000_2020": "{}Covariates/RWA/Coastline/DST/rwa_dst_coastline_100m_2000_2020.tif".format(
            root_path),
        "rwa_dmsp_100m_2011": "{}Covariates/RWA/DMSP/rwa_dmsp_100m_2011.tif".format(
            root_path),
        "rwa_esaccilc_dst_water_100m_2000_2012": "{}Covariates/RWA/ESA_CCI_Water/DST/rwa_esaccilc_dst_water_100m_2000_2012.tif".format(
            root_path),
        "rwa_osm_dst_roadintersec_100m_2016": "{}Covariates/RWA/OSM/DST/rwa_osm_dst_roadintersec_100m_2016.tif".format(
            root_path),
        "rwa_osm_dst_waterway_100m_2016": "{}Covariates/RWA/OSM/DST/rwa_osm_dst_waterway_100m_2016.tif".format(
            root_path),
        "rwa_osm_dst_road_100m_2016": "{}Covariates/RWA/OSM/DST/rwa_osm_dst_road_100m_2016.tif".format(
            root_path),
        "rwa_srtm_slope_100m": "{}Covariates/RWA/Slope/rwa_srtm_slope_100m.tif".format(
            root_path),
        "rwa_srtm_topo_100m": "{}Covariates/RWA/Topo/rwa_srtm_topo_100m.tif".format(
            root_path),
        "rwa_viirs_100m_2015": "{}Covariates/RWA/VIIRS/rwa_viirs_100m_2015.tif".format(
            root_path),
        "rwa_wdpa_dst_cat1_100m_2015": "{}Covariates/RWA/WDPA/WDPA_1/rwa_wdpa_dst_cat1_100m_2015.tif".format(
            root_path),
    },
    "nga": {
        "buildings_google": "{}OtherBuildings/NGA/NGA_gbp_BCB_v1_count.tif".format(root_path),
        "buildings_maxar": "{}OtherBuildings/NGA/NGA_mbp_BCB_v3_count.tif".format(root_path),
        "buildings_google_mean_area": "{}OtherBuildings/NGA/NGA_gbp_BCB_v1_mean_area.tif".format(root_path),
        "buildings_maxar_mean_area": "{}OtherBuildings/NGA/NGA_mbp_BCB_v3_mean_area.tif".format(root_path),
        "nga_tt50k_100m_2000": "{}Covariates/NGA/Accessibility/nga_tt50k_100m_2000.tif".format(
            root_path),
        "nga_dst_bsgme_100m_2015": "{}Covariates/NGA/BSGM/2015/DTE/nga_dst_bsgme_100m_2015.tif".format(
            root_path),
        "nga_dst_ghslesaccilcgufghsll_100m_2014": "{}Covariates/NGA/BuiltSettlement/2014/DTE/nga_dst_ghslesaccilcgufghsll_100m_2014.tif".format(
            root_path),
        "nga_dst_coastline_100m_2000_2020": "{}Covariates/NGA/Coastline/DST/nga_dst_coastline_100m_2000_2020.tif".format(
            root_path),
        "nga_dmsp_100m_2011": "{}Covariates/NGA/DMSP/nga_dmsp_100m_2011.tif".format(
            root_path),
        "nga_esaccilc_dst_water_100m_2000_2012": "{}Covariates/NGA/ESA_CCI_Water/DST/nga_esaccilc_dst_water_100m_2000_2012.tif".format(
            root_path),
        "nga_osm_dst_roadintersec_100m_2016": "{}Covariates/NGA/OSM/DST/nga_osm_dst_roadintersec_100m_2016.tif".format(
            root_path),
        "nga_osm_dst_waterway_100m_2016": "{}Covariates/NGA/OSM/DST/nga_osm_dst_waterway_100m_2016.tif".format(
            root_path),
        "nga_osm_dst_road_100m_2016": "{}Covariates/NGA/OSM/DST/nga_osm_dst_road_100m_2016.tif".format(
            root_path),
        "nga_srtm_slope_100m": "{}Covariates/NGA/Slope/nga_srtm_slope_100m.tif".format(
            root_path),
        "nga_srtm_topo_100m": "{}Covariates/NGA/Topo/nga_srtm_topo_100m.tif".format(
            root_path),
        "nga_viirs_100m_2015": "{}Covariates/NGA/VIIRS/nga_viirs_100m_2015.tif".format(
            root_path),
        "nga_wdpa_dst_cat1_100m_2015": "{}Covariates/NGA/WDPA/WDPA_1/nga_wdpa_dst_cat1_100m_2015.tif".format(
            root_path),
    },
    "moz": {
        "buildings_google": "{}OtherBuildings/MOZ/MOZ_gbp_BCB_v1_count.tif".format(root_path),
        "buildings_maxar": "{}OtherBuildings/MOZ/MOZ_mbp_BCB_v3_count.tif".format(root_path),
        "buildings_google_mean_area": "{}OtherBuildings/MOZ/MOZ_gbp_BCB_v1_mean_area.tif".format(root_path),
        "buildings_maxar_mean_area": "{}OtherBuildings/MOZ/MOZ_mbp_BCB_v3_mean_area.tif".format(root_path),
        "moz_tt50k_100m_2000": "{}Covariates/MOZ/Accessibility/moz_tt50k_100m_2000.tif".format(
            root_path),
        "moz_dst_bsgme_100m_2015": "{}Covariates/MOZ/BSGM/2015/DTE/moz_dst_bsgme_100m_2015.tif".format(
            root_path),
        "moz_dst_ghslesaccilcgufghsll_100m_2014": "{}Covariates/MOZ/BuiltSettlement/2014/DTE/moz_dst_ghslesaccilcgufghsll_100m_2014.tif".format(
            root_path),
        "moz_dst_coastline_100m_2000_2020": "{}Covariates/MOZ/Coastline/DST/moz_dst_coastline_100m_2000_2020.tif".format(
            root_path),
        "moz_dmsp_100m_2011": "{}Covariates/MOZ/DMSP/moz_dmsp_100m_2011.tif".format(
            root_path),
        "moz_esaccilc_dst_water_100m_2000_2012": "{}Covariates/MOZ/ESA_CCI_Water/DST/moz_esaccilc_dst_water_100m_2000_2012.tif".format(
            root_path),
        "moz_osm_dst_roadintersec_100m_2016": "{}Covariates/MOZ/OSM/DST/moz_osm_dst_roadintersec_100m_2016.tif".format(
            root_path),
        "moz_osm_dst_waterway_100m_2016": "{}Covariates/MOZ/OSM/DST/moz_osm_dst_waterway_100m_2016.tif".format(
            root_path),
        "moz_osm_dst_road_100m_2016": "{}Covariates/MOZ/OSM/DST/moz_osm_dst_road_100m_2016.tif".format(
            root_path),
        "moz_srtm_slope_100m": "{}Covariates/MOZ/Slope/moz_srtm_slope_100m.tif".format(
            root_path),
        "moz_srtm_topo_100m": "{}Covariates/MOZ/Topo/moz_srtm_topo_100m.tif".format(
            root_path),
        "moz_viirs_100m_2015": "{}Covariates/MOZ/VIIRS/moz_viirs_100m_2015.tif".format(
            root_path),
        "moz_wdpa_dst_cat1_100m_2015": "{}Covariates/MOZ/WDPA/WDPA_1/moz_wdpa_dst_cat1_100m_2015.tif".format(
            root_path),
    },
    "zmb" : {
        "buildings_google": "{}OtherBuildings/ZMB/ZMB_own_google_bcount.tif".format(root_path),
        "buildings_maxar": "{}OtherBuildings/ZMB/ZMB_buildings_v2_0_count.tif".format(root_path),
        "buildings_google_mean_area": "{}OtherBuildings/ZMB/ZMB_own_google_barea_v3.tif".format(root_path),
        "buildings_maxar_mean_area": "{}OtherBuildings/ZMB/ZMB_buildings_v2_0_mean_area.tif".format(root_path),
        "zmb_tt50k_100m_2000": "{}Covariates/ZMB/Accessibility/zmb_tt50k_100m_2000.tif".format(
            root_path),
        "zmb_dst_bsgme_100m_2015": "{}Covariates/ZMB/BSGM/2015/DTE/zmb_dst_bsgme_100m_2015.tif".format(
            root_path),
        "zmb_dst_ghslesaccilcgufghsll_100m_2014": "{}Covariates/ZMB/BuiltSettlement/2014/DTE/zmb_dst_ghslesaccilcgufghsll_100m_2014.tif".format(
            root_path),
        "zmb_dst_coastline_100m_2000_2020": "{}Covariates/ZMB/Coastline/DST/zmb_dst_coastline_100m_2000_2020.tif".format(
            root_path),
        "zmb_dmsp_100m_2011": "{}Covariates/ZMB/DMSP/zmb_dmsp_100m_2011.tif".format(
            root_path),
        "zmb_esaccilc_dst_water_100m_2000_2012": "{}Covariates/ZMB/ESA_CCI_Water/DST/zmb_esaccilc_dst_water_100m_2000_2012.tif".format(
            root_path),
        "zmb_osm_dst_roadintersec_100m_2016": "{}Covariates/ZMB/OSM/DST/zmb_osm_dst_roadintersec_100m_2016.tif".format(
            root_path),
        "zmb_osm_dst_waterway_100m_2016": "{}Covariates/ZMB/OSM/DST/zmb_osm_dst_waterway_100m_2016.tif".format(
            root_path),
        "zmb_osm_dst_road_100m_2016": "{}Covariates/ZMB/OSM/DST/zmb_osm_dst_road_100m_2016.tif".format(
            root_path),
        "zmb_srtm_slope_100m": "{}Covariates/ZMB/Slope/zmb_srtm_slope_100m.tif".format(
            root_path),
        "zmb_srtm_topo_100m": "{}Covariates/ZMB/Topo/zmb_srtm_topo_100m.tif".format(
            root_path),
        "zmb_viirs_100m_2015": "{}Covariates/ZMB/VIIRS/zmb_viirs_100m_2015.tif".format(
            root_path),
        "zmb_wdpa_dst_cat1_100m_2015": "{}Covariates/ZMB/WDPA/WDPA_1/zmb_wdpa_dst_cat1_100m_2015.tif".format(
            root_path),
    },
    "zaf" : {
        "buildings_google": "{}OtherBuildings/ZAF/ZAF_own_google_bcount.tif".format(root_path),
        # "buildings_maxar": "{}OtherBuildings/ZAF/ZAF_buildings_v1_1_count.tif".format(root_path),
        "buildings_google_mean_area": "{}OtherBuildings/ZAF/ZAF_own_google_meanarea.tif".format(root_path),
        # "buildings_maxar_mean_area": "{}OtherBuildings/ZAF/ZAF_buildings_v1_1_mean_area.tif".format(root_path),
        # "zaf_tt50k_100m_2000": "{}Covariates/ZAF/Accessibility/zaf_tt50k_100m_2000.tif".format(
        #     root_path),
        # "zaf_dst_bsgme_100m_2015": "{}Covariates/ZAF/BSGM/2015/DTE/zaf_dst_bsgme_100m_2015.tif".format(
        #     root_path),
        # "zaf_dst_ghslesaccilcgufghsll_100m_2014": "{}Covariates/ZAF/BuiltSettlement/2014/DTE/zaf_dst_ghslesaccilcgufghsll_100m_2014.tif".format(
        #      root_path),
        # "zaf_dst_coastline_100m_2000_2020": "{}Covariates/ZAF/Coastline/DST/zaf_dst_coastline_100m_2000_2020.tif".format(
        #     root_path),
        # "zaf_dmsp_100m_2011": "{}Covariates/ZAF/DMSP/zaf_dmsp_100m_2011.tif".format(
        #     root_path),
        # "zaf_esaccilc_dst_water_100m_2000_2012": "{}Covariates/ZAF/ESA_CCI_Water/DST/zaf_esaccilc_dst_water_100m_2000_2012.tif".format(
        #     root_path),
        # "zaf_osm_dst_roadintersec_100m_2016": "{}Covariates/ZAF/OSM/DST/zaf_osm_dst_roadintersec_100m_2016.tif".format(
        #     root_path),
        # "zaf_osm_dst_waterway_100m_2016": "{}Covariates/ZAF/OSM/DST/zaf_osm_dst_waterway_100m_2016.tif".format(
        #     root_path),
        # "zaf_osm_dst_road_100m_2016": "{}Covariates/ZAF/OSM/DST/zaf_osm_dst_road_100m_2016.tif".format(
        #     root_path),
        # "zaf_srtm_slope_100m": "{}Covariates/ZAF/Slope/zaf_srtm_slope_100m.tif".format(
        #     root_path),
        # "zaf_srtm_topo_100m": "{}Covariates/ZAF/Topo/zaf_srtm_topo_100m.tif".format(
        #     root_path),
        # "zaf_viirs_100m_2015": "{}Covariates/ZAF/VIIRS/zaf_viirs_100m_2015.tif".format(
        #     root_path),
        # "zaf_wdpa_dst_cat1_100m_2015": "{}Covariates/ZAF/WDPA/WDPA_1/zaf_wdpa_dst_cat1_100m_2015.tif".format(
        #     root_path),
    },
    "dza" : {
        "buildings_google": "{}OtherBuildings/DZA/DZA_own_google_bcount.tif".format(root_path),
        # "buildings_maxar": "{}OtherBuildings/DZA/DZA_buildings_v1_1_count.tif".format(root_path),
        "buildings_google_mean_area": "{}OtherBuildings/DZA/DZA_own_google_meanarea.tif".format(root_path),
        # "buildings_maxar_mean_area": "{}OtherBuildings/DZA/DZA_buildings_v1_1_mean_area.tif".format(root_path),
        "dza_tt50k_100m_2000": "{}Covariates/DZA/Accessibility/dza_tt50k_100m_2000.tif".format(
            root_path),
        "dza_dst_bsgme_100m_2015": "{}Covariates/DZA/BSGM/2015/DTE/dza_dst_bsgme_100m_2015.tif".format(
            root_path),
        "dza_dst_ghslesaccilcgufghsll_100m_2014": "{}Covariates/DZA/BuiltSettlement/2014/DTE/dza_dst_ghslesaccilcgufghsll_100m_2014.tif".format(
             root_path),
        "dza_dst_coastline_100m_2000_2020": "{}Covariates/DZA/Coastline/DST/dza_dst_coastline_100m_2000_2020.tif".format(
            root_path),
        "dza_dmsp_100m_2011": "{}Covariates/DZA/DMSP/dza_dmsp_100m_2011.tif".format(
            root_path),
        "dza_esaccilc_dst_water_100m_2000_2012": "{}Covariates/DZA/ESA_CCI_Water/DST/dza_esaccilc_dst_water_100m_2000_2012.tif".format(
            root_path),
        "dza_osm_dst_roadintersec_100m_2016": "{}Covariates/DZA/OSM/DST/dza_osm_dst_roadintersec_100m_2016.tif".format(
            root_path),
        "dza_osm_dst_waterway_100m_2016": "{}Covariates/DZA/OSM/DST/dza_osm_dst_waterway_100m_2016.tif".format(
            root_path),
        "dza_osm_dst_road_100m_2016": "{}Covariates/DZA/OSM/DST/dza_osm_dst_road_100m_2016.tif".format(
            root_path),
        "dza_srtm_slope_100m": "{}Covariates/DZA/Slope/dza_srtm_slope_100m.tif".format(
            root_path),
        "dza_srtm_topo_100m": "{}Covariates/DZA/Topo/dza_srtm_topo_100m.tif".format(
            root_path),
        "dza_viirs_100m_2015": "{}Covariates/DZA/VIIRS/dza_viirs_100m_2015.tif".format(
            root_path),
        "dza_wdpa_dst_cat1_100m_2015": "{}Covariates/DZA/WDPA/WDPA_1/dza_wdpa_dst_cat1_100m_2015.tif".format(
            root_path),
    },
    "mar" : {
        # "buildings_google": "{}OtherBuildings/MAR/MAR_own_google_bcount.tif".format(root_path),
        # "buildings_maxar": "{}OtherBuildings/MAR/MAR_buildings_v1_1_count.tif".format(root_path),
        # "buildings_google_mean_area": "{}OtherBuildings/MAR/MAR_own_google_meanarea.tif".format(root_path),
        # "buildings_maxar_mean_area": "{}OtherBuildings/MAR/MAR_buildings_v1_1_mean_area.tif".format(root_path),
        "mar_tt50k_100m_2000": "{}Covariates/MAR/Accessibility/mar_tt50k_100m_2000.tif".format(
            root_path),
        "mar_dst_bsgme_100m_2015": "{}Covariates/MAR/BSGM/2015/DTE/mar_dst_bsgme_100m_2015.tif".format(
            root_path),
        "mar_dst_ghslesaccilcgufghsll_100m_2014": "{}Covariates/MAR/BuiltSettlement/2014/DTE/mar_dst_ghslesaccilcgufghsll_100m_2014.tif".format(
             root_path),
        "mar_dst_coastline_100m_2000_2020": "{}Covariates/MAR/Coastline/DST/mar_dst_coastline_100m_2000_2020.tif".format(
            root_path),
        "mar_dmsp_100m_2011": "{}Covariates/MAR/DMSP/mar_dmsp_100m_2011.tif".format(
            root_path),
        "mar_esaccilc_dst_water_100m_2000_2012": "{}Covariates/MAR/ESA_CCI_Water/DST/mar_esaccilc_dst_water_100m_2000_2012.tif".format(
            root_path),
        "mar_osm_dst_roadintersec_100m_2016": "{}Covariates/MAR/OSM/DST/mar_osm_dst_roadintersec_100m_2016.tif".format(
            root_path),
        "mar_osm_dst_waterway_100m_2016": "{}Covariates/MAR/OSM/DST/mar_osm_dst_waterway_100m_2016.tif".format(
            root_path),
        "mar_osm_dst_road_100m_2016": "{}Covariates/MAR/OSM/DST/mar_osm_dst_road_100m_2016.tif".format(
            root_path),
        "mar_srtm_slope_100m": "{}Covariates/MAR/Slope/mar_srtm_slope_100m.tif".format(
            root_path),
        "mar_srtm_topo_100m": "{}Covariates/MAR/Topo/mar_srtm_topo_100m.tif".format(
            root_path),
        "mar_viirs_100m_2015": "{}Covariates/MAR/VIIRS/mar_viirs_100m_2015.tif".format(
            root_path),
        "mar_wdpa_dst_cat1_100m_2015": "{}Covariates/MAR/WDPA/WDPA_1/mar_wdpa_dst_cat1_100m_2015.tif".format(
            root_path),
    },
    "mli" : {
        # "buildings_google": "{}OtherBuildings/MAR/MAR_own_google_bcount.tif".format(root_path),
        # "buildings_maxar": "{}OtherBuildings/MAR/MAR_buildings_v1_1_count.tif".format(root_path),
        # "buildings_google_mean_area": "{}OtherBuildings/MAR/MAR_own_google_meanarea.tif".format(root_path),
        # "buildings_maxar_mean_area": "{}OtherBuildings/MAR/MAR_buildings_v1_1_mean_area.tif".format(root_path),
        "mli_tt50k_100m_2000": "{}Covariates/MAR/Accessibility/mli_tt50k_100m_2000.tif".format(
            root_path),
        "mli_dst_bsgme_100m_2015": "{}Covariates/MAR/BSGM/2015/DTE/mli_dst_bsgme_100m_2015.tif".format(
            root_path),
        "mli_dst_ghslesaccilcgufghsll_100m_2014": "{}Covariates/MAR/BuiltSettlement/2014/DTE/mli_dst_ghslesaccilcgufghsll_100m_2014.tif".format(
             root_path),
        "mli_dst_coastline_100m_2000_2020": "{}Covariates/MAR/Coastline/DST/mli_dst_coastline_100m_2000_2020.tif".format(
            root_path),
        "mli_dmsp_100m_2011": "{}Covariates/MAR/DMSP/mli_dmsp_100m_2011.tif".format(
            root_path),
        "mli_esaccilc_dst_water_100m_2000_2012": "{}Covariates/MAR/ESA_CCI_Water/DST/mli_esaccilc_dst_water_100m_2000_2012.tif".format(
            root_path),
        "mli_osm_dst_roadintersec_100m_2016": "{}Covariates/MAR/OSM/DST/mli_osm_dst_roadintersec_100m_2016.tif".format(
            root_path),
        "mli_osm_dst_waterway_100m_2016": "{}Covariates/MAR/OSM/DST/mli_osm_dst_waterway_100m_2016.tif".format(
            root_path),
        "mli_osm_dst_road_100m_2016": "{}Covariates/MAR/OSM/DST/mli_osm_dst_road_100m_2016.tif".format(
            root_path),
        "mli_srtm_slope_100m": "{}Covariates/MAR/Slope/mli_srtm_slope_100m.tif".format(
            root_path),
        "mli_srtm_topo_100m": "{}Covariates/MAR/Topo/mli_srtm_topo_100m.tif".format(
            root_path),
        "mli_viirs_100m_2015": "{}Covariates/MAR/VIIRS/mli_viirs_100m_2015.tif".format(
            root_path),
        "mli_wdpa_dst_cat1_100m_2015": "{}Covariates/MAR/WDPA/WDPA_1/mli_wdpa_dst_cat1_100m_2015.tif".format(
            root_path),
    },
    "civ" : {
        # "buildings_google": "{}OtherBuildings/MAR/MAR_own_google_bcount.tif".format(root_path),
        # "buildings_maxar": "{}OtherBuildings/MAR/MAR_buildings_v1_1_count.tif".format(root_path),
        # "buildings_google_mean_area": "{}OtherBuildings/MAR/MAR_own_google_meanarea.tif".format(root_path),
        # "buildings_maxar_mean_area": "{}OtherBuildings/MAR/MAR_buildings_v1_1_mean_area.tif".format(root_path),
        "civ_tt50k_100m_2000": "{}Covariates/MAR/Accessibility/civ_tt50k_100m_2000.tif".format(
            root_path),
        "civ_dst_bsgme_100m_2015": "{}Covariates/MAR/BSGM/2015/DTE/civ_dst_bsgme_100m_2015.tif".format(
            root_path),
        "civ_dst_ghslesaccilcgufghsll_100m_2014": "{}Covariates/MAR/BuiltSettlement/2014/DTE/civ_dst_ghslesaccilcgufghsll_100m_2014.tif".format(
             root_path),
        "civ_dst_coastline_100m_2000_2020": "{}Covariates/MAR/Coastline/DST/civ_dst_coastline_100m_2000_2020.tif".format(
            root_path),
        "civ_dmsp_100m_2011": "{}Covariates/MAR/DMSP/civ_dmsp_100m_2011.tif".format(
            root_path),
        "civ_esaccilc_dst_water_100m_2000_2012": "{}Covariates/MAR/ESA_CCI_Water/DST/civ_esaccilc_dst_water_100m_2000_2012.tif".format(
            root_path),
        "civ_osm_dst_roadintersec_100m_2016": "{}Covariates/MAR/OSM/DST/civ_osm_dst_roadintersec_100m_2016.tif".format(
            root_path),
        "civ_osm_dst_waterway_100m_2016": "{}Covariates/MAR/OSM/DST/civ_osm_dst_waterway_100m_2016.tif".format(
            root_path),
        "civ_osm_dst_road_100m_2016": "{}Covariates/MAR/OSM/DST/civ_osm_dst_road_100m_2016.tif".format(
            root_path),
        "civ_srtm_slope_100m": "{}Covariates/MAR/Slope/civ_srtm_slope_100m.tif".format(
            root_path),
        "civ_srtm_topo_100m": "{}Covariates/MAR/Topo/civ_srtm_topo_100m.tif".format(
            root_path),
        "civ_viirs_100m_2015": "{}Covariates/MAR/VIIRS/civ_viirs_100m_2015.tif".format(
            root_path),
        "civ_wdpa_dst_cat1_100m_2015": "{}Covariates/MAR/WDPA/WDPA_1/civ_wdpa_dst_cat1_100m_2015.tif".format(
            root_path),
    }
}

no_data_values_Sat2Pop = {
    "rwa":{
        "BuildingPreds_Own": -99999,
        "DeepFeatures_Own0": -99999,
        "DeepFeatures_Own1": -99999,
        "DeepFeatures_Own2": -99999,
        "DeepFeatures_Own3": -99999,
        "DeepFeatures_Own4": -99999,
        "DeepFeatures_Own5": -99999,
        "DeepFeatures_Own6": -99999,
        "DeepFeatures_Own7": -99999,
        "DeepFeatures_Own8": -99999,
        "DeepFeatures_Own9": -99999,
        "DeepFeatures_Own10": -99999,
        "DeepFeatures_Own11": -99999,
        "DeepFeatures_Own12": -99999,
        "DeepFeatures_Own13": -99999,
        "DeepFeatures_Own14": -99999,
        "DeepFeatures_Own15": -99999,
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
        "tza_dmsp_100m_2011": -32767.,
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
    "nga":{
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
        "nga_tt50k_100m_2000": -99999,
        "nga_dst_bsgme_100m_2015": -99999,
        "nga_dst_ghslesaccilcgufghsll_100m_2014": -99999,
        "nga_dst_coastline_100m_2000_2020": -99999,
        "nga_dmsp_100m_2011": 32767.,
        "nga_esaccilc_dst_water_100m_2000_2012":-99999, 
        "nga_osm_dst_roadintersec_100m_2016":-99999, 
        "nga_osm_dst_waterway_100m_2016": -99999,
        "nga_osm_dst_road_100m_2016": -99999,
        "nga_srtm_slope_100m": 255.,
        "nga_srtm_topo_100m": 32767.,
        "nga_viirs_100m_2015": 3.4028e+38,
        "nga_wdpa_dst_cat1_100m_2015": -99999,
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
    },
    "zaf":{
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
        "zaf_tt50k_100m_2000": -99999,
        "zaf_dst_bsgme_100m_2015": -99999,
        "zaf_dst_ghslesaccilcgufghsll_100m_2014": -99999,
        "zaf_dst_coastline_100m_2000_2020": -99999,
        "zaf_dmsp_100m_2011": 32767.,
        "zaf_esaccilc_dst_water_100m_2000_2012":-99999, 
        "zaf_osm_dst_roadintersec_100m_2016":-99999, 
        "zaf_osm_dst_waterway_100m_2016": -99999,
        "zaf_osm_dst_road_100m_2016": -99999,
        "zaf_srtm_slope_100m": 255.,
        "zaf_srtm_topo_100m": 32767.,
        "zaf_viirs_100m_2015": 3.4028235e+38,
        "zaf_wdpa_dst_cat1_100m_2015": -99999,
    },
    "dza":{
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
        "dza_tt50k_100m_2000": -99999,
        "dza_dst_bsgme_100m_2015": -99999,
        "dza_dst_ghslesaccilcgufghsll_100m_2014": -99999,
        "dza_dst_coastline_100m_2000_2020": -99999,
        "dza_dmsp_100m_2011": 32767.,
        "dza_esaccilc_dst_water_100m_2000_2012":-99999, 
        "dza_osm_dst_roadintersec_100m_2016":-99999, 
        "dza_osm_dst_waterway_100m_2016": -99999,
        "dza_osm_dst_road_100m_2016": -99999,
        "dza_srtm_slope_100m": 255.,
        "dza_srtm_topo_100m": 32767.,
        "dza_viirs_100m_2015": 3.4028235e+38,
        "dza_wdpa_dst_cat1_100m_2015": -99999,
    },
    "mar":{
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
        "mar_tt50k_100m_2000": -99999,
        "mar_dst_bsgme_100m_2015": -99999,
        "mar_dst_ghslesaccilcgufghsll_100m_2014": -99999,
        "mar_dst_coastline_100m_2000_2020": -99999,
        "mar_dmsp_100m_2011": 32767.,
        "mar_esaccilc_dst_water_100m_2000_2012":-99999, 
        "mar_osm_dst_roadintersec_100m_2016":-99999, 
        "mar_osm_dst_waterway_100m_2016": -99999,
        "mar_osm_dst_road_100m_2016": -99999,
        "mar_srtm_slope_100m": 255.,
        "mar_srtm_topo_100m": 32767.,
        "mar_viirs_100m_2015": 3.4028235e+38,
        "mar_wdpa_dst_cat1_100m_2015": -99999,
    },
    "mli":{
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
        "mli_tt50k_100m_2000": -99999,
        "mli_dst_bsgme_100m_2015": -99999,
        "mli_dst_ghslesaccilcgufghsll_100m_2014": -99999,
        "mli_dst_coastline_100m_2000_2020": -99999,
        "mli_dmsp_100m_2011": 32767.,
        "mli_esaccilc_dst_water_100m_2000_2012":-99999, 
        "mli_osm_dst_roadintersec_100m_2016":-99999, 
        "mli_osm_dst_waterway_100m_2016": -99999,
        "mli_osm_dst_road_100m_2016": -99999,
        "mli_srtm_slope_100m": 255.,
        "mli_srtm_topo_100m": 32767.,
        "mli_viirs_100m_2015": 3.4028235e+38,
        "mli_wdpa_dst_cat1_100m_2015": -99999,
    },
    "civ":{
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
        "civ_tt50k_100m_2000": -99999,
        "civ_dst_bsgme_100m_2015": -99999,
        "civ_dst_ghslesaccilcgufghsll_100m_2014": -99999,
        "civ_dst_coastline_100m_2000_2020": -99999,
        "civ_dmsp_100m_2011": 32767.,
        "civ_esaccilc_dst_water_100m_2000_2012":-99999, 
        "civ_osm_dst_roadintersec_100m_2016":-99999, 
        "civ_osm_dst_waterway_100m_2016": -99999,
        "civ_osm_dst_road_100m_2016": -99999,
        "civ_srtm_slope_100m": 255.,
        "civ_srtm_topo_100m": 32767.,
        "civ_viirs_100m_2015": 3.4028235e+38,
        "civ_wdpa_dst_cat1_100m_2015": -99999,
    }
}

norms = {
    "tza": {
        "buildings_j": (0.00089380914, 8.41622997e-03),
        # "buildings": (0.265996819, 1.83158563e+00),
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
        # "buildings": (0.265996819, 1.83158563e+00),
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
        # "buildings": (0.265996819, 1.83158563e+00),
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
        # "buildings": (0.265996819, 1.83158563e+00),
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
    "nga":{
        "buildings_j": (0.00089380914, 8.41622997e-03),
        # "buildings": (0.265996819, 1.83158563e+00),
        "esaccilc_dst011_100m_2000": (2.81727052, 5.69885715e+00),
        "esaccilc_dst040_100m_2000": (0.44520899, 2.72345595e+00),
        "esaccilc_dst130_100m_2000": (3.09648584, 4.70480562e+00),
        "esaccilc_dst140_100m_2000": (9.18009012, 9.85124076e+00),
        "esaccilc_dst150_100m_2000": (108.52330299, 8.17261502e+01),
        "esaccilc_dst160_100m_2000": (8.65616757, 9.26884634e+00),
        "esaccilc_dst190_100m_2000": (37.38046272, 3.10730075e+01),
        "esaccilc_dst200_100m_2000": (64.73759992, 4.46013049e+01),
        "nga_tt50k_100m_2000": (209.1351, 188.1936),
        "nga_dst_bsgme_100m_2015": (3.2670, 4.2283),
        "nga_dst_ghslesaccilcgufghsll_100m_2014": (3.3381, 4.2542),
        "nga_dst_coastline_100m_2000_2020": (698.8104,  326.6751),
        "nga_dmsp_100m_2011": (71.8836 ,  399.8602 ),
        "nga_esaccilc_dst_water_100m_2000_2012":(18.1815, 14.8920), 
        "nga_osm_dst_roadintersec_100m_2016":(19.8096, 29.5883), 
        "nga_osm_dst_waterway_100m_2016": (15.2655, 15.4519), 
        "nga_osm_dst_road_100m_2016": (3.7082,  5.6608), 
        "nga_srtm_slope_100m": (3.2089 , 3.8157),
        "nga_srtm_topo_100m": (1143.9080 , 402.7680),
        "nga_viirs_100m_2015": (0.2205 , 1.0825),
        "nga_wdpa_dst_cat1_100m_2015": (388.8701 , 221.6834),
        'buildings_merge_mean_area': (26.3673, 48.1988)
    },
    "moz":{
        "buildings_j": (0.00089380914, 8.41622997e-03),
        # "buildings": (0.265996819, 1.83158563e+00),
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
        # "buildings": (0.265996819, 1.83158563e+00),
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
    },
    "zaf":{
        "buildings_j": (0.00089380914, 8.41622997e-03),
        # "buildings": (0.265996819, 1.83158563e+00),
        "esaccilc_dst011_100m_2000": (2.81727052, 5.69885715e+00),
        "esaccilc_dst040_100m_2000": (0.44520899, 2.72345595e+00),
        "esaccilc_dst130_100m_2000": (3.09648584, 4.70480562e+00),
        "esaccilc_dst140_100m_2000": (9.18009012, 9.85124076e+00),
        "esaccilc_dst150_100m_2000": (108.52330299, 8.17261502e+01),
        "esaccilc_dst160_100m_2000": (8.65616757, 9.26884634e+00),
        "esaccilc_dst190_100m_2000": (37.38046272, 3.10730075e+01),
        "esaccilc_dst200_100m_2000": (64.73759992, 4.46013049e+01),
        "zaf_tt50k_100m_2000": (209.1351, 188.1936),
        "zaf_dst_bsgme_100m_2015": (3.2670, 4.2283),
        "zaf_dst_ghslesaccilcgufghsll_100m_2014": (3.3381, 4.2542),
        "zaf_dst_coastline_100m_2000_2020": (698.8104,  326.6751),
        "zaf_dmsp_100m_2011": (71.8836 ,  399.8602 ),
        "zaf_esaccilc_dst_water_100m_2000_2012":(18.1815, 14.8920), 
        "zaf_osm_dst_roadintersec_100m_2016":(19.8096, 29.5883), 
        "zaf_osm_dst_waterway_100m_2016": (15.2655, 15.4519), 
        "zaf_osm_dst_road_100m_2016": (3.7082,  5.6608), 
        "zaf_srtm_slope_100m": (3.2089 , 3.8157),
        "zaf_srtm_topo_100m": (1143.9080 , 402.7680),
        "zaf_viirs_100m_2015": (0.2205 , 1.0825),
        "zaf_wdpa_dst_cat1_100m_2015": (388.8701 , 221.6834),
        'buildings_merge_mean_area': (26.3673, 48.1988),
        'buildings_google_mean_area': (26.3673, 48.1988)
    },
    "dza":{
        "buildings_j": (0.00089380914, 8.41622997e-03),
        # "buildings": (0.265996819, 1.83158563e+00),
        "esaccilc_dst011_100m_2000": (2.81727052, 5.69885715e+00),
        "esaccilc_dst040_100m_2000": (0.44520899, 2.72345595e+00),
        "esaccilc_dst130_100m_2000": (3.09648584, 4.70480562e+00),
        "esaccilc_dst140_100m_2000": (9.18009012, 9.85124076e+00),
        "esaccilc_dst150_100m_2000": (108.52330299, 8.17261502e+01),
        "esaccilc_dst160_100m_2000": (8.65616757, 9.26884634e+00),
        "esaccilc_dst190_100m_2000": (37.38046272, 3.10730075e+01),
        "esaccilc_dst200_100m_2000": (64.73759992, 4.46013049e+01),
        "dza_tt50k_100m_2000": (209.1351, 188.1936),
        "dza_dst_bsgme_100m_2015": (3.2670, 4.2283),
        "dza_dst_ghslesaccilcgufghsll_100m_2014": (3.3381, 4.2542),
        "dza_dst_coastline_100m_2000_2020": (698.8104,  326.6751),
        "dza_dmsp_100m_2011": (71.8836 ,  399.8602 ),
        "dza_esaccilc_dst_water_100m_2000_2012":(18.1815, 14.8920), 
        "dza_osm_dst_roadintersec_100m_2016":(19.8096, 29.5883), 
        "dza_osm_dst_waterway_100m_2016": (15.2655, 15.4519), 
        "dza_osm_dst_road_100m_2016": (3.7082,  5.6608), 
        "dza_srtm_slope_100m": (3.2089 , 3.8157),
        "dza_srtm_topo_100m": (1143.9080 , 402.7680),
        "dza_viirs_100m_2015": (0.2205 , 1.0825),
        "dza_wdpa_dst_cat1_100m_2015": (388.8701 , 221.6834),
        'buildings_merge_mean_area': (26.3673, 48.1988),
        'buildings_google_mean_area': (26.3673, 48.1988)
    },
    "mar":{
        "buildings_j": (0.00089380914, 8.41622997e-03),
        # "buildings": (0.265996819, 1.83158563e+00),
        "esaccilc_dst011_100m_2000": (2.81727052, 5.69885715e+00),
        "esaccilc_dst040_100m_2000": (0.44520899, 2.72345595e+00),
        "esaccilc_dst130_100m_2000": (3.09648584, 4.70480562e+00),
        "esaccilc_dst140_100m_2000": (9.18009012, 9.85124076e+00),
        "esaccilc_dst150_100m_2000": (108.52330299, 8.17261502e+01),
        "esaccilc_dst160_100m_2000": (8.65616757, 9.26884634e+00),
        "esaccilc_dst190_100m_2000": (37.38046272, 3.10730075e+01),
        "esaccilc_dst200_100m_2000": (64.73759992, 4.46013049e+01),
        "mar_tt50k_100m_2000": (209.1351, 188.1936),
        "mar_dst_bsgme_100m_2015": (3.2670, 4.2283),
        "mar_dst_ghslesaccilcgufghsll_100m_2014": (3.3381, 4.2542),
        "mar_dst_coastline_100m_2000_2020": (698.8104,  326.6751),
        "mar_dmsp_100m_2011": (71.8836 ,  399.8602 ),
        "mar_esaccilc_dst_water_100m_2000_2012":(18.1815, 14.8920), 
        "mar_osm_dst_roadintersec_100m_2016":(19.8096, 29.5883), 
        "mar_osm_dst_waterway_100m_2016": (15.2655, 15.4519), 
        "mar_osm_dst_road_100m_2016": (3.7082,  5.6608), 
        "mar_srtm_slope_100m": (3.2089 , 3.8157),
        "mar_srtm_topo_100m": (1143.9080 , 402.7680),
        "mar_viirs_100m_2015": (0.2205 , 1.0825),
        "mar_wdpa_dst_cat1_100m_2015": (388.8701 , 221.6834),
        'buildings_merge_mean_area': (26.3673, 48.1988),
        'buildings_google_mean_area': (26.3673, 48.1988)
    },
    "mli":{
        "buildings_j": (0.00089380914, 8.41622997e-03),
        # "buildings": (0.265996819, 1.83158563e+00),
        "esaccilc_dst011_100m_2000": (2.81727052, 5.69885715e+00),
        "esaccilc_dst040_100m_2000": (0.44520899, 2.72345595e+00),
        "esaccilc_dst130_100m_2000": (3.09648584, 4.70480562e+00),
        "esaccilc_dst140_100m_2000": (9.18009012, 9.85124076e+00),
        "esaccilc_dst150_100m_2000": (108.52330299, 8.17261502e+01),
        "esaccilc_dst160_100m_2000": (8.65616757, 9.26884634e+00),
        "esaccilc_dst190_100m_2000": (37.38046272, 3.10730075e+01),
        "esaccilc_dst200_100m_2000": (64.73759992, 4.46013049e+01),
        "mli_tt50k_100m_2000": (209.1351, 188.1936),
        "mli_dst_bsgme_100m_2015": (3.2670, 4.2283),
        "mli_dst_ghslesaccilcgufghsll_100m_2014": (3.3381, 4.2542),
        "mli_dst_coastline_100m_2000_2020": (698.8104,  326.6751),
        "mli_dmsp_100m_2011": (71.8836 ,  399.8602 ),
        "mli_esaccilc_dst_water_100m_2000_2012":(18.1815, 14.8920), 
        "mli_osm_dst_roadintersec_100m_2016":(19.8096, 29.5883), 
        "mli_osm_dst_waterway_100m_2016": (15.2655, 15.4519), 
        "mli_osm_dst_road_100m_2016": (3.7082,  5.6608), 
        "mli_srtm_slope_100m": (3.2089 , 3.8157),
        "mli_srtm_topo_100m": (1143.9080 , 402.7680),
        "mli_viirs_100m_2015": (0.2205 , 1.0825),
        "mli_wdpa_dst_cat1_100m_2015": (388.8701 , 221.6834),
        'buildings_merge_mean_area': (26.3673, 48.1988),
        'buildings_google_mean_area': (26.3673, 48.1988)
    },
    "civ":{
        "buildings_j": (0.00089380914, 8.41622997e-03),
        # "buildings": (0.265996819, 1.83158563e+00),
        "esaccilc_dst011_100m_2000": (2.81727052, 5.69885715e+00),
        "esaccilc_dst040_100m_2000": (0.44520899, 2.72345595e+00),
        "esaccilc_dst130_100m_2000": (3.09648584, 4.70480562e+00),
        "esaccilc_dst140_100m_2000": (9.18009012, 9.85124076e+00),
        "esaccilc_dst150_100m_2000": (108.52330299, 8.17261502e+01),
        "esaccilc_dst160_100m_2000": (8.65616757, 9.26884634e+00),
        "esaccilc_dst190_100m_2000": (37.38046272, 3.10730075e+01),
        "esaccilc_dst200_100m_2000": (64.73759992, 4.46013049e+01),
        "civ_tt50k_100m_2000": (209.1351, 188.1936),
        "civ_dst_bsgme_100m_2015": (3.2670, 4.2283),
        "civ_dst_ghslesaccilcgufghsll_100m_2014": (3.3381, 4.2542),
        "civ_dst_coastline_100m_2000_2020": (698.8104,  326.6751),
        "civ_dmsp_100m_2011": (71.8836 ,  399.8602 ),
        "civ_esaccilc_dst_water_100m_2000_2012":(18.1815, 14.8920), 
        "civ_osm_dst_roadintersec_100m_2016":(19.8096, 29.5883), 
        "civ_osm_dst_waterway_100m_2016": (15.2655, 15.4519), 
        "civ_osm_dst_road_100m_2016": (3.7082,  5.6608), 
        "civ_srtm_slope_100m": (3.2089 , 3.8157),
        "civ_srtm_topo_100m": (1143.9080 , 402.7680),
        "civ_viirs_100m_2015": (0.2205 , 1.0825),
        "civ_wdpa_dst_cat1_100m_2015": (388.8701 , 221.6834),
        'buildings_merge_mean_area': (26.3673, 48.1988),
        'buildings_google_mean_area': (26.3673, 48.1988)
    }
}

metadata = {
    "tza": {
        "wp_no_data": [0, 1],
        "wp_covariates_no_data": -9999,
        "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": 'preprocessed_data_3_tza.pkl',
        "rst_wp_regions_path": '{}OtherBuildings/TZA/tza_subnational_2000_2020_sid.tif'.format(root_path)
    },
    "uga":{
        "wp_no_data": [0, 1],
        "wp_covariates_no_data": -9999,
        "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": 'preprocessed_data_3_uga.pkl',
        "rst_wp_regions_path": '{}OtherBuildings/UGA/uga_wpop_regions.tif'.format(root_path)
    },
    "cod":{
        "wp_no_data": [0, 1],
        "wp_covariates_no_data": -9999,
        "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": 'preprocessed_data_3_cod.pkl',
        "rst_wp_regions_path": '{}OtherBuildings/COD/cod_wpop_regions.tif'.format(root_path)
    },
    "rwa":{
        "wp_no_data": [0, 1],
        "wp_covariates_no_data": -9999,
        "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": 'preprocessed_data_3_rwa.pkl',
        "preproc_data_path_Sat2Pop": 'preprocessed_data_3_rwa.pkl',
        # "preproc_data_path_Sat2Pop": 'preprocessed_data_4_Sat2Pop_rwa.pkl',
        "rst_wp_regions_path": '{}OtherBuildings/RWA/rwa_wpop_regions.tif'.format(root_path)
    },
    "nga":{
        "wp_no_data": [0, 1],
        "wp_covariates_no_data": -9999,
        "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": 'preprocessed_data_3_nga.pkl',
        "rst_wp_regions_path": '{}OtherBuildings/NGA/nga_wpop_regions.tif'.format(root_path)
    },
    "moz":{
        "wp_no_data": [0, 1],
        "wp_covariates_no_data": -9999,
        "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": 'preprocessed_data_3_moz.pkl',
        "rst_wp_regions_path": '{}OtherBuildings/MOZ/moz_wpop_regions.tif'.format(root_path)
    },
    "zmb":{
        "wp_no_data": [0],
        "wp_covariates_no_data": -9999,
        "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": 'preprocessed_data_3_zmb.pkl',
        "rst_wp_regions_path": '{}OtherBuildings/ZMB/zmb_adm4_sid.tif'.format(root_path)
    },
    "zaf":{
        "wp_no_data": [0],
        "wp_covariates_no_data": -9999,
        # "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": 'preprocessed_data_3_zaf.pkl',
        "preproc_data_path_Sat2Pop": 'preprocessed_data_4_zaf.pkl',
        "rst_wp_regions_path": '{}OtherBuildings/ZAF/Subnational/zaf_subnational_admin_2000_2020_SID.tif'.format(root_path)
    },
    "dza":{
        "wp_no_data": [0],
        "wp_covariates_no_data": -9999,
        # "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": 'preprocessed_data_3_dza.pkl',
        "rst_wp_regions_path": '{}OtherBuildings/DZA/Subnational/dza_subnational_admin_2000_2020_SID.tif'.format(root_path)
    },
    "mar":{
        "wp_no_data": [0],
        "wp_covariates_no_data": -9999,
        # "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": 'preprocessed_data_3_mar.pkl',
        "rst_wp_regions_path": '{}OtherBuildings/MAR/Subnational/mar_subnational_admin_2000_2020_SID.tif'.format(root_path)
    }, 
    "mli":{
        "wp_no_data": [0],
        "wp_covariates_no_data": -9999,
        # "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": 'preprocessed_data_3_mli.pkl',
        "rst_wp_regions_path": '{}OtherBuildings/MLI/Subnational/mli_subnational_admin_2000_2020_SID.tif'.format(root_path)
    }, 
    "civ":{
        "wp_no_data": [0],
        "wp_covariates_no_data": -9999,
        # "hd_no_data": [0],
        "scale_maxar_to_google": None,
        "preproc_data_path": 'preprocessed_data_3_mli.pkl',
        "rst_wp_regions_path": '{}OtherBuildings/CIV/Subnational/civ_subnational_admin_2000_2020_SID.tif'.format(root_path)
    }
}

input_paths["tza_f2"] = {
    # "buildings": input_paths["tza"]["buildings"],
    "tza_viirs_100m_2016" : "{}Covariates/TZA/VIIRS/tza_viirs_100m_2016.tif".format(root_path)
}

input_paths["tza_f3"] = {
    "buildings_google": input_paths["tza"]["buildings_google"],
    "buildings_maxar": input_paths["tza"]["buildings_maxar"],
    "buildings_google_mean_area": input_paths["tza"]["buildings_google_mean_area"],
    "buildings_maxar_mean_area": input_paths["tza"]["buildings_maxar_mean_area"],
    "tza_viirs_100m_2015" : input_paths["tza"]["tza_viirs_100m_2015"]
}

metadata["tza_f2"] = metadata["tza"]
metadata["tza_f3"] = metadata["tza"]
# metadata["uga"] = metadata["tza"]

# Columns of shapefiles
col_coarse_level_seq_id = "GR_SID"
col_finest_level_seq_id = "SID"
