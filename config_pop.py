
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
        # "buildings_j": "{}OtherBuildings/TZA/tza_gbuildings.tif".format(root_path),
        "buildings_google": "{}OtherBuildings/TZA/TZA_gbp_BCB_v1_count.tif".format(root_path),
        "buildings_maxar": "{}OtherBuildings/TZA/TZA_mbp_BCB_v3_count.tif".format(root_path),
        "buildings_google_mean_area": "{}OtherBuildings/TZA/TZA_gbp_BCB_v1_mean_area.tif".format(root_path),
        "buildings_maxar_mean_area": "{}OtherBuildings/TZA/TZA_mbp_BCB_v3_mean_area.tif".format(root_path),
        # "esaccilc_dst011_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst011_100m_2015.tif".format(
        #     root_path),
        # "esaccilc_dst040_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst040_100m_2015.tif".format(
        #     root_path),
        # "esaccilc_dst130_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst130_100m_2015.tif".format(
        #     root_path),
        # "esaccilc_dst140_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst140_100m_2015.tif".format(
        #     root_path),
        # "esaccilc_dst150_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst150_100m_2015.tif".format(
        #     root_path),
        # "esaccilc_dst160_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst160_100m_2015.tif".format(
        #     root_path),
        # "esaccilc_dst190_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst190_100m_2015.tif".format(
        #     root_path), # Urban Areas
        # "esaccilc_dst200_100m_2000": "{}Covariates/TZA/ESA_CCI_Annual/2015/tza_esaccilc_dst200_100m_2015.tif".format(
        #     root_path), 
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

        # "buildings_j": (0.00089380914, 8.41622997e-03),
        # # "buildings": (0.265996819, 1.83158563e+00),
        # "esaccilc_dst011_100m_2000": (2.81727052, 5.69885715e+00),
        # "esaccilc_dst040_100m_2000": (0.44520899, 2.72345595e+00),
        # "esaccilc_dst130_100m_2000": (3.09648584, 4.70480562e+00),
        # "esaccilc_dst140_100m_2000": (9.18009012, 9.85124076e+00),
        # "esaccilc_dst150_100m_2000": (108.52330299, 8.17261502e+01),
        # "esaccilc_dst160_100m_2000": (8.65616757, 9.26884634e+00),
        # "esaccilc_dst190_100m_2000": (37.38046272, 3.10730075e+01),
        # "esaccilc_dst200_100m_2000": (64.73759992, 4.46013049e+01),
        # "tza_tt50k_100m_2000": (3.76670969e+02, 3.24293088e+02),
        # "tza_dst_bsgme_100m_2015": (1.00138409e+01, 1.08199418e+01),
        # "tza_dst_ghslesaccilcgufghsll_100m_2014": (1.01325672e+01, 10.82786076),
        # "tza_dst_coastline_100m_2000_2020": (4.69711371e+02,  269.56063334),
        # "tza_dmsp_100m_2011": (9.96103462e+00 ,  133.46285072 ),
        # "tza_esaccilc_dst_water_100m_2000_2012":(2.27107890e+01, 19.28008101), 
        # "tza_osm_dst_roadintersec_100m_2016":(45.36450679, 43.17709196), 
        # "tza_osm_dst_waterway_100m_2016": (2.06357978e+01, 21.3063267),
        # "tza_osm_dst_road_100m_2016": (12.8190051,  15.32399314),
        # "tza_srtm_slope_100m": (3.44179414e+00 , 4.7920747),
        # "tza_srtm_topo_100m": (1.02034711e+03 , 483.97733166),
        # "tza_viirs_100m_2015": (9.87479157e-02 , 0.49287229),
        # "tza_wdpa_dst_cat1_100m_2015": (2.37557718e+02 , 122.54531703),
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

        # "buildings_j": (0.00089380914, 8.41622997e-03),
        # # "buildings": (0.265996819, 1.83158563e+00),
        # "esaccilc_dst011_100m_2000": (2.81727052, 5.69885715e+00),
        # "esaccilc_dst040_100m_2000": (0.44520899, 2.72345595e+00),
        # "esaccilc_dst130_100m_2000": (3.09648584, 4.70480562e+00),
        # "esaccilc_dst140_100m_2000": (9.18009012, 9.85124076e+00),
        # "esaccilc_dst150_100m_2000": (108.52330299, 8.17261502e+01),
        # "esaccilc_dst160_100m_2000": (8.65616757, 9.26884634e+00),
        # "esaccilc_dst190_100m_2000": (37.38046272, 3.10730075e+01),
        # "esaccilc_dst200_100m_2000": (64.73759992, 4.46013049e+01),
        # "uga_tt50k_100m_2000": (3.76670969e+02, 3.24293088e+02),
        # "uga_dst_bsgme_100m_2015": (1.00138409e+01, 1.08199418e+01),
        # "uga_dst_ghslesaccilcgufghsll_100m_2014": (1.01325672e+01, 10.82786076),
        # "uga_dst_coastline_100m_2000_2020": (4.69711371e+02,  269.56063334),
        # "uga_dmsp_100m_2011": (9.96103462e+00 ,  133.46285072 ),
        # "uga_esaccilc_dst_water_100m_2000_2012":(2.27107890e+01, 19.28008101), 
        # "uga_osm_dst_roadintersec_100m_2016":(45.36450679, 43.17709196), 
        # "uga_osm_dst_waterway_100m_2016": (2.06357978e+01, 21.3063267),
        # "uga_osm_dst_road_100m_2016": (12.8190051,  15.32399314),
        # "uga_srtm_slope_100m": (3.44179414e+00 , 4.7920747),
        # "uga_srtm_topo_100m": (1.02034711e+03 , 483.97733166),
        # "uga_viirs_100m_2015": (9.87479157e-02 , 0.49287229),
        # "uga_wdpa_dst_cat1_100m_2015": (2.37557718e+02 , 122.54531703),
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
        "rst_wp_regions_path": '{}OtherBuildings/RWA/rwa_wpop_regions.tif'.format(root_path)
    }
}

input_paths["tza_f2"] = {
    # "buildings": input_paths["tza"]["buildings"],
    "tza_viirs_100m_2016" : "{}Covariates/TZA/VIIRS/tza_viirs_100m_2016.tif".format(root_path)
}

metadata["tza_f2"] = metadata["tza"]
# metadata["uga"] = metadata["tza"]

# Columns of shapefiles
col_coarse_level_seq_id = "GR_SID"
col_finest_level_seq_id = "SID"
