import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from distutils.util import strtobool


def show_feat_importance(feat_importance_path, metric, country_code, output_path, num_feats_to_show, 
                         adjusted, include_building_counts):
    
    with open(feat_importance_path, 'rb') as handle:
        data = pickle.load(handle)
    
    metric_orig_val = data["{}/{}".format(country_code, metric)]
    if adjusted == 1:
        metric_orig_val = data["{}/adjusted/{}".format(country_code, metric)]

    feat_names = data['feat_importance']['feature_names']
    
    scores = []
    offset_idx = 1
    if include_building_counts == 1:
        offset_idx = 0
    for i in range(offset_idx, len(feat_names)):   
        feat_name = feat_names[i]
        
        feat_importance = data['feat_importance']['not_adj'][feat_name]
        if adjusted == 1:
            feat_importance = data['feat_importance']['adj'][feat_name]
        
        score_importance = feat_importance['importance_score']
        if metric in ["mape", "mae"]:
            score_importance *= -1
        
        score_values = np.array(feat_importance['array_metrics'])
        avg = np.mean(score_values)
        stdev = np.std(score_values)
        print("feat {}, score: avg {} std {}".format(feat_name, avg, stdev))
        
        scores.append(score_importance)
    
    scores = np.array(scores)
    sorted_idxs = np.argsort(scores)
    sorted_idxs = sorted_idxs[::-1]
    
    print("---------- Feature importance based on {}".format(metric))
    for idx in sorted_idxs:
        print(scores[idx], feat_names[idx+offset_idx])
    
    # Labels for the features
    labels = {
        "buildings_merge" : "Building count",
        "buildings_merge_mean_area" : "Building mean area",
        "{}_tt50k_100m_2000".format(country_code) : "Travel time to cities",
        "{}_dst_bsgme_100m_2015".format(country_code) : "Dst. to BSGM buildings",
        "{}_dst_ghslesaccilcgufghsll_100m_2014".format(country_code) : "Dst. to GHSL buildings",
        "{}_dst_coastline_100m_2000_2020".format(country_code) : "Dst. to coastline",
        "{}_dmsp_100m_2011".format(country_code) : "Night time light DMSP",
        "{}_esaccilc_dst_water_100m_2000_2012".format(country_code) : "Dst. to water ESA",
        "{}_osm_dst_roadintersec_100m_2016".format(country_code) : "Dst. to road intersection",
        "{}_osm_dst_waterway_100m_2016".format(country_code): "Dst. to waterway OSM",
        "{}_osm_dst_road_100m_2016".format(country_code) : "Dst. to road",
        "{}_srtm_slope_100m".format(country_code) : "Slope",
        "{}_srtm_topo_100m".format(country_code) : "Elevation",
        "{}_viirs_100m_2015".format(country_code) : "Night time light VIIRS",
        "{}_wdpa_dst_cat1_100m_2015".format(country_code) : "Dst. to protected areas"
    }
    # Save feature importance figure
    sorted_feat_names = [labels[feat_names[idx+offset_idx]] for idx in sorted_idxs[::-1]]
    sorted_scores = [scores[idx] for idx in sorted_idxs[::-1]]
    sorted_feat_names = sorted_feat_names[-num_feats_to_show:]
    sorted_scores = sorted_scores[-num_feats_to_show:]
    fig = plt.figure(figsize=(6, 5))
    plt.barh(sorted_feat_names, sorted_scores, color='gray')
    if metric in ["mae", "mape"]:
        plt.xlabel("{} increase when permuting feature values".format(metric.upper()))
    else:
        plt.xlabel("{} decrease when permuting feature values".format(metric.upper()))
    plt.title("Feature importance")
    plt.savefig(output_path, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feat_importance_path", type=str, help="Preprocessed data of feature importance (pickle file)")
    parser.add_argument("metric", type=str, help="metric (e.g., r2, mae, mape)")
    parser.add_argument("country_code", type=str, help="country code")
    parser.add_argument("output_path", type=str, help="Feature importance figure path")
    parser.add_argument("num_feats_to_show", type=int, help="Num. features to show")
    parser.add_argument("adjusted", type=lambda x: bool(strtobool(x)), help="Adjusted: True, Not-adjusted: False")
    parser.add_argument("include_building_counts", type=lambda x: bool(strtobool(x)), help="Include: True, Not-include: False")
    args = parser.parse_args()

    show_feat_importance(args.feat_importance_path, args.metric, args.country_code, args.output_path, args.num_feats_to_show, 
                         args.adjusted, args.include_building_counts)


if __name__ == "__main__":
    main()
