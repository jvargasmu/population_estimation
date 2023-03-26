import argparse
import numpy as np
from osgeo import gdal
from matplotlib import cm


def compute_percentiles_from_popmap(pop_map_path):
    # Read geo tif image
    pop_map = gdal.Open(pop_map_path).ReadAsArray().astype(np.float32)
    print(f"max_val {np.max(pop_map[~np.isnan(pop_map)])}")
    # Compute the "large_cap_val": values higher than this are considered outliers
    # and "cap_val" is the minimum value of the last interval
    cap_val_orig_float = np.percentile(pop_map[~np.isnan(pop_map)].flatten(), 99.99)
    large_cap_val =  int(np.percentile(pop_map[~np.isnan(pop_map)].flatten(), 99.9999))
    cap_val = (int(cap_val_orig_float) // 50) * 50
    print(f"cap_val_orig_float {cap_val_orig_float} cap_val: {cap_val}, large_cap_val {large_cap_val}")

    # create invervals and colors
    max_size_intervals = 8
    intervals = []
    int_div = 10
    if cap_val > 200:
        intervals += [1, 25, 50] 
        int_div = 50
    elif cap_val > 100:
        intervals += [1, 5, 25, 50]
        int_div = 25
    else:
        intervals += [1, 5]

    last_fixed_val = intervals[-1]
    range_to_be_filled = cap_val - last_fixed_val
    num_intervals_to_be_filled = max_size_intervals - len(intervals) - 2
    interval_size = range_to_be_filled / (num_intervals_to_be_filled + 1.0)

    for i in range(num_intervals_to_be_filled):
        interval_val = last_fixed_val + ((i+1)*interval_size)
        intervals.append((int(interval_val) // int_div) * int_div)
    
    intervals += [cap_val, large_cap_val]
    print(f"intervals {intervals}")

    # Get colors
    viridis = cm.get_cmap('viridis', 256)
    interval_colors = []
    for i in range(len(intervals) - 1):
        viridis_index = int((((intervals[i] - intervals[0]) / (cap_val - intervals[0]))*255) + 0.5)
        color_dec = viridis(viridis_index)
        color_hex = '#%02x%02x%02x' %  tuple((np.array(color_dec[:3])*255).astype(int).tolist())
        interval_colors.append(color_hex)
    
    print(f"reversed interval_colors {interval_colors[::-1]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pop_map_path", type=str, default="", help="Pop map path")
    args = parser.parse_args()

    compute_percentiles_from_popmap(args.pop_map_path)


if __name__ == "__main__":
    main()
