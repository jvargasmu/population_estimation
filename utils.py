import fiona
from osgeo import gdal
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.utils import check_array
from sklearn.model_selection import KFold
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
import matplotlib.ticker as ticker
from tqdm import tqdm
import copy
from pylab import figure, imshow, matshow, grid, savefig
import torch
import pickle
import h5py
import wandb
import psutil
import os
import pdb
import config_pop as cfg

def get_properties_dict(data_dict_orig):
    data_dict = []
    for data_row in data_dict_orig:
        data_dict.append(data_row["properties"])
    return data_dict


def read_input_raster_data_to_np(input_paths, keys=None):
    #assuming every covariate has same dimensions
    first_name = list(input_paths.keys())[0]
    hwdims = gdal.Open(input_paths[first_name]).ReadAsArray().astype(np.float32).shape
    fdim = input_paths.__len__()
    inputs = np.zeros((fdim,) + hwdims, dtype=np.float32) 
    for i,kinp in enumerate(input_paths.keys()):
        print("read {}".format(input_paths[kinp]))
        inputs[i] = gdal.Open(input_paths[kinp]).ReadAsArray().astype(np.float32)
    return inputs


def read_input_raster_data_to_np_buildings(input_paths, keys=None):
    #assuming every covariate has same dimensions
    first_name = list(input_paths.keys())[0]
    hwdims = gdal.Open(input_paths[first_name]).ReadAsArray().astype(np.float32).shape
    fdim = input_paths.__len__()
    inputs = np.zeros((fdim,) + hwdims, dtype=np.float32) 
    for i,kinp in enumerate(input_paths.keys()):
        if ("buildings_google" in kinp) or ("buildings_maxar" in kinp):
            print("read {}".format(input_paths[kinp]))
            inputs[i] = gdal.Open(input_paths[kinp]).ReadAsArray().astype(np.float32)
    return inputs



def read_input_raster_data(input_paths):
    inputs = {}
    for kinp in input_paths.keys():
        inputs[kinp] = gdal.Open(input_paths[kinp]).ReadAsArray().astype(np.float32)
    
    for suffix in ["", "_mean_area"]:
        buildings_feat = "buildings{}".format(suffix)
        buildings_google_feat = "buildings_google{}".format(suffix)
        buildings_maxar_feat = "buildings_maxar{}".format(suffix)
        if buildings_feat not in inputs.keys():
            
            if (buildings_google_feat in inputs.keys()) and (buildings_maxar_feat in inputs.keys()):
                inputs[buildings_feat] = np.maximum(inputs[buildings_google_feat], inputs[buildings_maxar_feat])
                del inputs[buildings_google_feat]
                del inputs[buildings_maxar_feat]
            
            elif buildings_google_feat in inputs.keys():
                inputs[buildings_feat] = inputs[buildings_google_feat]
                del inputs[buildings_google_feat]
            
            elif buildings_maxar_feat in inputs.keys():
                inputs[buildings_feat] = inputs[buildings_maxar_feat]
                del inputs[buildings_maxar_feat]
    
    orig_input_keys = list(inputs.keys())
    new_list_of_keys = ["buildings", "buildings_mean_area"] + orig_input_keys[:-2]
    inputs = {k:inputs[k] for k in new_list_of_keys}
    
    return inputs


def read_shape_layer_data(shape_layer_path):
    with fiona.open(shape_layer_path) as reader:
        layer_data_orig = [elem for elem in reader]
    layer_data = get_properties_dict(layer_data_orig)
    return layer_data

def mean_absolute_percentage_error(y_true, y_pred): 

    y_true = check_array(y_true.reshape(-1,1))
    y_pred = check_array(y_pred.reshape(-1,1))
    
    zeromask = (y_true!=0)
    y_true, y_pred = y_true[zeromask], y_pred[zeromask]  

    percentage_error = (y_true - y_pred) / y_true

    return np.mean(np.abs(percentage_error)) * 100, percentage_error * 100


def my_mean_absolute_error(y_pred,y_true):
    y_pred = y_pred.astype(np.float32)
    y_true = y_true.astype(np.float32)
    
    errors = y_pred - y_true
    output_errors = np.average(np.abs(errors), axis=0)
    return output_errors, errors

def density_scatter( x , y, ax = None, sort = True, bins = 20, millions = True, cmap='inferno', ** kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    @ticker.FuncFormatter
    def million_formatter(x, pos):
        return "%.1fM" % (x/1E6)
    @ticker.FuncFormatter
    def tousends_formatter(x, pos):
        return "%.0fk" % (x/1E3)

    def add_identity(axes, *line_args, **line_kwargs):
        identity, = axes.plot([], [], *line_args, **line_kwargs)
        def callback(axes):
            low_x, high_x = axes.get_xlim()
            low_y, high_y = axes.get_ylim()
            low = max(low_x, low_y)
            high = min(high_x, high_y)
            identity.set_data([low, high], [low, high])
        callback(axes)
        axes.callbacks.connect('xlim_changed', callback)
        axes.callbacks.connect('ylim_changed', callback)
        return axes
        
    if ax is None :
        fig , ax = plt.subplots(dpi=200)
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = False )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, cmap=cmap, **kwargs )
    if millions:
        ax.xaxis.set_major_formatter(tousends_formatter)
        ax.yaxis.set_major_formatter(tousends_formatter) 
    plt.xticks(rotation='45') 
    plt.xlabel("Census Data")
    plt.ylabel("Prediced Census Data") 

    maxi = np.max([x.max(), y.max()])
    plt.xlim([0, maxi])
    plt.ylim([0, maxi])

    norm = Normalize(vmin = np.min(z), vmax = int(np.max(z)))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm, cmap=cmap), ax=ax)
    # cbar = fig.colorbar(ax=ax)
    cbar.ax.set_ylabel('#Samples')
    add_identity(ax, color='r', ls='--', alpha=0.5)

    return fig, ax

def pop_colorbar(min=0, max=465, cmap="viridis", ylabel='[Population/ha]', fontsize=10):
    fig , ax = plt.subplots(dpi=400)
    norm = Normalize(vmin=min, vmax=max)
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm, cmap=cmap), ax=ax) 
    cbar.ax.set_ylabel('[Population/ha]', fontsize=fontsize)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    fig.show()
    plt.savefig('foo.png', bbox_inches = "tight")


def compute_performance_metrics_arrays(preds, gt): 
    
    metrics = {}

    preds = np.squeeze(preds)
    gt = np.squeeze(gt)

    if len(preds.shape)==2:
        # bayes
        # preds, vars = np.split(preds,[1], axis=1)
        stds = np.sqrt(preds[:,1])
        preds = preds[:,0]

        metrics.update({
            "aux/stds/histogram_std": stds, "aux/stds/min_stds": np.min(stds),
            "aux/stds/max_stds":  np.max(stds), "aux/stds/median_stds": np.median(stds),
            "aux/stds/mean_stds":  np.mean(stds), "aux/stds/std_stds": np.std(stds)
        })
    
    r2 = r2_score(gt, preds)
    
    mae, errors = my_mean_absolute_error(gt, preds)
    mse = mean_squared_error(gt, preds)
    mape, percentage_error = mean_absolute_percentage_error(gt,preds)

    # fig, ax = density_scatter(gt,preds, bins=150, s=20, cmap="cividis")
    # fig.show()
    # plt.savefig('foo.png', bbox_inches = "tight")

    # pop_colorbar(0,465)

    metrics.update({
    "r2": r2, "mae": mae, "mse": mse, "mape": mape,
    "aux/errors/errors": errors, "aux/errors/min_errors": np.min(errors), "aux/errors/max_errors":  np.max(errors), "aux/errors/median_error":  np.median(errors), "aux/errors/mean_error":  np.mean(errors), "aux/errors/std_error":  np.std(errors), 
    # "aux/errors/abs/abs_errors": np.abs(errors), "aux/errors/abs/min_abs_errors": np.min(np.abs(errors)), "aux/errors/abs/max_abs_error":  np.max(np.abs(errors)), "aux/errors/abs/median_abs_error":  np.median(np.abs(errors)), "aux/errors/abs/mean_abs_error": np.mean(np.abs(errors)), "aux/errors/abs/std_abs_error": np.std(np.abs(errors)),
    "aux/errors_percentage/percentage_errors": percentage_error, "aux/errors_percentage/min_percentage_errors": np.min(percentage_error), "aux/errors_percentage/max_percentage_error":  np.max(percentage_error), "aux/errors_percentage/median_percentage_error":  np.median(percentage_error), "aux/errors_percentage/mean_percentage_error":  np.mean(percentage_error), "aux/errors_percentage/std_percentage_error": np.std(percentage_error),
    # "aux/errors_percentage/abs/abs_percentage_errors": np.abs(percentage_error), "aux/errors_percentage/abs/min_abs_percentage_errors": np.min(np.abs(percentage_error)), "aux/errors_percentage/abs/max_abs_percentage_errors":  np.max(np.abs(percentage_error)), "aux/errors_percentage/abs/median_abs_percentage_error":  np.median(np.abs(percentage_error)), "aux/errors_percentage/abs/mean_abs_percentage_error":  np.mean(np.abs(percentage_error)), "aux/errors_percentage/abs/std_abs_percentage_error":  np.std(np.abs(percentage_error))
    #"scatterplot": [fig,ax]
    })

    return metrics


def compute_performance_metrics(preds_dict, gt_dict):
    assert len(preds_dict) == len(gt_dict)

    preds = []
    gt = []
    ids = preds_dict.keys()
    for id in ids:
        preds.append(preds_dict[id])
        gt.append(gt_dict[id])

    preds = np.array(preds).astype(np.float)
    gt = np.array(gt).astype(np.float)

    return compute_performance_metrics_arrays(preds, gt)


def write_geolocated_image(image, output_path, src_geo_transform, src_projection):
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_path, image.shape[1], image.shape[0], 1, gdal.GDT_Float32, options=['COMPRESS=LZW'])
    outdata.SetGeoTransform(src_geo_transform)
    outdata.SetProjection(src_projection)
    outdata.GetRasterBand(1).WriteArray(image)
    outdata.FlushCache()
    outdata = None
    ds = None


def convert_str_to_int_keys(data_dict_orig):
    data_dict = {}
    for k in data_dict_orig.keys():
        data_dict[int(k)] = data_dict_orig[k]
    return data_dict


def convert_dict_vals_str_to_float(data_dict_orig):
    return {k: float(data_dict_orig[k]) for k in data_dict_orig.keys()}


def preprocess_census_targets(data_dict_orig):
    data_dict = convert_str_to_int_keys(data_dict_orig)
    data_dict = convert_dict_vals_str_to_float(data_dict)
    return data_dict


def create_map_of_valid_ids(regions, no_valid_ids):
    map_valid_ids = np.ones(regions.shape).astype(np.uint32)
    for id in no_valid_ids:
        map_valid_ids[regions == id] = 0
    return map_valid_ids


def create_valid_mask_array(num_ids, valid_ids):
    valid_ids_mask = np.zeros(num_ids)
    for id in valid_ids:
        valid_ids_mask[id] = 1
    return valid_ids_mask


def compute_grouped_values(data, valid_ids, id_to_gid):
    # Initialize target values
    grouped_data = {}
    for id in valid_ids:
        gid = id_to_gid[id]
        if gid not in grouped_data.keys():
            grouped_data[gid] = 0
    # Aggregate targets
    for id in valid_ids:
        gid = id_to_gid[id]
        grouped_data[gid] += data[id]
    return grouped_data


def transform_dict_to_array(data_dict):
    return np.array([data_dict[k] for k in data_dict.keys()]).astype(np.float32)


def transform_dict_to_matrix(data_dict):
    assert len(data_dict.keys()) > 0
    # get size of matrix
    keys = list(data_dict.keys())
    num_rows = len(keys)
    first_row = data_dict[keys[0]]
    col_keys = list(first_row.keys())
    num_cols = len(col_keys)
    # fill matrix
    data_array = np.zeros((num_rows, num_cols)).astype(np.float32)
    for i, rk in enumerate(keys):
        for j, ck in enumerate(col_keys):
            data_array[i, j] = data_dict[rk][ck]

    return data_array


def compute_features_from_raw_inputs(inputs, feats_list):
    inputs_mat = []
    for feat in feats_list:
        inputs_mat.append(inputs[feat])
    inputs_mat = np.array(inputs_mat)
    all_features = inputs_mat.reshape((inputs_mat.shape[0], -1))
    all_features = all_features.transpose()
    return all_features


def mostly_non_empty_map(map_valid_ids, feats_list, inputs, threshold = 0.99, min_val = 0.001):
    map_empty_feats = np.random.rand(map_valid_ids.shape[0], map_valid_ids.shape[1]) < threshold
    for k in feats_list:
        min_threshold = 0
        max_threshold = 1000.0
        for k in inputs.keys():
            inputs[k][inputs[k] > max_threshold] = 0
            inputs[k][inputs[k] < min_threshold] = 0
        map_empty_feats = np.multiply(map_empty_feats, inputs[k] <= min_val)

    mostly_non_empty = (1 - map_empty_feats).astype(np.bool)
    return mostly_non_empty


def calculate_densities(census, area, map=None):
    density = {}
    for key, value in census.items():
        density[key] = value / area[key]
    if map is None:
        return density

    #write into map
    # making sure that all the values are contained in the 
    diffkey = set(area.keys()) - set(census.keys())
    mapping = copy.deepcopy(density)
    for key in diffkey:
        mapping[key] = 0

    #vectorized mapping of the integer keys (assumes keys are integers, and not excessively large compared to the length of the dicct)
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))
    mapping_ar = np.zeros(k.max()+1,dtype=v.dtype) #k,v from approach #1
    mapping_ar[k] = v
    density_map = mapping_ar[map] 
    return density, density_map
    
    
def plot_2dmatrix(matrix,fig=1):
    if torch.is_tensor(matrix):
        if matrix.is_cuda:
            matrix = matrix.cpu()
        matrix = matrix.numpy()
    figure(fig)
    matshow(matrix, interpolation='nearest')
    grid(True)
    savefig('outputs/last_plot.png')


def accumulate_values_by_region(map, ids, regions):
    sums = {}
    for id in tqdm(ids):
        sums[id]= map[regions==id].sum()
    return sums


def bbox2(img):
    rows = torch.any(img, axis=1)
    cols = torch.any(img, axis=0)
    rmin, rmax = torch.where(rows)[0][[0, -1]]
    cmin, cmax = torch.where(cols)[0][[0, -1]]

    return rmin, rmax+1, cmin, cmax+1


class PatchDataset(torch.utils.data.Dataset):
    """Patch dataset."""
    def __init__(self, rawsets, memory_mode, device, validation_split): 
        self.device = device
        
        print("Preparing dataloader for: ", list(rawsets.keys()))
        self.loc_list = []
        self.BBox = {}
        self.features = {}
        self.Ys = {}
        self.Masks = {}
        for i, (name, rs)  in tqdm(enumerate(rawsets.items())):

            with open(rs['vars'], "rb") as f: 
                tr_census, tr_regions, tr_valid_data_mask, tY, tMasks, tBBox = pickle.load(f)

            self.BBox[name] = tBBox
            if memory_mode:
                self.features[name] = h5py.File(rs["features"], 'r')["features"][:]
            else:
                self.features[name] = h5py.File(rs["features"], 'r')["features"]
            self.Ys[name] =  tY  
            self.Masks[name] = tMasks
            self.loc_list.extend( [(name, k) for k,_ in enumerate(tBBox)])

        self.dims = self.features[name].shape[1]
        
    def __len__(self):
        return len(self.variables[0])

    def getsingleitem(self, idx):
        output = []
        name, k = self.idx_to_loc(idx)
        rmin, rmax, cmin, cmax = self.BBox[name][k]
        X = torch.from_numpy(self.features[name][:,:,rmin:rmax, cmin:cmax])
        Y = torch.from_numpy(self.Ys[name][k]) 
        Mask = torch.from_numpy(self.Masks[name][k]) 
        return X, Y, Mask

    def __getitem__(self, idx):
        return self.getsingleitem(idx)

class MultiPatchDataset(torch.utils.data.Dataset):
    """Patch dataset."""
    def __init__(self, datalocations, train_dataset_name, train_level, memory_mode, device,
        validation_split, validation_fold, loss_weights, sampler_weights, val_valid_ids={}, build_pairs=True, random_seed_folds=1610,
        index_permutation_feat=None, permutation_random_seed=42, remove_feat_idxs=None):

        self.device = device    
        print("Preparing dataloader for: ", list(datalocations.keys()))
        self.features = {}
        self.loc_list, self.loc_list_train, self.loc_list_val = [],[],[]
        self.loc_list_hout = []
        self.all_weights, self.all_sampler_weights,  self.all_natural_weights = [],[],[]
        self.BBox, self.BBox_train, self.BBox_val, self.BBox_hout = {},{},{},{}
        self.Ys, self.Ys_train, self.Ys_val, self.Ys_hout = {},{},{},{} 
        self.tregid, self.max_tregid = {},{}
        self.tregid_val, self.max_tregid_val = {},{}
        self.tregid_hout, self.max_tregid_hout = {},{}
        self.Masks, self.Masks_train, self.Masks_val, self.Masks_hout = {},{},{},{}
        self.regMasks, self.regMasks_train, self.regMasks_val, self.regMasks_hout = {},{},{},{}
        self.weight_list = {}
        self.memory_disag, self.memory_disag_val, self.feature_names = {},{},{}
        self.memory_disag_hout = {}
        self.val_valid_ids = val_valid_ids
        self.memory_vars = {}
        self.source_census_val = {}
        self.source_census_hout = {}
        process = psutil.Process(os.getpid())
        
        for i, (name, rs) in tqdm(enumerate(datalocations.items())):
            print("Preparing dataloader: ", name)
            print("Initial:",process.memory_info().rss/1000/1000,"mb used")
            
            # get map of valid ids
            rst_wp_regions_path = cfg.metadata[name]["rst_wp_regions_path"]
            preproc_data_path = cfg.metadata[name]["preproc_data_path"]
            fine_regions = gdal.Open(rst_wp_regions_path).ReadAsArray().astype(np.uint32)
            with open(preproc_data_path, 'rb') as handle:
                pdata = pickle.load(handle)
            no_valid_ids = pdata["no_valid_ids"]
            map_valid_ids = create_map_of_valid_ids(fine_regions, no_valid_ids)

            with open(rs['train_vars_f'], "rb") as f:
                _, _, _, tY_f, tregid_f, tMasks_f, tregMasks_f, tBBox_f, _ = pickle.load(f)
            with open(rs['train_vars_c'], "rb") as f:
                _, _, _, tY_c, tregid_c, tMasks_c, tregMasks_c, tBBox_c, feature_names = pickle.load(f)

            self.feature_names[name] = feature_names
            # print("After loading trainvars",process.memory_info().rss/1000/1000,"mb used")

            if name not in self.val_valid_ids.keys():          
                with open(rs['eval_vars'], "rb") as f:
                    self.memory_vars[name] = pickle.load(f)
                    self.val_valid_ids[name] = self.memory_vars[name][4]
            # print("After loading of eval memory vars",process.memory_info().rss/1000/1000,"mb used")
            with open(rs['disag'], "rb") as f:
                self.memory_disag[name] = pickle.load(f) 

            # print("After loading of disag memory",process.memory_info().rss/1000/1000,"mb used")

            if memory_mode[i]=='m':
                #self.features[name] = h5py.File(rs["features"], 'r', driver='core')["features"]
                features = h5py.File(rs["features"], 'r')["features"][:]
                if index_permutation_feat is not None:
                    print("read file and permute feature : {}".format(self.feature_names[name][index_permutation_feat]))
                    num_images = features.shape[0]
                    num_channels = features.shape[1]
                    height = features.shape[2]
                    width = features.shape[3]
                    features = features.reshape(num_images, num_channels, height * width)
                    valid_features = features[:, :, map_valid_ids.flatten() == 1]
                    num_valid_samples =  valid_features.shape[2]
                    np.random.seed(permutation_random_seed)
                    permutation_indexes = np.arange(num_valid_samples)
                    np.random.shuffle(permutation_indexes)
                    valid_features[:, index_permutation_feat, :] = valid_features[:, index_permutation_feat, permutation_indexes]
                    features[:,:,map_valid_ids.flatten() == 1] = valid_features
                    features = features.reshape(num_images, num_channels, height, width)
                    del valid_features
                
                if remove_feat_idxs is not None:
                    new_features = []
                    for idx in range(len(feature_names)):
                        if idx not in remove_feat_idxs:
                            new_features.append(features[:, idx, :, :])
                    features = np.concatenate(new_features)
                    features = np.expand_dims(features, axis=0)
                
                self.features[name] = features     

            elif memory_mode[i]=='d':
                self.features[name] = h5py.File(rs["features"], 'r')["features"]
            else:
                raise Exception(f"Wrong memory mode for {name}. It should be 'd' or 'm' in a comma separated list. No spaces!")
            # print("After loading of features",process.memory_info().rss/1000/1000,"mb used")
            
            # Validation split strategy:
            # We always split the coarse patches into 5 folds, then we look up fine patches that belong to those coarse validation patches
            np.random.seed(random_seed_folds)
            if validation_fold is not None:
                trainidxs, validxs, houtidxs = [],[],[]
                n_samples = len(tY_c)
                n_splits = 5
                for spl in range(n_splits):
                    orig_indices = np.arange(n_samples)
                    np.random.shuffle(orig_indices)
                    idx_offset = n_samples
                    indices = np.concatenate((orig_indices, orig_indices, orig_indices))

                    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
                    fold_sizes[: n_samples % n_splits] += 1
                    current = 0
                    for fold_size in fold_sizes:
                        val_start, val_stop = current, current + fold_size
                        hout_start, hout_stop = current - fold_size, current
                        train_start, train_stop = current + fold_size, current + fold_size * (n_splits - 2)
                        
                        trainidxs.append(indices[idx_offset+train_start:idx_offset+train_stop])
                        validxs.append(indices[idx_offset+val_start:idx_offset+val_stop])
                        houtidxs.append(indices[idx_offset+hout_start:idx_offset+hout_stop])
                        
                        current = val_stop
                
                choice_val_c = validxs[validation_fold]
                choice_hout_c = houtidxs[validation_fold]
            else:
                n_samples = len(tY_c)
                split_int =int(len(tY_c)*validation_split)
                orig_indices = np.arange(n_samples)
                np.random.shuffle(orig_indices)
                choice_val_c = orig_indices[:split_int]
                choice_hout_c = np.array([], dtype=np.int64)
                # no_holdout at this point, there is no case where we need a custom split and a non-zero length holdout
                # if validation_split > 0.0 and holdout:
                #     choice_hout_c = orig_indices[-split_int:]
            
            ind_val_hout_c = np.zeros(len(tY_c), dtype=bool)
            ind_val_hout_c[choice_val_c] = True 
            # For the "ac" option the holdout can still be used in the training data
            ind_val_hout_c[choice_hout_c] = True 
            ind_train_c = ~ind_val_hout_c 

            ind_val_c =  np.zeros(len(tY_c), dtype=bool)
            ind_val_c[choice_val_c] = True 

            tY_f = np.asarray(tY_f)
            tY_c = np.asarray(tY_c)
            tMasks_f = np.asarray(tMasks_f, dtype=object)
            tMasks_c = np.asarray(tMasks_c, dtype=object)
            tregMasks_f = np.asarray(tregMasks_f, dtype=object)
            tregMasks_c = np.asarray(tregMasks_c, dtype=object)
            tBBox_f = np.asarray(tBBox_f)
            tBBox_c = np.asarray(tBBox_c)
            tregid_f = np.asarray(tregid_f).astype(np.int16)
            tregid_c = np.asarray(tregid_c).astype(np.int16)

            tregid_val_c = tregid_c[choice_val_c]
            tregid_hout_c = tregid_c[choice_hout_c]

            # Prepare validation variables
            # If we took the coarse level as training, we need to translate the ind_val to the fine level and get the fine level patches for validation!
            choice_val_f = np.where(np.in1d(self.memory_disag[name][0],tregid_val_c)[self.val_valid_ids[name]])[0] 
            ind_val_f = np.zeros(len(tY_f), dtype=bool)
            ind_val_f[choice_val_f] = True 
            
            choice_hout_f = np.where(np.in1d(self.memory_disag[name][0],tregid_hout_c)[self.val_valid_ids[name]])[0] 
            ind_hout_f = np.zeros(len(tY_f), dtype=bool)
            ind_hout_f[choice_hout_f] = True 
            
            ind_val_hout_f = np.zeros(len(tY_f), dtype=bool)
            ind_val_hout_f[choice_val_f] = True
            ind_val_hout_f[choice_hout_f] = True
            ind_train_f = ~ind_val_hout_f

            if train_level[i]=='f':
                tY, tregid, tMasks, tregMasks, tBBox = tY_f, tregid_f, tMasks_f, tregMasks_f, tBBox_f
                ind_train = ind_train_f
                # ind_val = ind_val_f
            elif train_level[i] in ['c','ac']:
                tY, tregid, tMasks, tregMasks, tBBox = tY_c, tregid_c, tMasks_c, tregMasks_c, tBBox_c
                ind_train = ind_train_c
                # ind_val = ind_val_c

            tY = np.asarray(tY).astype(np.float32)
            tMasks = np.asarray(tMasks, dtype=object)
            tregMasks = np.asarray(tregMasks, dtype=object)
            tBBox = np.asarray(tBBox)

            # Prepare validation variables. Validation should be on the same level es training!! 
            if train_level[i]=='f':
                self.BBox_val[name] = tBBox_f[ind_val_f]
                valid_val_boxes = (self.BBox_val[name][:,1]-self.BBox_val[name][:,0]) * (self.BBox_val[name][:,3]-self.BBox_val[name][:,2])>0
                self.BBox_val[name] = self.BBox_val[name][valid_val_boxes]
                self.Ys_val[name] =  tY_f[ind_val_f][valid_val_boxes] 
                self.tregid_val[name] = tregid_f[ind_val_f][valid_val_boxes]
                target_to_source_val = self.memory_disag[name][0].clone()
                target_to_source_val[~np.in1d(self.memory_disag[name][0], tregid_val_c)] = 0
                # coarse_regid_val = self.memory_disag[name][0][self.tregid_val[name]].unique(return_counts=True)[0] # consistency check: this should be the same as "tregid_val_c"
                self.source_census_val[name] = { key: value for key,value in self.memory_disag[name][1].items() if key in tregid_val_c}
                self.memory_disag_val[name] = target_to_source_val, self.source_census_val[name], self.memory_disag[name][2]
                if self.tregid_val[name].__len__()>0:
                    self.max_tregid_val[name] = np.max(self.tregid_val[name])
                self.Masks_val[name] = tMasks_f[ind_val_f][valid_val_boxes]
                self.regMasks_val[name] = tregMasks_f[ind_val_f][valid_val_boxes]
                self.loc_list_val.extend( [(name, k) for k,_ in enumerate(self.BBox_val[name])])
            elif train_level[i] in ['c','ac']:
                self.BBox_val[name] = tBBox_c[ind_val_c]
                valid_val_boxes = (self.BBox_val[name][:,1]-self.BBox_val[name][:,0]) * (self.BBox_val[name][:,3]-self.BBox_val[name][:,2])>0
                self.BBox_val[name] = self.BBox_val[name][valid_val_boxes]
                self.Ys_val[name] =  tY_c[ind_val_c][valid_val_boxes] 
                self.tregid_val[name] = tregid_c[ind_val_c][valid_val_boxes]
                target_to_source_val = self.memory_disag[name][0].clone()
                target_to_source_val[~np.in1d(self.memory_disag[name][0], tregid_val_c)] = 0
                # coarse_regid_val = self.memory_disag[name][0][self.tregid_val[name]].unique(return_counts=True)[0] # consistency check: this should be the same as "tregid_val_c"
                self.source_census_val[name] = { key: value for key,value in self.memory_disag[name][1].items() if key in tregid_val_c}
                self.memory_disag_val[name] = target_to_source_val, self.source_census_val[name], self.memory_disag[name][2]
                if self.tregid_val[name].__len__()>0:
                    self.max_tregid_val[name] = np.max(self.tregid_val[name])
                self.Masks_val[name] = tMasks_c[ind_val_c][valid_val_boxes]
                self.regMasks_val[name] = tregMasks_c[ind_val_c][valid_val_boxes]
                self.loc_list_val.extend( [(name, k) for k,_ in enumerate(self.BBox_val[name])])
            
            # Prepare the holdout (test) variables #TODO: refactor val and hout variables computation
            self.BBox_hout[name] = tBBox_f[ind_hout_f]
            valid_hout_boxes = (self.BBox_hout[name][:,1]-self.BBox_hout[name][:,0]) * (self.BBox_hout[name][:,3]-self.BBox_hout[name][:,2])>0
            self.BBox_hout[name] = self.BBox_hout[name][valid_hout_boxes]
            self.Ys_hout[name] =  tY_f[ind_hout_f][valid_hout_boxes] 
            self.tregid_hout[name] = tregid_f[ind_hout_f][valid_hout_boxes]
            target_to_source_hout = self.memory_disag[name][0].clone()
            target_to_source_hout[~np.in1d(self.memory_disag[name][0], tregid_hout_c)] = 0
            # coarse_regid_hout = self.memory_disag[name][0][self.tregid_hout[name]].unique(return_counts=True)[0] # consistency check: this should be the same as "tregid_hout_c"
            self.source_census_hout[name] = { key: value for key,value in self.memory_disag[name][1].items() if key in tregid_hout_c}
            self.memory_disag_hout[name] = target_to_source_hout, self.source_census_hout[name], self.memory_disag[name][2]
            if self.tregid_hout[name].__len__()>0:
                self.max_tregid_hout[name] = np.max(self.tregid_hout[name])
            self.Masks_hout[name] = tMasks_f[ind_hout_f][valid_hout_boxes]
            self.regMasks_hout[name] = tregMasks_f[ind_hout_f][valid_hout_boxes]
            self.loc_list_hout.extend( [(name, k) for k,_ in enumerate(self.BBox_hout[name])])

            # Prepare the training variables
            if train_level[i]=='ac':
                self.BBox_train[name] = tBBox
                valid_train_boxes = (self.BBox_train[name][:,1]-self.BBox_train[name][:,0]) * (self.BBox_train[name][:,3]-self.BBox_train[name][:,2])>0
                self.BBox_train[name] = self.BBox_train[name][valid_train_boxes] 
                self.Ys_train[name] =  tY[valid_train_boxes]
                self.Masks_train[name] = tMasks[valid_train_boxes]
                self.regMasks_train[name] = tregMasks[valid_train_boxes]
                if name in train_dataset_name:
                    self.loc_list_train.extend( [(name, k) for k,_ in enumerate(self.BBox_train[name])])
            else:
                self.BBox_train[name] = tBBox[ind_train]
                valid_train_boxes = (self.BBox_train[name][:,1]-self.BBox_train[name][:,0]) * (self.BBox_train[name][:,3]-self.BBox_train[name][:,2])>0
                self.BBox_train[name] = self.BBox_train[name][valid_train_boxes] 
                self.Ys_train[name] =  tY[ind_train][valid_train_boxes]
                self.Masks_train[name] = tMasks[ind_train][valid_train_boxes]
                self.regMasks_train[name] = tregMasks[ind_train][valid_train_boxes]
                if name in train_dataset_name:
                    self.loc_list_train.extend( [(name, k) for k,_ in enumerate(self.BBox_train[name])])

            # Prepare the complete variables, we only use the finest level for this
            self.BBox[name] = tBBox_f
            valid_boxes = (self.BBox[name][:,1]-self.BBox[name][:,0]) * (self.BBox[name][:,3]-self.BBox[name][:,2])>0
            self.BBox[name] = self.BBox[name][valid_boxes] 
            self.Ys[name] =  tY_f[valid_boxes]
            self.tregid[name] = tregid_f[valid_boxes]
            self.max_tregid[name] = np.max(self.tregid[name])
            self.Masks[name] = tMasks_f[valid_boxes]
            self.regMasks[name] = tregMasks_f[valid_boxes]
            self.loc_list.extend( [(name, k) for k,_ in enumerate(self.BBox[name])])

            # Initialize sample weights
            self.weight_list[name] =  torch.tensor([loss_weights[i]]*len(self.Ys_train[name]), requires_grad=False)
            self.all_weights.extend(self.weight_list[name])
            self.all_sampler_weights.extend( [sampler_weights[i]] * len(self.Ys_train[name]) )
            self.all_natural_weights.extend([len(self.Ys_train[name])] * len(self.Ys_train[name]))
            print("Final usage",process.memory_info().rss/1000/1000,"mb used")

        self.dims = self.features[name].shape[1]

        if build_pairs:  
            
            num_single = len(self.loc_list_train)
            indicies = range(num_single)
            max_pix_forward = 20000

            bboxlist = [ self.BBox[name][k] for name,k in self.loc_list_train ]
            patchsize = [ (bb[1]-bb[0])*(bb[3]-bb[2]) for bb in bboxlist]
            patchsize = np.asarray(patchsize)

            pairs = [[indicies[i],indicies[j]] for i in range(num_single) for j in range(i+1, num_single)]
            pairs = np.asarray(pairs) 
            sumpixels_pairs12 = np.take(patchsize, pairs[:,0]) + np.take(patchsize, pairs[:,1])  
            pairs = pairs[np.asarray(sumpixels_pairs12)<max_pix_forward**2]
            self.small_pairs = pairs[np.asarray(sumpixels_pairs12)>0]

            # triplets = [[indicies[i],indicies[j],indicies[k]] for i in tqdm(range(num_single)) for j in range(i+1, num_single) for k in range(j+1, num_single)]
            # triplets = np.asarray(triplets, dtype=object)
            # sumpixels_triplets = [(patchsize[id1]+patchsize[id2]+patchsize[id3]) for id1,id2,id3 in triplets ]
            # self.small_triplets = triplets[np.asarray(sumpixels_triplets)<max_pix_forward**2]

            # prepare the weights
            self.all_sample_ids = list(self.small_pairs) #+ list(self.small_triplets)
            self.custom_sampler_weights = [ self.all_sampler_weights[idx1]+self.all_sampler_weights[idx2] for idx1,idx2 in self.all_sample_ids ]
            self.natural_sampler_weights = [ self.all_natural_weights[idx1]+self.all_natural_weights[idx2] for idx1,idx2 in self.all_sample_ids ]
        
        else:
            num_single = len(self.loc_list_train)
            self.small_pairs = np.expand_dims(np.arange(num_single, dtype=int), axis=1)
            
            self.all_sample_ids = list(self.small_pairs)
            self.custom_sampler_weights = [ self.all_sampler_weights[idx1[0]] for idx1 in self.all_sample_ids ]
            self.natural_sampler_weights = [ self.all_natural_weights[idx1[0]] for idx1 in self.all_sample_ids ]
            

        print("Dataloader ready.")

    def __len__(self):
        # this will return the length when the data is used for training with a dataloader
        return self.all_sample_ids.__len__()
    
    def len_val(self):
        # this will return the length of the validation dataset
        return len(self.loc_list_val)
    
    def len_all_samples(self, name=None):
        # length when we merge training and validation together
        if name is not None:
            return len(self.Ys[name])
        return len(self.loc_list)

    def idx_to_loc(self, idx):
        return self.loc_list[idx]

    def idx_to_loc_train(self, idx):
        return self.loc_list_train[idx]

    def idx_to_loc_val(self, idx):
        return self.loc_list_val[idx]

    def idx_to_loc_hout(self, idx):
        return self.loc_list_hout[idx]
    
    def num_feats(self):
        return self.dims

    def get_single_item(self, idx, name=None): 
        if name is None:
            # should not be idx_to_loc_val?
            name, k = self.idx_to_loc_val(idx)
            # name, k = self.idx_to_loc(idx)
        else:
            k = idx 
        rmin, rmax, cmin, cmax = self.BBox[name][k]
        X = torch.tensor(self.features[name][0,:,rmin:rmax, cmin:cmax])
        Y = torch.tensor(self.Ys[name][k])
        Mask = torch.tensor(self.Masks[name][k]) 
        census_id = torch.tensor(self.tregid[name][k])
        return X, Y, Mask, name, census_id

    def get_single_training_item(self, idx, name=None): 
        if name is None:
            name, k = self.idx_to_loc_train(idx)
        else:
            k = idx
        rmin, rmax, cmin, cmax = self.BBox_train[name][k]
        X = torch.tensor(self.features[name][0,:,rmin:rmax, cmin:cmax])
        Y = torch.tensor(self.Ys_train[name][k])
        Mask = torch.tensor(self.Masks_train[name][k])
        weight = self.weight_list[name][k]
        return X, Y, Mask, name, weight

    def get_single_validation_item(self, idx, name=None, return_BB=False): 
        if name is None:
            name, k = self.idx_to_loc_val(idx)
        else:
            k = idx
        rmin, rmax, cmin, cmax = self.BBox_val[name][k]
        X = torch.tensor(self.features[name][0,:,rmin:rmax, cmin:cmax])
        Y = torch.tensor(self.Ys_val[name][k])
        Mask = torch.tensor(self.Masks_val[name][k])
        census_id = torch.tensor(self.tregid_val[name][k])
        if np.prod(X.shape[1:])==0:
            raise Exception("no values")
        if return_BB:
            return X, Y, Mask, name, census_id, self.BBox_val[name][k], torch.tensor(self.regMasks_hout[name][k])
        else:
            return X, Y, Mask, name, census_id
    
    def get_single_holdout_item(self, idx, name=None, return_BB=False): 
        if name is None:
            name, k = self.idx_to_loc_hout(idx)
        else:
            k = idx
        rmin, rmax, cmin, cmax = self.BBox_hout[name][k]
        X = torch.tensor(self.features[name][0,:,rmin:rmax, cmin:cmax])
        Y = torch.tensor(self.Ys_hout[name][k])
        Mask = torch.tensor(self.Masks_hout[name][k])
        census_id = torch.tensor(self.tregid_hout[name][k])
        if np.prod(X.shape[1:])==0:
            raise Exception("no values")
        if return_BB:
            return X, Y, Mask, name, census_id, self.BBox_hout[name][k], torch.tensor(self.regMasks_hout[name][k])
        else:
            return X, Y, Mask, name, census_id

    def __getitem__(self,idx):
        idxs = self.all_sample_ids[idx] 
        sample = []
        for i in idxs:
            sample.append(self.get_single_training_item(i))
        
        return sample


def NormL1(outputs, targets, eps=1e-8):
    loss = torch.abs(outputs - targets) / torch.clamp(outputs + targets, min=eps)
    return loss.mean()

def LogL1(outputs, targets, eps=1e-8):
    return torch.abs(torch.log(outputs+1) - torch.log(targets+1)).mean()

def LogL2(outputs, targets, eps=1e-8):
    return ((torch.log(outputs+1) - torch.log(targets+1))**2).mean()

def LogoutputL1(outputs, targets, eps=1e-8):
    return torch.abs(outputs - torch.log(targets)).mean() 

def LogoutputL2(outputs, targets, eps=1e-8):
    loss = (outputs - torch.log(targets))**2
    return loss.mean()

def myMSEloss(y, target):
    return ((y-target)**2).mean()
