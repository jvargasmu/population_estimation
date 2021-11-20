import fiona
from osgeo import gdal
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.utils import check_array
from tqdm import tqdm
import copy
from pylab import figure, imshow, matshow, grid, savefig
import torch
import pickle
import h5py

def get_properties_dict(data_dict_orig):
    data_dict = []
    for data_row in data_dict_orig:
        data_dict.append(data_row["properties"])
    return data_dict


def read_input_raster_data_to_np(input_paths):
    #assuming every covariate has same dimensions
    first_name = list(input_paths.keys())[0]
    hwdims = gdal.Open(input_paths[first_name]).ReadAsArray().astype(np.float32).shape
    fdim = input_paths.__len__()
    inputs = np.zeros((fdim,) + hwdims, dtype=np.float32) 
    for i,kinp in enumerate(input_paths.keys()):
        print("read {}".format(input_paths[kinp]))
        inputs[i] = gdal.Open(input_paths[kinp]).ReadAsArray().astype(np.float32)
    return inputs


def read_input_raster_data(input_paths):
    inputs = {}
    for kinp in input_paths.keys():
        print("read {}".format(input_paths[kinp]))
        inputs[kinp] = gdal.Open(input_paths[kinp]).ReadAsArray().astype(np.float32)
    return inputs


def read_shape_layer_data(shape_layer_path):
    with fiona.open(shape_layer_path) as reader:
        layer_data_orig = [elem for elem in reader]
    layer_data = get_properties_dict(layer_data_orig)
    return layer_data

def mean_absolute_percentage_error(y_true, y_pred): 

    y_true = check_array(y_true.reshape(-1,1))
    y_pred = check_array(y_pred.reshape(-1,1))
    
    zeromask = (y_true==0)
    y_true, y_pred = y_true[zeromask], y_pred[zeromask]  

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


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

    r2 = r2_score(gt, preds)
    mae = mean_absolute_error(gt, preds)
    mse = mean_squared_error(gt, preds)
    mape = mean_absolute_percentage_error(gt,preds)

    return r2, mae, mse, mape


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

    return rmin, rmax, cmin, cmax


class PatchDataset(torch.utils.data.Dataset):
    """Patch dataset."""
    def __init__(self, rawsets, memory_mode, device): 
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
    def __init__(self, rawsets, memory_mode, device):
        self.device = device
        
        print("Preparing dataloader for: ", list(rawsets.keys()))
        self.loc_list = []
        self.BBox = {}
        self.features = {}
        self.Ys = {}
        self.Masks = {}
        for i, (name, rs) in (enumerate(rawsets.items())):

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
        
        num_single = len(self.loc_list)
        indicies = range(num_single)
        max_pix_forward = 20000

        bboxlist = [ self.BBox[name][k] for name,k in self.loc_list ]
        patchsize = [ (bb[1]-bb[0])*(bb[3]-bb[2]) for bb in bboxlist]
        patchsize = np.asarray(patchsize)

        pairs = [[indicies[i],indicies[j]] for i in range(num_single) for j in range(i+1, num_single)]
        pairs = np.asarray(pairs) 
        sumpixels_pairs12 = np.take(patchsize, pairs[:,0]) + np.take(patchsize, pairs[:,1])  
        self.small_pairs = pairs[np.asarray(sumpixels_pairs12)<max_pix_forward**2]

        # triplets = [[indicies[i],indicies[j],indicies[k]] for i in tqdm(range(num_single)) for j in range(i+1, num_single) for k in range(j+1, num_single)]
        # triplets = np.asarray(triplets, dtype=object)
        # sumpixels_triplets = [(patchsize[id1]+patchsize[id2]+patchsize[id3]) for id1,id2,id3 in triplets ]
        # self.small_triplets = triplets[np.asarray(sumpixels_triplets)<max_pix_forward**2]

        self.all_sample_ids = list(self.small_pairs) #+ list(self.small_triplets)


    def __len__(self):
        return self.all_sample_ids.__len__()

    def idx_to_loc(self, idx):
        return self.loc_list[idx]
    
    def num_feats(self):
        return self.dims

    def getsingleitem(self, idx):
        output = []
        name, k = self.idx_to_loc(idx)
        rmin, rmax, cmin, cmax = self.BBox[name][k]
        X = torch.from_numpy(self.features[name][0,:,rmin:rmax, cmin:cmax])
        Y = torch.from_numpy(self.Ys[name][k])
        Mask = torch.from_numpy(self.Masks[name][k]) 
        return X, Y, Mask

    def __getitem__(self,idx):
        idxs = self.all_sample_ids[idx]

        sample = []
        for i in idxs:
            sample.append(self.getsingleitem(i))
        
        return sample


def NormL1(outputs, targets, eps=1e-8):
    loss = torch.abs(outputs - targets) / torch.clip(outputs + targets, min=eps)
    return loss.mean()

def LogL1(outputs, targets, eps=1e-8):
    loss = torch.abs(torch.log(outputs+1) - torch.log(targets+1))
    return loss.mean()

def LogoutputL1(outputs, targets, eps=1e-8):
    loss = torch.abs(outputs - torch.log(targets))
    return loss.mean()

def LogoutputL2(outputs, targets, eps=1e-8):
    loss = (outputs - torch.log(targets+1))**2
    return loss.mean()

def save_as_geoTIFF(src_filename,  dst_filename, raster): 
 
    from osgeo import gdal, osr

    src_filename ='/path/to/source.tif'
    dst_filename = '/path/to/destination.tif'

    # Opens source dataset
    src_ds = gdal.Open(src_filename)
    format = "GTiff"
    driver = gdal.GetDriverByName(format)

    # Open destination dataset
    dst_ds = driver.CreateCopy(dst_filename, src_ds, 0)

    # Specify raster location through geotransform array
    # (uperleftx, scalex, skewx, uperlefty, skewy, scaley)
    # Scale = size of one pixel in units of raster projection
    # this example below assumes 100x100
    gt = [-7916400, 100, 0, 5210940, 0, -100]

    # Set location
    dst_ds.SetGeoTransform(gt)

    # Get raster projection
    epsg = 3857
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dest_wkt = srs.ExportToWkt()

    # Set projection
    dst_ds.SetProjection(dest_wkt)

    # Close files
    dst_ds = None
    src_ds = None



    # src_path = '/home/pf/pfstaff/projects/andresro/typhoon/baselines'
    # src_path_profiles = '/scratch/for_andres/before/orig'

    # algs = ['cva-2', 'mad-2']
    # tiles = ['T51PUR', 'T51PUT', 'T51PVR', 'T51PWQ']
    
    # for alg in algs:
    #     for tile in tiles:
    #         print(f'Processing {alg}/{tile}...')
    #         mat_fname = os.path.join(src_path, alg, f'{tile}.mat')
    #         mat_file = sio.loadmat(mat_fname)
    #         cm = mat_file['CM'].astype('float32')
            
    #         profile_src_path = os.path.join(src_path_profiles, f'{tile}_mean_12.tif')
    #         profile_src = rasterio.open(profile_src_path)
    #         profile = profile_src.profile.copy()
    #         profile['count'] = 1
    #         profile_src.close()
            
    #         out_path = os.path.join(alg, f'{tile}_mean_12.tif')
    #         writer = rasterio.open(out_path, 'w', **profile)
            
    #         writer.write(cm, indexes=1)
    #         writer.close()
            
    # print('End')