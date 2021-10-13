import argparse
import numpy as np
from osgeo import gdal
import config_pop as cfg
import pickle
from utils import read_input_raster_data, create_map_of_valid_ids, compute_features_from_raw_inputs, compute_performance_metrics
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region
import torch
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn as nn
from building_disagg_baseline import disaggregate_weighted_by_preds


class PopNN(nn.Module):

    def __init__(self, channels_in=4, weights_regularizer=None):
        super(PopNN, self).__init__()

        self.channels_in = channels_in
        self.spatial_net = nn.Sequential(nn.Linear(2, 32),
                                         nn.ReLU(), nn.Linear(32, 512))
        self.color_net = nn.Sequential(nn.Linear(channels_in - 2, 32),
                                       nn.ReLU(), nn.Linear(32, 512))
        self.head_net = nn.Sequential(nn.ReLU(),
                                      nn.Linear(512, 32),
                                      nn.ReLU(), nn.Linear(32, 1))

        if weights_regularizer is None:
            reg_spatial = 0.0001
            reg_color = 0.001
            reg_head = 0.0001
        else:
            reg_spatial = weights_regularizer[0]
            reg_color = weights_regularizer[1]
            reg_head = weights_regularizer[2]

        self.params_with_regularizer = []
        self.params_with_regularizer += [{'params': self.spatial_net.parameters(), 'weight_decay': reg_spatial}]
        self.params_with_regularizer += [{'params': self.color_net.parameters(), 'weight_decay': reg_color}]
        self.params_with_regularizer += [{'params': self.head_net.parameters(), 'weight_decay': reg_head}]

    def forward(self, input):
        input_spatial = input[:, self.channels_in - 2:]
        input_color = input[:, 0:self.channels_in - 2]
        merged_features = self.spatial_net(input_spatial) + self.color_net(input_color)
        return self.head_net(merged_features)


class PixelFeaturesTestDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.len = len(self.features)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.from_numpy(self.features[index])


def compute_dict_region_to_valid_indexes(regions):
    num_regions = len(np.unique(regions))
    region_to_valinds = {id: [] for id in range(1, num_regions + 1)}

    for i in range(len(regions)):
        region_id = regions[i]
        region_to_valinds[region_id].append(region_id)

    for region_id in range(1, num_regions + 1):
        region_to_valinds[region_id] = np.array(region_to_valinds[region_id]).astype(np.uint32)

    return region_to_valinds


def train_nn_pop(preproc_data_path, rst_wp_regions_path, output_dir, dataset_name):
    # Read input data
    input_paths = cfg.input_paths[dataset_name]
    with open(preproc_data_path, 'rb') as handle:
        pdata = pickle.load(handle)

    wp_rst_regions = gdal.Open(rst_wp_regions_path).ReadAsArray().astype(np.uint32)
    valid_ids = pdata["valid_ids"]
    no_valid_ids = pdata["no_valid_ids"]
    inputs = read_input_raster_data(input_paths)
    input_buildings = inputs["buildings"]
    id_to_cr_id = pdata["id_to_cr_id"]
    num_coarse_regions = pdata["num_coarse_regions"]
    cr_census_arr = pdata["cr_census_arr"]
    geo_metadata = pdata["geo_metadata"]
    valid_census = pdata["valid_census"]
    feats_list = inputs.keys()
    wp_ids = list(np.unique(wp_rst_regions))
    num_wp_ids = len(wp_ids)
    buildings = np.multiply(input_buildings, (input_buildings < 255).astype(np.float32))
    buildings_mask = buildings > 0
    # Binary map representing a pixel belong to a region with valid id
    map_valid_ids = create_map_of_valid_ids(wp_rst_regions, no_valid_ids)
    # Get map of coarse level regions
    cr_regions = compute_map_with_new_labels(wp_rst_regions, id_to_cr_id, map_valid_ids)
    # Disaggregate population using building maps as weights
    disagg_population = disaggregate_weighted_by_preds(cr_census_arr, buildings,
                                                       map_valid_ids, cr_regions, num_coarse_regions, output_dir,
                                                       mask=buildings_mask, save_images=False)
    initial_targets = disagg_population.flatten()
    # Get features
    raw_features = compute_features_from_raw_inputs(inputs, feats_list)
    # Add spatial features
    x = np.arange(input_buildings.shape[1])
    y = np.arange(input_buildings.shape[0])
    y_grid, x_grid = np.meshgrid(y, x, indexing='ij')
    x_feat = np.expand_dims(x_grid.flatten(), axis=1)
    y_feat = np.expand_dims(y_grid.flatten(), axis=1)
    # Get map of coarse level regions
    cr_regions = compute_map_with_new_labels(wp_rst_regions, id_to_cr_id, map_valid_ids)
    # Create input array (only valid pixels and valid regions)
    cr_regions_mask = cr_regions > 0
    valid_pixels = np.multiply(cr_regions_mask, map_valid_ids.astype(np.bool))
    valid_pixels = np.multiply(valid_pixels, buildings_mask)
    valid_pixels_flat = valid_pixels.flatten()
    # Get valid features
    all_features = np.concatenate([raw_features, y_feat, x_feat], axis=1)
    valid_features = all_features[valid_pixels_flat]
    valid_initial_targets = initial_targets[valid_pixels_flat]
    valid_cr_regions = cr_regions.flatten()[valid_pixels_flat]
    raw_feat_mean = np.mean(valid_features, axis=0)
    raw_feat_std = np.std(valid_features, axis=0)
    valid_features = (valid_features - raw_feat_mean) / raw_feat_std
    print("valid_features.shape {}".format(valid_features.shape))
    # Setup Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_input_channels = valid_features.shape[1]
    params = {"weights_regularizer": [0.0001, 0.001, 0.001],
                      "lr": 0.001,
                      "loss": 'l1',
                      "num_epochs": 5000,
                      "num_epochs_show_logs": 100,
                      "eval_batch_size" : 256}

    pop_net = PopNN(channels_in=num_input_channels, weights_regularizer=params['weights_regularizer']).train().to(device)

    optimizer = optim.Adam(pop_net.params_with_regularizer, lr=params['lr'])
    if params['loss'] == 'mse':
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.L1Loss()

    # Train model for several epochs : TODO: data augmentation
    region_to_valinds = compute_dict_region_to_valid_indexes(valid_cr_regions)
    for epoch_number in range(1, params["num_epochs"]+1):

        accumulated_loss = 0
        for region_id in range(1, num_coarse_regions):

            region_valinds = region_to_valinds[region_id]
            features_numpy = valid_features[region_valinds]
            region_target = cr_census_arr[region_id]

            features = torch.from_numpy(features_numpy).float().to(device)
            target = torch.tensor(region_target).float().to(device)

            optimizer.zero_grad()

            preds = pop_net(features)
            total_preds = torch.sum(preds)
            loss = loss_fn(total_preds, target)

            accumulated_loss += loss.item()

            loss.backward()
            optimizer.step()

        if epoch_number % params['num_epochs_show_logs'] == 0:
            print("epoch {} acc_loss {}".format(epoch_number, accumulated_loss))

    # Evaluate with data of the whole image
    eval_batch_size = params["eval_batch_size"]
    eval_dataset = PixelFeaturesTestDataset(valid_features)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                                  batch_size=eval_batch_size, shuffle=False)

    num_valid_features = len(valid_features)
    valid_preds = np.zeros(num_valid_features).astype(np.float32)
    current_index = 0
    for features in eval_loader:
        with torch.no_grad():
            pop_net.eval()
            features = features.float().to(device)

            preds = pop_net(features)
            pres_np_array = preds.view(-1).cpu().detach().numpy()
            if current_index + eval_batch_size > num_valid_features:
                valid_preds[current_index:] = pres_np_array
            else:
                valid_preds[current_index:current_index+eval_batch_size] = pres_np_array

            current_index += len(features)

    all_preds = np.zeros(len(all_features)).astype(np.float32)
    all_preds[valid_pixels_flat] = valid_preds
    pred_map = all_preds.reshape(buildings.shape)
    print("Inference Done")

    # Get building maps with values between 0 and 1 (sometimes 255 represent no data values)
    unnorm_weights = pred_map
    mask = np.multiply(input_buildings > 0, (input_buildings < 255))

    # Disaggregate population using pred maps as weights
    disagg_population = disaggregate_weighted_by_preds(cr_census_arr, unnorm_weights,
                                                       map_valid_ids, cr_regions, num_coarse_regions, output_dir,
                                                       mask=mask, save_images=True, geo_metadata=geo_metadata)

    # Aggregate pixel level predictions to the finest level region
    agg_preds_arr = compute_accumulated_values_by_region(wp_rst_regions, disagg_population, map_valid_ids, num_wp_ids)
    agg_preds = {id: agg_preds_arr[id] for id in valid_ids}

    preds_and_gt_dict = {}
    for id in valid_census.keys():
        preds_and_gt_dict[id] = {"pred": agg_preds[id], "gt": valid_census[id]}

    # Save predictions
    preds_and_gt_path = "{}preds_and_gt.pkl".format(output_dir)
    with open(preds_and_gt_path, 'wb') as handle:
        pickle.dump(preds_and_gt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Compute metrics
    r2, mae, mse = compute_performance_metrics(agg_preds, valid_census)
    print("r2 {} mae {} mse {}".format(r2, mae, mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preproc_data_path", type=str, help="Preprocessed data of regions (pickle file)")
    parser.add_argument("rst_wp_regions_path", type=str,
                        help="Raster of WorldPop administrative boundaries information")
    parser.add_argument("output_dir", type=str, help="Output dir ")
    parser.add_argument("dataset_name", type=str, help="Dataset name")
    args = parser.parse_args()

    train_nn_pop(args.preproc_data_path, args.rst_wp_regions_path, args.output_dir, args.dataset_name)


if __name__ == "__main__":
    main()
