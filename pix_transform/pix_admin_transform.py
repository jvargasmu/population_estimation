import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import sys
import wandb

from utils import plot_2dmatrix, accumulate_values_by_region, compute_performance_metrics, bbox2, PatchDataset
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region
from pix_transform_utils.utils import upsample

from pix_transform.pix_transform_net import PixTransformNet

DEFAULT_PARAMS = {'feature_downsampling': 1,
            'spatial_features_input': False,
            'weights_regularizer': 0.001, # spatial color head
            'loss': 'l1',
            "predict_log_values": False,
            'optim': 'adam',
            'lr': 0.001,
            "epochs": 25,
            'logstep': 1,
            }

if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm


def PixAdminTransform(guide_img, source, valid_mask=None, params=DEFAULT_PARAMS, validation_data=None, orig_guide_res=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(guide_img.shape) < 3:
        guide_img = np.expand_dims(guide_img, 0)

    if valid_mask is None:
        valid_mask = np.ones_like(guide_img)

    if validation_data is not None:
        validation_census, target_img, validation_regions, validation_ids, valid_validation_ids = validation_data
        num_validation_ids = np.unique(validation_regions).__len__()
        target_img[~valid_mask] = target_img[valid_mask].mean()

    source_census , source_regions, source_map = source
    n_channels, hr_height, hr_width = guide_img.shape

    # source_img = source_img.squeeze()
    lr_height, lr_width = source_regions.shape

    # TODO: also try with the downsampled version, for this we also need to downsample the regions which might be tricky/expensive (categorical labels), or we just divide the bounding box coordinates by the factor
    assert (params["feature_downsampling"]==1)

    guide_img_mean = guide_img[:,valid_mask].mean(1)
    guide_img_std = guide_img[:,valid_mask].std(1)
    guide_img = ((guide_img.transpose((1,2,0)) - guide_img_mean ) / guide_img_std).transpose((2,0,1))

    source_arr = list(source_census.values())

    if params['spatial_features_input']:
        x = np.linspace(-0.5, 0.5, hr_height)
        y = np.linspace(-0.5, 0.5, hr_width)
        x_grid, y_grid = np.meshgrid(x, y, indexing='ij')

        x_grid = np.expand_dims(x_grid, axis=0)
        y_grid = np.expand_dims(y_grid, axis=0)

        guide_img = np.concatenate([guide_img, x_grid, y_grid], axis=0)
        n_channels += 2

    #### prepare_patches #########################################################################

    # Move all important variable to pytorch
    guide_img = torch.from_numpy(guide_img).type(torch.float32)
    # source_img = torch.from_numpy(source_img).float()
    valid_mask = torch.from_numpy(valid_mask).type(torch.BoolTensor)
    if target_img is not None:
        target_img = torch.from_numpy(target_img).float()
        validation_regions = torch.from_numpy(validation_regions.astype(np.int16))
    torch_valid_validation_ids = torch.from_numpy(valid_validation_ids.astype(np.bool8))
    source_map = torch.from_numpy(source_map)

    # source_img_mean = torch.tensor(source_img_mean)
    # source_img_std = torch.tensor(source_img_std)
    
    # Iterate throuh the image an cut out examples
    X,Y,Masks = [],[],[]
    for regid in tqdm(source_census.keys()):
        mask = regid==source_regions
        rmin, rmax, cmin, cmax = bbox2(mask)
        X.append(guide_img[:,rmin:rmax, cmin:cmax])
        Y.append(torch.tensor(source_census[regid]))
        Masks.append(torch.tensor(mask[rmin:rmax, cmin:cmax]))


    train_data = PatchDataset(X, Y, Masks, device=device)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    ###############################################################################################

    #### setup network ############################################################################
    mynet = PixTransformNet(channels_in=guide_img.shape[0],
                            weights_regularizer=params['weights_regularizer'],
                            device=device).train().to(device)
    wandb.watch(mynet)
    
    optimizer = optim.Adam(mynet.params_with_regularizer, lr=params['lr'])
    if params['loss'] == 'mse':
        myloss = torch.nn.MSELoss()
    elif params['loss'] == 'l1':
        myloss = torch.nn.L1Loss()
    else:
        print("unknown loss!")
        return
    ###############################################################################################

    epochs = params["epochs"] # params["batch_size"] * params["iteration"] // (guide_patches.shape[0])
    with tqdm(range(0, epochs), leave=True) as tnr:
        # tnr.set_description("epoch {}".format(0))
        if target_img is not None:
            tnr.set_postfix(R2=-99., MAEc=100000.)
        else:
            tnr.set_postfix(consistency=-1.)
        for epoch in tnr:
            for data,y,mask in train_loader:
                if mask.sum()<1:
                    optimizer.zero_grad()
                    continue
                
                optimizer.zero_grad()

                y_pred = mynet(data, (0,1), mask)
                loss = myloss(y_pred, y[0].to(device))

                loss.backward()
                optimizer.step()

                torch.cuda.empty_cache()

            if epoch % params['logstep'] == 0:
                with torch.no_grad():
                    mynet.eval()

                    # batchwise passing for whole image
                    predicted_target_img = mynet.forward_batchwise(guide_img.unsqueeze(0),
                        norm=(0,1),
                        predict_map=True
                    ).squeeze()

                    # replace masked values with the mean value, this way the artefacts when upsampling are mitigated
                    predicted_target_img[~valid_mask] = 1e-10

                    if target_img is not None:

                        if params["predict_log_values"]:
                            abs_predicted_target_img = torch.exp(abs_predicted_target_img)

                        # convert resolution back to feature resolution
                        if params["feature_downsampling"]!=1:
                            full_res_predicted_target_image = upsample(predicted_target_img, params["feature_downsampling"], orig_guide_res=orig_guide_res)
                        else:
                            full_res_predicted_target_image = predicted_target_img
                        
                        # Aggregate by fine administrative boundary
                        agg_preds_arr = compute_accumulated_values_by_region(
                            validation_regions.numpy().astype(np.uint32),
                            full_res_predicted_target_image.cpu().numpy().astype(np.float32),
                            valid_validation_ids,
                            num_validation_ids
                        )
                        agg_preds = {id: agg_preds_arr[id] for id in validation_ids}
                        r2, mae, mse = compute_performance_metrics(agg_preds, validation_census)

                    if target_img is not None:
                        tnr.set_postfix(R2=r2, zMAEc=mae)
                        wandb.log({"r2": r2, "mae": mae, "mse": mse})

                    mynet.train()
                    print(mae,r2)

    # compute final prediction, un-normalize, and back to numpy
    mynet.eval()
    predicted_target_img = mynet.forward_batchwise(guide_img.unsqueeze(0),
        norm=(0,1),
        predict_map=True
    ).squeeze()
    predicted_target_img = predicted_target_img.cpu().detach().squeeze().numpy()

    return predicted_target_img
