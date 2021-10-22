import numpy as np
from numpy.core.numeric import zeros_like

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import sys
import wandb

from utils import plot_2dmatrix, accumulate_values_by_region, compute_performance_metrics, bbox2, \
     PatchDataset, MultiPatchDataset, NormL1, LogL1, LogoutputL1, LogoutputL2
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region
from pix_transform_utils.utils import upsample

from pix_transform.pix_transform_net import PixTransformNet, PixScaleNet

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
        validation_census, target_img, validation_regions, validation_ids, valid_validation_ids, target_to_source = validation_data
        num_validation_ids = np.unique(validation_regions).__len__()
        target_img[~valid_mask] = 1e-10 #target_img[valid_mask].mean()

    source_census , source_regions, source_map = source
    source_regions = source_regions.to(device)

    n_channels, hr_height, hr_width = guide_img.shape

    # source_img = source_img.squeeze()
    lr_height, lr_width = source_regions.shape

    # TODO: also try with the downsampled version, for this we also need to downsample the regions which might be tricky/expensive (categorical labels), or we just divide the bounding box coordinates by the factor
    assert (params["feature_downsampling"]==1)

    source_arr = list(source_census.values())

    # TODO: Remove!
    if params['spatial_features_input']:
        x = np.linspace(-0.5, 0.5, hr_height)
        y = np.linspace(-0.5, 0.5, hr_width)
        x_grid, y_grid = np.meshgrid(x, y, indexing='ij')

        x_grid = np.expand_dims(x_grid, axis=0)
        y_grid = np.expand_dims(y_grid, axis=0)

        guide_img = np.concatenate([guide_img, x_grid, y_grid], axis=0)
        n_channels += 2

    #### prepare_patches #########################################################################
    
    # Iterate throuh the image an cut out examples
    valid_mask = valid_mask.to(device)
    X,Y,Masks = [],[],[]
    for regid in tqdm(source_census.keys()):
        mask = (regid==source_regions) * valid_mask
        rmin, rmax, cmin, cmax = bbox2(mask)
        X.append(guide_img[:,rmin:rmax, cmin:cmax])
        Y.append(torch.tensor(source_census[regid]))
        Masks.append(torch.tensor(mask[rmin:rmax, cmin:cmax]))
    valid_mask = valid_mask.cpu()
    


    if params["admin_augment"]:
        train_data = MultiPatchDataset(X, Y, Masks, device=device)
    else:
        train_data = PatchDataset(X, Y, Masks, device=device)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    ###############################################################################################

    #### setup network ############################################################################
    ###############################################################################################

    if params['loss'] == 'mse':
        myloss = torch.nn.MSELoss()
    elif params['loss'] == 'l1':
        myloss = torch.nn.L1Loss()
    elif params['loss'] == 'NormL1':
        myloss = NormL1
    elif params['loss'] == 'LogL1':
        myloss = LogL1
    elif params['loss'] == 'LogoutputL1':
        myloss = LogoutputL1
    elif params['loss'] == 'LogoutputL2':
        myloss = LogoutputL2
    else:
        print("unknown loss!")
        return
        
    if params['Net']=='PixNet':
        mynet = PixTransformNet(channels_in=guide_img.shape[0],
                                weights_regularizer=params['weights_regularizer'],
                                device=device).train().to(device)
    elif params['Net']=='ScaleNet':
            mynet = PixScaleNet(channels_in=guide_img.shape[0],
                            weights_regularizer=params['weights_regularizer'],
                            device=device, loss=params['loss'], kernel_size=params['kernel_size'],
                            ).train().to(device)

    optimizer = optim.Adam(mynet.params_with_regularizer, lr=params['lr'])

    if params["load_state"] is not None:
        checkpoint = torch.load('checkpoints/best_r2_{}.pth'.format(params["load_state"]))
        mynet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    wandb.watch(mynet)

    epochs = params["epochs"]
    itercounter = 0
    with tqdm(range(0, epochs), leave=True) as tnr:
        best_r2=-1e12
        best_mae=1e12
        best_r2_adj=-1e12
        for epoch in tnr:
            for sample in tqdm(train_loader):
                optimizer.zero_grad()
                
                if isinstance(sample, list):
                    y_pred = mynet.forward_one_or_more(sample)
                     #check if any valid values are there, else skip   
                    if y_pred is None:
                        continue
                    
                    # Sum over the census data per patch
                    y = [torch.sum(torch.tensor([samp[1] for samp in sample]))]

                else:
                     #check if any valid values are there, else skip   
                    if sample[2].sum()<1:
                        continue
                    
                    X,y,mask = sample
                    y_pred = mynet(X, mask)

                loss = myloss(y_pred, y[0])

                loss.backward()
                optimizer.step()

                itercounter += 1
                torch.cuda.empty_cache()

                # if epoch % params['logstep'] == 0:
                if itercounter>=( source_census.keys().__len__() * params['logstep'] ):
                    itercounter = 0

                    with torch.no_grad():
                        mynet.eval()

                        # batchwise passing for whole image
                        predicted_target_img = mynet.forward_batchwise(guide_img.unsqueeze(0),
                            predict_map=True
                        )

                        # replace masked values with the mean value, this way the artefacts when upsampling are mitigated
                        predicted_target_img[~valid_mask] = 1e-10

                        if target_img is not None:
                            
                            # Aggregate by fine administrative boundary
                            agg_preds_arr = compute_accumulated_values_by_region(
                                validation_regions.numpy().astype(np.uint32),
                                predicted_target_img.cpu().numpy().astype(np.float32),
                                valid_validation_ids.numpy().astype(np.uint32),
                                num_validation_ids
                            )
                            agg_preds = {id: agg_preds_arr[id] for id in validation_ids}
                            r2, mae, mse = compute_performance_metrics(agg_preds, validation_census)
                            log_dict = {"r2": r2, "mae": mae, "mse": mse, 'train/loss': loss}

                            compute_constrained_map = True
                            if compute_constrained_map:
                                predicted_target_img_adjusted = torch.zeros_like(predicted_target_img, device=device)
                                predicted_target_img = predicted_target_img.to(device)

                                agg_preds_cr_arr = np.zeros(len(target_to_source.unique()))
                                for finereg in target_to_source.unique():
                                    finregs_to_sum = torch.nonzero(target_to_source==finereg)
                                    agg_preds_cr_arr[finereg] = agg_preds_arr[target_to_source==finereg].sum()
                                
                                agg_preds_cr = {id: agg_preds_cr_arr[id] for id in source_census.keys()}
                                scalings = {id: source_census[id]/agg_preds_cr[id] for id in source_census.keys()}


                                for idx in scalings.keys():
                                    mask = [source_regions==idx]
                                    predicted_target_img_adjusted[mask] = predicted_target_img[mask]*scalings[idx]

                                # Aggregate by fine administrative boundary
                                agg_preds_adj_arr = compute_accumulated_values_by_region(
                                    validation_regions.numpy().astype(np.uint32),
                                    predicted_target_img_adjusted.cpu().numpy().astype(np.float32),
                                    valid_validation_ids.numpy().astype(np.uint32),
                                    num_validation_ids
                                )
                                agg_preds_adj = {id: agg_preds_adj_arr[id] for id in validation_ids}
                                r2_adj, mae_adj, mse_adj = compute_performance_metrics(agg_preds_adj, validation_census)
                                log_dict.update( {"adjusted/r2": r2_adj, "adjusted/mae": mae_adj, "adjusted/mse": mse_adj} )

                                predicted_target_img_adjusted = predicted_target_img_adjusted.cpu()
                                predicted_target_img = predicted_target_img.cpu()

                            if r2>best_r2:
                                best_r2 = r2
                                log_dict["best_r2"] = best_r2
                                torch.save({'model_state_dict':mynet.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'epoch':epoch, 'log_dict':log_dict},
                                    'checkpoints/best_r2_{}.pth'.format(wandb.run.name) )
                            
                            if mae<best_mae:
                                best_mae = mae
                                log_dict["best_mae"] = best_mae
                                torch.save({'model_state_dict':mynet.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'epoch':epoch, 'log_dict':log_dict},
                                    'checkpoints/best_mae_{}.pth'.format(wandb.run.name) )
                            
                            if r2_adj>best_r2_adj:
                                best_r2_adj = r2_adj
                                log_dict["adjusted/best_r2"] = best_r2_adj
                                torch.save({'model_state_dict':mynet.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'epoch':epoch, 'log_dict':log_dict},
                                    'checkpoints/best_r2_adj_{}.pth'.format(wandb.run.name) )

                        if target_img is not None:
                            tnr.set_postfix(R2=r2, zMAEc=mae)
                            wandb.log(log_dict)
                            
                        mynet.train() 
                        torch.cuda.empty_cache()

    # compute final prediction, un-normalize, and back to numpy
    with torch.no_grad():
        mynet.eval()

        checkpoint = torch.load('checkpoints/best_r2_{}.pth'.format(wandb.run.name) )
        mynet.load_state_dict(checkpoint['model_state_dict'])

        predicted_target_img = mynet.forward_batchwise(guide_img.unsqueeze(0),
            predict_map=True
        )

        compute_constrained_map = True
        if compute_constrained_map:
            agg_preds_cr_arr = np.zeros(len(target_to_source.unique()))
            for finereg in target_to_source.unique():
                finregs_to_sum = torch.nonzero(target_to_source==finereg)
                agg_preds_cr_arr[finereg] = agg_preds_arr[target_to_source==finereg].sum()
            
            agg_preds_cr = {id: agg_preds_cr_arr[id] for id in source_census.keys()}
            scalings = {id: source_census[id]/agg_preds_cr[id] for id in source_census.keys()}

            predicted_target_img_adjusted = torch.zeros_like(predicted_target_img)

            for idx in tqdm(scalings.keys()):
                mask = [source_regions==idx]
                predicted_target_img_adjusted[mask] = predicted_target_img[mask]*scalings[idx]

    return predicted_target_img, predicted_target_img_adjusted