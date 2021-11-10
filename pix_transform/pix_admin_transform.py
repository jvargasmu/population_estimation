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

DEFAULT_PARAMS = {
            'weights_regularizer': 0.001, # spatial color head
            'loss': 'l1',
            'optim': 'adam',
            'lr': 0.001,
            "epochs": 25,
            'logstep': 1,
            }

if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm


def eval_my_model(mynet, guide_img, valid_mask, validation_regions,
    valid_validation_ids, num_validation_ids, validation_ids, validation_census, 
    target_img, device, 
    best_scores=[1.,0.,1.],
    optimizer=None, epoch=0,
    disaggregation_data=None, return_scale=False,
    dataset_name="unspecifed_dataset"):

    best_r2, best_mae, best_r2_adj = best_scores

    res = {}

    with torch.no_grad():
        mynet.eval()

        # batchwise passing for whole image
        return_vals = mynet.forward_batchwise(guide_img.unsqueeze(0),
            predict_map=True,
            return_scale=return_scale
        )
        if return_scale:
            predicted_target_img, scales = return_vals
            res["scales"] = scales.squeeze()
        else:
            predicted_target_img = return_vals
        

        # replace masked values with the mean value, this way the artefacts when upsampling are mitigated
        predicted_target_img[~valid_mask] = 1e-10

        # Aggregate by fine administrative boundary
        agg_preds_arr = compute_accumulated_values_by_region(
            validation_regions.numpy().astype(np.uint32),
            predicted_target_img.cpu().numpy().astype(np.float32),
            valid_validation_ids.numpy().astype(np.uint32),
            num_validation_ids
        )
        agg_preds = {id: agg_preds_arr[id] for id in validation_ids}
        r2, mae, mse = compute_performance_metrics(agg_preds, validation_census)
        log_dict = {"r2": r2, "mae": mae, "mse": mse}

        if disaggregation_data is not None:
            target_to_source, source_census, source_regions = disaggregation_data

            predicted_target_img_adjusted = torch.zeros_like(predicted_target_img, device=device)
            predicted_target_img = predicted_target_img.to(device)

            # agg_preds_cr_arr = np.zeros(len(target_to_source.unique()))
            agg_preds_cr_arr = np.zeros(target_to_source.unique().max()+1)
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

            res["predicted_target_img"] = predicted_target_img
        res["predicted_target_img"] = predicted_target_img


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
    
    # if return_scale:
    #     return predicted_target_img, predicted_target_img_adjusted, log_dict, [best_r2, best_mae, best_r2_adj], scales.squeeze()
    # else:
    #     return predicted_target_img, predicted_target_img_adjusted, log_dict, [best_r2, best_mae, best_r2_adj]

    return res, log_dict, [best_r2, best_mae, best_r2_adj]

def PixAdminTransform(
    training_source,
    validation_data=None,
    disaggregation_data=None,
    params=DEFAULT_PARAMS
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if validation_data is not None: 

        # val_features, val_census, val_regions, val_map, val_valid_ids, val_map_valid_ids, val_guide_res, val_valid_data_mask = validation_data
        
        # num_validation_ids = np.unique(val_regions).__len__()
        # val_map[~val_valid_data_mask] = 1e-10 

    # if disaggregation_data is not None:
    #     fine_to_cr, val_cr_census, val_cr_regions = disaggregation_data

    #### prepare_patches #########################################################################
    
    # Iterate throuh the image an cut out examples

    X,Y,Masks = [],[],[]
    for train_dataset_name in training_source.keys():
        tr_features, tr_census, tr_regions, tr_map, tr_guide_res, tr_valid_data_mask = training_source[train_dataset_name] 
        
        tr_regions = tr_regions.to(device)
        tr_valid_data_mask = tr_valid_data_mask.to(device)
        
        for regid in tqdm(tr_census.keys()):
            mask = (regid==tr_regions) * tr_valid_data_mask
            rmin, rmax, cmin, cmax = bbox2(mask)
            X.append(tr_features[:,rmin:rmax, cmin:cmax])
            Y.append(torch.tensor(tr_census[regid]))
            Masks.append(torch.tensor(mask[rmin:rmax, cmin:cmax]))
            
        tr_regions = tr_regions.cpu()
        tr_valid_data_mask = tr_valid_data_mask.cpu()
        
    if params["admin_augment"]:
        train_data = MultiPatchDataset(X, Y, Masks, device=device)
    else:
        train_data = PatchDataset(X, Y, Masks, device=device)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

    #### setup loss/network ############################################################################

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
        raise Exception("unknown loss!")
        
    if params['Net']=='PixNet':
        mynet = PixTransformNet(channels_in=tr_features.shape[0],
                                weights_regularizer=params['weights_regularizer'],
                                device=device).train().to(device)
    elif params['Net']=='ScaleNet':
            mynet = PixScaleNet(channels_in=tr_features.shape[0],
                            weights_regularizer=params['weights_regularizer'],
                            device=device, loss=params['loss'], kernel_size=params['kernel_size'],
                            ).train().to(device)

    optimizer = optim.Adam(mynet.params_with_regularizer, lr=params['lr'])

    if params["load_state"] is not None:
        checkpoint = torch.load('checkpoints/best_r2_{}.pth'.format(params["load_state"]))
        mynet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    wandb.watch(mynet)

    if params['eval_only']:

        #TODO: if we do not cross validation we can do disagregation,
        
        log_dict = {}
        for test_dataset_name in validation_data.keys():
            val_features, val_census, val_regions, val_map, val_valid_ids, val_map_valid_ids, val_guide_res, val_valid_data_mask = validation_data[test_dataset_name]

            res, this_log_dict, best_scores = eval_my_model(
                mynet, val_features, val_valid_data_mask, val_regions,
                val_map_valid_ids, np.unique(val_regions).__len__(), val_valid_ids, val_census, 
                val_map, device,
                disaggregation_data=disaggregation_data[test_dataset_name],
                # fine_to_cr, val_cr_census, val_cr_regions,
                return_scale=True, dataset_name=test_dataset_name
            )
            for key in this_log_dict.keys():
                log_dict[test_dataset_name+'/'+key] = this_log_dict[key]

        wandb.log(log_dict)
                
        return res
    
    #### train network ############################################################################

    epochs = params["epochs"]
    itercounter = 0
    with tqdm(range(0, epochs), leave=True) as tnr:

        # initialize the best score variables
        best_scores = {}
        for test_dataset_name in validation_data.keys():
            best_scores[train_dataset_name] = [-1e12, 1e12, -1e12]

        for epoch in tnr:
            for sample in tqdm(train_loader):
                optimizer.zero_grad()
                
                # Feed forward the network
                y_pred = mynet.forward_one_or_more(sample)
                
                #check if any valid values are there, else skip   
                if y_pred is None:
                    continue
                
                # Sum over the census data per patch
                y = [torch.sum(torch.tensor([samp[1] for samp in sample]))]
                
                # Backwards
                loss = myloss(y_pred, y[0])
                loss.backward()
                optimizer.step()

                itercounter += 1
                torch.cuda.empty_cache()

                # if epoch % params['logstep'] == 0:
                # if itercounter>=( tr_census.keys().__len__() * params['logstep'] ):
                if itercounter>=( 2000*params['logstep'] ):
                    itercounter = 0

                    # Evaluate Model and save model

                    log_dict = {}
                    for test_dataset_name in validation_data.keys():
                        val_features, val_census, val_regions, val_map, val_valid_ids, val_map_valid_ids, val_guide_res, val_valid_data_mask = validation_data[test_dataset_name]


                        res, this_log_dict, this_best_scores = eval_my_model(
                            mynet, val_features, val_valid_data_mask, val_regions,
                            val_map_valid_ids, np.unique(val_regions).__len__(), val_valid_ids, val_census, 
                            val_map, device, 
                            best_scores[train_dataset_name], optimizer=optimizer,
                            disaggregation_data=disaggregation_data[test_dataset_name], epoch=epoch,
                            dataset_name=test_dataset_name
                        )
                        for key in this_log_dict.keys():
                            log_dict[test_dataset_name+'/'+key] = this_log_dict[key]

                        best_scores[test_dataset_name] = this_best_scores

                    log_dict['train/loss'] = loss 

                    # if val_fine_map is not None:
                    tnr.set_postfix(R2=log_dict[list(validation_data.keys())[0]+'/r2'],
                                    zMAEc=log_dict[list(validation_data.keys())[0]+'/mae'])
                    wandb.log(log_dict)
                        
                    mynet.train() 
                    torch.cuda.empty_cache()

    # compute final prediction, un-normalize, and back to numpy
    with torch.no_grad():
        mynet.eval()

        checkpoint = torch.load('checkpoints/best_r2_{}.pth'.format(wandb.run.name) )
        mynet.load_state_dict(checkpoint['model_state_dict'])

        log_dict = {}
        res = {}
        for test_dataset_name in validation_data.keys():
            val_features, val_census, val_regions, val_map, val_valid_ids, val_map_valid_ids, val_guide_res, val_valid_data_mask = validation_data[test_dataset_name]

            this_res, log_dict, best_scores = eval_my_model(
                mynet, val_features, val_valid_data_mask, val_regions,
                val_map_valid_ids, np.unique(val_regions).__len__(), val_valid_ids, val_census, 
                val_map, device,
                best_scores[train_dataset_name], optimizer=optimizer,
                disaggregation_data=disaggregation_data[test_dataset_name], return_scale=True,
                dataset_name=test_dataset_name
            )
            for key in this_log_dict.keys():
                log_dict[test_dataset_name+'/'+key] = this_log_dict[key]
            res[test_dataset_name] = this_res
            

    return res