import numpy as np
from numpy.core.numeric import zeros_like
import os
import logging
logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import sys
import wandb
import h5py
import pickle
from pathlib import Path
import random

from utils import plot_2dmatrix, accumulate_values_by_region, compute_performance_metrics, bbox2, \
     PatchDataset, MultiPatchDataset, NormL1, LogL1, LogoutputL1, LogoutputL2, compute_performance_metrics_arrays
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region
from pix_transform_utils.utils import upsample

from pix_transform.pix_transform_net import PixTransformNet, PixScaleNet


if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm


def disag_map(predicted_target_img, agg_preds_arr, disaggregation_data):

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unfold disagg data
    target_to_source, source_census, source_regions = disaggregation_data

    predicted_target_img_adjusted = torch.zeros_like(predicted_target_img, device=device)
    predicted_target_img = predicted_target_img.to(device)

    # agg_preds_cr_arr = np.zeros(len(target_to_source.unique()))
    agg_preds_cr_arr = np.zeros(target_to_source.unique().max()+1)
    for finereg in target_to_source.unique():
        finregs_to_sum = torch.nonzero(target_to_source==finereg)
        agg_preds_cr_arr[finereg] = agg_preds_arr[target_to_source==finereg].sum()
    
    agg_preds_cr = {id: agg_preds_cr_arr[id] for id in source_census.keys()}
    scalings = {id: torch.tensor(source_census[id]/agg_preds_cr[id]).to(device) for id in source_census.keys()}

    for idx in (scalings.keys()):
        mask = [source_regions==idx]
        predicted_target_img_adjusted[mask] = predicted_target_img[mask]*scalings[idx]

    return predicted_target_img_adjusted.cpu()


def disag_and_eval_map(predicted_target_img, agg_preds_arr, validation_regions, valid_validation_ids,
    num_validation_ids, validation_ids, validation_census, disaggregation_data):

    predicted_target_img_adjusted = disag_map(predicted_target_img, agg_preds_arr, disaggregation_data)

    # Aggregate by fine administrative boundary
    agg_preds_adj_arr = compute_accumulated_values_by_region(
        validation_regions.numpy().astype(np.uint32),
        predicted_target_img_adjusted.cpu().numpy().astype(np.float32),
        valid_validation_ids.numpy().astype(np.uint32),
        num_validation_ids
    )
    agg_preds_adj = {id: agg_preds_adj_arr[id] for id in validation_ids}
    metrics = compute_performance_metrics(agg_preds_adj, validation_census)
    log_dict = {}
    for key,value in metrics.items():
        log_dict["adjusted/"+key] = value
    # log_dict = {"adjusted/r2": r2_adj, "adjusted/mae": mae_adj, "adjusted/mse": mse_adj, "adjusted/mape": mape_adj} 

    predicted_target_img = predicted_target_img.cpu()
    return predicted_target_img_adjusted.cpu(), log_dict
    

def eval_my_model(mynet, guide_img, valid_mask, validation_regions,
    valid_validation_ids, num_validation_ids, validation_ids, validation_census,
    disaggregation_data=None, return_scale=False,
    dataset_name="unspecifed_dataset"):

    res = {}

    with torch.no_grad():
        mynet.eval()

        # batchwise passing for whole image
        return_vals = mynet.forward_batchwise(
            guide_img,
            predict_map=True,
            return_scale=return_scale,
            forward_only=True
        )
        if return_scale:
            predicted_target_img, scales = return_vals
            res["scales"] = scales.squeeze()
        else:
            predicted_target_img = return_vals
        
        # replace masked values with the mean value, this way the artefacts when upsampling are mitigated
        predicted_target_img[~valid_mask] = 1e-10

        res["predicted_target_img"] = predicted_target_img

        # Aggregate by fine administrative boundary
        agg_preds_arr = compute_accumulated_values_by_region(
            validation_regions.numpy().astype(np.uint32),
            predicted_target_img.cpu().numpy().astype(np.float32),
            valid_validation_ids.numpy().astype(np.uint32),
            num_validation_ids
        )
        agg_preds = {id: agg_preds_arr[id] for id in validation_ids}
        metrics = compute_performance_metrics(agg_preds, validation_census)
        # log_dict = {"r2": r2, "mae": mae, "mse": mse, "mape": mape}

        if disaggregation_data is not None:

            predicted_target_img_adjusted, adj_logs = disag_and_eval_map(predicted_target_img, agg_preds_arr, validation_regions, valid_validation_ids,
                num_validation_ids, validation_ids, validation_census, disaggregation_data)
            metrics.update(adj_logs)

            predicted_target_img_adjusted = predicted_target_img_adjusted.cpu() 
            predicted_target_img = predicted_target_img.cpu()

    return res, metrics


def checkpoint_model(mynet, optimizerstate, epoch, log_dict, dataset_name, best_scores):

    best_r2, best_mae, best_mape, best_r2_adj, best_mae_adj, best_mape_adj = best_scores

    if log_dict["r2"]>best_r2:
        best_r2 = log_dict["r2"]
        log_dict["best_r2"] = best_r2
        torch.save({'model_state_dict':mynet.state_dict(), 'optimizer_state_dict':optimizerstate, 'epoch':epoch, 'log_dict':log_dict},
            'checkpoints/best_r2_{}_{}.pth'.format(dataset_name, wandb.run.name) )
    
    if log_dict["mae"]<best_mae:
        best_mae =log_dict["mae"]
        log_dict["best_mae"] = best_mae
        torch.save({'model_state_dict':mynet.state_dict(), 'optimizer_state_dict':optimizerstate, 'epoch':epoch, 'log_dict':log_dict},
            'checkpoints/best_mae_{}_{}.pth'.format(dataset_name, wandb.run.name) )

    if log_dict["mape"]<best_mape:
        best_mape =log_dict["mape"]
        log_dict["best_mape"] = best_mape
        torch.save({'model_state_dict':mynet.state_dict(), 'optimizer_state_dict':optimizerstate, 'epoch':epoch, 'log_dict':log_dict},
            'checkpoints/best_mape_{}_{}.pth'.format(dataset_name, wandb.run.name) )
    
    if "adjusted/r2" in log_dict.keys() and log_dict["adjusted/r2"]>best_r2_adj:
        best_r2_adj = log_dict["adjusted/r2"]
        log_dict["adjusted/best_r2"] = best_r2_adj
        torch.save({'model_state_dict':mynet.state_dict(), 'optimizer_state_dict':optimizerstate, 'epoch':epoch, 'log_dict':log_dict},
            'checkpoints/best_r2_adj_{}_{}.pth'.format(dataset_name, wandb.run.name) )

    if "adjusted/mae" in log_dict.keys() and log_dict["adjusted/mae"]<best_mae_adj:
        best_mae_adj = log_dict["adjusted/mae"]
        log_dict["adjusted/best_mae"] = best_mae_adj
        torch.save({'model_state_dict':mynet.state_dict(), 'optimizer_state_dict':optimizerstate, 'epoch':epoch, 'log_dict':log_dict},
            'checkpoints/best_mae_adj_{}_{}.pth'.format(dataset_name, wandb.run.name) )


    if "adjusted/mape" in log_dict.keys() and log_dict["adjusted/mape"]<best_mape_adj:
        best_mape_adj = log_dict["adjusted/mape"]
        log_dict["adjusted/best_mape"] = best_mape_adj
        torch.save({'model_state_dict':mynet.state_dict(), 'optimizer_state_dict':optimizerstate, 'epoch':epoch, 'log_dict':log_dict},
            'checkpoints/best_mape_adj_{}_{}.pth'.format(dataset_name, wandb.run.name) )

    best_scores = best_r2, best_mae, best_mape, best_r2_adj, best_mae_adj, best_mape_adj

    return best_scores


def PixAdminTransform(
    training_source,
    validation_data=None,
    disaggregation_data=None,
    params=None,
    save_ds=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### prepare Dataset #########################################################################

    if params["admin_augment"]:
        train_data = MultiPatchDataset(training_source, params['memory_mode'], device, params["validation_split"], params["validation_fold"], params["weights"], params["custom_sampler_weights"])
    else:
        train_data = PatchDataset(training_source, params['memory_mode'], device, params["validation_split"])
    if params["sampler"] in ['custom', 'natural']:
        weights = train_data.all_natural_weights if params["sampler"]=="natural" else train_data.custom_sampler_weights
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=False)
        shuffle = False
    else:
        logging.info(f'Using no weighted sampler') 
        sampler = None
        shuffle = True
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=shuffle, sampler=sampler, num_workers=0)

    # load test data into memory
    for name,v in validation_data.items(): 
        with open(v['vars'], "rb") as f:
            v['memory_vars'] = pickle.load(f)
        with open(v['disag'], "rb") as f:
            v['memory_disag'] = pickle.load(f)
        
        # check if we can reuse the features from the training
        if name in train_data.features:
            v['features_disk'] = train_data.features[name]
        else:
            v['features_disk'] = h5py.File(v["features"], 'r')["features"]


    # Fix all random seeds
    torch.manual_seed(params["random_seed"])
    random.seed(params["random_seed"])
    np.random.seed(params["random_seed"])

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
        mynet = PixTransformNet(channels_in=train_data.num_feats(),
                                weights_regularizer=params['weights_regularizer'],
                                device=device).train().to(device)
    elif params['Net']=='ScaleNet':
            mynet = PixScaleNet(channels_in=train_data.num_feats(),
                            weights_regularizer=params['weights_regularizer'],
                            device=device, loss=params['loss'], kernel_size=params['kernel_size'],
                            dropout=params["dropout"]
                            ).train().to(device)

    if params["optim"]=="adam":
        optimizer = optim.Adam(mynet.params_with_regularizer, lr=params['lr'])
    elif params["optim"]=="adamw":
        optimizer = optim.AdamW(mynet.params_with_regularizer, lr=params['lr'], weight_decay=params["weights_regularizer_adamw"])

    if params["load_state"] is not None:
        checkpoint = torch.load('checkpoints/best_r2_{}.pth'.format(params["load_state"]))
        mynet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    wandb.watch(mynet)

    if params['eval_only']:

        #TODO: if we do not cross validation we can do disagregation,
        
        log_dict = {}
        for test_dataset_name, values in validation_data.items():
            val_census, val_regions, val_map, val_valid_ids, val_map_valid_ids, val_guide_res, val_valid_data_mask = values['memory_vars']
            val_features = values["features_disk"]

            res, this_log_dict = eval_my_model(
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
    batchiter = 0

    # initialize the best score variables
    best_scores, best_val_scores = {}, {}
    for test_dataset_name in validation_data.keys():
        best_scores[test_dataset_name] = [-1e12, 1e12, 1e12, -1e12, 1e12, 1e12]
        best_val_scores[test_dataset_name] = [-1e12, 1e12, 1e12, -1e12, 1e12, 1e12]

    with tqdm(range(0, epochs), leave=True) as tnr:
        for epoch in tnr:
            for sample in tqdm(train_loader):
                optimizer.zero_grad()
                
                # Feed forward the network
                y_pred_list = mynet.forward_one_or_more(sample)
                
                #check if any valid values are there, else skip   
                if y_pred_list is None:
                    continue
                
                # Sum over the census data per patch 
                y_pred = torch.sum(torch.stack([pred*samp[3] for pred,samp in zip(y_pred_list, sample)]))
                y_gt = torch.sum(torch.tensor([samp[1]*samp[3] for samp in sample]))
                
                # Backwards
                loss = myloss(y_pred, y_gt)
                loss.backward()
                optimizer.step()

                itercounter += 1
                batchiter += 1
                torch.cuda.empty_cache()

                if itercounter>=( params['logstep'] ):
                    itercounter = 0

                    # Validate and Test the model and save model
                    log_dict = {}
                    
                    # Validation
                    if params["validation_split"]>0.:
                        for name in training_source.keys():
                            logging.info(f'Validating dataset of {name}')
                            agg_preds,val_census = [],[]
                            for idx in range(len(train_data.Ys_val[name])):
                                X, Y, Mask = train_data.get_single_validation_item(idx, name) 
                                agg_preds.append(mynet.forward(X, Mask, forward_only=True).detach().cpu().numpy())
                                val_census.append(Y.cpu().numpy())

                            metrics = compute_performance_metrics_arrays(np.asarray(agg_preds), np.asarray(val_census))
                            # this_log_dict = {"r2": r2, "mae": mae, "mse": mse, "mape": mape}
                            best_val_scores[test_dataset_name] = checkpoint_model(mynet, optimizer.state_dict(), epoch, metrics, name, best_val_scores[test_dataset_name])
                            for key in metrics.keys():
                                log_dict[name + '/validation/' + key ] = metrics[key]
                            torch.cuda.empty_cache()

                    # Evaluation Model
                    for test_dataset_name, values in validation_data.items():
                        logging.info(f'Testing dataset of {test_dataset_name}')
                        val_census, val_regions, val_map, val_valid_ids, val_map_valid_ids, val_guide_res, val_valid_data_mask = values['memory_vars']
                        val_features = values["features_disk"]
                        
                        _, this_log_dict = eval_my_model(
                            mynet, val_features, val_valid_data_mask, val_regions,
                            val_map_valid_ids, np.unique(val_regions).__len__(), val_valid_ids, val_census, 
                            disaggregation_data=values['memory_disag'],
                            dataset_name=test_dataset_name, return_scale=True
                        )

                        # Model checkpointing and update best scores
                        best_scores[test_dataset_name] = checkpoint_model(mynet, optimizer.state_dict(), epoch, this_log_dict, test_dataset_name, best_scores[test_dataset_name])
                        for key in this_log_dict.keys():
                            log_dict[test_dataset_name+'/'+key] = this_log_dict[key]
                        torch.cuda.empty_cache()

                    log_dict['train/loss'] = loss 
                    log_dict['batchiter'] = batchiter
                    log_dict['epoch'] = epoch

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
        for test_dataset_name, values in validation_data.items():
            val_census, val_regions, val_map, val_valid_ids, val_map_valid_ids, val_guide_res, val_valid_data_mask = values['memory_vars']
            val_features = values["features_disk"]
            this_res, log_dict, best_scores = eval_my_model(
                mynet, val_features, val_valid_data_mask, val_regions,
                val_map_valid_ids, np.unique(val_regions).__len__(), val_valid_ids, val_census, 
                val_map, device,
                best_scores[test_dataset_name], optimizer=optimizer,
                disaggregation_data=values['memory_disag'], return_scale=True,
                dataset_name=test_dataset_name
            )
            for key in this_log_dict.keys():
                log_dict[test_dataset_name+'/'+key] = this_log_dict[key]
            res[test_dataset_name] = this_res
            

    return res