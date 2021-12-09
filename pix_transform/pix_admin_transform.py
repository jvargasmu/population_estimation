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

from bayesian_dl.loss import GaussianNLLLoss, LaplacianNLLLoss


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


    scalings_array = torch.tensor(list(scalings.values())).numpy()
    log_dict = {
    "disaggregation/scalings_": wandb.Histogram(scalings_array), "disaggregation/mean_scaling": np.mean(scalings_array),
    "disaggregation/median_scaling": np.median(scalings_array), "disaggregation/min_scaling": np.min(scalings_array),
    "disaggregation/max_scaling": np.max(scalings_array)  }

    return predicted_target_img_adjusted.cpu(), log_dict

def disag_wo_map(agg_preds_arr, disaggregation_data):

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unfold disagg data
    target_to_source, source_census, source_regions = disaggregation_data
 
    # agg_preds_cr_arr = np.zeros((target_to_source.unique().max()+1).type(torch.int).item())
    agg_preds_cr_arr = np.zeros((target_to_source.unique().max()+1))
    for finereg in target_to_source.unique(): 
        agg_preds_cr_arr[finereg] = agg_preds_arr[target_to_source==finereg].sum()
    
    agg_preds_cr = {id: agg_preds_cr_arr[id] for id in source_census.keys()}

    scalings = {}
    for id in source_census.keys():
        if agg_preds_cr[id]==0:
            scalings[id] = torch.tensor(1)
        else:
            scalings[id] = torch.tensor(source_census[id]/agg_preds_cr[id])
    # scalings = { torch.tensor(1) if agg_preds_cr[id]==0 else id: torch.tensor(source_census[id]/agg_preds_cr[id]) for id in source_census.keys()}
    
    agg_preds_arr_adj = agg_preds_arr.clone()

    for id,s in scalings.items():
        mask = target_to_source==id 
        agg_preds_arr_adj[mask] = agg_preds_arr[mask] * s

    scalings_array = torch.tensor(list(scalings.values())).numpy()
    log_dict = {
    "disaggregation/scalings_": wandb.Histogram(scalings_array), "disaggregation/mean_scaling": np.mean(scalings_array),
    "disaggregation/median_scaling": np.median(scalings_array), "disaggregation/min_scaling": np.min(scalings_array),
    "disaggregation/max_scaling": np.max(scalings_array)  }

    return agg_preds_arr_adj, log_dict


def disag_and_eval_map(predicted_target_img, agg_preds_arr, validation_regions, valid_validation_ids,
    num_validation_ids, validation_ids, validation_census, disaggregation_data):

    predicted_target_img_adjusted, log_dict = disag_map(predicted_target_img, agg_preds_arr, disaggregation_data)
    
    # Aggregate by fine administrative boundary
    agg_preds_adj_arr = compute_accumulated_values_by_region(
        validation_regions.numpy().astype(np.uint32),
        predicted_target_img_adjusted.cpu().numpy().astype(np.float32),
        valid_validation_ids.numpy().astype(np.uint32),
        num_validation_ids
    )
    agg_preds_adj = {id: agg_preds_adj_arr[id] for id in validation_ids} 
    predicted_target_img = predicted_target_img.cpu()
   
    metrics = compute_performance_metrics(agg_preds_adj, validation_census)
    for key,value in metrics.items():
        log_dict["adjusted/"+key] = value 

    return predicted_target_img_adjusted.cpu(), log_dict
    

def eval_my_model(mynet, guide_img, valid_mask, validation_regions,
    valid_validation_ids, num_validation_ids, validation_ids, validation_census,
    dataset,
    disaggregation_data=None, return_scale=False,
    dataset_name="unspecifed_dataset",
    full_eval=False):

    res = {}
    metrics = {}

    with torch.no_grad():
        mynet.eval()
        
        if full_eval:

            # batchwise passing for whole image
            logging.info(f'Classic eval started')
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

            if len(predicted_target_img.shape)==3:
                res["variances"] = predicted_target_img[1]
                predicted_target_img = predicted_target_img[0]
            
            # replace masked values with the mean value, this way the artefacts when upsampling are mitigated
            predicted_target_img[~valid_mask] = 1e-16

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
            logging.info(f'Classic eval finished')

            if disaggregation_data is not None:

                logging.info(f'Classic disag started') 
                predicted_target_img_adjusted, adj_logs = disag_and_eval_map(predicted_target_img, agg_preds_arr, validation_regions, valid_validation_ids,
                    num_validation_ids, validation_ids, validation_census, disaggregation_data)
                metrics.update(adj_logs)
                logging.info(f'Classic disag finsihed')

                res["predicted_target_img_adjusted"] = predicted_target_img_adjusted.cpu()  
                predicted_target_img_adjusted = predicted_target_img_adjusted.cpu()
                predicted_target_img = predicted_target_img.cpu()

        else:
            # Fast evaluation pipeline
            logging.info(f'Samplewise eval started')
            agg_preds2 = {}
            agg_preds_arr = torch.zeros((dataset.max_tregid[dataset_name]+1,))
            for idx in tqdm(range(dataset.len_all_samples(dataset_name))):
                X, Y, Mask, census_id = dataset.get_single_item(idx, dataset_name) 
                prediction = mynet.forward(X, Mask, forward_only=True).detach().cpu().numpy()

                if isinstance(prediction, np.ndarray):
                    prediction = prediction[0]
                agg_preds2[census_id.item()] = prediction.item()
                agg_preds_arr[census_id.item()] = prediction.item()

            agg_preds3 = {id: agg_preds_arr[id].item() for id in validation_ids}

            for cid in validation_census.keys():
                if cid not in agg_preds3.keys():
                    agg_preds3[cid] = 0

            this_metrics = compute_performance_metrics(agg_preds3, validation_census)
            metrics.update(this_metrics)
            logging.info(f'Samplewise eval finished')

            if disaggregation_data is not None: 
                logging.info(f'Fast disag started') 

                for cid in validation_regions.unique():
                    if cid.item() not in agg_preds3.keys():
                        agg_preds3[cid.item()] = 0

                # Do the disagregation without the map
                agg_preds_arr_adj, log_dict = disag_wo_map(agg_preds_arr, disaggregation_data)
                for key,value in log_dict.items():
                    metrics["adjusted/coarse/"+key] = value 
                logging.info(f'Fast disag finished') 
                agg_preds_adj = {id: agg_preds_arr_adj[id].item() for id in validation_ids}                
                this_metrics = compute_performance_metrics(agg_preds_adj, validation_census)
                for key,value in this_metrics.items():
                    metrics["adjusted/coarse/"+key] = value  

                # "fake" new dissagregation data and reuse the function
                # Do the disagregation on country level 
                disaggregation_data_coarsest = \
                    [torch.zeros(disaggregation_data[0].shape, dtype=int), {0: sum(list(disaggregation_data[1].values()))}, disaggregation_data[2] ]
              
                agg_preds_arr_country_adj, log_dict = disag_wo_map(agg_preds_arr, disaggregation_data_coarsest)
                for key,value in log_dict.items():
                    metrics["adjusted/country/"+key] = value 
                metrics["country/pred"] = agg_preds_arr.sum()
                metrics["country/gt"] = disaggregation_data_coarsest[1][0]

                # metrics.update(log_dict)
                agg_preds_country_adj = {id: agg_preds_arr_country_adj[id].item() for id in validation_ids}                
                this_metrics = compute_performance_metrics(agg_preds_country_adj, validation_census)
                for key,value in this_metrics.items():
                    metrics["adjusted/country/"+key] = value  

            logging.info(f'fast disag finished')

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
    datalocations,
    train_dataset_name,
    test_dataset_names,
    params,  
    save_ds=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### prepare Dataset #########################################################################
    # unique_datasets = set(list(validation_data.keys()) + list(training_source.keys()))

    if params["admin_augment"]:
        dataset = MultiPatchDataset(datalocations, train_dataset_name, params["train_level"], params['memory_mode'], device, 
            params["validation_split"], params["validation_fold"], params["weights"], params["custom_sampler_weights"])
    else:
        dataset = PatchDataset(training_source, params['memory_mode'], device, params["validation_split"])
    if params["sampler"] in ['custom', 'natural']:
        weights = dataset.all_natural_weights if params["sampler"]=="natural" else dataset.custom_sampler_weights
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=False)
        shuffle = False
    else:
        logging.info(f'Using no weighted sampler') 
        sampler = None
        shuffle = True
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle, sampler=sampler, num_workers=0)

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
    elif params['loss'] == 'gaussNLL':
        myloss = GaussianNLLLoss(max_clamp=20.)
    elif params['loss'] == 'laplaceNLL':
        myloss = LaplacianNLLLoss(max_clamp=20.)
    else:
        raise Exception("unknown loss!")
        
    if params['Net']=='PixNet':
        mynet = PixTransformNet(channels_in=dataset.num_feats(),
                                weights_regularizer=params['weights_regularizer'],
                                device=device).train().to(device)
    elif params['Net']=='ScaleNet':
            mynet = PixScaleNet(channels_in=dataset.num_feats(),
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

        #TODO: CV with 5 models here
        #TODO: evaluate 1 model here
        log_dict = {}
        for name in test_dataset_names:
                
            for test_dataset_name, values in validation_data.items():
                val_census, val_regions, val_map, val_valid_ids, val_map_valid_ids, val_guide_res, val_valid_data_mask = values['memory_vars']
                val_features = values["features_disk"]

                # res, this_log_dict = eval_my_model(
                #     mynet, val_features, val_valid_data_mask, val_regions,
                #     val_map_valid_ids, np.unique(val_regions).__len__(), val_valid_ids, val_census, 
                #     val_map, device,
                #     disaggregation_data=disaggregation_data[test_dataset_name],
                #     # fine_to_cr, val_cr_census, val_cr_regions,
                #     return_scale=True, dataset_name=test_dataset_name
                # )

                res, this_log_dict = eval_my_model(
                    mynet, val_features, val_valid_data_mask, val_regions,
                    val_map_valid_ids, np.unique(val_regions).__len__(), val_valid_ids, val_census, 
                    disaggregation_data=values['memory_disag'],
                    dataset_name=test_dataset_name, return_scale=True
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
    for test_dataset_name in test_dataset_names:
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
                y_pred = torch.stack([pred*samp[3] for pred,samp in zip(y_pred_list, sample)]).sum(0)
                y_gt = torch.tensor([samp[1]*samp[3] for samp in sample]).sum()

                # Backwards
                loss = myloss(y_pred, y_gt)
                loss.backward()
                optimizer.step()

                # logging
                train_log_dict = {}
                if len(y_pred)==2:
                    train_log_dict["train/y_pred_"] = y_pred[0]
                    train_log_dict["train/y_var"] = y_pred[1]
                else:
                    train_log_dict["train/y_pred"] = y_pred
                train_log_dict["train/y_gt"] = y_gt
                train_log_dict['train/loss'] = loss 
                train_log_dict['batchiter'] = batchiter
                train_log_dict['epoch'] = epoch 
                wandb.log(train_log_dict)

                itercounter += 1
                batchiter += 1
                torch.cuda.empty_cache()

                if itercounter>=( params['logstep'] ):
                    itercounter = 0

                    # Validate and Test the model and save model
                    log_dict = {}
                    
                    # Validation
                    if params["validation_split"]>0. or (params["validation"] is not None):
                        for name in test_dataset_names:
                            logging.info(f'Validating dataset of {name}')
                            agg_preds,val_census = [],[]
                            for idx in tqdm(range(len(dataset.Ys_val[name]))):
                                X, Y, Mask = dataset.get_single_validation_item(idx, name) 
                                agg_preds.append(mynet.forward(X, Mask, forward_only=True).detach().cpu().numpy())
                                val_census.append(Y.cpu().numpy())

                            metrics = compute_performance_metrics_arrays(np.asarray(agg_preds), np.asarray(val_census))
                            # this_log_dict = {"r2": r2, "mae": mae, "mse": mse, "mape": mape}
                            best_val_scores[name] = checkpoint_model(mynet, optimizer.state_dict(), epoch, metrics, name, best_val_scores[name])
                            for key in metrics.keys():
                                log_dict[name + '/validation/' + key ] = metrics[key]
                            torch.cuda.empty_cache()

                    # Evaluation Model
                    # for test_dataset_name, values in validation_data.items():
                    for name in test_dataset_names: 

                        logging.info(f'Testing dataset of {name}')
                        val_census, val_regions, val_map, val_valid_ids, val_map_valid_ids, val_guide_res, val_valid_data_mask = dataset.memory_vars[name]
                        val_features = dataset.features[name]
                        
                        res, this_log_dict = eval_my_model(
                            mynet, val_features, val_valid_data_mask, val_regions,
                            val_map_valid_ids, np.unique(val_regions).__len__(), val_valid_ids, val_census,
                            dataset=dataset,
                            disaggregation_data=dataset.memory_disag[name],
                            dataset_name=name, return_scale=True
                        )
 
                        log_images = False
                        if log_images:
                            if len(res['scales'].shape)==3:
                                this_log_dict["viz/scales"] = wandb.Image(res['scales'][0])
                                this_log_dict["viz/scales_var"] = wandb.Image(res['scales'][1])
                                this_log_dict["viz/predicted_target_img"] = wandb.Image(res['predicted_target_img'])
                                this_log_dict["viz/predicted_target_img_var"] = wandb.Image(res['variances'])
                                this_log_dict["viz/predicted_target_img_adjusted"] = wandb.Image(res['predicted_target_img_adjusted'])

                        # Model checkpointing and update best scores
                        best_scores[name] = checkpoint_model(mynet, optimizer.state_dict(), epoch, this_log_dict, name, best_scores[test_dataset_name])
                        for key in this_log_dict.keys():
                            log_dict[name+'/'+key] = this_log_dict[key]
                        torch.cuda.empty_cache()

                    # log_dict['train/loss'] = loss 
                    log_dict['batchiter'] = batchiter
                    log_dict['epoch'] = epoch

                    # if val_fine_map is not None:
                    tnr.set_postfix(R2=log_dict[list(datalocations.keys())[-1]+'/r2'],
                                    zMAEc=log_dict[list(datalocations.keys())[-1]+'/mae'])
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