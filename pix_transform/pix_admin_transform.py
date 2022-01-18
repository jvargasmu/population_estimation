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

from pix_transform.evaluation import disag_map, disag_wo_map, disag_and_eval_map, eval_my_model, checkpoint_model

if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm


def PixAdminTransform(
    datalocations,
    train_dataset_name,
    test_dataset_names,
    params):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### prepare Dataset #########################################################################
    # unique_datasets = set(list(validation_data.keys()) + list(training_source.keys()))

    if params["admin_augment"]:
        dataset = MultiPatchDataset(datalocations, train_dataset_name, params["train_level"], params['memory_mode'], device, 
            params["validation_split"], params["validation_fold"], params["weights"], params["custom_sampler_weights"])
    else:
        raise Exception("option not available")
        dataset = PatchDataset(training_source, params['memory_mode'], device, params["validation_split"])

    # Fix all random seeds
    torch.manual_seed(params["random_seed"])
    random.seed(params["random_seed"])
    np.random.seed(params["random_seed"])

    if params["sampler"] in ['custom', 'natural']:
        weights = dataset.all_natural_weights if params["sampler"]=="natural" else dataset.custom_sampler_weights
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=False)
        shuffle = False
    else:
        logging.info(f'Using no weighted sampler') 
        sampler = None
        shuffle = True
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle, sampler=sampler, num_workers=0)

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
                        dropout=params["dropout"],
                        input_scaling=params["input_scaling"], output_scaling=params["output_scaling"],
                        datanames=train_dataset_name
                        ).train().to(device)

    #Optimizer
    if params["optim"]=="adam":
        optimizer = optim.Adam(mynet.params_with_regularizer, lr=params['lr'])
    elif params["optim"]=="adamw":
        optimizer = optim.AdamW(mynet.params_with_regularizer, lr=params['lr'], weight_decay=params["weights_regularizer_adamw"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params["lr_scheduler_step"], gamma=0.5)

    # Load from state
    if params["load_state"] is not None:
        checkpoint = torch.load('checkpoints/best_mape_{}_VAL_{}.pth'.format(test_dataset_names[0], params["load_state"]))
        mynet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    wandb.watch(mynet)

    if params['eval_only']:

        #TODO: CV with 5 models here
        #TODO: evaluate 1 model here
        #TODO: move 5fold feature
        log_dict = {}
        res_dict = {}
        for name in test_dataset_names:
            
            if params["validation_split"]>0. or (params["validation"] is not None):
                
                logging.info(f'Validating dataset of {name}')
                agg_preds,val_census = [],[]
                agg_preds_arr = torch.zeros((dataset.max_tregid_val[name]+1,))

                for idx in tqdm(range(len(dataset.Ys_val[name])), disable=params["silent_mode"]):
                    X, Y, Mask, name, census_id = dataset.get_single_validation_item(idx, name) 
                    agg_preds.append(mynet.forward(X, Mask, name=name, forward_only=True).detach().cpu().numpy())
                    val_census.append(Y.cpu().numpy())

                metrics = compute_performance_metrics_arrays(np.asarray(agg_preds), np.asarray(val_census))
                for key in metrics.keys():
                    log_dict[name + '/validation/' + key ] = metrics[key]
                torch.cuda.empty_cache()

        # Evaluation Model
        for name in test_dataset_names: 

            logging.info(f'Testing dataset of {name}')
            val_census, val_regions, val_map, val_valid_ids, val_map_valid_ids, _, val_valid_data_mask, _, _ = dataset.memory_vars[name]
            val_features = dataset.features[name]
            
            res, this_log_dict = eval_my_model(
                mynet, val_features, val_valid_data_mask, val_regions,
                val_map_valid_ids, np.unique(val_regions).__len__(), val_valid_ids, val_census,
                dataset=dataset,
                disaggregation_data=dataset.memory_disag[name],
                dataset_name=name, return_scale=True, full_eval=True
            )

            log_images = False
            if log_images:
                if len(res['scales'].shape)==3:
                    this_log_dict["viz/scales"] = wandb.Image(res['scales'][0])
                    this_log_dict["viz/scales_var"] = wandb.Image(res['scales'][1])
                    this_log_dict["viz/predicted_target_img"] = wandb.Image(res['predicted_target_img'])
                    this_log_dict["viz/predicted_target_img_var"] = wandb.Image(res['variances'])
                    this_log_dict["viz/predicted_target_img_adjusted"] = wandb.Image(res['predicted_target_img_adjusted'])

            # Model log collection
            for key in res.keys():
                res_dict[name+'/'+key] = res[key]

            for key in this_log_dict.keys():
                log_dict[name+'/'+key] = this_log_dict[key]

            torch.cuda.empty_cache()

        # log_dict['train/loss'] = loss 
        log_dict['batchiter'] = 0
        log_dict['epoch'] = 0
        wandb.log(log_dict)
            
        mynet.train() 
        torch.cuda.empty_cache()
                        
        return res_dict, log_dict
    
    #### train network ############################################################################

    epochs = params["epochs"]
    itercounter = 0
    batchiter = 0

    # initialize the best score variables
    best_scores, best_val_scores = {}, {}
    for test_dataset_name in test_dataset_names:
        best_scores[test_dataset_name] = [-1e12, 1e12, 1e12, -1e12, 1e12, 1e12]
        best_val_scores[test_dataset_name] = [-1e12, 1e12, 1e12, -1e12, 1e12, 1e12]

    with tqdm(range(0, epochs), leave=True, disable=params["silent_mode"]) as tnr:
        for epoch in tnr:
            for sample in tqdm(train_loader, disable=params["silent_mode"]):
                optimizer.zero_grad()
                
                # Feed forward the network
                y_pred_list = mynet.forward_one_or_more(sample)
                
                #check if any valid values are there, else skip   
                if y_pred_list is None:
                    continue
                
                # Sum over the census data per patch 
                y_pred = torch.stack([pred*samp[4] for pred,samp in zip(y_pred_list, sample)]).sum(0)
                y_gt = torch.tensor([samp[1]*samp[4] for samp in sample]).sum()

                # Backwards
                loss = myloss(y_pred, y_gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mynet.parameters(), params["grad_clip"])
                optimizer.step()
                scheduler.step()

                # logging
                train_log_dict = {}
                if batchiter % 50 == 0: 
                    if len(y_pred)==2:
                        train_log_dict["train/y_pred_"] = y_pred[0]
                        train_log_dict["train/y_var"] = y_pred[1]
                    else:
                        train_log_dict["train/y_pred"] = y_pred
                    train_log_dict["train/y_gt"] = y_gt
                    train_log_dict['train/loss'] = loss 
                    train_log_dict['epoch'] = epoch 
                    train_log_dict['batchiter'] = batchiter
                    train_log_dict['current_lr'] = optimizer.param_groups[0]["lr"]
                    wandb.log(train_log_dict)

                itercounter += 1
                batchiter += 1

                if mynet.output_scaling:
                    mynet.normalize_out_scales()

                torch.cuda.empty_cache()

                if itercounter>=( params['logstep'] ):
                    itercounter = 0

                    # Validate and Test the model and save model
                    log_dict = {}
                    
                    # Validation
                    if params["validation_split"]>0. or (params["validation_fold"] is not None):
                        for name in test_dataset_names:
                            logging.info(f'Validating dataset of {name}')
                            agg_preds,val_census = [],[]
                            agg_preds_arr = torch.zeros((dataset.max_tregid[name]+1,))

                            for idx in tqdm(range(len(dataset.Ys_val[name])), disable=params["silent_mode"]):
                                X, Y, Mask, name, census_id = dataset.get_single_validation_item(idx, name)
                                pred = mynet.forward(X, Mask, name=name, forward_only=True).detach().cpu().numpy()
                                agg_preds.append(pred)
                                val_census.append(Y.cpu().numpy())
                                if isinstance(pred, np.ndarray):
                                    pred = pred[0] 
                                agg_preds_arr[census_id.item()] = pred.item()

                            metrics = compute_performance_metrics_arrays(np.asarray(agg_preds), np.asarray(val_census)) 
                            best_val_scores[name] = checkpoint_model(mynet, optimizer.state_dict(), epoch, metrics, '/'+name+'/VAL/', best_val_scores[name])
                            for key in metrics.keys():
                                log_dict[name + '/validation/' + key ] = metrics[key]
                            
                            # Disaggregation per coarse val census
                            agg_preds_arr_adj, this_metrics_dis = disag_wo_map(agg_preds_arr, dataset.memory_disag_val[name])
                            for key,value in this_metrics_dis.items():
                                log_dict[name + "/validation/adjusted/coarse/"+key] = value  
                                
                            this_metrics = compute_performance_metrics_arrays(agg_preds_arr_adj[dataset.tregid_val[name]].numpy(), np.asarray(val_census))  
                            for key,value in this_metrics.items():
                                log_dict[name + "/validation/adjusted/coarse/"+key] = value  

                            # "fake" new dissagregation data and reuse the function
                            # Do the disagregation on country level
                            tts = torch.zeros(dataset.memory_disag_val[name][0].shape, dtype=int)
                            tts[torch.where(dataset.memory_disag_val[name][0])] = 1
                            disaggregation_data_coarsest_val = [tts, {1: sum(list(dataset.memory_disag_val[name][1].values()))}, dataset.memory_disag_val[name][2] ]
                        
                            agg_preds_arr_country_adj, this_metrics_cl = disag_wo_map(agg_preds_arr, disaggregation_data_coarsest_val)
                            for key,value in this_metrics_cl.items():
                                log_dict[name + "/validation/adjusted/country_like/"+key] = value  
                            # agg_preds_arr_adj[dataset.tregid_val[name]]
                            this_metrics_cl = compute_performance_metrics_arrays(agg_preds_arr_country_adj[dataset.tregid_val[name]].numpy(), np.asarray(val_census))  
                            for key,value in this_metrics_cl.items():
                                log_dict[name + "/validation/adjusted/country_like/"+key] = value  
                            
                            torch.cuda.empty_cache()

                    # Evaluation Model: Evaluates the training and validation regions at the same time.
                    # for test_dataset_name, values in validation_data.items():
                    for name in test_dataset_names: 

                        logging.info(f'Testing dataset of {name}')
                        val_census, val_regions, val_map, val_valid_ids, val_map_valid_ids, _, val_valid_data_mask, _, _ = dataset.memory_vars[name]
                        val_features = dataset.features[name]
                        
                        res, this_log_dict = eval_my_model(
                            mynet, val_features, val_valid_data_mask, val_regions,
                            val_map_valid_ids, np.unique(val_regions).__len__(), val_valid_ids, val_census,
                            dataset=dataset,
                            disaggregation_data=dataset.memory_disag[name],
                            dataset_name=name, return_scale=True, silent_mode=params["silent_mode"]
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
                        best_scores[name] = checkpoint_model(mynet, optimizer.state_dict(), epoch, this_log_dict,  '/'+name+'/ALL/', best_scores[test_dataset_name])
                        for key in this_log_dict.keys():
                            log_dict[name+'/'+key] = this_log_dict[key]
                        torch.cuda.empty_cache()

                    # log_dict['train/loss'] = loss 
                    log_dict['batchiter'] = batchiter
                    log_dict['epoch'] = epoch

                    # if val_fine_map is not None:
                    tnr.set_postfix(R2=log_dict[test_dataset_names[-1]+'/r2'],
                                    zMAEc=log_dict[test_dataset_names[-1]+'/mae'])
                    wandb.log(log_dict)
                        
                    mynet.train() 
                    torch.cuda.empty_cache()

                    if batchiter>=params["maxstep"]:
                        maxstep_reached = True
                        break
            else:
                # Continue if the inner loop was not broken.
                continue
            break

    # compute final prediction, un-normalize, and back to numpy
    with torch.no_grad():
        mynet.eval()
        
        Path('checkpoints/{}'.format('Final')).mkdir(parents=True, exist_ok=True) 

        saved_dict = {'model_state_dict': mynet.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch, 'log_dict': log_dict}
        if mynet.input_scaling:
            saved_dict["input_scales_bias"] = [mynet.in_scale, mynet.in_bias]
        if mynet.output_scaling:
            saved_dict["output_scales_bias"] = [mynet.out_scale, mynet.out_bias] 

        torch.save(saved_dict,
            'checkpoints/{}{}.pth'.format('Final/Maxstepstate_', wandb.run.name) )
        torch.cuda.empty_cache()

        # Validate and Test the model and save model
        log_dict = {}
        res_dict = {}

        # Evaluation Model
        # for test_dataset_name, values in validation_data.items():
        for name in test_dataset_names: 

            logging.info(f'Testing dataset of {name}')
            val_census, val_regions, val_map, val_valid_ids, val_map_valid_ids, _, val_valid_data_mask, _, _ = dataset.memory_vars[name]
            val_features = dataset.features[name]
            
            res, this_log_dict = eval_my_model(
                mynet, val_features, val_valid_data_mask, val_regions,
                val_map_valid_ids, np.unique(val_regions).__len__(), val_valid_ids, val_census,
                dataset=dataset,
                disaggregation_data=dataset.memory_disag[name],
                dataset_name=name, return_scale=True, silent_mode=params["silent_mode"], full_eval=True
            )

            # Model log collection
            for key in res.keys():
                res_dict[name+'/'+key] = res[key]

            for key in this_log_dict.keys():
                log_dict[name+'/'+key] = this_log_dict[key]
                
        wandb.log(log_dict)
        
    return res_dict, log_dict 
