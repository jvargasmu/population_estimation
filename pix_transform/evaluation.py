

import logging

#from torch._C import AliasDb
logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import numpy as np
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
from tqdm import tqdm
import os
import pdb

from utils import plot_2dmatrix, accumulate_values_by_region, compute_performance_metrics, bbox2, \
     PatchDataset, MultiPatchDataset, NormL1, LogL1, LogoutputL1, LogoutputL2, compute_performance_metrics_arrays
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region

from pix_transform.pix_transform_net import PixScaleNet
import config_pop as cfg


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
        # finregs_to_sum = torch.nonzero(target_to_source==finereg)
        agg_preds_cr_arr[finereg] = agg_preds_arr[target_to_source==finereg].sum()
    
    agg_preds_cr = {id: agg_preds_cr_arr[id] for id in source_census.keys()}
    scalings = {id: torch.tensor(source_census[id]/agg_preds_cr[id]).to(device) for id in source_census.keys()}

    for idx in (scalings.keys()):
        mask = [source_regions==idx]
        if not scalings[idx].isnan() and (not scalings[idx].isinf()):
            predicted_target_img_adjusted[mask] = predicted_target_img[mask]*scalings[idx]
        else:
            predicted_target_img[mask] = predicted_target_img[mask]*scalings[idx]
            scalings[idx] = 99


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
    full_eval=False, silent_mode=True):

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
            
            if len(predicted_target_img.shape)==4: #TODO: verify in which case len(predicted_target_img.shape) == 3
                if predicted_target_img.shape[1] > 1:
                    res["variances"] = predicted_target_img[0, 1]
                predicted_target_img = predicted_target_img[0, 0]
            
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
            # agg_preds2 = {}
            agg_preds_arr = torch.zeros((dataset.max_tregid[dataset_name]+1,))
            for idx in tqdm(range(dataset.len_all_samples(dataset_name)), disable=silent_mode):
                X, Y, Mask, name, census_id = dataset.get_single_item(idx, dataset_name) 
                prediction = mynet.forward(X, Mask, name=name, forward_only=True).detach().cpu().numpy()

                if isinstance(prediction, np.ndarray) and prediction.shape.__len__()==1:
                    prediction = prediction[0]
                # agg_preds2[census_id.item()] = prediction.item()
                agg_preds_arr[census_id.item()] = prediction.item()
                torch.cuda.empty_cache()

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
    
    # if "in_scale" in dir(mynet):
    #     if name in mynet.in_scale.keys():
    #         in_scales = mynet.in_scale[name][0,:,0,0].detach().cpu().numpy()
    #         in_biases = mynet.in_bias[name][0,:,0,0].detach().cpu().numpy() 

    #     elif "mean_in_scale" in dir(mynet): 
    #         in_scales = mynet.mean_in_scale[0,:,0,0].detach().cpu().numpy()
    #         in_biases = mynet.mean_in_bias[0,:,0,0].detach().cpu().numpy()

    #     for i,(sc,bi) in enumerate(zip(in_scales, in_biases)):
    #         if mynet.pop_target:
    #             fname = dataset.feature_names[name][i]
    #         else:
    #             fname = dataset.feature_names[name][i+1]
    #         metrics["input_scaling/"+fname] = sc
    #         metrics["input_bias/"+fname] = bi

    # if "out_scale" in dir(mynet): 
    #     if name in mynet.out_scale.keys():
    #         metrics["output_scaling/output_scaling"] = mynet.out_scale[name].detach().cpu().numpy()
    #         metrics["output_scaling/output_bias"] = mynet.out_bias[name].detach().cpu().numpy() 

    #     elif "mean_out_scale" in dir(mynet): 
    #         metrics["output_scaling/output_scaling"] = mynet.mean_out_scale.detach().cpu().numpy()
    #         metrics["output_scaling/output_bias"] = mynet.mean_out_bias.detach().cpu().numpy()

    return res, metrics

def log_scales(mynet, datalocations, dataset, metrics):
    for name in datalocations:
        if "in_scale" in dir(mynet):
            if name in mynet.in_scale.keys():
                in_scales = mynet.in_scale[name][0,:,0,0].detach().cpu().numpy()
                in_biases = mynet.in_bias[name][0,:,0,0].detach().cpu().numpy() 

            elif "mean_in_scale" in dir(mynet): 
                in_scales = mynet.mean_in_scale[0,:,0,0].detach().cpu().numpy()
                in_biases = mynet.mean_in_bias[0,:,0,0].detach().cpu().numpy()

            for i,(sc,bi) in enumerate(zip(in_scales, in_biases)):
                if mynet.pop_target:
                    fname = dataset.feature_names[name][i]
                else:
                    fname = dataset.feature_names[name][i+1]
                metrics[name + "/input_scaling/"+fname] = sc
                metrics[name + "/input_bias/"+fname] = bi

        if "out_scale" in dir(mynet): 
            if name in mynet.out_scale.keys():
                metrics[name + "/output_scaling/output_scaling"] = mynet.out_scale[name].detach().cpu().numpy()
                metrics[name + "/output_scaling/output_bias"] = mynet.out_bias[name].detach().cpu().numpy() 

            elif "mean_out_scale" in dir(mynet): 
                metrics[name + "/output_scaling/output_scaling"] = mynet.mean_out_scale.detach().cpu().numpy()
                metrics[name + "/output_scaling/output_bias"] = mynet.mean_out_bias.detach().cpu().numpy()
    
    return metrics



def checkpoint_model(mynet, optimizerstate, epoch, log_dict, dataset_name, best_scores):

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    Path('checkpoints/best_r2{}'.format(dataset_name)).mkdir(parents=True, exist_ok=True)
    Path('checkpoints/best_mae{}'.format(dataset_name)).mkdir(parents=True, exist_ok=True)
    Path('checkpoints/best_mape{}'.format(dataset_name)).mkdir(parents=True, exist_ok=True)
    Path('checkpoints/best_r2_adj{}'.format(dataset_name)).mkdir(parents=True, exist_ok=True)
    Path('checkpoints/best_mae_adj{}'.format(dataset_name)).mkdir(parents=True, exist_ok=True)
    Path('checkpoints/best_mape_adj{}'.format(dataset_name)).mkdir(parents=True, exist_ok=True) 
    
    best_r2, best_mae, best_mape, best_r2_adj, best_mae_adj, best_mape_adj = best_scores

    saved_dict = {'model_state_dict': mynet.state_dict(), 'optimizer_state_dict': optimizerstate, 'epoch': epoch, 'log_dict': log_dict}
    if mynet.input_scaling:
        saved_dict["input_scales_bias"] = [mynet.in_scale, mynet.in_bias]
    if mynet.output_scaling:
        saved_dict["output_scales_bias"] = [mynet.out_scale, mynet.out_bias] 

    if log_dict["r2"]>best_r2:
        best_r2 = log_dict["r2"]
        log_dict["best_r2"] = best_r2
        torch.save(saved_dict,
            'checkpoints/best_r2{}{}.pth'.format(dataset_name, wandb.run.name) )
    
    if log_dict["mae"]<best_mae:
        best_mae = log_dict["mae"]
        log_dict["best_mae"] = best_mae
        torch.save(saved_dict,
            'checkpoints/best_mae{}{}.pth'.format(dataset_name, wandb.run.name) )

    if log_dict["mape"]<best_mape:
        best_mape = log_dict["mape"]
        log_dict["best_mape"] = best_mape
        torch.save(saved_dict,
            'checkpoints/best_mape{}{}.pth'.format(dataset_name, wandb.run.name) )
    
    if "adjusted/r2" in log_dict.keys() and log_dict["adjusted/r2"]>best_r2_adj:
        best_r2_adj = log_dict["adjusted/r2"]
        log_dict["adjusted/best_r2"] = best_r2_adj
        torch.save(saved_dict,
            'checkpoints/best_r2_adj{}{}.pth'.format(dataset_name, wandb.run.name) )

    if "adjusted/mae" in log_dict.keys() and log_dict["adjusted/mae"]<best_mae_adj:
        best_mae_adj = log_dict["adjusted/mae"]
        log_dict["adjusted/best_mae"] = best_mae_adj
        torch.save(saved_dict,
            'checkpoints/best_mae_adj{}{}.pth'.format(dataset_name, wandb.run.name) )

    if "adjusted/mape" in log_dict.keys() and log_dict["adjusted/mape"]<best_mape_adj:
        best_mape_adj = log_dict["adjusted/mape"]
        log_dict["adjusted/best_mape"] = best_mape_adj
        torch.save(saved_dict,
            'checkpoints/best_mape_adj{}{}.pth'.format(dataset_name, wandb.run.name) )

    best_scores = best_r2, best_mae, best_mape, best_r2_adj, best_mae_adj, best_mape_adj

    return best_scores


def eval_generic_model(datalocations, train_dataset_name,  test_dataset_names, params, Mynets, Datasets, memory_vars):
    
    log_dict = {}
    res_dict = {}
    
    for name in test_dataset_names: 

        agg_preds = []
        val_census = []
        pop_ests = []
        agg_preds_arr = torch.zeros((Datasets[0].max_tregid[name]+1,))
        BBoxes = []
        census_ids = []
        Scales = [] 

        val_census, val_regions, val_map, _, val_valid_ids, val_map_valid_ids, _, val_valid_data_mask, _, _, _ = memory_vars[name]
        res = {}

        guide_res = memory_vars[name][6] 
        res["predicted_target_img"] = torch.zeros(guide_res, dtype=torch.float16)
        res["variances"] = torch.zeros(guide_res, dtype=torch.float16) 
        res["scales"] = torch.zeros((2,)+guide_res, dtype=torch.float16)
        res["scales"][:] = float('nan')
        res["fold_map"] = torch.zeros(guide_res, dtype=torch.float16)
        res["fold_map"][:] = float('nan')
        res["id_map"] = torch.zeros(guide_res, dtype=torch.float16)
        res["fold_map"][:] = float('nan')

        logging.info(f'Cross Validating dataset of {name}')

        for k in range(5):

            dataset = Datasets[k]
            mynet = Mynets[k]

            with torch.no_grad():
                mynet.eval()
                val_census_list = []

                for idx in tqdm(range(len(dataset.Ys_hout[name])), disable=params["silent_mode"]):
                    X, Y, Mask, name, census_id, BB, regMasks = dataset.get_single_holdout_item(idx, name, return_BB=True) 
                    pop_est, scale = mynet.forward(X, mask=None, name=name, predict_map=True, forward_only=True)

                    rmin, rmax, cmin, cmax = BB

                    res["predicted_target_img"][rmin:rmax, cmin:cmax][Mask] = pop_est[:,0,Mask].to(torch.float16)
                    if pop_est.shape[1]==2:
                        res["variances"][rmin:rmax, cmin:cmax][Mask] = pop_est[:,1,Mask].to(torch.float16)  
                    res["scales"][:,rmin:rmax, cmin:cmax][:,regMasks] = scale[0,:,regMasks].to(torch.float16)
                    # res["scales"][:,rmin:rmax, cmin:cmax] = scale[0,:].to(torch.float16)
                    res["fold_map"][rmin:rmax, cmin:cmax][torch.tensor(regMasks)] = 1.0 * k  
                    res["id_map"][rmin:rmax, cmin:cmax][regMasks] = census_id.to(torch.float16) #1.0 * census_id
                    
                    pred = pop_est[0,0,Mask].sum().detach().cpu().numpy()
                    agg_preds.append(pred)
                    val_census_list.append(Y.cpu().numpy())
                    agg_preds_arr[census_id.item()] = pop_est[0,0,Mask].sum().detach().cpu().item()
                    
                    census_ids.append(census_id)
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()
        
        res["scales"][torch.isinf(res["scales"])] = np.nan

        # calculate this for all folds
        agg_preds3 = {id: agg_preds_arr[id].item() for id in val_valid_ids}
        metrics = compute_performance_metrics(agg_preds3, val_census) 
        for key in metrics.keys():
            log_dict[name + '/' + key ] = metrics[key]

        logging.info(f'Classic disag started') 

        predicted_target_img_adjusted, adj_logs = disag_and_eval_map(res["predicted_target_img"], agg_preds_arr, val_regions, val_map_valid_ids,
            np.unique(val_regions).__len__(), val_valid_ids, val_census, dataset.memory_disag[name])
        for key in adj_logs.keys():
            res_dict[name + '/' + key] = adj_logs[key] #TODO: should we include this log entry in res_dict
            log_dict[name + '/' + key] = adj_logs[key]
        logging.info(f'Classic disag finsihed')

        # "fake" new dissagregation data and reuse the function
        # Do the disagregation on country level # TODO: change to holdout set
        tts = torch.zeros(dataset.memory_disag[name][0].shape, dtype=int)
        tts[torch.where(dataset.memory_disag[name][0])] = 1
        disaggregation_data_coarsest_val = [tts, {1: sum(list(dataset.memory_disag[name][1].values()))}, dataset.memory_disag[name][2] ]
    
        agg_preds_arr_country_adj, this_metrics_cl = disag_wo_map(agg_preds_arr, disaggregation_data_coarsest_val)
        for key,value in this_metrics_cl.items():
            log_dict[name + "/adjusted/country_like/"+key] = value
        
        this_metrics_cl = compute_performance_metrics_arrays(agg_preds_arr_country_adj[dataset.tregid[name]].numpy(), np.asarray(list(val_census.values())))  
        for key,value in this_metrics_cl.items():
            log_dict[name + "/adjusted/country_like/"+key] = value

        res["predicted_target_img_adjusted"] = predicted_target_img_adjusted.cpu()  
        predicted_target_img_adjusted = predicted_target_img_adjusted.cpu() 

        for key in res.keys():
            res_dict[name + '/' + key] = res[key]

    log_dict["batchiter"] = 0 
    log_dict["epoch"] = 0 
    
    wandb.log(log_dict) 
    return res_dict, log_dict

def Eval5Fold_PixAdminTransform(
    datalocations,
    train_dataset_name,
    test_dataset_names,
    params):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    memory_vars,val_valid_ids = {},{}
    for i, (name,rs) in enumerate(datalocations.items()):
        with open(rs['eval_vars'], "rb") as f:
            memory_vars[name] = pickle.load(f)
            val_valid_ids[name] = memory_vars[name][4]

    index_permutation_feat = None
    permutation_random_seed = 42
    if "index_permutation_feat" in params.keys():
        index_permutation_feat = params["index_permutation_feat"]
        permutation_random_seed = params["permutation_random_seed"]
    # make 5 datasets for each fold
    Datasets = []
    for k in range(5): 
        Datasets.append(
            MultiPatchDataset(datalocations, train_dataset_name, params["train_level"], params['memory_mode'], device, 
                params["validation_split"], k, params["weights"], params["custom_sampler_weights"], val_valid_ids, build_pairs=False,  random_seed_folds=params["random_seed_folds"],
                index_permutation_feat=index_permutation_feat, permutation_random_seed=permutation_random_seed, remove_feat_idxs=params["remove_feat_idxs"])
        )

        calculate_mean_std = False
        if calculate_mean_std and k==0:

            def update(existingAggregate, newValue):
                # Welford's online algorithm: Update
                (count, mean, M2) = existingAggregate
                count += 1
                delta = newValue - mean
                mean += delta / count
                delta2 = newValue - mean
                M2 += delta * delta2
                return (count, mean, M2)

            # Retrieve the mean, variance and sample variance from an aggregate
            def finalize(existingAggregate):
                # Welford's online algorithm: Output
                (count, mean, M2) = existingAggregate
                if count < 2:
                    return float("nan")
                else:
                    (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
                    return (mean, variance, sampleVariance)

            for name in test_dataset_names:

                print("****************************************************************")
                print("Calculating Mean and Std for fold", k, "and dataset", name)
                print("****************************************************************")
                summation = 0
                sumsqrt = 0
                n = Datasets[k].BBox_train[name].__len__()
                npix = 0
                initial = torch.zeros((Datasets[k].feature_names[name].__len__()))
                existingAggregate = [0,initial.clone(),initial.clone()]

                for i in (range(n)):
                    X, _, Mask, _, _ = Datasets[k].get_single_training_item(i, name)
                    X_masked = X[:,Mask]
                    this_npix = X_masked.shape[1]
                    npix += this_npix

                    for ii in range(this_npix):
                        existingAggregate = update(existingAggregate,X_masked[:,ii])

                mean_w, var_w, sampleVariance_w = finalize(existingAggregate)
                stddev_w = torch.sqrt(var_w)
                sample_stddev_w = torch.sqrt(sampleVariance_w)

                for j,fname in enumerate(Datasets[k].feature_names[name]):
                    # print("Train Fold", k, "; Dataset", name, "Featurename:", fname ,"; Mean=, Stdv= (", mean_w[j].item(), ",", sample_stddev_w[j].item(), ")")
                    print(mean_w[j].item(), ",", sample_stddev_w[j].item(), "," , fname, ", Train Fold", k, "; Dataset", name, "Featurename:", fname ,"; Mean=, Stdv= (", mean_w[j].item(), ",", sample_stddev_w[j].item(), ")")


    # Fix all random seeds
    # torch.manual_seed(params["random_seed"])
    # random.seed(params["random_seed"])
    # np.random.seed(params["random_seed"])
    # Fix all random seeds
    torch.manual_seed(params["random_seed"])
    random.seed(params["random_seed"])
    np.random.seed(params["random_seed"])
    torch.cuda.manual_seed(params["random_seed"])
    torch.cuda.manual_seed_all(params["random_seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(params["random_seed"])

    # load 5 models
    Mynets = []
    for k in range(5):
        mynet = PixScaleNet(channels_in=Datasets[0].num_feats(),
                    weights_regularizer=params['weights_regularizer'],
                    device=device, loss=params['loss'], kernel_size=params['kernel_size'],
                    dropout=params["dropout"],
                    input_scaling=params["input_scaling"], output_scaling=params["output_scaling"],
                    datanames=train_dataset_name, small_net=params["small_net"], pop_target=params["population_target"]
                    ).train().to(device)


        # Loading from checkpoint
        if params["e5f_metric"] == "final":
            checkpoint = torch.load('checkpoints/Final/Maxstepstate_{}.pth'.format(params["eval_5fold"][k]))
        elif params["e5f_metric"] in ["best_mape_avg","best_r2_avg","best_mae_avg","best_mape_adj_avg","best_r2_adj_avg","best_mae_adj_avg"]: 
            checkpoint = torch.load('checkpoints/{}/AVG/VAL/{}.pth'.format(params["e5f_metric"].split("_avg")[0], params["eval_5fold"][k])) 
        else: 
            checkpoint = torch.load('checkpoints/{}/{}/VAL/{}.pth'.format(params["e5f_metric"], test_dataset_names[0], params["eval_5fold"][k])) 
        
        mynet.load_state_dict(checkpoint['model_state_dict'])
        if "input_scales_bias" in checkpoint.keys():
            mynet.in_scale, mynet.in_bias = checkpoint["input_scales_bias"][0], checkpoint["input_scales_bias"][1]
        if "output_scales_bias" in checkpoint.keys():
            mynet.out_scale, mynet.out_bias = checkpoint["output_scales_bias"][0], checkpoint["output_scales_bias"][1]

        mynet.eval()
        Mynets.append(mynet)

    return eval_generic_model(datalocations, train_dataset_name,  test_dataset_names, params, Mynets, Datasets, memory_vars)






def EvalModel_PixAdminTransform(
    datalocations,
    train_dataset_name,
    test_dataset_names,
    params):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    memory_vars,val_valid_ids = {},{}
    for i, (name,rs) in enumerate(datalocations.items()):
        with open(rs['eval_vars'], "rb") as f:
            memory_vars[name] = pickle.load(f)
            val_valid_ids[name] = memory_vars[name][4]

    # make 5 datasets for each fold 
    dataset = MultiPatchDataset(datalocations, train_dataset_name, params["train_level"], params['memory_mode'], device, 
            validation_fold=None, loss_weights=params["weights"], sampler_weights=params["custom_sampler_weights"], val_valid_ids=val_valid_ids, validation_split=0.0, build_pairs=False,  random_seed_folds=params["random_seed_folds"])

    # calculate_mean_std = False
         
    torch.manual_seed(params["random_seed"])
    random.seed(params["random_seed"])
    np.random.seed(params["random_seed"])
    torch.cuda.manual_seed(params["random_seed"])
    torch.cuda.manual_seed_all(params["random_seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(params["random_seed"])


    mynet = PixScaleNet(channels_in=dataset.num_feats(),
                weights_regularizer=params['weights_regularizer'],
                device=device, loss=params['loss'], kernel_size=params['kernel_size'],
                dropout=params["dropout"],
                input_scaling=params["input_scaling"], output_scaling=params["output_scaling"],
                datanames=train_dataset_name, small_net=params["small_net"], pop_target=params["population_target"]
                ).train().to(device)


    # Loading from checkpoint
    if params["e5f_metric"] == "final":
        checkpoint = torch.load('checkpoints/Final/Maxstepstate_{}.pth'.format(params["eval_5fold"]))
    elif params["e5f_metric"] in ["best_mape_avg","best_r2_avg","best_mae_avg","best_mape_adj_avg","best_r2_adj_avg","best_mae_adj_avg"]: 

        checkpoint = torch.load('checkpoints/{}/AVG/VAL/{}.pth'.format(params["e5f_metric"].split("_avg")[0], params["eval_model"])) 
    else:
        #TODO: This works for one country in the test set. We need to verify if this would work for several countries in test_dataset_names
        checkpoint = torch.load('checkpoints/{}/{}/VAL/{}.pth'.format(params["e5f_metric"], test_dataset_names[0], params["val_valid_ids"])) 
    

    mynet.load_state_dict(checkpoint['model_state_dict'])
    if "input_scales_bias" in checkpoint.keys():
        mynet.in_scale, mynet.in_bias = checkpoint["input_scales_bias"][0], checkpoint["input_scales_bias"][1]
    if "output_scales_bias" in checkpoint.keys():
        mynet.out_scale, mynet.out_bias = checkpoint["output_scales_bias"][0], checkpoint["output_scales_bias"][1]

    mynet.eval()

    log_dict,res_dict = {},{}
    for name in test_dataset_names: 
        logging.info(f'Testing dataset of {name}')
        val_census, val_regions, val_map, _, val_valid_ids, val_map_valid_ids, _, val_valid_data_mask, _, _, _ = memory_vars[name]
        val_features = dataset.features[name]
        
        print((val_regions>0).sum())
        print(val_valid_data_mask.sum())

        res, this_log_dict = eval_my_model(
            mynet, val_features, val_valid_data_mask, val_regions,
            val_map_valid_ids, np.unique(val_regions).__len__(), val_valid_ids, val_census,
            dataset=dataset,
            disaggregation_data=dataset.memory_disag[name],
            dataset_name=name, return_scale=True, silent_mode=params["silent_mode"], full_eval=True
        )


        for key in this_log_dict.keys():
            log_dict[name+'/'+key] = this_log_dict[key]
        for key in res.keys():
            res_dict[name+'/'+key] = res[key]
        torch.cuda.empty_cache()

    log_dict["batchiter"] = 0

    wandb.log(log_dict)
    return res_dict, log_dict


def Eval5Fold_FeatureImportance(
    datalocations,
    train_dataset_name,
    test_dataset_names,
    params):
    
    metric_name = params["e5f_metric"].split("_")[1]
    country_code = test_dataset_names[0] # TODO: implement method for multiple countries (now is just picking the first test country)
    
    with open(datalocations[country_code]['train_vars_c'], "rb") as f:
        _, _, _, tY_c, tregid_c, tMasks_c, tregMasks_c, tBBox_c, feature_names = pickle.load(f)
    
    # Obtain original results witout 
    res_orig, log_dict_orig = Eval5Fold_PixAdminTransform(
            datalocations=datalocations,
            train_dataset_name=train_dataset_name,
            test_dataset_names=test_dataset_names,
            params=params, 
        )
    
    metric_orig = log_dict_orig["{}/{}".format(country_code, metric_name)]
    metric_orig_adj = log_dict_orig["{}/adjusted/{}".format(country_code, metric_name)]
    
    # Obtain results by applying permutation of features
    num_features = len(feature_names)
    num_permutations = params["eval_feat_importance"]
    feat_importance = {"not_adj": {}, "adj": {}}
    # permute each features
    for i in range(num_features):
        feat_name = feature_names[i]
        array_metric = []
        array_metric_adj = []
        print("permute feature : {}".format(feat_name))
        # permute it several times 
        for k in range(num_permutations):
            params["index_permutation_feat"] = i
            params["permutation_random_seed"] = params["random_seed"] + k
            res, log_dict = Eval5Fold_PixAdminTransform(
                datalocations=datalocations,
                train_dataset_name=train_dataset_name,
                test_dataset_names=test_dataset_names,
                params=params, 
            )
            # obtain metric value
            metric = log_dict["{}/{}".format(country_code, metric_name)]
            metric_adj = log_dict["{}/adjusted/{}".format(country_code, metric_name)]
            array_metric.append(metric)
            array_metric_adj.append(metric_adj)
            
            log_dict_orig["log_feat_{}_perm_{}".format(feat_name, k)] = log_dict
        
        array_metric = np.array(array_metric)
        array_metric_adj = np.array(array_metric_adj)
        
        avg_metric = np.mean(array_metric)
        avg_metric_adj = np.mean(array_metric_adj)
        # compute the difference in performance
        feat_importance["not_adj"][feat_name] =  { "importance_score" : metric_orig - avg_metric, 
                                                "avg_metric" : avg_metric,
                                                "array_metrics" : array_metric    
                                                }
        feat_importance["adj"][feat_name] = { "importance_score" : metric_orig_adj - avg_metric_adj, 
                                                "avg_metric" : avg_metric_adj,
                                                "array_metrics" : array_metric_adj    
                                                }
    
    feat_importance["feature_names"] = feature_names
    log_dict_orig["feat_importance"] = feat_importance
    print("{} : not-adj {} adj {}".format(metric_name, metric_orig, metric_orig_adj))
    print(feat_importance)
    
    return res_orig, log_dict_orig