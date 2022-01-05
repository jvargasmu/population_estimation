

import logging
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

from utils import plot_2dmatrix, accumulate_values_by_region, compute_performance_metrics, bbox2, \
     PatchDataset, MultiPatchDataset, NormL1, LogL1, LogoutputL1, LogoutputL2, compute_performance_metrics_arrays
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region

from pix_transform.pix_transform_net import PixScaleNet

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

                if isinstance(prediction, np.ndarray):
                    prediction = prediction[0]
                # agg_preds2[census_id.item()] = prediction.item()
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
    
    if "in_scale" in dir(mynet):
        if dataset_name in mynet.in_scale.keys():
            in_scales = mynet.in_scale[dataset_name][0,:,0,0].detach().cpu().numpy()
            in_biases = mynet.in_bias[dataset_name][0,:,0,0].detach().cpu().numpy() 
            for i,(sc,bi) in enumerate(zip(in_scales[::-1], in_biases[::-1])):
                fname = dataset.feature_names[dataset_name][-i]
                metrics["input_scaling/"+fname] = sc
                metrics["input_bias/"+fname] = bi

        elif "mean_in_scale" in dir(mynet): 
            in_scales = mynet.mean_in_scale[0,:,0,0].detach().cpu().numpy()
            in_biases = mynet.mean_in_bias[0,:,0,0].detach().cpu().numpy() 
            for i,(sc,bi) in enumerate(zip(in_scales[::-1], in_biases[::-1])):
                fname = dataset.feature_names[dataset_name][-i]
                metrics["input_scaling/"+fname] = sc
                metrics["input_bias/"+fname] = bi

    if "out_scale" in dir(mynet): 
        if dataset_name in mynet.out_scale.keys():
            metrics["output_scaling/output_scaling"] = mynet.out_scale[dataset_name].detach().cpu().numpy()
            metrics["output_scaling/output_bias"] = mynet.out_bias[dataset_name].detach().cpu().numpy() 

        elif "mean_out_scale" in dir(mynet): 
            metrics["output_scaling/output_scaling"] = mynet.mean_out_scale.detach().cpu().numpy()
            metrics["output_scaling/output_bias"] = mynet.mean_out_bias.detach().cpu().numpy()

    return res, metrics


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
        best_mae =log_dict["mae"]
        log_dict["best_mae"] = best_mae
        torch.save(saved_dict,
            'checkpoints/best_mae{}{}.pth'.format(dataset_name, wandb.run.name) )

    if log_dict["mape"]<best_mape:
        best_mape =log_dict["mape"]
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

    #TODO: CV with 5 models here 
    #TODO: move 5fold feature
    log_dict = {}
    res_dict = {}
    
    for name in test_dataset_names: 

        agg_preds = []
        val_census = []
        #agg_preds_arr = torch.zeros((dataset.max_tregid_val[name]+1,))
        pop_ests = []
        BBoxes = []
        census_ids = []
        logging.info(f'Validating dataset of {name}')

        for k in range(5):

            dataset = Datasets[k]
            mynet = Mynets[k]


            for idx in tqdm(range(len(dataset.Ys_val[name])), disable=params["silent_mode"]):
                X, Y, Mask, name, census_id, BB = dataset.get_single_validation_item(idx, name, return_BB=True) 
                pop_est, scale = mynet.forward(X, mask=None, name=name, predict_map=True, forward_only=True)
                pop_ests.append(pop_est.detach().cpu().numpy())
                agg_preds.append(pop_est.sum((0,2,3)).detach().cpu().numpy())
                #agg_preds.append(mynet.forward(X, Mask, name=name, forward_only=True).detach().cpu().numpy())
                val_census.append(Y.cpu().numpy())
                census_ids.append(census_id)
                BBoxes.append(BB)

        # calculate this for all folds
        metrics = compute_performance_metrics_arrays(np.asarray(agg_preds), np.asarray(val_census))
        for key in metrics.keys():
            log_dict[name + '/validation/' + key ] = metrics[key]
        torch.cuda.empty_cache()

        #TODO here: Build together the whole amp from the list
        """
        Pseudocode:
        1. Initialize Tensor
        2. Bring to GPU
        3. Iterate through list
            3.1 Insert at each position
        """


    # Evaluation Model
    for name in test_dataset_names: 

        logging.info(f'Testing dataset of {name}')
        val_census, val_regions, val_map, val_valid_ids, val_map_valid_ids, _, val_valid_data_mask, _, _ = memory_vars[name]
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

    return res, log_dir




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
            val_valid_ids[name] = memory_vars[name][3]

    #TODO: make 5 datasets for each fold
    Datasets = []
    for k in range(5): 
        Datasets.append(
            MultiPatchDataset(datalocations, train_dataset_name, params["train_level"], params['memory_mode'], device, 
                params["validation_split"], k, params["weights"], params["custom_sampler_weights"], val_valid_ids, build_pairs=False)
        )

    # Fix all random seeds
    torch.manual_seed(params["random_seed"])
    random.seed(params["random_seed"])
    np.random.seed(params["random_seed"])

    # mynet = PixScaleNet(channels_in=Datasets[0].num_feats(),
    #                 weights_regularizer=params['weights_regularizer'],
    #                 device=device, loss=params['loss'], kernel_size=params['kernel_size'],
    #                 dropout=params["dropout"],
    #                 input_scaling=params["input_scaling"], output_scaling=params["output_scaling"],
    #                 datanames=train_dataset_name
    #                 ).train().to(device)

    # TODO: load 5 models
    Mynets = []
    for k in range(5):
        mynet = PixScaleNet(channels_in=Datasets[0].num_feats(),
                    weights_regularizer=params['weights_regularizer'],
                    device=device, loss=params['loss'], kernel_size=params['kernel_size'],
                    dropout=params["dropout"],
                    input_scaling=params["input_scaling"], output_scaling=params["output_scaling"],
                    datanames=train_dataset_name
                    ).train().to(device)

        checkpoint = torch.load('checkpoints/Final/Maxstepstate_{}.pth'.format(params["eval_5fold"][k]))
        mynet.load_state_dict(checkpoint['model_state_dict'])
        Mynets.append(mynet)

    return eval_generic_model(datalocations, train_dataset_name,  test_dataset_names, params, Mynets, Datasets, memory_vars)
