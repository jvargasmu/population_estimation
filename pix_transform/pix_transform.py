import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import sys
from utils import plot_2dmatrix, accumulate_values_by_region, compute_performance_metrics
from cy_utils import compute_map_with_new_labels, compute_accumulated_values_by_region, compute_disagg_weights, \
    set_value_for_each_region
from pix_transform_utils.utils import upsample

from pix_transform.pix_transform_net import PixTransformNet

DEFAULT_PARAMS = {'greyscale': False,
                  'channels': -1,
                  'bicubic_input': False,
                  'spatial_features_input': True,
                  'weights_regularizer': [0.0001, 0.001, 0.001],  # spatial color head
                  'loss': 'l1',

                  'optim': 'adam',
                  'lr': 0.0001,

                  'batch_size': 32,
                  'patch_size': 8,
                  'iteration': 1024 * 32,

                  'logstep': 512,
                  }

if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm


def PixTransform(source_img, guide_img, valid_mask=None, params=DEFAULT_PARAMS, validation_data=None, orig_guide_res=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(guide_img.shape) < 3:
        guide_img = np.expand_dims(guide_img, 0)

    if params["channels"] > 0:
        guide_img = guide_img[0:params["channels"], :, :]

    if params['greyscale']:
        guide_img = np.mean(guide_img, axis=0, keepdims=True)

    if valid_mask is None:
        valid_mask = np.ones_like(guide_img)

    if validation_data is not None:
        validation_census, target_img, validation_regions, validation_ids, valid_validation_ids = validation_data
        num_validation_ids = np.unique(validation_regions).__len__()
        target_img[~valid_mask] = target_img[valid_mask].mean()
    source_img[~valid_mask] = source_img[valid_mask].mean()

    n_channels, hr_height, hr_width = guide_img.shape

    source_img = source_img.squeeze()
    lr_height, lr_width = source_img.shape

    assert (hr_height == lr_height)
    assert (hr_width == lr_width)
    # assert (hr_height % lr_height == 0)

    PS = params["patch_size"]
    Mh = hr_height // PS
    Mw = hr_width // PS

    # normalize guide and source\
    masked_guide_img = np.ma.array(guide_img, mask=~np.tile(valid_mask[None], (n_channels,1,1)))
    guide_img_mean = np.ma.getdata( masked_guide_img.mean((1,2)) )
    guide_img_std = np.ma.getdata( masked_guide_img.std((1,2)) )
    guide_img = ((guide_img.transpose((1,2,0)) - guide_img_mean ) / guide_img_std).transpose((2,0,1))

    maskedsource_img = np.ma.array(source_img, mask=~valid_mask)
    source_img_mean = np.ma.getdata( maskedsource_img.mean() )
    source_img_std = np.ma.getdata( maskedsource_img.std() )
    source_img = (source_img - source_img_mean) / source_img_std
    if target_img is not None:
        target_img = (target_img - source_img_mean) / source_img_std

    if params['spatial_features_input']:
        x = np.linspace(-0.5, 0.5, hr_height)
        y = np.linspace(-0.5, 0.5, hr_width)
        x_grid, y_grid = np.meshgrid(x, y, indexing='ij')

        x_grid = np.expand_dims(x_grid, axis=0)
        y_grid = np.expand_dims(y_grid, axis=0)

        guide_img = np.concatenate([guide_img, x_grid, y_grid], axis=0)
        n_channels += 2

    #### prepare_patches #########################################################################
    # guide_patches is M^2 x C x D x D
    # source_pixels is M^2 x 1

    # Move all important variable to pytorch
    guide_img = torch.from_numpy(guide_img).float()
    source_img = torch.from_numpy(source_img).float()
    valid_mask = torch.from_numpy(valid_mask).type(torch.BoolTensor)
    if target_img is not None:
        target_img = torch.from_numpy(target_img).float()#.to(device)
        validation_regions = torch.from_numpy(validation_regions.astype(np.int32))#.to(device)

    source_img_mean = torch.tensor(source_img_mean)
    source_img_std = torch.tensor(source_img_std)
    
    # Iterate throuh the image an cut out examples
    guide_patches = guide_img.unfold(1,PS,PS).unfold(2,PS,PS).contiguous().view(n_channels, -1, PS, PS ).permute(1,0,2,3).to(device)
    source_patches = source_img.unfold(0,PS,PS).unfold(1,PS,PS).contiguous().view(1, -1, PS, PS ).permute(1,0,2,3).to(device)
    mask_patches = valid_mask.unfold(0,PS,PS).unfold(1,PS,PS).type(torch.BoolTensor).contiguous().view(1, -1, PS, PS ).permute(1,0,2,3).to(device)

    train_data = torch.utils.data.TensorDataset(guide_patches, source_patches,mask_patches)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    ###############################################################################################

    #### setup network ############################################################################
    mynet = PixTransformNet(channels_in=guide_img.shape[0],
                            weights_regularizer=params['weights_regularizer']).train().to(device)
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
            for (x, y, mask) in (train_loader):
                if mask.sum()<1:
                    optimizer.zero_grad()
                    continue
                
                optimizer.zero_grad()

                y_pred = mynet(x)

                loss = myloss(y_pred[mask], y[mask])

                loss.backward()
                optimizer.step()

            if epoch % params['logstep'] == 0:
                with torch.no_grad():
                    mynet.eval()
                    # batchwise passing for whole image
                    predicted_target_img = mynet.forward_batchwise(guide_img.unsqueeze(0)).squeeze()
                    # replace masked values with the mean value, this way the artefacts when upsampling are mitigated
                    predicted_target_img[~valid_mask] = 0#predicted_target_img[valid_mask].mean()


                    if target_img is not None:

                        # revert normalization
                        abs_predicted_target_img =  (source_img_std * predicted_target_img) + source_img_mean 
                        abs_source_img = (source_img_std * source_img) + source_img_mean 
                        abs_target_img = (source_img_std * target_img) + source_img_mean 

                        if params["predict_log_values"]:
                            abs_predicted_target_img = torch.exp(abs_predicted_target_img)
                            abs_source_img = torch.exp(abs_source_img)
                            abs_target_img = torch.exp(abs_target_img)

                        # mse_predicted_source_img = F.mse_loss((source_img_std * predicted_target_img)[valid_mask], (source_img_std * source_img)[valid_mask])
                        mse_predicted_source_img = F.mse_loss(abs_predicted_target_img[valid_mask], abs_source_img[valid_mask]).cpu().numpy()
                        # mse_predicted_target_img = F.mse_loss((source_img_std * predicted_target_img)[valid_mask], (source_img_std * target_img)[valid_mask])
                        # mse_predicted_target_img = F.mse_loss(abs_predicted_target_img[valid_mask].cpu(), abs_target_img[valid_mask]).cpu().numpy()
                    
                        # convert resolution back to feature resolution
                        if params["feature_downsampling"]!=1:
                            full_res_predicted_target_image = upsample(abs_predicted_target_img, params["feature_downsampling"], orig_guide_res=orig_guide_res)
                            full_upsamp_source_image = upsample(abs_source_img, params["feature_downsampling"], orig_guide_res=orig_guide_res)
                            # full_upsamp_target_image = upsample(abs_target_img, params["feature_downsampling"], orig_guide_res=orig_guide_res)
                            # plot_2dmatrix(torch.log(upsample(abs_source_img, params["feature_downsampling"], orig_guide_res=orig_guide_res)))
                        else:
                            full_res_predicted_target_image = upsample(abs_predicted_target_img, params["feature_downsampling"], orig_guide_res=orig_guide_res)
                            full_upsamp_source_image = upsample(abs_source_img, params["feature_downsampling"], orig_guide_res=orig_guide_res)
                        
                        #TODO: interpolate the missing boarder values that occure due the downsampling process

                        agg_preds_arr = compute_accumulated_values_by_region(
                            validation_regions.numpy().astype(np.uint32),
                            full_res_predicted_target_image.cpu().numpy().astype(np.float32),
                            valid_validation_ids,
                            num_validation_ids
                        )
                        agg_preds = {id: agg_preds_arr[id] for id in validation_ids}
                        
                        # source_img_back = compute_accumulated_values_by_region(
                        #     validation_regions.numpy().astype(np.uint32),
                        #     full_upsamp_source_image.cpu().numpy().astype(np.float32),
                        #     valid_validation_ids,
                        #     num_validation_ids
                        # )
                        # agg_source_back = {id: source_img_back[id] for id in validation_ids}

                        # r2_source, mae_source, mse_source = compute_performance_metrics(agg_preds, agg_source_back) # Try to overfit this as a test!
                        # r2_ts, mae_ts, mse_ts = compute_performance_metrics(validation_census, agg_source_back)

                        
                        # agg_preds = accumulate_values_by_region(full_res_predicted_target_image, list(validation_census.keys()), validation_regions)
                        r2, mae, mse = compute_performance_metrics(agg_preds, validation_census)
                        # source_image_consistency = myloss(
                        #     source_img_std * F.avg_pool2d(predicted_target_img.unsqueeze(0), D),
                        #     source_img_std * source_img.unsqueeze(0))
                    if target_img is not None:
                        tnr.set_postfix(R2=r2, zMAEc=mae, zMSEs=mse_predicted_source_img) #MSEs=mse_predicted_source_img, MSEt=mse_predicted_target_img,
                                        
                    else:
                        tnr.set_postfix(MSEs=mse_predicted_source_img)
                    mynet.train()

    # compute final prediction, un-normalize, and back to numpy
    mynet.eval()
    predicted_target_img = mynet(guide_img.unsqueeze(0)).squeeze()
    predicted_target_img = source_img_mean + source_img_std * predicted_target_img
    predicted_target_img = predicted_target_img.cpu().detach().squeeze().numpy()

    return predicted_target_img
