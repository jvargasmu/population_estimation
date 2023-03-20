import numpy as np
import torch.nn as nn
import torch
from torch.nn.modules.container import Sequential
from tqdm import tqdm
from utils import plot_2dmatrix


class PixScaleNet(nn.Module):

    def __init__(self, channels_in=5, kernel_size=1, weights_regularizer=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu", loss=None, dropout=0.,
        exp_max_clamp=20, pred_var = True, input_scaling=False, output_scaling=False, datanames=None, small_net=False, pop_target=False):
        super(PixScaleNet, self).__init__()

        # Define params
        self.pop_target = pop_target
        self.channels_in = channels_in - 1
        if pop_target:
            self.channels_in = channels_in
        self.device = device
        self.exp_max_clamp = exp_max_clamp
        self.pred_var = pred_var
        self.input_scaling = input_scaling
        self.output_scaling = output_scaling
        self.datanames = datanames
        
        # for some special kinds of loss functions we need an exp(..) transformation
        self.exptransform_outputs = loss in ['LogoutputL1', 'LogoutputL2']

        # bayesian case is not in paper
        self.bayesian = loss in ['gaussNLL', 'laplaceNLL']
        self.out_dim = 2 if self.bayesian else 1

        # define hidden layers
        n1 = 128
        n2 = 128
        n3 = 128
        k1,k2,k3,k4 = kernel_size 
        self.convnet = torch.any(torch.tensor(kernel_size)>1)

        self.params_with_regularizer = []

        # in case of input/output scaling per country (not used in paper)
        if self.input_scaling and (datanames is not None):
            print("using elementwise input scaling")
            self.in_scale = {}
            self.in_bias = {}
            for name in datanames:
                self.in_scale[name] = torch.ones( (1,self.channels_in,1,1), requires_grad=True, device=device)
                self.in_bias[name] = torch.zeros( (1,self.channels_in,1,1), requires_grad=True, device=device) 
                self.params_with_regularizer += [{'params':self.in_scale[name]}] 
                self.params_with_regularizer += [{'params':self.in_bias[name]}]
        
        # in case of input/output scaling per country (not used in paper)  
        if self.output_scaling and (datanames is not None):
            print("using elementwise output scaling")
            self.out_scale = {}
            self.out_bias = {}
            for name in datanames:
                self.out_scale[name] = torch.ones( (1), requires_grad=True, device=device) 
                self.out_bias[name] =  torch.zeros( (1), requires_grad=True, device=device) 
                self.params_with_regularizer += [{'params':self.out_scale[name]}] 
                self.params_with_regularizer += [{'params':self.out_bias[name]}]
        
        # Define the core architecture
        self.occratenet = nn.Sequential(
                        nn.Dropout(p=dropout, inplace=True),                        
                        nn.Conv2d(self.channels_in, n1, (k1,k1), padding=(k1-1)//2),   nn.Dropout(p=dropout, inplace=True),    nn.ReLU(inplace=True),
                        nn.Conv2d(n1, n2, (k2,k2), padding=(k2-1)//2),              nn.Dropout(p=dropout, inplace=True),    nn.ReLU(inplace=True),
                        nn.Conv2d(n2, n3, (k3, k3),padding=(k3-1)//2),              nn.Dropout(p=dropout, inplace=True),    nn.ReLU(inplace=True), 
                        )
        
        self.occrate_layer = nn.Sequential(nn.Conv2d(n3, 1, (k4, k4),padding=(k4-1)//2), nn.Softplus() )
        self.occrate_var_layer = nn.Sequential( nn.Conv2d(n3, 1, (k4, k4),padding=(k4-1)//2), nn.Softplus() if pred_var else nn.Identity(inplace=True) )
 
        self.params_with_regularizer += [{'params':self.occratenet.parameters(),'weight_decay':weights_regularizer}]
        self.params_with_regularizer += [{'params':self.occrate_layer.parameters(),'weight_decay':weights_regularizer}]
        self.params_with_regularizer += [{'params':self.occrate_var_layer.parameters(),'weight_decay':weights_regularizer}]


    def forward(self, inputs, mask=None, name=None, predict_map=False, forward_only=False):

        if len(inputs.shape)==3:
            inputs = inputs.unsqueeze(0)

        if mask is not None and len(mask.shape)==2:
            mask = mask.unsqueeze(0)

        # Check if the image is too large for singe forward pass
        PS = 2500 if forward_only else 1000 
        if torch.tensor(inputs.shape[-2:]).prod()>PS**2:
            return self.forward_batchwise(inputs, mask, name, predict_map=predict_map, forward_only=forward_only)
        
        if (mask is not None) and (not predict_map):
            mask = mask.to(self.device)
            if not self.convnet:
                inputs = inputs[:,:,mask[0]].unsqueeze(3)
                mask = mask[mask].unsqueeze(0).unsqueeze(2) 
            mask = mask.cpu()
        
        # check inputs
        inputs[inputs>1e32] = 0

        # Apply network
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).to(self.device)
        else:
            inputs = inputs.to(self.device)

        # get buildings, which are the first layer
        buildings = inputs[:,0:1,:,:]
        
        if self.pop_target:
            # population target
            data = inputs
        else:
            # occupany target
            data = inputs[:,1:,:,:]

        if self.input_scaling:
            data = self.perform_scale_inputs(data, name)

        # Forward Occupancy_core
        feats = self.occratenet(data)
        
        if self.pop_target:
            pop_est = self.occrate_layer(feats)
            if self.bayesian:
                raise Exception("not implemented")
            else:
                if self.output_scaling:
                    pop_est = self.perform_scale_output(pop_est, name)
                    pop_est[:,:,buildings[0,0]==0] *= 0.
                
                occrate = pop_est / buildings
                occrate[:,:,buildings[0,0]==0] *= 0.
        else:
            occrate = self.occrate_layer(feats)
            if self.bayesian:
                # bayesian case is not in paper

                if self.pred_var:
                    var = self.occrate_var_layer(feats)
                else:
                    var = torch.exp(self.occrate_var_layer(feats)) 

                occrate = torch.cat([occrate, var], 1)

                # Not in paper
                if self.output_scaling:
                    occrate = self.perform_scale_output(occrate, name)
                    occrate[:,:,buildings[0,0]==0] *= 0.
                
                # Multiply the number of Buildings with the occupancy rate
                pop_est = torch.mul(buildings, occrate[:,0])

                # Variance Propagation for bayesian approach
                pop_est = torch.cat([pop_est,  torch.mul(torch.square(buildings), occrate[:,1])], 1)
            else:
                if self.output_scaling:
                    occrate = self.perform_scale_output(occrate, name)

                # Multiply the number of Buildings with the occupancy rate
                pop_est = torch.mul(buildings, occrate)
        data = data.cpu()

        # backtransform if necessary before(!) summation
        if self.exptransform_outputs:
            pop_est = pop_est.exp() 
        
        # Check if masking should be applied
        if mask is not None: 
            if self.bayesian:
                return pop_est[0,:,mask[0]].sum(1).cpu()
            else:
                return pop_est[0,mask].sum().cpu()
        else:
            # check if the output should be the map or the sum
            if not predict_map:
                return pop_est.sum((0,2,3)).cpu()
            else:
                return pop_est.cpu(), occrate.cpu()

    # Not in paper
    def perform_scale_inputs(self, data, name):
        if name not in list(self.in_scale.keys()):
            self.calculate_mean_input_scale()
            return (data - self.mean_in_bias) / self.mean_in_scale #+ self.mean_in_bias
        else:
            return (data - self.in_bias[name]) /self.in_scale[name]


    # Not in paper
    def calculate_mean_input_scale(self):
        self.mean_in_scale = 0
        self.mean_in_bias = 0
        for name in list(self.in_scale.keys()):
            self.mean_in_scale += self.in_scale[name]
            self.mean_in_bias += self.in_bias[name]
        self.mean_in_scale = self.mean_in_scale/self.in_scale.keys().__len__()
        self.mean_in_bias = self.mean_in_bias/self.in_scale.keys().__len__()


    # Not in paper
    def perform_scale_output(self, preds, name):
        """
        Inputs:
            - preds : tensor of shape (1,d,h,w). Where d is 1 for the non bayesian case and 2 (pred & var) for the bayesian case.
            - name: the name of the country the patch is located.
        Output:
            - Scaled and clamped predictions
        """
        if self.bayesian:
            if name not in list(self.out_scale.keys()):
                self.calculate_mean_output_scale()  
                preds_0 = preds[:,0:1]*self.mean_out_scale #+ self.mean_out_bias
                preds_1 = preds[:,1:2]*torch.square(self.mean_out_scale)
                preds = torch.cat([preds_0,preds_1], 1)  
            else:
                preds_0 = preds[:,0:1]*self.out_scale[name] #+ self.out_bias[name]
                preds_1 = preds[:,1:2]*torch.square(self.out_scale[name])
                preds = torch.cat([preds_0,preds_1], 1) 
        else: 
            if name not in list(self.out_scale.keys()):
                self.calculate_mean_output_scale()
                preds = preds*self.mean_out_scale #+ self.mean_out_bias
            else:
                preds = preds*self.out_scale[name] #+ self.out_bias[name]
                        
        # Ensure that there are no negative occ-rates and variances
        return preds.clamp(min=0)

    # Not in paper
    def normalize_out_scales(self):
        with torch.no_grad():
            average_scale = torch.sum(torch.cat(list(self.out_scale.values()))) / list(self.out_scale.keys()).__len__()
            
            for key in list(self.out_scale.keys()):
                self.out_scale[key] /= average_scale
            
    # Not in paper
    def calculate_mean_output_scale(self):
        self.mean_out_scale = 0
        self.mean_out_bias = 0
        for name in list(self.out_scale.keys()):
            self.mean_out_scale += self.out_scale[name]
            self.mean_out_bias += self.out_bias[name]
        self.mean_out_scale = self.mean_out_scale/self.out_scale.keys().__len__()
        self.mean_out_bias = self.mean_out_bias/self.out_scale.keys().__len__()


    def forward_batchwise(self, inputs, mask=None, name=None, predict_map=False, return_scale=False, forward_only=False): 
        # Memory optimized function to sparsely forward large administrative regions. 
        #choose a responsible patch that does not exceed the GPU memory
        PS = 1800 if forward_only else 900
        extra_low_memory = False
        if extra_low_memory:
            PS = 32 if self.convnet else PS
        else:
            PS = 64 if self.convnet else PS
        oh, ow = inputs.shape[-2:]
        if predict_map:
            outvar = torch.zeros((1,self.out_dim,oh, ow), dtype=torch.float32, device='cpu')
            scale = torch.zeros((1,self.out_dim,oh, ow), dtype=torch.float32, device='cpu')
        else:
            outvar = 0

        # Cuts the image into small patches and forwards them piece by piece, and throws away empty patches
        sums = []
        for hi in range(0,oh,PS):
            for oi in range(0,ow,PS):
                if (not predict_map) and (not self.convnet):
                    if mask is not None and mask[:,hi:hi+PS,oi:oi+PS].sum()>0:
                        outvar += self( inputs[:,:,hi:hi+PS,oi:oi+PS][:,:,mask[0,hi:hi+PS,oi:oi+PS]].unsqueeze(3), name=name, forward_only=forward_only)
                elif (not predict_map) and self.convnet:
                    this_mask = mask[:,hi:hi+PS,oi:oi+PS]
                    if this_mask.sum()>0:
                        out = self( inputs[:,:,hi:hi+PS,oi:oi+PS], mask=this_mask, predict_map=True, name=name)
                        outvar += out.sum().cpu()
                else:
                    # for mappredictions, we do not sum up.
                    outvar[:,:,hi:hi+PS,oi:oi+PS], scale[:,:,hi:hi+PS,oi:oi+PS] = self( inputs[:,:,hi:hi+PS,oi:oi+PS], name=name, predict_map=True, forward_only=forward_only)

        if not predict_map:
            return outvar
        else:
            return outvar, scale
        

    def forward_one_or_more(self, sample, mask=None):
        # Forwards all the administrative regions in the batch. 
        # In practice we only tested batchsize=1 because of memory requirements

        summings = []
        valid_samples  = 0 
        for i, inp in enumerate(sample):
            if inp[2].sum()>0:

                summings.append( self(inp[0], inp[2], inp[3][0]).cpu())
                valid_samples += 1

        if valid_samples==0:
            return None    
        return summings
