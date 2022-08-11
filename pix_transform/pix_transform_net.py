import numpy as np
import torch.nn as nn
import torch
from torch.nn.modules.container import Sequential
from tqdm import tqdm
from utils import plot_2dmatrix

class PixTransformNet(nn.Module):

    def __init__(self, channels_in=5, kernel_size = 1, weights_regularizer = None, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(PixTransformNet, self).__init__()

        self.channels_in = channels_in
        self.device = device

        n1 = 128
        n2 = 128
        n3 = 128
        kernel_size = 1

        self.net = nn.Sequential(nn.Conv2d(channels_in,n1,(1,1),padding=0),
                                      nn.ReLU(inplace=True),nn.Conv2d(n1, n2,(kernel_size,kernel_size),padding=(kernel_size-1)//2),
                                      nn.ReLU(inplace=True),nn.Conv2d(n2, n3, (kernel_size,kernel_size),padding=(kernel_size-1)//2),
                                      nn.ReLU(inplace=True),nn.Conv2d(n3, 1, (1, 1),padding=0),
                                      nn.ReLU(inplace=True))

        if weights_regularizer is None:
            regularizer = 0.001
        else:
            # reg_spatial = weights_regularizer[0]
            regularizer = weights_regularizer
            # reg_head = weights_regularizer[2]
        
        self.params_with_regularizer = []
        # self.params_with_regularizer += [{'params':self.spatial_net.parameters(),'weight_decay':reg_spatial}]
        self.params_with_regularizer += [{'params':self.net.parameters(),'weight_decay':regularizer}]
        # self.params_with_regularizer += [{'params':self.head_net.parameters(),'weight_decay':reg_head}]


    def forward(self, inputs, mask=None, predict_map=False):

        # Check if the image is too large for singe forward pass
        if torch.tensor(inputs.shape[2:4]).prod()>400**2:
            return self.forward_batchwise(inputs, mask)

        # Apply network
        inputs = self.net(inputs.to(self.device))
        
        # Check if masking should be applied
        if mask is not None:
            mask = mask.to(self.device)
            return inputs[:,mask].sum()
        else:

            # check if the output should be the map or the sum
            if not predict_map:
                return inputs.sum()
            else:
                return inputs
                

    def forward_batchwise(self, inputs,mask=None, predict_map=False): 

        #choose a responsible patch that does not exceed the GPU memory
        PS = 150
        oh, ow = inputs.shape[-2:]
        if not predict_map:
            outvar = 0
        else:
            outvar = torch.zeros((1,1,oh, ow), dtype=inputs.dtype, device='cpu')

        sums = []
        for hi in range(0,oh,PS):
            for oi in range(0,ow,PS):
                if not predict_map:
                    if mask[:,hi:hi+PS,oi:oi+PS].sum()>0:
                        outvar += self( inputs[:,:,hi:hi+PS,oi:oi+PS][:,:,mask[0,hi:hi+PS,oi:oi+PS]].unsqueeze(3))
                else:
                    outvar[:,:,hi:hi+PS,oi:oi+PS] = self( inputs[:,:,hi:hi+PS,oi:oi+PS], predict_map=True).cpu()
                    
        if not predict_map:    
            return outvar
        else:
            return outvar.squeeze()

    def forward_one_or_more(self, sample, mask=None):

        total_sum = 0
        valid_samples  = 0
        for i, inp in enumerate(sample):
            if inp[2].sum()>0:

                total_sum += self(inp[0], inp[2])
                valid_samples += 1

        if valid_samples==0:
            return None    
        return total_sum


class PixScaleNet(nn.Module):

    def __init__(self, channels_in=5, kernel_size=1, weights_regularizer=0.001, hidden_neurons=128,
        device="cuda" if torch.cuda.is_available() else "cpu", loss=None, dropout=0.,
        exp_max_clamp=20, pred_var = True, input_scaling=False, output_scaling=False, datanames=None, small_net=False, pop_target=False):
        super(PixScaleNet, self).__init__()

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

        self.exptransform_outputs = loss in ['LogoutputL1', 'LogoutputL2']
        self.bayesian = loss in ['gaussNLL', 'laplaceNLL']
        self.out_dim = 2 if self.bayesian else 1

        # k1,k2,k3,k4 = kernel_size
        Netlist = []
        for i,k in enumerate(kernel_size[:-1]):
            if i==0:
                incells = self.channels_in
            else:
                incells = hidden_neurons
            convblock = [nn.Dropout(p=dropout, inplace=False), nn.Conv2d(incells, hidden_neurons, (k,k), padding=(k-1)//2, padding_mode="reflect"), nn.ReLU(inplace=False)]
            Netlist.extend(convblock)
        self.occratenet =nn.Sequential(*Netlist)

        self.convnet = torch.any(torch.tensor(kernel_size)>1)

        self.params_with_regularizer = []

        if self.input_scaling and (datanames is not None):
            print("using elementwise input scaling")
            self.in_scale = {}
            self.in_bias = {}
            for name in datanames:
                self.in_scale[name] = torch.ones( (1,self.channels_in,1,1), requires_grad=True, device=device)
                self.in_bias[name] = torch.zeros( (1,self.channels_in,1,1), requires_grad=True, device=device) 
                self.params_with_regularizer += [{'params':self.in_scale[name]}] 
                self.params_with_regularizer += [{'params':self.in_bias[name]}]
            
        if self.output_scaling and (datanames is not None):
            print("using elementwise output scaling")
            self.out_scale = {}
            self.out_bias = {}
            for name in datanames:
                self.out_scale[name] = torch.ones( (1), requires_grad=True, device=device) 
                self.out_bias[name] =  torch.zeros( (1), requires_grad=True, device=device) 
                self.params_with_regularizer += [{'params':self.out_scale[name]}] 
                self.params_with_regularizer += [{'params':self.out_bias[name]}]
             

        self.occrate_layer = nn.Sequential(nn.Conv2d(hidden_neurons, 1, (kernel_size[-1], kernel_size[-1]),padding=(kernel_size[-1]-1)//2, padding_mode="reflect"), nn.Softplus() )
        self.occrate_var_layer = nn.Sequential( nn.Conv2d(hidden_neurons, 1, (kernel_size[-1], kernel_size[-1]),padding=(kernel_size[-1]-1)//2, padding_mode="reflect"), nn.Softplus() if pred_var else nn.Identity(inplace=False) )
 
        self.params_with_regularizer += [{'params':self.occratenet.parameters(),'weight_decay':weights_regularizer}]
        self.params_with_regularizer += [{'params':self.occrate_layer.parameters(),'weight_decay':weights_regularizer}]
        self.params_with_regularizer += [{'params':self.occrate_var_layer.parameters(),'weight_decay':weights_regularizer}]


    def forward(self, inputs, mask=None, name=None, predict_map=False, forward_only=False):

        if mask is not None:
            if mask.sum()==0 and not self.convnet:
                return self.return_zero(inputs, predict_map)
            
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

        buildings = inputs[:,0:1,:,:]
        
        if self.pop_target:
            data = inputs
        else:
            data = inputs[:,1:,:,:]

        if self.input_scaling:
            data = self.perform_scale_inputs(data, name)

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
                if self.pred_var:
                    var = self.occrate_var_layer(feats)
                else:
                    var = torch.exp(self.occrate_var_layer(feats)) 

                occrate = torch.cat([occrate, var], 1)
                if self.output_scaling:
                    occrate = self.perform_scale_output(occrate, name)
                    occrate[:,:,buildings[0,0]==0] *= 0.
                    
                pop_est = torch.mul(buildings, occrate[:,0])

                # Variance Propagation
                pop_est = torch.cat([pop_est,  torch.mul(torch.square(buildings), occrate[:,1])], 1)
            else:
                if self.output_scaling:
                    occrate = self.perform_scale_output(occrate, name)
                    #occrate[:,:,buildings[0,0]==0] *= 0.
                pop_est = torch.mul(buildings, occrate)
        data = data.cpu()

        # backtransform if necessary before(!) summation
        if self.exptransform_outputs:
            #TODO: change this part when using a new loss function
            pop_est = pop_est.exp() 
        
        # Check if masking should be applied
        if mask is not None: 
            if self.bayesian:
                return pop_est[0,:,mask[0]].sum(1).cpu()
            else:
                return pop_est[0,mask].sum().cpu()
            #return pop_est.sum((0,2,3)).cpu()
        else:
            # check if the output should be the map or the sum
            if not predict_map:
                return pop_est.sum((0,2,3)).cpu()
            else:
                return pop_est.cpu(), occrate.cpu()


    def return_zero(self, inputs, predict_map):
        # Makes a dummy output for the case that the region contains no buildings
        zero_map = inputs[0]*torch.zeros_like(inputs[0])
        if not predict_map:
            return zero_map.sum().cpu()
        else:
            return zero_map.cpu(), zero_map.cpu()

    def perform_scale_inputs(self, data, name):
        if name not in list(self.in_scale.keys()):
            self.calculate_mean_input_scale()
            return (data - self.mean_in_bias) / self.mean_in_scale #+ self.mean_in_bias
        else:
            return (data - self.in_bias[name]) /self.in_scale[name]


    def calculate_mean_input_scale(self):
        self.mean_in_scale = 0
        self.mean_in_bias = 0
        for name in list(self.in_scale.keys()):
            self.mean_in_scale += self.in_scale[name]
            self.mean_in_bias += self.in_bias[name]
        self.mean_in_scale = self.mean_in_scale/self.in_scale.keys().__len__()
        self.mean_in_bias = self.mean_in_bias/self.in_scale.keys().__len__()


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

    def normalize_out_scales(self):
        with torch.no_grad():
            average_scale = torch.sum(torch.cat(list(self.out_scale.values()))) / list(self.out_scale.keys()).__len__()
            
            for key in list(self.out_scale.keys()):
                self.out_scale[key] /= average_scale
            
    def calculate_mean_output_scale(self):
        self.mean_out_scale = 0
        self.mean_out_bias = 0
        for name in list(self.out_scale.keys()):
            self.mean_out_scale += self.out_scale[name]
            self.mean_out_bias += self.out_bias[name]
        self.mean_out_scale = self.mean_out_scale/self.out_scale.keys().__len__()
        self.mean_out_bias = self.mean_out_bias/self.out_scale.keys().__len__()


    def forward_batchwise(self, inputs, mask=None, name=None, predict_map=False, return_scale=False, forward_only=False): 

        #choose a responsible patch that does not exceed the GPU memory
        PS = 1800 if forward_only else 900
        PS = 64 if self.convnet else PS
        oh, ow = inputs.shape[-2:]
        if predict_map:
            outvar = torch.zeros((1,self.out_dim,oh, ow), dtype=torch.float32, device='cpu')
            scale = torch.zeros((1,self.out_dim,oh, ow), dtype=torch.float32, device='cpu')
        else:
            outvar = 0

        sums = []
        for hi in range(0,oh,PS):
            for oi in range(0,ow,PS):
                if (not predict_map) and (not self.convnet):
                    if mask is not None and mask[:,hi:hi+PS,oi:oi+PS].sum()>0:
                        outvar += self( inputs[:,:,hi:hi+PS,oi:oi+PS][:,:,mask[0,hi:hi+PS,oi:oi+PS]].unsqueeze(3), name=name, forward_only=forward_only)
                elif (not predict_map) and self.convnet:
                    this_mask = mask[:,hi:hi+PS,oi:oi+PS]
                    if this_mask.sum()>0:
                        # out = self( inputs[:,:,hi:hi+PS,oi:oi+PS], mask=this_mask, predict_map=True, name=name)[0].cpu()
                        out = self( inputs[:,:,hi:hi+PS,oi:oi+PS], mask=this_mask, predict_map=True, name=name)
                        outvar += out.sum().cpu()
                else:
                    outvar[:,:,hi:hi+PS,oi:oi+PS], scale[:,:,hi:hi+PS,oi:oi+PS] = self( inputs[:,:,hi:hi+PS,oi:oi+PS], name=name, predict_map=True, forward_only=forward_only)

        if not predict_map:
            return outvar
        else:
            return outvar, scale
        

    def forward_one_or_more(self, sample, mask=None):

        summings = []
        valid_samples  = 0 
        for i, inp in enumerate(sample):
            if inp[2].sum()>0:

                summings.append( self(inp[0], inp[2], inp[3][0]).cpu())
                valid_samples += 1

        if valid_samples==0:
            return None    
        return summings
