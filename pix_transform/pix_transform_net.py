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

    def __init__(self, channels_in=5, kernel_size=1, weights_regularizer=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu", loss=None, dropout=0.,
        exp_max_clamp=20, pred_var = True, input_scaling=False, output_scaling=False, datanames=None):
        super(PixScaleNet, self).__init__()

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

        n1 = 128
        n2 = 128
        n3 = 128
        k1,k2,k3,k4 = kernel_size 
        self.convnet = torch.any(torch.tensor(kernel_size)>1)

        self.params_with_regularizer = []

        if self.input_scaling and (datanames is not None):
            print("using elementwise input scaling")
            self.in_scale = {}
            self.in_bias = {}
            for name in datanames:
                self.in_scale[name] = torch.ones( (1,channels_in-1,1,1), requires_grad=True, device=device)
                self.in_bias[name] = torch.zeros( (1,channels_in-1,1,1), requires_grad=True, device=device)
                self.params_with_regularizer += [{'params':self.in_scale[name],'weight_decay':weights_regularizer}]
                self.params_with_regularizer += [{'params':self.in_bias[name],'weight_decay':weights_regularizer}]
            
        if self.output_scaling and (datanames is not None):
            print("using elementwise output scaling")
            self.out_scale = {}
            self.out_bias = {}
            for name in datanames:
                self.out_scale[name] = torch.ones( (1), requires_grad=True, device=device)
                # self.out_scale[name].data.fill_(1.)
                self.out_bias[name] =  torch.zeros( (1), requires_grad=True, device=device)
                # self.out_bias[name].data.fill_(0.)
                self.params_with_regularizer += [{'params':self.out_scale[name],'weight_decay':weights_regularizer}]
                self.params_with_regularizer += [{'params':self.out_bias[name],'weight_decay':weights_regularizer}]
            
        if dropout>0.0:
            self.scalenet = nn.Sequential(
                            nn.Dropout(p=dropout, inplace=True),                        nn.Conv2d(channels_in-1, n1, (k1,k1), padding=(k1-1)//2),
                            nn.Dropout(p=dropout, inplace=True), nn.ReLU(inplace=True), nn.Conv2d(n1, n2, (k2,k2), padding=(k2-1)//2),
                            nn.Dropout(p=dropout, inplace=True), nn.ReLU(inplace=True), nn.Conv2d(n2, n3, (k3, k3),padding=(k3-1)//2),
                            nn.Dropout(p=dropout, inplace=True), nn.ReLU(inplace=True),# nn.Conv2d(n3, self.out_dim, (k4, k4),padding=(k4-1)//2),
                                                                #  nn.ReLU(inplace=True)
                            )
        else:
            self.scalenet = nn.Sequential(        nn.Conv2d(channels_in-1, n1, (k1,k1), padding=(k1-1)//2),
                            nn.ReLU(inplace=True),nn.Conv2d(n1, n2, (k2,k2), padding=(k2-1)//2),
                            nn.ReLU(inplace=True),nn.Conv2d(n2, n3, (k3,k3), padding=(k3-1)//2),
                        #   nn.ReLU(inplace=True),nn.Conv2d(n3, n3, (k3, k3),padding=(k3-1)//2),
                        #   nn.ReLU(inplace=True),nn.Conv2d(n3, n3, (k3, k3),padding=(k3-1)//2),
                            nn.ReLU(inplace=True),#nn.Conv2d(n3, self.out_dim, (k4, k4),padding=(k4-1)//2),
                            # nn.ReLU(inplace=True)
                            )

        self.scale_layer = nn.Sequential(nn.Conv2d(n3, 1, (k4, k4),padding=(k4-1)//2), nn.Softplus() )
        self.var_layer = nn.Sequential( nn.Conv2d(n3, 1, (k4, k4),padding=(k4-1)//2), nn.Softplus() if pred_var else nn.Identity(inplace=True) )
 
        self.params_with_regularizer += [{'params':self.scalenet.parameters(),'weight_decay':weights_regularizer}]
        self.params_with_regularizer += [{'params':self.scale_layer.parameters(),'weight_decay':weights_regularizer}]
        self.params_with_regularizer += [{'params':self.var_layer.parameters(),'weight_decay':weights_regularizer}]


    def forward(self, inputs, mask=None, name=None, predict_map=False, forward_only=False):

        # Check if the image is too large for singe forward pass
        PS = 1500 if forward_only else 1500
        if torch.tensor(inputs.shape[2:4]).prod()>PS**2:
            return self.forward_batchwise(inputs, mask, name)

        if mask is not None and len(mask.shape)==2:
            mask = mask.unsqueeze(0)
        
        if len(inputs.shape)==3:
            inputs = inputs.unsqueeze(0)

        if (mask is not None) and (not predict_map):
            mask = mask.to(self.device)
            inputs = inputs[:,:,mask[0]].unsqueeze(3)

        # Apply network
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).to(self.device)
        else:
            inputs = inputs.to(self.device)

        buildings = inputs[:,0:1,:,:]
        data = inputs[:,1:,:,:]

        if self.input_scaling:
            data = self.perform_scale_inputs(data, name)

        data = self.scalenet(data)
        scale = self.scale_layer(data)
        pop_est = torch.mul(buildings, scale)
        if self.bayesian:
            if self.pred_var:
                var = self.var_layer(data)
            else:
                var = torch.exp(self.var_layer(data))
            # Variance Propagation
            scale = torch.cat([scale, var], 1)
            pop_est = torch.cat([pop_est,  torch.mul(torch.square(buildings), var)], 1)

        if self.output_scaling:
            pop_est = self.perform_scale_output(pop_est, name)
        
        # backtransform if necessary before(!) summation
        if self.exptransform_outputs:
            #TODO: change this part when using a new loss function
            pop_est = pop_est.exp() 
        
        # Check if masking should be applied
        if mask is not None:
            mask = mask.to(self.device)
            return pop_est.sum((0,2,3)).cpu()
        else:
            # check if the output should be the map or the sum
            if not predict_map:
                return pop_est.sum((0,2,3)).cpu()
            else:
                return pop_est.cpu(), scale.cpu()


    def perform_scale_inputs(self, data, name):
        if name not in self.datanames:
            self.calculate_mean_input_scale()
            return data*self.mean_in_scale + self.mean_in_bias
        else:
            return data*self.in_scale[name] + self.in_bias[name]

    def calculate_mean_input_scale(self):
        self.mean_in_scale = 0
        self.mean_in_bias = 0
        for name in self.datanames:
            self.mean_in_scale += self.in_scale[name]
            self.mean_in_bias += self.in_bias[name]
        self.mean_in_scale = self.mean_in_scale/self.datanames.__len__()
        self.mean_in_bias = self.mean_in_bias/self.datanames.__len__()


    def perform_scale_output(self, preds, name):
        if name not in self.datanames:
            self.calculate_mean_input_scale()
            return preds*self.mean_out_scale + self.mean_out_bias
        else:
            return preds*self.out_scale[name] + self.out_bias[name]

    def calculate_mean_output_scale(self):
        self.mean_out_scale = 0
        self.mean_out_bias = 0
        for name in self.datanames:
            self.mean_out_scale += self.out_scale[name]
            self.mean_out_bias += self.out_bias[name]
        self.mean_out_scale = self.mean_out_scale/self.datanames.__len__()
        self.mean_out_bias = self.mean_out_bias/self.datanames.__len__()


    def forward_batchwise(self, inputs, mask=None, name=None, predict_map=False, return_scale=False, forward_only=False): 

        #choose a responsible patch that does not exceed the GPU memory
        PS = 1300 if forward_only else 1300
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
                    if mask[:,hi:hi+PS,oi:oi+PS].sum()>0:
                        outvar += self( inputs[:,:,hi:hi+PS,oi:oi+PS][:,:,mask[0,hi:hi+PS,oi:oi+PS]].unsqueeze(3), name=name, forward_only=forward_only)
                elif (not predict_map) and self.convnet:
                    out, _ = self( inputs[:,:,hi:hi+PS,oi:oi+PS], predict_map=True)
                    outvar += out.sum().cpu()
                else:
                    outvar[:,:,hi:hi+PS,oi:oi+PS], scale[:,:,hi:hi+PS,oi:oi+PS] = self( inputs[:,:,hi:hi+PS,oi:oi+PS], name=name, predict_map=True, forward_only=forward_only)

        if predict_map:    
            out = outvar.squeeze()
        else:
            out = outvar

        if return_scale:
            out = [out,scale]
        
        return out

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
