from numpy import dtype
import torch.nn as nn
import torch


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

    def __init__(self, channels_in=5, kernel_size=1, weights_regularizer=0.001, device="cuda" if torch.cuda.is_available() else "cpu", loss=None):
        super(PixScaleNet, self).__init__()

        self.channels_in = channels_in
        self.device = device
        self.exptransform_outputs = loss in ['LogoutputL1', 'LogoutputL2']

        n1 = 128
        n2 = 128
        n3 = 256
        k1,k2,k3,k4 = kernel_size 
        self.convnet = torch.any(torch.tensor(kernel_size)>1)

        self.scalenet = nn.Sequential(                      nn.Conv2d(channels_in-1, n1, (k1,k1), padding=(k1-1)//2),
                                      nn.ReLU(inplace=True),nn.Conv2d(n1, n2, (k2,k2), padding=(k2-1)//2),
                                      nn.ReLU(inplace=True),nn.Conv2d(n2, n3, (k3, k3),padding=(k3-1)//2),
                                    #   nn.ReLU(inplace=True),nn.Conv2d(n3, n3, (k3, k3),padding=(k3-1)//2),
                                    #   nn.ReLU(inplace=True),nn.Conv2d(n3, n3, (k3, k3),padding=(k3-1)//2),
                                      nn.ReLU(inplace=True),nn.Conv2d(n3, 1, (k4, k4),padding=(k4-1)//2),
                                      nn.ReLU(inplace=True)
                                      )
        
        self.params_with_regularizer = []
        self.params_with_regularizer += [{'params':self.scalenet.parameters(),'weight_decay':weights_regularizer}]


    def forward(self, inputs, mask=None, predict_map=False):

        # Check if the image is too large for singe forward pass
        if torch.tensor(inputs.shape[2:4]).prod()>1500**2:
            return self.forward_batchwise(inputs, mask)

        if (mask is not None) and (not predict_map):
            mask = mask.to(self.device)
            inputs = inputs[:,:,mask[0]].unsqueeze(3)

        # Apply network
        inputs = inputs.to(self.device)
        buildings = inputs[:,0:1,:,:]
        inputs = inputs[:,1:,:,:]

        scale = self.scalenet(inputs) 
        inputs = torch.mul(buildings, scale)
        
        # backtransform if necessary before(!) summation
        if self.exptransform_outputs:
            inputs = inputs.exp() 
                
        # Check if masking should be applied
        if mask is not None:
            mask = mask.to(self.device)
            return inputs.sum().cpu()
        else:
            # check if the output should be the map or the sum
            if not predict_map:
                return inputs.sum().cpu()
            else:
                return inputs.cpu(), scale.cpu()


    def forward_batchwise(self, inputs,mask=None, predict_map=False, return_scale=False): 

        #choose a responsible patch that does not exceed the GPU memory
        PS = 1200
        oh, ow = inputs.shape[-2:]
        if predict_map:
            outvar = torch.zeros((1,1,oh, ow), dtype=inputs.dtype, device='cpu')
            scale = torch.zeros((1,1,oh, ow), dtype=inputs.dtype, device='cpu')
        else:
            outvar = 0

        sums = []
        for hi in range(0,oh,PS):
            for oi in range(0,ow,PS):
                if (not predict_map) and (not self.convnet):
                    if mask[:,hi:hi+PS,oi:oi+PS].sum()>0:
                        outvar += self( inputs[:,:,hi:hi+PS,oi:oi+PS][:,:,mask[0,hi:hi+PS,oi:oi+PS]].unsqueeze(3))
                elif (not predict_map) and self.convnet:
                    out, _ = self( inputs[:,:,hi:hi+PS,oi:oi+PS], predict_map=True)
                    outvar += out.sum().cpu()
                else:
                    outvar[:,:,hi:hi+PS,oi:oi+PS], scale[:,:,hi:hi+PS,oi:oi+PS] = self( inputs[:,:,hi:hi+PS,oi:oi+PS], predict_map=True)

        if not predict_map:    
            out = outvar
        else:
            out = outvar.squeeze()

        if return_scale:
            out = [out,scale]
        
        return out

    def forward_one_or_more(self, sample, mask=None):

        total_sum = 0
        valid_samples  = 0
        for i, inp in enumerate(sample):
            if inp[2].sum()>0:

                total_sum += self(inp[0], inp[2]).cpu()
                valid_samples += 1

        if valid_samples==0:
            return None    
        return total_sum








