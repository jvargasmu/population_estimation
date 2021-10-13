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


    def forward(self, input, norm,  mask=None, predict_map=False):

        if torch.tensor(input.shape[2:4]).prod()>150**2:
            return self.forward_batchwise(input, norm, mask)

        mean, std = norm

        input = input.to(self.device)

        input = self.net(input)
        
        # Check if masking should be applied
        if mask is not None:
            #return ((std *  self.head_net(input)[:,mask] ) + mean).sum()
            mask = mask.to(self.device)
            return input[:,mask].sum()
        else:
            # input = ((std *  self.head_net(input) ) + mean)
        
            # check if the output should be the map or the sum
            if not predict_map:
                return input.sum()
            else:
                return input
                

    def forward_batchwise(self, input, norm, mask=None, predict_map=False): 

        #choose a responsible patch that does not exceed the GPU memory
        PS = 150
        oh, ow = input.shape[-2:]
        if not predict_map:
            outvar = 0
        else:
            outvar = torch.zeros((1,1,oh, ow), dtype=input.dtype, device=input.device)

        sums = []
        for hi in range(0,oh,PS):
            for oi in range(0,ow,PS):
                if not predict_map:
                    if mask[:,hi:hi+PS,oi:oi+PS].sum()>0:
                        outvar += self( input[:,:,hi:hi+PS,oi:oi+PS][:,:,mask[0,hi:hi+PS,oi:oi+PS]].unsqueeze(3), norm)
                else:
                    outvar[:,:,hi:hi+PS,oi:oi+PS] = self( input[:,:,hi:hi+PS,oi:oi+PS], norm, predict_map=True)
                    
        return outvar