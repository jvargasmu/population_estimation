from numpy import dtype
import torch.nn as nn
import torch


class PixTransformNet(nn.Module):

    def __init__(self, channels_in=5, kernel_size = 1,weights_regularizer = None):
        super(PixTransformNet, self).__init__()

        self.channels_in = channels_in
        
        # self.spatial_net = nn.Sequential(nn.Conv2d(2,32,(1,1),padding=0),
        #                                  nn.ReLU(),nn.Conv2d(32,1024,(kernel_size,kernel_size),padding=(kernel_size-1)//2))
        # self.color_net = nn.Sequential(nn.Conv2d(channels_in,32-2,(1,1),padding=0),
        #                                nn.ReLU(),nn.Conv2d(32,1024,(kernel_size,kernel_size),padding=(kernel_size-1)//2))
        self.color_net = nn.Sequential(nn.Conv2d(channels_in,32,(1,1),padding=0),
                                       nn.ReLU(),nn.Conv2d(32,128,(kernel_size,kernel_size),padding=(kernel_size-1)//2))
        self.head_net = nn.Sequential(nn.ReLU(),nn.Conv2d(128, 32, (kernel_size,kernel_size),padding=(kernel_size-1)//2),
                                      nn.ReLU(),nn.Conv2d(32, 1, (1, 1),padding=0))

        if weights_regularizer is None:
            # reg_spatial = 0.0001
            reg_color = 0.001
            reg_head = 0.0001
        else:
            # reg_spatial = weights_regularizer[0]
            reg_color = weights_regularizer[1]
            reg_head = weights_regularizer[2]
        
        self.params_with_regularizer = []
        # self.params_with_regularizer += [{'params':self.spatial_net.parameters(),'weight_decay':reg_spatial}]
        self.params_with_regularizer += [{'params':self.color_net.parameters(),'weight_decay':reg_color}]
        self.params_with_regularizer += [{'params':self.head_net.parameters(),'weight_decay':reg_head}]


    def forward(self, input):

        # input_spatial = input[:,self.channels_in-2:,:,:]
        # input_color = input[:,0:self.channels_in-2,:,:]
        input_color = input[:,0:self.channels_in,:,:]

        merged_features = self.color_net(input_color) #self.spatial_net(input_spatial)
        
        return self.head_net(merged_features)

    def forward_batchwise(self, input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #choose a responsible patch that does not exceed the GPU memory
        PS = 400
        oh, ow = input.shape[-2:]
        output = torch.zeros((1,1,oh, ow), dtype=input.dtype, device=input.device)

        for hi in range(0,oh,PS):
            for oi in range(0,ow,PS):
                output[:,:,hi:hi+PS,oi:oi+PS] = self(input[:,:,hi:hi+PS,oi:oi+PS].to(device))
   

        return output