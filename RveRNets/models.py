import torch
from torch import nn
from torch.nn.functional import relu


class RveRNet(nn.Module):

    def __init__(self, backbone1, backbone2, input_size, output_size):
        super().__init__()
        self.roi_module = backbone1
        self.extra_roi_module = backbone2
        self.freeze()
        self.integrated_fc1 = nn.Linear(self.get_total_feature_dim(input_size), 640) 
        self.integrated_fc2 = nn.Linear(640, output_size)
        self.softmax = nn.LogSoftmax(1)
        
        
    def convert_to_tensor(self, input_obj):
        '''
        The output of the DeiTForImageClassification model is an instance of ImageClassifierOutput, and the Tensor is in the logits attribute of this instance. Therefore, to prevent a TypeError, a separate method was created to handle this.
        '''
        if type(input_obj) is not torch.Tensor:
            return input_obj.logits
        else:
            return input_obj
        
    
    def forward(self, x1, x2):
        out1 = self.roi_module(x1)
        out2 = self.extra_roi_module(x2)
        out1 = self.convert_to_tensor(out1)
        out2 = self.convert_to_tensor(out2)
        x = torch.cat((out1, out2), dim=1)
        x = relu(self.integrated_fc1(x))
        x = self.integrated_fc2(x)        
        return self.softmax(x)
    
    
    def freeze(self, backbone=[]):
        if len(backbone) == 0:
            backbone=[self.roi_module, self.extra_roi_module]
        for bb in backbone:
            for param in bb.parameters():
                param.requires_grad = False        
            
    
    def unfreeze(self, backbone=[]):
        if len(backbone) == 0:
            backbone=[self.roi_module, self.extra_roi_module]
        for bb in backbone:
            for param in bb.parameters():
                param.requires_grad = True 
    
    def get_total_feature_dim(self, input_size):
        # Assuming all backbones output features of the same dimension
        # Modify accordingly if this assumption is incorrect
        return self.get_feature_dim(self.roi_module, input_size) + self.get_feature_dim(self.extra_roi_module, input_size)        
    
    
    def get_feature_dim(self, backbone, input_size):
        # Assuming all backbones output features of the same dimension
        # Modify accordingly if this assumption is incorrect

        try:
            return backbone(torch.zeros(1, 3, input_size, input_size)).size(1)
        except AttributeError as ae:
            return backbone(torch.zeros(1, 3, input_size, input_size)).logits.size(1)  # For DeiT
    
    