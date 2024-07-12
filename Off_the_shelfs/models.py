import torch
from torch import nn
from torch.nn.functional import relu
  

class CustomNet(nn.Module):
    def __init__(self, backbone, input_size, output_size):
        super().__init__()
        self.backbone = backbone        
        self.freeze()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.get_feature_dim(input_size), output_size),
            nn.LogSoftmax(1)
        )
    
    def forward(self, x):
        xo = self.backbone(x)
        try:
            return self.classifier(xo)
        except TypeError as te:
            return self.classifier(xo.logits) # For DeiT
    
    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_feature_dim(self, input_size):
        # Assuming all backbones output features of the same dimension
        # Modify accordingly if this assumption is incorrect

        try:
            return self.backbone(torch.zeros(1, 3, input_size, input_size)).size(1)
        except AttributeError as ae:
            return self.backbone(torch.zeros(1, 3, input_size, input_size)).logits.size(1)  # For DeiT