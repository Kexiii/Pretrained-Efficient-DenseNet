import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import numpy.testing as npt
import torch.utils.model_zoo as model_zoo
import densenet_efficient

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

def pretrained_efficient_densenet169():
    model_eff = densenet_efficient.densenet169()
    weights = model_zoo.load_url(model_urls['densenet169'])
    
    # Load weights except for the bottleneck layer
    eff_dict = model_eff.state_dict()
    pretrained_dict = {k: v for k, v in weights.items() if k in eff_dict}
    eff_dict.update(pretrained_dict) 
    model_eff.load_state_dict(eff_dict)
    
    # Extract bottleneck weights 
    bottleneck_weights = []
    for k,v in weights.items():
        ks = k.split('.')
        if len(ks) == 6:
            if ks[4] == '1':
                bottleneck_weights.append(v)
    
    # Load bottleneck weights
    index = 0
    for layer in model_eff.modules():
        if isinstance(layer, densenet_efficient._EfficientDensenetBottleneck):
            layer.norm_weight = nn.Parameter(bottleneck_weights[index*5+0])
            layer.norm_bias = nn.Parameter(bottleneck_weights[index*5+1])
            layer.conv_weight = nn.Parameter(bottleneck_weights[index*5+4])
            index += 1
    assert index == 82, "82 bottleneck layers"
    
    return model_eff