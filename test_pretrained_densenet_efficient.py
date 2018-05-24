import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import numpy.testing as npt
from pretrained_densenet_efficient import pretrained_efficient_densenet169
import torch.utils.model_zoo as model_zoo

def test():
    model_eff = pretrained_efficient_densenet169()
    model_ori = models.densenet169(pretrained=True)
     
    fake_input = Variable(torch.randn((1,3,224,224)))
    ori_output = model_ori(fake_input)
    eff_output = model_eff(fake_input)
    
    ori_output = ori_output.data.numpy()
    eff_output = eff_output.data.numpy()
    
    npt.assert_almost_equal(ori_output,eff_output)
    
    fake_input = Variable(torch.randn((2,3,224,224)))
    ori_output = model_ori(fake_input)
    eff_output = model_eff(fake_input)
    
    ori_output = ori_output.data.numpy()
    eff_output = eff_output.data.numpy()
    
    npt.assert_almost_equal(ori_output,eff_output)
    print("Okay...")
    
if __name__ == '__main__':
    test()