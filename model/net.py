import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable, Function
import torch.nn.functional as F
import math

# class GradReverse(Function):
#     def __init__(self, lambd):
#         self.lambd = lambd

#     def forward(self, x):
#         return x.view_as(x)

#     def backward(self, grad_output):
#         return (grad_output * self.lambd)


# def grad_reverse(x, lambd=-1.0):
#     return GradReverse(lambd)(x)

class GradReverse(Function):

    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -1.0)


def grad_reverse(x):
    # return GradReverse(lambd)(x)
    grad = GradReverse()
    return grad.apply(x)

class ResBase50(nn.Module):
    def __init__(self):
        super(ResBase50, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool

    def forward(self, x, reversed=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if reversed == True:
            x = grad_reverse(x)
        x = self.layer1(x)
        # if reversed == True:
        #     x = grad_reverse(x, -1)
        x = self.layer2(x)
        # if reversed == True:
        #     x = grad_reverse(x, -1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
    
class _BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(self.num_features).cuda())
            self.bias = nn.Parameter(torch.Tensor(self.num_features).cuda())
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(self.num_features).cuda())
        self.register_buffer('running_var', torch.ones(self.num_features).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)

class ResClassifier(nn.Module):
    def __init__(self, class_num=31, extract=True, dropout_p=0.5):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
            )
        
        self.fc2 = nn.Linear(1000, class_num)
        self.extract = extract
        self.dropout_p = dropout_p

    def forward(self, x):
        x = x.view(x.size(0),2048)
        fc1_emb = self.fc1(x)
        if self.training:
            fc1_emb.mul_(math.sqrt(1 - self.dropout_p))            
        logit = self.fc2(fc1_emb)

        if self.extract:
            return fc1_emb, logit         
        return logit

