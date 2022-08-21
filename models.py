import os
import shutil
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch libs
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from sklearn.metrics import *

import matplotlib.pyplot as plt
from PIL import Image

from utils import *


class FocalLoss(object):
    """
    This class encapsulates the focal loss function
    """
    def __init__(self):
        self.softmax = softmax = nn.Softmax(dim=1)

    def __call__(self, outputs, labels):
        pred = self.softmax(outputs)
        pred.to(GetDevice())

        labels = torch.nn.functional.one_hot(labels)
        labels.to(GetDevice())

        loss = torch.sum(
            -torch.tensor([3.444, 1.409], dtype=torch.float).to(GetDevice()) * \
            labels * torch.pow(1.0 - pred, 2.0) * torch.log(pred),
            dim=1
        )
        return torch.sum(loss, dim=0)


def FreezeModelAndSetEval(model):
    """
    Set model to evaluation mode and freeze model parameters
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


# Batch norm

class EnsembleModelV2(nn.Module):
    """
    This class encapsulates the ensemble model version 2.
    """
    def __init__(self, models, train_base=False):
        super(EnsembleModelV2, self).__init__()

        # Free parameters in base models
        self.models = models
        for model in self.models:
            if self.training and train_base:
                print('Base model trainable')
                model.train()
            else:
                print('Base model freeze')
                FreezeModelAndSetEval(model)

        num_models = len(self.models)
        num_class = 2

        self.len_feature = num_models * num_class

        self.weight_layer = nn.Linear(self.len_feature * self.len_feature, num_models)

        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(p=0.5, inplace=False)

        self.batch_norm = nn.BatchNorm1d(self.len_feature * self.len_feature)


        self.to(GetDevice())


    def forward(self, x):
        # model, batch, prediction
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # batch, prediction * model
        concat_outputs = torch.cat(outputs, dim=1)

        outer_product = torch.bmm(concat_outputs.unsqueeze(2), \
                                  concat_outputs.unsqueeze(1))

        flatten = outer_product.view(-1, self.len_feature * self.len_feature)


        ### Dropout
        # flatten = self.dropout(flatten)

        ### Batch norm
        normalization = self.batch_norm(flatten)

        # batch, model
        learner_weights = self.tanh(self.weight_layer(normalization))

        # model, 1, batch, prediction
        outputs_unsqueeze = []
        for output in outputs:
            outputs_unsqueeze.append(output.unsqueeze(dim=0))
        # model, batch, prediction
        all_precitions = torch.cat(outputs_unsqueeze, dim=0)
        # batch, model, prediction
        all_precitions = all_precitions.permute(1, 0, 2)

        # batch, model, 1
        learner_weights = learner_weights.unsqueeze(-1)

        return torch.sum(torch.mul(all_precitions, learner_weights), dim=1)


class EnsembleModel(nn.Module):
    """
    This class encapsulates the ensemble model version 1
    """
    def __init__(self, models):
        super(EnsembleModel, self).__init__()

        # Free parameters in base models
        self.models = models
        for model in self.models:
            FreezeModelAndSetEval(model)
            # print(model)

        num_models = len(self.models)
        num_class = 2

        self.len_feature = num_models * num_class

        self.weight_layer = nn.Linear(self.len_feature * self.len_feature, num_models)

        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(p=0.5, inplace=False)

        self.to(GetDevice())

    def forward(self, x):
        # model, batch, prediction
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # batch, prediction * model
        concat_outputs = torch.cat(outputs, dim=1)

        outer_product = torch.bmm(concat_outputs.unsqueeze(2), \
                                  concat_outputs.unsqueeze(1))

        flatten = outer_product.view(-1, self.len_feature * self.len_feature)


        ### Dropout
        # flatten = self.dropout(flatten)

        # batch, model
        learner_weights = self.tanh(self.weight_layer(flatten))

        # model, 1, batch, prediction
        outputs_unsqueeze = []
        for output in outputs:
            outputs_unsqueeze.append(output.unsqueeze(dim=0))
        # model, batch, prediction
        all_precitions = torch.cat(outputs_unsqueeze, dim=0)
        # batch, model, prediction
        all_precitions = all_precitions.permute(1, 0, 2)

        # batch, model, 1
        learner_weights = learner_weights.unsqueeze(-1)

        return torch.sum(torch.mul(all_precitions, learner_weights), dim=1)


def GetInceptionV3Model(model=models.inception_v3(pretrained=True)):\
    """
    This function will create a inception model of version 3. It will also
    replace the classes to 2 instead of the default num classes.
    """
    num_classes = 2
    # Replace the last layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.to(GetDevice())

    return model


def GetVgg16NetModel(model=models.vgg16(pretrained=True)):
    """
    This function will create a VGG16 model. It will also
    replace the classes to 2 instead of the default num classes.
    """
    num_classes = 2
    # Replace the last layer
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.to(GetDevice())

    return model


def GetAlexNetModel(model=models.alexnet(pretrained=True)):
    """
    This function will create a AlexNet model. It will also
    replace the classes to 2 instead of the default num classes.
    """
    num_classes = 2
    # Replace the last layer
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.to(GetDevice())

    return model


def GetResnet18Model(model=models.resnet18(pretrained=True)):
    """
    This function will create a ResNet18 model. It will also
    replace the classes to 2 instead of the default num classes.
    """
    num_classes = 2
    # Replace the last layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.to(GetDevice())

    return model


class GetAlexNetModelV1(nn.Module):
    """
    This class encapsulates the modified version of AlexNet Model.
    """
    def __init__(self):
        super(GetAlexNetModelV1, self).__init__()

        # Number of classes to predict
        num_classes = 2

        self.base_model = models.alexnet(pretrained=True)
        self.relu = nn.ReLU(inplace=True)
        
        # Feature maps from 5th intermediate features layer
        self.features_5 = nn.Sequential(*list(self.base_model.features.children())[:-7])
        self.dropout_5 = nn.Dropout(p=0.5, inplace=False)
        self.f5_linear_1 = nn.Linear(192 * 13 * 13, 4096)
        self.f5_linear_2 = nn.Linear(4096, 256)

        # Feature maps from 9th intermediate features layer
        self.features_9 = nn.Sequential(*list(self.base_model.features.children())[:-3])
        self.dropout_9 = nn.Dropout(p=0.5, inplace=False)
        self.f9_linear_1 = nn.Linear(256 * 13 * 13, 4096)
        self.f9_linear_2 = nn.Linear(4096, 256)

        # Replace the last layer
        num_features = self.base_model.classifier[6].in_features
        self.base_model.classifier[6] = nn.Linear(num_features, 256)

        # Combined output
        self.final_linear_1 = nn.Linear(256 * 3, num_classes)

    def forward(self, x):
        # Original model output
        y = self.base_model(x)
        y = self.relu(y)

        # Intermedia layer output
        f5 = self.features_5(x)
        f5_flatten = f5.view(-1, 192 * 13 * 13)
        f5_flatten = self.dropout_5(f5_flatten)
        f5_l1 = self.relu(self.f5_linear_1(f5_flatten))
        f5_l1 = self.dropout_5(f5_l1)
        f5_l2 = self.relu(self.f5_linear_2(f5_l1))

        f9 = self.features_9(x)
        f9_flatten = f9.view(-1, 256 * 13 * 13)
        f9_flatten = self.dropout_9(f9_flatten)
        f9_l1 = self.relu(self.f9_linear_1(f9_flatten))
        f9_l1 = self.dropout_9(f9_l1)
        f9_l2 = self.relu(self.f9_linear_2(f9_l1))

        final = torch.cat((y, f5_l2, f9_l2), 1)
        final_l1 = self.final_linear_1(final)

        return final_l1
