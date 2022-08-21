import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from sklearn.metrics import *

import matplotlib.pyplot as plt
from PIL import Image

from utils import *
from models import *

# model name
model_name = 'ensemble_model_v9_dataset_v3'

# Flag for controlling train the model or evaluate the model
# True means the model will be trained and will be save the trained
# model into a local dir.
# training = True
training = False

# Check GPU on the local machine
CheckGpuAvailability()
ClearGpu()

# Load dataset
# 224 * 224
train_dataset, val_dataset = GetDataSet(
	train_data_path='./datasets_v3/train', val_data_path='./datasets_v3/val'
)

print(train_dataset)
print(val_dataset)

# Load dataset into train and val
train_loader, val_loader = LoadData(
	train_dataset=train_dataset, val_dataset=val_dataset, batch_size=32)


# Load base models
alexnet_path = './saved_models/alexnet_v1_dataset_v3_5.pth'
alexnet = GetAlexNetModel()
alexnet.load_state_dict(torch.load(alexnet_path))

inception_v3_path = './saved_models/inception_v3_v1_dataset_v3_2.pth'
inception_v3 = GetInceptionV3Model()
inception_v3.load_state_dict(torch.load(inception_v3_path))

vgg16_path = './saved_models/vgg16_v1_dataset_v3_5.pth'
vgg16 = GetVgg16NetModel()
vgg16.load_state_dict(torch.load(vgg16_path))

resnet18_path = './saved_models/resnet18_v1_dataset_v3_1.pth'
resnet18 = GetResnet18Model()
resnet18.load_state_dict(torch.load(resnet18_path))

# All of the baseline models
models=[alexnet, vgg16, inception_v3, resnet18]

# Loss Function
criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss()

# Create EnsembleModel Version 2
model = EnsembleModelV2(models=models)
print(model)

# Train the model and save the trained models
if (training):
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	# optimizer = optim.Adam(model.parameters(), \
 #                       lr=0.001, betas=(0.9, 0.999), \
 #                       eps=1e-08, weight_decay=0, \
 #                       amsgrad=False)

	all_model_names, train_loss = TrainModelV2(model=model, criterion=criterion, \
		                           optimizer=optimizer, dataloader=train_loader, \
		                           output_name=model_name, epochs=10)

	np.save('./saved_models/' + model_name + '_model_names', all_model_names)
	np.save('./saved_models/' + model_name + '_train_loss', train_loss)
	ClearGpu()
# Evaluate the models
else:

	all_model_paths = np.load('./saved_models/' + model_name + '_model_names.npy').tolist()
	

	# all_val_loss = GetTrainingMeticsV2(
	# 	model, all_model_paths, val_loader, criterion
	# )
	# np.save('./saved_models/' + model_name + '_val_loss', all_val_loss)

	# all_train_loss = np.load('./saved_models/' + model_name + '_train_loss.npy').tolist()
	# all_val_loss = np.load('./saved_models/' + model_name + '_val_loss.npy')

	# PlotLoss(all_train_loss, all_val_loss)


	# best_model_idx = np.argmin(all_val_loss)
	# print('Best model version is:')
	# print(all_model_paths[best_model_idx])
	# print('Loss on validation set is: {}'.format(all_val_loss[best_model_idx]))
	

	# Get model metrics
	# model.load_state_dict(torch.load(all_model_paths[best_model_idx]))
	# EvalModel(model=model, val_loader=val_loader)

	for path in all_model_paths:
		model.load_state_dict(torch.load(path))
		EvalModel(model=model, val_loader=val_loader)

	ClearGpu()
