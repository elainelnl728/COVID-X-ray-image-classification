import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import pycuda.driver as cuda
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

# Model name
model_name = 'inception_v3_v2_dataset_v3'

# Flag for controlling train the model or evaluate the model
# True means the model will be trained and will be save the trained
# model into a local dir.
# training = True
training = False

# Check GPU on the local machine
CheckGpuAvailability()
ClearGpu()

# Load dataset into train and val
train_dataset, val_dataset = GetDataSetForInception(
	train_data_path='./datasets_v3/train', val_data_path='./datasets_v3/val'
)

print(train_dataset)
print(val_dataset)

# Load dataset into loader
train_loader, val_loader = LoadData(
	train_dataset=train_dataset, val_dataset=val_dataset, batch_size=32)

# Loss function
criterion = nn.CrossEntropyLoss()

# Train the model and save the trained models
if (training):
	model = GetInceptionV3Model()

	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	all_model_names, train_loss = TrainModelV2Inception(model=model, criterion=criterion, \
		                           optimizer=optimizer, dataloader=train_loader, \
		                           output_name=model_name, epochs=15)

	np.save('./saved_models/' + model_name + '_model_names', all_model_names)
	np.save('./saved_models/' + model_name + '_train_loss', train_loss)

	ClearGpu()
# Evaluate the model
else:

	all_model_paths = np.load('./saved_models/' + model_name + '_model_names.npy').tolist()
	all_train_loss = np.load('./saved_models/' + model_name + '_train_loss.npy').tolist()

	model = GetInceptionV3Model()

	all_val_loss = GetTrainingMeticsV2(
		model, all_model_paths, val_loader, criterion
	)
	np.save('./saved_models/' + model_name + '_val_loss', all_val_loss)

	all_val_loss = np.load('./saved_models/' + model_name + '_val_loss.npy')

	PlotLoss(all_train_loss, all_val_loss)


	best_model_idx = np.argmin(all_val_loss)
	print('Best model version is:')
	print(all_model_paths[best_model_idx])
	print('Loss on validation set is: {}'.format(all_val_loss[best_model_idx]))
	

	# Get model metrics
	model.load_state_dict(torch.load(all_model_paths[best_model_idx]))
	EvalModel(model=model, val_loader=val_loader)

	# model.load_state_dict(torch.load(all_model_paths[-1]))
	# EvalModel(model=model, val_loader=train_loader)

	ClearGpu()
