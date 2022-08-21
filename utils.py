import os
import shutil
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

# torch libs
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from sklearn.metrics import *
import seaborn as sn
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image


def ClearGpu():
  """
  Clear up the GPU
  """
  gc.collect()
  torch.cuda.empty_cache()


def CheckGpuAvailability():
  """
  Check to see if a GPU is available on the machine
  """
  if torch.cuda.is_available():  
    print('GPU is available')
    print('Current device #: {}\n'.format(torch.cuda.current_device()))
  else:
    print('No GPU\n')


def GetDevice():
  """
  Get the device info.
  """
  # Get device
  if torch.cuda.is_available():  
    dev = "cuda:0"
  else:
    dev = "cpu"
  return torch.device(dev)


def GetDataSetV2(path):
  """
  Get the version 2 of the dataset
  """
  preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  covid_len = len(os.listdir(path + "/covid19"))
  non_covid_len = len(os.listdir(path + "/non_covid"))

  print(" covid19: {} \n non_covid: {} \n".format(covid_len, non_covid_len))

  return datasets.ImageFolder(path, preprocess)


# def GetDataIndicesV2(dataset, sample_rate=1.0):
#   total_example = len(dataset)
#   print('# examples: {}'.format(total_example))

#   indices = list(range(total_example))
#   # np.random.seed(random_seed)
#   np.random.shuffle(indices)

#   sampled_example = int(total_example * sample_rate)
#   num_train = int(sampled_example * 0.8)
#   num_val = sampled_example - num_train  

#   split = int(num_val)

#   train_idx, val_idx = indices[split:sampled_example], indices[:split]

#   return train_idx, val_idx


def GetDataLoaderV2(dataset, train_idx, val_idx, batch_size=32):
  """
  Get the version 2 of the data laoder
  """
  train_sampler = SubsetRandomSampler(train_idx)
  val_sampler = SubsetRandomSampler(val_idx)

  train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
  val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

  print('# batches training: {}'.format(len(train_loader)))
  print('# batches validation: {}'.format(len(val_loader)))

  return train_loader, val_loader


# def GetAllDataV2(dataset, train_idx, val_idx):
#   train_sampler = SubsetRandomSampler(train_idx)
#   val_sampler = SubsetRandomSampler(val_idx)

#   train_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=len(train_sampler), sampler=train_sampler)
#   val_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=len(val_sampler), sampler=val_sampler)

#   print('# batches training: {}'.format(len(train_loader)))
#   print('# batches validation: {}'.format(len(val_loader)))

#   all_train_inputs, all_train_labels = next(iter(train_loader))
#   all_val_inputs, all_val_labels = next(iter(val_loader))
  
#   return all_train_inputs, all_train_labels, all_val_inputs, all_val_labels

def TrainModelV2Inception(model, criterion, optimizer, dataloader, output_name, epochs=5, start_epoch=0):
  """
  Change model to training mode, since some module behavior is different in
  in training and evaluation modes.
  """
  model.train()

  print("Model starts training")

  all_model_names = []
  train_loss = []
  for e in range(epochs):
    epoch = e + start_epoch

    start = time.time()
    accumulated_loss = 0.0
    cnt = 0
    for i, data in enumerate(dataloader, 0):
      inputs, labels = data
      inputs, labels = inputs.to(GetDevice()), labels.to(GetDevice())

      with torch.set_grad_enabled(True):
        # For inception V3 model, model returns output, aux
        outputs, aux = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accumulated_loss += loss.item()
        cnt += 1

      display_step = 10
      if (i % display_step == display_step-1):
        print("Epoch: %d, batch: %d, loss: %.3f" %(epoch + 1, i + 1, loss.item()))

    train_loss.append(accumulated_loss / cnt)

    end = time.time()
    
    trained_model_path = "./saved_models/" + output_name + "_" + str(epoch) +".pth"
    torch.save(model.state_dict(), trained_model_path)
    all_model_names.append(trained_model_path)

    print('Epoch: %d, train_loss: %.5f, time: %d' %(epoch + 1, accumulated_loss / cnt, end - start))
    print('----------------------------------')

  print("Model finishes training")
  return all_model_names, train_loss


def TrainModelV2(model, criterion, optimizer, dataloader, output_name, epochs=5, start_epoch=0):
  """
  Change model to training mode, since some module behavior is different in
  in training and evaluation modes.
  """
  model.train()

  print("Model starts training")

  all_model_names = []
  train_loss = []
  for e in range(epochs):
    epoch = e + start_epoch

    start = time.time()
    accumulated_loss = 0.0
    cnt = 0
    for i, data in enumerate(dataloader, 0):
      inputs, labels = data
      inputs, labels = inputs.to(GetDevice()), labels.to(GetDevice())

      with torch.set_grad_enabled(True):
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accumulated_loss += loss.item()
        cnt += 1

      display_step = 10
      if (i % display_step == display_step-1):
        print("Epoch: %d, batch: %d, loss: %.3f" %(epoch + 1, i + 1, loss.item()))

    train_loss.append(accumulated_loss / cnt)

    end = time.time()
    
    trained_model_path = "./saved_models/" + output_name + "_" + str(epoch) +".pth"
    torch.save(model.state_dict(), trained_model_path)
    all_model_names.append(trained_model_path)

    print('Epoch: %d, train_loss: %.5f, time: %d' %(epoch + 1, accumulated_loss / cnt, end - start))
    print('----------------------------------')

    ClearGpu()

  print("Model finishes training")
  return all_model_names, train_loss


def GetModelMetrics(model, dataloader, criterion):
  """
  Calculate and return the metrics of the models.
  """
  model.eval()

  total_loss = 0.0
  cnt = 0
  for data, target in dataloader: 
    data, target = data.to(GetDevice()), target.to(GetDevice())

    output = model(data)
    loss = criterion(output, target)
    total_loss += loss.item()
    cnt += 1
    print(cnt)

  return total_loss / cnt
    

def GetTrainingMeticsV2(model, all_model_paths, val_loader, criterion):
  """
  Get the version 2 of the metrics
  """
  all_val_loss = []
  for model_path in all_model_paths:
    model.load_state_dict(torch.load(model_path))

    print('------------------------')
    print(model_path)
    print('Validation loss...')
    val_loss = GetModelMetrics(model, val_loader, criterion)

    all_val_loss.append(val_loss)

  return all_val_loss


# def GetModelMetricsInception(model, dataloader, criterion):
#   model.eval()

#   total_loss = 0.0
#   cnt = 0
#   for data, target in dataloader: 
#     output = model(data)
#     loss = criterion(output, target)
#     total_loss += loss.item()
#     cnt += 1
#     print(cnt)

#   return total_loss / cnt


# def GetTrainingMeticsV2Inception(model, all_model_paths, val_loader, criterion):
#   all_val_loss = []
#   for model_path in all_model_paths:
#     model.load_state_dict(torch.load(model_path))

#     print('------------------------')
#     print(model_path)
#     print('Validation loss...')
#     val_loss = GetModelMetricsInception(model, val_loader, criterion)

#     all_val_loss.append(val_loss)

#   return all_val_loss

####################### V1 version #########################

def GetDataSetForInception(train_data_path, val_data_path):
  """
  Function to get the dataset for Inception Model
  """
  train_data_path_covid_len = len(os.listdir(train_data_path + "/covid"))
  train_data_path_other_len = len(os.listdir(train_data_path + "/others"))

  val_data_path_covid_len = len(os.listdir(val_data_path + "/covid"))
  val_data_path_other_len = len(os.listdir(val_data_path + "/others"))

  print(" train_covid19: {} \n train_non_covid: {} \n\n val_covid19: {} \n val_non_covid19: {}".
        format(train_data_path_covid_len, train_data_path_other_len, \
               val_data_path_covid_len, val_data_path_other_len))

  preprocess = transforms.Compose([
      # Inception V3 input size is 299
      transforms.Resize(350),
      transforms.CenterCrop(299),
      #####
      transforms.Grayscale(num_output_channels=3),
      transforms.RandomHorizontalFlip(p=0.5),
      #####
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  preprocess_val = transforms.Compose([
      # Inception V3 input size is 299
      transforms.Resize(350),
      transforms.CenterCrop(299),
      #####
      transforms.Grayscale(num_output_channels=3),
      #####
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  train_dataset = datasets.ImageFolder(train_data_path, preprocess)
  val_dataset = datasets.ImageFolder(val_data_path, preprocess_val)

  return train_dataset, val_dataset


def GetDataSet(train_data_path, val_data_path):
  """
  A helper function to load the dataset
  """
  train_data_path_covid_len = len(os.listdir(train_data_path + "/covid"))
  train_data_path_other_len = len(os.listdir(train_data_path + "/others"))

  val_data_path_covid_len = len(os.listdir(val_data_path + "/covid"))
  val_data_path_other_len = len(os.listdir(val_data_path + "/others"))

  print(" train_covid19: {} \n train_non_covid: {} \n\n val_covid19: {} \n val_non_covid19: {}".
        format(train_data_path_covid_len, train_data_path_other_len, \
               val_data_path_covid_len, val_data_path_other_len))

  preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      #####
      transforms.Grayscale(num_output_channels=3),
      transforms.RandomHorizontalFlip(p=0.5),
      #####
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  preprocess_val = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      #####
      transforms.Grayscale(num_output_channels=3),
      #####
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  train_dataset = datasets.ImageFolder(train_data_path, preprocess)
  val_dataset = datasets.ImageFolder(val_data_path, preprocess_val)

  return train_dataset, val_dataset


def LoadData(train_dataset, val_dataset, batch_size=32):
  """
  Load the dataset into train and val loader
  """
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

  print('# batches training: ' + str(len(train_loader)))
  print('# batches testing: ' + str(len(val_loader)))

  return train_loader, val_loader


def GetAllData(train_dataset, val_dataset):
  """
  Load the train and val dataset into loader
  """
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
  all_train_inputs, all_train_labels = next(iter(train_loader))

  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)
  all_val_inputs, all_val_labels = next(iter(val_loader))

  return all_train_inputs, all_train_labels, all_val_inputs, all_val_labels


def GetTestLoss(model, criterion, all_val_inputs, all_val_labels):
  """
  Get the loss
  """
  test_pred = model(all_val_inputs)
  test_loss = criterion(test_pred, all_val_labels)
  return test_loss.item()

def GetTrainLoss(model, criterion, all_train_inputs, all_train_labels):
  """
  Get the train loss
  """
  train_pred = model(all_train_inputs)
  train_loss = criterion(train_pred, all_train_labels)
  return train_loss.item()


def TrainModel(model, criterion, optimizer, dataloader, output_name, all_data, epochs=5):
  """
  Change model to training mode, since some module behavior is different in
  in training and evaluation modes.
  """
  print('Prepare all data for metrics evaluation')
  all_train_inputs, all_train_labels, all_val_inputs, all_val_labels = all_data

  model.train()

  all_test_loss = []
  all_train_loss = []
  for epoch in range(epochs):
    start = time.time()
    accumulated_loss = 0.0

    for i, data in enumerate(dataloader, 0):
      inputs, labels = data

      with torch.set_grad_enabled(True):
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accumulated_loss += loss.item()

      display_step = 3
      if (i % display_step == display_step-1):
        print("Epoch: %d, batch: %d, loss: %.3f" %(epoch + 1, i + 1, accumulated_loss/display_step))
        accumulated_loss= 0.0
    
    end = time.time()
    print('------------------------')
    trained_model_path = "./saved_models/" + output_name + "_" + str(epoch) +".pth"
    torch.save(model.state_dict(), trained_model_path)

    train_loss = GetTrainLoss(model, criterion, all_train_inputs, all_train_labels)
    test_loss = GetTestLoss(model, criterion, all_val_inputs, all_val_labels)

    all_train_loss.append(train_loss)
    all_test_loss.append(test_loss)
    print('Epoch: %d, Train loss: %.3f, Test loss: %.3f, time: %d' %(epoch + 1, train_loss, test_loss, end - start))
    print('------------------------')

  print("Model finishes training")
  return all_train_loss, all_test_loss


def classification_metrics(Y_pred, Y_true):
    acc, auc, precision, recall, f1score = accuracy_score(Y_true, Y_pred), \
                                           roc_auc_score(Y_true, Y_pred), \
                                           precision_score(Y_true, Y_pred), \
                                           recall_score(Y_true, Y_pred), \
                                           f1_score(Y_true, Y_pred)
    return acc, auc, precision, recall, f1score

def processs_prediction(Y_pred, Y_true, label_of_interest):
  pred = [1 if(i == label_of_interest) else 0 for i in Y_pred]
  label = [1 if(i == label_of_interest) else 0 for i in Y_true]
  return pred, label


def eval_model(model, dataloader):
    # Change to evaluation mode
    model.eval()

    Y_pred = []
    Y_test = []

    for data, target in dataloader: 
        # data, target = data.to(GetDevice()), target.to(GetDevice())
        data = data.to(GetDevice())

        softmax_layer = nn.Softmax(dim=1)
        outputs = softmax_layer(model(data))
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().detach().numpy()

        target = target.numpy()
        target = target.reshape((len(target), 1))
        
        Y_test.append(target)
        Y_pred.append(y_pred)
        
    Y_pred = np.concatenate(Y_pred, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)

    return Y_pred, Y_test


def EvalModel(model, val_loader):
  y_pred, y_true = eval_model(model, val_loader)

  y_pred_0, y_true_0 = processs_prediction(y_pred, y_true, label_of_interest=0)
  y_pred_1, y_true_1 = processs_prediction(y_pred, y_true, label_of_interest=1)

  # print(y_pred_0)
  # print(y_pred_1)

  acc, auc, precision, recall, f1 = classification_metrics(y_pred_0, y_true_0)
  print('--------------------------------------')
  print('Covid19 metrics:')
  print(f"accuracy: {acc:.3f}\nAUC: {auc:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}")
  print('\n')

  acc, auc, precision, recall, f1 = classification_metrics(y_pred_1, y_true_1)
  print('Non-Covid19 metrics:')
  print(f"accuracy: {acc:.3f}\nAUC: {auc:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}")
  print('--------------------------------------')
  print('\n')

  tn, fp, fn, tp = confusion_matrix(y_true_0, y_pred_0).ravel()
  total  = len(y_pred_1)
  matrix = [[tp/total*100.0, fn/total*100.0], [fp/total*100.0, tn/total*100.0]]
  df_cm = pd.DataFrame(matrix, range(2), range(2))
  # plt.figure(figsize=(10,7))
  # sn.set(font_scale=1.4) # for label size
  sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="YlGnBu") # font size

  plt.show()


# def eval_model_inception(model, dataloader):
#     # Change to evaluation mode
#     model.eval()

#     Y_pred = []
#     Y_test = []

#     for data, target in dataloader: 
#         softmax_layer = nn.Softmax(dim=1)
#         outputs = model(data)
#         outputs = softmax_layer(outputs)
#         _, y_pred = torch.max(outputs, 1)
#         y_pred = y_pred.detach().numpy()

#         target = target.numpy()
#         target = target.reshape((len(target), 1))
        
#         Y_test.append(target)
#         Y_pred.append(y_pred)
        
#     Y_pred = np.concatenate(Y_pred, axis=0)
#     Y_test = np.concatenate(Y_test, axis=0)

#     return Y_pred, Y_test


# def EvalModelInception(model, val_loader):
#   y_pred, y_true = eval_model_inception(model, val_loader)

#   y_pred_0, y_true_0 = processs_prediction(y_pred, y_true, label_of_interest=0)
#   y_pred_1, y_true_1 = processs_prediction(y_pred, y_true, label_of_interest=1)

#   # print(y_pred_0)
#   # print(y_pred_1)

#   acc, auc, precision, recall, f1 = classification_metrics(y_pred_0, y_true_0)
#   print('Covid19 metrics:')
#   print('--------------------------------------')
#   print(f"accuracy: {acc:.3f}\nAUC: {auc:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}")
#   print('\n')

#   acc, auc, precision, recall, f1 = classification_metrics(y_pred_1, y_true_1)
#   print('Non-Covid19 metrics:')
#   print('--------------------------------------')
#   print(f"accuracy: {acc:.3f}\nAUC: {auc:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}")
#   print('\n')


def PlotLoss(alexnet_train_loss, alexnet_val_loss):
  plt.subplots(figsize=(7, 5))
  plt.plot(range(len(alexnet_train_loss)), alexnet_train_loss, marker='.', \
    markersize=12, label='Train loss')
  plt.plot(range(len(alexnet_val_loss)), alexnet_val_loss, marker='.', \
    markersize=12, label='Val loss')
  plt.xlabel('Epoch', fontsize='x-large')
  plt.ylabel('Loss', fontsize='x-large')
  plt.legend(fontsize='x-large')
  plt.grid()
  plt.show()
