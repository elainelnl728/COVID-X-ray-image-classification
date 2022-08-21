# Introduction
This dir contains all of the souce code for team 1135. There are many models which includes the four baseline modesl AlexNet, VGG, ResNet and Inception. Those four baseline models are used for get the baseline metrics so that we are able to make improvements.

Apart from the four baseline models, there are few customized models such as ensemble v1 and ensemble v2. The purpose to have those models are for competing with the four baseline models.

This dir also contains a dataset_v3 which includes Curated COVID19 dataset and COVID Chest X-Ray dataset. In order to increase the train difficult, the two dataset are combined and the train dataset includes covid19 and others. The others dir is the combination of normal and pneumonia. The val dir also contains covid19 and others. It is the same as train.


## How to run?
- Install python3 on the mahcine
- Install the necessary libs
- Install libs by running 'pip install lib-name'
- python3 file_name.py


## Notes:
- There is flag called training in each model file. The flag is to control train or evaluate. If you want to train the model, set the flag to true. Also, the train model will be saived into the saved_models dir.
- The model training may take 2-3 hours depends on the epoch and GPU.