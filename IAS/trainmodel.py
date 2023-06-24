from datetime import datetime
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,50).__str__()
from pathlib import Path
import numpy as np
import copy
import cv2
from tqdm import tqdm
import gc
from torch import cuda
import pandas as pd
import ipywidgets as widgets
import torch
import torchvision.transforms as tf
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics.functional import accuracy as acc

import segmentationtraining as st

def train(TrainFolder, ValidFolder, epochs, batchSize, TestFolder=None, Learning_Rate=1e-5, SchedulerName='Plateau', Scheduler_Patience=12, percentagOfUnlabelledTiles=0.075, model=None):
    # if there is a logfile it continues otherwise it starts from scratch

    """global train_with_depthmap
    if os.path.exists(os.path.join(folder, "CHMdepths")) and len(os.listdir(os.path.join(folder, "CHMdepths"))) != 0:
        train_with_depthmap=True
        input_channels=4
    else:
        train_with_depthmap=False
        input_channels=3
    """
    input_channels=3

    if model is None:  #loads the default model configuration
        model = smp.UnetPlusPlus(
        encoder_name="resnet152",
        encoder_weights="imagenet",
        in_channels=input_channels,
        classes=7,
        activation='softmax')

    model_naming_title = st.Netname(model)
    log_path = 'LOG for MC-' + model_naming_title + '.csv'

    if os.path.exists(log_path):
        st.trainFromLastMC(model, TrainFolder, ValidFolder, epochs, batchSize, TestFolder=TestFolder, Learning_Rate=Learning_Rate,SchedulerName='Plateau', Scheduler_Patience=Scheduler_Patience, percentagOfUnlabelledTiles=percentagOfUnlabelledTiles)
    else:
        st.trainStartMC(model, TrainFolder, ValidFolder, epochs, batchSize, TestFolder=TestFolder, Learning_Rate=Learning_Rate,SchedulerName='Plateau', Scheduler_Patience=Scheduler_Patience, percentagOfUnlabelledTiles=percentagOfUnlabelledTiles)

    return

ListFolders=[]
for file in os.listdir(".\\Data\\trainData\\"):
    d = os.path.join(".\\Data\\trainData\\", file)
    if os.path.isdir(d):
        ListFolders.append(file)

Trainbox = widgets.Select(
    options=ListFolders,
    value=ListFolders[0],
    rows=12,
    description='Train:',
    disabled=False)

Validbox = widgets.Select(
    options=ListFolders,
    value=ListFolders[1],
    rows=12,
    description='Validation:',
    disabled=False)

batch_size =int((torch.cuda.get_device_properties(0).total_memory/804896768)/8)*8
Batchbox=widgets.IntText(value=batch_size, description='Batch Size:', disabled=False)

EpochsBox=widgets.IntText(value=300, description='Epochs:', disabled=False)

Accept = widgets.Button(description='Accept', disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon='check')

def on_button_clicked(b):
    epochs=EpochsBox.value
    batch_size=Batchbox.value

    TrainFolder=Trainbox.value
    TrainFolder="./Data/trainData/"+TrainFolder
    ValidFolder=Validbox.value
    ValidFolder="./Data/trainData/"+ValidFolder



    print(epochs,    TrainFolder,    ValidFolder,    batch_size)
    train(TrainFolder, ValidFolder, epochs, batch_size, TestFolder=None, Learning_Rate=1e-5, SchedulerName='Plateau', Scheduler_Patience=12, percentagOfUnlabelledTiles=0.075, model=None)

Accept.on_click(on_button_clicked)

row1=widgets.HBox([ Trainbox, Validbox])
row2=widgets.HBox([Batchbox, EpochsBox])
display(row1,row2 , Accept)