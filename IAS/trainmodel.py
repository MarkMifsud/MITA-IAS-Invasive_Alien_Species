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

input_channels = 3
output_channels = 7

percentagOfUnlabelledTiles = 0.075
model = None
Learning_Rate = 1e-5
TestFolder = None
Scheduler_Patience = 12

SchedulerName='Plateau'

def train(TrainFolder, ValidFolder, epochs, batchSize, TestFolder=TestFolder, Learning_Rate=Learning_Rate, SchedulerName=SchedulerName, Scheduler_Patience=Scheduler_Patience, percentagOfUnlabelledTiles=percentagOfUnlabelledTiles, model=model):
    # if there is a logfile it continues otherwise it starts from scratch

    """global train_with_depthmap
    if os.path.exists(os.path.join(folder, "CHMdepths")) and len(os.listdir(os.path.join(folder, "CHMdepths"))) != 0:
        train_with_depthmap=True
        input_channels=4
    else:
        train_with_depthmap=False
        input_channels=3
    """
    global input_channels
    global output_channels

    if model is None:  #loads the default model configuration
        model = smp.UnetPlusPlus(
        encoder_name="resnet152",
        encoder_weights="imagenet",
        in_channels=input_channels,
        classes=output_channels,
        activation='softmax')

    model_naming_title = st.Netname(model)
    log_path = 'LOG for MC-' + model_naming_title + '.csv'
    #print(log_path)

    if os.path.exists(log_path):
        st.trainFromLastMC(model, TrainFolder, ValidFolder, epochs, batchSize, TestFolder=TestFolder, Learning_Rate=Learning_Rate,SchedulerName=SchedulerName, Scheduler_Patience=Scheduler_Patience, percentagOfUnlabelledTiles=percentagOfUnlabelledTiles)
    else:
        st.trainStartMC(model, TrainFolder, ValidFolder, epochs, batchSize, TestFolder=TestFolder, Learning_Rate=Learning_Rate,SchedulerName=SchedulerName, Scheduler_Patience=Scheduler_Patience, percentagOfUnlabelledTiles=percentagOfUnlabelledTiles)

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

InChannelsBox=widgets.IntText(value=3, description='In Channels:', disabled=False)
OutChannelsBox=widgets.IntText(value=7, description='Out Channels:', disabled=False)
EpochsBox=widgets.IntText(value=300, description='Epochs:', disabled=False)

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

    global input_channels
    global output_channels
    output_channels=OutChannelsBox.value
    input_channels=InChannelsBox.value
    row0.close()
    row1.close()
    row2.close()
    Accept.close()

    if output_channels <= 2:
        output_channels = 2
        AcceptClass = widgets.Button(description='Accept', disabled=False,
                                button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                tooltip='Click me',
                                icon='check')
        SingleClassBox =widgets.IntText(value=10, description='Class:', disabled=False)

        def on_class_accept(b):
            st.singleclass =SingleClassBox.value
            SingleClassBox.close()
            AcceptClass.close()
            print("training only on class: ", st.singleclass, end=" " )
            print(epochs, "epochs on:", TrainFolder, ValidFolder, "Batch size:",batch_size)
            train(TrainFolder, ValidFolder, epochs, batch_size)

        AcceptClass.on_click(on_class_accept)
        display(SingleClassBox, AcceptClass)


    else:
        print(epochs, "epochs on:", TrainFolder, ValidFolder, "Batch size:",batch_size)
        train(TrainFolder, ValidFolder, epochs, batch_size)

Accept.on_click(on_button_clicked)

row0=widgets.HBox([InChannelsBox , OutChannelsBox])
row1=widgets.HBox([ Trainbox, Validbox])
row2=widgets.HBox([Batchbox, EpochsBox])
display(row0, row1, row2, Accept)