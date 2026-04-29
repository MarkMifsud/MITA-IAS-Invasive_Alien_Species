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

# --- logging to an Output widget, shared for this UI ---
current_output_widget = None

def log(*args, **kwargs):
    if current_output_widget is not None:
        with current_output_widget:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


# --- discover folders as before ---
ListFolders = []
for file in os.listdir(".\\Data\\trainData\\"):
    d = os.path.join(".\\Data\\trainData\\", file)
    if os.path.isdir(d):
        ListFolders.append(file)

# --- persistent container + log widget ---
container = widgets.VBox()
log_output = widgets.Output()

# --- globals for widgets so we can rebuild UIs ---
Trainbox = None
Validbox = None
Batchbox = None
InChannelsBox = None
OutChannelsBox = None
EpochsBox = None
Accept = None
SingleClassBox = None
AcceptClass = None


def build_main_ui():
    """Build the main training UI (train/valid/batch/epochs/etc)."""
    global Trainbox, Validbox, Batchbox, InChannelsBox, OutChannelsBox, EpochsBox, Accept

    # main widgets
    Trainbox = widgets.Select(
        options=ListFolders,
        value=ListFolders[0] if len(ListFolders) > 0 else None,
        rows=12,
        description='Train:',
        disabled=False
    )

    Validbox = widgets.Select(
        options=ListFolders,
        value=ListFolders[1] if len(ListFolders) > 1 else None,
        rows=12,
        description='Validation:',
        disabled=False
    )

    batch_size = int((torch.cuda.get_device_properties(0).total_memory / 804896768) / 8) * 8
    Batchbox = widgets.IntText(value=batch_size, description='Batch Size:', disabled=False)

    InChannelsBox = widgets.IntText(value=3, description='In Channels:', disabled=False)
    OutChannelsBox = widgets.IntText(value=7, description='Out Channels:', disabled=False)
    EpochsBox = widgets.IntText(value=300, description='Epochs:', disabled=False)

    Accept = widgets.Button(
        description='Accept',
        disabled=False,
        button_style='',
        tooltip='Start training',
        icon='check'
    )

    row0 = widgets.HBox([InChannelsBox, OutChannelsBox])
    row1 = widgets.HBox([Trainbox, Validbox])
    row2 = widgets.HBox([Batchbox, EpochsBox])

    # layout: log + main UI
    container.children = [log_output, row0, row1, row2, Accept]

    Accept.on_click(on_button_clicked)


def show_single_class_ui():
    """Show the single-class selection UI, hiding the main UI."""
    global SingleClassBox, AcceptClass

    SingleClassBox = widgets.IntText(value=10, description='Class:', disabled=False)
    AcceptClass = widgets.Button(
        description='Accept',
        disabled=False,
        button_style='',
        tooltip='Train single class',
        icon='check'
    )

    # hide main UI, show log + single-class UI
    container.children = [log_output, SingleClassBox, AcceptClass]

    AcceptClass.on_click(on_class_accept)


def restore_main_ui():
    """Rebuild and show the main UI again (log persists)."""
    log_output.clear_output()
    build_main_ui()


def on_button_clicked(b):
    """Main Accept button: start training or go to single-class UI."""
    global input_channels, output_channels, current_output_widget

    epochs = EpochsBox.value
    batch_size = Batchbox.value

    TrainFolder = "./Data/trainData/" + Trainbox.value
    ValidFolder = "./Data/trainData/" + Validbox.value

    output_channels = OutChannelsBox.value
    input_channels = InChannelsBox.value

    # route logs to our log_output
    current_output_widget = log_output

    if output_channels <= 2:
        # single-class mode: set to 2 and ask which class
        output_channels = 2
        show_single_class_ui()
    else:
        # hide main UI, keep log visible
        container.children = [log_output]

        with log_output:
            log(f"{epochs} epochs on: {TrainFolder} (train), {ValidFolder} (valid), Batch size: {batch_size}")
            train(TrainFolder, ValidFolder, epochs, batch_size)

        # after training, restore main UI
        restore_main_ui()


def on_class_accept(b):
    """Accept button for single-class training."""
    global current_output_widget

    epochs = EpochsBox.value
    batch_size = Batchbox.value

    TrainFolder = "./Data/trainData/" + Trainbox.value
    ValidFolder = "./Data/trainData/" + Validbox.value

    current_output_widget = log_output

    with log_output:
        st.singleclass = SingleClassBox.value
        log(f"Training only on class: {st.singleclass}")
        log(f"{epochs} epochs on: {TrainFolder} (train), {ValidFolder} (valid), Batch size: {batch_size}")

        # hide single-class UI, keep log visible
        container.children = [log_output]

        train(TrainFolder, ValidFolder, epochs, batch_size)

    # after training, restore main UI (single-class UI can be shown again on next run)
    restore_main_ui()


# --- initial display ---
build_main_ui()
display(container)
