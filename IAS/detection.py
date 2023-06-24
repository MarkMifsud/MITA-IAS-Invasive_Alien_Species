import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 50).__str__()
import visualisations2 as vis2
import numpy as np
import copy
import cv2
import gc
import pandas as pd
import torch
import torchvision.transforms as tf
import segmentation_models_pytorch as smp
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch import cuda
from segmentation_models_pytorch.metrics.functional import accuracy as acc
import ipywidgets as widgets

input_layers_count = 3
classes_count = 7


model = smp.UnetPlusPlus(
    encoder_name="resnet152",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use None or `imagenet` pre-trained weights for encoder initialization
    in_channels=input_layers_count,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=classes_count,  # model output channels (number of classes in your dataset, add +1 for background)
    activation='softmax',  # deprecated for some models.  Last activation is self(x)
)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

"""
ListBASEModels = ["Unet++"]
Basebox=widgets.Select(
    options=ListBASEModels,
    value=ListBASEModels[0],
    rows=12,
    description='Base:',
    disabled=False
)

"""

Start = widgets.Button(description='Start Detection', disabled=False,
                       button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                       tooltip='Click me', icon='eye')


def DetectionLoop(b):
    global Start
    Start.close()
    ListRasters = os.listdir(".\\Data\\source\\rasters")
    Rasterbox = widgets.Select(
        options=ListRasters,
        value=ListRasters[0],
        rows=12,
        description='Raster:',
        disabled=False)

    ListModels = os.listdir(".\\Models\\_usable\\")  # checkpoints that are deemed usable for detection
    Modelbox = widgets.Select(
        options=ListModels,
        value=ListModels[0],
        rows=12,
        description='Model:',
        disabled=False)

    Tilebox = widgets.IntText(value=1024, description='Tile Size:', disabled=False)

    t = datetime.now()
    DateTime = str(t.hour) + str(t.minute) + "-" + str(t.day) + "-" + str(t.month) + "-" + str(t.year)
    Savebox = widgets.Text(value='detection' + DateTime, placeholder='detection' + DateTime, description='Save As:',
                           disabled=False)

    Accept = widgets.Button(description='Accept & Proceed', disabled=False,
                            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me',
                            icon='check')

    def on_button_clicked(b):
        print("Loading data and model...", end=" ")
        global Net
        tile_size = Tilebox.value
        file = Rasterbox.value
        epoch = Modelbox.value
        epoch = ".\\Models\\_usable\\" + epoch
        save_as = Savebox.value
        save_as = '.\\Results\\' + save_as
        raster = '.\\Data\\source\\rasters\\' + file
        Net = model.to(device)
        Net.load_state_dict(torch.load(epoch))

        ArgmaxMap = vis2.ArgmaxMapOnly(Net, raster, tilesize=tile_size)
        vis2.Argmax2Output(ArgmaxMap, save_as=save_as)
        row1.close()
        row2.close()
        Tilebox.close()
        Accept.close()
        Net = model.to(device)
        del ArgmaxMap
        gc.collect()
        cuda.empty_cache()
        print('Detection Completed')
        display(Start)

    Accept.on_click(on_button_clicked)
    row1 = widgets.HBox([Modelbox, Rasterbox])
    row2 = widgets.HBox([Tilebox, Savebox])
    display(row1, row2, Accept)

    Start = widgets.Button(description='Start Detection', disabled=False,
                           button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                           tooltip='Click me', icon='eye')
    Start.on_click(DetectionLoop)


Start.on_click(DetectionLoop)
display(Start)
