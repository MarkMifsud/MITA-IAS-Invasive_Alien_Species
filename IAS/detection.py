# ============================================
#   FIXED, CLASS-BASED, RESTARTABLE DETECTION UI
# ============================================

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
from IPython.display import display

# -------------------------
# CONFIG  (defaults for mutliclass models)
# -------------------------
usable_models_directory = ".\\Models\\_usable\\"
input_layers_count = 3
classes_count = 7

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# -------------------------
# DETECTION UI CONTROLLER
# -------------------------
class DetectionUI:
    def __init__(self):
        self.container = widgets.VBox()
        self.log_output = widgets.Output()
        self.current_output_widget = None

        self.model = None
        self.Net = None

        display(self.container)
        self.build_start_ui()
        

    # -------------------------
    # LOGGING
    # -------------------------
    def log(self, *args, **kwargs):
        if self.current_output_widget is not None:
            with self.current_output_widget:
                print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    # -------------------------
    # UI BUILDERS
    # -------------------------
    def build_start_ui(self):
        """Start button screen."""
        self.current_output_widget = self.log_output
        self.log_output.clear_output()

        self.Start = widgets.Button(
            description='Start Detection',
            icon='eye',
            tooltip='Click to begin',
            button_style=''
        )
        self.Start.on_click(self.start_detection)

        self.container.children = [self.log_output, self.Start]

    def start_detection(self, b):
        """Transition to model/raster/tile selection."""
        self.current_output_widget = self.log_output
        self.log_output.clear_output()

        # Load available models
        global classes_count
        global usable_models_directory
        if classes_count <= 2:
            classes_count = 2
            usable_models_directory = ".\\Models\\_usable_1_class\\"
        ListModels = os.listdir(usable_models_directory)
        if len(ListModels) == 0:
            with self.log_output:
                print("No single-class models found in:", usable_models_directory)
            return

        # Build widgets
        ListRasters = os.listdir(".\\Data\\source\\rasters")

        self.Rasterbox = widgets.Select(
            options=ListRasters,
            value=ListRasters[0],
            rows=12,
            description='Raster:'
        )

        self.Modelbox = widgets.Select(
            options=ListModels,
            value=ListModels[0],
            rows=12,
            description='Model:'
        )

        self.Tilebox = widgets.IntText(value=1024, description='Tile Size:')

        t = datetime.now()
        DateTime = f"{t.hour}{t.minute}-{t.day}-{t.month}-{t.year}"
        self.Savebox = widgets.Text(
            value='detection' + DateTime,
            description='Save As:'
        )

        self.Accept = widgets.Button(
            description='Accept & Proceed',
            icon='check',
            tooltip='Run detection'
        )
        self.Accept.on_click(self.run_detection)

        row1 = widgets.HBox([self.Modelbox, self.Rasterbox])
        row2 = widgets.HBox([self.Tilebox, self.Savebox])

        self.container.children = [self.log_output, row1, row2, self.Accept]

    # -------------------------
    # DETECTION LOGIC
    # -------------------------
    def run_detection(self, b):
        """Hide UI, show log, run detection, then return to Start."""
        self.current_output_widget = self.log_output
        self.log_output.clear_output()

        # Extract selections
        tile_size = self.Tilebox.value
        raster_file = self.Rasterbox.value
        epoch_file = self.Modelbox.value
        epoch_path = usable_models_directory + epoch_file
        save_name = self.Savebox.value
        save_as = '.\\Results\\' + save_name
        raster_path = '.\\Data\\source\\rasters\\' + raster_file

        # Hide UI
        self.container.children = [self.log_output]

        with self.log_output:
            print("Loading model and data...")

        # Load model (with auto-detect output classes)
        if self.model is None:
            self.model = smp.UnetPlusPlus(
                encoder_name="resnet152",
                encoder_weights="imagenet",
                in_channels=input_layers_count,
                classes=classes_count,
                activation='softmax'
            )

        try:
            self.model.load_state_dict(torch.load(epoch_path))
        except Exception as error:
            # Auto-detect output size
            detect_output_size = int(str(error).split("shape torch.Size([")[1].split(",")[0])
            self.model = smp.UnetPlusPlus(
                encoder_name="resnet152",
                encoder_weights="imagenet",
                in_channels=input_layers_count,
                classes=detect_output_size,
                activation='softmax'
            )

        self.Net = self.model.to(device)
        self.Net.load_state_dict(torch.load(epoch_path))

        with self.log_output:
            print("Model loaded.")
            print("Performing detection...")

        # Run detection
        ArgmaxMap = vis2.ArgmaxMapOnly(self.Net, raster_path, tilesize=tile_size)
        vis2.Argmax2Output(ArgmaxMap, save_as=save_as)

        del ArgmaxMap,self.Net
        gc.collect()
        cuda.empty_cache()

        with self.log_output:
            print("Detection completed.")
            print("Saved to:", save_as)

        # Reset UI
        self.build_start_ui()


# -------------------------
# RUN UI
# -------------------------

#DetectionUI()
