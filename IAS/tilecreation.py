import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

from pathlib import Path
import shutil
import glob
import math
import numpy as np
import copy 
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import cuda
import gc
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

current_output_widget = None

def create_dir(path):
    """ To creating a directory if it does not exist"""
    if not os.path.exists(path): os.makedirs(path)


def log(*args, **kwargs):
    if current_output_widget is not None:
        with current_output_widget:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def SplitMap(map, mask, size=256, stride=240, prefixletter=None):
    """
    map: the raster to split
    mask: the mask to split
    size:  size in pixels X and Y for each tile
    prefixletter: use different letter for each raster to distinguish its tiles from other rasters and also prevent over writing
    """
    
    if size%16!=0: 
        log("Size must be divisible by 16")
        return
        
    if stride%16!=0 and stride<=size: 
        log("Stride must be divisible by 16 and be less or equal to Size")
        return
    
    if prefixletter==None:
        log("prefixletter cannot be None. Assign prefixletter=\"[any letter]\" to SplitMap function " )
        return
    
    if os.path.exists(map)==False:
        log("Raster Map not found, Recheck the path " )
        return
    
    if os.path.exists(mask)==False:
        log("Label Map not found, Recheck the path " )
        return
    
    gc.collect()
    cuda.empty_cache()
    
    #log("Loading Images...")
    Img=cv2.imread(map, cv2.IMREAD_COLOR)
    
    if type(Img)==type(None):
        log("Raster not found")
        return
    
    #if label is in human-visible and RGB format
    #Lbl=cv2.imread(mask, 0)
    #else
    Lbl=cv2.imread(mask, cv2.COLOR_GRAY2BGR )
    #Lbl=cv2.imread(mask, cv2.COLOR_GRAY2BGR ) 
    
    """
    Lbl=torch.from_numpy(Lbl.astype(np.int8)).to(device)
    if int(torch.max(Lbl))==0:
        Lbl=cv2.imread(mask, cv2.COLOR_GRAY2BGR )
    else:
        Lbl=Lbl.cpu().detach().numpy()
    """
    if type(Lbl)==type(None):
        log("Label not found")
        return
    
    #log("Images loaded")
    height = Img.shape[0]
    width = Img.shape[1]
    
    striderX=stride #how much it moves
    striderY=stride
    
    sizeX=size # how large it is
    sizeY=size
    
    addPixelsX=striderX-(width%striderX)  # how much pixels to add to make all input images equal to size
    newWidth=width+addPixelsX
    addPixelsY=striderY-(height%striderY)
    newHeight=height+addPixelsY
    
    """
    if addPixelsX%2==0: 
        addPixelsLeft=addPixelsX/2
        addPixelsX=addPixelsX/2
    else:
        addPixelsLeft=0
        
    if addPixelsY%2==0: 
        addPixelsTop=addPixelsY/2
        addPixelsY=addPixelsY/2
    else:
        addPixelsTop=0
    
    #Img = cv2.copyMakeBorder(Img, int(addPixelsTop), int(addPixelsY), int(addPixelsLeft), int(addPixelsX),cv2.BORDER_CONSTANT, value=[0,0,0])
    #Lbl = cv2.copyMakeBorder(Lbl, int(addPixelsTop), int(addPixelsY), int(addPixelsLeft), int(addPixelsX),cv2.BORDER_CONSTANT, value=[0,0,0])
    """
    
    Img = cv2.copyMakeBorder(Img, 0, int(addPixelsY), 0, int(addPixelsX),cv2.BORDER_CONSTANT, value=[0,0,0])
    Lbl = cv2.copyMakeBorder(Lbl, 0, int(addPixelsY), 0, int(addPixelsX),cv2.BORDER_CONSTANT, value=[0,0,0])
    gc.collect()
    #log("Images resized")
    
    xMoves=newWidth/striderX
    yMoves=newHeight/striderY
    
    Lbl=Lbl.astype(np.int8)
    #Lbl=torch.from_numpy(Lbl.astype(np.int8)).to(device)
    Lbl=torch.from_numpy(Lbl).to(device)

    
    #tensorise=tf.ToTensor()
    #TempLbl=torch.from_numpy(Lbl).to(device)
    

    #nonZeroes=torch.count_nonzero(TempLbl)
    #pixelCount=TempLbl.shape[1]*TempLbl.shape[2]
    #ClassProportion= int(nonZeroes)/pixelCount # the fraction of the image that contains the class
    
    #del TempLbl
    gc.collect()
    cuda.empty_cache()

    
    titles=["tilename","image", "mask","class","set"]
    tileDB = pd.DataFrame( columns=titles)    

    create_dir("./data/datasets/images/")
    create_dir("./data/datasets/labels/")
    #maskpath="data/train/labels/" old path
    #imgpath="data/train/images/" old path
    
    pixelCount=0 
    ClassPixels=0 # the fraction of the image that contains the class
    countEmptyTiles=0
    log("Processing:", end="")
    for j in range(int(yMoves)):
        #log(" row:",j, "|",end=" ")
        log("." , end="")
        for i in range(int(xMoves)):
            #log(i, end=" ")
            segment=Img[j*striderY:(j*striderY)+sizeY, i*striderX:(i*striderX)+sizeX]
            
            TempSegment=torch.from_numpy(segment).to(device)
            RelevantPixels=torch.count_nonzero(TempSegment)
            
            c=0
       
            if int(RelevantPixels)==0:
                countEmptyTiles=countEmptyTiles+1
                #log("Empty segment ",countEmptyTiles, " found")
            else:
                name=prefixletter+"_i"+str(i)+"j"+str(j)
                imgdir=f"data/datasets/images/"+name+".tif"            
                cv2.imwrite( imgdir, segment)

                maskSegment=Lbl[j*striderY:(j*striderY)+sizeY, i*striderX:(i*striderX)+sizeX]        
                c=torch.count_nonzero(maskSegment)
                maskdir="none"
                if int(c)!=0:  #  if tile has no classes it is not saved in order to save disk space. It is then created for training programmatically
                    maskdir=f"data/datasets/labels/"+name+".tif"
                    cv2.imwrite( maskdir, maskSegment.cpu().numpy())
                
                newtile=pd.DataFrame([[name, imgdir, maskdir, int(c), "" ]], columns=titles)
                tileDB=pd.concat([tileDB, newtile])
                
                pixelCount=pixelCount+RelevantPixels #(size*size)
                ClassPixels=ClassPixels+c
                #log(name," non0s",int(RelevantPixels)," Labeled:",int(c), " max:",int(maskSegment.max()))
                
        #log("")
        gc.collect()
        cuda.empty_cache()

    
    ClassProportion=ClassPixels/pixelCount
    
    tileDB=tileDB.sort_values(by=['class'], ascending=True)
     
    if os.path.exists("./data/datasets/tileDatabase.csv"):
        allTilesDB=pd.read_csv("./data/datasets/tileDatabase.csv", sep=",")
        allTilesDB=pd.concat([allTilesDB,tileDB])
        allTilesDB.to_csv("./data/datasets/tileDatabase.csv", sep=",")
        
    else:
        tileDB.to_csv("./data/datasets/tileDatabase.csv", sep=",")
 
    #log("Empty tiles not included",countEmptyTiles)
    return ClassProportion, int(ClassPixels), int(pixelCount)

    


class TileUIController:
    def __init__(self):
        self.container = widgets.VBox()
        self.build_ui()          # <-- build first
        display(self.container)  # <-- display after children exist

    def build_ui(self):
        # UI widgets
        self.Textbox = widgets.Text(
            value='', placeholder='raster name', description='Raster to tile:'
        )
        self.Tilebox = widgets.IntText(value=256, description='Tile Size:')
        self.Stridebox = widgets.IntText(value=256, description='Stride:')
        self.Accept = widgets.Button(description='Accept & Proceed')

        # Output widget for logs (persists across runs)
        self.log_output = widgets.Output()

        # Layout
        self.container.children = [
            self.log_output,
            self.Textbox,
            self.Tilebox,
            self.Stridebox,
            self.Accept
        ]

        self.Accept.on_click(self.on_accept)

    def on_accept(self, b):
        file = self.Textbox.value
        size = self.Tilebox.value
        stride = self.Stridebox.value

        global current_output_widget
        current_output_widget = self.log_output

        # Hide UI, keep log visible
        self.container.children = [self.log_output]

        with self.log_output:
            print(f"Starting run for: {file}")

            label_path = f".\\Data\\source\\labels\\{file}.tif"
            raster_path = f".\\Data\\source\\rasters\\{file}.tif"

            if not os.path.exists(label_path):
                print(f"No .tif file named {file} was found in /Data/source/labels.")
                print("Please check the file name and try again.")
                self.build_ui()
                return

            a, b, c = SplitMap(
                raster_path,
                label_path,
                size=size,
                stride=stride,
                prefixletter=file.upper()
            )

            print(f"| {file} DONE")

        # Rebuild UI (log stays)
        self.build_ui()










