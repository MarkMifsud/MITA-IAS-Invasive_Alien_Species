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


import torch
import torchvision.transforms as tf
from torch import cuda
import gc


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def create_dir(path):
	""" To creating a directory if it does not exist"""
	if not os.path.exists(path): os.makedirs(path)


def SplitMap(map, mask, size=256, stride=240, prefixletter=None):
	"""
	map: the raster to split
	mask: the mask to split
	size:  size in pixels X and Y for each tile
	prefixletter: use different letter for each raster to distinguish its tiles from other rasters and also prevent over writing
	"""
	
	if size%16!=0: 
		print("Size must be divisible by 16")
		return
		
	if stride%16!=0 and stride<=size: 
		print("Stride must be divisible by 16 and be less or equal to Size")
		return
	
	if prefixletter==None:
		print("prefixletter cannot be None. Assign prefixletter=\"[any letter]\" to SplitMap function " )
		return
	
	if os.path.exists(map)==False:
		print("Raster Map not found, Recheck the path " )
		return
	
	if os.path.exists(mask)==False:
		print("Label Map not found, Recheck the path " )
		return
	
	gc.collect()
	cuda.empty_cache()
	
	print("Loading Images...")
	Img=cv2.imread(map, cv2.IMREAD_COLOR)
    
	if type(Img)==type(None):
		print("Raster not found")
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
		print("Label not found")
		return
	
	print("Images loaded")
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
	print("Images resized")
	
	xMoves=newWidth/striderX
	yMoves=newHeight/striderY
	
	#Lbl=Lbl/255
	#Lbl=Lbl*10
	
	"""  for single class only
	Lbl=Lbl==10
	Lbl=Lbl.astype(np.int8)
	Lbl=Lbl*10
	"""
	
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
	print("Starting loop: ",end="")
	for j in range(int(yMoves)):
		print(" row:",j, "|",end=" ")
		for i in range(int(xMoves)):
			print(i, end=" ")
			segment=Img[j*striderY:(j*striderY)+sizeY, i*striderX:(i*striderX)+sizeX]
			
			TempSegment=torch.from_numpy(segment).to(device)
			RelevantPixels=torch.count_nonzero(TempSegment)
			
			c=0
       
			if int(RelevantPixels)==0:
				countEmptyTiles=countEmptyTiles+1
				#print("Empty segment ",countEmptyTiles, " found")
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
				print(name," non0s",int(RelevantPixels)," Labeled:",int(c), " max:",int(maskSegment.max()))
				
		print("")	
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
 
	print("Empty tiles not included",countEmptyTiles)
	return ClassProportion, int(ClassPixels), int(pixelCount)



def SplitDataMaps(rastermaps,classmaps,size=256, stride=240, trainsplit=0.8):

	"""
	trainsplit: the amount of data to be used for training.  The rest is split equally for val and test
	"""
	print("Analysing data for balancing class in  Train-Valid-Test split...")
	ClassProportion = 0
	nonZeroes = 0
	pixelCount = 0
	for i in range(len(rastermaps)):
		print("processing: ",rastermaps[i]," & ",classmaps[i],"...", end="")
		c,n,p=SplitMap(rastermaps[i], classmaps[i], size, stride, prefixletter=chr(97+i))
		ClassProportion=ClassProportion+c
		nonZeroes=nonZeroes+n 
		pixelCount=pixelCount+p
		print("DONE")

	trainsize=trainsplit
	valsize=(1-trainsplit)/2
	testsize=(1-trainsplit)/2
	
	#how many tiles each subset has
	traintiles=0
	valtiles=0
	testtiles=0

	# how many pixels in the class each subset has
	trainclass=0
	valclass=0
	testclass=0

	create_dir("data/datasets/train/images/")
	create_dir("data/datasets/train/labels/")
	
	create_dir("data/datasets/valid/images/")
	create_dir("data/datasets/valid/labels/")

	create_dir("data/datasets/test/images/")
	create_dir("data/datasets/test/labels/")

	tileDB=pd.read_csv("data/tileDatabase.csv", sep=",")
	
	
	for i in tileDB.index:
    
		if  tileDB['class'][i]==0:
			#print(traintiles*(valsize*10),"<= ",valtiles*(trainsize*10), end="" )
			if traintiles*(valsize*10)<=valtiles*(trainsize*10):            
				tileDB['set'][i]="train"
				traintiles+=1
				#print(" train")
												
				shutil.move(tileDB['label'], "data/datasets/train/labels/"+tileDB['name']+".png")
				shutil.move(tileDB['image'], "data/datasets/train/images/"+tileDB['name']+".tif")
				
				continue
            
			if  valtiles*(testsize*10)>testtiles*(valsize*10):
				tileDB['set'][i]="test"
				testtiles+=1
				shutil.move(tileDB['label'], "data/datasets/test/labels/"+tileDB['name']+".png")
				shutil.move(tileDB['image'], "data/datasets/test/images/"+tileDB['name']+".tif")
			else:
				tileDB['set'][i]="val"
				valtiles+=1
				shutil.move(tileDB['label'], "data/datasets/valid/labels/"+tileDB['name']+".png")
				shutil.move(tileDB['image'], "data/datasets/valid/images/"+tileDB['name']+".tif")
		else:
			#print(trainclass*(valsize*10),"<=",valclass*(trainsize*10)," | ",valclass*(testsize*10), ">",testclass*(valsize*10), end="")
			if trainclass*(valsize*10)<=valclass*(trainsize*10):            
				tileDB['set'][i]="train"
				trainclass+=tileDB['class'][i]
				shutil.move(tileDB['label'], "data/datasets/train/labels/"+tileDB['name']+".png")
				shutil.move(tileDB['image'], "data/datasets/train/images/"+tileDB['name']+".tif")
				continue
            
			if  valclass*(testsize*10)>testclass*(valsize*10):
				tileDB['set'][i]="test"
				testclass+=tileDB['class'][i]
				shutil.move(tileDB['label'], "data/datasets/test/labels/"+tileDB['name']+".png")
				shutil.move(tileDB['image'], "data/datasets/test/images/"+tileDB['name']+".tif") 
			else:
				tileDB['set'][i]="val"
				valclass+=tileDB['class'][i]
				shutil.move(tileDB['label'], "data/datasets/valid/labels/"+tileDB['name']+".png")
				shutil.move(tileDB['image'], "data/datasets/valid/images/"+tileDB['name']+".tif")
            
		#print(" ", tileDB['set'][i])
	return

	
	
	
























