import os
import glob
import cv2
import math
import gc 
import copy
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms as tf
from torch import cuda
from tqdm import tqdm
import pandas as pd

from datetime import datetime

import numpy as np
#from PIL import Image 
#import matplotlib.pyplot as plt 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	
transformImg= tf.Compose([tf.ToPILImage(),tf.ToTensor(),tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ]) #function to adapt image
tensorize=tf.ToTensor()

blanks=0  #the number of pixels outside the raster area of interest

def prepareImages(image_file_path,label_file_path, tilesize=3200, singleclass=10):
	global blanks
	striderX=tilesize  #the size of the prediction maps. Must be divisible by 16 and 6912x6400 max for 24Gb Gpu
	striderY=tilesize
	
	image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
	if type(image)!=type(None):
		print("processing Image...")
	else:
		print("Image not found")
		return
	
	image=torch.from_numpy(image).to(device)
	nonblanks=torch.count_nonzero(image[:,:,3])
	blanks=image.shape[0]*image.shape[1]-nonblanks

	#print ("empty pixels:",int(blanks), "   filled pixles:", int(nonblanks))
	del nonblanks
	blanklayer=torch.logical_not(image[:,:,3])*255
	cv2.imwrite('exclude_mask.png',blanklayer.cpu().detach().numpy())
	del blanklayer

	image=cv2.imread(image_file_path, cv2.IMREAD_COLOR)
	gc.collect()
	cuda.empty_cache()

	height = image.shape[0]
	width = image.shape[1]
	channels = image.shape[2]
	#print("H:",height, "  W:", width, "  C:", channels)


	addPixelsX=striderX-(width%striderX)
	newWidth=width+addPixelsX
	#print (addPixelsX, " more pixels to be added to the left. New width is:", newWidth)

	addPixelsY=striderY-(height%striderY)
	newHeight=height+addPixelsY
	#print (addPixelsY, " more pixels to be added at the bottom. New height is:", newHeight)

	image = cv2.copyMakeBorder(image, 0, addPixelsY, 0, addPixelsX,cv2.BORDER_CONSTANT, value=[0,0,0])
	
	xMoves=newWidth/striderX
	yMoves=newHeight/striderY
	#print("Loop will move ", xMoves, " horizontally, for ", yMoves," different heights")

	# Exclude Area outside Raster
	#First we load the map with zeros 
	blank=cv2.imread('exclude_mask.png', cv2.COLOR_GRAY2BGR )
	blank= cv2.copyMakeBorder(blank, 0, addPixelsY, 0, addPixelsX,cv2.BORDER_CONSTANT, value=[1])
	blank=blank==0
	blank=blank.astype(np.int8)
	""" All This is used to exclude pixels outside the raster area"""
	#print("image: ", image.shape, "  blank: ",  blank.shape)	

	Lbl=cv2.imread(label_file_path, cv2.COLOR_GRAY2BGR )
	if type(image)!=type(None):
		print("processing Label...")
	else:
		print("Label not found")
		return image, blank
	
	if Lbl.max()==255:
		Lbl=Lbl[:,:,1]/255 #enable only for human viewable white on black map
	else:
		if singleclass=='all': Lbl=Lbl/10
		else: Lbl=Lbl==singleclass

	Lbl=Lbl.astype(np.uint8)
	Lbl = cv2.copyMakeBorder(Lbl, 0, addPixelsY, 0, addPixelsX,cv2.BORDER_CONSTANT, value=[0,0,0])
	print(Lbl.shape)

	return image, Lbl, blank, xMoves, yMoves, striderX, striderY, height, width,newHeight,newWidth


def Predict(Net, image_file_path, label_file_path ,tilesize=3200,singleclass=10):

	image, Lbl, blank, xMoves, yMoves, striderX, striderY,height, width,newHeight,newWidth=prepareImages(image_file_path, label_file_path,  tilesize,singleclass) 
	gc.collect()
	cuda.empty_cache()

	result= np.zeros((newHeight,newWidth), dtype=np.float16) #.astype(np.int16)
	maxpred=-2
	minpred=0

	Net.eval()
	for j in tqdm(range(int(yMoves))):
		#print ("row ", j, end=": ")
		for i in range(int(xMoves)):
			#print( i, end=" ")
			segment=image[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]
        
			with torch.no_grad():
				segment=transformImg(segment)
				segment=torch.autograd.Variable(segment, requires_grad=False).to(device).unsqueeze(0) # Load image
				Pred=Net(segment)
				Pred=Pred[0][1]
        
			if float(Pred.max())>maxpred:
				maxpred=float(Pred.max())
				#print("max:",maxpred)
        
			if float(Pred.min())<minpred:
				minpred=float(Pred.min())
				#print("min:",minpred)
            
			b=blank[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX]
			b=torch.from_numpy(b).to(device)
			Pred=Pred*b
        
			result[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX]=Pred.cpu().detach().numpy()
		#print("|")	
		
	return result, minpred, maxpred,image, Lbl, blank, xMoves, yMoves, striderX, striderY,height, width	,newHeight,newWidth


def WhiteMap(Net,image_file_path,label_file_path,tilesize=3200, CutoffPoint=0.5,save_name='White Prediction Map'):
	
	result, minpred, maxpred,image, Lbl, blank, xMoves, yMoves, striderX, striderY,height, width,newHeight,newWidth=Predict(Net,image_file_path,label_file_path,tilesize,singleclass=10)
	
	# generate White prediction map
	result=result>CutoffPoint
	result=result/(maxpred-CutoffPoint)
	result=result*65535
	print("saving White Map...")
	cv2.imwrite( save_name+'.png',result[0:height, 0:width])
	print(save_name,".png has been saved")
	
	
def GrayMap(Net,image_file_path,label_file_path, CutoffPoint=0.5,save_name='Gray Prediction Map'):
	
	result, minpred, maxpred,image, Lbl, blank, xMoves, yMoves, striderX, striderY,height, width,newHeight,newWidth=Predict(Net,image_file_path,label_file_path,tilesize,singleclass=10)
	result=result-CutoffPoint
	result=result/(maxpred+CutoffPoint)
	result=result*65535
	print("saving result...")
	cv2.imwrite( save_name+'.png',result[0:height, 0:width])
	print(save_name,".png has been saved")


def HeatMap(Net,image_file_path,label_file_path, tilesize=3200,CutoffPoint=0,save_name='HeatMap'):
	
	result, minpred, maxpred,image, Lbl, blank, xMoves, yMoves, striderX, striderY,height, width,newHeight,newWidth=Predict(Net,image_file_path,label_file_path,tilesize,singleclass=10)
	result=result-CutoffPoint
	#result=result/(maxpred+CutoffPoint)
	result=result*65535 
	print("Generating Heat Map")
	for j in tqdm(range(int(yMoves))):
		#print ("row ", j, end=": ")
		for i in range(int(xMoves)):
			#print( i, end=" ")
			Pred=result[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]
			Pred=torch.from_numpy(Pred)
			newp= torch.zeros((striderX,striderY,3),dtype=torch.int16)

			LowProb=Pred<=32768
			LowProb=LowProb.type(torch.int16)
			LowProb=Pred*LowProb #only the pixels in prediction with low probability (with their float value)

			MidProb=torch.logical_and(Pred<=49152,Pred>32768) # ,dtype=torch.int16)
			MidProb=(Pred*MidProb) #only the pixels in prediction with probability 50 to 75% (with their float value)

			HiProb=Pred>49152
			HiProb=HiProb.type(torch.int16)
			HiProb=HiProb*Pred #only the pixels in prediction with probability>75% (with their float value)

			LowRed=Pred*LowProb*2
			del LowProb
			gc.collect()
			cuda.empty_cache()
        
			LowRed=LowRed.type(torch.int16)

			MidRed=65536-((MidProb-32768)*4)
			MidRed=MidRed.type(torch.int16)

			newp[:,:,2]=LowRed+MidRed
			del LowRed
			del MidRed
			gc.collect()
			cuda.empty_cache()

			MidGreen=(MidProb-32768)*4
			MidGreen=MidGreen.type(torch.int16)
			del MidProb
			gc.collect()
			cuda.empty_cache()

			HiGreen=65536-((HiProb-49152)*4)

			newp[:,:,1]=MidGreen+HiGreen
			del MidGreen
			del HiGreen
			gc.collect()
			cuda.empty_cache()

			newp[:,:,0]=(HiProb-49152)*4
			del HiProb

			b=blank[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]
			b=torch.from_numpy(b).to(device)
			newp[:,:,0]=newp[:,:,0]*b
			newp[:,:,1]=newp[:,:,1]*b
			newp[:,:,2]=newp[:,:,2]*b
			del b
        
			image[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX]=newp.cpu().detach().numpy()
			gc.collect()
			cuda.empty_cache()
			#END  Generate heat map using GPU
    	
	return cv2.imwrite( save_name+'.png',image[0:height, 0:width])


def UnevenHeatMap(Net,image_file_path,label_file_path, tilesize=3200,CutoffPoint=0,save_name='HeatMap'):
	
	result, minpred, maxpred,image, Lbl, blank, xMoves, yMoves, striderX, striderY,height, width,newHeight,newWidth=Predict(Net,image_file_path,label_file_path,tilesize,singleclass=10)
	result=result-minpred
	max=np.max(result)
	result=result/max
	print ("Recommended Threshold:", (minpred*-1)/max )
	result=result*65535 
	

	for j in range(int(yMoves)):
		print ("row ", j, end=": ")
		for i in range(int(xMoves)):
			print( i, end=" ")
			Pred=result[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]
			Pred=torch.from_numpy(Pred)
			newp= torch.zeros((striderX,striderY,3),dtype=torch.int16)

			LowProb=Pred<=32768
			LowProb=LowProb.type(torch.int16)
			LowProb=Pred*LowProb #only the pixels in prediction with low probability (with their float value)

			MidProb=torch.logical_and(Pred<=49152,Pred>32768) # ,dtype=torch.int16)
			MidProb=(Pred*MidProb) #only the pixels in prediction with probability 50 to 75% (with their float value)

			HiProb=Pred>49152
			HiProb=HiProb.type(torch.int16)
			HiProb=HiProb*Pred #only the pixels in prediction with probability>75% (with their float value)

			LowRed=Pred*LowProb*2
			del LowProb
			gc.collect()
			cuda.empty_cache()
        
			LowRed=LowRed.type(torch.int16)

			MidRed=65536-((MidProb-32768)*4)
			MidRed=MidRed.type(torch.int16)

			newp[:,:,2]=LowRed+MidRed
			del LowRed
			del MidRed
			gc.collect()
			cuda.empty_cache()

			MidGreen=(MidProb-32768)*4
			MidGreen=MidGreen.type(torch.int16)
			del MidProb
			gc.collect()
			cuda.empty_cache()

			HiGreen=65536-((HiProb-49152)*4)

			newp[:,:,1]=MidGreen+HiGreen
			del MidGreen
			del HiGreen
			gc.collect()
			cuda.empty_cache()

			newp[:,:,0]=(HiProb-49152)*4
			del HiProb

			b=blank[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]
			b=torch.from_numpy(b) #.to(device)
			newp[:,:,0]=newp[:,:,0]*b
			newp[:,:,1]=newp[:,:,1]*b
			newp[:,:,2]=newp[:,:,2]*b
			del b
        
			image[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX]=newp.cpu().detach().numpy()
			gc.collect()
			cuda.empty_cache()
			#END  Generate heat map using GPU
    	
	return cv2.imwrite( save_name+'.png',image[0:height, 0:width])


def ConfusionMatrixForModel(Net, image_file_path, label_file_path,singleclass=10,tilesize=3200, threshold=0.5,save_name='Confusion Matrix'):
	global blanks
	image, Lbl, blank, xMoves, yMoves, striderX, striderY,height, width,newHeight,newWidth=prepareImages(image_file_path, label_file_path,  tilesize,singleclass) 

	gc.collect()
	cuda.empty_cache()

	torch.set_default_tensor_type('torch.cuda.FloatTensor') #puts all tensors on GPU by default

	Net.eval()
	print("Generating Confusion Map")
	for j in tqdm(range(int(yMoves))):
		#print ("\n row:", j, end="|")
		for i in range(int(xMoves)):
			#print (" col:", i, end=" ")
			#print("height from ",(j*striderY),"to row",(j*striderY)+striderY, "col ",i*striderX,"to ",(i*striderX)+striderX)
			segment=image[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]
			label=Lbl[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]
        
			with torch.no_grad():
				segment=transformImg(segment)
				segment=torch.autograd.Variable(segment, requires_grad=False).to(device).unsqueeze(0) # Load image
				Pred=Net(segment)
			del segment
			#gc.collect()
			#cuda.empty_cache()
       
			Pred=Pred[0][1]   #are you sure?
			Pred=Pred>=threshold
			label=label!=0
			label=tensorize(label)
			label=label[0].to(device)
        
			match=torch.eq(Pred,label)
			del Pred
			gc.collect()
			cuda.empty_cache()
			match=match.to(device)
 
			map= torch.zeros((striderY,striderX,3),dtype=torch.int8).to(device)
    
			"""
			P predict
			L label
			Q where equal (true p and true n)
	
			L and Q = where L is true and P matchs  (True Positives)  turn to 255 and call it green
			not L and Q = True Negatives 
			L and notQ = L is true but no match (False Negative) turn to 255 and call it Blue
			notL and Not Q = L is false and L+P do no match = False positives turn to 255 and call it Red

			Blue=torch.logical_and(label,torch.logical_not(Pred), out=torch.empty((striderX,striderY), dtype=torch.int8) )
			Green= torch.logical_and(label,Pred, out=torch.empty((striderX,striderY), dtype=torch.int8))
			Red=torch.logical_and(torch.logical_not(label),torch.logical_not(Pred), out=torch.empty((striderX,striderY), dtype=torch.int8)) 
			"""
                
			map[:,:,0]=torch.logical_and(label,torch.logical_not(match), out=torch.empty((striderY,striderX), dtype=torch.int8) )
			map[:,:,1]=torch.logical_and(label,match, out=torch.empty((striderY,striderX), dtype=torch.int8))
			map[:,:,2]=torch.logical_and(torch.logical_not(label),torch.logical_not(match), out=torch.empty((striderY,striderX), dtype=torch.int8))
          
			b=blank[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]
			b=torch.from_numpy(b).to(device)
			map[:,:,0]=map[:,:,0]*b
			map[:,:,1]=map[:,:,1]*b
			map[:,:,2]=map[:,:,2]*b
        
			del b
			gc.collect()
			cuda.empty_cache()
         
			map=map*254
			map=map.cpu().detach().numpy()

			image[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX]=copy.deepcopy(map)    
			# END   Generating Confusion Matrix Map
         
			del map
			gc.collect()
			cuda.empty_cache()
	
	
	image=image[0:height, 0:width]
	image.shape
	print (save_name," | ", image.shape, " | Saved Successfully= ", cv2.imwrite(save_name+'.png',image))

	del Lbl
	del blank

	gc.collect()
	#image=cv2.imread('512 A5 confMat.png', cv2.IMREAD_COLOR)
	image=torch.from_numpy(image).to(device)
 
	#Generate Results
	imagesize=(height*width)-blanks
	fn=torch.count_nonzero(image[:,:,0])
	tp=torch.count_nonzero(image[:,:,1])
	fp=torch.count_nonzero(image[:,:,2])
	tn=imagesize-(fn+tp+fp)

	acc= ((tp+tn))/imagesize 


	#Output Result
	print("TP:",int(tp),"  FP:",int(fp))
	print("FN:",int(fn),"  TN:",int(tn))

	print("\nCorrect  :",int((tp+tn))," =", float(((tp+tn)/imagesize)*100),"%")
	print("Incorrect:",int((fp+fn))," =",float(((fp+fn)/imagesize)*100),"%")

	print("\nAccuracy:", round(float(acc*100),3),"%" )

	return int(tp),int(fp),int(fn),int(tn)




def Precision(TP,FP):
    return TP/(TP+FP)

def Recall(TP,FN):
    return TP/(TP+FN)

def Fone(TP,FP,FN):
	p=Precision(TP,FP)
	r=Recall(TP,FN)
	return 2*(  (p*r)/(p+r) )

def fpfnRatio(FP,FN):
	return (FP + 1)/(FN + 1)



def Netname(Net): #Genrates string based on Model and backbone for naming files
	string=str(Net)[0:50]
	model_name=""
	backbone=""
	writemode=True
	writemode2=False

	for c in string:
		if writemode2==True and c=='(': break
		if c=='(':
			writemode=False
			
		if writemode: model_name=model_name+c
		if c==':': writemode2=True
		if writemode2: backbone=backbone+c
		
	#backbone= backbone[2:11]
	return model_name+'-'+backbone[2:11]



Default_threshold_List=[0.35,0.4,0.5,0.6,0.65]


def PRCurve(Net, image_file_path, label_file_path, threshold_List=Default_threshold_List,singleclass=10):

	PRCurve_titles=['threshold','Precision','Recall','F1-score']
	PRCurve_DB=pd.DataFrame( columns=PRCurve_titles)
	t=datetime.now()
	DateTime =str(t.hour)+str(t.minute)+"-"+str(t.day)+"-"+str(t.month)+"-"+str(t.year)
	path="PRCurveMaps-"+DateTime
	os.makedirs(path)

	for t in threshold_List:
		tp,fp,fn,tn=ConfusionMatrixForModel(Net, image_file_path, label_file_path, singleclass=singleclass, threshold=t, save_name="./"+path+"/"+Netname(Net)+str(t)+"CM")
		p=Precision(tp,fp )
		r=Recall(tp,fn)
		Fone = 2*(  (p*r)/(p+r) )
		new_entry=pd.DataFrame([[t,p, r, Fone ]], columns=PRCurve_titles)
		PRCurve_DB=pd.concat([PRCurve_DB, new_entry])
		PRCurve_DB.to_csv("PRCurve Data for"+Netname(Net)+".csv", sep=",")

	return PRCurve_DB

def LoadNet(model,checkpoint):
	Net = model # Load net
	Net=Net.to(device)
	Net.load_state_dict(torch.load(checkpoint))
	return Net

	
def EpochSelection(model, log_path, image_file_path, label_file_path,  EpochsToTest=[24,49,74,99], CutoffPoint=0, singleclass=10,tilesize=3200,threshold=0.5):
	log_DB=pd.read_csv(log_path, sep=",",index_col=0)
    
	result_titles=['Epoch','CheckPoint', 'TruePositives','FalsePositives','FalseNegatives','TrueNegatives', 'Precision', 'Recall', 'F1','Acc','FP-FN Ratio']
	result_DB=pd.DataFrame( columns=result_titles)
    
	s=log_DB.head(1)  # [log_DB['Epoch'] == 0]
	s=s['Session'][0]
	
	dataset=os.path.basename(os.path.basename(image_file_path))
	dataset=dataset[0:-4]
    
	for e in EpochsToTest:
		c=log_DB[log_DB['Epoch'] == e]
		c=c['CheckPoint']
		if c.empty: 
			print("epoch ",e," was not trained. Skipping it. " )
			continue
		
		c=str(c[0])

		if c=='not saved':
			print("epoch ",e,"is ", c, ". Skipping testing. " )
			continue
		else:
			print("loading epoch:",e, " at Checkpoint:", c )
			Net=LoadNet(model,"./"+c)
		
			#print("Developing confusion Map")
			TP,FP,FN,TN=ConfusionMatrixForModel(Net, image_file_path, label_file_path,singleclass=singleclass,save_name="./"+s+"/"+Netname(Net)+'ConfMatrix-Epoch'+str(e)+"-"+dataset)
			p=Precision(TP,FP )
			r=Recall(TP,FN)
			Fone = 2*((p*r)/(p+r))
			acc= (TP+TN)/(TP+FP+FN+TN)
			fpfnRatio= (FP + 1)/(FN + 1)
			new_entry=pd.DataFrame([[e,c,TP,FP,FN,TN,p,r,Fone,acc,fpfnRatio]], columns=result_titles)
			result_DB=pd.concat([result_DB, new_entry])
			result_DB.to_csv("./"+s+"/epoch selection results "+dataset+".csv", sep=",")
			#result_DB.to_csv("epoch selection results.csv", sep=",")
			gc.collect()
			cuda.empty_cache()
			#print("Developing Heat Map")
			HeatMap(Net,image_file_path,label_file_path, tilesize=3200,CutoffPoint=0,save_name="./"+s+"/"+Netname(Net)+'HeatMap-Epoch'+str(e)+"-"+dataset)
			gc.collect()
			cuda.empty_cache()
        
	return result_DB

	
	
def function_list():
	print("LIST OF FUNCTIONS:")
	print("")
	print("visualsations.WhiteMap: Creates a Black & White prediction map. Black<50% , White>50% ")
	print("visualsations.GrayMap: Creates a grayscale prediction map. Black=0% , White=100% ")
	print("visualsations.HeatMap: Creates a Graded Heat Map. Black=0% , Red=50%, Green=75%, Blue=100% ")
	print("visualsations.SimpleHeatMap: Creates a Simplified Heat Map. 0%=<Black <25%<= Red <50%<= Green <75%<= Blue")
	#print("visualsations.RainbowHeatMap: Like Simplified Heat Map but 0%=<Black <25%<= Red <50%<= Green <75%<= Blue")
	print("visualsations.ConfusionMatrixForModel: Maps the Confusion Matrix. Black=TN Red=FP Green=TP Blue=FN")
	print("visualsations.PRCurve: Generates a Confusion Map, a CSV file with Confusion Matrix details including Precision & Recall and plots a PRCurve for a number of Thresholds")
	print("visualsations.Netname: Generates string based on Model and backbone for naming files according to that data")
    
	print("")
	print("________________________________")
	print("MORE DETAILS:")
	print("")
	print("WhiteMap(Net,image_file_path,label_file_path,singleclass=10,tilesize=3200, CutoffPoint=0.5,save_name='White Prediction Map') ")
	print("")
	print("GrayMap(Net,image_file_path,label_file_path,singleclass=10,tilesize=3200, CutoffPoint=0.5,save_name='White Prediction Map') ")
	print("")
	print("HeatMap(Net,image_file_path,label_file_path, singleclass=10,tilesize=3200,CutoffPoint=0,save_name='HeatMap') ")
	print("returns  boolean value indicating if saving the heatmap was successful")
	print("")
	print("ConfusionMatrixForModel(Net, image_file_path, label_file_path,singleclass=10,tilesize=3200, threshold=0.5,save_name='Confusion Matrix')")
	print("returns TP,FP,FN,TN ")
	print("")
	print("PRCurve(Net, image_file_path, label_file_path, threshold_List=Default_threshold_List, singleclass=10 )")
	print("returns a pandas datasheet for the PRCurve ")
	
	print("")
	print("EpochSelection(model, log_path, image_file_path, label_file_path,  EpochsToTest=[24,49,74,99], CutoffPoint=0, singleclass=10,tilesize=3200,threshold=0.5) ")
	print("returns a pandas datasheet for each epoch. Also saves Confusion Maps and heat maps for each in the Checkpoints folder")