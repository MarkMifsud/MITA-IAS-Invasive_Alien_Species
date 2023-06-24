import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,50).__str__()
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

transformGrayscaleImg= tf.Compose([tf.ToPILImage(),tf.ToTensor()]) #function to adapt image

blanks=0  #the number of pixels outside the raster area of interest
grayscale=False

height=0
width=0
xMoves=0
yMoves=0
striderX=0
striderY=0
newHeight=0
newWidth=0


def LoadNet(model,checkpoint):
    Net = model # Load net
    Net=Net.to(device)
    Net.load_state_dict(torch.load(checkpoint))
    return Net


def LoadPredictionTensor(file_path):
    tensor=torch.load(file_path)
    return tensor.to(device)


def CountBlanks(image_file_path):
    global blanks
    image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
    if type(image)!=type(None):
        print("processing Image...")
    else:
        print("Image not found")
        return

    image=torch.from_numpy(image).to(device)
    nonblanks=torch.count_nonzero(image[:,:,3])  #layer 4 is alpha, nonblanks
    blanks=image.shape[0]*image.shape[1]-nonblanks

    del image
    gc.collect()
    cuda.empty_cache()

    return int(blanks)



def prepareImagesAllClasses(image_file_path,label_file_path, tilesize, ignoreLabel=False):
    global blanks  # number of blank pixels in the raster

    global height
    global width
    global xMoves
    global yMoves
    global newHeight
    global newWidth

    global striderX
    global striderY
    striderX=tilesize
    striderY=tilesize

    image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
    if type(image)!=type(None):
        print("processing Image...")
    else:
        print("Image not found")
        return

    image=torch.from_numpy(image).to(device)
    nonblanks=torch.count_nonzero(image[:,:,3])  #layer 4 is alpha, nonblanks 
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

    if ignoreLabel==False:  #used to process only the image
        Lbl=cv2.imread(label_file_path, cv2.COLOR_GRAY2BGR )
        if type(image)!=type(None):
            print("processing Label...")
        else:
            print("Label not found")
            return image, blank

        Lbl=Lbl/10
        Lbl=Lbl.astype(np.uint8)
        Lbl = cv2.copyMakeBorder(Lbl, 0, addPixelsY, 0, addPixelsX,cv2.BORDER_CONSTANT, value=[0])
        print(image.shape,' ',Lbl.shape)
    else:
        Lbl=None


    return image, Lbl, blank, xMoves, yMoves, height, width,newHeight,newWidth



def PredictAllClasses(Net, image, blank, xMoves, yMoves, striderX, striderY, height, width,newHeight,newWidth):
    global grayscale
    gc.collect()
    cuda.empty_cache()



    Net.eval()
    with torch.no_grad():
        segment=image[0:striderY, 0:striderX]
        segment=transformImg(segment)
        segment=torch.autograd.Variable(segment, requires_grad=False).to(device).unsqueeze(0) # Load image
        Pred=Net(segment)
        Pred_class_count=Pred.size(dim=1)

    result= torch.zeros((Pred_class_count,newHeight,newWidth), dtype=torch.float16).to(device) #.astype(np.int16)

    for j in tqdm(range(int(yMoves))):
        #print ("row ", j, end=": ")
        for i in range(int(xMoves)):
            #print( i, end=" ")
            segment=image[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]
        
            with torch.no_grad():
                if grayscale:
                    segment=cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
                    segment=transformGrayscaleImg(segment)
                else:
                    segment=transformImg(segment)

                segment=torch.autograd.Variable(segment, requires_grad=False).to(device).unsqueeze(0) # Load image
                Pred=Net(segment)
                """
                if singleclass=='all':
                    Pred=torch.argmax(Pred[0], dim=0)
                else:
                    Pred=Pred[0][int(int(singleclass/10))] 
                """

            # now we set predictions of pixels outside rasters to zero
            b=blank[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX]
            b=torch.from_numpy(b).to(device)
            Pred=Pred*b
            b=b==0
            b=b*6000
            Pred[0,0]=Pred[0,0]+b


 
 
            result[:,j*striderY:(j*striderY)+striderY, i*striderX: (i*striderX)+striderX ]= Pred[0] # .cpu().detach().numpy()
        #print("|")

    return result


def GeneratePredictionWithLabel(Net, image_file_path,label_file_path, tilesize=1600, save_name='RawPrediction'):
    image, Lbl, blank, xMoves, yMoves, height, width,newHeight,newWidth=prepareImagesAllClasses(image_file_path,label_file_path, tilesize, ignoreLabel=False)
    striderX=tilesize
    striderY=tilesize

    RawPrediction=PredictAllClasses(Net, image, blank, xMoves, yMoves, striderX, striderY, height, width,newHeight,newWidth)

    # removing padding
    RawPrediction=RawPrediction[:,0:height, 0:width]
    Lbl=Lbl[0:height, 0:width] 

    del image
    #del blank

    gc.collect()
    cuda.empty_cache()

    if save_name!= None:  
        torch.save(RawPrediction, save_name+'.pred')
        print("SAVED "+save_name+'.pred  to file')


    return RawPrediction, Lbl

def ArgmaxMapOnly(Net, image_file_path, tilesize=1600, save_name=None):

    image, Lbl, blank, xMoves, yMoves, height, width,newHeight,newWidth=prepareImagesAllClasses(image_file_path , None, tilesize, ignoreLabel=True)
    striderX=tilesize
    striderY=tilesize

    Pred=PredictAllClasses(Net, image, blank, xMoves, yMoves, striderX, striderY, height, width,newHeight,newWidth)
    Pred=Pred[:,0:height, 0:width]
    
    ArgmaxMap= np.zeros((height,width),dtype=np.int8)
    
    for row in tqdm(range(height)): #for all rows, pixels in each row are processed in parallel
        pred=Pred[:,row][:]
        pred=torch.argmax(pred, dim=0)
        #print(pred.shape, torch.count_nonzero(pred))
        #pred=torch.argmax(pred, dim=0) #argmax prediction for the row
        ArgmaxMap[row]=pred.cpu().detach().numpy()

    if save_name!=None: cv2.imwrite('Argmax-'+save_name+'.png', ArgmaxMap)
    del Pred
    del pred
    del image
    gc.collect()
    cuda.empty_cache()
    return  ArgmaxMap


def Pred2Result(Pred, Lbl, save_as):
    conf_matrix=torch.zeros([7,7])

    ArgmaxMap= np.zeros((Lbl.shape[0],Lbl.shape[1]),dtype=np.int8)
    
    for row in tqdm(range(Lbl.shape[0])): #for all rows, pixels in each row are processed in parallel
        label=Lbl[row]

        pred=Pred[:,row][:]
        pred=torch.argmax(pred, dim=0) #argmax prediction for the row
        
    
        for i in range(7):  #labels
            tempLabel=label==i
            for j in range(7):  #predictions
                tempPred=pred==j
                match=torch.logical_and(torch.from_numpy(tempLabel).to(device),tempPred)
                conf_matrix[i,j]=conf_matrix[i,j]+torch.count_nonzero(match)
        
        ArgmaxMap[row]=pred.cpu().detach().numpy()
        
    blankpixels=int(blanks)
    conf_matrix[0,0]=conf_matrix[0,0]-blankpixels # subtraction compensates for non raster pixels

    class_names = ('None', 'Arundo', 'Opuntia', 'Agri','Eucalyp', 'Agave', 'Acacia')
    conf_matrix=conf_matrix.type(torch.int32).cpu().detach().numpy()

    # Create pandas dataframe
    dataframe = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    dataframe.to_excel(save_as+".xlsx")
    cv2.imwrite('Argmax-'+save_as+'.png',ArgmaxMap)
    
    return dataframe, ArgmaxMap

def Argmax2ConfMap(ArgmaxMap,Lbl,save_as,species=1):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    H=Lbl.shape[0]
    W=Lbl.shape[1]
    ArgmaxMap=ArgmaxMap==species
    Lbl=Lbl==species
    
    ArgmaxMap=torch.from_numpy(ArgmaxMap).to(device)
    Lbl=torch.from_numpy(Lbl).to(device)
    match=torch.eq(ArgmaxMap,Lbl).to(device)
    """
    del ArgmaxMap
    gc.collect()
    cuda.empty_cache()
    """
    
    map= torch.zeros((H,W,3),dtype=torch.int8).to(device)
    #print(ArgmaxMap.device," ",Lbl.device," ",match.device," ",map.device, " ")
    map[:,:,0]=torch.logical_and(Lbl,torch.logical_not(match), out=torch.empty((H,W), dtype=torch.int8) )
    map[:,:,1]=torch.logical_and(Lbl,match, out=torch.empty((H,W), dtype=torch.int8))
    map[:,:,2]=torch.logical_and(torch.logical_not(Lbl),torch.logical_not(match), out=torch.empty((H,W), dtype=torch.int8))
    """
    blank=cv2.imread('exclude_mask.png', cv2.COLOR_GRAY2BGR)
    blank=torch.from_numpy(blank).to(device)
    map[:,:,0]=map[:,:,0]*blank
    map[:,:,1]=map[:,:,1]*blank
    map[:,:,2]=map[:,:,2]*blank
    """
    
    map=map.cpu().detach().numpy()
    map=map*254
    cv2.imwrite(save_as+'-'+str(species)+'0.png', map)
    del map
    gc.collect()
    cuda.empty_cache()
    return
    
    
def Argmax2Output(ArgmaxMap,save_as='Species Detection'):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    H=ArgmaxMap.shape[0]
    W=ArgmaxMap.shape[1]
    ArgmaxMap=torch.from_numpy(ArgmaxMap)
    map= torch.zeros((H,W,3),dtype=torch.uint8).to(device)
    
    palette= [[0,0,0],[0, 200, 0], [0,0,240], [120, 32,32], [254, 0, 254],[128,0, 128], [0,255,255]]
    channels=[0,1,2]
    
    torch.where(ArgmaxMap==3,0,ArgmaxMap)
    map[:,:,0]=ArgmaxMap
    map[:,:,1]=ArgmaxMap
    map[:,:,2]=ArgmaxMap
    
    for color in range(len(palette)):
        if (color==0 or color==3): continue
        for channel in channels:
            map[:,:,channel]=torch.where(map[:,:,channel]==color,palette[color][channel],map[:,:,channel])
            
      
    map=map.cpu().detach().numpy()
    
    cv2.imwrite(save_as+'.tif', map)
    print(save_as+'.tif  SAVED')
    del map
    gc.collect()
    cuda.empty_cache()
    return




def PaddedPredictionWithLabel(Net, image_file_path,label_file_path, tilesize=1600, save_name='RawPrediction'):

    global height
    global width
    global xMoves
    global yMoves
    global height
    global width
    global newHeight
    global newWidth

    image, Lbl, blank, xMoves, yMoves, height, width,newHeight,newWidth=prepareImagesAllClasses(image_file_path,label_file_path, tilesize)
    striderX=tilesize
    striderY=tilesize

    RawPrediction=PredictAllClasses(Net, image, blank, xMoves, yMoves, striderX, striderY, height, width,newHeight,newWidth)


    del image
    del blank
    gc.collect()
    cuda.empty_cache()

    if save_name!= None:
        torch.save(RawPrediction, save_name+'.Pred')
        print("SAVED "+save_name+'.Pred  to file')


    return RawPrediction, Lbl



def HeatMapFromPrediction(result ,CutoffPoint=0):
    # takes a single class prediction (result) and outputs a heatmap

    global blanks  #a number of blanks
    global grayscale

    global height
    global width
    global xMoves
    global yMoves
    global height
    global width
    global newHeight
    global newWidth


    #print("result.shape: ",result.shape)
    #result=result-CutoffPoint
    #result=result/(maxpred+CutoffPoint)
    #result=result*65535
    print("Generating Heat Map")

    output=torch.zeros((newHeight,newWidth,3), dtype=torch.int16)

    for j in tqdm(range(int(yMoves))):
        #print ("row ", j, end=": ")
        for i in range(int(xMoves)):
            #print( i, end=" ")
            Pred=result[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]
            ############Pred=torch.from_numpy(Pred)
            Pred=Pred*65535
            newp= torch.zeros((striderX,striderY,3),dtype=torch.int16)

            LowProb=torch.logical_and( 16384<Pred , Pred<=32768 )
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

            """
            b=blank[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]
            b=torch.from_numpy(b).to(device)
            newp[:,:,0]=newp[:,:,0]*b
            newp[:,:,1]=newp[:,:,1]*b
            newp[:,:,2]=newp[:,:,2]*b
            del b
            """

            output[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX]=newp.cpu().detach()
            gc.collect()
            cuda.empty_cache()
            #END  Generate heat map using GPU

    return output[0:height, 0:width].cpu().detach().numpy()


def ConfusionMatrixFromPerdiction(pred, Lbl,singleclass,tilesize=3200, threshold=0.5):
    global blanks  #a number of blanks
    global grayscale

    global height
    global width
    global xMoves
    global yMoves
    global height
    global width
    global newHeight
    global newWidth

    gc.collect()
    cuda.empty_cache()

    image=torch.zeros( (newHeight,newWidth,3) ,dtype=torch.int8).to(device)

    torch.set_default_tensor_type('torch.cuda.FloatTensor') #puts all tensors on GPU by default


    print("Generating Confusion Map")
    for j in tqdm(range(int(yMoves))):
        #print ("\n row:", j, end="|")
        for i in range(int(xMoves)):

            label=Lbl[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]
            Pred=pred[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]

            Pred=Pred>=threshold
            label=label==int(singleclass/10)
            label=tensorize(label)
            label=label[0].to(device)
        
            match=torch.eq(Pred,label)
            del Pred
            gc.collect()
            cuda.empty_cache()
            match=match.to(device)
 
            map= torch.zeros((striderY,striderX,3),dtype=torch.int8).to(device)
            map[:,:,0]=torch.logical_and(label,torch.logical_not(match), out=torch.empty((striderY,striderX), dtype=torch.int8) )
            map[:,:,1]=torch.logical_and(label,match, out=torch.empty((striderY,striderX), dtype=torch.int8))
            map[:,:,2]=torch.logical_and(torch.logical_not(label),torch.logical_not(match), out=torch.empty((striderY,striderX), dtype=torch.int8))

            gc.collect()
            cuda.empty_cache()
         
            map=map*254
            map=map.type(torch.int8) #.cpu().detach().numpy()

            image[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX]=copy.deepcopy(map)
            # END   Generating Confusion Matrix Map
         
            del map
            gc.collect()
            cuda.empty_cache()


    image=image[0:height, 0:width]

    del Lbl
    #del blank
    gc.collect()

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

    return image.cpu().detach().numpy() , int(tp),int(fp),int(fn),int(tn)


def FindBestEpochs(log_path):
    log_DB = pd.read_csv(log_path, sep=",", index_col=0)
    LastEpoch = int(log_DB.tail(1)['Epoch'])
    best_train_loss = log_DB['Train-Loss'].min()
    best_val_loss = log_DB['Val-Loss'].min()
    best_test_loss = log_DB['Test-Loss'].min()
    best_test_acc = log_DB['Test-Acc'].max()

    c = log_DB[log_DB['Train-Loss'] == log_DB['Train-Loss'].min()]
    best_train_loss = int(c['Epoch'][0])

    c = log_DB[log_DB['Val-Loss'] == log_DB['Val-Loss'].min()]
    best_val_loss = int(c['Epoch'][0])

    if log_DB['Test-Acc'].max() != 0:
        c = log_DB[log_DB['Test-Loss'] == log_DB['Test-Loss'].min()]
        best_test_loss = int(c['Epoch'][0])
        c = log_DB[log_DB['Test-Acc'] == log_DB['Test-Acc'].max()]
        best_test_acc = int(c['Epoch'][0])
        list = [best_train_loss, best_val_loss, best_test_loss, best_test_acc, LastEpoch]
    else:
        list = [best_train_loss, best_val_loss, LastEpoch]

    i = -1
    for element in list:  # eliminates duplicates
        i = i + 1
        j = -1
        for compared in list:
            j = j + 1
            if i == j:
                continue
            else:
                if element == compared: list[j] = 0

    for element in list:
        try:
            list.remove(0)
        except:
            break

    try:
        list.remove(0)
    except:
        print("")

    return list




def Precision(TP,FP):
    if TP+FP != 0:
        return TP/(TP+FP)
    else:
        return 0

def Recall(TP,FN):
    if TP+FN!=0:
        return TP/(TP+FN)
    else:
        return 0

def Fone(TP,FP,FN):
    p=Precision(TP,FP)
    r=Recall(TP,FN)
    if p+r!=0:
        return 2*(  (p*r)/(p+r) )
    else:
        return 0

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


def EvaluateRasters(model, log_path, raster_list=[],label_list=[],EpochsToTest=None,classList=[10,50],tsholdList=[0.5],TileSizeList=[3200]):

    start=datetime.now()
    print ("Started at: ", start)


    #______BEGIN Verifications__________

    if len(raster_list)!=len(label_list) or len(label_list)==0:
        print("raster list and label list don't match in size, or are empty")
        return

    if EpochsToTest==None:
        EpochsToTest=FindBestEpochs(log_path)

    #______END Verifications__________


    log_DB=pd.read_csv(log_path, sep=",",index_col=0)
    
    result_titles=['Raster','Epoch','CheckPoint','Tile Size','Threshold','TruePositives','FalsePositives','FalseNegatives','TrueNegatives', 'Precision', 'Recall', 'F1','Acc','FP-FN Ratio']
    result_DB=pd.DataFrame( columns=result_titles)
    
    s=log_DB.head(1)  # [log_DB['Epoch'] == 0]
    s=s['Session'][0]

    
    for i in range(len(raster_list)):  #repeats for rasters and saved tensors
        dataset=os.path.basename(os.path.basename(raster_list[i]))
        dataset=dataset[0:-4]
        for size in TileSizeList:
            image, Lbl, blank, xMoves, yMoves, height, width,newHeight,newWidth=prepareImagesAllClasses(raster_list[i],label_list[i], size)

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
                    netname=Netname(Net)


                striderX=size
                striderY=size
                prediction=PredictAllClasses(Net, image, blank, xMoves, yMoves, striderX, striderY, height, width,newHeight,newWidth)

                t=datetime.now()
                DateTime =str(t.hour)+str(t.minute)+"-"+str(t.day)+"-"+str(t.month)+"-"+str(t.year)

                for cls in classList:
                    Pred=prediction[int(cls/10)] #pick the prediction layer to work with


                    for th in tsholdList:
                        cm,TP,FP,FN,TN=ConfusionMatrixFromPerdiction(Pred, Lbl, tilesize=3200, threshold=th)
                        cv2.imwrite("./"+s+"/"+Netname(Net)+'CONFUSIONmap-Epoch'+str(e)+"-"+dataset+'-Class '+str(cls)+'-TH'+str(th*100)+'-'+DateTime +'.png', cm )
                        del cm
                        gc.collect()
                        torch.cuda.empty_cache()

                        p=Precision(TP,FP )
                        r=Recall(TP,FN)
                        f_one = Fone(TP,FP,FN)
                        acc= (TP+TN)/(TP+FP+FN+TN)
                        fpfnRatio= (FP + 1)/(FN + 1)
                        new_entry=pd.DataFrame([[i,e,c,size,th,TP,FP,FN,TN,p,r,f_one,acc,fpfnRatio]], columns=result_titles)
                        result_DB=pd.concat([result_DB, new_entry])
                        result_DB.to_csv("./"+s+"/Evaluation Results "+netname+'-'+DateTime+".csv", sep=",")

                    heatmap=HeatMapFromPrediction(Pred, xMoves, yMoves, striderX, striderY,height, width,newHeight,newWidth)
                    cv2.imwrite("./"+s+"/"+Netname(Net)+'HEATmap-Epoch'+str(e)+"-"+dataset+'-Class '+str(cls)+'-'+DateTime +'.png', heatmap )
                    del heatmap
                    gc.collect()
                    torch.cuda.empty_cache()

                del prediction
                del Net
                gc.collect()
                cuda.empty_cache()

        del image
        del Lbl
        del blank
        gc.collect()
        cuda.empty_cache()

    end=datetime.now()
    print ("Finished at: ", end)
    duration=end-start
    print ("Duration: ",  duration.minutes, "   (in seconds): ", duration.seconds)



def ArgmaxMCNoBG(Net, RawPrediction, Lbl , blank, tilesize=3200,  singleclass=10, save_name='Argmax'):
    global blanks
    global grayscale
    global height
    global width
    global xMoves
    global yMoves
    global height
    global width
    global newHeight
    global newWidth
    #image, Lbl, blank, xMoves, yMoves, striderX, striderY,height, width,newHeight,newWidth=prepareImages(image_file_path, label_file_path,  tilesize)

    gc.collect()
    cuda.empty_cache()

    torch.set_default_tensor_type('torch.cuda.FloatTensor') #puts all tensors on GPU by default

    Net.eval()
    print("Generating Correct Map for mutliclass ")
    for j in tqdm(range(int(yMoves))):
        #print ("\n row:", j, end="|")
        for i in range(int(xMoves)):
            #print (" col:", i, end=" ")
            #print("height from ",(j*striderY),"to row",(j*striderY)+striderY, "col ",i*striderX,"to ",(i*striderX)+striderX)
            Pred=RawPrediction[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]
            label=Lbl[j*striderY:(j*striderY)+striderY, i*striderX:(i*striderX)+striderX ]

            Pred[0][0]=-6000
            Pred[0][1]=-6000
            Pred[0][2]=-6000
            #Pred[0][3]=-6000
            
            #print(' Pred Shape: ', Pred.shape, ' --->', end=' ')
 
            Pred=torch.argmax(Pred[0], dim=0)

            #label=(label/10)
            label= torch.from_numpy(label).to(device)
            label=label==singleclass/10

            Pred=Pred==(singleclass/10)

            match=torch.eq(Pred,label)
            del Pred
            gc.collect()
            cuda.empty_cache()
            match=match.to(device)

            #label=label!=0
 
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


"""
def EvaluateTensors(tensor_list=[], activation_list[],classList=[10,50],tsholdList=[0.5],TileSizeList=[3200]):
    
    start=datetime.now()
    print ("Started at: ", start)
    

    #______BEGIN Verifications__________
    
    if len(tensor_list)==0:
        print("Tensor list is empty")
        return
    
    #______END Verifications__________
    
    result_titles=['Raster','Epoch','CheckPoint','Tile Size','Threshold','TruePositives','FalsePositives','FalseNegatives','TrueNegatives', 'Precision', 'Recall', 'F1','Acc','FP-FN Ratio']
    result_DB=pd.DataFrame( columns=result_titles)
    
    s=log_DB.head(1)  # [log_DB['Epoch'] == 0]
    s=s['Session'][0]
    
    
    for i in range(len(tensor_list)):  #repeats for rasters and saved tensors
        dataset=os.path.basename(os.path.basename(raster_list[i]))
        dataset=dataset[0:-4]
        for size in TileSizeList:
            image, Lbl, blank, xMoves, yMoves, height, width,newHeight,newWidth=prepareImagesAllClasses(raster_list[i],label_list[i], size)
        
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
                    netname=Netname(Net)
            
            
                striderX=size
                striderY=size
                prediction=PredictAllClasses(Net, image, blank, xMoves, yMoves, striderX, striderY, height, width,newHeight,newWidth)
                
                t=datetime.now()
                DateTime =str(t.hour)+str(t.minute)+"-"+str(t.day)+"-"+str(t.month)+"-"+str(t.year)
            
                for cls in classList:
                    Pred=prediction[int(cls/10)] #pick the prediction layer to work with

            
                    for th in tsholdList:
                        cm,TP,FP,FN,TN=ConfusionMatrixFromPerdiction(Pred, Lbl,xMoves, yMoves, striderX, striderY,height, width,newHeight,newWidth,cls,tilesize=3200, threshold=th)
                        cv2.imwrite("./"+s+"/"+Netname(Net)+'CONFUSIONmap-Epoch'+str(e)+"-"+dataset+'-Class '+str(cls)+'-TH'+str(th*100)+'-'+DateTime +'.png', cm )
                        del cm
                        gc.collect()
                        torch.cuda.empty_cache()
                
                        p=Precision(TP,FP )
                        r=Recall(TP,FN)
                        f_one = Fone(TP,FP,FN)
                        acc= (TP+TN)/(TP+FP+FN+TN)
                        fpfnRatio= (FP + 1)/(FN + 1)
                        new_entry=pd.DataFrame([[i,e,c,size,th,TP,FP,FN,TN,p,r,f_one,acc,fpfnRatio]], columns=result_titles)
                        result_DB=pd.concat([result_DB, new_entry])
                        result_DB.to_csv("./"+s+"/Evaluation Results "+netname+'-'+DateTime+".csv", sep=",")

                    heatmap=HeatMapFromPrediction(Pred, xMoves, yMoves, striderX, striderY,height, width,newHeight,newWidth)
                    cv2.imwrite("./"+s+"/"+Netname(Net)+'HEATmap-Epoch'+str(e)+"-"+dataset+'-Class '+str(cls)+'-'+DateTime +'.png', heatmap )
                    del heatmap
                    gc.collect()
                    torch.cuda.empty_cache()
                
                del prediction
                del Net
                gc.collect()
                cuda.empty_cache()
            
        del image
        del Lbl
        del blank 
        gc.collect()
        cuda.empty_cache()
    
    end=datetime.now()
    print ("Finished at: ", end)
    duration=end-start
    print ("Duration: ",  duration.minutes, "   (in seconds): ", duration.seconds)
    
"""

