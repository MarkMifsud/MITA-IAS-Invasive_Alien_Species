import cv2
import numpy as np
import copy
import gc
import os


folderexists=False
while folderexists==False:
    folder = input("Enter the name of the exported folder:")
    if not os.path.exists(".\\" +folder+"\\"):
        print(' no folder named '+folder+' was found in /Data. Please check the folder name and location, then try again')
    else:
        folderexists = True
        print('Joining images in '+folder)
        break


images_or_lables = 'images'



overlapH=0
overlapW=0

path=".\\" + folder + "\\" + images_or_lables + "\\000000000000.tif"
image1 = cv2.imread(path, cv2.IMREAD_UNCHANGED)

imageHeight=image1.shape[0]
imageWidth=image1.shape[1]

bottom_row0=image1[imageHeight-1,:]
right_column0=image1[:,imageWidth-1]
#del image1
gc.collect()

path=".\\"+folder+"\\"+images_or_lables+"\\000000000002.tif"
image2 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
left_coloumn2=image2[:,0]
if (right_column0==image2[:,0]).all(): overlapW=1
#del image2
gc.collect()

path=".\\"+folder+"\\"+images_or_lables+"\\000000000001.tif"
image3 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
if (bottom_row0==image3[0,:]).all() : overlapH=1
#del image3
gc.collect()

goalHeight=(imageHeight*2)-overlapH
goalWidth=(imageWidth*2)-overlapW

print( goalHeight, " ",goalWidth )


images_or_lables='images'
result=np.zeros((goalHeight,goalWidth,4), dtype=np.uint8)

path=".\\"+folder+"\\"+images_or_lables+"\\000000000000.tif"
image1 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
result[0:imageHeight, 0:imageWidth ]=copy.deepcopy(image1)
del image1
gc.collect()

path=".\\"+folder+"\\"+images_or_lables+"\\000000000002.tif"
image2 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
result[0:imageHeight, imageWidth-overlapW:imageWidth*2]=copy.deepcopy(image2)
del image2
gc.collect()


path=".\\"+folder+"\\"+images_or_lables+"\\000000000001.tif"
image3 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
result[imageHeight-overlapH:imageHeight*2, 0:imageWidth]=copy.deepcopy(image3)
del image3
gc.collect()

path=".\\"+folder+"\\"+images_or_lables+"\\000000000003.tif"
image4 = cv2.imread(path, cv2.IMREAD_UNCHANGED)

result[imageHeight-overlapH:imageHeight*2, imageWidth-overlapW:imageWidth*2]=copy.deepcopy(image4)
del image4
gc.collect()

saved = cv2.imwrite('./source/rasters/'+folder+'.tif',result)
if saved: print('Stiched raster saved as:  Data/source/rasters/'+folder+'.tif')
saved = False

from PIL import Image
import matplotlib.pyplot as plt

labels_exist = False
for i in range(0, 4):
    path = ".\\" + folder + "\\labels\\00000000000" + str(i) + ".tif"
    if os.path.exists(path):
        labels_exist = True

if labels_exist:

    images_or_lables = 'labels'
    result = np.zeros((goalHeight, goalWidth), dtype=np.uint8)

    path = ".\\" + folder + "\\" + images_or_lables + "\\000000000000.tif"
    if os.path.exists(path):
        image1 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        result[0:imageHeight, 0:imageWidth] = copy.deepcopy(image1)
        del image1
        gc.collect()

    path = ".\\" + folder + "\\" + images_or_lables + "\\000000000002.tif"
    if os.path.exists(path):
        image2 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        result[0:imageHeight, imageWidth - overlapW:imageWidth * 2] = copy.deepcopy(image2)
        del image2
        gc.collect()

    path = ".\\" + folder + "\\" + images_or_lables + "\\000000000001.tif"
    if os.path.exists(path):
        image3 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        result[imageHeight - overlapH:imageHeight * 2, 0:imageWidth] = copy.deepcopy(image3)
        del image3
        gc.collect()

    path = ".\\" + folder + "\\" + images_or_lables + "\\000000000003.tif"
    if os.path.exists(path):
        image4 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        result[imageHeight - overlapH:imageHeight * 2, imageWidth - overlapW:imageWidth * 2] = copy.deepcopy(image4)
        del image4
        gc.collect()

    saved = cv2.imwrite('./source/' + images_or_lables + '/' + folder + '.tif', result)
    if saved: print('Stiched label saved as:  Data/source/labels/' + folder + '.tif')
    saved = False
    plt.imshow(result)

else:
    print("no label tiles found")

print("TILE STITCHING COMPLETE")