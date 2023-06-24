# MITA-IAS-Invasive_Alien_Species
Attempting to detect Invasive plant species from drone images for monitoring and controlling them. A collaboration between Ambjent Malta and University of Malta, funded by MITA.


|   |   |
|---|---|
|**Folder Name**|**Description and Contents**|
|IAS|Contains all files associated with the project, like:<br><br>·       Jupyter Notebook files to run the programs (.ipynb files)<br><br>·       Files containing necessary Python code (.py files)<br><br>·       Subfolders like Data, Models, Results…|
|IAS/Models|This is where the trained Deep-Learning models are to be stored.<br><br>Each subfolder in here contains saved checkpoints (trained models) for a specific trained model (.torch files) as well as some other files that are produced as a result of training (.optim .sched .csv…)<br><br>When using the project, at least one of the checkpoints stored here will be used as the model that performs the detection of species, and will have to be referenced.|
|IAS/Models  <br>/_usable|This folder is for model checkpoints that are deemed good to use for species detection|
|IAS/Data|This is where data that the AI models use is stored.  These consist mostly of rasters and labels in .tif format that were exported from ARCGIS ortho-mosaics and shapefiles.<br><br>This is also where the programs expect files to be stored by default.<br><br>NOTE:  The AI models only work with pixel data so anything placed here for the AI model to use must be exported in .tif format first.<br><br>In general, this subfolder contains:<br><br>·       Subfolders containing image data the models work on<br><br>·       Some files used to analyse or manipulate the various image data contained herein|
|IAS/Data  <br>/source|This is where full size rasters and corresponding labels are to be stored.<br><br>**_IAS/Data/source/rasters_** contains the full colour, raster image of the location as captured by the drone.<br><br>**_IAS/Data/source/labels_** contains the corresponding label map for each full colour raster.  The label map identifies species and where they are present in the location. These label maps are usually understood to be labelled manually by a person.<br><br>Example:   **_IAS/Data/source/rasters/Ar1.tif_** is a colour raster and has a corresponding grayscale bitmap at **_IAS/Data/source/labels/Ar1.tif_**  to map the labelled species contained in the location.<br><br>If the label is available, this raster can be used to:<br><br>1)    Train new models<br><br>2)   Compare the detection performed by a model against the labelling performed by a person|
|IAS/Data  <br>/traindata|This subfolder contains images and labels in a format used to train models.<br><br>These consist of 256 by 256 sized tiles (other sizes are possible) that can be derived from the larger images inside **_IAS/Data/source_** .|
|IAS/Results|When models perform detection, resulting images are saved in this location|

Full instructions on how to use the system are in   ***IAS/UserManual/***  for non technical people. 


