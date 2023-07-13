from datetime import datetime
import os
from pathlib import Path
import numpy as np
import copy
import cv2
from tqdm import tqdm
import gc
from torch import cuda
import pandas as pd
import random

import torch
import torchvision.transforms as tf
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics.functional import accuracy as acc

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

plottable_data = None
height = 256
width = 256
Net = None
batch_size = 1
optimizer = None
scheduler = None
logFilePath = None
seed = 41
train_grayscale = False

criterion = None

singleclass = None  # a single class a model is trained on specifically.  All other values in a label are omitted.
excludeClasses = []


def set_seed(s=41):
    global seed
    seed = s
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(s)
    # random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)


tensorise = tf.ToTensor()


def AdaptMask(Lbl):  # function to adapt mask to Tensor
    global singleclass
    global excludeClasses
    Lbl = Lbl.astype(np.float32)

    if singleclass == None:
        Lbl = Lbl / 10
    else:
        Lbl = (Lbl == singleclass)

    Lbl = Lbl.astype(int)
    Lbl = tensorise(Lbl)
    return Lbl


transformImg = tf.Compose([tf.ToPILImage(), tf.ToTensor(),
                           tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # function to adapt image
# Normalize parameters are suggested by PyTorch documentation

transformGrayscaleImg = tf.Compose([tf.ToPILImage(), tf.ToTensor()])  # function to adapt image


def ReadRandomImage(folder):  # Used if training on randomly selected images
    ListOfImages = os.listdir(os.path.join(folder, "images"))  # Create list of images
    idx = np.random.randint(0, len(ListImages))  # Select random image
    Img = cv2.imread(os.path.join(folder, "images", ListOfImages[idx]), cv2.IMREAD_COLOR)[:, :, 0:3]
    Img = transformImg(Img)

    if Path(os.path.join(folder, "labels", ListOfImages[idx])).is_file():
        Lbl = cv2.imread(os.path.join(folder, "labels", ListOfImages[idx]), cv2.COLOR_GRAY2BGR)
        Lbl = AdaptMask(Lbl)
    else:
        Lbl = torch.zeros(width, height, dtype=torch.int32)

    return Img, Lbl


def LoadBatch(folder):  # Load batch of Random images
    global batch_size
    images = torch.zeros([batch_size, 3, height, width])
    labels = torch.zeros([batch_size, height, width])
    for i in range(batch_size):
        images[i], labels[i] = ReadRandomImage(folder)

    return images, labels


def LoadNext(batchNum, batchSize, folder):
    # global batch_size
    global height
    global width
    global train_grayscale

    ListOfImages = os.listdir(os.path.join(folder, "images"))

    if train_grayscale:
        images = torch.zeros([batchSize, 1, height, width])
    else:
        images = torch.zeros([batchSize, 3, height, width])
    labels = torch.zeros([batchSize, height, width])

    for item in range(batchSize):
        idx = (batchNum * batchSize) + item
        # print ("idx:",idx, "  path:", os.path.join(folder, "labels", ListOfImages[idx]) )
        if train_grayscale:
            # Img=cv2.cvtColor(cv2.imread(os.path.join(folder, "images", ListOfImages[idx]),  cv2.COLOR_GRAY2BGR ), cv2.COLOR_BGR2GRAY)
            Img = cv2.imread(os.path.join(folder, "images", ListOfImages[idx]), cv2.COLOR_BGR2GRAY)
            Img = transformGrayscaleImg(Img)
        else:
            Img = cv2.imread(os.path.join(folder, "images", ListOfImages[idx]), cv2.IMREAD_COLOR)[:, :, 0:3]
            Img = transformImg(Img)

        # now we check if the label exists.  We read it ELSE generate blank tensor
        if Path(os.path.join(folder, "labels", ListOfImages[idx])).is_file():
            Lbl = cv2.imread(os.path.join(folder, "labels", ListOfImages[idx]), cv2.COLOR_GRAY2BGR)  # [:,:,0:3]
            Lbl = AdaptMask(Lbl)
        else:
            Lbl = torch.zeros(width, height, dtype=torch.int32)

        images[item] = Img
        labels[item] = Lbl

    return images, labels


def learn(imgs, lbls):
    global Net
    global optimizer
    global criterion

    images = torch.autograd.Variable(imgs, requires_grad=False).to(device)  # Load image
    labels = torch.autograd.Variable(lbls, requires_grad=False).to(device)  # Load labels
    Pred = Net(images)  # ['out'] # make prediction
    Net.zero_grad()
    if criterion is None:
        criterion = smp.losses.DiceLoss('multiclass', log_loss=False, from_logits=True, smooth=0.0, ignore_index=None,
                                        eps=1e-07)
    # criterion = torch.nn.CrossEntropyLoss() # Set loss function
    Loss = criterion(Pred, labels.long())  # Calculate cross entropy loss
    Loss.backward()  # Backpropogate loss
    this_loss = copy.deepcopy(Loss.data).cpu().detach().numpy()  # this_loss=Loss.data.cpu().numpy()
    optimizer.step()  # not used see if necessary
    return this_loss


def validate(imgs, lbls, vbatch_size):
    global Net
    global criterion

    images = torch.autograd.Variable(imgs, requires_grad=False).to(device)  # Load image
    labels = torch.autograd.Variable(lbls, requires_grad=False).to(device)  # Load labels
    Pred = Net(images)  # ['out'] # make prediction
    Net.zero_grad()
    if criterion == None:
        criterion = smp.losses.DiceLoss('multiclass', log_loss=False, from_logits=True, smooth=0.0, ignore_index=None,
                                        eps=1e-07)
    Loss = criterion(Pred, labels.long())  # Calculate cross entropy loss
    this_loss = copy.deepcopy(Loss.data).cpu().detach().numpy()

    tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(Pred, dim=1), labels.long(), mode='multiclass',
                                           num_classes=Pred.shape[1])
    accuracy = acc(tp, fp, fn, tn, reduction="micro")

    return this_loss, float(accuracy)


def validateOneClass(imgs, lbls, vbatch_size, ValidateClass=1):  # DEPRECATED
    global Net

    images = torch.autograd.Variable(imgs, requires_grad=False).to(device)  # Load image
    labels = torch.autograd.Variable(lbls, requires_grad=False).to(device)  # Load labels
    Pred = Net(images)  # ['out'] # make prediction
    Net.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()  # Set loss function
    Loss = criterion(Pred, labels.long())  # Calculate cross entropy loss
    this_loss = copy.deepcopy(Loss.data).cpu().detach().numpy()

    seg = torch.argmax(Pred[0:vbatch_size], dim=1)
    seg = seg == ValidateClass  # class 1 only  everything else is zero
    lbls = lbls.to(device)
    lbls = lbls == ValidateClass

    tp, fp, fn, tn = smp.metrics.get_stats(seg, lbls.long().to(device), mode='binary', threshold=0.5)
    accuracy = acc(tp, fp, fn, tn, reduction="macro")
    # match=torch.logical_and(seg,lbls)
    # match=torch.count_nonzero(match)
    # accuracy=int(match)/(height*width)

    return this_loss, float(accuracy)


def Netname(Net):  # Genrates string based on Model and backbone for naming files
    string = str(Net)[0:50]
    model_name = ""
    backbone = ""
    writemode = True
    writemode2 = False

    for c in string:
        if writemode2 == True and c == '(': break
        if c == '(':
            writemode = False

        if writemode: model_name = model_name + c
        if c == ':': writemode2 = True
        if writemode2: backbone = backbone + c

    # backbone= backbone[2:11]
    return model_name + '-' + backbone[2:11]


def LoadNet(model, checkpoint):
    global Net
    Net = model  # Load net
    Net = Net.to(device)
    Net.load_state_dict(torch.load(checkpoint))
    return Net


def trainStart(model, TrainFolder, ValidFolder, epochs, batchSize, TrainOnClass=10, TestFolder=None, Learning_Rate=1e-5,
               logName=None, SchedulerName='Plateau'):
    # log will not save test loss in this version

    global singleclass
    singleclass = TrainOnClass

    global batch_size
    batch_size = batchSize
    vbatch_size = batchSize * 4
    ListImages = os.listdir(os.path.join(TrainFolder, "images"))  # Create list of images        
    vListImages = os.listdir(os.path.join(ValidFolder, "images"))  # Create list of validation images
    unbatched = len(ListImages) % batch_size
    batch_counts = round((len(ListImages) - unbatched) / batch_size)
    vunbatched = len(vListImages) % vbatch_size
    vbatch_counts = round((len(vListImages) - vunbatched) / vbatch_size)

    if TestFolder != None:
        tbatch_size = batchSize * 4
        tListImages = os.listdir(os.path.join(TestFolder, "images"))  # Create list of test images
        tunbatched = len(tListImages) % tbatch_size
        tbatch_counts = round((len(tListImages) - tunbatched) / tbatch_size)
    else:
        tbatch_counts = 0

    global Net
    global scheduler
    global optimizer
    # load model 
    Net = model  # Load net
    Net = Net.to(device)
    model_naming_title = Netname(Net)
    optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)  # Create adam optimizer
    if SchedulerName == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, batch_counts)
    elif SchedulerName == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    else:
        print("SchedulerName parameter must be Cosine or Plateau. " + SchedulerName + " is not recognised")
        return

    # autodetect height and width of training tiles
    global height
    global width
    tempImage = cv2.imread(os.path.join(TrainFolder, "images", ListImages[0]), cv2.IMREAD_COLOR)
    height = tempImage.shape[0]
    width = tempImage.shape[1]
    del tempImage

    global logFilePath
    global plottable_data

    t = datetime.now()
    DateTime = str(t.hour) + str(t.minute) + "-" + str(t.day) + "-" + str(t.month) + "-" + str(t.year)
    path = 'Models/' + model_naming_title + "-" + DateTime
    del t

    if not os.path.exists(path): os.makedirs(path)

    if logName == None:
        log_path = 'LOG for ' + model_naming_title + '.csv'
    else:
        log_path = logName

    if not os.path.exists(log_path):
        log_path2 = "./" + path + '/LOG for ' + model_naming_title + "-" + DateTime + '.csv'
        log_titles = ['Epoch', 'Train-Loss', 'Val-Loss', 'Val-Acc', 'Test-Loss', 'Test-Acc', 'Time', 'Learn-Rate',
                      'Session', 'CheckPoint']
        log_DB = pd.DataFrame(columns=log_titles)
        log_DB2 = pd.DataFrame(columns=log_titles)
        model_naming_title = "-" + model_naming_title + ".torch"

        print(" Folder for checkpoints:'", path, "' was created")
        print(" Training Log File:'", log_path, "' will be created... Starting training from scratch")
        # Learning_Rate
        best_loss = 1
        EpochsStartFrom = 0  # in case training is restarted from a previously saved epoch, this continues the sequence
    # and prevents over-writing models and logs in the loss database 


    else:
        print("Log file for ", model_naming_title, " already present as: ", log_path)
        print("Training will not start, to prevent overwriting")
        print(" ")
        print("If you really want to start from scratch, please move the existent log file")
        print("If you want to continue training please use the trainfromLast function instead")
        return

    del DateTime
    trainingDetails(path, TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder), tbatch_counts, '0',
                    epochs - 1, finished=False)
    # _________________________PREPERATION DONE_________________#
    # _________________________TRAINING LOOP STARTS FROM HERE_________________#
    with torch.autograd.set_detect_anomaly(False):
        gc.collect()
        cuda.empty_cache()

        for itr in range(epochs):  # Training loop
            start_time = datetime.now()
            train_loss = 0
            runs = batch_counts
            vruns = vbatch_counts

            Net.train()
            for batchNum in tqdm(range(batch_counts)):
                images, labels = LoadNext(batchNum, batch_size, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            if unbatched > 0:
                images, labels = LoadNext(batch_counts + 1, unbatched, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                runs = batch_counts + 1
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            # uncomment if you want to train on a random batch too
            # images,labels=LoadBatch(TrainFolder) 
            # train_loss+=learn(images, labels)
            # runs=batch_counts+1

            train_loss = train_loss / (runs)  # +1) #averages the loss on all batches
            if SchedulerName == 'Cosine':
                scheduler.step()
            else:
                scheduler.step(train_loss)

            # BEGIN Validation 
            Net.eval()
            with torch.no_grad():
                valid_loss = 0
                ValACC = 0

                for vbatchNum in tqdm(range(vbatch_counts)):
                    images, labels = LoadNext(vbatchNum, vbatch_size, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy

                if vunbatched > 0:
                    images, labels = LoadNext(vbatch_counts + 1, vunbatched, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy
                    vruns = vbatch_counts + 1

                valid_loss = valid_loss / (vruns)  # averages the loss on all batches
                ValACC = ValACC / (vruns)
                # END   Validation

                test_loss = 0
                testACC = 0
                if TestFolder != None:
                    truns = tbatch_counts
                    for tbatchNum in tqdm(range(tbatch_counts)):
                        images, labels = LoadNext(tbatchNum, tbatch_size, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, tbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy

                    if tunbatched > 0:
                        images, labels = LoadNext(tbatch_counts + 1, tunbatched, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, tbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy
                        truns = tbatch_counts + 1

                    test_loss = test_loss / (truns)  # averages the loss on all batches
                    testACC = testACC / (truns)
                # END Test                

            duration = datetime.now() - start_time

            if train_loss <= best_loss:
                best_loss = copy.deepcopy(train_loss)

            t = datetime.now()
            checkpoint = path + "/" + str(itr + EpochsStartFrom) + "-" + str(t.hour) + str(
                t.minute) + model_naming_title
            print("Saving Model: ", "./" + checkpoint)
            torch.save(Net.state_dict(), "./" + checkpoint)

            """
            else:
                if itr!=epochs-1:  checkpoint="not saved"
                else:
                    checkpoint=path+"/"+str(itr+EpochsStartFrom)+"-LAST EPOCH-"+model_naming_title
                    torch.save(Net.state_dict(),"./"+checkpoint)
                    print(" Saving LAST EPOCH: ./",checkpoint)
            """
            torch.save(optimizer.state_dict(), path + "/optimiser.optim")
            torch.save(scheduler, path + "/scheduler.sched")

            print(itr + EpochsStartFrom, "=> TrainLoss=", train_loss, "  ValLoss=", valid_loss, "  valACC=", ValACC,
                  " TestLoss=", test_loss, " testACC=", testACC, " lr:", GetLastLR(SchedulerName), " Time:",
                  duration.seconds)
            new_log_entry = pd.DataFrame([[itr + EpochsStartFrom, train_loss, valid_loss, ValACC, test_loss, testACC,
                                           duration.seconds, GetLastLR(SchedulerName), path, checkpoint]],
                                         columns=log_titles)
            log_DB = pd.concat([log_DB, new_log_entry])
            log_DB.to_csv(log_path, sep=",")
            log_DB2 = pd.concat([log_DB2, new_log_entry])
            log_DB2.to_csv(log_path2, sep=",")

    # _________________________TRAINING LOOP ENDS HERE_________________#
    trainingDetails(path, TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder), tbatch_counts, '0',
                    checkpoint, finished=True)
    plottable_data = log_DB
    print("____FINISHED Training______")
    return log_DB


def trainFromLast(model, TrainFolder, ValidFolder, epochs, batchSize, TrainOnClass=10, TestFolder=None, Learning_Rate=0,
                  SchedulerName='Plateau'):
    global singleclass
    singleclass = TrainOnClass

    global Net
    global scheduler
    global optimizer
    Net = model  # Load net
    Net = Net.to(device)
    gc.collect()
    cuda.empty_cache()

    global batch_size
    batch_size = batchSize
    vbatch_size = batchSize * 4
    ListImages = os.listdir(os.path.join(TrainFolder, "images"))  # Create list of images        
    vListImages = os.listdir(os.path.join(ValidFolder, "images"))  # Create list of validation images
    unbatched = len(ListImages) % batch_size
    batch_counts = round((len(ListImages) - unbatched) / batch_size)
    vunbatched = len(vListImages) % vbatch_size
    vbatch_counts = round((len(vListImages) - vunbatched) / vbatch_size)

    if TestFolder != None:
        tbatch_size = batchSize * 4
        tListImages = os.listdir(os.path.join(TestFolder, "images"))  # Create list of test images
        tunbatched = len(tListImages) % tbatch_size
        tbatch_counts = round((len(tListImages) - tunbatched) / tbatch_size)
    else:
        tbatch_counts = 0

    global height
    global width
    # autodetect height and width of training tiles
    tempImage = cv2.imread(os.path.join(TrainFolder, "images", ListImages[0]), cv2.IMREAD_COLOR)
    height = tempImage.shape[0]
    width = tempImage.shape[1]
    del tempImage

    global plottable_data
    global logFilePath
    model_naming_title = Netname(Net)
    log_path = 'LOG for ' + model_naming_title + '.csv'
    logFilePath = log_path
    log_titles = ['Epoch', 'Train-Loss', 'Val-Loss', 'Val-Acc', 'Test-Loss', 'Test-Acc', 'Time', 'Learn-Rate',
                  'Session', 'CheckPoint']
    log_DB = pd.DataFrame(columns=log_titles)

    if os.path.exists(log_path):
        print("A log file for ", model_naming_title, " was found as: ", log_path)
        log_DB = pd.read_csv(log_path, sep=",", index_col=0)
        path = log_DB.tail(1)['Session']
        path = str(path[0])

        log_path2 = "./" + path + '/LOG for ' + path[7:] + '.csv'

        best_loss = log_DB['Train-Loss'].min()  # smallest loss value
        LastEpoch = int(log_DB.tail(1)['Epoch'])
        LastCheckpoint = log_DB.tail(1)['CheckPoint']
        if Learning_Rate == 0: Learning_Rate = float(log_DB.tail(1)['Learn-Rate'])  # the last learning rate logged
        EpochsStartFrom = LastEpoch + 1

        if os.path.exists(path):
            print("Folder for checkpoints: ", path, " was found")
        elif os.path.exists('Models/' + path):
            path = 'Models/' + path
            print("Folder for checkpoints: ", path, " was found")
        else:
            print("Folder for checkpoints: ", path, " or Models/" + path + " were not found.  Training cannot continue")
            del epochs
            print(
                " Please restore the folder and restart this notebook, or start training from scratch by manually deleting the older log")
            return

        if os.path.exists("./" + LastCheckpoint[0]):
            checkpoint = "./" + LastCheckpoint[0]  # Path to trained model
            print("Training to continue from checkpoint:", checkpoint)
        elif os.path.exists("./Models/" + LastCheckpoint[0]):
            checkpoint = "./Models/" + LastCheckpoint[0]  # Path to trained model
            print("Training to continue from checkpoint:", checkpoint)
        else:
            print("Last Checkpoint was found at NEITHER ./" + LastCheckpoint[0] + " NOR at ./Models/" + LastCheckpoint[
                0] + "  .  Training cannot continue")
            # print(" Please specify a path to a saved checkpoint manually in the next cell")
            return

    else:
        print(" Training Log File:  '", log_path, "'  was not found...  ")
        print(
            " Either restore the log file, pass a different model_naming_title to trainFromLast() , or start training from scratch with trainStart() ")
        return
    # _________________________PREPERATION DONE_________________#

    Net.load_state_dict(torch.load(checkpoint))  # if this gives an error check the training setup in previous 2 cells

    optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)

    if os.path.exists('./' + path + "/optimiser.optim"):
        optimizer.load_state_dict(torch.load('./' + path + "/optimiser.optim"))

    if SchedulerName == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, batch_counts)
    elif SchedulerName == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    else:
        print("SchedulerName parameter must be Cosine or Plateau. " + SchedulerName + " is not recognised")
        return

    if os.path.exists('./' + path + "/scheduler.sched"):  scheduler = torch.load('./' + path + "/scheduler.sched")

    model_naming_title = "-" + model_naming_title + ".torch"
    # log_path2="./"+path+'/LOG for '+path+'.csv'
    log_DB2 = pd.read_csv(log_path2, sep=",", index_col=0)

    trainingDetails(path, TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder), tbatch_counts,
                    checkpoint, EpochsStartFrom + epochs, finished=False)
    # _________________________TRAINING STARTS FROM HERE_________________#

    with torch.autograd.set_detect_anomaly(False):
        for itr in range(epochs):  # Training loop
            start_time = datetime.now()
            train_loss = 0
            runs = batch_counts
            vruns = vbatch_counts

            Net.train()
            for batchNum in tqdm(range(batch_counts)):
                images, labels = LoadNext(batchNum, batch_size, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            if unbatched > 0:
                images, labels = LoadNext(batch_counts + 1, unbatched, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                runs = batch_counts + 1
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            # uncomment if you want to train on a random batch too
            """
            images,labels=LoadBatch(TrainFloder) 
            train_loss+=learn(images, labels)
            runs=batch_counts+1
            """

            train_loss = train_loss / (runs)  # +1) #averages the loss on all batches
            if SchedulerName == 'Cosine':
                scheduler.step()
            else:
                scheduler.step(train_loss)
            print(itr + EpochsStartFrom, "=> TrainLoss=", train_loss)
            # BEGIN Validation 

            Net.eval()
            with torch.no_grad():
                valid_loss = 0
                ValACC = 0

                for vbatchNum in tqdm(range(vbatch_counts)):
                    images, labels = LoadNext(vbatchNum, vbatch_size, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy

                if vunbatched > 0:
                    images, labels = LoadNext(vbatch_counts + 1, vunbatched, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy
                    vruns = vbatch_counts + 1

                valid_loss = valid_loss / (vruns)  # averages the loss on all batches
                ValACC = ValACC / (vruns)
                print(itr + EpochsStartFrom, "=> ValLoss=", valid_loss, " valACC=", ValACC)
                # END   Validation

                test_loss = 0
                testACC = 0
                if TestFolder != None:
                    truns = tbatch_counts
                    for tbatchNum in tqdm(range(tbatch_counts)):
                        images, labels = LoadNext(tbatchNum, tbatch_size, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, tbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy

                    if tunbatched > 0:
                        images, labels = LoadNext(tbatch_counts + 1, tunbatched, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, tbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy
                        truns = tbatch_counts + 1

                    test_loss = test_loss / (truns)  # averages the loss on all batches
                    testACC = testACC / (truns)
                    print(itr + EpochsStartFrom, "=>  TestLoss=", test_loss, " testACC=", testACC)
                # END Test                

            duration = datetime.now() - start_time

            if train_loss <= best_loss:
                best_loss = copy.deepcopy(train_loss)

            t = datetime.now()
            checkpoint = path + "/" + str(itr + EpochsStartFrom) + "-" + str(t.hour) + str(
                t.minute) + model_naming_title
            print("Saving Model: ", "./" + checkpoint, end="")
            torch.save(Net.state_dict(), "./" + checkpoint)

            """
            else:
                if itr!=epochs-1:  checkpoint="not saved"
                else:
                    checkpoint=path+"/"+str(itr+EpochsStartFrom)+"-LAST EPOCH-"+model_naming_title
                    torch.save(Net.state_dict(),"./"+checkpoint)
                    print(" Saving LAST EPOCH: ./",checkpoint)
            """
            print("lr:", GetLastLR(SchedulerName), "  Time:", duration.seconds)
            # new_log_entry=pd.DataFrame([[itr+EpochsStartFrom, train_loss, valid_loss,ValACC, float(scheduler.state_dict()["_last_lr"][0]),path,checkpoint]], columns=log_titles)
            new_log_entry = pd.DataFrame([[itr + EpochsStartFrom, train_loss, valid_loss, ValACC, test_loss, testACC,
                                           duration.seconds, GetLastLR(SchedulerName), path, checkpoint]],
                                         columns=log_titles)
            log_DB = pd.concat([log_DB, new_log_entry])
            log_DB.to_csv(log_path, sep=",")
            log_DB2 = pd.concat([log_DB2, new_log_entry])
            log_DB2.to_csv(log_path2, sep=",")

    # _________________________TRAINING LOOP ENDS HERE_________________#
    trainingDetails(path, TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder), tbatch_counts,
                    str(EpochsStartFrom), checkpoint, finished=True)
    plottable_data = log_DB
    print("____FINISHED Training______")
    return log_DB


def LoadNextRandomBatch(BatchOfImages, folder):
    # global batch_size
    batchSize = len(BatchOfImages)
    global height
    global width
    global train_grayscale

    if train_grayscale:
        images = torch.zeros([batchSize, 1, height, width])
    else:
        images = torch.zeros([batchSize, 3, height, width])
    labels = torch.zeros([batchSize, height, width])

    item = 0
    for image in BatchOfImages:
        # print (" path:", os.path.join(folder, "images", image ) )
        if train_grayscale:
            Img = cv2.imread(os.path.join(folder, "images", image), cv2.COLOR_BGR2GRAY)
            Img = transformGrayscaleImg(Img)
        else:
            Img = cv2.imread(os.path.join(folder, "images", image), cv2.IMREAD_COLOR)[:, :, 0:3]
            Img = transformImg(Img)

        # now we check if the label exists.  We read it ELSE generate blank tensor
        if Path(os.path.join(folder, "labels", image)).is_file():
            Lbl = cv2.imread(os.path.join(folder, "labels", image), cv2.COLOR_GRAY2BGR)
            Lbl = AdaptMask(Lbl)
        else:
            Lbl = torch.zeros(width, height, dtype=torch.int32)

        images[item] = Img
        labels[item] = Lbl
        item = item + 1

    return images, labels


def trainStartMC(model, TrainFolder, ValidFolder, epochs, batchSize, TestFolder=None, Learning_Rate=1e-5, logName=None,
                 SchedulerName='Plateau', Scheduler_Patience=3, percentagOfUnlabelledTiles=0.3):
    if percentagOfUnlabelledTiles > 1 or percentagOfUnlabelledTiles < 0:
        print("percentagOfUnlabelledTiles must be between 0 and 1.  Training cannot start.")
        return

    global batch_size
    batch_size = batchSize
    ListImages = os.listdir(os.path.join(TrainFolder, "images"))  # Create list of images
    ListOfLabelledImages = os.listdir(os.path.join(TrainFolder, "labels"))
    UnlabelledImages = copy.deepcopy(ListImages)
    for item in ListOfLabelledImages:
        UnlabelledImages.remove(item)

    amountInSample = int(len(UnlabelledImages) * percentagOfUnlabelledTiles)
    unbatched = (len(ListOfLabelledImages) + amountInSample) % batch_size
    batch_counts = round((len(ListOfLabelledImages) + amountInSample - unbatched) / batch_size)

    vListImages = os.listdir(os.path.join(ValidFolder, "images"))  # Create list of validation images
    if len(vListImages) < (batchSize * 4):
        vbatch_size = batchSize * 4
    else:
        vbatch_size = batchSize
    vunbatched = len(vListImages) % vbatch_size
    vbatch_counts = round((len(vListImages) - vunbatched) / vbatch_size)

    if TestFolder != None:
        tListImages = os.listdir(os.path.join(TestFolder, "images"))  # Create list of test images
        if len(tListImages) < (batchSize * 4):
            tbatch_size = batchSize * 4
        else:
            tbatch_size = batchSize
        tunbatched = len(tListImages) % tbatch_size
        tbatch_counts = round((len(tListImages) - tunbatched) / tbatch_size)
    else:
        tbatch_counts = 0

    global Net
    global scheduler
    global optimizer
    # load model 
    Net = model  # Load net
    Net = Net.to(device)
    model_naming_title = Netname(Net)
    optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)  # Create adam optimizer
    if SchedulerName == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, batch_counts)
    elif SchedulerName == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=Scheduler_Patience)
    else:
        print("SchedulerName parameter must be Cosine or Plateau. " + SchedulerName + " is not recognised")
        return

    global height
    global width
    # autodetect width and height of tiles
    tempImage = cv2.imread(os.path.join(TrainFolder, "images", ListImages[0]), cv2.IMREAD_COLOR)
    height = tempImage.shape[0]
    width = tempImage.shape[1]
    del tempImage

    global logFilePath
    global plottable_data

    t = datetime.now()
    DateTime = str(t.hour) + str(t.minute) + "-" + str(t.day) + "-" + str(t.month) + "-" + str(t.year)
    path = 'Models/' + model_naming_title + "-" + DateTime
    del t

    if not os.path.exists(path): os.makedirs(path)

    if logName == None:
        log_path = 'LOG for MC-' + model_naming_title + '.csv'
    else:
        log_path = logName

    if not os.path.exists(log_path):
        log_path2 = "./" + path + '/LOG for MC-' + model_naming_title + "-" + DateTime + '.csv'
        log_titles = ['Epoch', 'Train-Loss', 'Val-Loss', 'Val-Acc', 'Test-Loss', 'Test-Acc', 'Time', 'Learn-Rate',
                      'Session', 'CheckPoint']
        log_DB = pd.DataFrame(columns=log_titles)
        log_DB2 = pd.DataFrame(columns=log_titles)
        model_naming_title = "-" + model_naming_title + ".torch"

        print(" Folder for checkpoints:'", path, "' was created")
        print(" Training Log File:'", log_path, "' will be created... Starting training from scratch")
        Learning_Rate
        best_loss = 1
        EpochsStartFrom = 0  # in case training is restarted from a previously saved epoch, this continues the sequence
    # and prevents over-writing models and logs in the loss database
    else:
        print("Log file for ", model_naming_title, " already present as: ", log_path)
        print("Training will not start, to prevent overwriting")
        print(" ")
        print("If you really want to start from scratch, please move the existent log file")
        print("If you want to continue training please use the trainfromLast function instead")
        return

    del DateTime
    trainingDetails(path, TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder), tbatch_counts, '0',
                    epochs - 1, finished=False)
    # _________________________PREPERATION DONE_________________#
    # _________________________TRAINING LOOP STARTS FROM HERE_________________#

    with torch.autograd.set_detect_anomaly(False):
        gc.collect()
        cuda.empty_cache()
        for itr in range(epochs):  # Training loop
            start_time = datetime.now()
            train_loss = 0
            runs = batch_counts
            vruns = vbatch_counts

            random.shuffle(UnlabelledImages)
            sampleOfUnlabelledImages = UnlabelledImages[0:amountInSample]
            TrainSample = copy.deepcopy(ListOfLabelledImages)
            TrainSample.extend(sampleOfUnlabelledImages)
            random.shuffle(TrainSample)

            Net.train()
            for count in tqdm(range(batch_counts)):
                BatchOfImages = TrainSample[(count * batch_size):(count * batch_size) + batch_size - 1]
                # print((count*batch_size)," ",(count*batch_size)+batch_size-1)
                images, labels = LoadNextRandomBatch(BatchOfImages, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            if unbatched > 0:
                BatchOfImages = TrainSample[(batch_counts * batch_size):(batch_counts * batch_size) + unbatched - 1]
                images, labels = LoadNextRandomBatch(BatchOfImages, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                runs = batch_counts + 1
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            train_loss = train_loss / (runs)  # +1) #averages the loss on all batches
            if SchedulerName == 'Cosine':
                scheduler.step()
            else:
                scheduler.step(train_loss)

            del TrainSample
            # BEGIN Validation 
            Net.eval()
            with torch.no_grad():
                valid_loss = 0
                ValACC = 0

                for vbatchNum in tqdm(range(vbatch_counts)):
                    images, labels = LoadNext(vbatchNum, vbatch_size, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy

                if vunbatched > 0:
                    images, labels = LoadNext(vbatch_counts + 1, vunbatched, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy
                    vruns = vbatch_counts + 1

                valid_loss = valid_loss / (vruns)  # averages the loss on all batches
                ValACC = ValACC / (vruns)
                # END   Validation

                test_loss = 0
                testACC = 0
                if TestFolder != None:
                    truns = tbatch_counts
                    for tbatchNum in tqdm(range(tbatch_counts)):
                        images, labels = LoadNext(tbatchNum, tbatch_size, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy

                    if tunbatched > 0:
                        images, labels = LoadNext(tbatch_counts + 1, tunbatched, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy
                        truns = tbatch_counts + 1

                    test_loss = test_loss / (truns)  # averages the loss on all batches
                    testACC = testACC / (truns)
                # END Test                

            duration = datetime.now() - start_time

            if train_loss <= best_loss:
                best_loss = copy.deepcopy(train_loss)

            t = datetime.now()
            checkpoint = path + "/" + str(itr + EpochsStartFrom) + "-" + str(t.hour) + str(
                t.minute) + model_naming_title
            print("Saving Model: ", "./" + checkpoint)
            torch.save(Net.state_dict(), "./" + checkpoint)

            torch.save(optimizer.state_dict(), path + "/optimiser.optim")
            torch.save(scheduler, path + "/scheduler.sched")

            """
            else:
                if itr!=epochs-1:  checkpoint="not saved"
                else:
                    checkpoint=path+"/"+str(itr+EpochsStartFrom)+"-LAST EPOCH-"+model_naming_title
                    torch.save(Net.state_dict(),"./"+checkpoint)
                    print(" Saving LAST EPOCH: ./",checkpoint)
            """
            print(itr + EpochsStartFrom, "=> TrainLoss=", train_loss, "  ValLoss=", valid_loss, "  valACC=", ValACC,
                  " TestLoss=", test_loss, " testACC=", testACC, " lr:", GetLastLR(SchedulerName), " Time:",
                  duration.seconds)
            new_log_entry = pd.DataFrame([[itr + EpochsStartFrom, train_loss, valid_loss, ValACC, test_loss, testACC,
                                           duration.seconds, GetLastLR(SchedulerName), path, checkpoint]],
                                         columns=log_titles)
            log_DB = pd.concat([log_DB, new_log_entry])
            log_DB.to_csv(log_path, sep=",")
            log_DB2 = pd.concat([log_DB2, new_log_entry])
            log_DB2.to_csv(log_path2, sep=",")

    # _________________________TRAINING LOOP ENDS HERE_________________#
    trainingDetails(path, TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder), tbatch_counts, '0',
                    checkpoint, finished=True)
    plottable_data = log_DB
    print("____FINISHED Training______")
    return log_DB


def trainFromLastMC(model, TrainFolder, ValidFolder, epochs, batchSize, TestFolder=None, logName=None, Learning_Rate=0,
                    SchedulerName='Plateau', Scheduler_Patience=3, percentagOfUnlabelledTiles=0.3):
    global Net
    global scheduler
    global optimizer
    Net = model  # Load net
    Net = Net.to(device)
    gc.collect()
    cuda.empty_cache()

    if percentagOfUnlabelledTiles > 1 or percentagOfUnlabelledTiles < 0:
        print("percentagOfUnlabelledTiles must be between 0 and 1.  Training cannot start.")
        return
    global batch_size
    batch_size = batchSize
    ListImages = os.listdir(os.path.join(TrainFolder, "images"))  # Create list of images
    ListOfLabelledImages = os.listdir(os.path.join(TrainFolder, "labels"))
    UnlabelledImages = copy.deepcopy(ListImages)
    for item in ListOfLabelledImages:
        UnlabelledImages.remove(item)

    amountInSample = int(len(UnlabelledImages) * percentagOfUnlabelledTiles)
    unbatched = (len(ListOfLabelledImages) + amountInSample) % batch_size
    batch_counts = round((len(ListOfLabelledImages) + amountInSample - unbatched) / batch_size)

    vListImages = os.listdir(os.path.join(ValidFolder, "images"))  # Create list of validation images
    if len(vListImages) < (batchSize * 4):
        vbatch_size = batchSize * 4
    else:
        vbatch_size = batchSize
    vunbatched = len(vListImages) % vbatch_size
    vbatch_counts = round((len(vListImages) - vunbatched) / vbatch_size)

    if TestFolder != None:
        tListImages = os.listdir(os.path.join(TestFolder, "images"))  # Create list of test images
        if len(tListImages) < (batchSize * 4):
            tbatch_size = batchSize * 4
        else:
            tbatch_size = batchSize
        tunbatched = len(tListImages) % tbatch_size
        tbatch_counts = round((len(tListImages) - tunbatched) / tbatch_size)
    else:
        tbatch_counts = 0

    global height
    global width
    # autodetect height and width of training tiles
    tempImage = cv2.imread(os.path.join(TrainFolder, "images", ListImages[0]), cv2.IMREAD_COLOR)
    height = tempImage.shape[0]
    width = tempImage.shape[1]
    del tempImage

    global plottable_data
    global logFilePath
    model_naming_title = Netname(Net)
    if logName==None: log_path = 'LOG for MC-' + model_naming_title + '.csv'
    else: log_path = logName
    logFilePath = log_path
    log_titles = ['Epoch', 'Train-Loss', 'Val-Loss', 'Val-Acc', 'Test-Loss', 'Test-Acc', 'Time', 'Learn-Rate',
                  'Session', 'CheckPoint']
    log_DB = pd.DataFrame(columns=log_titles)

    if os.path.exists(log_path):
        print("A log file for MC", model_naming_title, " was found as: ", log_path)
        log_DB = pd.read_csv(log_path, sep=",", index_col=0)
        path = log_DB.tail(1)['Session']
        path = str(path[0])

        log_path2 = "./" + path + '/LOG for MC-' + path[7:] + '.csv'

        best_loss = log_DB['Train-Loss'].min()  # smallest loss value
        LastEpoch = int(log_DB.tail(1)['Epoch'])
        LastCheckpoint = log_DB.tail(1)['CheckPoint']
        if Learning_Rate == 0: Learning_Rate = float(log_DB.tail(1)['Learn-Rate'])  # the last learning rate logged
        EpochsStartFrom = LastEpoch + 1

        if os.path.exists(path):
            print("Folder for checkpoints: ", path, " was found")
        elif os.path.exists('Models/' + path):
            path = 'Models/' + path
            print("Folder for checkpoints: ", path, " was found")
        else:
            print("Folder for checkpoints: ", path, " or Models/" + path + " were not found.  Training cannot continue")
            del epochs
            print(
                " Please restore the folder and restart this notebook, or start training from scratch in appropriate notebook")
            return

        if os.path.exists("./" + LastCheckpoint[0]):
            checkpoint = "./" + LastCheckpoint[0]  # Path to trained model
            print("Training to continue from checkpoint:", checkpoint)
        elif os.path.exists("./Models/" + LastCheckpoint[0]):
            checkpoint = "./Models/" + LastCheckpoint[0]  # Path to trained model
            print("Training to continue from checkpoint:", checkpoint)
        else:
            print("Last Checkpoint was found at NEITHER ./" + LastCheckpoint[0] + " NOR at ./Models/" + LastCheckpoint[
                0] + "  .  Training cannot continue")
            # print(" Please specify a path to a saved checkpoint manually in the next cell")
            return

    else:
        print(" Training Log File:  '", log_path, "'  was not found...  ")
        print(
            " Either restore the log file, pass a different model_naming_title to trainFromLast() , or start training from scratch with trainStart() ")
        return
    # _________________________PREPERATION DONE_________________#

    Net.load_state_dict(torch.load(checkpoint))  # if this gives an error check the training setup in previous 2 cells

    optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)

    if os.path.exists('./' + path + "/optimiser.optim"):
        optimizer.load_state_dict(torch.load('./' + path + "/optimiser.optim"))

    if SchedulerName == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, batch_counts)
    elif SchedulerName == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=Scheduler_Patience)
    else:
        print("SchedulerName parameter must be Cosine or Plateau. " + SchedulerName + " is not recognised")
        return

    if os.path.exists('./' + path + "/scheduler.sched"):  scheduler = torch.load('./' + path + "/scheduler.sched")

    model_naming_title = "-" + model_naming_title + ".torch"
    # log_path2="./"+path+'/LOG for '+path+'.csv'
    log_DB2 = pd.read_csv(log_path2, sep=",", index_col=0)

    trainingDetails(path, TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder), tbatch_counts,
                    checkpoint, EpochsStartFrom + epochs, finished=False)
    # _________________________TRAINING STARTS FROM HERE_________________#

    with torch.autograd.set_detect_anomaly(False):
        for itr in range(epochs):  # Training loop
            start_time = datetime.now()
            train_loss = 0
            runs = batch_counts
            vruns = vbatch_counts

            random.shuffle(UnlabelledImages)
            sampleOfUnlabelledImages = UnlabelledImages[0:amountInSample]
            TrainSample = copy.deepcopy(ListOfLabelledImages)
            TrainSample.extend(sampleOfUnlabelledImages)
            random.shuffle(TrainSample)

            Net.train()
            for count in tqdm(range(batch_counts)):
                BatchOfImages = TrainSample[(count * batch_size):(count * batch_size) + batch_size - 1]
                # print((count*batch_size)," ",(count*batch_size)+batch_size-1)
                images, labels = LoadNextRandomBatch(BatchOfImages, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            if unbatched > 0:
                BatchOfImages = TrainSample[(batch_counts * batch_size):(batch_counts * batch_size) + unbatched - 1]
                images, labels = LoadNextRandomBatch(BatchOfImages, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                runs = batch_counts + 1
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            train_loss = train_loss / (runs)  # +1) #averages the loss on all batches
            if SchedulerName == 'Cosine':
                scheduler.step()
            else:
                scheduler.step(train_loss)
            print(itr + EpochsStartFrom, "=> TrainLoss=", train_loss)
            # BEGIN Validation 

            Net.eval()
            with torch.no_grad():
                valid_loss = 0
                ValACC = 0

                for vbatchNum in tqdm(range(vbatch_counts)):
                    images, labels = LoadNext(vbatchNum, vbatch_size, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy

                if vunbatched > 0:
                    images, labels = LoadNext(vbatch_counts + 1, vunbatched, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy
                    vruns = vbatch_counts + 1

                valid_loss = valid_loss / (vruns)  # averages the loss on all batches
                ValACC = ValACC / (vruns)
                print(itr + EpochsStartFrom, "=> ValLoss=", valid_loss, " valACC=", ValACC)
                # END   Validation

                test_loss = 0
                testACC = 0
                if TestFolder != None:
                    truns = tbatch_counts
                    for tbatchNum in tqdm(range(tbatch_counts)):
                        images, labels = LoadNext(tbatchNum, tbatch_size, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy

                    if tunbatched > 0:
                        images, labels = LoadNext(tbatch_counts + 1, tunbatched, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy
                        truns = tbatch_counts + 1

                    test_loss = test_loss / (truns)  # averages the loss on all batches
                    testACC = testACC / (truns)
                    print(itr + EpochsStartFrom, "=>  TestLoss=", test_loss, " testACC=", testACC)
                # END Test                

            duration = datetime.now() - start_time

            if train_loss <= best_loss:
                best_loss = copy.deepcopy(train_loss)

            t = datetime.now()
            checkpoint = path + "/" + str(itr + EpochsStartFrom) + "-" + str(t.hour) + str(
                t.minute) + model_naming_title
            print("Saving Model: ", "./" + checkpoint, end="")
            torch.save(Net.state_dict(), "./" + checkpoint)

            """
            else:
                if itr!=epochs-1:  checkpoint="not saved"
                else:
                    checkpoint=path+"/"+str(itr+EpochsStartFrom)+"-LAST EPOCH-"+model_naming_title
                    torch.save(Net.state_dict(),"./"+checkpoint)
                    print(" Saving LAST EPOCH: ./",checkpoint)
            """

            torch.save(optimizer.state_dict(), path + "/optimiser.optim")
            torch.save(scheduler, path + "/scheduler.sched")

            print("lr:", GetLastLR(SchedulerName), "  Time:", duration.seconds)
            # new_log_entry=pd.DataFrame([[itr+EpochsStartFrom, train_loss, valid_loss,ValACC, float(scheduler.state_dict()["_last_lr"][0]),path,checkpoint]], columns=log_titles)
            new_log_entry = pd.DataFrame([[itr + EpochsStartFrom, train_loss, valid_loss, ValACC, test_loss, testACC,
                                           duration.seconds, GetLastLR(SchedulerName), path, checkpoint]],
                                         columns=log_titles)
            log_DB = pd.concat([log_DB, new_log_entry])
            log_DB.to_csv(log_path, sep=",")
            log_DB2 = pd.concat([log_DB2, new_log_entry])
            log_DB2.to_csv(log_path2, sep=",")

    # _________________________TRAINING LOOP ENDS HERE_________________#
    trainingDetails(path, TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder), tbatch_counts,
                    str(EpochsStartFrom), checkpoint, finished=True)
    plottable_data = log_DB
    print("____FINISHED Training______")
    return log_DB


"""
def trainStartMC(model, TrainFolder, ValidFolder,epochs, batchSize,TestFolder=None,Learning_Rate=1e-5, logName=None,SchedulerName='Plateau'):
        
    global batch_size
    batch_size=batchSize
    vbatch_size=batchSize*4
    ListImages=os.listdir(os.path.join(TrainFolder, "images")) # Create list of images        
    vListImages=os.listdir(os.path.join(ValidFolder, "images")) # Create list of validation images
    unbatched=len(ListImages)%batch_size
    batch_counts=round((len(ListImages)-unbatched)/batch_size)
    vunbatched=len(vListImages)%vbatch_size
    vbatch_counts=round((len(vListImages)-vunbatched)/vbatch_size)
        
    if TestFolder!=None: 
        tbatch_size=batchSize*4
        tListImages=os.listdir(os.path.join(TestFolder, "images")) # Create list of test images
        tunbatched=len(tListImages)%tbatch_size
        tbatch_counts=round((len(tListImages)-tunbatched)/tbatch_size)
    else:
        tbatch_counts=0

    global Net
    global scheduler
    global optimizer
    #load model 
    Net = model # Load net
    Net=Net.to(device)
    model_naming_title=Netname(Net)
    optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate) # Create adam optimizer
    if SchedulerName=='Cosine':
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,batch_counts)
    elif SchedulerName=='Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    else:
        print("SchedulerName parameter must be Cosine or Plateau. "+SchedulerName+" is not recognised")
        return


    #autodetect height and width of training tiles
    global height
    global width
    tempImage=cv2.imread(os.path.join(TrainFolder, "images", ListImages[0]), cv2.IMREAD_COLOR)
    height=tempImage.shape[0]
    width=tempImage.shape[1]
    del tempImage
    

    global logFilePath
    global plottable_data
    
    t=datetime.now()
    DateTime =str(t.hour)+str(t.minute)+"-"+str(t.day)+"-"+str(t.month)+"-"+str(t.year)
    path='Models/'+model_naming_title+"-"+DateTime
    del t
    
    if not os.path.exists(path): os.makedirs(path)

    if logName==None:
        log_path='LOG for MC-'+model_naming_title+'.csv'
    else:
        log_path=logName

    if not os.path.exists(log_path):
        log_path2="./"+path+'/LOG for MC-'+model_naming_title+"-"+DateTime+'.csv'
        log_titles=['Epoch','Train-Loss','Val-Loss', 'Val-Acc', 'Test-Loss','Test-Acc', 'Time','Learn-Rate','Session','CheckPoint']
        log_DB=pd.DataFrame( columns=log_titles)
        log_DB2=pd.DataFrame( columns=log_titles)
        model_naming_title="-"+model_naming_title+".torch"

        print(" Folder for checkpoints:'",path,"' was created")
        print(" Training Log File:'",log_path,"' will be created... Starting training from scratch")
        Learning_Rate
        best_loss=1
        EpochsStartFrom=0  #in case training is restarted from a previously saved epoch, this continues the sequence
        # and prevents over-writing models and logs in the loss database 

        
    else: 
        print("Log file for ",model_naming_title, " already present as: ", log_path)
        print("Training will not start, to prevent overwriting")
        print(" ")
        print("If you really want to start from scratch, please move the existent log file")
        print("If you want to continue training please use the trainfromLast function instead")
        return
    
    del DateTime
    trainingDetails(path,TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder) ,tbatch_counts, '0', epochs-1, finished=False)
    #_________________________PREPERATION DONE_________________#
    #_________________________TRAINING LOOP STARTS FROM HERE_________________#
    with torch.autograd.set_detect_anomaly(False):
        gc.collect()
        cuda.empty_cache()

        for itr in range(epochs): # Training loop
            start_time= datetime.now()
            train_loss=0
            runs=batch_counts
            vruns=vbatch_counts
            
            Net.train()
            for batchNum in tqdm(range(batch_counts)):
                images,labels=LoadNext(batchNum,batch_size,TrainFolder)
                train_loss=train_loss+learn(images, labels)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()
       
            if unbatched>0:
                images,labels=LoadNext(batch_counts+1,unbatched,TrainFolder)
                train_loss=train_loss+learn(images, labels)
                runs=batch_counts+1
                del images
                del labels
                gc.collect()
                cuda.empty_cache()
    
            #uncomment if you want to train on a random batch too
            #images,labels=LoadBatch(TrainFolder) 
            #train_loss+=learn(images, labels)
            #runs=batch_counts+1
         

            train_loss=train_loss/(runs) # +1) #averages the loss on all batches
            if SchedulerName=='Cosine':
                scheduler.step()
            else:
                scheduler.step(train_loss)
        
            #BEGIN Validation 
            Net.eval()
            with torch.no_grad():
                valid_loss=0
                ValACC=0
        
                for vbatchNum in tqdm(range(vbatch_counts)):
                    images,labels=LoadNext(vbatchNum,vbatch_size,ValidFolder)
                    newVloss,batch_accuracy=validate(images, labels,vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss=valid_loss+newVloss
                    ValACC=ValACC+batch_accuracy
       
                if vunbatched>0:
                    images,labels=LoadNext(vbatch_counts+1,vunbatched,ValidFolder)
                    newVloss,batch_accuracy=validate(images, labels,vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss=valid_loss+newVloss
                    ValACC=ValACC+batch_accuracy
                    vruns=vbatch_counts+1
        
                valid_loss=valid_loss/(vruns) #averages the loss on all batches
                ValACC=ValACC/(vruns)
                #END   Validation
                
                test_loss=0
                testACC=0
                if TestFolder!=None:
                    truns=tbatch_counts
                    for tbatchNum in tqdm(range(tbatch_counts)):
                        images,labels=LoadNext(tbatchNum,tbatch_size,TestFolder)
                        newVloss,batch_accuracy=validate(images, labels,vbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss=test_loss+newVloss
                        testACC=testACC+batch_accuracy
       
                    if tunbatched>0:
                        images,labels=LoadNext(tbatch_counts+1,tunbatched,TestFolder)
                        newVloss,batch_accuracy=validate(images, labels,vbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss=test_loss+newVloss
                        testACC=testACC+batch_accuracy
                        truns=tbatch_counts+1
        
                    test_loss=test_loss/(truns) #averages the loss on all batches
                    testACC=testACC/(truns)
                    #END Test                
            
     
            duration=datetime.now()-start_time
        
            if train_loss<=best_loss:
                best_loss=copy.deepcopy(train_loss)
    
            t=datetime.now()
            checkpoint=path+"/"+str(itr+EpochsStartFrom)+"-"+str(t.hour)+str(t.minute)+model_naming_title
            print("Saving Model: ", "./"+checkpoint)
            torch.save(Net.state_dict(),"./"+checkpoint)
            
            torch.save(optimizer.state_dict(),path+"/optimiser.optim")
            torch.save(scheduler,path+"/scheduler.sched")
    
            ####
            else:
                if itr!=epochs-1:  checkpoint="not saved"
                else:
                    checkpoint=path+"/"+str(itr+EpochsStartFrom)+"-LAST EPOCH-"+model_naming_title
                    torch.save(Net.state_dict(),"./"+checkpoint)
                    print(" Saving LAST EPOCH: ./",checkpoint)
            ####    
            print(itr+EpochsStartFrom,"=> TrainLoss=",train_loss,"  ValLoss=", valid_loss,  "  valACC=",ValACC, " TestLoss=", test_loss," testACC=",testACC, " lr:",  GetLastLR(SchedulerName), " Time:", duration.seconds)
            new_log_entry=pd.DataFrame([[itr+EpochsStartFrom, train_loss, valid_loss,ValACC,test_loss,testACC, duration.seconds,GetLastLR(SchedulerName),path,checkpoint]], columns=log_titles)
            log_DB=pd.concat([log_DB, new_log_entry])
            log_DB.to_csv(log_path, sep=",")
            log_DB2=pd.concat([log_DB2, new_log_entry])
            log_DB2.to_csv(log_path2, sep=",")

    #_________________________TRAINING LOOP ENDS HERE_________________#
    trainingDetails(path,TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder),tbatch_counts, '0', checkpoint, finished=True)
    plottable_data =log_DB
    print("____FINISHED Training______")
    return log_DB

def trainFromLastMC(model, TrainFolder, ValidFolder, epochs, batchSize, TestFolder=None, Learning_Rate=0,SchedulerName='Plateau'):
    global Net
    global scheduler
    global optimizer
    Net = model # Load net
    Net=Net.to(device)
    gc.collect()
    cuda.empty_cache()
    
    global batch_size
    batch_size=batchSize
    vbatch_size=batchSize*4
    ListImages=os.listdir(os.path.join(TrainFolder, "images")) # Create list of images        
    vListImages=os.listdir(os.path.join(ValidFolder, "images")) # Create list of validation images
    unbatched=len(ListImages)%batch_size
    batch_counts=round((len(ListImages)-unbatched)/batch_size)
    vunbatched=len(vListImages)%vbatch_size
    vbatch_counts=round((len(vListImages)-vunbatched)/vbatch_size)

    if TestFolder!=None: 
        tbatch_size=batchSize*4
        tListImages=os.listdir(os.path.join(TestFolder, "images")) # Create list of test images
        tunbatched=len(tListImages)%tbatch_size
        tbatch_counts=round((len(tListImages)-tunbatched)/tbatch_size)
    else:
        tbatch_counts=0
    
    global height
    global width
    #autodetect height and width of training tiles
    tempImage=cv2.imread(os.path.join(TrainFolder, "images", ListImages[0]), cv2.IMREAD_COLOR)
    height=tempImage.shape[0]
    width=tempImage.shape[1]
    del tempImage
    
    global plottable_data
    global logFilePath
    model_naming_title=Netname(Net)
    log_path='LOG for MC-'+model_naming_title+'.csv'
    logFilePath=log_path
    log_titles=['Epoch','Train-Loss','Val-Loss', 'Val-Acc', 'Test-Loss','Test-Acc', 'Time','Learn-Rate','Session','CheckPoint']
    log_DB=pd.DataFrame( columns=log_titles)

    if os.path.exists(log_path):
        print("A log file for MC",model_naming_title," was found as: ",log_path)
        log_DB=pd.read_csv(log_path, sep=",", index_col=0)
        path=log_DB.tail(1)['Session']
        path=str(path[0])
        
        log_path2="./"+path+'/LOG for MC-'+path[7:]+'.csv'
        
        best_loss=log_DB['Train-Loss'].min() #smallest loss value
        LastEpoch=int(log_DB.tail(1)['Epoch'])
        LastCheckpoint=log_DB.tail(1)['CheckPoint']
        if Learning_Rate==0: Learning_Rate=float(log_DB.tail(1)['Learn-Rate'])  #the last learning rate logged
        EpochsStartFrom=LastEpoch+1
    
        if os.path.exists(path):
            print("Folder for checkpoints: ",path, " was found")
        elif os.path.exists('Models/'+path):
            path='Models/'+path
            print("Folder for checkpoints: ",path, " was found")
        else:
            print("Folder for checkpoints: ",path, " or Models/"+path+" were not found.  Training cannot continue")
            del epochs
            print(" Please restore the folder and restart this notebook, or start training from scratch in appropriate notebook")
            return
        
        if os.path.exists("./"+LastCheckpoint[0]):
            checkpoint ="./"+LastCheckpoint[0] # Path to trained model
            print("Training to continue from checkpoint:", checkpoint)
        elif os.path.exists("./Models/"+LastCheckpoint[0]):
            checkpoint ="./Models/"+LastCheckpoint[0] # Path to trained model
            print("Training to continue from checkpoint:", checkpoint)
        else: 
            print("Last Checkpoint was found at NEITHER ./"+LastCheckpoint[0]+" NOR at ./Models/"+LastCheckpoint[0]+"  .  Training cannot continue")
            #print(" Please specify a path to a saved checkpoint manually in the next cell")
            return
            
    else:
        print(" Training Log File:  '",log_path,"'  was not found...  ")
        print(" Either restore the log file, pass a different model_naming_title to trainFromLast() , or start training from scratch with trainStart() ")
        return
    #_________________________PREPERATION DONE_________________#
    
    Net.load_state_dict(torch.load(checkpoint))  #if this gives an error check the training setup in previous 2 cells

    
    optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate)
    
    if os.path.exists('./'+path+"/optimiser.optim"):
        optimizer.load_state_dict(torch.load('./'+path+"/optimiser.optim"))
    
    
    if SchedulerName=='Cosine':
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,batch_counts)
    elif SchedulerName=='Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    else:
        print("SchedulerName parameter must be Cosine or Plateau. "+SchedulerName+" is not recognised")
        return
        
    if os.path.exists('./'+path+"/scheduler.sched"):  scheduler=torch.load('./'+path+"/scheduler.sched")

    
    
    
    
    model_naming_title="-"+model_naming_title+".torch"
    #log_path2="./"+path+'/LOG for '+path+'.csv'
    log_DB2=pd.read_csv(log_path2, sep=",", index_col=0)

    trainingDetails(path,TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder),tbatch_counts, checkpoint, EpochsStartFrom+epochs, finished=False)
    #_________________________TRAINING STARTS FROM HERE_________________#

    with torch.autograd.set_detect_anomaly(False):
        for itr in range(epochs): # Training loop
            start_time= datetime.now()
            train_loss=0
            runs=batch_counts
            vruns=vbatch_counts
    
            Net.train()
            for batchNum in tqdm(range(batch_counts)):
                images,labels=LoadNext(batchNum,batch_size,TrainFolder)
                train_loss=train_loss+learn(images, labels)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()
       
            if unbatched>0:
                images,labels=LoadNext(batch_counts+1,unbatched,TrainFolder)
                train_loss=train_loss+learn(images, labels)
                runs=batch_counts+1
                del images
                del labels
                gc.collect()
                cuda.empty_cache()
    
            #uncomment if you want to train on a random batch too
            ###
            ###images,labels=LoadBatch(TrainFloder) 
            #train_loss+=learn(images, labels)
            #runs=batch_counts+1
            ###

            train_loss=train_loss/(runs) # +1) #averages the loss on all batches
            if SchedulerName=='Cosine':
                scheduler.step()
            else:
                scheduler.step(train_loss)
            print(itr+EpochsStartFrom,"=> TrainLoss=",train_loss)
            #BEGIN Validation 
            
            Net.eval()
            with torch.no_grad():
                valid_loss=0
                ValACC=0
        
                for vbatchNum in tqdm(range(vbatch_counts)):
                    images,labels=LoadNext(vbatchNum,vbatch_size,ValidFolder)
                    newVloss,batch_accuracy=validate(images, labels,vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss=valid_loss+newVloss
                    ValACC=ValACC+batch_accuracy
       
                if vunbatched>0:
                    images,labels=LoadNext(vbatch_counts+1,vunbatched,ValidFolder)
                    newVloss,batch_accuracy=validate(images, labels,vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss=valid_loss+newVloss
                    ValACC=ValACC+batch_accuracy
                    vruns=vbatch_counts+1
        
                valid_loss=valid_loss/(vruns) #averages the loss on all batches
                ValACC=ValACC/(vruns)
                print(itr+EpochsStartFrom,"=> ValLoss=", valid_loss,  " valACC=",ValACC)
                #END   Validation
                
                test_loss=0
                testACC=0
                if TestFolder!=None:
                    truns=tbatch_counts
                    for tbatchNum in tqdm(range(tbatch_counts)):
                        images,labels=LoadNext(tbatchNum,tbatch_size,TestFolder)
                        newVloss,batch_accuracy=validate(images, labels,vbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss=test_loss+newVloss
                        testACC=testACC+batch_accuracy
       
                    if tunbatched>0:
                        images,labels=LoadNext(tbatch_counts+1,tunbatched,TestFolder)
                        newVloss,batch_accuracy=validate(images, labels,vbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss=test_loss+newVloss
                        testACC=testACC+batch_accuracy
                        truns=tbatch_counts+1
        
                    test_loss=test_loss/(truns) #averages the loss on all batches
                    testACC=testACC/(truns)
                    print(itr+EpochsStartFrom,"=>  TestLoss=", test_loss," testACC=",testACC)
                    #END Test                
            
     
            duration=datetime.now()-start_time
        
            if train_loss<=best_loss:
                best_loss=copy.deepcopy(train_loss)
    
            t=datetime.now()
            checkpoint=path+"/"+str(itr+EpochsStartFrom)+"-"+str(t.hour)+str(t.minute)+model_naming_title
            print("Saving Model: ", "./"+checkpoint, end="")
            torch.save(Net.state_dict(),"./"+checkpoint)
    
            ###
            #else:
            #    if itr!=epochs-1:  checkpoint="not saved"
            #    else:
            #        checkpoint=path+"/"+str(itr+EpochsStartFrom)+"-LAST EPOCH-"+model_naming_title
            #        torch.save(Net.state_dict(),"./"+checkpoint)
            #        print(" Saving LAST EPOCH: ./",checkpoint)
            ###    
            print("lr:", GetLastLR(SchedulerName), "  Time:", duration.seconds)
            #new_log_entry=pd.DataFrame([[itr+EpochsStartFrom, train_loss, valid_loss,ValACC, float(scheduler.state_dict()["_last_lr"][0]),path,checkpoint]], columns=log_titles)
            new_log_entry=pd.DataFrame([[itr+EpochsStartFrom, train_loss, valid_loss,ValACC,test_loss,testACC, duration.seconds , GetLastLR(SchedulerName),path,checkpoint]], columns=log_titles)
            log_DB=pd.concat([log_DB, new_log_entry])
            log_DB.to_csv(log_path, sep=",")
            log_DB2=pd.concat([log_DB2, new_log_entry])
            log_DB2.to_csv(log_path2, sep=",")

    #_________________________TRAINING LOOP ENDS HERE_________________#
    trainingDetails(path,TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder) ,tbatch_counts, str(EpochsStartFrom), checkpoint, finished=True)
    plottable_data =log_DB
    print("____FINISHED Training______")
    return log_DB
"""


def plotTraining():
    global plottable_data
    import matplotlib.pyplot as plt

    xAxis = plottable_data['Epoch']
    yAxis = plottable_data['Train-Loss']
    yAxis2 = plottable_data['Val-Loss']

    plt.plot(xAxis, yAxis, xAxis, yAxis2)

    plt.title('Loss Chart')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()


def fineTune(model, checkpoint_path, TrainFolder, ValidFolder, epochs, batchSize, TrainFolder2=None, TestFolder=None,
             Learning_Rate=1e-5, ShedulerName='Plateau'):
    global height
    global width
    tempImage = cv2.imread(os.path.join(TrainFolder, "images", ListImages[0]), cv2.IMREAD_COLOR)
    height = tempImage.shape[0]
    width = tempImage.shape[1]
    del tempImage

    global Net
    global optimizer

    model_naming_title = Netname(model)
    Net = model  # Load net
    Net = Net.to(device)
    print("Loading checkpoint ", checkpoint_path, end=" ...")
    Net.load_state_dict(torch.load(checkpoint_path))
    optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)  # Create adam optimizer
    if SchedulerName == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, batch_counts)
    elif SchedulerName == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    else:
        print("SchedulerName parameter must be Cosine or Plateau. " + SchedulerName + " is not recognised")
        return
    gc.collect()
    cuda.empty_cache()
    print("... DONE")

    t = datetime.now()
    DateTime = str(t.hour) + str(t.minute) + "-" + str(t.day) + "-" + str(t.month) + "-" + str(t.year)

    path = 'Models/FINETUNE' + model_naming_title + "-" + DateTime
    del t
    del DateTime
    if not os.path.exists(path): os.makedirs(path)

    global batch_size
    batch_size = batchSize
    vbatch_size = batchSize * 4

    global plottable_data
    global logFilePath

    log_path = 'Log for FINETUNING ' + model_naming_title + '.csv'

    log_path2 = "./" + path + '/LOG for FINETUNING ' + model_naming_title + "-" + DateTime + '.csv'
    logFilePath = log_path
    log_titles = ['Epoch', 'Train-Loss', 'Val-Loss', 'Val-Acc', 'Test-Loss', 'Test-Acc', 'Time', 'Learn-Rate',
                  'Session', 'CheckPoint']
    log_DB = pd.DataFrame(columns=log_titles)

    ListImages = os.listdir(os.path.join(TrainFolder, "images"))  # Create list of images        
    unbatched = len(ListImages) % batch_size
    batch_counts = round((len(ListImages) - unbatched) / batch_size)
    vListImages = os.listdir(os.path.join(ValidFolder, "images"))  # Create list of validation images
    vunbatched = len(vListImages) % vbatch_size
    vbatch_counts = round((len(vListImages) - vunbatched) / vbatch_size)

    if TestFolder != None:
        tbatch_size = batchSize * 4
        tListImages = os.listdir(os.path.join(TestFolder, "images"))  # Create list of test images
        tunbatched = len(tListImages) % tbatch_size
        tbatch_counts = round((len(tListImages) - tunbatched) / tbatch_size)
    else:
        tbatch_counts = 0

    if TrainFolder2 != None:
        ListImages2 = os.listdir(os.path.join(TrainFolder2, "images"))  # Create list of test images
        unbatched2 = len(ListImages2) % tbatch_size
        batch_counts2 = round((len(ListImages2) - unbatched2) / batch_size)
    else:
        batch_counts2 = 0

    trainingDetails(path, str(TrainFolder) + ' and ' + str(TrainFolder2),
                    str(batch_counts) + ' and ' + str(batch_counts2), ValidFolder, vbatch_counts, str(TestFolder),
                    tbatch_counts, '0', epochs - 1, finished=False)

    EpochsStartFrom = 0
    # _________________________TRAINING LOOP STARTS FROM HERE_________________#
    with torch.autograd.set_detect_anomaly(False):
        gc.collect()
        cuda.empty_cache()

        for itr in range(epochs):  # Training loop
            start_time = datetime.now()
            train_loss = 0
            runs = batch_counts
            vruns = vbatch_counts

            Net.train()
            for batchNum in tqdm(range(batch_counts)):
                images, labels = LoadNext(batchNum, batch_size, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            if unbatched > 0:
                images, labels = LoadNext(batch_counts + 1, unbatched, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                runs = batch_counts + 1
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            if TrainFolder2 != None:
                for batchNum in tqdm(range(batch_counts2)):
                    images, labels = LoadNext(batchNum, batch_size, TrainFolder2)
                    train_loss = train_loss + learn(images, labels)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()

                runs = runs + batch_counts2

                if unbatched2 > 0:
                    images, labels = LoadNext(batch_counts + 1, unbatched2, TrainFolder2)
                    train_loss = train_loss + learn(images, labels)
                    runs = batch_counts + 1
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()

            train_loss = train_loss / (runs)  # +1) #averages the loss on all batches
            if SchedulerName == 'Cosine':
                scheduler.step()
            else:
                scheduler.step(train_loss)
            print(itr + EpochsStartFrom, "=> TrainLoss=", train_loss)

            # BEGIN Validation 
            Net.eval()
            with torch.no_grad():
                valid_loss = 0
                ValACC = 0

                for vbatchNum in tqdm(range(vbatch_counts)):
                    images, labels = LoadNext(vbatchNum, vbatch_size, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy

                if vunbatched > 0:
                    images, labels = LoadNext(vbatch_counts + 1, vunbatched, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy
                    vruns = vbatch_counts + 1

                valid_loss = valid_loss / (vruns)  # averages the loss on all batches
                ValACC = ValACC / (vruns)
                # END   Validation

                test_loss = 0
                testACC = 0
                if TestFolder != None:
                    truns = tbatch_counts
                    for tbatchNum in tqdm(range(tbatch_counts)):
                        images, labels = LoadNext(tbatchNum, tbatch_size, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, tbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy

                    if tunbatched > 0:
                        images, labels = LoadNext(tbatch_counts + 1, tunbatched, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, tbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy
                        truns = tbatch_counts + 1

                    test_loss = test_loss / (truns)  # averages the loss on all batches
                    testACC = testACC / (truns)
                # END Test                

            duration = datetime.now() - start_time

            if train_loss <= best_loss:
                best_loss = copy.deepcopy(train_loss)

            t = datetime.now()
            checkpoint = path + "/" + str(itr + EpochsStartFrom) + "-" + str(t.hour) + str(
                t.minute) + model_naming_title
            print("Saving Model: ", "./" + checkpoint)
            torch.save(Net.state_dict(), "./" + checkpoint)

            """
            else:
                if itr!=epochs-1:  checkpoint="not saved"
                else:
                    checkpoint=path+"/"+str(itr+EpochsStartFrom)+"-LAST EPOCH-"+model_naming_title
                    torch.save(Net.state_dict(),"./"+checkpoint)
                    print(" Saving LAST EPOCH: ./",checkpoint)
            """
            print(itr + EpochsStartFrom, "=> TrainLoss=", train_loss, "  ValLoss=", valid_loss, "  valACC=", ValACC,
                  " TestLoss=", test_loss, " testACC=", testACC, " lr:", scheduler.state_dict()["_last_lr"][0],
                  " Time:", duration.seconds)
            new_log_entry = pd.DataFrame([[itr + EpochsStartFrom, train_loss, valid_loss, ValACC, test_loss, testACC,
                                           duration.seconds, float(scheduler.state_dict()["_last_lr"][0]), path,
                                           checkpoint]], columns=log_titles)
            log_DB = pd.concat([log_DB, new_log_entry])
            log_DB.to_csv(log_path, sep=",")
            log_DB2 = pd.concat([log_DB2, new_log_entry])
            log_DB2.to_csv(log_path2, sep=",")

    # _________________________TRAINING LOOP ENDS HERE_________________#
    trainingDetails(path, TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder), tbatch_counts, '0',
                    checkpoint, finished=True)
    print("____FINISHED FINE TUNING______")
    plottable_data = log_DB
    return log_DB


def GenerateLog(model, CheckpointFolder, TrainFolder, ValidFolder, TestFolder=None, batchSize=192):
    print("This function needs updating to use the latest validation functions")

    def getEpochNum(c):  # takes checkpoint name and returns the check point number
        epochNum = ''
        for l in c:
            if l != '-':
                epochNum = epochNum + l
            else:
                break
        return epochNum

    global height
    global width
    # global optimizer
    # global logFilePath
    global batch_size
    batch_size = batchSize

    global Net
    Net = None
    gc.collect()
    cuda.empty_cache()
    Net = model  # Load net
    Net = Net.to(device)

    ListImages = os.listdir(os.path.join(TrainFolder, "images"))  # Create list of images        
    vListImages = os.listdir(os.path.join(ValidFolder, "images"))  # Create list of validation images
    if TestFolder != None:
        tListImages = os.listdir(os.path.join(TestFolder, "images"))  # Create list of test images

    unbatched = len(ListImages) % batch_size
    batch_counts = round((len(ListImages) - unbatched) / batch_size)
    vunbatched = len(vListImages) % batch_size
    vbatch_counts = round((len(vListImages) - vunbatched) / batch_size)
    if TestFolder != None:
        tListImages = os.listdir(os.path.join(TestFolder, "images"))  # Create list of test images
        tbatch_size = batchSize
        tunbatched = len(tListImages) % tbatch_size
        tbatch_counts = round((len(tListImages) - tunbatched) / tbatch_size)

    log_path = "./" + CheckpointFolder + '/LOG for RENAME HERE.csv'
    DBtitles = ['Epoch', 'Train-Loss', 'Val-Loss', 'Val-Acc', 'Test-Loss', 'Test-Acc', 'Time', 'Learn-Rate', 'Session',
                'CheckPoint']
    log_DB = pd.DataFrame(columns=DBtitles)

    s = os.path.dirname(CheckpointFolder)

    temp = os.listdir(CheckpointFolder)
    CheckPointList = []
    for c in temp:
        if c[-6:] == '.torch':
            CheckPointList.append(c)
    del temp

    i = 0
    for c in CheckPointList:
        print("Evaluating Checkpoint: ", str(c))
        Net.load_state_dict(torch.load('./' + CheckpointFolder + '/' + c))
        Net.eval()
        gc.collect()
        cuda.empty_cache()
        start_time = datetime.now()
        with torch.no_grad():
            train_loss = 0
            runs = batch_counts
            vruns = vbatch_counts

            for batchNum in tqdm(range(batch_counts)):
                images, labels = LoadNext(batchNum, batch_size, TrainFolder)
                this_loss, batch_accuracy = validate(images, labels, batch_size)
                train_loss = this_loss + train_loss
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            if unbatched > 0:
                images, labels = LoadNext(batch_counts + 1, unbatched, TrainFolder)
                this_loss, batch_accuracy = validate(images, labels, batch_size)
                train_loss = this_loss + train_loss
                runs = batch_counts + 1
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            train_loss = train_loss / (runs)  # +1) #averages the loss on all batches

            valid_loss = 0
            ValACC = 0

            for vbatchNum in tqdm(range(vbatch_counts)):
                images, labels = LoadNext(vbatchNum, batch_size, ValidFolder)
                newVloss, batch_accuracy = validate(images, labels, batch_size)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()
                valid_loss = valid_loss + newVloss
                ValACC = ValACC + batch_accuracy

            if vunbatched > 0:
                images, labels = LoadNext(vbatch_counts + 1, vunbatched, ValidFolder)
                newVloss, batch_accuracy = validate(images, labels, vunbatched)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()
                valid_loss = valid_loss + newVloss
                ValACC = ValACC + batch_accuracy
                vruns = vbatch_counts + 1

            valid_loss = valid_loss / (vruns)  # averages the loss on all batches
            ValACC = ValACC / (vruns)

            test_loss = 0
            testACC = 0
            if TestFolder != None:
                truns = tbatch_counts
                for tbatchNum in tqdm(range(tbatch_counts)):
                    images, labels = LoadNext(tbatchNum, tbatch_size, TestFolder)
                    newVloss, batch_accuracy = validate(images, labels, tbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    test_loss = test_loss + newVloss
                    testACC = testACC + batch_accuracy

                if tunbatched > 0:
                    images, labels = LoadNext(tbatch_counts + 1, tunbatched, TestFolder)
                    newVloss, batch_accuracy = validate(images, labels, tbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    test_loss = test_loss + newVloss
                    testACC = testACC + batch_accuracy
                    truns = tbatch_counts + 1

                test_loss = test_loss / (truns)  # averages the loss on all batches
                testACC = testACC / (truns)

        # ['Epoch','Train-Loss','Val-Loss', 'Val-Acc', 'Test-Loss','Test-Acc', 'Time','Learn-Rate','Session','CheckPoint']
        # print("checkpoint:", i, "  ", c, "  loss:",valid_loss )
        duration = datetime.now() - start_time
        new_entry = pd.DataFrame([[getEpochNum(c), train_loss, valid_loss, ValACC, test_loss, testACC, duration.seconds,
                                   'unknown', CheckpointFolder + '/' + c, c]], columns=DBtitles)
        log_DB = pd.concat([log_DB, new_entry])
        log_DB.to_csv(log_path, sep=",")
        i = i + 1

    print("DONE:  Log saved as: ", log_path)
    return log_DB


def validateCheckpoints(model, log_path, ValidFolder, batchSize=128):
    global height
    global width
    global Net
    global batch_size
    batch_size = batchSize

    vListImages = os.listdir(os.path.join(ValidFolder, "images"))  # Create list of validation images

    # autodetect height and width of training tiles
    tempImage = cv2.imread(os.path.join(ValidFolder, "images", vListImages[0]), cv2.IMREAD_COLOR)
    height = tempImage.shape[0]
    width = tempImage.shape[1]
    del tempImage

    log_DB = pd.read_csv(log_path, sep=",", index_col=0)
    # log_DB=log_DB[log_DB['CheckPoint'][0]!= 'not saved']

    DBtitles = ['Epoch', 'CheckPoint', 'Val-loss', 'Acc']
    val_DB = pd.DataFrame(columns=DBtitles)

    dataset = os.path.basename(os.path.dirname(ValidFolder))

    s = log_DB[log_DB['Epoch'] == 0]
    s = s['Session'][0]

    i = 0

    for c in log_DB['CheckPoint']:

        if c == 'not saved':
            new_entry = pd.DataFrame([[i, c, 0, 0]], columns=DBtitles)
            val_DB = pd.concat([val_DB, new_log_entry])
            # valDB.to_csv("./"+s+"/validation-"+Netname(Net)+"-"+dataset+".csv", sep=",")
            i = i + 1
            continue

        if os.path.exists('./' + c):
            Net = LoadNet(model, './' + c)
        elif os.path.exists('./Models/' + c):
            Net = LoadNet(model, './Models/' + c)
        else:
            print("Checkpoint ", c, " not found.  ABORTING!!")
            continue

        vunbatched = len(vListImages) % batch_size
        vbatch_counts = round((len(vListImages) - vunbatched) / batch_size)
        valid_loss = 0
        ValACC = 0

        Net.eval()
        with torch.no_grad():
            valid_loss = 0
            ValACC = 0

            for vbatchNum in tqdm(range(vbatch_counts)):
                images, labels = LoadNext(vbatchNum, batch_size, ValidFolder)
                newVloss, batch_accuracy = validate(images, labels, batch_size)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()
                valid_loss = valid_loss + newVloss
                ValACC = ValACC + batch_accuracy

            if vunbatched > 0:
                images, labels = LoadNext(vbatch_counts + 1, vunbatched, ValidFolder)
                newVloss, batch_accuracy = validate(images, labels, vunbatched)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()
                valid_loss = valid_loss + newVloss
                ValACC = ValACC + batch_accuracy
                vruns = vbatch_counts + 1

        valid_loss = valid_loss / (vruns)  # averages the loss on all batches
        ValACC = ValACC / (vruns)

        print("checkpoint:", i, "  ", c, "  loss:", valid_loss)
        new_entry = pd.DataFrame([[i, c, valid_loss, ValACC]], columns=DBtitles)
        val_DB = pd.concat([val_DB, new_entry])
        val_DB.to_csv("./" + s + "/validation-" + Netname(Net) + "-" + dataset + ".csv", sep=",")
        i = i + 1

    print("DONE:  Saved as: ./" + s + "/validation-" + Netname(Net) + "-" + dataset + ".csv")


def validateCheckpointsMC(model, log_path, ValidFolder, batchSize=128, ValidateClass=1):
    global height
    global width
    global Net
    global batch_size

    batch_size = batchSize

    vListImages = os.listdir(os.path.join(ValidFolder, "images"))  # Create list of validation images

    # autodetect height and width of training tiles
    tempImage = cv2.imread(os.path.join(ValidFolder, "images", vListImages[0]), cv2.IMREAD_COLOR)
    height = tempImage.shape[0]
    width = tempImage.shape[1]
    del tempImage

    log_DB = pd.read_csv(log_path, sep=",", index_col=0)
    # log_DB=log_DB[log_DB['CheckPoint'][0]!= 'not saved']

    DBtitles = ['Epoch', 'CheckPoint', 'Val-loss', 'Acc']
    val_DB = pd.DataFrame(columns=DBtitles)

    dataset = os.path.basename(os.path.dirname(ValidFolder))

    s = log_DB[log_DB['Epoch'] == 0]
    s = s['Session'][0]

    i = 0

    for c in log_DB['CheckPoint']:

        if c == 'not saved':
            new_entry = pd.DataFrame([[i, c, 0, 0]], columns=DBtitles)
            val_DB = pd.concat([val_DB, new_log_entry])
            # valDB.to_csv("./"+s+"/validation-"+Netname(Net)+"-"+dataset+".csv", sep=",")
            i = i + 1
            continue

        if os.path.exists('./' + c):
            Net = LoadNet(model, './' + c)
        elif os.path.exists('./Models/' + c):
            Net = LoadNet(model, './Models/' + c)
        else:
            print("Checkpoint ", c, " not found.  ABORTING!!")
            continue

        vunbatched = len(vListImages) % batch_size
        vbatch_counts = round((len(vListImages) - vunbatched) / batch_size)
        valid_loss = 0
        ValACC = 0

        Net.eval()
        with torch.no_grad():
            valid_loss = 0
            ValACC = 0

            for vbatchNum in tqdm(range(vbatch_counts)):
                images, labels = LoadNext(vbatchNum, batch_size, ValidFolder)
                newVloss, batch_accuracy = validateOneClass(images, labels, batch_size, ValidateClass=ValidateClass)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()
                valid_loss = valid_loss + newVloss
                ValACC = ValACC + batch_accuracy

            if vunbatched > 0:
                images, labels = LoadNext(vbatch_counts + 1, vunbatched, ValidFolder)
                newVloss, batch_accuracy = validateOneClass(images, labels, vunbatched, ValidateClass=ValidateClass)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()
                valid_loss = valid_loss + newVloss
                ValACC = ValACC + batch_accuracy
                vruns = vbatch_counts + 1

        valid_loss = valid_loss / (vruns)  # averages the loss on all batches
        ValACC = ValACC / (vruns)

        print("checkpoint:", i, "  ", c, "  loss:", valid_loss)
        new_entry = pd.DataFrame([[i, c, valid_loss, ValACC]], columns=DBtitles)
        val_DB = pd.concat([val_DB, new_entry])
        val_DB.to_csv("./" + s + "/validation-" + Netname(Net) + "-" + dataset + ".csv", sep=",")
        i = i + 1

    print("DONE:  Saved as: ./" + s + "/validation-" + Netname(Net) + "-" + dataset + ".csv")


def SeedSensitivity(model, seedList, TrainFolder, ValidFolder, epochs, batchSize):
    for s in seedList:
        trainSet(model, TrainFolder, ValidFolder, epochs, batchSize)
    # report

    return


def trainingDetails(folder, train, batches, val, vbatches, test, tbatches, start, end, finished=False):
    global Net
    global seed
    global height
    global width
    global batch_size
    global scheduler
    global optimizer

    t = datetime.now()
    DateTime = str(t.hour) + str(t.minute) + "-" + str(t.day) + "-" + str(t.month) + "-" + str(t.year)

    f = open('./' + folder + '/training details ' + DateTime + '.txt', 'a')

    if finished == False:
        f.write('Date Time: ')
        f.write(DateTime)
        f.write('\n')
        f.write('Model Details: ')
        f.write(str(Net))
        f.write('\n')
        f.write('Seed: ' + str(seed) + '\n')
        f.write('Scheduler:' + str(scheduler) + '\n')
        f.write('Optimizer:' + str(optimizer) + '\n')
        f.write('Train: ' + str(train) + '  Batches: ' + str(batches) + '\n')
        f.write('Val: ' + str(val) + '  Batches: ' + str(vbatches) + '\n')
        f.write('Test: ' + str(test) + '  Batches: ' + str(tbatches) + '\n')
        f.write('Size of tiles: Height: ' + str(height) + '  Width: ' + str(width) + '\n')
        f.write('Batch Size: ' + str(batch_size) + '\n')
        f.write('Name of machine: ' + str(os.environ['COMPUTERNAME']) + '\n')
        f.write('Username: ' + str(os.environ.get('USERNAME')) + '\n')
        f.write('Start from epoch: ' + str(start) + '\n')
        f.write('\n')
    else:
        f.write('Til epoch: ' + str(end) + '\n')
        f.write('Finished successfully: True ')
        f.write('\n')
        f.write('\n')
        f.write('\n')

    f.close()

    del f


"""    
def trainFromPoint(Net, TrainFolder, ValidFolder,epochs, batch_size, checkpoint_path, model_naming_title=Netname(Net), width=256,height=256):

    ListImages=os.listdir(os.path.join(TrainFolder, "images")) # Create list of images        
    vListImages=os.listdir(os.path.join(ValidFolder, "images")) # Create list of validation images
    


"""


def GetLastLR(SchedulerName):
    global scheduler
    if SchedulerName == 'Plateau':
        last_lr = scheduler.state_dict()["_last_lr"][0]
    else:
        last_lr = scheduler.get_last_lr()[0]

    return last_lr


def OLDtrainStart(model, TrainFolder, ValidFolder, epochs, batchSize, TestFolder=None, Learning_Rate=1e-5):
    # log will not save test loss in this version
    global plottable_data
    global batch_size
    global Net
    global height
    global width
    global optimizer
    global logFilePath

    batch_size = batchSize
    vbatch_size = batchSize * 4
    ListImages = os.listdir(os.path.join(TrainFolder, "images"))  # Create list of images        
    vListImages = os.listdir(os.path.join(ValidFolder, "images"))  # Create list of validation images

    if TestFolder != None:
        tbatch_size = batchSize * 4
        tListImages = os.listdir(os.path.join(TestFolder, "images"))  # Create list of test images
        tunbatched = len(tListImages) % tbatch_size
        tbatch_counts = round((len(tListImages) - tunbatched) / tbatch_size)

    # autodetect height and width of training tiles
    tempImage = cv2.imread(os.path.join(TrainFolder, "images", ListImages[0]), cv2.IMREAD_COLOR)
    height = tempImage.shape[0]
    width = tempImage.shape[1]
    del tempImage

    # load model 
    Net = model  # Load net
    Net = Net.to(device)
    model_naming_title = Netname(Net)

    t = datetime.now()
    DateTime = str(t.hour) + str(t.minute) + "-" + str(t.day) + "-" + str(t.month) + "-" + str(t.year)

    path = model_naming_title + "-" + DateTime
    del t
    del DateTime
    if not os.path.exists(path): os.makedirs(path)

    log_path = 'LOG for ' + model_naming_title + '.csv'
    logFilePath = log_path

    if not os.path.exists(log_path):
        log_path2 = "./" + path + '/LOG for ' + path + '.csv'
        log_titles = ['Epoch', 'Train-Loss', 'Val-Loss', 'Acc', 'Learn-Rate', 'Session', 'CheckPoint']
        log_DB = pd.DataFrame(columns=log_titles)
        log_DB2 = pd.DataFrame(columns=log_titles)
        model_naming_title = "-" + model_naming_title + ".torch"

        print(" Folder for checkpoints:'", path, "' was created")
        print(" Training Log File:'", log_path, "' will be created... Starting training from scratch")
        Learning_Rate
        best_loss = 1
        EpochsStartFrom = 0  # in case training is restarted from a previously saved epoch, this continues the sequence
        # and prevents over-writing models and logs in the loss database 

        optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)  # Create adam optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        unbatched = len(ListImages) % batch_size
        batch_counts = round((len(ListImages) - unbatched) / batch_size)
        vunbatched = len(vListImages) % vbatch_size
        vbatch_counts = round((len(vListImages) - vunbatched) / vbatch_size)
        valid_loss = 0
        ValACC = 0

    else:
        print("Log file for ", model_naming_title, " already present as: ", log_path)
        print("Training will not start, to prevent overwriting")
        print(" ")
        print("If you really want to start from scratch, please move the existent log file")
        print("If you want to continue training please use the trainfromLast function instead")
        return

    # _________________________PREPERATION DONE_________________#
    # _________________________TRAINING LOOP STARTS FROM HERE_________________#
    with torch.autograd.set_detect_anomaly(False):
        gc.collect()
        cuda.empty_cache()

        for itr in range(epochs):  # Training loop
            start_time = datetime.now()
            train_loss = 0
            runs = batch_counts
            vruns = vbatch_counts

            for batchNum in tqdm(range(batch_counts)):
                images, labels = LoadNext(batchNum, batch_size, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            if unbatched > 0:
                images, labels = LoadNext(batch_counts + 1, unbatched, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                runs = batch_counts + 1
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            # uncomment if you want to train on a random batch too
            # images,labels=LoadBatch(TrainFolder) 
            # train_loss+=learn(images, labels)
            # runs=batch_counts+1

            train_loss = train_loss / (runs)  # +1) #averages the loss on all batches
            scheduler.step(train_loss)

            # BEGIN Validation 

            with torch.no_grad():
                valid_loss = 0
                ValACC = 0

                for vbatchNum in tqdm(range(vbatch_counts)):
                    images, labels = LoadNext(vbatchNum, vbatch_size, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy

                if vunbatched > 0:
                    images, labels = LoadNext(vbatch_counts + 1, vunbatched, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy
                    vruns = vbatch_counts + 1

                valid_loss = valid_loss / (vruns)  # averages the loss on all batches
                ValACC = ValACC / (vruns)
                # END   Validation

                test_loss = 0
                testACC = 0
                if TestFolder != None:
                    for tbatchNum in tqdm(range(tbatch_counts)):
                        images, labels = LoadNext(tbatchNum, tbatch_size, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, tbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy

                    if tunbatched > 0:
                        images, labels = LoadNext(tbatch_counts + 1, tunbatched, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, tbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy
                        truns = tbatch_counts + 1

                    test_loss = test_loss / (truns)  # averages the loss on all batches
                    testACC = testACC / (truns)
                # END Test                

            duration = datetime.now() - start_time

            if train_loss <= best_loss:
                best_loss = copy.deepcopy(train_loss)

            t = datetime.now()
            checkpoint = path + "/" + str(itr + EpochsStartFrom) + "-" + str(t.hour) + str(
                t.minute) + model_naming_title
            print("Saving Model: ", "./" + checkpoint)
            torch.save(Net.state_dict(), "./" + checkpoint)

            """
            else:
                if itr!=epochs-1:  checkpoint="not saved"
                else:
                    checkpoint=path+"/"+str(itr+EpochsStartFrom)+"-LAST EPOCH-"+model_naming_title
                    torch.save(Net.state_dict(),"./"+checkpoint)
                    print(" Saving LAST EPOCH: ./",checkpoint)
            """
            print(itr + EpochsStartFrom, "=> TrainLoss=", train_loss, "  ValLoss=", valid_loss, "ACC=", ValACC, "lr:",
                  GetLastLR(SchedulerName), " Time:", duration.seconds)
            new_log_entry = pd.DataFrame([[itr + EpochsStartFrom, train_loss, valid_loss, ValACC,
                                           float(scheduler.state_dict()["_last_lr"][0]), path, checkpoint]],
                                         columns=log_titles)
            log_DB = pd.concat([log_DB, new_log_entry])
            log_DB.to_csv(log_path, sep=",")
            log_DB2 = pd.concat([log_DB2, new_log_entry])
            log_DB2.to_csv(log_path2, sep=",")

    # _________________________TRAINING LOOP ENDS HERE_________________#

    print("____FINISHED Training______")
    return log_DB


def getScoresFast(Pred):
    scores = []
    batchsize = Pred.shape[0]
    batchsizescores = []
    for i in range(batchsize):
        batchsizescores.append(0)
    # base case
    if Pred.shape[0] == 1:
        a = int(torch.count_nonzero(torch.argmax(Pred[0], dim=0)))
        scores.append(a)
    else:
        # recursive case
        a = int(torch.count_nonzero(torch.argmax(Pred, dim=1)))
        if a != 0:
            scores.extend(getScoresFast(Pred[0:int(Pred.shape[0] / 2)]))
            scores.extend(getScoresFast(Pred[int(Pred.shape[0] / 2):int(Pred.shape[0])]))
        else:
            scores.extend(batchsizescores)

    return scores


def BgToTrainOn(UnlabelledImages, amountInSample, path, itr, folder, IncorrectPixelsAcceptedInTile=1):
    # instead of the backGround tiles being picked randomly, we pick the ones that predict the worst. 
    #    # First we run an accuracy test on each one and save a score

    print("Evaluating background sample")
    global Net
    batchSize = 64
    scores = []
    batchsizescores = []
    for i in range(batchSize):
        batchsizescores.append(0)
    Net.eval()
    count = 0
    images = torch.zeros([batchSize, 3, height, width])

    for item in tqdm(UnlabelledImages):
        Img = cv2.imread(os.path.join(folder, "images", item), cv2.IMREAD_COLOR)[:, :, 0:3]
        Img = transformImg(Img)
        images[count] = Img
        count = count + 1
        if count == batchSize:
            count = 0
            images = torch.autograd.Variable(images, requires_grad=False).to(device)

            with torch.no_grad(): Pred = Net(images)  # ['out'] # make prediction
            scores.extend(getScoresFast(Pred))

            images = torch.zeros([batchSize, 3, height, width])
            del Pred
            gc.collect()
            cuda.empty_cache()

    # for remnant images that don't fit in a batch we do check scores individually
    images = torch.autograd.Variable(images, requires_grad=False).to(device)
    with torch.no_grad():
        Pred = Net(images)  # ['out'] # make prediction
    a = int(torch.count_nonzero(torch.argmax(Pred, dim=1)))
    for i in range(count):
        a = int(torch.count_nonzero(torch.argmax(Pred[i], dim=0)))
        scores.append(a)
    del Pred
    gc.collect()
    cuda.empty_cache()

    print("Re-adjusting background sample...", end="")
    BGImageDataset = {'image': UnlabelledImages, 'misses': scores}
    BGImageDataFrame = pd.DataFrame(BGImageDataset)
    result = BGImageDataFrame[BGImageDataFrame.misses > IncorrectPixelsAcceptedInTile].sort_values(by=['misses'],
                                                                                                   ascending=False)
    result.to_excel("./" + path + '/BGtiles stats at epoch ' + str(itr) + '.xlsx')
    if amountInSample > result.shape[0]:
        WorstBgTiles = result['image'].tolist()
    else:
        WorstBgTiles = result.head(amountInSample)['image'].tolist()
    print("DONE and saved at ./" + path + '/BGtiles stats at epoch ' + str(itr) + '.xlsx')
    return WorstBgTiles


def inTestBGat(itr, TestBGat):
    answer = False
    for i in TestBGat:
        if itr == i: answer = True
    return answer


def trainStartBG(model, TrainFolder, ValidFolder, epochs, batchSize, TestFolder=None, Learning_Rate=1e-5, logName=None,
                 Scheduler_Patience=3, percentagOfUnlabelledTiles=0.075,
                 TestBGat=[0, 8, 24, 40, 56, 88, 120, 152, 184, 216, 48, 280], IncorrectPixelsAcceptedInTile=1,
                 ReAdjustUnlabbeledTilesList=False):
    if percentagOfUnlabelledTiles > 1 or percentagOfUnlabelledTiles < 0:
        print("percentagOfUnlabelledTiles must be between 0 and 1.  Training cannot start.")
        return

    global batch_size
    batch_size = batchSize
    vListImages = os.listdir(os.path.join(ValidFolder, "images"))  # Create list of validation images
    if len(vListImages) < (batchSize * 4):
        vbatch_size = batchSize * 4
    else:
        vbatch_size = batchSize
    vunbatched = len(vListImages) % vbatch_size
    vbatch_counts = round((len(vListImages) - vunbatched) / vbatch_size)

    if TestFolder != None:
        tListImages = os.listdir(os.path.join(TestFolder, "images"))  # Create list of test images
        if len(tListImages) < (batchSize * 4):
            tbatch_size = batchSize * 4
        else:
            tbatch_size = batchSize
        tunbatched = len(tListImages) % tbatch_size
        tbatch_counts = round((len(tListImages) - tunbatched) / tbatch_size)
    else:
        tbatch_counts = 0

    global Net
    global scheduler
    global optimizer
    # load model 
    Net = model  # Load net
    Net = Net.to(device)
    model_naming_title = Netname(Net)
    optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)  # Create adam optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=Scheduler_Patience)
    SchedulerName = 'Plateau'

    global logFilePath
    global plottable_data

    t = datetime.now()
    DateTime = str(t.hour) + str(t.minute) + "-" + str(t.day) + "-" + str(t.month) + "-" + str(t.year)
    path = 'Models/' + model_naming_title + "-" + DateTime
    del t

    if not os.path.exists(path): os.makedirs(path)

    if logName == None:
        log_path = 'LOG for MC-' + model_naming_title + '.csv'
    else:
        log_path = logName

    if not os.path.exists(log_path):
        log_path2 = "./" + path + '/LOG for MC-' + model_naming_title + "-" + DateTime + '.csv'
        log_titles = ['Epoch', 'Train-Loss', 'Val-Loss', 'Val-Acc', 'Test-Loss', 'Test-Acc', 'Time', 'Learn-Rate',
                      'Session', 'CheckPoint']
        log_DB = pd.DataFrame(columns=log_titles)
        log_DB2 = pd.DataFrame(columns=log_titles)
        model_naming_title = "-" + model_naming_title + ".torch"

        print(" Folder for checkpoints:'", path, "' was created")
        print(" Training Log File:'", log_path, "' will be created... Starting training from scratch")
        best_loss = 1
        EpochsStartFrom = 0  # in case training is restarted from a previously saved epoch, this continues the sequence
    # and prevents over-writing models and logs in the loss database
    else:
        print("Log file for ", model_naming_title, " already present as: ", log_path)
        print("Training will not start, to prevent overwriting")
        print(" ")
        print("If you really want to start from scratch, please move the existent log file")
        print("If you want to continue training please use the trainfromLast function instead")
        return

    del DateTime

    # for list of Images, ListOfLabelledImages and UnlabelledImages
    ListImages = os.listdir(os.path.join(TrainFolder, "images"))  # Create list of images
    ListOfLabelledImages = os.listdir(os.path.join(TrainFolder, "labels"))
    UnlabelledImages = copy.deepcopy(ListImages)

    global height
    global width
    # autodetect width and height of tiles
    tempImage = cv2.imread(os.path.join(TrainFolder, "images", ListImages[0]), cv2.IMREAD_COLOR)
    height = tempImage.shape[0]
    width = tempImage.shape[1]
    del tempImage

    for item in ListOfLabelledImages:
        UnlabelledImages.remove(item)

    amountInSample = int(len(UnlabelledImages) * percentagOfUnlabelledTiles)
    unbatched = (len(ListOfLabelledImages) + amountInSample) % batch_size
    batch_counts = round((len(ListOfLabelledImages) + amountInSample - unbatched) / batch_size)

    trainingDetails(path, TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder), tbatch_counts, '0',
                    epochs - 1, finished=False)
    # _________________________PREPERATION DONE_________________#

    BGSampleReadjusted = False

    # _________________________TRAINING LOOP STARTS FROM HERE_________________#

    with torch.autograd.set_detect_anomaly(False):
        gc.collect()
        cuda.empty_cache()
        for itr in range(epochs):  # Training loop
            start_time = datetime.now()
            train_loss = 0
            vruns = vbatch_counts

            if inTestBGat(itr + EpochsStartFrom, TestBGat):
                sampleOfUnlabelledImages = BgToTrainOn(UnlabelledImages, amountInSample, path, itr + EpochsStartFrom,
                                                       TrainFolder, IncorrectPixelsAcceptedInTile=1)
                BGSampleSize = len(sampleOfUnlabelledImages)
                unbatched = (len(ListOfLabelledImages) + BGSampleSize) % batch_size
                batch_counts = round((len(ListOfLabelledImages) + BGSampleSize - unbatched) / batch_size)
                BGSampleReadjusted = ReAdjustUnlabbeledTilesList  # boolen showing if smple of bg tiles has been re-adjusted
            else:
                if BGSampleReadjusted == False:
                    random.shuffle(UnlabelledImages)
                    sampleOfUnlabelledImages = UnlabelledImages[0:amountInSample]
                    unbatched = (len(ListOfLabelledImages) + amountInSample) % batch_size
                    batch_counts = round((len(ListOfLabelledImages) + amountInSample - unbatched) / batch_size)

            TrainSample = copy.deepcopy(ListOfLabelledImages)
            TrainSample.extend(sampleOfUnlabelledImages)
            random.shuffle(TrainSample)

            runs = batch_counts

            Net.train()
            for count in tqdm(range(batch_counts)):
                BatchOfImages = TrainSample[(count * batch_size):(count * batch_size) + batch_size - 1]
                # print((count*batch_size)," ",(count*batch_size)+batch_size-1)
                images, labels = LoadNextRandomBatch(BatchOfImages, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            if unbatched > 0:
                BatchOfImages = TrainSample[(batch_counts * batch_size):(batch_counts * batch_size) + unbatched - 1]
                images, labels = LoadNextRandomBatch(BatchOfImages, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                runs = batch_counts + 1
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            train_loss = train_loss / (runs)  # +1) #averages the loss on all batches
            scheduler.step(train_loss)

            del TrainSample
            # BEGIN Validation 
            Net.eval()
            with torch.no_grad():
                valid_loss = 0
                ValACC = 0

                for vbatchNum in tqdm(range(vbatch_counts)):
                    images, labels = LoadNext(vbatchNum, vbatch_size, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy

                if vunbatched > 0:
                    images, labels = LoadNext(vbatch_counts + 1, vunbatched, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy
                    vruns = vbatch_counts + 1

                valid_loss = valid_loss / (vruns)  # averages the loss on all batches
                ValACC = ValACC / (vruns)
                # END   Validation

                test_loss = 0
                testACC = 0
                if TestFolder != None:
                    truns = tbatch_counts
                    for tbatchNum in tqdm(range(tbatch_counts)):
                        images, labels = LoadNext(tbatchNum, tbatch_size, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy

                    if tunbatched > 0:
                        images, labels = LoadNext(tbatch_counts + 1, tunbatched, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy
                        truns = tbatch_counts + 1

                    test_loss = test_loss / (truns)  # averages the loss on all batches
                    testACC = testACC / (truns)
                # END Test                

            duration = datetime.now() - start_time

            if train_loss <= best_loss:
                best_loss = copy.deepcopy(train_loss)

            t = datetime.now()
            checkpoint = path + "/" + str(itr + EpochsStartFrom) + "-" + str(t.hour) + str(
                t.minute) + model_naming_title
            print("Saving Model: ", "./" + checkpoint)
            torch.save(Net.state_dict(), "./" + checkpoint)

            torch.save(optimizer.state_dict(), path + "/optimiser.optim")
            torch.save(scheduler, path + "/scheduler.sched")

            print(itr + EpochsStartFrom, "=> TrainLoss=", train_loss, "  ValLoss=", valid_loss, "  valACC=", ValACC,
                  " TestLoss=", test_loss, " testACC=", testACC, " lr:", GetLastLR(SchedulerName), " Time:",
                  duration.seconds)
            new_log_entry = pd.DataFrame([[itr + EpochsStartFrom, train_loss, valid_loss, ValACC, test_loss, testACC,
                                           duration.seconds, GetLastLR(SchedulerName), path, checkpoint]],
                                         columns=log_titles)
            log_DB = pd.concat([log_DB, new_log_entry])
            log_DB.to_csv(log_path, sep=",")
            log_DB2 = pd.concat([log_DB2, new_log_entry])
            log_DB2.to_csv(log_path2, sep=",")

    # _________________________TRAINING LOOP ENDS HERE_________________#
    trainingDetails(path, TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder), tbatch_counts, '0',
                    checkpoint, finished=True)
    plottable_data = log_DB
    print("____FINISHED Training______")
    return log_DB


def trainFromLastBG(model, TrainFolder, ValidFolder, epochs, batchSize, TestFolder=None, Learning_Rate=1e-5,
                    logName=None, Scheduler_Patience=3, percentagOfUnlabelledTiles=0.075,
                    TestBGat=[8, 24, 40, 56, 88, 120, 152, 184, 216, 248, 280], IncorrectPixelsAcceptedInTile=1,
                    ReAdjustUnlabbeledTilesList=False):
    if percentagOfUnlabelledTiles > 1 or percentagOfUnlabelledTiles < 0:
        print("percentagOfUnlabelledTiles must be between 0 and 1.  Training cannot start.")
        return

    global batch_size
    batch_size = batchSize
    vListImages = os.listdir(os.path.join(ValidFolder, "images"))  # Create list of validation images
    if len(vListImages) < (batchSize * 4):
        vbatch_size = batchSize * 4
    else:
        vbatch_size = batchSize
    vunbatched = len(vListImages) % vbatch_size
    vbatch_counts = round((len(vListImages) - vunbatched) / vbatch_size)

    if TestFolder != None:
        tListImages = os.listdir(os.path.join(TestFolder, "images"))  # Create list of test images
        if len(tListImages) < (batchSize * 4):
            tbatch_size = batchSize * 4
        else:
            tbatch_size = batchSize
        tunbatched = len(tListImages) % tbatch_size
        tbatch_counts = round((len(tListImages) - tunbatched) / tbatch_size)
    else:
        tbatch_counts = 0

    global Net
    global scheduler
    global optimizer
    # load model 
    Net = model  # Load net
    Net = Net.to(device)
    model_naming_title = Netname(Net)
    optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)  # Create adam optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=Scheduler_Patience)
    SchedulerName = 'Plateau'

    # _________________________PREPERATION ABout log files START_________________#
    global logFilePath
    global plottable_data
    model_naming_title = Netname(Net)
    log_path = 'LOG for MC-' + model_naming_title + '.csv'
    logFilePath = log_path
    log_titles = ['Epoch', 'Train-Loss', 'Val-Loss', 'Val-Acc', 'Test-Loss', 'Test-Acc', 'Time', 'Learn-Rate',
                  'Session', 'CheckPoint']
    log_DB = pd.DataFrame(columns=log_titles)

    if os.path.exists(log_path):
        print("A log file for MC", model_naming_title, " was found as: ", log_path)
        log_DB = pd.read_csv(log_path, sep=",", index_col=0)
        log_DB2 = pd.read_csv(log_path, sep=",", index_col=0)
        path = log_DB.tail(1)['Session']
        path = str(path[0])

        log_path2 = "./" + path + '/LOG for MC-' + path[7:] + '.csv'

        best_loss = log_DB['Train-Loss'].min()  # smallest loss value
        LastEpoch = int(log_DB.tail(1)['Epoch'])
        LastCheckpoint = log_DB.tail(1)['CheckPoint']
        if Learning_Rate == 0: Learning_Rate = float(log_DB.tail(1)['Learn-Rate'])  # the last learning rate logged
        EpochsStartFrom = LastEpoch + 1

        if os.path.exists(path):
            print("Folder for checkpoints: ", path, " was found")
        elif os.path.exists('Models/' + path):
            path = 'Models/' + path
            print("Folder for checkpoints: ", path, " was found")
        else:
            print("Folder for checkpoints: ", path, " or Models/" + path + " were not found.  Training cannot continue")
            del epochs
            print(
                " Please restore the folder and restart this notebook, or start training from scratch in appropriate notebook")
            return

        if os.path.exists("./" + LastCheckpoint[0]):
            checkpoint = "./" + LastCheckpoint[0]  # Path to trained model
            print("Training to continue from checkpoint:", checkpoint)
        elif os.path.exists("./Models/" + LastCheckpoint[0]):
            checkpoint = "./Models/" + LastCheckpoint[0]  # Path to trained model
            print("Training to continue from checkpoint:", checkpoint)
        else:
            print("Last Checkpoint was found at NEITHER ./" + LastCheckpoint[0] + " NOR at ./Models/" + LastCheckpoint[
                0] + "  .  Training cannot continue")
            # print(" Please specify a path to a saved checkpoint manually in the next cell")
            return

    else:
        print(" Training Log File:  '", log_path, "'  was not found...  ")
        print(
            " Either restore the log file, pass a different model_naming_title to trainFromLast() , or start training from scratch with trainStart() ")
        return

    model_naming_title = "-" + model_naming_title + ".torch"
    # _________________________PREPERATION ABout log files DONE_________________#

    # for list of Images, ListOfLabelledImages and UnlabelledImages
    ListImages = os.listdir(os.path.join(TrainFolder, "images"))  # Create list of images
    ListOfLabelledImages = os.listdir(os.path.join(TrainFolder, "labels"))
    UnlabelledImages = copy.deepcopy(ListImages)

    global height
    global width
    # autodetect width and height of tiles
    tempImage = cv2.imread(os.path.join(TrainFolder, "images", ListImages[0]), cv2.IMREAD_COLOR)
    height = tempImage.shape[0]
    width = tempImage.shape[1]
    del tempImage

    for item in ListOfLabelledImages:
        UnlabelledImages.remove(item)

    amountInSample = int(len(UnlabelledImages) * percentagOfUnlabelledTiles)
    unbatched = (len(ListOfLabelledImages) + amountInSample) % batch_size
    batch_counts = round((len(ListOfLabelledImages) + amountInSample - unbatched) / batch_size)

    trainingDetails(path, TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder), tbatch_counts, '0',
                    epochs - 1, finished=False)
    # _________________________PREPERATION DONE_________________#

    BGSampleReadjusted = False
    Net.load_state_dict(torch.load(checkpoint))

    # _________________________TRAINING LOOP STARTS FROM HERE_________________#

    with torch.autograd.set_detect_anomaly(False):
        gc.collect()
        cuda.empty_cache()
        for itr in range(epochs):  # Training loop
            start_time = datetime.now()
            train_loss = 0
            vruns = vbatch_counts

            if inTestBGat(itr + EpochsStartFrom, TestBGat):
                sampleOfUnlabelledImages = BgToTrainOn(UnlabelledImages, amountInSample, path, itr + EpochsStartFrom,
                                                       TrainFolder, IncorrectPixelsAcceptedInTile=1)
                BGSampleSize = len(sampleOfUnlabelledImages)
                unbatched = (len(ListOfLabelledImages) + BGSampleSize) % batch_size
                batch_counts = round((len(ListOfLabelledImages) + BGSampleSize - unbatched) / batch_size)
                BGSampleReadjusted = ReAdjustUnlabbeledTilesList  # boolen showing if smple of bg tiles has been re-adjusted
            else:
                if BGSampleReadjusted == False:
                    random.shuffle(UnlabelledImages)
                    sampleOfUnlabelledImages = UnlabelledImages[0:amountInSample]
                    unbatched = (len(ListOfLabelledImages) + amountInSample) % batch_size
                    batch_counts = round((len(ListOfLabelledImages) + amountInSample - unbatched) / batch_size)

            TrainSample = copy.deepcopy(ListOfLabelledImages)
            TrainSample.extend(sampleOfUnlabelledImages)
            random.shuffle(TrainSample)

            runs = batch_counts

            Net.train()
            for count in tqdm(range(batch_counts)):
                BatchOfImages = TrainSample[(count * batch_size):(count * batch_size) + batch_size - 1]
                # print((count*batch_size)," ",(count*batch_size)+batch_size-1)
                images, labels = LoadNextRandomBatch(BatchOfImages, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            if unbatched > 0:
                BatchOfImages = TrainSample[(batch_counts * batch_size):(batch_counts * batch_size) + unbatched - 1]
                images, labels = LoadNextRandomBatch(BatchOfImages, TrainFolder)
                train_loss = train_loss + learn(images, labels)
                runs = batch_counts + 1
                del images
                del labels
                gc.collect()
                cuda.empty_cache()

            train_loss = train_loss / (runs)  # +1) #averages the loss on all batches
            scheduler.step(train_loss)

            del TrainSample
            # BEGIN Validation 
            Net.eval()
            with torch.no_grad():
                valid_loss = 0
                ValACC = 0

                for vbatchNum in tqdm(range(vbatch_counts)):
                    images, labels = LoadNext(vbatchNum, vbatch_size, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy

                if vunbatched > 0:
                    images, labels = LoadNext(vbatch_counts + 1, vunbatched, ValidFolder)
                    newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                    del images
                    del labels
                    gc.collect()
                    cuda.empty_cache()
                    valid_loss = valid_loss + newVloss
                    ValACC = ValACC + batch_accuracy
                    vruns = vbatch_counts + 1

                valid_loss = valid_loss / (vruns)  # averages the loss on all batches
                ValACC = ValACC / (vruns)
                # END   Validation

                test_loss = 0
                testACC = 0
                if TestFolder != None:
                    truns = tbatch_counts
                    for tbatchNum in tqdm(range(tbatch_counts)):
                        images, labels = LoadNext(tbatchNum, tbatch_size, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy

                    if tunbatched > 0:
                        images, labels = LoadNext(tbatch_counts + 1, tunbatched, TestFolder)
                        newVloss, batch_accuracy = validate(images, labels, vbatch_size)
                        del images
                        del labels
                        gc.collect()
                        cuda.empty_cache()
                        test_loss = test_loss + newVloss
                        testACC = testACC + batch_accuracy
                        truns = tbatch_counts + 1

                    test_loss = test_loss / (truns)  # averages the loss on all batches
                    testACC = testACC / (truns)
                # END Test                

            duration = datetime.now() - start_time

            if train_loss <= best_loss:
                best_loss = copy.deepcopy(train_loss)

            t = datetime.now()
            checkpoint = path + "/" + str(itr + EpochsStartFrom) + "-" + str(t.hour) + str(
                t.minute) + model_naming_title
            print("Saving Model: ", "./" + checkpoint)
            torch.save(Net.state_dict(), "./" + checkpoint)

            torch.save(optimizer.state_dict(), path + "/optimiser.optim")
            torch.save(scheduler, path + "/scheduler.sched")

            print(itr + EpochsStartFrom, "=> TrainLoss=", train_loss, "  ValLoss=", valid_loss, "  valACC=", ValACC,
                  " TestLoss=", test_loss, " testACC=", testACC, " lr:", GetLastLR(SchedulerName), " Time:",
                  duration.seconds)
            new_log_entry = pd.DataFrame([[itr + EpochsStartFrom, train_loss, valid_loss, ValACC, test_loss, testACC,
                                           duration.seconds, GetLastLR(SchedulerName), path, checkpoint]],
                                         columns=log_titles)
            log_DB = pd.concat([log_DB, new_log_entry])
            log_DB.to_csv(log_path, sep=",")
            log_DB2 = pd.concat([log_DB2, new_log_entry])
            log_DB2.to_csv(log_path2, sep=",")

    # _________________________TRAINING LOOP ENDS HERE_________________#
    trainingDetails(path, TrainFolder, batch_counts, ValidFolder, vbatch_counts, str(TestFolder), tbatch_counts, '0',
                    checkpoint, finished=True)
    plottable_data = log_DB
    print("____FINISHED Training______")
    return log_DB


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

if __name__ == "__main__":
    set_seed()
    print(
        'Default loss function is DICE.  You can set the loss function using segmentationtraining.criterion=myLossFunction()')