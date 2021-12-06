# This is a sample Python script.
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
import os
import glob


'''
LAST UPDATED 11/10/2021, lsdr
'''

## Process images in parallel

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file

OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
sep = os.path.sep
os.chdir(OR_PATH) # Come back to the directory where the code resides , all files will be left on this directory



n_epoch = 10
BATCH_SIZE = 32
LR = 0.001

## Image processing
CHANNELS = 3         # Channel should remain 3 because color is a differentiator
IMAGE_SIZE = 400

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
SAVE_MODEL = True

torch.random.seed()
np.random.seed(0)

#---- Define the model ---- #

def augmentation_step(X, rotate=False):

    transform = transforms.RandomApply([transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip()], p=0.5)
    rotation = transforms.RandomApply([transforms.RandomRotation(degrees=(0, 270))], p=0.3)
    normalize = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    transforms.CenterCrop(300)])

    X = transform(X)
    if rotate:
        X = rotation(X)
    X = normalize(X)

    return X


class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, list_IDs,type_data,target_type, augmentation=False):
        #Initialization'
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.target_type = target_type
        self.augmentation = augmentation

    def __len__(self):
        #Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        #Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label

        if self.type_data == 'train':
            y = train_df.label.get(ID)
        else:
            y = val_df.label.get(ID)

        labels_ohe = [y]
        y = torch.FloatTensor(labels_ohe)

        if self.type_data == 'train':
            file = train_df.path.get(ID)
        else:
            file = val_df.path.get(ID)

        img = cv2.imread(file)

        img= cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))

        # Augmentation only for train
        X = torch.FloatTensor(img)

        X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))

        if self.augmentation:
            X = augmentation_step(X, rotate=True)
        else:
            transform_norm = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.CenterCrop(300)
            ])
            X = transform_norm(X)

        return X, y

def process_data(df,data_type,target_type, augmentation=False):
    # ---------------------- Parameters for the data loader --------------------------------
    # Data Loaders
    list_of_ids = list(df.index)
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True}

    data_set = Dataset(list_of_ids, data_type,target_type, augmentation=augmentation)
    data_ds = data.DataLoader(data_set, **params)

    return data_ds

def save_model(model):
    # Open the file

    print(model, file=open('summary.txt', "w"))

def model_definition():
    # Define a Keras sequential model
    # Compile the model

    model = models.efficientnet_b0(pretrained=True)
    model.classifier = nn.Linear(1280, OUTPUTS_a)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)

    save_model(model)

    return model, optimizer, criterion, #scheduler

def train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on):
    # Use a breakpoint in the code line below to debug your script.

    model, optimizer, criterion,  = model_definition()  #scheduler

    cont = 0
    train_loss_item = list([])
    test_loss_item = list([])

    pred_labels_per_hist = list([])

    model.phase = 0

    met_test_best = 0
    for epoch in range(n_epoch):
        train_loss, steps_train = 0, 0

        model.train()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        train_hist = list([])
        test_hist = list([])

        with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch)) as pbar:

            for xdata,xtarget in train_ds:

                xdata, xtarget = xdata.to(device), xtarget.to(device)

                optimizer.zero_grad()

                output = model(xdata)

                loss = criterion(output, xtarget)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                cont += 1

                steps_train += 1

                train_loss_item.append([epoch, loss.item()])

                pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                if len(pred_labels_per_hist) == 0:
                    pred_labels_per_hist = pred_labels_per
                else:
                    pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                if len(train_hist) == 0:
                    train_hist = xtarget.cpu().numpy()
                else:
                    train_hist = np.vstack([train_hist, xtarget.cpu().numpy()])

                pbar.update(1)
                pbar.set_postfix_str("Test Loss: {:.5f}".format(train_loss / steps_train))

                pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        # Metric Evaluation
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        avg_train_loss = train_loss / steps_train

        ## Finish with Training

        ## Testing the model

        model.eval()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        test_loss, steps_test = 0, 0
        met_test = 0

        with torch.no_grad():

            with tqdm(total=len(test_ds), desc="Epoch {}".format(epoch)) as pbar:

                for xdata,xtarget in test_ds:

                    xdata, xtarget = xdata.to(device), xtarget.to(device)

                    optimizer.zero_grad()

                    output = model(xdata)

                    loss = criterion(output, xtarget)

                    test_loss += loss.item()
                    cont += 1

                    steps_test += 1

                    test_loss_item.append([epoch, loss.item()])

                    pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                    if len(pred_labels_per_hist) == 0:
                        pred_labels_per_hist = pred_labels_per
                    else:
                        pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                    if len(test_hist) == 0:
                        tast_hist = xtarget.cpu().numpy()
                    else:
                        test_hist = np.vstack([test_hist, xtarget.cpu().numpy()])

                    pbar.update(1)
                    pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))

                    pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                    real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        #acc_test = accuracy_score(real_labels[1:], pred_labels)

        avg_test_loss = test_loss / steps_test

        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in train_metrics.items():
            xstrres = xstrres +' Train '+met+ ' {:.5f}'.format(dat)


        xstrres = xstrres + " - "
        for met, dat in test_metrics.items():
            xstrres = xstrres + ' Test '+met+ ' {:.5f}'.format(dat)
            if met == save_on:
                met_test = dat

        print(xstrres)

        if met_test > met_test_best and SAVE_MODEL:

            torch.save(model.state_dict(), "model.pt")
            xdf_dset_results = val_df.copy()

            ## The following code creates a string to be saved as 1,2,3,3,
            ## However, since this classification is binary only one output is made rather than 1,2,3,3
            ## This code will be used to validate the model
            xfinal_pred_labels = []
            for i in range(len(pred_labels)):
                joined_string = ",".join(str(int(e)) for e in pred_labels[i])
                xfinal_pred_labels.append(joined_string)

            xdf_dset_results['results'] = xfinal_pred_labels

            xdf_dset_results.to_excel('results.xlsx', index = False)
            print("The model has been saved!")
            met_test_best = met_test


def metrics_func(metrics, aggregates, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    xcont = 1
    xsum = 0
    xavg = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet =hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont +1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum/xcont
    # Ask for arguments for each metric

    return res_dict

#Load my train data
train_dataset_0_all = glob.glob(OR_PATH + '/training_data/fold_0/all/*.bmp')
train_dataset_0_hem = glob.glob(OR_PATH + '/training_data/fold_0/hem/*.bmp')
train_dataset_1_all = glob.glob(OR_PATH + '/training_data/fold_1/all/*.bmp')
train_dataset_1_hem = glob.glob(OR_PATH + '/training_data/fold_1/hem/*.bmp')
train_dataset_2_all = glob.glob(OR_PATH + '/training_data/fold_2/all/*.bmp')
train_dataset_2_hem = glob.glob(OR_PATH + '/training_data/fold_2/hem/*.bmp')
# Merging happens here:-
A=[]
H=[]
A.extend(train_dataset_0_all)
A.extend(train_dataset_1_all)
A.extend(train_dataset_2_all)

H.extend(train_dataset_0_hem)
H.extend(train_dataset_1_hem)
H.extend(train_dataset_2_hem)
H.extend(H)  # double the instances of the number of minority classes

# Create labels as list for dataframe :-
Label_A = [1]*len(A)
Label_H = [0]*len(H)


print('ALL Sample Count',len(A))
print('Hem Sample Count',len(H))
# Converting to pandas dataframe for easier access:-
A.extend(H)
Label_A.extend(Label_H)
train_df = pd.DataFrame({'path':A, 'label':Label_A})
train_df = train_df.sample(frac=1).reset_index(drop=True)  # Shuffle the data

# Include Validation data as well :-
valid_data = pd.read_csv(OR_PATH  + '/validation_data/C-NMC_test_prelim_phase_data_labels.csv')

av = valid_data[valid_data['labels'] == 1]
hv = valid_data[valid_data['labels'] == 0]

# list of paths to validation data images
VAL_PATH = OR_PATH + '/validation_data/C-NMC_test_prelim_phase_data/'
AVL = [VAL_PATH + i for i in list(av.new_names)]
HVL = [VAL_PATH + i for i in list(hv.new_names)]

# Create labels :-
Label_AVL = [1]*len(AVL)
Label_HVL = [0]*len(HVL)

# Converting to pandas dataframe for easier access:-
AVL.extend(HVL)
Label_AVL.extend(Label_HVL)
val_df = pd.DataFrame({'path':AVL, 'label':Label_AVL})
val_df  = val_df .sample(frac=1).reset_index(drop=True) # Shuffle the data


OUTPUTS_a = 1

list_of_metrics = ['acc', 'f1_weighted', 'coh']
list_of_agg = ['avg']

train_ds = process_data(train_df, 'train', target_type = 1, augmentation=True)

val_ds = process_data(val_df, 'val', target_type = 1)

train_and_test(train_ds, val_ds, list_of_metrics, list_of_agg, save_on='acc')


# Augmentation = True; efficientnet_b0
#