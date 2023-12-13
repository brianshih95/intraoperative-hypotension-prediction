# Deep Learning for Prediction of Intraoperative Hypotension by Analysing Biosignal Waveforms Acquired from Patient Monitoring

# Code for GradCAM for model explanation

# Author: Yu Seong Chu, BSc (dbtjd9968@yonsei.ac.kr) and Solam Lee, MD (solam@yonsei.ac.kr)

# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import pickle, os, warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from random import randint

import cv2
from torch.autograd import Function
from scipy.interpolate import interp1d

import re

warnings.filterwarnings("ignore")

model_path = './model/44009/pt/classification_invasive_multi_5min_epoch_best.pt' # Path for pretrained model
match = re.search(r'(\d+)', model_path)
if match:
    random_key = int(match.group(1))
else:
    print("Wrong model path!!!")
    exit(1)

# Prespecifications
task = 'classification' # either 'classification' or 'regression'
invasive = True # either True or False
multi = True # either True or False
pred_lag = 300 # 300 for 5-min, 600 for 10-min, 900 for 15-min prediction or others

cuda_number = 0 # -1 for multi GPU support
num_workers = 0
batch_size = 64
max_epoch = 100

train_ratio = 0.6 # Size for training dataset
valid_ratio = 0.1 # Size for validation dataset
test_ratio = 0.3 # Size for test dataset

dr_classification = 0.3 # Drop out ratio for classification model
dr_regression = 0.0 # Drop out ratio for regression model

# Establish dataset
class dnn_dataset(torch.utils.data.Dataset):
    def __init__(self, abp, ecg, ple, co2, target, invasive, multi):
        self.invasive, self.multi = invasive, multi
        self.abp, self.ecg, self.ple, self.co2 = abp, ecg, ple, co2
        self.target = target
        
    def __getitem__(self, index):
        if self.invasive == True:
            if self.multi == True: # Invasive multi-channel model
                return np.float32( np.vstack (( np.array ( self.abp[index] ),
                                                np.array ( self.ecg[index] ),
                                                np.array ( self.ple[index] ),
                                                np.array ( self.co2[index] ) ) ) ), np.float32(self.target[index])
            else: # Invasive mono-channel model (arterial pressure-only model)
                return np.float32( np.array ( self.abp[index] ) ), np.float32(self.target[index])       
        else:
            if self.multi == True: # Non-invasive multi-channel model
                return np.float32( np.vstack (( np.array ( self.ecg[index] ),
                                                np.array ( self.ple[index] ),
                                                np.array ( self.co2[index] ) ) ) ), np.float32(self.target[index])
            else: # Non-invasive mono-channel model (photoplethysmography-only model)
                return np.float32( np.array ( self.ple[index] ) ), np.float32(self.target[index])

    def __len__(self):
        return len(self.target)


# Uni-dimensional convolutional neural network

class Net(nn.Module):
    def __init__(self, task, invasive, multi):
    
        self.task, self.invasive, self.multi = task, invasive, multi
        super(Net, self).__init__()

        if self.task == 'classification':
            self.dr = dr_classification
            self.final = 2
        else:
            self.dr = dr_regression
            self.final = 1

        if self.multi == True:
            self.inc = 4 if self.invasive == True else 3
        else:
            self.inc = 1
                        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.inc, out_channels=64, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(320, self.final),
            nn.Dropout(self.dr)
        )
        
        self.activation = nn.Sigmoid()

    
    def forward(self, x):
        
        x = x.view(x.shape[0], self.inc, -1)
    
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        
        out = out.view(x.shape[0], out.size(1) * out.size(2))

        out = self.fc(out)
        if self.task == 'classification':
            out = self.activation(out)
        
        return out


# Read dataset

processed_dir = './processed/'

file_list = np.char.split ( np.array ( os.listdir(processed_dir) ), '.' )
case_list = []
for caseid in file_list:
    case_list.append ( int ( caseid[0] ) )
print ( 'N of total cases: {}'.format ( len ( case_list ) ) )

cases = {}
cases['train'], cases['valid+test'] = train_test_split ( case_list,
                                                        test_size=(valid_ratio+test_ratio),
                                                        random_state=random_key )
cases['valid'], cases['test'] = train_test_split ( cases['valid+test'],
                                                  test_size=(test_ratio/(valid_ratio+test_ratio)),
                                                  random_state=random_key )

for phase in [ 'train', 'valid', 'test' ]:
    print ( "- N of {} cases: {}".format(phase, len(cases[phase])) )

for idx, caseid in enumerate(case_list):
    filename = processed_dir + str ( caseid ) + '.pkl'
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        data['caseid'] = [ caseid ] * len ( data['abp'] )
        raw_records = raw_records.append ( pd.DataFrame ( data ) ) if idx > 0 else pd.DataFrame ( data )
        
raw_records = raw_records[(raw_records['map']>=20)&(raw_records['map']<=160)].reset_index(drop=True) # Exclude abnormal range



# Define loader and model

if task == 'classification':
    task_target = 'hypo'
    criterion = nn.BCELoss()
else:
    task_target = 'map'
    criterion = nn.MSELoss()

print ( '\n===== Task: {}, Seed: {} =====\n'.format ( task, random_key ) )
print ( 'Invasive: {}\nMulti: {}\nPred lag: {}\n'.format ( invasive, multi, pred_lag ))

records = raw_records.loc[ ( raw_records['input_length']==30 ) &
                            ( raw_records['pred_lag']==pred_lag ) ]

records = records [ records.columns.tolist()[-1:] + records.columns.tolist()[:-1] ]
print ( 'N of total records: {}'.format ( len ( records ) ))

split_records = {}
for phase in ['train', 'valid', 'test']:
    split_records[phase] = records[records['caseid'].isin(cases[phase])].reset_index(drop=True)
    print ('- N of {} records: {}'.format ( phase, len ( split_records[phase] )))

print ( '' )

ext = {}
for phase in [ 'train', 'valid', 'test' ]:
    ext[phase] = {}
    for x in [ 'abp', 'ecg', 'ple', 'co2', 'hypo', 'map' ]:
        ext[phase][x] = split_records[phase][x]

dataset, loader = {}, {}
epoch_loss, epoch_auc = {}, {}

for phase in [ 'train', 'valid', 'test' ]:
    dataset[phase] = dnn_dataset ( ext[phase]['abp'],
                                    ext[phase]['ecg'],
                                    ext[phase]['ple'],
                                    ext[phase]['co2'],
                                    ext[phase][task_target],
                                    invasive = invasive, multi = multi )
    loader[phase] = torch.utils.data.DataLoader(dataset[phase],
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle = True if phase == 'train' else False )
    epoch_loss[phase], epoch_auc[phase] = [], []


torch.cuda.set_device(cuda_number)
DNN = Net( task = task, invasive = invasive, multi = multi )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DNN = DNN.to(device)

# GradCAM

class FeatureExtractor():

    def __init__(self, model):
        self.model = model
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name == '2':
                x.register_hook(self.save_gradient)
                outputs += [x]
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        return outputs, x


class ModelOutputs():

    def __init__(self, model, feature_module):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        
        return target_activations, x


def interpolation(mask):
    x = np.linspace(0,5,21)
    y = mask
    fq = interp1d(x,y,kind = 'quadratic')

    xint = np.linspace(x.min(), x.max(), 3000)
    yintq = fq(xint)
    yintq = yintq - np.min(yintq)
    yintq = yintq / (np.max(yintq) +0.00001)
    return yintq

class GradCam:
    def __init__(self, model, feature_module, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        
        one_hot[0][index] = 1
        
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        
        one_hot.backward(retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)


        for i, w in enumerate(weights):
            cam += w * target[i, :]
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) +0.00001)
        return cam


DNN.load_state_dict(torch.load(model_path))
grad_cam = GradCam(model=DNN, feature_module=DNN.conv7, use_cuda=True)


index = 0
for dnn_inputs, dnn_target in loader['train']:
    DNN.eval()
    for i in range(len(dnn_inputs)):
        target_index = dnn_target[i]
        grad_cam_input = dnn_inputs[i].unsqueeze(0).cuda()
        output = DNN(grad_cam_input).cpu()
        if output[0][0] > 0.5:
            predicted_index = 1
        else:
            predicted_index = 0

        index = index + 1
        mask = grad_cam(grad_cam_input, 0)
        print(mask.shape)
        if target_index == predicted_index:
            if mask.sum() == 0:
                pass

            else:
                interpolted_mask = interpolation(mask)
                print(interpolted_mask.shape)
                a = np.append(np.expand_dims(interpolted_mask, axis=0), np.expand_dims(interpolted_mask, axis=0), axis=0)
                for i in range(50):
                    a = np.append(a,np.expand_dims(interpolted_mask, axis=0),axis = 0)
                heatmap = cv2.applyColorMap(np.uint8(255 * a), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                f, axes = plt.subplots(5, 1, figsize=(10, 12))
                
                x = np.arange(0,30,0.01)
                y1  = np.array(grad_cam_input[0][0].cpu())
                print(y1.shape)
                y2  = np.array(grad_cam_input[0][1].cpu())
                y3  = np.array(grad_cam_input[0][2].cpu())
                y4  = np.array(grad_cam_input[0][3].cpu())

                axes[0].imshow(heatmap)
                axes[0].axis('off')
                axes[0].set_title("heatmap")

                axes[1].plot(x, y1)
                axes[1].set_title("1ch input")
                axes[1].set_xlim([0, 30])

                axes[2].plot(x, y2)
                axes[2].set_title("2ch input")
                axes[2].set_xlim([0, 30])

                axes[3].plot(x, y3)
                axes[3].set_title("3ch input")
                axes[3].set_xlim([0, 30])

                axes[4].plot(x, y4)
                axes[4].set_title("4ch input")
                axes[4].set_xlim([0, 30])


                f.tight_layout()
                heatmap_dir = './model/'+str(random_key)+'/heatmap/'
                if not ( os.path.isdir( heatmap_dir ) ):
                    os.makedirs ( os.path.join ( heatmap_dir ) )
                train_dir = heatmap_dir + 'Train/'
                if not ( os.path.isdir( train_dir ) ):
                    os.makedirs ( os.path.join ( train_dir ) )
                plt.savefig(train_dir+str(index)+'.jpg')
        else:
            pass