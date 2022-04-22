# ## Imports

import os
import time
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps

from tqdm import tqdm, tqdm_notebook

import torch
from torch import nn, cuda
from torch.autograd import Variable 
import torch.nn.functional as F
import torchvision as vision
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, ReduceLROnPlateau

from sklearn.metrics import f1_score

from Test_dataset import TestDataset
from CIFAR10Policy import CIFAR10Policy
from SubPolicy import SubPolicy

torch.cuda.set_device(0)


# ## CosineAnnealing

class CosineAnnealingWithRestartsLR(_LRScheduler):
    
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=1, restart_decay=0.95):
        self.T_max = T_max
        self.T_mult = T_mult
        self.next_restart = T_max
        self.eta_min = eta_min
        self.restarts = 0
        self.last_restart = 0
        self.T_num = 0
        self.restart_decay = restart_decay
        super(CosineAnnealingWithRestartsLR,self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.Tcur = self.last_epoch - self.last_restart
        if self.Tcur >= self.next_restart:
            self.next_restart *= self.T_mult
            self.last_restart = self.last_epoch
            self.T_num += 1
        learning_rate = [(self.eta_min + ((base_lr)*self.restart_decay**self.T_num - self.eta_min) * (1 + math.cos(math.pi * self.Tcur / self.next_restart)) / 2) for base_lr in self.base_lrs]
        return learning_rate



# seed value fix
# The seed value must be fixed so that results can be compared each time the hyper parameter is changed.
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 2019
seed_everything(SEED)




use_cuda = cuda.is_available()
use_cuda


# ## Dataset


class TrainDataset(Dataset):
    def __init__(self, df, mode='train', transforms=None):
        self.df = df
        self.mode = mode
        self.transform = transforms[self.mode]
        
    def __len__(self):
        return len(self.df)
            
    def __getitem__(self, idx):
        
        image = Image.open(TRAIN_IMAGE_PATH / self.df['img_file'][idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.df['class'][idx]

        return image, label





target_size = (224, 224)

data_transforms = {
    'train': vision.transforms.Compose([
        vision.transforms.Resize(target_size),
        vision.transforms.RandomHorizontalFlip(),
        vision.transforms.RandomRotation(20),
        CIFAR10Policy(),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
    ]),
    'valid': vision.transforms.Compose([
        vision.transforms.Resize(target_size),
        vision.transforms.RandomResizedCrop(target_size, scale=(0.8,1.0)),
        vision.transforms.RandomHorizontalFlip(),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
    ]),
    'test': vision.transforms.Compose([
        vision.transforms.Resize((224,224)),
        vision.transforms.RandomResizedCrop(target_size, scale=(0.8,1.0)),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
    ]),
}




TRAIN_IMAGE_PATH = Path('train_images_crop_224_224/1/')



df = pd.read_csv("image_5folds.csv")

df.head()


len(df[df['fold'] == 0]), len(df[df['fold'] == 1]), len(df[df['fold'] == 2]), len(df[df['fold'] == 3]), len(df[df['fold'] == 4])



fold_num = 3 #change


train_df = df.loc[df['fold'] != fold_num]
valid_df = df.loc[df['fold'] == fold_num]



train_df = train_df[['img_file', 'class']].reset_index(drop=True)
valid_df = valid_df[['img_file', 'class']].reset_index(drop=True)



num_classes = train_df['class'].nunique()
y_true = valid_df['class'].values # for cv score



print("number of train dataset: {}".format(len(train_df)))
print("number of valid dataset: {}".format(len(valid_df)))
print("number of classes to predict: {}".format(num_classes))


# ## Training



def train_one_epoch(model, criterion, train_loader, optimizer, accumulation_step=2):
    
    model.train()
    train_loss = 0.
    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(train_loader):
            
        inputs, targets = inputs.cuda(), targets.cuda()

        

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        
        if accumulation_step:
            if (i+1) % accumulation_step == 0:  
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()
        

        train_loss += loss.item() / len(train_loader)
        
    return train_loss


def validation(model, criterion, valid_loader):
    
    model.eval()
    valid_preds = np.zeros((len(valid_dataset), num_classes))
    val_loss = 0.
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(valid_loader):

            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs).detach()
            loss = criterion(outputs, targets)
            valid_preds[i * batch_size: (i+1) * batch_size] = outputs.cpu().numpy()
            
            val_loss += loss.item() / len(valid_loader)
            
        y_pred = np.argmax(valid_preds, axis=1)
        val_score = f1_score(y_true, y_pred, average='micro')  
        
    return val_loss, val_score    



def pick_best_score(result1, result2):
    if result1['best_score'] < result2['best_score']:
        return result2
    else:
        return result1
    
def pick_best_loss(result1, result2):
    if result1['best_loss'] < result2['best_loss']:
        return result1
    else:
        return result2



def train_model(num_epochs=60, accumulation_step=4, cv_checkpoint=False, fine_tune=False, weight_file_name='best_model_resnet152_fold{}.pt'.format(fold_num), **train_kwargs):
    
    # choose scheduler
    if fine_tune:
        lr = 0.00001
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
    else:    
        lr = 0.01
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.025)
        eta_min = 1e-6
        T_max = 10
        T_mult = 1
        restart_decay = 0.97
        scheduler = CosineAnnealingWithRestartsLR(optimizer,T_max=T_max, eta_min=eta_min, T_mult=T_mult, restart_decay=restart_decay)

    train_result = {}
    train_result['weight_file_name'] = weight_file_name
    best_epoch = -1
    best_score = 0.
    lrs = []
    score = []
    
    for epoch in range(num_epochs):
        
        start_time = time.time()

        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, accumulation_step)
        val_loss, val_score = validation(model, criterion, valid_loader)
        score.append(val_score)
    
        # model save (score or loss?)
        if cv_checkpoint:
            if val_score > best_score:
                best_score = val_score
                train_result['best_epoch'] = epoch + 1
                train_result['best_score'] = round(best_score, 5)
                torch.save(model.state_dict(), weight_file_name)
        else:
            if val_loss < best_loss:
                best_loss = val_loss
                train_result['best_epoch'] = epoch + 1
                train_result['best_loss'] = round(best_loss, 5)
                torch.save(model.state_dict(), weight_file_name)
        
        elapsed = time.time() - start_time
        
        lr = [_['lr'] for _ in optimizer.param_groups]
        print("Epoch {} - train_loss: {:.4f}  val_loss: {:.4f}  cv_score: {:.4f}  lr: {:.6f}  time: {:.0f}s".format(
                epoch+1, train_loss, val_loss, val_score, lr[0], elapsed))
        
        for param_group in optimizer.param_groups:
            lrs.append(param_group['lr'])
        
        # scheduler update
        if fine_tune:
            if cv_checkpoint:
                scheduler.step(val_score)
            else:
                scheduler.step(val_loss)
        else:
            scheduler.step()
     
    return train_result, lrs, score



batch_size = 32

train_dataset = TrainDataset(train_df, mode='train', transforms=data_transforms)
valid_dataset = TrainDataset(valid_df, mode='valid', transforms=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# ## Model

model = models.resnet152(pretrained=True)
model.fc = nn.Linear(2048, num_classes)
model.cuda()




criterion = nn.CrossEntropyLoss()

train_kwargs = dict(
    train_loader=train_loader,
    valid_loader=valid_loader,
    model=model,
    criterion=criterion,
    )


print("training starts")
num_epochs = 100
result, lrs, score = train_model(num_epochs=num_epochs, accumulation_step=2, cv_checkpoint=True, fine_tune=False, weight_file_name='best_model_resnet152_fold{}.pt'.format(fold_num), **train_kwargs)
print(result)


# ## Learning rate plot

plt.figure(figsize=(18,4))
plt.subplot(1,2,1)
plt.plot(lrs, 'b')
plt.xlabel('Epochs', fontsize=12, fontweight='bold')
plt.ylabel('Learning rate', fontsize=14, fontweight='bold')
plt.title('Learning rate schedule', fontsize=15, fontweight='bold')

x = [x for x in range(0, num_epochs, 10)]
y = [0.01, 0.005, 0.000001]
ylabel = ['1e-2', '1e-4', '1e-6']
plt.xticks(x)
plt.yticks(y, ylabel)

plt.subplot(1,2,2)
plt.plot(score, 'r')
plt.xlabel('Epochs', fontsize=12, fontweight='bold')
plt.ylabel('Valid score', fontsize=14, fontweight='bold')
plt.title('F1 Score', fontsize=15, fontweight='bold')

x = [x for x in range(0, num_epochs, 10)]

plt.show()
plt.savefig("/home/haykim/dataset/output_resnet152/fold{}_lr_f1score_plot.png".format(fold_num)) #change

