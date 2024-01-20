import os
import pdb
import cv2
import sys
import time
import torch
import random
import shutil
import pickle
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import albumentations 
from tqdm import tqdm
from model import Unet
from copy import deepcopy
import torch.optim as optim
from shutil import copytree
import matplotlib.pyplot as plt
from optimizer import FastAdaBelief
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from matplotlib import pyplot as plt

#  from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from albumentations.pytorch import ToTensor
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, sampler
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)

warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def mask2rle(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return fname, masks
    
# Dataloader

class SteelDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "train_images",  image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'] # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1) # 4x256x1600
        if str(image_id)+'.npy' in os.listdir('predictions/'):
           mask = np.load('predictions/'+str(image_id)+'.npy')
        return idx, img, mask

    def __len__(self):
        return len(self.fnames)
        
    def set_idx(self, idx, mask):
        image_id, _ = make_mask(idx, self.df)
        np.save('predictions/'+image_id+'.npy', mask)


def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5), # only horizontal flip as of now
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms

def provider(
    data_folder,
    df_path,
    phase,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=4,
):
    '''Returns dataloader for the model training'''
    df = pd.read_csv(df_path)
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    #df['ImageId'], df['ClassId'] = zip(*df[['ImageId','ClassId']])
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
    df = train_df if phase == "train" else val_df
    image_dataset = SteelDataset(df, data_folder, mean, std, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )

    return dataloader
    
# Utility functions (Dice and IoU metric implementations, metric logger for training and validation)

def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

#         dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
#         dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
#         dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos

class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice.tolist())
        self.dice_pos_scores.extend(dice_pos.tolist())
        self.dice_neg_scores.extend(dice_neg.tolist())
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]

def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou

if not os.path.exists('/tmp/.cache/torch/checkpoints/'):
   os.makedirs('/tmp/.cache/torch/checkpoints/', exist_ok=True)
shutil.copyfile("resnet18.pth", "/tmp/.cache/torch/checkpoints/resnet18-5c106cde.pth")

model = Unet("resnet18", encoder_weights="imagenet", classes=4, activation=None)
model = model.to('cuda')

def best_entropy(p,target,ths=0.5):
    
    # NegEntropy
    a = p * np.log(p + 1e-8)
    b = (1-p) * np.log(1-p + 1e-8)
    mask_a = np.isfinite(a)
    mask_b = np.isfinite(b)
    filtered_arr_a = a[mask_a]
    filtered_arr_b = b[mask_b]
    NegEntropy = np.nanmean(filtered_arr_a) + np.nanmean(filtered_arr_b)
    
    # BCE
    a = target * np.log(p + 1e-8)
    b = (1-target) * np.log(1-p + 1e-8)
    mask_a = np.isfinite(a)
    mask_b = np.isfinite(b)
    filtered_arr_a = a[mask_a]
    filtered_arr_b = b[mask_b]    
    BCE = np.nanmean(filtered_arr_a) + np.nanmean(filtered_arr_b)
    
    return (-BCE - NegEntropy)/2

class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model, device='cuda'):
        self.num_workers = 31
        self.batch_size = {"train": 32, "val": 32}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 4e-5
        self.num_epochs = 20
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = device
        #torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.net_copy = deepcopy(self.net)
        self.net = self.net.to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        #self.optimizer = FastAdaBelief(self.net.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                data_folder=data_folder,
                df_path=train_df_path,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.early_stopping_warmup = 15
        self.early_stopping_phase_2 = 20
    
    def forward(self, images, targets):
        """ forward of single model """
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        """ training epoch """
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)

        self.optimizer.zero_grad()
        for itr, batch in tqdm(enumerate(dataloader), total=len(dataloader)): # replace `dataloader` with `tk0` for tqdm
            indexes, images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss, iou

    def iterate_get_preds_and_labels(self, phase="val", topk=20):
        """obtenir les predicitons et les labels dans la meme output"""

        outputs_pred_label = []
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        self.optimizer.zero_grad()

        liste_entropies = []
        list_indexes = []
        liste_outputs = []
        for itr, batch in tqdm(enumerate(dataloader), total=len(dataloader)): # replace `dataloader` with `tk0` for tqdm
            indexes, images, targets = batch
            images, targets = torch.Tensor(images),torch.Tensor(targets)
            _, outputs = self.forward(images, targets)
            images = images.detach().cpu().numpy()
            outputs = torch.softmax(outputs, axis=-1).detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            for idx in range(len(outputs)):
                entropy = 0
                for j in range(4):
                    entropy += 0.25 * best_entropy(outputs[idx][j],targets[idx][j])
                liste_entropies.append(entropy)
                list_indexes.append(indexes[idx])
                liste_outputs.append(outputs[idx])
                
            list_indexes = np.array(list_indexes)
            liste_outputs = np.array(liste_outputs)
            liste_entropies = np.array(liste_entropies)
            index_sorted = np.argsort(liste_entropies)[-topk:]
            liste_entropies = liste_entropies[index_sorted]
            liste_outputs = liste_outputs[index_sorted]
            list_indexes = list_indexes[index_sorted]
            
            list_indexes = list(list_indexes)
            liste_outputs = list(liste_outputs)
            liste_entropies = list(liste_entropies)

        liste_outputs = np.array(liste_outputs)

        for idx in range(len(list_indexes)):
            output = deepcopy(liste_outputs[idx])
            output[output>=0.5] = 1
            output[output<0.5] = 0
            dataloader.dataset.set_idx(list_indexes[idx], output)

    def start(self):
        ok = 0
        while True:

            if ok == 0:
                ok = 1
                var = torch.load('model.pth')
                self.net_copy.load_state_dict(var['state_dict'])
                self.best_loss = var['best_loss']
            else:
                self.net = deepcopy(self.net_copy)
            self.net = self.net.to(self.device)
            
            # PHASE I warmup
            early_stopping = 0
            epoch = 0
            while early_stopping < self.early_stopping_warmup:
                self.iterate(epoch, "train")

                state = {
                    "epoch": epoch,
                    "best_loss": self.best_loss,
                    "state_dict": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                with torch.no_grad():
                    val_loss, iou = self.iterate(epoch, "val")
                    self.scheduler.step(val_loss)
                if val_loss < self.best_loss:
                    print("******** New optimal found, saving state ********")
                    state["best_loss"] = val_loss
                    state["iou"] = iou
                    torch.save(state, "./model.pth")
                    self.best_loss = val_loss
                    early_stopping = 0
                else:
                    early_stopping += 1 
                epoch += 1

            var = torch.load('model.pth')
            self.net.load_state_dict(var['state_dict'])

            # PHASE II Active learning
            print('PHASE II')
            early_stopping = 0
            while early_stopping < self.early_stopping_phase_2:
                # PHASE IIA predictions on train dataset and correcting data
                print('PHASE IIA')
                self.iterate_get_preds_and_labels(phase="train")

                # PHASE IIB training epoch on corrected data
                print('PHASE IIB')
                self.iterate(epoch, phase='train')

                state = {
                    "epoch": epoch,
                    "state_dict": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }

                with torch.no_grad():
                    val_loss, iou = self.iterate(epoch, "val")
                    self.scheduler.step(val_loss)
                if val_loss < self.best_loss:
                    print("******** New optimal found, saving state ********")
                    state["best_loss"] = val_loss
                    state['iou'] = iou
                    torch.save(state, "./model_.pth")
                    self.best_loss = val_loss

                    early_stopping = 0
                else:
                    early_stopping += 1
                epoch += 1
            ok = 1

sample_submission_path = 'severstal-steel-defect-detection/sample_submission.csv'
train_df_path = 'severstal-steel-defect-detection/train2.csv'
data_folder = "severstal-steel-defect-detection/"
test_data_folder = "severstal-steel-defect-detection/test_images"
shutil.copyfile('severstal-steel-defect-detection/train.csv', 'severstal-steel-defect-detection/train2.csv')
os.makedirs('predictions/', exist_ok=True)

# environ 1 heure d'entrainement
model_trainer = Trainer(model)

model_trainer.start()
