import torch
import os
from torch.autograd import Variable
from PIL import Image
import math
import torch.nn.functional as F
import numpy as np

class ImageDataset(torch.utils.data.Dataset): 
    def __init__(self, data, transform=None): 
        self.imgs = data # path, label
        self.transform = transform 
  
    # Defining the length of the dataset 
    def __len__(self): 
        return len(self.imgs) 
  
    # Defining the method to get an item from the dataset 
    def __getitem__(self, index): 
        path, target = self.imgs[index]
        sample = self.pil_loader(path)
  
        # Applying the transform 
        if self.transform: 
            sample = self.transform(sample) 
          
        return sample, target, index
    
    def pil_loader(self, path):
      # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
      with open(path, 'rb') as f:
          img = Image.open(f)
        #   return img.convert('L')
          return img.convert('RGB')
      

class NLNLCrossEntropyLoss():
    def __init__(self, weight_pos, weight_neg, num_classes, epsilon = 1e-7):
        
        self.epsilon = epsilon
        
        # Reversed?
        self.weight_pos = weight_neg.to("cuda")
        self.weight_neg = weight_pos.to("cuda")
        self.num_classes = num_classes

    def loss(self, y_pred, y_pred_neg, y_true, y_true_neg, ignore_index = -100):
        # Initialize loss to zero
        loss = 0.0
        loss_neg = 0.0

        # negative
        for i in range(self.num_classes):
            mask = y_pred_neg[:,i]!=ignore_index
            if mask.sum(dim=0) > 0:
                loss_neg_mean = ((self.weight_neg[i] * y_true_neg[:,i]
                                * torch.log(y_pred_neg[:,i] + self.epsilon))*mask).sum(dim=0)/mask.sum(dim=0)
                loss_neg += -1 * loss_neg_mean
                print("loss_neg:" + str(loss_neg))
                if math.isnan(loss_neg):
                    print(y_pred_neg[:,i])
                    print(loss_neg_mean)
                    print(mask)
                    print(self.weight_neg[i])
                    print(y_true_neg[:,i])
                    break

        # positive
        for i in range(self.num_classes):
            mask = y_pred[:,i]!=ignore_index
            if mask.sum(dim=0) > 0:
                loss_mean = ((self.weight_pos[i] * y_true[:,i]
                                * torch.log(y_pred[:,i] + self.epsilon))*mask).sum(dim=0)/mask.sum(dim=0)
                loss += -1 * loss_mean
                print("loss:" + str(loss))
        # return Variable(loss, requires_grad = True)
        return loss + loss_neg

# Proved to be the same as NLLLoss
class NLNLCrossEntropyLossNL():
    def __init__(self, weight, num_classes, epsilon = 1e-7):
        
        self.epsilon = epsilon
        
        # Reversed?
        self.weight = weight.to("cuda")
        self.num_classes = num_classes

    def loss(self, logits, y_true_neg, ignore_index = -100):
        # Initialize loss to zero
        loss_neg = 0.0
        loss_neg_log = torch.log( torch.clamp(1.-F.softmax(logits, -1), min=1e-5, max=1.))

        for i in range(self.num_classes):
            mask = y_true_neg[:,i]!=ignore_index
            if mask.sum(dim=0) > 0:
                loss_neg_mean = ((self.weight[i]
                                * loss_neg_log[:,i] * y_true_neg[:,i])*mask).sum(dim=0)/mask.sum(dim=0)
                loss_neg += -1 * loss_neg_mean
        return loss_neg
    
class NLNLCrossEntropyLossPL():
    def __init__(self, weight, num_classes, epsilon = 1e-7):
        
        self.epsilon = epsilon
        
        # Reversed?
        self.weight = weight.to("cuda")
        self.num_classes = num_classes

    def loss(self, logits, y_true, ignore_index = -100):
        # Initialize loss to zero
        loss = 0.0
        loss_log = torch.log( torch.clamp(F.softmax(logits, -1), min=1e-5, max=1.))
        # loss_log = torch.log( torch.clamp(F.sigmoid(logits), min=1e-5, max=1.))
        for i in range(self.num_classes):
            mask = y_true[:,i]!=ignore_index
            if mask.sum(dim=0) > 0:
                loss_mean = ((self.weight[i]
                                * loss_log[:,i] * y_true[:,i])*mask).sum(dim=0)/mask.sum(dim=0)
                loss += -1 * loss_mean
        return loss
    
def LabelDropper(data, labels):
    print("Entered LabelDropper")
    print(labels)

    # drop unrelated
    for label in labels:
        if 'drop_index' in locals():
            drop_index = drop_index.intersection(data[(data['Finding Labels'].str.contains(label) == False)].index)
        else:
            drop_index = data[(data['Finding Labels'].str.contains(label) == False)].index

    data = data.drop(drop_index)

    for index in range(len(labels)):
        label = labels[index]
        for index2 in range(len(labels)):
            if index != index2:
                label2 = labels[index2]
                data = data.drop(data[(data['Finding Labels'].str.contains(label) == True) & (data['Finding Labels'].str.contains(label2) == True)].index)

        # finished dropping
        data.loc[data['Finding Labels'].str.contains(label) == True, 'Finding Labels'] = label

    return data