import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self,image):
        image = (image - self.mean) / self.std
        return image

class Resize(object):

    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):

        image_high = cv2.resize(image, dsize=(self.W*2, self.H*2), interpolation=cv2.INTER_LINEAR)
        image_low = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask_high = cv2.resize(mask, dsize=(self.W*2, self.H*2), interpolation=cv2.INTER_LINEAR) / 255
        mask_low = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR) / 255
        return image_low,image_high,mask_high,mask_low

class ToTensor(object):
    def __call__(self,image_low,image_high,mask_high,mask_low ):

        image_low = torch.from_numpy(image_low)
        image_low = image_low.permute(2, 0, 1)

        image_high = torch.from_numpy(image_high)
        image_high = image_high.permute(2, 0, 1)

        mask_high = torch.from_numpy(mask_high)
        mask_high = torch.unsqueeze(mask_high,0)

        mask_low = torch.from_numpy(mask_low)
        mask_low = torch.unsqueeze(mask_low, 0)
        return image_low,image_high,mask_high,mask_low

class Val_Datasets(Dataset):

    def __init__(self):

        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])

        self.normalize  = Normalize( mean=self.mean, std=self.std )
        self.resize = Resize(352,352)
        self.totensor   = ToTensor()

        self.img_dir = "data/val/imgs/"
        self.examples = []

        file_names = os.listdir(self.img_dir)

        for file_name in file_names:
            if file_name.find(".jpg") != -1 and file_name.startswith(".") == False:

                img_path = self.img_dir + file_name

                example = {}
                example["img_path"] = img_path
                example["label_name"] = file_name.replace(".jpg", ".png")
                self.examples.append(example)
        self.num_examples = len(self.examples)

    def __getitem__(self, idx):

        example = self.examples[idx]

        img_path = example["img_path"]
        image = cv2.imread(img_path)[:,:,::-1].astype(np.float32)

        image_low = (image - self.mean) / self.std
        image_low = cv2.resize(image_low,(352,352))
        image_low = torch.from_numpy(image_low)
        image_low = image_low.permute(2,0,1)

        image_high = cv2.resize(image,dsize=(1024,1024),interpolation=cv2.INTER_NEAREST)
        image_high = (image_high - self.mean) / self.std
        image_high = image_high.transpose((2,0,1)).astype(np.float32)
        image_high = torch.from_numpy(image_high)

        return image_low,image_high

    def __len__(self):
        return self.num_examples


