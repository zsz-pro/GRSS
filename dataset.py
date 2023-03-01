from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import cv2

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


img_opt_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

img_sar_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

mask_transform = MaskToTensor()


class RSDataset(Dataset):
    def __init__(self, class_name, mode=None, img_transform=img_transform, mask_transform=mask_transform, sync_transforms=None):
        # 数据相关
        self.class_names = class_name
        self.mode = mode
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.sync_transform = sync_transforms
        self.sync_img_mask = []
        key_word = 'patches'
        if mode == "src": # for gid
            img_dir = os.path.join(root, 'rgb')
            mask_dir = os.path.join(root, 'label')
        else: # for whu-opt-sar
            img_dir = os.path.join(root, 'rgb')
            mask_dir = os.path.join(root, 'label')

        for img_filename in os.listdir(img_dir):
            img_mask_pair = (os.path.join(img_dir, img_filename),
                             os.path.join(mask_dir,
                                          img_filename.replace("MSS1.jpg", "MSS1_label.png").replace("MSS2.jpg", "MSS2_label.png")))
            self.sync_img_mask.append(img_mask_pair)
        print(self.sync_img_mask)
        if (len(self.sync_img_mask)) == 0:
            print("Found 0 data, please check your dataset!")

    def __getitem__(self, index):
        img_path, mask_path = self.sync_img_mask[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        # transform
        if self.sync_transform is not None:
            img, mask = self.sync_transform(img, mask)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.sync_img_mask)

    def classes(self):
        return self.class_names




class WHUOPTSARDataset(Dataset):
    def __init__(self, class_name, root, mode=None, img_sar_transform=img_sar_transform, img_opt_transform=img_opt_transform, mask_transform=mask_transform, sync_transforms=None):
        # 数据相关
        self.class_names = class_name
        self.mode = mode
        self.img_sar_transform = img_sar_transform
        self.img_opt_transform = img_opt_transform
        self.mask_transform = mask_transform
        self.sync_transform = sync_transforms
        self.sync_img_mask = []

        img_sar_dir = os.path.join(root, 'sar')
        img_opt_dir = os.path.join(root, 'rgb')
        # mask_dir = os.path.join(root, 'lbl')
        mask_dir = os.path.join(root, 'dsm')

        for img_filename in os.listdir(img_sar_dir):
            img_mask_pair = (os.path.join(img_sar_dir, img_filename),
                             os.path.join(img_opt_dir, img_filename),
                             os.path.join(mask_dir, img_filename))
            self.sync_img_mask.append(img_mask_pair)
        # print(self.sync_img_mask)

        if (len(self.sync_img_mask)) == 0:
            print("Found 0 data, please check your dataset!")

    def __getitem__(self, index):
        img_sar_path, img_opt_path, mask_path = self.sync_img_mask[index]
        img_sar = Image.open(img_sar_path)#mode:F32
        img_opt = Image.open(img_opt_path)#mode:rgb
        dsm = Image.open(mask_path)  #mode:F32      
        mask = dsm.convert('L')#L:8bit
            
        mask = np.array(mask)
        ret2, thresh2 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ekernel = np.ones((7, 7), np.uint8)
        dkernel = np.ones((7, 7), np.uint8)
        #开运算 
        erosion = cv2.dilate(thresh2, dkernel, iterations=1)
        mask = cv2.erode(erosion, ekernel, iterations=1)
        mask = Image.fromarray(mask)
        # transform
        if self.sync_transform is not None:
            img_sar, img_opt, mask, dsm = self.sync_transform(img_sar, img_opt, mask, dsm)#flip
        if self.img_sar_transform is not None:
            img_sar = self.img_sar_transform(img_sar)
            dsm = self.img_sar_transform(dsm)   #both sar&dsm are float32,trans2tensor
            img_opt = self.img_opt_transform(img_opt)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)#float32->torch.int64???
        return img_sar, img_opt, mask,dsm

    def __len__(self):
        return len(self.sync_img_mask)

    def classes(self):
        return self.class_names

if __name__ ==  "__main__":
    pass