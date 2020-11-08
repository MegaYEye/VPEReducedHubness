import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from PIL import Image
import scipy.misc as m
import pickle
import glob
from tqdm import tqdm

def save_data(path="/media/lisa/1494EA6E94EA5232/datasets/23079_29550_bundle_archive/asl_alphabet_train/asl_alphabet_train"):
    f_iterator = glob.glob(os.path.join(path, "*/*.jpg"))
    X = []
    y = []
    for f in tqdm(f_iterator):
        data = Image.open(f).resize((64,64)).convert('RGB')
        data = np.asarray(data)
        X.append(data)
        label = ord(f.split("/")[-2])-ord('A')
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    np.save("X.npy", X)
    np.save("y.npy",y)
    return X,y

def save_template(path="/media/lisa/1494EA6E94EA5232/datasets/23079_29550_bundle_archive/asl_alphabet_template"):
    f_iterator = glob.glob(os.path.join(path, "*.png"))
    files = sorted([f for f in f_iterator])
    T = []
    for f in files:
        data = Image.open(f).resize((64,64)).convert('RGB')
        data = np.asarray(data)
        T.append(data) 
    T = np.array(T)
    print(T.shape)
    np.save("T.npy",T)
    return T
        
# save_data()

class ASLLoader(Dataset):
    def __init__(self, root="./", exp=None,split="train", is_transform=False, img_size=(64,64), augmentations=None, prototype_sampling_rate=0.005, template_choice="iconic"):
        root = os.path.join(root,"ASL/")
        
        y = np.load(os.path.join(root,"y.npy"))
        choice = np.random.choice(len(y),1000)
        
        X = np.load(os.path.join(root,"X.npy"))#[choice]
        y = np.load(os.path.join(root,"y.npy"))#[choice]
        T = np.load(os.path.join(root,"T.npy"))
        if template_choice == "iconic":
            T = np.load(os.path.join(root,"T_new.npy"))
        X = X.astype(np.uint8)
        T = T.astype(np.uint8)
        y = y.astype(np.long)
        # X[X==255]=250
        # T[T==255]=250
        # X[X==0]=255
        # T[T==0]=255
        if template_choice=="iconic":
            pass
        else:
            X[X==0]=255
            T[T==0]=255            

        if split == "train":
            mask = y<16
            self.X = X[mask]
            self.y = y[mask]
            self.T = T
        elif split == "val":
            # mask = (y>=16) & (y<21)
            mask = y>=16
            self.X = X[mask]
            self.y = y[mask]
            self.T = T
        elif split == "test":
            mask = y>=16
            # mask = y>=21
            self.X = X[mask]
            self.y = y[mask]
            self.T = T    
        
        self.targets = self.y          
        
        self.augmentations = augmentations
        self.is_transform=is_transform
        self.img_size=img_size
        self.label_set = np.unique(self.y)
        self.class_map={c:i for i,c in enumerate(self.label_set)}
        self.class_imap={i:c for i,c in enumerate(self.label_set)}
        self.class_index={lb: np.arange(len(y))[y==lb] for lb in self.label_set}
        self.n_classes = len(self.label_set)
        self.tr_class= torch.LongTensor(np.arange(len(self.label_set)))
        self.te_class= torch.LongTensor(np.arange(len(self.label_set)))
        self.proto_rate = prototype_sampling_rate
        self.template_choice = template_choice
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, index):
        img = self.X[index]
        y = self.y[index]
        template = self.T[y]
        gt = self.class_map[y]

        if random.random() < self.proto_rate:
            img = np.copy(template)

        if self.augmentations is not None:
            img, template = self.augmentations(img, template)

        img = torch.from_numpy(img).float().transpose(2,0)/255.0
        template = torch.from_numpy(template).float().transpose(2,0)/255.0
        # emb = torch.FloatTensor(self.cifar100_w2v[self.class_imap[gt]][1])
        return img, gt, template

    def load_template(self, target, augmentations=None):
        # if augmentation is not specified, use self.augmentations. Unless use input augmentation option.
        if augmentations is None:
            augmentations = self.augmentations
    
        #print(target)
        target_img = []
        embs = []
        for i in range(self.n_classes):
            if i in target:
                img = self.T[self.class_imap[i]]
            else:
                continue
            if augmentations is not None:
                img, _ = augmentations(img, img)
                
            img = torch.from_numpy(img).float().transpose(2,0)/255.0
            target_img.append(img)
            #print(img.shape)
        #print(len(target_img))
        return torch.stack(target_img,dim=0)

if __name__ == "__main__":
    asl = ASLLoader(split="train")
    img, gt, template = asl[10]
    templates = asl.load_template([1,2,3])
    templates = asl.load_template([1,2,3])
