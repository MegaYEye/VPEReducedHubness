import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.misc as m
# from augmentations import *
# from models import get_model
import random

class belga2toplogoLoader(Dataset):

  def __init__(self, root, exp, split='train', is_transform=False, img_size=None, augmentations=None, prototype_sampling_rate=0.005):
    super().__init__()

    if split == 'train':
        self.proto_rate = prototype_sampling_rate
    else:
        self.proto_rate = 0.0
    self.inputs = []
    self.targets = []
    self.class_names = []

    if split == 'train':
        self.split = 'belga'
        self.n_classes = 37 #
        self.tr_class = torch.LongTensor([12, 16, 28, 30]) # belga classes
        self.te_class = torch.LongTensor([0,1,2,3,4,5,6,7,8,9,10,11,  13,14,15,  17,18,19,20,21,22,23,24,25,26,27,  29,  31,32,33,34,35,36]) # belga classes
    elif split == 'test':
        self.split = 'toplogo10'
        self.n_classes = 11
        self.tr_class = torch.LongTensor([0,1,3,7,9])
        self.te_class = torch.LongTensor([2,4,5,6,8,10])
    elif split == 'val':
        self.split = 'flickr32'
        self.n_classes = 32
        self.tr_class = torch.LongTensor([7, 13, 25, 28]) # belga classes
        self.te_class = torch.LongTensor([0,1,2,3,4,5,6,  8,9,10,11,12,  14,15,16,17,18,19,20,21,22,23,24,  26,27,  29,30,31]) # belga classes

    self.img_size = img_size
    self.is_transform = is_transform
    self.augmentations = augmentations
    self.mean = np.array([125.00, 125.00, 125.00]) # average intensity

    self.root = root
    self.dataPath = root + exp + '/' + self.split + '_impaths.txt'
    self.labelPath = root + exp + '/' + self.split + '_imclasses.txt'

    f_data = open(self.dataPath,'r')
    f_label = open(self.labelPath,'r')
    data_lines = f_data.readlines()
    label_lines = f_label.readlines()

    for i in range(len(data_lines)):
      self.inputs.append(root+data_lines[i][0:-1])
      self.targets.append(int(label_lines[i].split()[0])) # label: [road class, wet/dry, video index]
    
    classnamesPath = root + exp + '/' + self.split + '_classnames.txt'
    f_classnames = open(classnamesPath, 'r')
    data_lines = f_classnames.readlines()
    for i in range(len(data_lines)):
        self.class_names.append(data_lines[i][0:-1])

    assert(self.n_classes == len(self.class_names))

    print('%s %s %d classes'%(split, self.split, len(self.class_names)))
    print('Load %s: %s: %d samples'%(split, self.split,  len(self.targets)))


  def __len__(self):
    return len(self.inputs)


  def __getitem__(self, index):
    img_path = self.inputs[index]
    gt = self.targets[index]
    gt = torch.ones(1).type(torch.LongTensor)*gt
    
    if self.split == 'test':
        pdb.set_trace()

    # Load images and templates. perform augmentations
    img = m.imread(img_path)
    img = np.array(img, dtype=np.uint8)
    template = m.imread(self.root + self.split + '/template_ordered/%02d.jpg'%(gt))
    template = np.array(template, dtype=np.uint8)

    if random.random() < self.proto_rate:
        img = np.copy(template)

    if self.augmentations is not None:
        img, template = self.augmentations(img, template)

    if self.is_transform:
        img = self.transform(img)
        template = self.transform(template)

    gt = gt-1

    return img, gt, template
    
  def transform(self, img):
    img = img.astype(np.float64)
    img -= self.mean
    if self.img_size is not None:
      img = m.imresize(img, (self.img_size[0], self.img_size[1]))
    # Resize scales images from 0 to 255, thus we need
    # to divide by 255.0
    img = img.astype(float) / 255.0
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    
    return img


  def load_template(self, target, augmentations=None):

    # if augmentation is not specified, use self.augmentations. Unless use input augmentation option.
    if augmentations is None:
        augmentations = self.augmentations
    img_paths = []
    
    for id in target:
        img_paths.append(self.root + self.split +'/template_ordered/%02d.jpg'%(id+1))

    target_img = []
    for img_path in img_paths:
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        if augmentations is not None:
            img, _ = augmentations(img, img)
        if self.transform:
            img = self.transform(img)

        target_img.append(img)

    return torch.stack(target_img, dim=0)