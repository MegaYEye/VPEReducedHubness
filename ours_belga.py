from argparse import ArgumentParser
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import math
import os
from pathlib import Path

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.models import densenet121
import torchvision
from torchvision import datasets, transforms

from loader import get_loader, get_data_path
from models import get_model
from augmentations import *

os.environ["OMP_NUM_THREADS"] = "3" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "3" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "3" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "3" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "3" # export NUMEXPR_NUM_THREADS=1
torch.set_num_threads(4) # this is important for torch 1.2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! otherwise it will go 100 degree..


# Setup
parser = ArgumentParser(description='Variational Prototyping Encoder (VPE)')
parser.add_argument('--seed',       type=int,   default=42,             help='Random seed')
parser.add_argument('--arch',       type=str,   default='vaeIdsiaStn',  help='network type: vaeIdsia, vaeIdsiaStn')
parser.add_argument('--dataset',    type=str,   default='belga2flickr', help='dataset to use [gtsrb, belga2flickr, belga2toplogo]') # for gtsrb2TT100K scenario, use main_train_test.py
parser.add_argument('--exp',        type=str,   default='exp_list',     help='training scenario')
parser.add_argument('--resume',     type=str,   default=None,           help='Resume training from previously saved model')

parser.add_argument('--epochs',     type=int,   default=1000,           help='Training epochs')
parser.add_argument('--lr',         type=float, default=1e-4,           help='Learning rate')
parser.add_argument('--batch_size', type=int,   default=128,            help='Batch size')

parser.add_argument('--img_cols',   type=int,   default=64,             help='resized image width')
parser.add_argument('--img_rows',   type=int,   default=64,             help='resized image height')
parser.add_argument('--workers',    type=int,   default=6,              help='Data loader workers')

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic=True

plt.switch_backend('agg')  # Allow plotting when running remotely

save_epoch = 100 # save log images per save_epoch

# 02 rotation + flip augmentation option
# Setup Augmentations
data_aug_tr= Compose([Scale(args.img_cols), # resize longer side of an image to the defined size
                      CenterPadding([args.img_rows, args.img_cols]), # zero pad remaining regions
                      RandomHorizontallyFlip(), # random horizontal flip
                      RandomRotate(180),
                      ColorDistort()])  # ramdom rotation

data_aug_te= Compose([Scale(args.img_cols), 
                     CenterPadding([args.img_rows, args.img_cols])])

result_path = 'results_' + args.dataset
if not os.path.exists(result_path):
  os.makedirs(result_path)
outimg_path =  "./img_log_" + args.dataset
if not os.path.exists(outimg_path):
  os.makedirs(outimg_path)

f_loss = open(os.path.join(result_path, "log_loss.txt"),'w')
f_loss.write('Network type: %s\n'%args.arch)
f_loss.write('Learning rate: %05f\n'%args.lr)
f_loss.write('batch-size: %s\n'%args.batch_size)
f_loss.write('img_cols: %s\n'%args.img_cols)
f_loss.write('Augmentation type: flip, centercrop\n\n')
f_loss.close()

f_iou = open(os.path.join(result_path, "log_acc.txt"),'w')
f_iou.close()

f_iou = open(os.path.join(result_path, "log_val_acc.txt"),'w')
f_iou.close()

# set up GPU
# we could do os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Data
data_loader = get_loader(args.dataset)
data_path = get_data_path(args.dataset)

tr_loader = data_loader(data_path, args.exp, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_tr,)
te_loader = data_loader(data_path, args.exp, is_transform=True, split='test', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_te)
val_loader = data_loader(data_path, args.exp, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_te)

trainloader = DataLoader(tr_loader, batch_size=args.batch_size, shuffle=True, pin_memory=True,worker_init_fn = np.random.seed(args.seed))#num_workers=args.workers, 
testloader = DataLoader(te_loader, batch_size=args.batch_size, shuffle=True, pin_memory=True,worker_init_fn = np.random.seed(args.seed))#num_workers=args.workers, num_workers=args.workers,
valloader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=True, pin_memory=True,worker_init_fn = np.random.seed(args.seed))#num_workers=args.workers, num_workers=args.workers,


# define model or load model
class DenseModel(nn.Module):
    def __init__(self, num_classes):
        super(DenseModel, self).__init__()
        self.densenet= densenet121(pretrained=False)
        self.input_dense = nn.Linear(300, 3*32*32)
        self.output_f = nn.Linear(1024,300)
        self.classifier = nn.Linear(300,num_classes)
        
    def forward(self, input): 
        x = self.input_dense(input)
        features = self.densenet.features(x.view(-1,3,32,32))
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        f = torch.flatten(out, 1)
        #print(f.shape) # [batch, 1024]
        f = self.output_f(f)*0.1+input
        out = self.classifier(f)
        #print(out.shape) # batch, 37]
        #print(out.shape)
        out = F.log_softmax(out, dim=1)
        return out,f   

class Model(nn.Module):
    def __init__(self, num_classes=37):
        super(Model,self).__init__()
        self.vpe = get_model(args.arch, n_classes=None)   
        self.dense = DenseModel(num_classes)
	       
    def forward(self, input): 
        recon, mu, logvar, stn = self.vpe(input)
        pred, feature = self.dense(mu)
        # pred=None
        # feature=None
        return recon, mu, logvar, stn, pred, feature
        
net = Model()
#net = get_model(args.arch, n_classes=None)
net.cuda()

if args.resume is not None:
  pre_params = torch.load(args.resume)
  net.init_params(pre_params)


reconstruction_function = nn.BCELoss()
reconstruction_function.reduction = 'sum'
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)
    # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE# + KLD

# Construct optimiser
optimizer = optim.Adam(net.parameters(), lr=args.lr,weight_decay=1e-5) # 1e-4

num_train = len(tr_loader.targets)
num_test = len(te_loader.targets)
num_val = len(val_loader.targets)
batch_iter = math.ceil(num_train/args.batch_size)
batch_iter_test = math.ceil(num_test/args.batch_size)
batch_iter_val = math.ceil(num_val/args.batch_size)

def train(e):
  n_classes = tr_loader.n_classes
  n_classes_te = te_loader.n_classes
  n_classes_val = val_loader.n_classes
  print('start train epoch: %d'%e)
  net.train()
  
  for i, (input, target, template) in enumerate(trainloader):
    optimizer.zero_grad()
    target = torch.squeeze(target)
    target_gpu = target.cuda()
    input, template = input.cuda(), template.cuda()

    _, mu_temp,_,_,pred_temp, fea_temp = net(template)
    recon, mu, logvar, input_stn, pred, fea = net(input)
    loss1 = loss_function(recon, template, mu, logvar) # reconstruction loss
    
    mu_expand = mu.view(1, mu.shape[0], mu.shape[1]).repeat([template.shape[0],1,1]) 
    template_expand = mu_temp.view(mu_temp.shape[0],1, mu_temp.shape[1]).repeat([1,mu.shape[0],1]) 
    dis = (mu_expand-template_expand).norm(p=2, dim=-1)*0.1

    contrastive_mask_same = target.view(1,target.shape[0]).repeat([target.shape[0],1])==target.view(target.shape[0],1).repeat([1,target.shape[0]])
    contrastive_mask_diff = ~contrastive_mask_same
    #same_dis = dis[contrastive_mask_same].topk(3, largest=True, sorted=False)[0].mean()
    #diff_dis = dis[contrastive_mask_diff].topk(3, largest=False, sorted=False)[0].mean()
    same_similarity = (-dis[contrastive_mask_same]).exp().mean()
    diff_similarity = (-dis[contrastive_mask_diff]).exp().mean()
    loss2 = -((same_similarity/(same_similarity+diff_similarity)).log())
    
    # if torch.isnan(diff_dis):
    #     loss2=0
    # else:
    #     loss2 = same_dis.exp()/(same_dis.exp()+diff_dis.exp())
    #     loss2 = (1e-7+loss2).log()
    loss4 = (mu-mu_temp).norm(dim=1,p=2).sum()#((mu-mu_temp).pow(2) + 1e-8).sum().sqrt()* 1.0 (mu-mu_temp).norm(dim=1,p=2).sum()#
    loss3 = F.nll_loss(pred, target_gpu)
    loss = loss1 + loss2 + loss3 + loss4
    print('Epoch:%d  Batch:%d/%d  loss1:%08f loss2:%08f loss3:%08f loss4:%08f'%(e, i, batch_iter, loss1/input.numel(), loss2, loss3, loss4))
    
    f_loss = open(os.path.join(result_path, "log_loss.txt"),'a')
    f_loss.write('Epoch:%d  Batch:%d/%d  loss:%08f\n'%(e, i, batch_iter, loss/input.numel()))
    f_loss.close()
    
    loss.backward()
    optimizer.step()

    if i < 1 and (e%save_epoch == 0):
      out_folder =  "%s/Epoch_%d_train"%(outimg_path, e)
      out_root = Path(out_folder)
      if not out_root.is_dir():
        os.mkdir(out_root)

      torchvision.utils.save_image(input, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(input_stn, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
      torchvision.utils.save_image(recon, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(template, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2)
    

  if e%save_epoch == 0:
    class_target = torch.LongTensor(list(range(n_classes)))
    class_template = tr_loader.load_template(class_target)
    class_template = class_template.cuda()
    with torch.no_grad():
      class_recon, class_mu, class_logvar, _,pred, fea = net(class_template)
    
    torchvision.utils.save_image(class_template, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2)
    torchvision.utils.save_image(class_recon, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2)
  
def score_NN(pred, class_feature, label, n_classes):

  sample_correct = torch.zeros(n_classes)
  sample_all = torch.zeros(n_classes)
  sample_rank = torch.zeros(n_classes, n_classes) # rank per class
  sample_distance = torch.ones(pred.shape[0], n_classes)*math.inf

  pred = pred.cpu() # batch x latent size
  class_feature = class_feature.cpu() # n_classes x latent size
  mean = class_feature.mean(0)
  pred -= mean
  class_feature -= mean
  #print(pred.shape, mean.shape)
  pred = pred/torch.norm(pred, p=2, dim=1, keepdim=True)
  class_feature = class_feature/torch.norm(class_feature, p=2, dim=1, keepdim=True)
  #print(pred.shape, mean.shape,class_feature.shape)
  label = label.numpy()
  for i in range(n_classes):
    cls_feat = class_feature[i,:]
    cls_mat = cls_feat.repeat(pred.shape[0],1)
    # euclidean distance
    sample_distance[:,i] = torch.norm(pred - cls_mat,p=2, dim=1)
  
  sample_distance = sample_distance.cpu().numpy()
  indices = np.argsort(sample_distance, axis=1) # sort ascending order

  for i in range(indices.shape[0]):
    rank = np.where(indices[i,:] == label[i])[0][0] # find rank
    sample_rank[label[i]][rank:] += 1 # update rank 
    sample_all[label[i]] += 1 # count samples per class
    if rank == 0:
      sample_correct[label[i]] += 1 # count rank 1 (correct classification)

  return sample_correct, sample_all, sample_rank


mean_scores = []
mean_rank = []
def test(e, best_acc, val_trigger):
  n_classes = te_loader.n_classes
  print('start test epoch: %d'%e)
  net.eval()
  accum_all = torch.zeros(n_classes)
  rank_all = torch.zeros(n_classes, n_classes) # rank per class
  accum_class = torch.zeros(n_classes)
  
  accum_all2 = torch.zeros(n_classes)
  rank_all2 = torch.zeros(n_classes, n_classes) # rank per class
  accum_class2 = torch.zeros(n_classes)
  # get template latent z
  class_target = torch.LongTensor(list(range(n_classes)))
  class_template = te_loader.load_template(class_target)
  class_template = class_template.cuda()
  with torch.no_grad():
    class_recon, class_mu, class_logvar, _, pred_temp, fea_temp = net(class_template)
  
  for i, (input, target, template) in enumerate(testloader):

    target = torch.squeeze(target)
    input, template = input.cuda(), template.cuda()
    with torch.no_grad():
      recon, mu, logvar, input_stn, pred, fea  = net(input)
    
    sample_correct, sample_all, sample_rank = score_NN(mu, class_mu, target, n_classes)
    accum_class += sample_correct
    accum_all += (sample_all)
    rank_all = rank_all + sample_rank # [class_id, topN]
    
    # sample_correct2, sample_all2, sample_rank2 = score_NN(fea, fea_temp, target, n_classes)
    # accum_class2 += sample_correct2
    # accum_all2 += sample_all2
    # rank_all2 = rank_all + sample_rank2 # [class_id, topN]
    print('Epoch:%d  Batch:%d/%d  processing...'%(e, i, batch_iter_test))

    if i < 1 and (e%save_epoch == 0):
      out_folder =  "%s/Epoch_%d_test"%(outimg_path, e)
      out_root = Path(out_folder)
      if not out_root.is_dir():
        os.mkdir(out_root)

      torchvision.utils.save_image(input, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(input_stn, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
      torchvision.utils.save_image(recon, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(template, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2)

  if e%save_epoch == 0:
    torchvision.utils.save_image(class_template, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2)  
    torchvision.utils.save_image(class_recon, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2)  

  acc_all = accum_class.sum() / accum_all.sum() 
  acc_cls = torch.div(accum_class, accum_all)
  
  # if np.isnan(acc_cls).sum()>0:
  #   print("....")

  rank_sample_avg = rank_all.sum(0) / accum_all.sum() # [class_id, topN]
  rank_cls = torch.div(rank_all, torch.transpose(accum_all.expand_as(rank_all),0,1))
  rank_cls_avg = torch.mean(rank_cls,dim=0)

  # acc_all2 = accum_class2.sum() / accum_all2.sum() 
  # acc_cls2 = torch.div(accum_class2, accum_all2)

  # write result part
  acc_trcls = torch.gather(acc_cls, 0, te_loader.tr_class)
  acc_tecls =torch.gather(acc_cls, 0, te_loader.te_class)

  print('========epoch(%d)========='%e)
  print('Seen Classes')
  for i, class_acc in enumerate(acc_trcls):
    print('cls:%d  acc:%02f'%(te_loader.tr_class[i], class_acc))
  print('Unseen Classes')
  for i, class_acc in enumerate(acc_tecls):
    print('cls:%d  acc:%02f'%(te_loader.te_class[i], class_acc))
  print('====================================')
  print('acc_avg:%02f'%acc_all)
  print('acc_cls:%02f'%acc_cls.mean())
  print('acc_trcls:%02f'%acc_trcls.mean())
  print('acc_tecls:%02f'%acc_tecls.mean())
  print('rank sample avg: %02f'%rank_sample_avg.mean())
  print('rank cls avg: %02f'%rank_cls_avg.mean())
  print('====================================')

  f_iou = open(os.path.join(result_path, "log_acc.txt"),'a')
  f_iou.write('epoch(%d), acc_cls: %04f, acc_trcls: %04f  acc_tecls: %04f  acc_all: %04f  top3: %04f  top5: %04f\n'%(e, acc_cls.mean(), acc_trcls.mean(), acc_tecls.mean(), acc_all, rank_sample_avg[2], rank_sample_avg[4]))
  f_iou.close()

  if val_trigger: # when validation performance higher than prev val_best performance.
    # in the paper, we report best accuracy triggered by validation performance.
    f_iou_class = open(os.path.join(result_path, "best_iou_triggeredByVal.txt"),'w')
    f_rank = open(os.path.join(result_path, "best_rank_triggeredByVal.txt"),'w')

    f_iou_class.write('Best score epoch:  %d\n'%e)
    f_iou_class.write('acc cls: %.4f  acc all: %.4f  rank mean: %.4f \n'%(acc_cls.mean(), acc_all, rank_all.mean()))
    f_iou_class.write('acc tr cls: %.4f  acc te cls: %.4f\n'%(acc_trcls.mean(), acc_tecls.mean()))
    f_iou_class.write('top3: %.4f  top5: %.4f\n'%(rank_sample_avg[2], rank_sample_avg[4]))

    f_iou_class.write('\nSeen classes\n')
    for i, class_acc in enumerate(acc_trcls):
      f_iou_class.write('cls:%d  acc:%02f\n'%(te_loader.tr_class[i], class_acc))
    f_iou_class.write('\nUnseen classes\n')
    for i, class_acc in enumerate(acc_tecls):
      f_iou_class.write('cls:%d  acc:%02f\n'%(te_loader.te_class[i], class_acc))
    f_iou_class.close()
    
    for i, rank_acc in enumerate(rank_sample_avg):
      f_rank.write('rank sample %d: %.4f\n'%(i+1, rank_acc))
    f_rank.write('\n')
    for i, rank_acc in enumerate(rank_cls_avg):
      f_rank.write('rank cls %d: %.4f\n'%(i+1, rank_acc))
    f_rank.close()

  if best_acc < acc_tecls.mean(): # update best score
    # best accuracy during the training stage. Just for reference.
    f_iou_class = open(os.path.join(result_path, "best_iou.txt"),'w')
    f_rank = open(os.path.join(result_path, "best_rank.txt"),'w')
    torch.save(net.state_dict(), os.path.join('%s_testBest_net.pth'%args.dataset)) # if best_acc == 0, then valBest model is saved

    best_acc = acc_tecls.mean()
    f_iou_class.write('Best score epoch:  %d\n'%e)
    f_iou_class.write('acc cls: %.4f  acc all: %.4f  rank mean: %.4f \n'%(acc_cls.mean(), acc_all, rank_all.mean()))
    f_iou_class.write('acc tr cls: %.4f  acc te cls: %.4f\n'%(acc_trcls.mean(), acc_tecls.mean()))
    f_iou_class.write('top3: %.4f  top5: %.4f\n'%(rank_sample_avg[2], rank_sample_avg[4]))

    f_iou_class.write('\nSeen classes\n')
    for i, class_acc in enumerate(acc_trcls):
      f_iou_class.write('cls:%d  acc:%02f\n'%(te_loader.tr_class[i], class_acc))
    f_iou_class.write('\nUnseen classes\n')
    for i, class_acc in enumerate(acc_tecls):
      f_iou_class.write('cls:%d  acc:%02f\n'%(te_loader.te_class[i], class_acc))
    f_iou_class.close()
    
    for i, rank_acc in enumerate(rank_sample_avg):
      f_rank.write('rank sample %d: %.4f\n'%(i+1, rank_acc))
    f_rank.write('\n')
    for i, rank_acc in enumerate(rank_cls_avg):
      f_rank.write('rank cls %d: %.4f\n'%(i+1, rank_acc))
    f_rank.close()


  ###### Plot scores
  mean_scores.append(acc_tecls.mean())
  es = list(range(len(mean_scores)))
  plt.plot(es, mean_scores, 'b-')
  plt.xlabel('Epoch')
  plt.ylabel('Unseen mean IoU')
  plt.savefig(os.path.join(result_path, 'unseen_ious.png'))
  plt.close()

  return best_acc

mean_scores_val = []
mean_rank_val = []

def validation(e, best_acc):
  n_classes = val_loader.n_classes
  print('start validation epoch: %d'%e)
  net.eval()

  accum_all = torch.zeros(n_classes)
  rank_all = torch.zeros(n_classes, n_classes)
  accum_class = torch.zeros(n_classes)

  # get template latent z
  class_target = torch.LongTensor(list(range(n_classes)))
  class_template = val_loader.load_template(class_target)
  class_template = class_template.cuda()
  with torch.no_grad():
    class_recon, class_mu, class_logvar, _, pred_temp,fea_temp = net(class_template)
  

  for i, (input, target, template) in enumerate(valloader):

    target = torch.squeeze(target)
    input, template = input.cuda(), template.cuda()
    with torch.no_grad():
      recon, mu, logvar, input_stn,pred,fea = net(input)
    
    sample_correct, sample_all, sample_rank = score_NN(mu, class_mu, target, n_classes)
    accum_class += sample_correct
    accum_all += (sample_all)
    rank_all = rank_all + sample_rank
    
    print('Epoch:%d  Batch:%d/%d  processing...'%(e, i, batch_iter_val))
    
    if i < 1 and (e%save_epoch == 0) :
      out_folder =  "%s/Epoch_%d_val"%(outimg_path, e)
      out_root = Path(out_folder)
      if not out_root.is_dir():
        os.mkdir(out_root)
      torchvision.utils.save_image(input, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(input_stn, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
      torchvision.utils.save_image(recon, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2)
      torchvision.utils.save_image(template, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2)

  if e%save_epoch == 0:
    torchvision.utils.save_image(class_template, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2)  
    torchvision.utils.save_image(class_recon, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2)  

  acc_all = accum_class.sum() / accum_all.sum() 
  acc_cls = torch.div(accum_class, accum_all)

  rank_sample_avg = rank_all.sum(0) / accum_all.sum() # [class_id, topN]
  rank_cls = torch.div(rank_all, torch.transpose(accum_all.expand_as(rank_all),0,1))
  rank_cls_avg = torch.mean(rank_cls,dim=0)
  
  # write result part
  acc_trcls = torch.gather(acc_cls, 0, val_loader.tr_class)
  acc_tecls =torch.gather(acc_cls, 0, val_loader.te_class)

  print('========epoch(%d)========='%e)
  print('Seen Classes')
  for i, class_acc in enumerate(acc_trcls):
    print('cls:%d  acc:%02f'%(val_loader.tr_class[i], class_acc))
  print('Unseen Classes')
  for i, class_acc in enumerate(acc_tecls):
    print('cls:%d  acc:%02f'%(val_loader.te_class[i], class_acc))
  print('====================================')
  print('acc_avg:%02f'%acc_all)
  print('acc_cls:%02f'%acc_cls.mean())
  print('acc_trcls:%02f'%acc_trcls.mean())
  print('acc_tecls:%02f'%acc_tecls.mean())
  print('rank sample avg:%02f'%rank_sample_avg.mean())
  print('rank cls avg:%02f'%rank_cls_avg.mean())
  print('====================================')

  f_iou = open(os.path.join(result_path, "log_val_acc.txt"),'a')
  f_iou.write('epoch(%d), acc_cls: %04f  acc_trcls: %04f  acc_tecls: %04f  acc_all: %04f  top3: %04f  top5: %04f\n'%(e, acc_cls.mean(), acc_trcls.mean(), acc_tecls.mean(), acc_all, rank_sample_avg[2], rank_sample_avg[4]))
  f_iou.close()

  if best_acc < acc_tecls.mean(): # update best score
    best_acc = acc_tecls.mean()
    f_iou_class = open(os.path.join(result_path, "best_iou_val.txt"),'w')
    f_iou_class.write('Best score epoch:  %d\n'%e)
    f_iou_class.write('acc cls: %.4f  acc all: %.4f  rank sample mean: %.4f  rank cls mean: %.4f\n'%(acc_cls.mean(), acc_all, rank_sample_avg.mean(), rank_cls_avg.mean()))
    f_iou_class.write('acc tr cls: %.4f  acc te cls: %.4f\n'%(acc_trcls.mean(), acc_tecls.mean()))
    f_iou_class.write('top3: %.4f  top5: %.4f\n'%(rank_sample_avg[2], rank_sample_avg[4]))

    f_iou_class.write('\nSeen classes\n')
    for i, class_acc in enumerate(acc_trcls):
      f_iou_class.write('cls:%d  acc:%02f\n'%(val_loader.tr_class[i], class_acc))
    f_iou_class.write('\nUnseen classes\n')
    for i, class_acc in enumerate(acc_tecls):
      f_iou_class.write('cls:%d  acc:%02f\n'%(val_loader.te_class[i], class_acc))
    f_iou_class.close()
    torch.save(net.state_dict(), os.path.join('%s_valBest_net.pth'%args.dataset))

    f_rank = open(os.path.join(result_path, "best_rank_val.txt"),'w')
    for i, rank_acc in enumerate(rank_sample_avg):
      f_rank.write('rank sample %d: %.4f\n'%(i+1, rank_acc))
    f_rank.write('\n')
    for i, rank_acc in enumerate(rank_cls_avg):
      f_rank.write('rank cls %d: %.4f\n'%(i+1, rank_acc))
    f_rank.close()

  # Plot scores
  mean_scores_val.append(acc_tecls.mean())
  es = list(range(len(mean_scores_val)))
  plt.plot(es, mean_scores_val, 'b-')
  plt.xlabel('Epoch')
  plt.ylabel('Mean IoU')
  plt.savefig(os.path.join(result_path, 'unseen_ious_val.png'))
  plt.close()

  return best_acc

if __name__ == "__main__":
  out_root = Path(outimg_path)
  if not out_root.is_dir():
    os.mkdir(out_root)

  best_acc = 0
  best_acc_val = 0

  for e in range(1, args.epochs + 1):
    val_trigger = False
    train(e)
    temp_acc_val = validation(e, best_acc_val)
    if temp_acc_val > best_acc_val:
      best_acc_val = temp_acc_val
      val_trigger = True # force test function to save log when validation performance is updated
    best_acc = test(e, best_acc, val_trigger)
    