import os
import argparse
import tqdm
import os
import argparse
import numpy as np
import tqdm
from itertools import chain
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from utils import CLEFImage, weights_init, print_args
from model.net import ResBase50, ResClassifier, grad_reverse

parser = argparse.ArgumentParser()
parser.add_argument("--data_list_root", default='data/OfficeHome/list')
parser.add_argument("--data_root", default='data/OfficeHome')
parser.add_argument("--source", default='Art')
parser.add_argument("--target", default='Clipart')
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=0)
parser.add_argument("--epoch", default=150, type=int)
parser.add_argument("--snapshot", default="")
parser.add_argument("--lr", default=0.001)
parser.add_argument("--class_num", default=65)
parser.add_argument("--extract", default=True)
parser.add_argument("--weight_entropy", default=0.1)
parser.add_argument("--dropout_p", default=0.5)
parser.add_argument("--task", default='None', type=str)
parser.add_argument("--post", default='-1', type=str)
parser.add_argument("--repeat", default='-1', type=str)
parser.add_argument("--num_k", default=1)
parser.add_argument("--result", default='record')
args = parser.parse_args()
print_args(args)

source_root = os.path.join(args.data_root, args.source)
source_label = os.path.join(args.data_list_root, args.source+'.txt')
target_root = os.path.join(args.data_root, args.target)
target_label = os.path.join(args.data_list_root, args.target+'.txt')

train_transform = transforms.Compose([
    transforms.Scale((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

source_set = CLEFImage(source_root, source_label, train_transform)
target_set = CLEFImage(target_root, target_label, train_transform)

source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)
target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)

netG = ResBase50().cuda()
netF = ResClassifier(class_num=args.class_num, extract=args.extract, dropout_p=args.dropout_p).cuda()
netF.apply(weights_init)


def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss

def get_entropy_loss(p_softmax):
    mask = p_softmax.ge(0.000001)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return args.weight_entropy*entropy / float(p_softmax.size(0))

opt_g = optim.SGD(netG.parameters(), lr=args.lr, weight_decay=0.0005)
opt_f = optim.SGD(netF.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
max_correct = 0
max_epoch = 0

for epoch in range(1, args.epoch+1):
    source_loader_iter = iter(source_loader)
    target_loader_iter = iter(target_loader)
    print(">>training " + args.task + " epoch : " + str(epoch))

    netG.train()
    netF.train()
    for i, (t_imgs, _) in tqdm.tqdm(enumerate(target_loader_iter)):
        try:
            s_imgs, s_labels = source_loader_iter.next()
        except:
            source_loader_iter = iter(source_loader)
            s_imgs, s_labels = source_loader_iter.next()

        if s_imgs.size(0) != args.batch_size or t_imgs.size(0) != args.batch_size:
            continue

        s_imgs = Variable(s_imgs.cuda())
        s_labels = Variable(s_labels.cuda())     
        t_imgs = Variable(t_imgs.cuda())
        
        opt_g.zero_grad()
        opt_f.zero_grad()

        for i in range(args.num_k):
        	t_bottleneck = netG(t_imgs, reversed=False)
        	_,t_logit = netF(t_bottleneck)
        	t_prob = F.softmax(t_logit)
        	t_entropy_loss = get_entropy_loss(t_prob)
        	loss = t_entropy_loss
        	loss.backward()
        	opt_g.step()
        	opt_f.step()
        	opt_g.zero_grad()
        	opt_f.zero_grad()

        s_bottleneck = netG(s_imgs)
        _,s_logit = netF(s_bottleneck)
        s_cls_loss = get_cls_loss(s_logit, s_labels)
        loss = s_cls_loss
        loss.backward()
        opt_g.step()
        opt_f.step()
        opt_g.zero_grad()
        opt_f.zero_grad()
 
        s_bottleneck = netG(s_imgs, reversed=False)
        t_bottleneck = netG(t_imgs, reversed=True) 
        _,s_logit = netF(s_bottleneck)
        _,t_logit = netF(t_bottleneck)  
        s_cls_loss = get_cls_loss(s_logit, s_labels)

        t_prob = F.softmax(t_logit)
        t_entropy_loss = get_entropy_loss(t_prob)
        loss = s_cls_loss +  t_entropy_loss
        loss.backward()
        opt_g.step()
        opt_f.step()
    
    # if epoch % 1 == 0:
    #     torch.save(netG.state_dict(), os.path.join(args.snapshot, "ImageHome_IAFN_" + args.task + "_netG_" + args.post + '.' + args.repeat + '_' + str(epoch) + ".pth"))
    #     torch.save(netF.state_dict(), os.path.join(args.snapshot, "ImageHome_IAFN_" + args.task + "_netF_" + args.post + '.' + args.repeat + '_'  + str(epoch) + ".pth"))
    netG.eval()
    netF.eval()
    correct = 0
    t_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    s_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    feat_s_list = []
    feat_t_list = []
    feat_s_label_list = []
    feat_t_label_list = []
    source_loader_iter = iter(s_loader)
    target_loader_iter = iter(t_loader)
    # for (t_imgs, t_labels) in t_loader:
    for i, (t_imgs, t_labels) in tqdm.tqdm(enumerate(target_loader_iter)):
        s_imgs, s_labels = source_loader_iter.next()
        t_imgs = Variable(t_imgs.cuda())
        t_bottleneck = netG(t_imgs, reversed=False)
        t_fc2_emb, t_logit = netF(t_bottleneck)
        s_imgs = Variable(s_imgs.cuda())

        pred = F.softmax(t_logit)
        pred = pred.data.cpu().numpy()
        pred = pred.argmax(axis=1)
        t_labels = t_labels.numpy()
        correct += np.equal(t_labels, pred).sum()
        if len(feat_t_list) == 0:
            feat_t_list = torch.Tensor.cpu(t_bottleneck).detach().numpy()
            feat_t_label_list = t_labels
            feat_s_list = torch.Tensor.cpu(netG(s_imgs)).detach().numpy()
            feat_s_label_list = s_labels.numpy()
        else:
            feat_t_list = np.concatenate((feat_t_list, torch.Tensor.cpu(t_bottleneck).detach().numpy()), axis=0)
            feat_t_label_list = np.concatenate((feat_t_label_list, t_labels), axis=0)
            feat_s_list = np.concatenate((feat_s_list, torch.Tensor.cpu(netG(s_imgs)).detach().numpy()), axis=0)
            feat_s_label_list = np.concatenate((feat_s_label_list, s_labels.numpy()), axis=0)
    t_imgs = []
    t_bottleneck = []
    t_logit = []
    t_fc2_emb = []
    pred = []
    t_labels = []

    # compute classification accuracy for target domain
    correct = correct * 1.0 / len(target_set)
    # correct_array.append(correct)

    if correct >= max_correct:
        max_correct = correct
        max_epoch = epoch
        if args.save:
            np.save("tsnedata/"+args.task + '_TSL3_feat_s_list_max.npy', feat_s_list) 
            np.save("tsnedata/"+args.task + '_TSL3_feat_t_list_max.npy', feat_t_list)
            np.save("tsnedata/"+args.task + '_TSL3_feat_s_label_list_max.npy', feat_s_label_list) 
            np.save("tsnedata/"+args.task + '_TSL3_feat_t_label_list_max.npy', feat_t_label_list)
    print ("Epoch {0} accuray: {1}; max acc: {2} @ {3}".format(epoch, correct, max_correct, max_epoch))
    result = open(os.path.join(args.result, "OfficeHome_sourceonly_" + args.task + '_' + args.post + '.' + args.repeat +"_score.txt"), "a")
    result.write("Epoch " + str(epoch) + ": " + str(correct) + "\n")
    result.close()

print ("Max: {0}".format(max_correct))
result = open(os.path.join(args.result, "OfficeHome_TSL_" + args.task+ "_revLayer_" + args.revLayer + '_' + args.post + '.' + args.repeat +"_score.txt"), "a")
result.write("Max: {0}".format(max_correct))
result.close()
