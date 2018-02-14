import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random
from layers import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate

class DataBowl3Detector(Dataset):
    def __init__(self, data_dir, split_path, config, phase = 'train',split_comber=None):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.max_stride = config['max_stride']
        self.stride = config['stride']
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']
        self.augtype = config['augtype']
        self.pad_value = config['pad_value']
        self.split_comber = split_comber
        idcs = np.genfromtxt(split_path, dtype=str)
        if not idcs.shape:
            idcs = np.array([idcs])

        if phase!='test':
            idcs = [f for f in idcs if (f not in self.blacklist)]

        self.filenames = [os.path.join(data_dir, '%s_img.npy' % idx) for idx in idcs]

        labels = []

        for idx in idcs:
            l = np.load(os.path.join(data_dir, '%s_label.npy' %idx))
            if np.all(l==0):
                l=np.array([])
            labels.append(l)

        self.sample_bboxes = labels
        if self.phase != 'test':
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0 :
                    for t in l:
                        self.bboxes.append([np.concatenate([[i],t])])
                        # Balance samples for different nodule sizes
                        """
                        if t[3]>sizelim:
                            self.bboxes.append([np.concatenate([[i],t])])
                        if t[3]>sizelim2:
                            self.bboxes+=[[np.concatenate([[i],t])]]*2
                        if t[3]>sizelim3:
                            self.bboxes+=[[np.concatenate([[i],t])]]*4
                        """
            self.bboxes = np.concatenate(self.bboxes,axis = 0)
            print 'total number of examples ', len(self.bboxes)

        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)
        self.cache = {}

    def __getitem__(self, idx,split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))#seed according to time

        # x% of input are random img crops for negative coverage
        # see __len__
        is_random_img  = False
        if self.phase !='test':
            if idx>=len(self.bboxes):
                is_random_crop = True
                idx = idx%len(self.bboxes)
                is_random_img = np.random.randint(2)
            else:
                is_random_crop = False
                idx = np.random.randint(0, len(self.bboxes))
        else:
            is_random_crop = False

        if self.phase != 'test':
            if not is_random_img:
                bbox = self.bboxes[idx]
                filename = self.filenames[int(bbox[0])]
                start = time.time()
                if filename in self.cache:
                    imgs = self.cache[filename]
                else:
                    imgs = np.load(filename)
                    self.cache[filename] = imgs
                bboxes = self.sample_bboxes[int(bbox[0])]
                isScale = self.augtype['scale'] and (self.phase=='train')
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes,isScale,is_random_crop)
                if self.phase=='train' and not is_random_crop:
                     sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                        ifflip = self.augtype['flip'], ifrotate=self.augtype['rotate'], ifswap = self.augtype['swap'])
            else:
                randimid = np.random.randint(len(self.filenames))
                filename = self.filenames[randimid]
                if filename in self.cache:
                    imgs = self.cache[filename]
                else:
                    imgs = np.load(filename)
                    self.cache[filename] = imgs
                bboxes = self.sample_bboxes[randimid]
                isScale = self.augtype['scale'] and (self.phase=='train')
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes,isScale=False,isRand=True)
            # if sample.shape[1] != 128 or sample.shape[2] != 128 or sample.shape[3] != 128:
            #     print filename, sample.shape
            label = self.label_mapping(sample.shape[1:], target, bboxes)
            sample = (sample.astype(np.float32)-128)/128
            #if filename in self.kagglenames and self.phase=='train':
            #    label[label==-1]=0
            return torch.from_numpy(sample), torch.from_numpy(label), coord
        else:
            imgs = np.load(self.filenames[idx])
            bboxes = self.sample_bboxes[idx]
            nh, nw = imgs.shape[1:]
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0,0], [0, ph - nh], [0, pw - nw]], 'constant',constant_values = self.pad_value)

            xx,yy = np.meshgrid(np.linspace(-0.5,0.5,imgs.shape[1]/self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[2]/self.stride),indexing ='ij')
            coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...]],0).astype('float32')
            imgs, nhw = self.split_comber.split(imgs)
            coord2, nhw2 = self.split_comber.split(coord,
                                                   side_len = self.split_comber.side_len/self.stride,
                                                   max_stride = self.split_comber.max_stride/self.stride,
                                                   margin = self.split_comber.margin/self.stride)
            assert np.all(nhw==nhw2)
            imgs = (imgs.astype(np.float32)-128)/128
            #print imgs.shape
            return torch.from_numpy(imgs), bboxes, torch.from_numpy(coord2), np.array(nhw)

    def __len__(self):
        if self.phase == 'train':
            return len(self.bboxes)/(1-self.r_rand)
            return 100
        elif self.phase =='val':
            return len(self.bboxes)
            return 100
        else:
            return len(self.sample_bboxes)


def augment(sample, target, bboxes, coord, ifflip = True, ifrotate=True, ifswap = True):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand()*180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],[np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
            newtarget[1:3] = np.dot(rotmat,target[1:3]-size/2)+size/2
            if np.all(newtarget[:3]>target[3]) and np.all(newtarget[:3]< np.array(sample.shape[1:4])-newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample,angle1,axes=(2,3),reshape=False)
                coord = rotate(coord,angle1,axes=(2,3),reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat,box[1:3]-size/2)+size/2
            else:
                counter += 1
                if counter ==3:
                    break
    if ifswap:
        if sample.shape[1]==sample.shape[2] and sample.shape[1]==sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder+1]))
            coord = np.transpose(coord,np.concatenate([[0],axisorder+1]))
            target[:3] = target[:3][axisorder]
            bboxes[:,:3] = bboxes[:,:3][:,axisorder]

    if ifflip:
#         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([np.random.randint(2),np.random.randint(2)])*2-1
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1]])
        coord = np.ascontiguousarray(coord[:,::flipid[0],::flipid[1]])
        for ax in range(2):
            if flipid[ax]==-1:
                target[ax] = np.array(sample.shape[ax+1])-target[ax]
                bboxes[:,ax]= np.array(sample.shape[ax+1])-bboxes[:,ax]
    return sample, target, bboxes, coord

class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.stride = config['stride']
        self.pad_value = config['pad_value']
    def __call__(self, imgs, target, bboxes,isScale=False,isRand=False):
        if isScale:
            radiusLim = [8.,120.]
            scaleLim = [0.75,1.25]
            scaleRange = [np.min([np.max([(radiusLim[0]/target[3]),scaleLim[0]]),1])
                         ,np.max([np.min([(radiusLim[1]/target[3]),scaleLim[1]]),1])]
            scale = np.random.rand()*(scaleRange[1]-scaleRange[0])+scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float')/scale).astype('int')
        else:
            crop_size=self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)
        bboxes = np.copy(bboxes)

        start = []
        for i in range(2):
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r)+ 1 - bound_size
                e = np.ceil (target[i] + r)+ 1 + bound_size - crop_size[i]
            else:
                s = np.max([imgs.shape[i+1]-crop_size[i]/2,imgs.shape[i+1]/2+bound_size])
                e = np.min([crop_size[i]/2,              imgs.shape[i+1]/2-bound_size])
                target = np.array([np.nan,np.nan,np.nan,np.nan])
            if s>e:
                start.append(np.random.randint(e,s))#!
            else:
                start.append(int(target[i])-crop_size[i]/2+np.random.randint(-bound_size/2,bound_size/2))
            # start.append(int(s))


        normstart = np.array(start).astype('float32')/np.array(imgs.shape[1:])-0.5
        normsize = np.array(crop_size).astype('float32')/np.array(imgs.shape[1:])
        xx,yy = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],self.crop_size[0]/self.stride),
                           np.linspace(normstart[1],normstart[1]+normsize[1],self.crop_size[1]/self.stride),indexing ='ij')
        coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...]],0).astype('float32')

        pad = []
        pad.append([0,0])
        for i in range(2):
            leftpad = max(0,-start[i])
            rightpad = max(0,start[i]+crop_size[i]-imgs.shape[i+1])
            pad.append([leftpad,rightpad])
        crop = imgs[:,
            max(start[0],0):min(start[0] + crop_size[0],imgs.shape[1]),
            max(start[1],0):min(start[1] + crop_size[1],imgs.shape[2])]
        crop = np.pad(crop,pad,'constant',constant_values =self.pad_value)
        for i in range(2):
            target[i] = target[i] - start[i]

        res = []
        for i in range(len(bboxes)):
            for j in range(2):
                bboxes[i][j] = bboxes[i][j] - start[j]
            if bboxes[i][0] > 0 and bboxes[i][0] < crop_size[0] and bboxes[i][1] > 0 and bboxes[i][1] < crop_size[0]:
                res.append(bboxes[i])
        # print 'total bboxes in this patch', len(res)

        if isScale:
            res = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop,[1,scale,scale],order=1)
            newpad = self.crop_size[0]-crop.shape[1:][0]
            if newpad<0:
                crop = crop[:,:-newpad,:-newpad]
            elif newpad>0:
                pad2 = [[0,0],[0,newpad],[0,newpad]]
                crop = np.pad(crop,pad2,'constant',constant_values =self.pad_value)
            for i in range(4):
                target[i] = target[i]*scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j]*scale
                if bboxes[i][0] >= 0 and bboxes[i][0] <= crop_size[0] and bboxes[i][1] >= 0 and bboxes[i][1] <= crop_size[0]:
                    res.append(bboxes[i])

        return crop, target, res, coord

class LabelMapping(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride'])
        self.num_neg = int(config['num_neg'])
        self.th_neg = config['th_neg']
        self.anchors = np.asarray(config['anchors'])
        self.phase = phase
        if phase == 'train':
            self.th_pos = config['th_pos_train']
        elif phase == 'val':
            self.th_pos = config['th_pos_val']


    def __call__(self, input_size, target, bboxes):
        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos

        output_size = []
        for i in range(2):
            assert(input_size[i] % stride == 0)
            output_size.append(input_size[i] / stride)

        # initialize label with all -1s
        label = -1 * np.ones(output_size + [len(anchors), 5], np.float32)
        offset = ((stride.astype('float')) - 1) / 2
        oh = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)

        # to get pool of definite negatives,
        # protect all voxels in label that touch nodule with 0s
        for bbox in bboxes:
            for i, anchor in enumerate(anchors):
                ih, iw = select_samples(bbox, anchor, th_neg, oh, ow)
                label[ih, iw, i, 0] = 0

        # randomly mark NUM_NEG negatives with -1
        if self.phase == 'train' and self.num_neg > 0:
            neg_h, neg_w, neg_a= np.where(label[:, :, :, 0] == -1)
            neg_idcs = random.sample(range(len(neg_h)), min(num_neg, len(neg_h)))
            neg_h, neg_w, neg_a = neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, 0] = 0
            label[neg_h, neg_w, neg_a, 0] = -1

        if np.isnan(target[0]):
            return label

        # Label all the anchors
        bbox = target
        #for bbox in bboxes:
        ih, iw, ia = [], [], []
        for i, anchor in enumerate(anchors):
            iih, iiw = select_samples(bbox, anchor, th_pos, oh, ow)
            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iih),), np.int64))
        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)
        flag = True

        # if no anchor box gives sufficient IOU, pick closest one
        if len(ih) == 0:
            pos = []
            for i in range(2):
                pos.append(max(0, int(np.round((bbox[i] - offset) / stride))))

            #TO-DO: should change according to rectangle
            idx = np.argmin(np.abs(np.log(bbox[3] / anchors)))
            pos.append(idx)
            flag = False
        else:
            idx = random.sample(range(len(ih)), 1)[0]
            pos = [ih[idx], iw[idx], ia[idx]]
        # print pos
        dh = (bbox[0] - oh[pos[0]]) / anchors[pos[2]]
        dw = (bbox[1] - ow[pos[1]]) / anchors[pos[2]]
        dy = np.log(bbox[2] / anchors[pos[2]])
        dx = np.log(bbox[3] / anchors[pos[2]])
        label[pos[0], pos[1], pos[2], :] = [1, dh, dw, dy, dx]
        
        return label

def select_samples(bbox, anchor, th, oh, ow):
    h, w, y, x = bbox
    #TO-DO: should change according to rectangle
    max_overlap = min(y, anchor)
    min_overlap = np.power(max(y, anchor), 3) * th / max_overlap / max_overlap
    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = h - 0.5 * np.abs(y - anchor) - (max_overlap - min_overlap)
        e = h + 0.5 * np.abs(y - anchor) + (max_overlap - min_overlap)
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]

        s = w - 0.5 * np.abs(x - anchor) - (max_overlap - min_overlap)
        e = w + 0.5 * np.abs(x - anchor) + (max_overlap - min_overlap)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64)

        lh, lw = len(ih), len(iw)
        ih = ih.reshape((-1, 1))
        iw = iw.reshape((1, -1))
        ih = np.tile(ih, (1, lw)).reshape((-1))
        iw = np.tile(iw, (lh, 1)).reshape((-1))
        centers = np.concatenate([
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis = 1)

        r0 = anchor / 2
        s0 = centers - r0
        e0 = centers + r0

        r1 = [y / 2, x / 2]
        s1 = bbox[:2] - r1
        s1 = s1.reshape((1, -1))
        e1 = bbox[:2] + r1
        e1 = e1.reshape((1, -1))

        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))

        intersection = overlap[:, 0] * overlap[:, 1]
        union = anchor * anchor + x * y - intersection

        iou = intersection / union

        mask = iou >= th
        #if th > 0.4:
         #   if np.sum(mask) == 0:
          #      print(['iou not large', iou.max()])
           # else:
            #    print(['iou large', iou[mask]])
        ih = ih[mask]
        iw = iw[mask]
        return ih, iw

def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

