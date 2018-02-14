import torch
import numpy as np
class SplitComb():
    def __init__(self,side_len,max_stride,stride,margin,pad_value):
        self.side_len = side_len
        self.max_stride = max_stride
        self.stride = stride
        self.margin = margin
        self.pad_value = pad_value
        
    def split(self, data, side_len = None, max_stride = None, margin = None):
        if side_len==None:
            side_len = self.side_len
        if max_stride == None:
            max_stride = self.max_stride
        if margin == None:
            margin = self.margin
        
        assert(side_len > margin)
        assert(side_len % max_stride == 0)
        assert(margin % max_stride == 0)

        splits = []
        _, h, w = data.shape

        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))
        
        nhw = [nh,nw]
        self.nhw = nhw
        
        pad = [ [0, 0],
                [margin, nh * side_len - h + margin],
                [margin, nw * side_len - w + margin]]
        data = np.pad(data, pad, 'edge')

        for ih in range(nh):
            for iw in range(nw):
                sh = ih * side_len
                eh = (ih + 1) * side_len + 2 * margin
                sw = iw * side_len
                ew = (iw + 1) * side_len + 2 * margin

                split = data[np.newaxis, :, sh:eh, sw:ew]
                splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits,nhw

    def combine(self, output, nhw = None, side_len=None, stride=None, margin=None):
        
        if side_len==None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        if nhw is None:
            nh = self.nhw[0]
            nw = self.nhw[1]
        else:
            nh,nw = nhw
        assert(side_len % stride == 0)
        assert(margin % stride == 0)
        side_len /= stride
        margin /= stride

        splits = []
        for i in range(len(output)):
            splits.append(output[i])

        # print len(splits)
        # print splits[0].shape
        output = -1000000 * np.ones((
            nh * side_len,
            nw * side_len,
            splits[0].shape[2],
            splits[0].shape[3]), np.float32)

        idx = 0
        for ih in range(nh):
            for iw in range(nw):
                sh = ih * side_len
                eh = (ih + 1) * side_len
                sw = iw * side_len
                ew = (iw + 1) * side_len

                split = splits[idx][margin:margin + side_len, margin:margin + side_len, margin:margin + side_len]
                output[sh:eh, sw:ew] = split
                idx += 1

        return output 
