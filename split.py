import os
import sys
import re
import numpy as np
from random import shuffle

import argparse

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--k-fold', default=10, type=int, metavar='N',
                    help='number of folders you want to split into')
parser.add_argument('--filename', default='filenames.npy', type=str, metavar='PATH',
                    help='directory to load the original filenames (default: filenames.npy)')
parser.add_argument('--save-dir', default='', type=str, metavar='PATH',
                    help='directory to save the split filenames (default: filenames.npy)')

def main():
    #Split the dataset into K folders
    args = parser.parse_args()
    
    k_fold = args.k_fold
    filenames = np.load(args.filename)
    save_dir = os.path.join(args.save_dir, "split")
    
    interval = len(filenames) / k_fold
    
    for i in xrange(k_fold):
        if i == k_fold - 1:
            train_subset = filenames[:i*interval]
            test_subset = filenames[i*interval:]
        else:
            train_subset = np.concatenate((filenames[:i*interval], filenames[(i+1)*interval:]))
            test_subset = filenames[i*interval:(i+1)*interval]
        
        idics = np.arange(len(train_subset))
        #np.random.shuffle(idics)
        
        filenames_train = train_subset
        
        filenames_val = test_subset
        
        print "---------------%d th folder ----------------------" % (i)
        print "Number of training patients: ", len(filenames_train)
        print "Number of validation patients: ", len(filenames_val)
        print "Total number of patients: ", len(filenames_val)+len(filenames_train)

        np.save(os.path.join(save_dir, "train_%d" % (i)), filenames_train)

        np.save(os.path.join(save_dir, "val_%d" % (i)), filenames_val)
        print "--------------------------------------------------"

if __name__ == '__main__':
    main()
