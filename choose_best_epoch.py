import numpy as np
import sys
import evaluator
import matplotlib.pyplot as plt

start = 100
end = 150
interval = 5

def main(start, end, interval, kfold):
    tpr_lists = []
    pth_lists = []
    tpr_lists_not_ign = []
    pth_lists_not_ign = []
    ckpts = range(start, end + 1, interval)
    
    best_score = 0
    best_epoch = "100"

    for i in ckpts:
        e = evaluator.Evaluator('val/%s/%s/bbox/test_%s' % (kfold, i, i), 'val', 'split/val_%s.npy' % (kfold), ckpt=i,pbb_cutoff=0.5)
#     fp_per_scan, tprs, pths, score = e.froc(ignore=ignore_val)
#     tpr_lists.append(tprs)
#     pth_lists.append(pths)
    
        _, tprs_not_ign, pths_not_ign, score = e.froc()
        tpr_lists_not_ign.append(tprs_not_ign)
        pth_lists_not_ign.append(pths_not_ign)
        
        if score > best_score:
	    best_score, best_epoch = score, i
        print i, 'all:', score

    print "Best score: ", best_score, " Best epoch: ", best_epoch
    return best_epoch

if __name__ == '__main__':
    print  main(start, end, interval, sys.argv[1])
