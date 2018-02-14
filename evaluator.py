import os
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_curve, roc_auc_score
sys.path.append('../tcai17/')
import utils
import matplotlib.pyplot as plt
import time

def is_hit(c_hat, c, d):
    """
    Hit criterion for LUNA16.

    c_hat: predicted center x,y,z
    c:     ground truth center
    d:     diameter of ground truth nodule
    """
    return np.linalg.norm(c_hat - c) < d/2.0

def s_to_p(s):
    """Convert scores array to probabilities."""
    from torch.autograd import Variable
    from torch import from_numpy
    from torch.nn import Sigmoid
    m = Sigmoid()
    input = Variable(from_numpy(s))
    return  m(input).data.numpy()

class Evaluator:
    """Suite of analytics for completed predictions.

    E.g., generate FROC statistics, get false positives, etc.
    """

    BASE = '/home/htang6/workspace/nod/training/detector/'

    # dir of preprocessed imgs to load spacing, ebox_origin, origin
    PREP = '/home/htang6/preprocess_result'

    # dir of .mhd raw files
    RAW = '/home/htang6/home/htang6/14HefeiCT/raw'

    def __init__(self, results_name, test_set, filenames, ckpt=None, pbb_cutoff=None, topk = None, classifier_pred_path = None):
        """
        results_name: dir with saved predictions, eg '05-31-17'
        test_set: 'test', 'val', 'train'
        ckpt: int for name of bbox dir, eg 061817/bbox/test2_100; none if just 'test2'
        pbb_cutoff: only load pbbs with prob > cutoff for efficiency
        """
        print 'WARNING: USING',self.PREP,'FOR SPACING, ORIGIN, EBOX_ORIGIN'
        self.test_set = test_set
        self.results_dir = os.path.join(self.BASE, 'results', results_name)
        self.bbox_dir = os.path.join(self.BASE, results_name)
        self.pbb_cutoff = pbb_cutoff
        self.topk = topk
        self.classifier_pred_path = classifier_pred_path

        sd = os.path.join(self.BASE, 'results', results_name, 'sub')
        if not os.path.exists(sd):
            os.makedirs(sd)

        self.filenames = np.load(os.path.join(self.BASE, filenames))

        if test_set == 'train' or test_set == 'val':
            self.generate_stats()

    def generate_stats(self):
        """Store PBBs in dataframe (self.pbbs) along with ground truth binary labels.

        self.n_annot : total number of nodule annotations

        self.rel = pbbs_df[pbbs_df['nod']==1]
        self.irr = pbbs_df[pbbs_df['nod']==0]
        self.lbbs_not_in_pbbs = lbbs_not_in_pbbs_df

        classifier_pred_path: optionally include classifier prediction scores
        """

        if self.test_set == 'test':
            print 'Error: test set has no labels'
            return

        lbbs_not_in_pbbs_df = pd.DataFrame(columns=['pid','z','y','x','d'])
        if self.classifier_pred_path is None:
            pbbs_df = pd.DataFrame(columns=['pid','prob','z','y','x','d','nod'])
        else:
            pbbs_df = pd.DataFrame(columns=['pid','prob','z','y','x','d','nod','c_prob'])

        n_annot = 0
        for name in self.filenames:
            #print name
            pbb = np.load(os.path.join(self.bbox_dir, name+'_pbb.npy'))
            # add nod
            pbb = np.concatenate([pbb, np.zeros((pbb.shape[0],1))], axis=1)

            # Include classifier scores
            # Use nan for patients that got pbbs but not classifier predictions
            # eg blacklist
            if self.classifier_pred_path is not None:
                pred_fname = os.path.join(self.classifier_pred_path, name+'_pred.npy')
                if os.path.exists(pred_fname):
                    cl_scores = np.load(pred_fname)
                else:
                    cl_scores = np.empty((pbb.shape[0],1))
                    cl_scores[:] = np.nan

                pbb = np.concatenate([pbb,cl_scores], axis=1)
            pbb = pbb[pbb[:,0].argsort()][::-1]
            lbb = np.load(os.path.join(self.bbox_dir, name+'_lbb.npy'))
            n_annot += len(lbb)
            lab_hits = np.zeros(len(lbb))

            # determine ground truth label of pbb
            # exclude relevant pbbs that are redundant for purposes of FROC

            #print 'pbb len', len(pbb)
            it = range(len(pbb)) if self.topk is None else range(min(len(pbb),self.topk))
            for i in it:

                if self.pbb_cutoff is not None and pbb[i,0] < self.pbb_cutoff:
                    break

                lbb_match = False
                redundant_hit = False
                for j in range(len(lbb)):
                    if is_hit(pbb[i][1:4], lbb[j][:3], lbb[j][3]):
                        if lab_hits[j] > 0:
                            redundant_hit = True
                            #print 'redundant tp!'
                            #print name, 'pbb', pbb[i], 'lbb', lbb[j]
                            #tp.append(pbb[i])
                        lab_hits[j] += 1
                        lbb_match = True
                        break
                if lbb_match:
                    pbb[i,5] = 1
                else:
                    pbb[i,5] = 0

                if not redundant_hit:
                    pbbs_df.loc[len(pbbs_df)] = [name] + list(pbb[i])
            missed = pd.DataFrame(columns=list('zyxd'), data = lbb[lab_hits == 0].reshape(-1,len(list('zyxd'))))
            missed['pid'] = name
            missed = missed[['pid','z','y','x','d']]
            lbbs_not_in_pbbs_df = pd.concat([lbbs_not_in_pbbs_df,missed], ignore_index=True)


        # convert scores to probabilities
        pbbs_probs = s_to_p(np.array(pbbs_df['prob']))
        pbbs_df['prob'] = pbbs_probs

        if self.classifier_pred_path is not None:
            pbbs_cprobs = s_to_p(np.array(pbbs_df['c_prob']))
            pbbs_df['c_prob'] = pbbs_cprobs

            # ensemble
            pbbs_df['ensemble'] = (pbbs_df['prob'] + pbbs_df['c_prob'])/2.0



        self.n_annot = n_annot
        self.pbbs = pbbs_df
        self.rel = pbbs_df[pbbs_df['nod']==1]
        self.irr = pbbs_df[pbbs_df['nod']==0]
        self.lbbs_not_in_pbbs = lbbs_not_in_pbbs_df
        print 'loaded {} pbbs'.format(len(pbbs_df))
        if self.test_set == 'train' or self.test_set == 'val':
            print 'saved pbbs missed {} out of {} annotations ({:.2%})'.format(len(lbbs_not_in_pbbs_df),
                                                                       n_annot,
                                                                           1.0 * len(lbbs_not_in_pbbs_df)/n_annot)


    def roc(self, by = 'prob', neg_th = -1):
        """Print roc statistics.

        by: 'c_prob' (classifier), 'prob' (detector), 'ensemble' probabilities to use for ranking.
        neg_th: minimum probability of non-nodule pbbs to include; should match negative set
                used for classifier training to compare AUROCs. without threshold, AUROC
                meaningless because most pbbs are very low probability nodules."""

        if self.test_set == 'test':
            print 'Error: test set has no labels'
            return

        neg = self.pbbs[self.pbbs['prob'] > neg_th]
        neg = neg[neg['nod']==0]
        pos = self.pbbs[self.pbbs['nod']==1]
        df = pd.concat([neg,pos])


        print 'AUROC: {:.3f}'.format(roc_auc_score(df['nod'],df[by]))

        fpr, tpr, ths = roc_curve(df['nod'],df[by])
        return fpr, tpr, ths


    def froc(self, ignore=[], n_scans=None):
        """Print FROC statistics and return (fp_per_scan,
                                             TPRs,
                                             probability_thresholds).
        ignore: list of patients to ignore pbbs for FROC
        n_scans: # scans used for fp_per_scan. If None, use len(self.filenames)
        """

        if self.test_set == 'test':
            print 'Error: test set has no labels'
            return

        irr = self.irr.loc[~self.irr['pid'].isin(ignore)]
        rel = self.rel.loc[~self.rel['pid'].isin(ignore)]

        irr = np.array(irr['prob'])
        rel = np.array(rel['prob'])
        irr = irr[irr.argsort()][::-1]
        rel = rel[rel.argsort()][::-1]
        tprs = []
        p_ths = []
        fp_per_scan = [1.0/8, 1.0/4, 1.0/2, 1.0, 2.0, 4.0, 8.0]
        if n_scans is None:
            n_scans = len(self.filenames)
        for nlf in fp_per_scan:
            irr_i = int(np.round(nlf * n_scans))
            # if not enough false positives, assume padded false positive list
            # with p=0
            prob_th = 0 if irr_i >= len(irr) else irr[irr_i]
            tpr = np.sum(rel > prob_th)/(1.0 * self.n_annot)
            tprs.append(tpr)
            p_ths.append(prob_th)
            print 'NLF: {}, TPR: {}, PROB_TH: {}'.format(nlf, tpr, prob_th)
        print '======'
        print 'avg TPR: {}'.format(np.mean(tprs))

        return (fp_per_scan, tprs, p_ths, np.mean(tprs))

        #plt.plot(fp_per_scan, tprs)
        #plt.show()

    def sub(self, ignore=[], k=30, save=False):
        """
        Return submission dataframe and/or save to csv.
        Use top k predictions per patient.

        ignore: list of patients to ignore pbbs for FROC

        filenames:  patient names for testing, eg. "LKDS-00002"
        bbox_dir: contains predicted bounding boxes, eg "LKDS-00002_pbb.npy"
        self.save_dir:   path to save csv file (None if not saving)
        self.PREP:  dir of preprocessed imgs to load spacing, ebox_origin, origin

        """
        #TODO: Use probability threshold instead of top k
        print 'WARNING: USING TOP K={} INSTEAD OF PROBABILITY THRESHOLD. FIX THIS!'.format(k)

        submission = []

        for name in self.filenames:
            if name in ignore:
                continue
            pbb = np.load(os.path.join(self.bbox_dir, "%s_pbb.npy" % (name)))
            pbb = pbb[pbb[:, 0].argsort()][::-1][:k]
            spacing = np.load(os.path.join(self.PREP, name + '_spacing.npy'))
            ebox_origin = np.load(os.path.join(self.PREP, name + '_ebox_origin.npy'))
            origin = np.load(os.path.join(self.PREP, name + '_origin.npy'))
            #print spacing

            for p in pbb:
                ebox_coord = p[[1, 2, 3]]
                whole_img_coord = ebox_coord + ebox_origin
                worldCoord = utils.voxelToWorldCoord(whole_img_coord, origin, spacing)
                submission.append([name, worldCoord[2], worldCoord[1], worldCoord[0], p[0]])

        submission = pd.DataFrame(submission, columns = ["seriesuid", "coordX", "coordY", "coordZ",
                                                         "probability"])
        scores = np.array(submission['probability'])
        probs = s_to_p(scores)
        submission['probability'] = probs

        if save:
            print "Saving submission to", self.save_dir
            submission.to_csv(self.save_dir, sep=',', index=False)
        return submission

    def fig(self, ix, raw=False, alt_df=None):
        """Generate interactive plot of a specified target.

        alt_df: df to use for fig; None if using self.pbb
        ix: index of the dataframe to plot
        raw: slider for raw image
        """

        if alt_df is None:
            df = self.pbbs
        else:
            df = alt_df

        row = df.loc[ix]
        print row
        pid, z, y, x, d = row['pid'], row['z'],\
                          row['y'],row['x'],row['d']

        import IPython.html.widgets as w
        if not raw:
            img = np.load(os.path.join(self.PREP, pid + '_clean.npy'))
        else:
            img, origin, spacing = utils.load_itk_image(os.path.join(self.RAW,pid + '.mhd'))

            # convert z,y,x,d to raw voxel coordinates
            v = np.array([z,y,x])
            ebox_origin = np.load(os.path.join(self.PREP,pid+'_ebox_origin.npy'))
            v = v + ebox_origin
            prep_spacing = np.load(os.path.join(self.PREP,pid+'_spacing.npy'))
            v = utils.voxelToWorldCoord(v, origin, prep_spacing)
            v = utils.worldToVoxelCoord(v, origin, spacing)
            z, y, x = v[0], v[1], v[2]
            d = d * prep_spacing[1] / spacing[1]

            # convert title
            row = row.copy()
            row['z'] = z
            row['y'] = y
            row['x'] = x
            row['d'] = d

        def fz(k):
            utils.showTargetImgComp(img, [k,y,x], plt, d=d, t=str(row))
        def fy(k):
            utils.showTargetImgComp(np.swapaxes(img,0,1), [k,z,x], plt, d=d, t=str(row))
        def fx(k):
            utils.showTargetImgComp(np.swapaxes(img,0,2), [k,y,z], plt, d=d, t=str(row))
        w.interact(fz, k=w.IntSlider(min=0,max=img.shape[0]-1,step=1,value=z))
        w.interact(fy, k=w.IntSlider(min=0,max=img.shape[1]-1,step=1,value=y))
        w.interact(fx, k=w.IntSlider(min=0,max=img.shape[2]-1,step=1,value=x))

class LunaEval(Evaluator):
    """Evaluates statistics for 10-fold CV test results by pooling
    the predictions for each subset.

    """
    # TODO: Current backwards compatible methods: froc(), fig()
    BASE = '/home/danielrk/lfz_luna/training/detector/results/'
    PREP = '/home/danielrk/lung/input/luna16/lfz_preprocess/'
    RAW = '/home/danielrk/lung/input/luna16/combined/'
    ROSTERS = '/home/danielrk/lung/input/luna16/'
    ABBREV = '/home/danielrk/lfz_luna/training/detector/labels/shorter.csv'
    def __init__(self, pbb_cutoff=None):

        self.test_set = 'train' # compatibility
        self.pbb_cutoff = pbb_cutoff

        self.generate_stats()

    def generate_stats(self):

        lbbs_not_in_pbbs_df = pd.DataFrame(columns=['pid','z','y','x','d'])
        pbbs_df = pd.DataFrame(columns=['pid','prob','z','y','x','d','nod'])

        n_annot = 0
        for k in range(10):
            filenames = np.load(os.path.join(self.ROSTERS, 'subset'+str(k)+'.npy'))
            bbox_dir = os.path.join(self.BASE,'luna_'+str(k),'bbox')
            for name in filenames:
                #print k, name
                pbb = np.load(os.path.join(bbox_dir, name+'_pbb.npy'))
                # add nod
                pbb = np.concatenate([pbb, np.zeros((pbb.shape[0],1))], axis=1)

                pbb = pbb[pbb[:,0].argsort()][::-1]
                lbb = np.load(os.path.join(bbox_dir, name+'_lbb.npy'))
                n_annot += len(lbb)
                lab_hits = np.zeros(len(lbb))

                # determine ground truth label of pbb
                # exclude relevant pbbs that are redundant for purposes of FROC
                #print 'pbb len', len(pbb)

                for i in range(len(pbb)):

                    if self.pbb_cutoff is not None and pbb[i,0] < self.pbb_cutoff:
                        break

                    lbb_match = False
                    redundant_hit = False
                    for j in range(len(lbb)):
                        if is_hit(pbb[i][1:4], lbb[j][:3], lbb[j][3]):
                            if lab_hits[j] > 0:
                                redundant_hit = True
                                #print 'redundant tp!'
                                #print name, 'pbb', pbb[i], 'lbb', lbb[j]
                                #tp.append(pbb[i])
                            lab_hits[j] += 1
                            lbb_match = True
                            break
                    if lbb_match:
                        pbb[i,5] = 1
                    else:
                        pbb[i,5] = 0

                    if not redundant_hit:
                        pbbs_df.loc[len(pbbs_df)] = [name] + list(pbb[i])
                #print lab_hits
                #print lbb
                missed = pd.DataFrame(columns=list('zyxd'), data = lbb[lab_hits == 0].reshape(-1,len(list('zyxd'))))
                missed['pid'] = name
                missed = missed[['pid','z','y','x','d']]
                lbbs_not_in_pbbs_df = pd.concat([lbbs_not_in_pbbs_df,missed], ignore_index=True)


        # convert scores to probabilities
        pbbs_probs = s_to_p(np.array(pbbs_df['prob']))
        pbbs_df['prob'] = pbbs_probs


        self.n_annot = n_annot
        self.pbbs = pbbs_df
        self.rel = pbbs_df[pbbs_df['nod']==1]
        self.irr = pbbs_df[pbbs_df['nod']==0]
        self.lbbs_not_in_pbbs = lbbs_not_in_pbbs_df
        print 'loaded {} pbbs'.format(len(pbbs_df))
        print 'saved pbbs missed {} out of {} annotations ({:.2%})'.format(len(lbbs_not_in_pbbs_df),
                                                                       n_annot,
                                                                       1.0 * len(lbbs_not_in_pbbs_df)/n_annot)

    def froc(self):
        n_scans = 888
        print 'assuming n_scans =', n_scans
        return Evaluator.froc(self, n_scans=n_scans)


    def sub(self, topk=30, save_dir=None):
        """
        Return submission dataframe and/or save to csv.
        Use top k predictions per patient.

        filenames:  patient names for testing, eg. "LKDS-00002"
        bbox_dir: contains predicted bounding boxes, eg "LKDS-00002_pbb.npy"
        save_dir:   path to save csv file (None if not saving)
        self.PREP:  dir of preprocessed imgs to load spacing, ebox_origin, origin

        """


        abbrevs = np.array(pd.read_csv(self.ABBREV, header=None))
        submission = []

        for k in range(10):
            filenames = np.load(os.path.join(self.ROSTERS, 'subset'+str(k)+'.npy'))
            bbox_dir = os.path.join(self.BASE,'luna_'+str(k),'bbox')
            for name in filenames:
                pbb = np.load(os.path.join(bbox_dir, "%s_pbb.npy" % (name)))
                pbb = pbb[pbb[:, 0].argsort()][::-1][:topk]
                spacing = np.load(os.path.join(self.PREP, name + '_spacing.npy'))
                ebox_origin = np.load(os.path.join(self.PREP, name + '_ebox_origin.npy'))
                origin = np.load(os.path.join(self.PREP, name + '_origin.npy'))
                #print spacing

                for p in pbb:
                    ebox_coord = p[[1, 2, 3]]
                    whole_img_coord = ebox_coord + ebox_origin
                    worldCoord = utils.voxelToWorldCoord(whole_img_coord, origin, spacing)
                    submission.append([abbrevs[int(name),1], worldCoord[2], worldCoord[1], worldCoord[0], p[0]])

        submission = pd.DataFrame(submission, columns = ["seriesuid", "coordX", "coordY", "coordZ",
                                                         "probability"])
        scores = np.array(submission['probability'])
        probs = s_to_p(scores)
        submission['probability'] = probs

        if save_dir is not None:
            print "Saving submission..."
            submission.to_csv(save_dir, sep=',', index=False)
        return submission


