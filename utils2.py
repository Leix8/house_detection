import numpy as np
import scipy.ndimage
import pandas as pd
import SimpleITK as sitk
import os
from PIL import Image
import dicom
import matplotlib.patches as patches
#import matplotlib.pyplot as plt

def showTargetImgComp(img, target, plt, targets=None, annos=None, d=None, t=None, voxelWidth=128, box=False):
    """
    Given patient CT scan, plot the area where target nodule resides, all input shape follows [z, y, x]

    img: The numpy array of image, input shape should follow [z, y, x]
    target: the coordinates of the target nodule, shape should follow [z, y, x, d]
    plt: matplotlib.pyplot module object
    d: draw rect around target of side length d; None if unwanted
    t: title of figure
    offset: The of set of z axis for showing the next and last few slices
    voxelWidth: The width of the bounded box
    """
    plt.rcParams['figure.figsize'] = (16.0, 16.0)
    fig,ax = plt.subplots(1,1)

    #s0 = max(int(np.round(target[0]-voxelWidth/2.0)),0)
    #e0 = min(int(np.round(target[0]+voxelWidth/2.0)), img.shape[0])

    s1 = max(int(np.round(target[0]-voxelWidth/2.0)),0)
    e1 = min(int(np.round(target[0]+voxelWidth/2.0)), img.shape[0])
    s2 = max(int(np.round(target[1]-voxelWidth/2.0)),0)
    e2 = min(int(np.round(target[1]+voxelWidth/2.0)), img.shape[1])
    yx_patch = img[s1:e1, s2:e2, :]
    #zx_patch = img[s0:e0, int(np.round(target[1]) + i), s2:e2]
    if d is not None:
        # top-left corner coord wrt patch
        #c0 = max(int(np.round(target[0] - d/2.0) - s0), 0)
        c1 = max(int(np.round(target[0] - d/2.0) - s1), 0)
        c2 = max(int(np.round(target[1] - d/2.0) - s2), 0)

        yx_nod_rect = patches.Rectangle((c2,c1),d,d,
                             linewidth=1,edgecolor='springgreen',facecolor='none')
        ax.add_patch(yx_nod_rect)

    if annos is not None:
        for t in annos:
            y = t[0] - target[0] + voxelWidth / 2 - t[2] / 2.0
            x = t[1] - target[1] + voxelWidth / 2 - t[2] / 2.0
            yx_nod_rect = patches.Rectangle((x, y),t[2],t[2],
                     linewidth=1,edgecolor='red',facecolor='none')
            ax.add_patch(yx_nod_rect)

    if targets is not None:
        for t in targets:
            y = t[0] - target[0] + voxelWidth / 2 - t[2] / 2.0
            x = t[1] - target[1] + voxelWidth / 2 - t[2] / 2.0
            yx_nod_rect = patches.Rectangle((x, y),t[2],t[2],
                     linewidth=1,edgecolor='springgreen',facecolor='none')
            ax.add_patch(yx_nod_rect)


    ax.imshow(yx_patch, cmap='gray')


    yx_rect = patches.Rectangle((s2,s1),voxelWidth,voxelWidth,
                             linewidth=1,edgecolor='r',facecolor='none')
    #zx_rect = patches.Rectangle((s2,s0),voxelWidth,voxelWidth,
    #                         linewidth=1,edgecolor='r',facecolor='none')
    if box:
        ax[1].add_patch(yx_rect)
    #ax[1,1].add_patch(zx_rect)
    # ax[1].imshow(img[int(np.round(target[0]))], cmap="gray")
    #ax[1,1].imshow(img[:,int(np.round(target[1]) + i),:], cmap="gray")
    if t is not None:
        plt.title(t)
    plt.show()

def convertToInt8(img):
    #Convert the pixel value to unsigned int8, ranging from 0 to 255
    img = (normalizePlanes(img)*255).astype("uint8")
    return img

def preprocess(img):
    img[img < -1000] = -1000
    img[img > 400] = 400
    return img

def load_itk_image(filename):
    #Load .mhd format image
    itkimage = sitk.ReadImage(filename)
    # print itkimage
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing

def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def voxelToWorldCoord(voxelCoord, origin, spacing):
    worldCoord = voxelCoord * spacing
    worldCoord += origin
    return worldCoord

def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def generateSubmission(filenames, k, bboxes_dir, img_dir, spacings, save_dir):
    """
    Generate the submission.csv for TCAI17

    filenames: All the patients name for testing, eg. "LKDS-00002"
    k: Top k predicted bboxes per patient
    bboxes_dir: The directory for stroing all the bounded boxes, all the bounded boxes should be named "LKDS-00002_pbb.npy"
    img_dir: The directory of the original .mhd, because we need the original origin information
    spacings: Numpy array of all the spacing information of patients
    save_dir: Where do you want to save the submission.csv file
    """
    submission = []

    for patient in filenames:
        pbb = np.load(os.path.join(bboxes_dir, "%s_pbb.npy" % (patient)))
        pbb = pbb[pbb[:, 0].argsort()][::-1][:k]
        spacing = spacings[spacings[:, 0] == patient][:, [1, 2, 3]]
        #print spacing

        _, numpyOrigin, _ = utils.load_itk_image(os.path.join(img_dir, "%s.mhd" % (patient)))

        for p in pbb:
            voxelCoord = p[[1, 2, 3]]
            worldCoord = utils.voxelToWorldCoord(voxelCoord, numpyOrigin, spacing)
            submission.append([patient, worldCoord[0][2], worldCoord[0][1], worldCoord[0][0], p[0]])

        submission = pd.DataFrame(submission, columns = ["seriesuid", "coordX", "coordY", "coordZ", "probability"])
        print "Saving submission..."
        submission.to_csv(save_dir, sep=',', index=False)

def resample(image, spacing, new_spacing=[1,1,1]):
    """
    Given image array, return (resampled image, new spacing).
    Steps:
    - Compute dimensions of image in mm x mm x mm
    - Divide by ideal new_spacing (eg 1mm x 1mm x 1mm) to get new shape
    - Round new shape (must be integers)
    - Compute "real" new_spacing using rounded new shape
    - Zoom array into new_spacing
    - Return (new_image, new_spacing)
    Can't guarantee new_spacing (because new shape has to be int),
    but something close.
    """


    # real dimensions = shape * spacing
    # new spacing = real dimensions / new shape (can be float!!)
    # => new shape = real dimensions / new spacing

    resize_factor = spacing * 1.0 / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape) # new_shape has to contain ints!
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor,
                                             mode='nearest')
    return image, new_spacing

def inRange(pbb, lbb):
    """
    Judge whether the predicted bounded box falls in the range of ground truth bounded box

    pbb: predicted bounded box[z, y, x]
    lbb: ground truth bounded box, [z, y, x, diameter]
    """
    p = np.array(pbb)
    l = np.array(lbb[:-1])
    if np.linalg.norm(p - l) <= lbb[-1]:
        return True
    else:
        return False

def froc(pbb, lbb, l, plt, sensitivity = [1.0/8, 1.0/4, 1.0/2, 1.0, 2.0, 4.0, 8.0], isDistribution=True):
    """
    Draw the FROC curve

    pbb: predicted bounded boxes, [[probability, z, y, x, diameter]]
    lbb: ground truth bounded boxes, [[z, y, x, diameter]]
    l: total number of CT-scans, or typically the number of patients
    plt: matplotlib.pyplot module
    sensitivity: List of the points of average false positive per patient/CT-scan
    ***Note: The varibale name may not be named correctly, but the TCAI somehow named it as sensitivity
    isDistribution: whether to plot the distribution of false positives and positives

    return
    y: The value of each point corresponding to the average false positive per patient/CT-scan
    """
    #L is total number of nodules
    #l is total number of CT-scans, or patients
    L = len(lbb)
    pbb = pbb[pbb[:, 0].argsort()][::-1]

    #Record whether each ground truth box is already detected
    l_flag = np.zeros((len(lbb),), np.int32)

    #NL the list of false positive
    #LL the list of true positive
    NL = []
    LL = []

    #Iterate over all predicted bounded boxes
    for p in pbb:
        flag = False
        matchi = 0
        #Iterate over all the labels(ground truth boxes)
        for i, label in enumerate(lbb):

            #If predicted coords are within the range of
            if inRange(p[1:-1], label):
                flag = True
                matchi = i

        #If the one is already detected, then put it into false positive
        if flag == True and l_flag[matchi] != 1:
            LL.append(p)
            l_flag[matchi] = 1
        else:
            NL.append(p)

    LL = np.array(LL)
    NL = np.array(NL)
    print LL.shape

    y = []
    for s in sensitivity:
        print s, l

        #According to the definition, find the index of the false positive one where it has the s*lth largest prob.
        index = int(s*l-1)

        #This is the threshold prob. for the calculating the true positive
        threshold = NL[index][0]

        #According to the definition, sensitivity(y-axis) is the number of true positive greater than the threshold
        #over the number of nodules in total
        y.append(len(LL[LL[:, 0]>=threshold]) / float(L))

    #Plot these points
    plt.plot(sensitivity, y)
    plt.xlabel("Average number of false positive per scan")
    plt.ylabel("Sensitivity")
    plt.legend(loc='best')
    plt.show()

    #Plot the scatter of true positives and false positives
    plt.ylim((-5,5))
    plt.plot(LL[:, 0], np.zeros_like(LL[:, 0]), 'rx', label = "True positive")
    plt.plot(NL[:, 0], np.zeros_like(NL[:, 0]) + 1, 'b*', label = "False positive")
    plt.legend(loc='best')
    plt.show()

    return y


