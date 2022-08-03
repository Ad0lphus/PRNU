"""from PIL import Image
from PIL.ExifTags import TAGS
  
# open the image
image = Image.open("test/data/nat/D09_I_t_0017.jpg")
exifdata = image.getexif()
for tagid in exifdata:
    tagname = TAGS.get(tagid, tagid)
    value = exifdata.get(tagid)
    if tagname == "Make":
        print(value)
    if tagname == "Model":
        print(value)"""
    
"""from locale import currency
from PIL import Image
from PIL.ExifTags import TAGS
import os
def rename_images(database):
    s=''
    CurDir = database.split('/')[-1]
    # rename each file as CurDir_filename
    for filename in os.listdir(database):
        os.rename(database+'/'+filename, database+'/'+CurDir+'_'+filename)
rename_images("datasets/union")"""

"""import sys
sys.path.insert(0, '..')
import os
import unittest

import numpy as np
from PIL import Image

import prnu

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


class TestPrnu(unittest.TestCase):

    def test_extract(self):
        im1 = np.array(Image.open('datasets/R307/fingerprint8c.jpg'))[:400, :500]
        w = prnu.extract_single(im1)
        self.assertSequenceEqual(w.shape, im1.shape[:2])

    def test_extract_multiple(self):
        im1 = np.asarray(Image.open('datasets/R307/fingerprint8c.jpg'))[:400, :500]
        im2 = np.asarray(Image.open('datasets/R307/fingerprint8c.jpg'))[:400, :500]

        imgs = [im1, im2]

        k_st = prnu.extract_multiple_aligned(imgs, processes=1)
        k_mt = prnu.extract_multiple_aligned(imgs, processes=2)

        self.assertTrue(np.allclose(k_st, k_mt, atol=1e-6))

    def test_crosscorr2d(self):
        im = np.asarray(Image.open('data/prnu1.jpg'))[:1000, :800]

        w_all = prnu.extract_single(im)

        y_os, x_os = 300, 150
        w_cut = w_all[y_os:, x_os:]

        cc = prnu.crosscorr_2d(w_cut, w_all)

        max_idx = np.argmax(cc.flatten())
        max_y, max_x = np.unravel_index(max_idx, cc.shape)

        peak_y = cc.shape[0] - 1 - max_y
        peak_x = cc.shape[1] - 1 - max_x

        peak_height = cc[max_y, max_x]

        self.assertSequenceEqual((peak_y, peak_x), (y_os, x_os))
        self.assertTrue(np.allclose(peak_height, 666995.0))

    def test_pce(self):
        im = np.asarray(Image.open('data/prnu1.jpg'))[:500, :400]

        w_all = prnu.extract_single(im)

        y_os, x_os = 5, 8
        w_cut = w_all[y_os:, x_os:]

        cc1 = prnu.crosscorr_2d(w_cut, w_all)
        cc2 = prnu.crosscorr_2d(w_all, w_cut)

        pce1 = prnu.pce(cc1)
        pce2 = prnu.pce(cc2)

        self.assertSequenceEqual(pce1['peak'], (im.shape[0] - y_os - 1, im.shape[1] - x_os - 1))
        self.assertTrue(np.allclose(pce1['pce'], 134611.58644973233))

        self.assertSequenceEqual(pce2['peak'], (y_os - 1, x_os - 1))
        self.assertTrue(np.allclose(pce2['pce'], 134618.03404934643))

    def test_gt(self):
        cams = ['a', 'b', 'c', 'd']
        nat = ['a', 'a', 'b', 'b', 'c', 'c', 'c']

        gt = prnu.gt(cams, nat)

        self.assertTrue(np.allclose(gt, [[1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, ], [0, 0, 0, 0, 1, 1, 1],
                                         [0, 0, 0, 0, 0, 0, 0, ]]))

    def test_stats(self):
        gt = np.array([[1, 0, 0, ], [0, 1, 0], [0, 0, 1]], np.bool)
        cc = np.array([[0.5, 0.2, 0.1], [0.1, 0.7, 0.1], [0.4, 0.3, 0.9]])

        stats = prnu.stats(cc, gt)

        self.assertTrue(np.allclose(stats['auc'], 1))
        self.assertTrue(np.allclose(stats['eer'], 0))
        self.assertTrue(np.allclose(stats['tpr'][-1], 1))
        self.assertTrue(np.allclose(stats['fpr'][-1], 1))
        self.assertTrue(np.allclose(stats['tpr'][0], 0))
        self.assertTrue(np.allclose(stats['fpr'][0], 0))

    def test_detection(self):
        nat = np.asarray(Image.open('datasets/authenticated/PRNUtest_fingerprint8c.jpg'))
        ff1 = np.asarray(Image.open('datasets/authenticated/PRNUtest_fingerprint8.bmp'))
        ff2 = np.asarray(Image.open('datasets/authenticated/PRNUtest_fingerprint8c.bmp'))

        nat = prnu.cut_ctr(nat, (500, 500, 3))
        ff1 = prnu.cut_ctr(ff1, (500, 500, 3))
        ff2 = prnu.cut_ctr(ff2, (500, 500, 3))

        w = prnu.extract_single(nat)
        k1 = prnu.extract_single(ff1)
        k2 = prnu.extract_single(ff2)

        pce1 = [{}] * 4
        pce2 = [{}] * 4

        for rot_idx in range(4):
            cc1 = prnu.crosscorr_2d(k1, np.rot90(w, rot_idx))
            pce1[rot_idx] = prnu.pce(cc1)

            cc2 = prnu.crosscorr_2d(k2, np.rot90(w, rot_idx))
            pce2[rot_idx] = prnu.pce(cc2)

        best_pce1 = np.max([p['pce'] for p in pce1])
        best_pce2 = np.max([p['pce'] for p in pce2])

        self.assertGreater(best_pce1, best_pce2)


if __name__ == '__main__':
    unittest.main()"""

# -*- coding: UTF-8 -*-
"""
@author: Luca Bondi (luca.bondi@polimi.it)
@author: Paolo Bestagini (paolo.bestagini@polimi.it)
@author: Nicolò Bonettini (nicolo.bonettini@polimi.it)
Politecnico di Milano 2018
"""

import os
from glob import glob
from multiprocessing import cpu_count, Pool

import numpy as np
from PIL import Image
import time

import prnu
import sys
import matplotlib.pyplot as plt
import argparse


'''
parser = argparse.ArgumentParser(description='This program extracts camera fingerprint using VDNet and VDID and compares them with the original implementation')
parser.add_argument("-denoiser", help="[original (default) | vdnet | vdid]", default='original')
parser.add_argument("-rm_zero_mean", help='Removes zero mean normalization', action='store_true',
                    default=False)
parser.add_argument("-rm_wiener", help='Removes Wiener filter', action='store_true',
                    default=False)
args = parser.parse_args()


def PRNU():
    """
    Main example script. Load a subset of flatfield and natural images from Dresden.
    For each device compute the fingerprint from all the flatfield images.
    For each natural image compute the noise residual.
    Check the detection performance obtained with cross-correlation and PCE
    :return:
    """
    start = time.time()

    denoiser = args.denoiser
    remove_zero_m = args.rm_zero_mean
    remove_wiener = args.rm_wiener
    prnu.define_param(denoiser, remove_zero_m, remove_wiener)

    print('Denoiser: ' + denoiser)
    print('Remove zero mean: ' + str(remove_zero_m))
    print('Remove wiener: ' + str(remove_wiener) + '\n')

    ff_dirlist = np.array(sorted(glob('Real/*.BMP'))) # load flatfield images 
    ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist]) # extract device name from filename

    nat_dirlist = np.array(sorted(glob('Real/*.BMP'))) # load natural images
    nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist]) # extract device name from filename
    print("ff_device: ", ff_device)
    print("nat_device: ", nat_device)
    print("\n")
    print("ff_dirlist: ", ff_dirlist)
    print("nat_dirlist: ", nat_dirlist)

    print('Computing fingerprints')
    fingerprint_device = sorted(np.unique(ff_device)) # extract unique device names

    k = []
    for device in fingerprint_device: # for each device
        imgs = [] # list of flatfield images
        for img_path in ff_dirlist[ff_device == device]: # for each flatfield image
            im = Image.open(img_path) # load image
            im_arr = np.asarray(im) # convert to array
            if im_arr.dtype != np.uint8: # if not uint8 convert to uint8
                print('Error while reading image: {}'.format(img_path)) # print error message
                continue
            if im_arr.ndim != 3: # if not 3D convert to 3D
                print('Image is not RGB: {}'.format(img_path))# print error message
                continue
            im_cut = prnu.cut_ctr(im_arr, (103, 103, 3)) # cut image to 103x103 pixels
            imgs += [im_cut] # add image to list
        k += [prnu.extract_multiple_aligned(imgs, processes=1)] # extract fingerprint for device


    k = np.stack(k, 0) # stack fingerprints


    print('Computing residuals') 

    imgs = []
    for img_path in nat_dirlist:
        imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (103, 103, 3))] # load natural images 


    w = []
    for img in imgs:
        w.append(prnu.extract_single(img))# extract fingerprint for device


    w = np.stack(w, 0) # stack fingerprints 

    # Computing Ground Truth
    gt = prnu.gt(fingerprint_device, nat_device) # compute ground truth for each device

    print('Computing cross correlation')
    cc_aligned_rot = prnu.aligned_cc(k, w)['cc']# compute cross correlation for each device and each rotation 

    print('Computing statistics cross correlation')
    stats_cc = prnu.stats(cc_aligned_rot, gt) # compute statistics for each device and each rotation 

    print('Computing PCE')
    pce_rot = np.zeros((len(fingerprint_device), len(nat_device))) # initialize PCE matrix

    for fingerprint_idx, fingerprint_k in enumerate(k): # for each device and each rotation 
        tn, tp, fp, fn = 0, 0, 0, 0
        pce_values = []
        natural_indices = []
        for natural_idx, natural_w in enumerate(w): # for each natural image and each rotation

            cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w) # compute cross correlation for each device and each rotation 
            prnu_pce = prnu.pce(cc2d)['pce'] # compute PCE for each device and each rotation
            pce_rot[fingerprint_idx, natural_idx] = prnu_pce # add PCE to matrix
            pce_values.append(prnu_pce) # add PCE to list
            natural_indices.append(natural_idx) # add natural image index to list
            if fingerprint_device[fingerprint_idx] == nat_device[natural_idx]: # if device name is the same
                if prnu_pce > 60.: # if PCE is greater than 60
                    tp += 1. # add 1 to true positive
                else: # if PCE is less than 60
                    fn += 1. # add 1 to false negative
            else: # if device name is different
                if prnu_pce > 60.: # if PCE is greater than 60
                    fp += 1. # add 1 to false positive
                else: # if PCE is less than 60
                    tn += 1. # add 1 to true negative
        tpr = tp / (tp + fn) # compute true positive rate
        fpr = fp / (fp + tn) # compute false positive rate

        plt.title('PRNU for ' + str(fingerprint_device[fingerprint_idx]) + ' - ' + denoiser) # set title
        plt.xlabel('query images') # set x label
        plt.ylabel('PRNU') # set y label

        plt.bar(natural_indices, pce_values) # plot bar chart
        plt.text(0.85, 0.85, 'TPR: ' + str(round(tpr, 2)) + '\nFPR: '+ str(round(fpr, 2)), 
         fontsize=10, color='k',
         ha='left', va='bottom',
         transform=plt.gca().transAxes)# add text to plot
        plt.axhline(y=60, color='r', linestyle='-') # add red line at 60
        plt.xticks(natural_indices) # set x ticks
        plt.savefig('plots_test/'+ denoiser + '/' +str(fingerprint_device[fingerprint_idx])+'.png') # save plot

        plt.clf() # clear plot

    print('Computing statistics on PCE') # compute statistics on PCE
    stats_pce = prnu.stats(pce_rot, gt) # compute statistics on PCE

    print('AUC on CC {:.2f}'.format(stats_cc['auc'])) # print AUC on CC
    print('AUC on PCE {:.2f}'.format(stats_pce['auc'])) # print AUC on PCE

    end = time.time() # end time
    elapsed = int(end - start) # elapsed time
    print('Elapsed time: '+ str(elapsed) + ' seconds') # print elapsed time
    """for fingerprint_idx, fingerprint_k in enumerate(k):
        print(fingerprint_device[fingerprint_idx])
        for natural_idx, natural_w in enumerate(w):
            #print image name and PCE
            print(nat_dirlist[natural_idx])
            # print value of PCE
            print(pce_rot[fingerprint_idx, natural_idx])
        print('\n')
    """
    list,l=[],[]
    ## print all image names whoes PCE is greater than 60 in a line for each device
    for fingerprint_idx, fingerprint_k in enumerate(k):
        #print(fingerprint_device[fingerprint_idx])
        for natural_idx, natural_w in enumerate(w):
            if pce_rot[fingerprint_idx, natural_idx] > 60.:
                #print(nat_dirlist[natural_idx],end='')
                #print(pce_rot[fingerprint_idx, natural_idx])
                l.append(nat_dirlist[natural_idx])
        list.append([fingerprint_device[fingerprint_idx],l])
        l=[]
    #print(list)
    for i in list:
        print(i)
    print('\n')




if __name__ == '__main__':
    PRNU()'''

dir = './datasets/R307'
curDir="R307"

# rename all files in the directory as curDir_filename
for filename in os.listdir(dir):
    os.rename(os.path.join(dir, filename), os.path.join(dir, curDir+"_"+filename))
