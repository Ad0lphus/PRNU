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

    ff_dirlist = np.array(sorted(glob('datasets/files/*'))) # load flatfield images 
    ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist]) # extract device name from filename

    nat_dirlist = np.array(sorted(glob('datasets/files/*'))) # load natural images
    nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist]) # extract device name from filename
    print("ff_device: ", ff_device)
    print("nat_device: ", nat_device)
    print("\n")
    print("ff_dirlist: ", ff_dirlist)
    print("nat_dirlist: ", nat_dirlist)
    '''
    difference between ff_device and nat_device is that ff_device contains the device name and nat_device contains the image name
    '''
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
            size = im_arr.shape # get image size
            print('Image size: {}'.format(size)) # print image size
            im_cut = prnu.cut_ctr(im_arr,size) # cut image
            #im_cut = prnu.cut_ctr(im_arr, (512, 512, 3)) # cut image to 512x512 pixels
            imgs += [im_cut] # add image to list
        
        # number of channels
        n_channels = imgs[0].shape[2]
        print('Number of channels: {}'.format(n_channels))

        # make sure all images have the same number of channels (3)
        print('file name: ', ff_dirlist[ff_device == device])
        k += [prnu.extract_multiple_aligned(imgs, processes=1)] # extract fingerprint for device
        '''
        Input image must have 1 or 3 channels is because the fingerprint is computed using the mean of the channels
        solution: convert to 3 channels
        '''



    k = np.stack(k, 0) # stack fingerprints


    print('Computing residuals') 

    imgs = []
    size = im_arr.shape # get image size
    print('Image size: {}'.format(size)) # print image size
    for img_path in nat_dirlist:
        imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), size)] # load natural images 



    w = []
    for img in imgs:
        w.append(prnu.extract_single(img))# extract fingerprint for device


    w = np.stack(w, 0) # stack fingerprints 

    # Computing Ground Truth
    gt = prnu.gt(fingerprint_device, nat_device) # compute ground truth for each device

    print('Computing cross correlation')
    cc_aligned_rot = prnu.aligned_cc(k, w)['cc']# compute cross correlation for each device and each rotation 

    print('Computing statistics cross correlation')
    #stats_cc = prnu.stats(cc_aligned_rot, gt) # compute statistics for each device and each rotation 

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
        print("tp : {}".format(tp)), print("fp : {}".format(fp)), print("tn : {}".format(tn)), print("fn : {}".format(fn))
        #tpr = tp / (tp + fn) # compute true positive rate
        #fpr = fp / (fp + tn) # compute false positive rate

        plt.title('PRNU for ' + str(fingerprint_device[fingerprint_idx]) + ' - ' + denoiser) # set title
        plt.xlabel('query images') # set x label
        plt.ylabel('PRNU') # set y label

        plt.bar(natural_indices, pce_values) # plot bar chart
        #plt.text(0.85, 0.85, 'TPR: ' + str(round(tpr, 2)) , 
         #fontsize=10, color='k',
         #ha='left', va='bottom',
         #transform=plt.gca().transAxes)# add text to plot
        plt.axhline(y=60, color='r', linestyle='-') # add red line at 60
        plt.xticks(natural_indices) # set x ticks
        plt.savefig('plots_test/'+ denoiser + '/' +str(fingerprint_device[fingerprint_idx])+'.png') # save plot

        plt.clf() # clear plot

    print('Computing statistics on PCE') # compute statistics on PCE
    stats_pce = prnu.stats(pce_rot, gt) # compute statistics on PCE

   # print('AUC on CC {:.2f}'.format(stats_cc['auc'])) # print AUC on CC
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
        #print(i,end='\n\n')
        print(i[0])
        for j in i[1]:
            print(j,end='\n')




if __name__ == '__main__':
    PRNU()

'''
PRNU is a tool for the analysis of natural images. It is based on the principle of fingerprinting, which is a method for the analysis of images.
The goal of PRNU is to identify the devices used in natural images. The analysis is based on the principle of cross-correlation.
The analysis is based on the principle of PCE. 

Working :
    1. Extract fingerprints from natural images
    2. Extract fingerprints from flatfield images
    3. Compute cross correlation between fingerprints
    4. Compute PCE between fingerprints
    5. Compute statistics on cross correlation
    6. Compute statistics on PCE
    7. Plot cross correlation
    8. Plot PCE
    9. Plot statistics on cross correlation
    10. Plot statistics on PCE
    11. Plot AUC on cross correlation
    12. Plot AUC on PCE

Exceptions : 
'''
