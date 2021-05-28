import cv2
import yaml
from tqdm import tqdm
import os
from dotmap import DotMap
import numpy as np

# Change the options below for each of the new submissions.
config = yaml.safe_load('''
##########################################################################
# This is the main config file for your method. Please fill the details. #
##########################################################################


##################
# Method Details #
##################
MethodTitle: "Kornia-AffNet-HardNet8-AdaLAM-tutorial"
MethodDescription: "OpenCV DoG keypoints + AffNet normalization, HardNet8 descriptor and AdaLAM postprocessing" #optional
SubmitterName : "Dmytro Mishkin" #optional
TeamName: "kornia" #optional
TeamMembers: "" #optional
PaperLink: "https://librecv.github.io/blog/2021/05/18/submitting-to-IMC2021-step-by-step.html" #optional
CodeLink: "" #optional
FreeformComments: "" #optional

#######################
# SimLocMatch Related # 
#######################

# Any other hyperparameter, to be logged along with the
# submission and / or used in your method _init() function
# This will also be logged at simlocmatch.com
params: 
  my_first_param: "hello"
  learning_rate: 0.001
  etc: "etc.."
''')

config = DotMap(config)
import os
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
import argparse
from adalam import AdalamFilter

def convert_kpts_to_imc(cv2_kpts):
    keypoints = np.array([(x.pt[0], x.pt[1]) for x in cv2_kpts ]).reshape(-1, 2)
    scales = np.array([12.0* x.size for x in cv2_kpts ]).reshape(-1, 1)
    angles = np.array([x.angle for x in cv2_kpts ]).reshape(-1, 1)
    responses = np.array([x.response for x in cv2_kpts]).reshape(-1, 1)
    return keypoints, scales, angles, responses


def extract_features(img_fname, detector, affine, descriptor, device, visualize=False):
    img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
    if visualize:
        plt.imshow(img)
    kpts = detector.detect(img, None)[:8000]
    # We will not train anything, so let's save time and memory by no_grad()
    with torch.no_grad():
        timg = K.image_to_tensor(img, False).float()/255.
        timg = timg.to(device)
        timg_gray = K.rgb_to_grayscale(timg)
        # kornia expects keypoints in the local affine frame format. 
        # Luckily, kornia_moons has a conversion function
        lafs = laf_from_opencv_SIFT_kpts(kpts, device=device)
        lafs_new = affine(lafs, timg_gray)
        if visualize:
            visualize_LAF(timg, lafs_new, 0)
        patches = KF.extract_patches_from_pyramid(timg_gray, lafs_new, 32)
        B, N, CH, H, W = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :) 
        descs = descriptor(patches.view(B * N, CH, H, W)).view(B * N, -1).detach().cpu().numpy()
    return kpts, descs, img
def match_adalam_with_magsac(kps1, kps2, descs1, descs2,
                               h1, w1, h2, w2):
                               
    matcher = AdalamFilter()
    kp1, s1, a1, r1 = convert_kpts_to_imc(kps1)
    kp2, s2, a2, r2 = convert_kpts_to_imc(kps2)
    idxs = matcher.match_and_filter(kp1, kp2,
                            descs1, descs2,
                            im1shape=(h1,w1),
                            im2shape=(h2,w2),
                            o1=a1.reshape(-1),
                            o2=a2.reshape(-1),
                            s1=s1.reshape(-1),
                            s2=s2.reshape(-1)).detach().cpu().numpy()
    if len(idxs) <= 15:
        return None
    src_pts = kp1[idxs[:,0]]
    dst_pts = kp2[idxs[:,1]]

    F, inliers_mask = cv2.findFundamentalMat(
            src_pts,
            dst_pts,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=0.25,
            confidence=0.99999,
            maxIters=100000)
    if np.array(inliers_mask).sum() < 15:
        return None
    inliers_mask = np.array(inliers_mask).astype(bool).reshape(-1)
    return np.concatenate([src_pts[inliers_mask], dst_pts[inliers_mask]], axis=1)


def compute_benchmark_results(images, pairs):
    device = torch.device('cpu')
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print ("GPU mode")
    except:
        print ('CPU mode')
    # SIFT (DoG) Detector
    sift_det =  cv2.SIFT_create(8000, contrastThreshold=-10000, edgeThreshold=-10000)
    # HardNet8 descriptor
    hardnet8 = KF.HardNet8(True).eval().to(device)
    # Affine shape estimator
    affnet = KF.LAFAffNetShapeEstimator(True).eval().to(device)
    # Loop through all the images and run detector/descriptor
    extraction_results = {}
    import os
    loaded = False
    #if os.path.isfile('features.pt'):
    #    extraction_results = torch.load('features.pt')
    #    loaded = True
    count = 0
    if not loaded:
        for image_key in tqdm(images):
            img_fname = os.path.join("simlocmatch-dataset",image_key)
            kpts, descrs, im =extract_features(img_fname, sift_det, affnet, hardnet8, device) 
            extraction_results[image_key] = {"kpts":kpts,"descrs":descrs, "im":im}
        #torch.save(extraction_results, 'features.pt')
    
    matches_results = {}
    # Loop through the benchmark image pairs        
    for pair_key in tqdm(pairs):
        image_key_L, image_key_R = pair_key
        descrs_L = extraction_results[image_key_L]['descrs']
        descrs_R = extraction_results[image_key_R]['descrs']
        kpts_L = extraction_results[image_key_L]['kpts']
        kpts_R = extraction_results[image_key_R]['kpts']
        im_L = extraction_results[image_key_L]['im']
        im_R = extraction_results[image_key_R]['im']
        h1,w1 = im_L.shape[:2]
        h2,w2 = im_R.shape[:2]
        matches_results[tuple(pair_key)] = match_adalam_with_magsac (kpts_L,kpts_R, descrs_L,  descrs_R, h1,w1, h2, w2)
        # Handling cases with no matching points found
        # if len(matches)==0:
           # matches_results[tuple(pair_key)] = None
    return matches_results
