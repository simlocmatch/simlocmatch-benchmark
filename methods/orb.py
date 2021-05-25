import cv2
import yaml
from tqdm import tqdm
import os
from dotmap import DotMap
import numpy as np

# Sample case for ORB. For a full list of the
# available options please see `your_custom_method.py`

config = yaml.safe_load('''
MethodTitle: "ORB"
MethodDescription: "OpenCV-ORB"

params:
  nfeatures: 1000
''')

config = DotMap(config)

def compute_benchmark_results(images, pairs):

    # Loop through all the images and run detector/descriptor
    extraction_results = {}
    for image_key in tqdm(images):
        im = cv2.imread(os.path.join("simlocmatch-dataset",image_key))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=config.params.nfeatures)
        kpts, descrs = orb.detectAndCompute(im, None)
        extraction_results[image_key] = {"kpts":kpts,"descrs":descrs, "im":im}

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
        
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING,crossCheck=True)
        matches = bf.match(descrs_L, descrs_R)
        # matches_img =cv2.drawMatches(im_L, kpts_L, im_R, kpts_R, matches, None, flags=2)
        # cv2.imshow("matches",matches_img)
        # cv2.waitKey(10000)

        pair_matches = []
        for m in matches:
            # Get the matching keypoints for each of the images
            idx_L = m.queryIdx
            idx_R = m.trainIdx

            # x,y -> column,row
            (x_L, y_L) = kpts_L[idx_L].pt
            (x_R, y_R) = kpts_R[idx_R].pt
            pair_matches.append((x_L, y_L,x_R, y_R))
        matches_results[tuple(pair_key)] = np.array(pair_matches)

        # Handling cases with no matching points found
        # if len(matches)==0:
           # matches_results[tuple(pair_key)] = None
           
    return matches_results
            

