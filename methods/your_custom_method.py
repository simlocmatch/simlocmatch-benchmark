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
MethodTitle: "Your method name"
MethodDescription: "Description goes here" #optional
SubmitterName : "" #optional
TeamName: "" #optional
TeamMembers: "" #optional
PaperLink: "" #optional
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

def compute_benchmark_results(images, pairs):

    matches_results = {}
    # Loop through the benchmark image pairs        
    for pair_key in tqdm(pairs):

        your_matches = [] # replace this with matches from your method
        matches_results[tuple(pair_key)] = your_matches

        # Handling cases with no matching points found
        # if len(matches)==0:
           # matches_results[tuple(pair_key)] = None
           
    return matches_results
            

