"""Run a given method to save results as needed for Simlocmatch CVPR 2021 benchmark.
Upload resulting files to https://simlocmatch.com
"""

import argparse
import logging
import os
from pathlib import Path
import methods.orb as orb
import coloredlogs
from lib.utils import export, image_loader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method", required=True, type=str, help="The path to your method definition file. E.g. orb. This has to be a file located in ./methods"
)

parser.add_argument(
    "--benchmark-version",
    type=str,
    required=True,
    choices=["cvpr2021-v1"],
    help="Which benchmark version to run.",
)


def main(args):

    # check if dataset is downloaded
    assert os.path.isdir('simlocmatch-dataset'), "Dataset not downloaded. Please run <bash prepare_data.sh>"

    # import the method file with config + matching code
    import importlib
    method = importlib.import_module("methods."+args.method)
    
    images, pairs = image_loader.load_image_pairs(os.path.join("image_lists","matching-"+args.benchmark_version+".txt"))
    method_results = method.compute_benchmark_results(images, pairs)
    
    export.to_h5(method_results, "matching", method.config, args)
    logging.info("Done.")
    logging.info("Results file saved and is ready for upload. Please check the root folder.")
    

if __name__ == "__main__":
    coloredlogs.install()
    args = parser.parse_args()
    main(args)
