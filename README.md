## SimLocMatch submissions for Image Matching: Local Features & Beyond - CVPR 2021

*SimLocMatch* is a dataset and benchmark that is based on synthetic
sequences, rendered under different challenging conditions. 

A significant advantage of *SimLocMatch* is that true and perfectly
accurate ground truth is available. This can lead to a rigorous
evaluation of matching methods, something that was a significant
drawback of benchmarks and datasets based on real data and SfM
pipelines for building pseudo ground truth. 

![banner](./imgs/banner.jpg)


### Downloading and setting up the data

To prepare for running the benchmark, first step is to download and prepare the data.
To do this, run 

```sh
bash prepare_data.sh
```

### Running your method on the challenge
A sample baseline is provided in `methods/orb.py` for the case of a trivial openCV based ORB matcher. 
It should act as a guideline of how to adapt to your method.

To run the orb baseline:
```sh
python3 compute_matching.py --method orb --benchmark-version cvpr2021-v1
```

To run the kornia baseline:
```sh
python3 compute_matching.py --method korniaadalam --benchmark-version cvpr2021-v1
```

This would require GPU and 32 Gb RAM, because all the descriptors are kept in memory for now. 

To run your method named `X`:
Create a `methods/X.py` file, replicating the methods from the sample ORB baseline.
Fill this file with the code required to run your method on the image pairs. 

Lastly, run the benchmark as below:
```sh
python3 compute_matching.py --method X --benchmark-version cvpr2021-v1
```
After all the pairs are traversed, you should have a `matching-X.h5` result file output.


### Results format
Please use the `compute_matching.py` script which saves the results in
a way that is compatible with the evaluation server.

Output is `.h5` file will contain results for each of the evaluation pairs

Note: SimLocMatch includes negative pairs in the evaluation,
i.e. pairs were no matches should be found. In the case that your
method is not able to match points between two images, then `None`
should be stored for this pair key.

I.e.

`matches_results[tuple(pair_key)] = None`

Please check the examples in the `methods` folder for more. 

### Submission of the results

Visit [simlocmatch.com](https://simlocmatch.com), and submit the resulting `.h5` file. 
Please note that upload might take some time due to the size of the results file (~5mins).


### CVPR 2021 Winning Methods
For the CVPR 2021 version of the challenge, we will use a combination of results w.r.t the following metrics:

- number of inliers (higher is better)
- matching success rate score (higher is better)
- number of matches found in negative pairs (cases with >0 will be penalized)

A method has to score high in all 3 areas in order to achieve a high rank overall.  
Details about the exact computation of the final ranking will be revealed closer to the challenge deadline. 

### Problems/Bugs/Evaluation Mishaps

Please note that [simlocmatch.com](https://simlocmatch.com) is in beta. 
For any issues, contact [info@simlocmatch.com](mailto:info@simlocmatch.com)
