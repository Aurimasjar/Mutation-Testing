### Code Clone Detection

 1. `cd clone`
 2. run `python pipeline.py --lang c` or `python pipeline.py --lang java` to generate preprocessed data for the two datasets.
 2. run `python train.py --lang c` to train on OJClone, `python train.py --lang java` on BigCLoneBench respectively.


### How to use it on your own dataset

Please refer to the `pkl` file in the corresponding directory. This file can be loaded by `pandas`.
