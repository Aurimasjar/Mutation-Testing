### Code Clone Detection

 1. `cd clone`
 2. run `python pipeline.py --lang c` or `python pipeline.py --lang java` to generate preprocessed data for the two datasets.
 3. run `python train.py --lang c` to train on OJClone, `python train.py --lang java` on BigCLoneBench respectively.
 4. run `python test.py --lang c` to test model results on OJClone, `python test.py --lang java` on BigCLoneBench respectively.

### Metrics description

All metrics used in research are described in `clone/metrics.py` file.
 1. In method calculate_old_c_metrics are presented metrics from the first version used for c language metrics model.
 2. In method calculate_c_metrics_2 are presented metrics from the second version used for c language metrics model.
 3. In method calculate_c_metrics are presented metrics from the third and final version used for c language metrics model.
 4. In method calculate_java_metrics are presented metrics for java language metrics model.


### How to use it on your own dataset

Please refer to the `pkl` file in the corresponding directory. This file can be loaded by `pandas`.
