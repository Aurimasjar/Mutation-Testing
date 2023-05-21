# Code Clone Detection

 1. `cd clone`
 2. run `python pipeline.py --lang c` or `python pipeline.py --lang java` to generate preprocessed data for the two datasets.
 3. run `python train.py --lang c` to train on OJClone, `python train.py --lang java` on BigCLoneBench respectively.
 4. run `python test.py --lang c` to test models on OJClone, `python test.py --lang java` on BigCLoneBench respectively.

### Metrics description

All metrics used in research are described in `clone/metrics.py` file.
 1. In method `calculate_old_c_metrics` are presented metrics from the first version used for c language metrics model.
 2. In method `calculate_c_metrics_2` are presented metrics from the second version used for c language metrics model.
 3. In method `calculate_c_metrics` are presented metrics from the third and final version used for c language metrics model.
 4. In method `calculate_java_metrics` are presented metrics for java language metrics model.

### Output

1. All metadata is stored in `clone/data_srw3` folder.
2. All generated images and their data is stored in `clone/images_srw3` folder.
3. All trained models are stored in `clone/output_srw3` folder.


# Equivalent Mutant Detection

 1. `cd clone`
 2. run `python pipeline.py --lang javamut` to generate preprocessed data for the mutant dataset.
 3. run `python train.py --lang javamut` to train on OJClone.
 4. run `python test.py --lang javamut` to test models on OJClone.

### Metrics description

Metrics used for mutants are described in `clone/metrics.py` file in method `calculate_javamut_metrics`.

### Output

1. All metadata is stored in `clone/data` folder.
2. All generated images and their data is stored in `clone/images` folder.
3. All trained models are stored in `clone/output` folder.

### Usage

Use `evaluate` function from `clone/eval.py` file to get the evaluations of code pair equivalence.
As an input pass a batch of code pairs and the method name that is compared.
Example provided in `mark_equivalent_mutants` function from `main.py` file.
