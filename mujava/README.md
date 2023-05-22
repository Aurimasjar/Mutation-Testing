# Mujava

Mujava tool was used both for mutant dataset production 
and mutant generation for new programs that were tested using mutation testing.

### Setup

1. To use mujava tool first 1.8 Java version is needed to be installed.
2. Three files are needed to be downloaded from https://cs.gmu.edu/~offutt/mujava/: 
`mujava.jar`, `openjava.jar` and `mujava.config`. 
The path in `mujava.config` must be set as a working mujava tool directory.

### Dataset creation

1. Set path in `mujava.config` file as `absolute_project_path\Mutation-Testing\mujava\dataset_session`.
2. To generate folder structure for mujava from `absolute_path\mujava` directory call this command:
`java mujava.makeMuJavaStructure`
3. Put original program to `mujava\dataset_session\src` and it's compiled code to `mujava\dataset_session\classes`.
4. To open a mujava window for mutant generation from `absolute_path\mujava` directory call this command:
`java mujava.gui.GenMutantsMain`. 
Use gui interface to mark class to mutate and selected mutation operators (use only method level mutation operators).
Decompiled original program is stored in `mujava\dataset_session\result\original` folder.
Decompiled mutants are stored in `mujava\dataset_session\result\traditional_mutants` folder.
5. Put program tests to `mujava\dataset_session\src` and it's compiled code to
`mujava\dataset_session\test_set`.
6. To open a mujava window for mutant execution for dataset creation from `absolute_path\mujava` directory call this command:
`java mujava.gui.RunTestMain > "absolute_project_path\Mutation-Testing\mujava\dataset_session\output.txt"`. 
Use gui interface to mark class, method and test case to test 
and check to execute only traditional mutants (that use only method level mutation operators).
Test results are stored in `mujava\dataset_session\output.txt` file.
7. `dataset_prep.py` file is used to scan the mutants and their test results 
and it forms the dataset that is stored in the `clone\data\javamut` folder: 
in `mut_funcs_all.csv` file all programs and their mutants are stored,
in `mut_pair_ids.csv` file all pairs of programs and their mutants are stored.

### Mutant generation for mutation testing

1. Set path in `mujava.config` file as `absolute_project_path\Mutation-Testing\mujava\program_session`.
2. To generate folder structure for mujava from `absolute_path\mujava` directory call this command:
`java mujava.makeMuJavaStructure`
3. Put original program to `mujava\program_session\src` and it's compiled code to `mujava\program_session\classes`.
4. To open a mujava window for mutant generation from `absolute_path\mujava` directory call this command:
`java mujava.gui.GenMutantsMain`. 
Use gui interface to mark class to mutate and selected mutation operators (use only method level mutation operators).
Decompiled original program is stored in `mujava\program_session\result\original` folder.
Decompiled mutants are stored in `mujava\program_session\result\traditional_mutants` folder.
5. `fix_package_structure` function from `mutants_prep.py` file is used to prepare codes for execution.
For each code package keyword is added to make them unique and folder method names are renamed to be readable.
