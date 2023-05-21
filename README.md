# Project Structure

1. Mutation testing prototype is implemented in the root folder.
2. Code clone detection and equivalent mutant detection models are implemented in the `clone` folder.
Refer to the `clone/README.md` for more info. 
3. Code designed for work with Mujava tool is implemented in the  `mujava` folder.
Refer to the `mujava/README.md` for more info.

### Mutation Testing Prototype

Mutation testing prototype is implemented in `main.py` file. 
Triangle inequality program is prepared as an example program.
* To calculate mutation score for selected method and initial test set call function `apply_mutation_testing`
* To calculate mutation scores for selected method, initial test set and new generated test sets 
using a genetic algorithm call function `apply_mutation_testing_with_ga_test_data_generation`

Both methods have additional parameters:
* Set parameter `set_initial_data` to `True` if the test set should be loaded from a file. 
Test sets are loaded from `test_sets` folder.
Otherwise, if parameter `set_initial_data` is `False` a random test set is generated.
* Set parameter `mark_eq_mutants` to `True` if the equivalent mutant detection models should be called to mark mutants.