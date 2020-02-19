#!/bin/bash

# 0. Test the global conditions
mprof run --python test_global_conditions.py old 0
mprof plot -o figures/global_function_with_conversion && rm ./*.dat
mprof run --python test_global_conditions.py old 1
mprof plot -o figures/global_function_with_seed && rm ./*.dat
mprof run --python test_global_conditions.py old 2
mprof plot -o figures/global_function_with_seed_and_conversion && rm ./*.dat

# At this point, we find that the general seed is
# the cause of the leak and can focus on the local
# conditions for the bug to happen

# 1. Test the local conditions with tf get_seed function
mprof run --python test_local_conditions.py tf_get_seed 0
mprof plot -o figures/local_tf_get_seed_function && rm ./*.dat
mprof run --python test_local_conditions.py tf_get_seed 1
mprof plot -o figures/local_tf_get_seed_function_determinist && rm ./*.dat
mprof run --python test_local_conditions.py tf_get_seed 2
mprof plot -o figures/local_tf_get_seed_function_only_random_factor && rm ./*.dat
mprof run --python test_local_conditions.py tf_get_seed 3
mprof plot -o figures/local_tf_get_seed_function_only_random_factor_op_seed && rm ./*.dat

# 2. Test the local conditions with
mprof run --python test_local_conditions.py patched_get_seed 0
mprof plot -o figures/local_patched_get_seed_function && rm ./*.dat
mprof run --python test_local_conditions.py patched_get_seed 1
mprof plot -o figures/local_patched_get_seed_function_determinist && rm ./*.dat
mprof run --python test_local_conditions.py patched_get_seed 2
mprof plot -o figures/local_patched_get_seed_function_only_random_factor && rm ./*.dat
mprof run --python test_local_conditions.py patched_get_seed 3
mprof plot -o figures/local_patched_get_seed_function_only_random_factor_op_seed && rm ./*.dat
