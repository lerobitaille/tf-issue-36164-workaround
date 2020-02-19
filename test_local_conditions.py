"""This script aims to show the memory leak that exists when tf has a
general seed and a conversion to numpy is done.
"""

import os
import sys
import random
import gorilla

import numpy as np
import tensorflow as tf
import tqdm

from tensorflow.python.eager import context
from tensorflow.python.framework import random_seed


DEFAULT_OP_SEED = 1923746
DATASET_SIZE = 1000
BATCH_SIZE = 32


#-----------------------------------------------------------------------


def transform_with_numpy(
        split_random_function=False,
        use_deterministic_factor=False,
        remove_contrast_function=False,
        use_op_seed=False):
    def transform(x):
        x = x.numpy() # Get a numpy version of x
        # Some NumPy operations here
        x = tf.convert_to_tensor(x) # Repass to tf version of x
        # A random TF function
        seed = DEFAULT_OP_SEED if use_op_seed else None
        if split_random_function:
            if use_deterministic_factor:
                factor = 0.5
            else:
                factor = tf.random.uniform([], seed=seed)
            if not remove_contrast_function:
                x = tf.image.adjust_contrast(x, factor)
        else:
            x = tf.image.random_contrast(x, 0.0, 1.0, seed=seed)
        return x
    return transform


#-----------------------------------------------------------------------


def get_example(T):
    example = tf.convert_to_tensor(
        np.random.rand(224, 224).astype(np.float32)
    )[tf.newaxis]
    return T(example)


def get_examples(T):
    for i in tqdm.tqdm(range(DATASET_SIZE)):
        examples = [get_example(T) for i in range(BATCH_SIZE)]
    return examples


#-----------------------------------------------------------------------


def better_get_seed(global_seed, op_seed):
    if op_seed is not None:
        return global_seed, op_seed
    else:
        return global_seed, DEFAULT_OP_SEED


def set_seed(seed=100):
    np.random.seed(seed)
    random.seed(seed)
    # Monkey Patch get_seed.
    func = lambda op_seed: better_get_seed(seed, op_seed)
    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    patch = gorilla.Patch(
        random_seed, 'get_seed', func, settings=settings)
    gorilla.apply(patch)


def set_seed_tf_get_seed(seed=100):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


#-----------------------------------------------------------------------


@profile
def function(seed_func):
    seed_func()
    T = transform_with_numpy(
        split_random_function=True)
    print(f'function')
    get_examples(T)


@profile
def function_deterministic(seed_func):
    seed_func()
    T = transform_with_numpy(
        split_random_function=True,
        use_deterministic_factor=True)
    print(f'function_deterministic')
    get_examples(T)


@profile
def function_only_random_factor(seed_func):
    seed_func()
    T = transform_with_numpy(
        split_random_function=True,
        remove_contrast_function=True)
    print(f'function_only_random_factor')
    get_examples(T)


@profile
def function_only_random_factor_op_seed(seed_func):
    seed_func()
    T = transform_with_numpy(
        split_random_function=True,
        remove_contrast_function=True,
        use_op_seed=True)
    print(f'function_only_random_factor_op_seed')
    get_examples(T)


#-----------------------------------------------------------------------


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('You should choose a seed func to use in '
              '[patched_get_seed, tf_get_seed]')
        print('You should also choose a tested func number in '
              '[0, 1, 2, 3]')
        exit()

    if sys.argv[1] not in ['patched_get_seed', 'tf_get_seed']:
        print('You should choose a seed func to use in '
              '[patched_get_seed, tf_get_seed]')
        exit()

    try:
        func_num = int(sys.argv[2])
    except ValueError:
        print('The tested func number should be an int')
        exit()

    if not (func_num < 4):
        print('The tested func number should be in [0, 1, 2, 3]')
        exit()

    if sys.argv[1] == 'patched_get_seed':
        seed_func = set_seed
    else:
        seed_func = set_seed_tf_get_seed

    # Disable INFO and WARNING tf messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Sets the memory growth, experimental...
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(
            physical_devices[0], True)

    # Executes functions
    if func_num == 0:
        function(seed_func)
    elif func_num == 1:
        function_deterministic(seed_func)
    elif func_num == 2:
        function_only_random_factor(seed_func)
    else:
        function_only_random_factor_op_seed(seed_func)
