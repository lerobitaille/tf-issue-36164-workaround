"""This script aims to show the memory leak that exists when tf has a
general seed and a conversion to numpy is done.
"""

import os
import sys
import random

import numpy as np
import tensorflow as tf
import tqdm


DATASET_SIZE = 1000
BATCH_SIZE = 32


#-----------------------------------------------------------------------


def transform_with_numpy():
    def transform(x):
        x = x.numpy() # Get a numpy version of x
        # Some NumPy operations here
        x = tf.convert_to_tensor(x) # Repass to tf version of x
        # A random TF function
        x = tf.image.random_contrast(x, 0.0, 1.0)
        return x
    return transform


def transform_with_tf():
    def transform(x):
        # Some TF operations here
        x = tf.convert_to_tensor(x) # Recast, just to isolate cause
        # A random TF function
        x = tf.image.random_contrast(x, 0.0, 1.0)
        return x
    return transform


#-----------------------------------------------------------------------


def get_example(T):
    example = tf.convert_to_tensor(
        np.random.rand(112, 112).astype(np.float32))[tf.newaxis]
    return T(example)


def get_examples(T):
    for i in tqdm.tqdm(range(DATASET_SIZE)):
        examples = [get_example(T) for i in range(BATCH_SIZE)]
    return examples


#-----------------------------------------------------------------------


def set_seed(seed=1000):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


#-----------------------------------------------------------------------


@profile
def function_with_conversion():
    print(f'function_with_conversion')
    T = transform_with_numpy()
    get_examples(T)


@profile
def function_with_seed():
    print(f'function_with_seed')
    set_seed()
    T = transform_with_tf()
    get_examples(T)


@profile
def function_with_seed_and_conversion():
    print(f'function_with_seed_and_conversion')
    set_seed()
    T = transform_with_numpy()
    get_examples(T)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('You should also choose a tested func number in [0, 1, 2]')
        exit()

    try:
        func_num = int(sys.argv[2])
    except ValueError:
        print('The tested func number should be an int')
        exit()

    if not (func_num < 3):
        print('The tested func number should be in [0, 1, 2]')
        exit()

    # Disable INFO and WARNING tf messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Sets the memory growth, experimental...
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if func_num == 0:
        function_with_conversion()
    elif func_num == 1:
        function_with_seed()
    else:
        function_with_seed_and_conversion()
