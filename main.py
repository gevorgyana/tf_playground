#!/home/i516739/anaconda3/envs/tf/bin/python

import tensorflow as tf
import numpy as np

batch_size = 32
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "aclImdb/train",
    batch_size = batch_size,
    validation_split = 0.2,
    subset = "training",
    seed = 1337,
)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "aclImdb/train",
    batch_size = batch_size,
    validation_split = 0.2,
    subset = "training",
    seed = 1337,
)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "aclImdb/test", batch_size = batch_size,
)

print(
    "number of batches in raw_train_ds: %d"
    % tf.data.experimental.cardinality(raw_train_ds)
)

print(
    "Number of batches in raw_test_ds: %d"
    % tf.data.experimental.cardinality()
)
