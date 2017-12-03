import numpy as np
import tensorflow as tf


def sparse_concat(input_list, features_M):
    data_shape = (len(input_list), features_M)
    indices = []
    values = []

    i = 0
    for item in input_list:
        values.extend(item['values'])
        item_indices = i * np.ones([len(item['indices']), 2], dtype=np.int32)
        item_indices[:, 1] = item['indices']
        indices.extend(item_indices.tolist())
        i = i + 1

    return tf.SparseTensorValue(indices, values, data_shape)


def sparsify(input_data):
    sparse_list = []
    for i in range(input_data.shape[0]):
        indices = []
        values = []
        for j in range(input_data.shape[1]):
            if input_data[i, j] >= 0.0001 or input_data[i, j] <= -0.0001:
                indices.append(j)
                values.append(input_data[i, j])
        sparse_list.append({'indices': indices, 'values': values})

    return sparse_list
