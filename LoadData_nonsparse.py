'''
Utilities for Loading data.
The input data file follows the same input for LibFM: http://www.libfm.org/libfm-1.42.manual.pdf

This version is modified from https://github.com/neural_factorization_machine, which is provided by:
Xiangnan He (xiangnanhe@gmail.com)
Lizi Liao (liaolizi.llz@gmail.com)

'''
import numpy as np
import random
import os
import pickle
from sparsify import sparse_concat


class LoadData(object):
    '''given the path of data, return the data format for DeepFM
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors with 0 or 1 entries
    Test_data: same as Train_data
    Validation_data: same as Train_data
    '''

    # Three files are needed in the path
    def __init__(self, path, dataset, loss_type, from_file=False, is_sparse=False):
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset + ".train.libfm"
        self.testfile = self.path + dataset + ".test.libfm"
        self.validationfile = self.path + dataset + ".validation.libfm"
        self.is_sparse = is_sparse
        if from_file:
            self.Train_data = pickle.load(open(os.path.join(self.path, dataset + '.train.dat')))
            self.Test_data = pickle.load(open(os.path.join(self.path, dataset + '.test.dat')))
            self.Validation_data = pickle.load(open(os.path.join(self.path, dataset + '.validation.dat')))
            self.features_M = self.Train_data['X'].shape[1]
        else:
            self.features_M = self.map_features()
            self.Train_data, self.Validation_data, self.Test_data = self.construct_data(loss_type)

    def map_features(self):  # map the feature entries in all files, kept in self.features dictionary
        features_train, self.train_num = self.read_features(self.trainfile)
        features_validation, self.validation_num = self.read_features(self.validationfile)
        features_test, self.test_num = self.read_features(self.testfile)
        return max([features_train, features_validation, features_test])

    def read_features(self, file):  # read a feature file
        features_M = 0
        num = 0
        for line in open(file, 'r'):
            items = line.strip().split(' ')
            num = num + 1
            for item in items[1:]:
                feature_id = int(item.strip().split(':')[0])
                if features_M < feature_id + 1:
                    features_M = feature_id + 1

        return features_M, num

    def construct_data(self, loss_type):
        X_, Y_, Y_for_logloss, X_sparse_list, X_sparse = self.read_data(self.trainfile, self.train_num)
        if loss_type == 'log_loss':
            Train_data = self.construct_dataset(X_, Y_for_logloss, X_sparse_list, X_sparse)
        else:
            Train_data = self.construct_dataset(X_, Y_, X_sparse_list, X_sparse)
        print("# of training:", len(Y_))

        X_, Y_, Y_for_logloss, X_sparse_list, X_sparse = self.read_data(self.validationfile, self.validation_num)
        if loss_type == 'log_loss':
            Validation_data = self.construct_dataset(X_, Y_for_logloss, X_sparse_list, X_sparse)
        else:
            Validation_data = self.construct_dataset(X_, Y_, X_sparse_list, X_sparse)
        print("# of validation:", len(Y_))

        X_, Y_, Y_for_logloss, X_sparse_list, X_sparse = self.read_data(self.testfile, self.test_num)
        if loss_type == 'log_loss':
            Test_data = self.construct_dataset(X_, Y_for_logloss, X_sparse_list, X_sparse)
        else:
            Test_data = self.construct_dataset(X_, Y_, X_sparse_list, X_sparse)
        print("# of test:", len(Y_))

        return Train_data, Validation_data, Test_data

    def read_data(self, file, data_num):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        if not self.is_sparse:
            X_ = np.zeros([data_num, self.features_M], dtype=np.float32)
        else:
            X_ = None
        Y_ = np.zeros([data_num], dtype=np.float32)
        X_sparse_list = []
        Y_for_logloss = np.zeros([data_num], dtype=np.float32)
        i = 0
        for line in open(file, 'r'):
            indices = []
            values = []
            items = line.strip().split(' ')
            Y_[i] = 1.0 * float(items[0])

            if float(items[0]) > 0:  # > 0 as 1; others as 0
                v = 1.0
            else:
                v = 0.0
            Y_for_logloss[i] = v

            for item in items[1:]:
                key_value_pair = item.strip().split(':')
                key = int(key_value_pair[0])
                value = float(key_value_pair[1])
                if not self.is_sparse:
                    X_[i, key] = float(value)
                indices.append(key)
                values.append(float(value))
            X_sparse_list.append({'indices': indices, 'values': values})

            i = i + 1

        X_sparse = sparse_concat(X_sparse_list, self.features_M)

        return X_, Y_, Y_for_logloss, X_sparse_list, X_sparse

    def construct_dataset(self, X_, Y_, X_sparse_list, X_sparse):
        Data_Dic = {}
        Data_Dic['Y'] = Y_
        Data_Dic['X'] = X_
        Data_Dic['X_sparse_list'] = X_sparse_list
        Data_Dic['X_sparse'] = X_sparse
        return Data_Dic

    def truncate_features(self):
        """
        Make sure each feature vector is of the same length
        """
        num_variable = len(self.Train_data['X'][0])
        for i in xrange(len(self.Train_data['X'])):
            num_variable = min([num_variable, len(self.Train_data['X'][i])])
        # truncate train, validation and test
        for i in xrange(len(self.Train_data['X'])):
            self.Train_data['X'][i] = self.Train_data['X'][i][0:num_variable]
        for i in xrange(len(self.Validation_data['X'])):
            self.Validation_data['X'][i] = self.Validation_data['X'][i][0:num_variable]
        for i in xrange(len(self.Test_data['X'])):
            self.Test_data['X'][i] = self.Test_data['X'][i][0:num_variable]
        return num_variable


def transform_data(dataset='data/banana/banana'):
    random.seed(2017)

    file_train = open('.'.join([dataset, 'train', 'libfm']), 'w')
    file_validation = open('.'.join([dataset, 'validation', 'libfm']), 'w')
    file_test = open('.'.join([dataset, 'test', 'libfm']), 'w')

    for line in open('.'.join([dataset, 'origin', 'libfm']), 'r'):
        num_random = random.random()
        if num_random < 0.7:
            file_train.writelines(line)
        elif num_random < 0.9:
            file_validation.writelines(line)
        else:
            file_test.writelines(line)

    file_train.close()
    file_validation.close()
    file_test.close()


def scale_percentile(matrix, mins=None, maxs=None):
    """scale a matrix to 0-1
    """
    if mins is None:
        mins = np.percentile(matrix, 1, axis=0)
    if maxs is None:
        maxs = np.percentile(matrix, 99, axis=0) - mins
    if all(maxs):
        matrix = (matrix - mins[None, :]) / maxs[None, :]

    matrix = matrix.clip(0, 1)  # Limit the value between 0 and 1
    return matrix, mins, maxs


if __name__ == '__main__':
    random.seed(2016)
    category = []
    num_train = 0
    num_validation = 0
    num_test = 0
    for i in range(19020):
        num_random = random.random()
        if num_random < 0.7:
            num_train = num_train + 1
            category.append(0)
        elif num_random < 0.9:
            num_validation = num_validation + 1
            category.append(1)
        else:
            num_test = num_test + 1
            category.append(2)

    data = [{'X': np.zeros([num_train, 10], dtype=np.float32), 'Y': np.zeros([num_train], dtype=np.float32)},
            {'X': np.zeros([num_validation, 10], dtype=np.float32), 'Y': np.zeros([num_validation], dtype=np.float32)},
            {'X': np.zeros([num_test, 10], dtype=np.float32), 'Y': np.zeros([num_test], dtype=np.float32)}]

    data_i = [0, 0, 0]
    i = 0
    for line in open('data/magic04/magic04.data'):
        item = line.strip().split(',')
        c = category[i]
        for j in range(10):
            data[c]['X'][data_i[c], j] = float(item[j])

        if item[10] == 'g':
            data[c]['Y'][data_i[c]] = 1.0
        else:
            data[c]['Y'][data_i[c]] = 0.0

        i = i + 1
        data_i[c] = data_i[c] + 1

    for i in range(3):
        rng_state = np.random.get_state()
        np.random.shuffle(data[i]['X'])
        np.random.set_state(rng_state)
        np.random.shuffle(data[i]['Y'])

    data[0]['X'], mins, maxs = scale_percentile(data[0]['X'])
    data[1]['X'], _, _ = scale_percentile(data[1]['X'])
    data[2]['X'], _, _ = scale_percentile(data[2]['X'])

    # print(data[0]['Y'])
    # print(data[1]['X'])
    # print(data[2]['X'])

    pickle.dump(data[0], open('data/magic04/magic04.train.dat', 'w'))
    pickle.dump(data[1], open('data/magic04/magic04.validation.dat', 'w'))
    pickle.dump(data[2], open('data/magic04/magic04.test.dat', 'w'))


