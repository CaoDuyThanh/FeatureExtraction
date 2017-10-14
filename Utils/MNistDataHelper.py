import numpy
import os
import pickle, cPickle
from DataHelper import DatasetHelper
from FileHelper import *
import xml.etree.cElementTree as ET
import gzip
import random

class MNistDataHelper(DatasetHelper):

    def _convert_type(self,
                      _variables,
                      _types):
        variables = []
        for _variable, _type in zip(_variables, _types):
            variables.append(_variable.astype(_type))
        return variables

    def __init__(self,
                 _train_label,
                 _test_label,
                 _dataset_path = None):
        DatasetHelper.__init__(self)

        # Check parameters
        check_not_none(_dataset_path, 'datasetPath'); check_path_exist(_dataset_path)

        # Set parameters
        self.dataset_path  = _dataset_path

        # Load the dataset
        with gzip.open(self.dataset_path, 'rb') as _file:
            try:
                _train_set, _valid_set, _test_set = pickle.load(_file, encoding='latin1')
            except:
                _train_set, _valid_set, _test_set = pickle.load(_file)

        self.train_set_x, self.train_set_y = _train_set
        self.valid_set_x, self.valid_set_y = _valid_set
        self.test_set_x, self.test_set_y   = _test_set

        self.train_set_x, self.train_set_y = self._convert_type([self.train_set_x, self.train_set_y], ['float32', 'int32'])
        self.valid_set_x, self.valid_set_y = self._convert_type([self.valid_set_x, self.valid_set_y], ['float32', 'int32'])
        self.test_set_x, self.test_set_y   = self._convert_type([self.test_set_x, self.test_set_y], ['float32', 'int32'])

        self.all_set_x = numpy.concatenate((self.train_set_x, self.valid_set_x, self.test_set_x), axis = 0)
        self.all_set_y = numpy.concatenate((self.train_set_y, self.valid_set_y, self.test_set_y), axis = 0)

        # Sort based on label
        _idx = numpy.argsort(self.all_set_y)
        self.all_set_x = self.all_set_x[_idx,]
        self.all_set_y = self.all_set_y[_idx,]

        self.train_set_x = []
        self.train_set_y = []
        _idx             = []
        for _label in _train_label:
            _idx.append(numpy.where(self.all_set_y == _label)[0])
        _idx = numpy.concatenate(tuple(_idx), axis = 0)
        self.train_set_x = self.all_set_x[_idx,]
        self.train_set_y = self.all_set_y[_idx,]
        _idx = range(len(self.train_set_x))
        random.shuffle(_idx)
        self.train_set_x = self.train_set_x[_idx,]
        self.train_set_y = self.train_set_y[_idx,]

        self.test_set_y = []
        self.test_set_y = []
        _idx            = []
        for _label in _test_label:
            _idx.append(numpy.where(self.all_set_y == _label)[0])
        _idx = numpy.concatenate(tuple(_idx), axis=0)
        self.test_set_x = self.all_set_x[_idx,]
        self.test_set_y = self.all_set_y[_idx,]
        _idx = range(len(self.test_set_x))
        random.shuffle(_idx)
        self.test_set_x = self.test_set_x[_idx,]
        self.test_set_y = self.test_set_y[_idx,]

        self.all_set_x   = self.all_set_x.reshape((len(self.all_set_x), 1, 28, 28))
        self.train_set_x = self.train_set_x.reshape((len(self.train_set_x), 1, 28, 28))
        self.test_set_x  = self.test_set_x.reshape((len(self.test_set_x), 1, 28, 28))


