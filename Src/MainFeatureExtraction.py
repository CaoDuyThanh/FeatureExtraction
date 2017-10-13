import numpy
import math
import random
import thread
import timeit
import gzip
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import multiprocessing
import pickle, cPickle
from joblib import Parallel, delayed
from random import shuffle
from Utils.FileHelper import *
from Utils.MNistDataHelper import *
from Models.FeatExtModel import *

########################################################################################################################
#                                                                                                                      #
#    CONFIGURATIONS SESSION                                                                                            #
#                                                                                                                      #
########################################################################################################################
# RECORD PATH
RECORD_PATH = 'FEAEXT_record.pkl'

# TRAIN | VALID | TEST RATIO
# TRAIN_RATIO = 0.8
# VALID_RATIO = 0.1
# TEST_RATIO  = 0.1

# TRAINING HYPER PARAMETER
TRAIN_STATE        = 1    # Training state
VALID_STATE        = 0    # Validation state
BATCH_SIZE         = 16
NUM_EPOCH          = 100
MAX_ITERATION      = 100000
LEARNING_RATE      = 0.001      # Starting learning rate
DISPLAY_FREQUENCY  = 100;       INFO_DISPLAY = '\r%sLearning rate = %f - Epoch = %d - Iter = %d - Cost = %f - Best cost = %f - Best prec = %f - Mags = %f'
SAVE_FREQUENCY     = 2000
VALIDATE_FREQUENCY = 2000

# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/MachineLearning/Dataset/MNIST/mnist.pkl.gz'

START_EPOCH     = 0
START_ITERATION = 0

# STATE PATH
STATE_PATH      = '../Pretrained/DAFE_CurrentState.pkl'
BEST_COST_PATH  = '../Pretrained/DAFE_Cost_Best.pkl'
BEST_PREC_PATH  = '../Pretrained/DAFE_Prec_Best.pkl'

#  GLOBAL VARIABLES
dataset         = None
FEAEXT_model    = None

########################################################################################################################
#                                                                                                                      #
#    LOAD DATASET SESSIONS                                                                                             #
#                                                                                                                      #
########################################################################################################################
def _load_dataset():
    global dataset
    dataset = MNistDataHelper(_dataset_path = DATASET_PATH,
                              _train_label  = [0, 1, 2, 3, 4, 6, 8, 9],
                              _test_label   = [5, 7])
    print ('|-- Load dataset ! Completed !')

########################################################################################################################
#                                                                                                                      #
#    CREATE FEATURE EXTRACTION MODEL                                                                                   #
#                                                                                                                      #
########################################################################################################################
def _create_FEAEXT_model():
    global FEAEXT_model
    FEAEXT_model = FEAEXTModel()

########################################################################################################################
#                                                                                                                      #
#    VALID FEATURE EXTRACTION MODEL..........                                                                          #
#                                                                                                                      #
########################################################################################################################
def _valid_model(_dataset, _valid_data, _pre_extract, _batch_size):
    return 0

########################################################################################################################
#                                                                                                                      #
#    TRAIN FEATURE EXTRACTION MODEL..........                                                                          #
#                                                                                                                      #
########################################################################################################################
def _train_model():
    global dataset, \
           FEAEXT_model
    # ===== Prepare dataset =====
    # ----- Get all data and devide into TRAIN | VALID | TEST set -----
    train_set_x = dataset.train_set_x
    train_set_y = dataset.train_set_y
    test_set_x  = dataset.test_set_x
    test_set_y  = dataset.test_set_y

    # ----- Shuffle data -----
    random.seed(123456)

    # ----- Divide into TRAIN|VALID|TEST set -----

    # ===== Load data record =====
    print ('|-- Load previous record !')
    iter_train_record = []
    cost_train_record = []
    iter_valid_record = []
    cost_valid_record = []
    best_valid_cost   = 10000
    if check_file_exist(RECORD_PATH, _throw_error = False):
        _file = open(RECORD_PATH)
        iter_train_record = pickle.load(_file)
        cost_train_record = pickle.load(_file)
        iter_valid_record = pickle.load(_file)
        cost_valid_record = pickle.load(_file)
        best_valid_cost   = pickle.load(_file)
        _file.close()
    print ('|-- Load previous record ! Completed !')

    # ===== Load state =====
    print ('|-- Load state !')
    if check_file_exist(STATE_PATH, _throw_error = False):
        _file = open(STATE_PATH)
        FEAEXT_model.load_state(_file)
        _file.close()
    print ('|-- Load state ! Completed !')

    # ===== Training start =====
    # ----- Temporary record -----
    _costs = []
    _mags  = []
    _epoch = START_EPOCH
    _iter  = START_ITERATION
    _learning_rate = LEARNING_RATE

    # ----- Train -----
    for _epoch in xrange(START_EPOCH, NUM_EPOCH):
        _idx = range(len(train_set_x))
        random.shuffle(_idx)
        train_set_x = train_set_x[_idx,]
        train_set_y = train_set_y[_idx,]
        _num_batch_trained_data = len(train_set_x) // BATCH_SIZE

        for _id_batch_trained_data in range(_num_batch_trained_data):
            _feat_batch = []
            for i in range(100):
                _id_batch_trained_feat = random(_num_batch_trained_data)
                _train_batch_x = train_set_x[ _id_batch_trained_feat * BATCH_SIZE :
                                             (_id_batch_trained_feat + 1) * BATCH_SIZE,]
                _train_batch_y = train_set_y[ _id_batch_trained_feat * BATCH_SIZE :
                                             (_id_batch_trained_feat + 1) * BATCH_SIZE, ]

                # Update
                result = FEAEXT_model.feat_train_func(TRAIN_STATE,
                                                      _learning_rate,
                                                      _train_batch_x,
                                                      _train_batch_y)
                _feat_batch.append(result[0])
            _feat_mean = numpy.mean(_feat_batch)

            _train_batch_x = train_set_x[_id_batch_trained_feat * BATCH_SIZE :
                                        (_id_batch_trained_feat + 1) * BATCH_SIZE, ]
            _train_batch_y = train_set_y[_id_batch_trained_feat * BATCH_SIZE :
                                        (_id_batch_trained_feat + 1) * BATCH_SIZE, ]
            _iter += 1
            result = FEAEXT_model.gens_train_func(TRAIN_STATE,
                                                  _learning_rate,
                                                  _train_batch_x,
                                                  _train_batch_y)

            # Temporary save info
            _costs.append(result[0])
            _train_end_time = timeit.default_timer()
            # Print information
            print '\r|-- Trained %d / %d batch - Time = %f' % (_num_batch_trained_data, _num_batch_train_data, _train_end_time - _train_start_time),

            if _iter % SAVE_FREQUENCY == 0:
                # Save record
                _file = open(RECORD_PATH, 'wb')
                pickle.dump(iter_train_record, _file, 2)
                pickle.dump(cost_train_record, _file, 2)
                pickle.dump(iter_valid_record, _file, 2)
                pickle.dump(cost_valid_record, _file, 2)
                pickle.dump(best_valid_cost, _file, 2)
                _file.close()
                print ('+ Save record ! Completed !')

                # Save state
                _file = open(STATE_PATH, 'wb')
                FEAEXT_model.save_state(_file)
                _file.close()
                print ('+ Save state ! Completed !')

            if _iter % DISPLAY_FREQUENCY == 0:
                # Print information of current training in progress
                print (INFO_DISPLAY % ('|-- ', _learning_rate, _epoch, _iter, numpy.mean(_costs), best_valid_cost, best_valid_prec, numpy.mean(_mags)))
                iter_train_record.append(_iter)
                cost_train_record.append(numpy.mean(_costs))
                _costs = []
                _mags  = []

            # if _iter % VALIDATE_FREQUENCY == 0:
            #     print ('------------------- Validate Model -------------------')
            #     _cost_valid, _prec_valid, _valid_pre_extract = _valid_model(_dataset     = dataset,
            #                                                                 _valid_data  = valid_data,
            #                                                                 _pre_extract = _valid_pre_extract,
            #                                                                 _batch_size  = 1)
            #     iter_valid_record.append(_iter)
            #     cost_valid_record.append(_cost_valid)
            #     print ('\n+ Validate model finished! Cost = %f - Prec = %f' % (_cost_valid, _prec_valid))
            #     print ('------------------- Validate Model (Done) -------------------')
            #
            #     # # Save model if its cost better than old one
                # if (_prec_valid > best_valid_prec):
                #     best_valid_prec = _prec_valid
                #
                #     # Save best model
                #     _file = open(BEST_PREC_PATH, 'wb')
                #     DAFeat_model.save_model(_file)
                #     _file.close()
                #     print ('+ Save best prec model ! Complete !')
                #
                # if (_cost_valid < best_valid_cost):
                #     best_valid_cost = _cost_valid
                #
                #     # Save best model
                #     _file = open(BEST_COST_PATH, 'wb')
                #     DAFeat_model.save_model(_file)
                #     _file.close()
                #     print ('+ Save best cost model ! Complete !')

def _test_model():
    global dataset, \
           FEAEXT_model
    # ===== Prepare dataset =====
    # ----- Get all data and devide into TRAIN | VALID | TEST set -----

    # ----- Shuffle data -----

    # ----- Divide into TRAIN|VALID|TEST set -----

    print ('------------------- Test Model -------------------')
    # ===== Load best state =====
    print ('|-- Load best model !')
    if check_file_exist(BEST_PREC_PATH, _throw_error=False):
        _file = open(BEST_PREC_PATH)
        FEAEXT_model.load_model(_file)
        _file.close()
    print ('|-- Load best model ! Completed !')

    _test_pre_extract = dict()
    _cost_test, _prec_test, _test_pre_extract = _valid_model(_dataset     = dataset,
                                                             _valid_data  = test_data,
                                                             _pre_extract = _test_pre_extract,
                                                             _batch_size  = 1)
    print ('\n+ Test model finished! Cost = %f - Prec = %f' % (_cost_test, _prec_test))
    print ('------------------- Test Model (Done) -------------------')

if __name__ == '__main__':
    _load_dataset()
    _create_FEAEXT_model()
    _train_model()
    # _test_model()