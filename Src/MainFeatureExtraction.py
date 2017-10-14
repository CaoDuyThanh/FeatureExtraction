import numpy
import math
import random
import thread
import timeit
import theano
import theano.tensor as T
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
NUM_EPOCH          = 1000
MAX_ITERATION      = 100000
LEARNING_RATE      = 0.00001      # Starting learning rate
DISPLAY_FREQUENCY  = 10;       INFO_DISPLAY = '\r%sLearning rate = %f - Epoch = %d - Iter = %d - Costf = %f - Costg = %f'
SAVE_FREQUENCY     = 10000
VALIDATE_FREQUENCY = 500
GENERATE_SAMPLE    = 100

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

    # ===== Load data record =====
    print ('|-- Load previous record !')
    iter_train_record  = []
    costf_train_record = []
    costg_train_record = []
    best_valid_cost    = 10000
    if check_file_exist(RECORD_PATH, _throw_error = False):
        _file = open(RECORD_PATH)
        iter_train_record  = pickle.load(_file)
        costf_train_record = pickle.load(_file)
        costg_train_record = pickle.load(_file)
        best_valid_cost    = pickle.load(_file)
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
    _costsf = []
    _costsg = []
    _epoch = START_EPOCH
    _iter  = START_ITERATION
    _learning_rate = LEARNING_RATE

    # ----- Train -----
    while _epoch < NUM_EPOCH:
        _idx = range(len(train_set_x))
        random.shuffle(_idx)
        train_set_x = train_set_x[_idx,]
        train_set_y = train_set_y[_idx,]
        _num_batch_trained_data = len(train_set_x) // BATCH_SIZE

        _epoch += 1
        for _id_batch_trained_data in range(_num_batch_trained_data):
            _feat_batch = []
            _train_start_time = timeit.default_timer()
            for i in range(20):
                _id_batch_trained_feat = random.randint(0, _num_batch_trained_data)
                _true_batch_x = train_set_x[ _id_batch_trained_feat * BATCH_SIZE :
                                             (_id_batch_trained_feat + 1) * BATCH_SIZE,]
                _true_batch_y = train_set_y[ _id_batch_trained_feat * BATCH_SIZE :
                                             (_id_batch_trained_feat + 1) * BATCH_SIZE, ]

                _noise_batch_x = FEAEXT_model.gens_gen_img_func1(_true_batch_x)[0]
                _train_batch_x = numpy.concatenate((_true_batch_x, _noise_batch_x), axis = 0)
                _train_batch_y = _true_batch_y

                # Update
                result = FEAEXT_model.feat_train_func(TRAIN_STATE,
                                                      _learning_rate,
                                                      _train_batch_x,
                                                      _train_batch_y)
                _feat_batch.append(result[0])
            _costsf.append(numpy.mean(_feat_batch))

            _train_batch_x = train_set_x[_id_batch_trained_data * BATCH_SIZE :
                                        (_id_batch_trained_data + 1) * BATCH_SIZE, ]
            _train_batch_y = train_set_y[_id_batch_trained_data * BATCH_SIZE :
                                        (_id_batch_trained_data + 1) * BATCH_SIZE, ]
            _iter += 1
            result = FEAEXT_model.gens_train_func(TRAIN_STATE,
                                                  _learning_rate,
                                                  _train_batch_x,
                                                  _train_batch_y)
            # Temporary save info
            _costsg.append(result[0])
            _train_end_time = timeit.default_timer()
            # Print information
            print '\r|-- Trained %d / %d batch - Time = %f' % (_id_batch_trained_data, _num_batch_trained_data, _train_end_time - _train_start_time),

            if _iter % SAVE_FREQUENCY == 0:
                # Save record
                _file = open(RECORD_PATH, 'wb')
                pickle.dump(iter_train_record, _file, 2)
                pickle.dump(costf_train_record, _file, 2)
                pickle.dump(costg_train_record, _file, 2)
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
                print (INFO_DISPLAY % ('|-- ', _learning_rate, _epoch, _iter, numpy.mean(_costsf), numpy.mean(_costsg)))
                iter_train_record.append(_iter)
                costf_train_record.append(numpy.mean(_costsf))
                costg_train_record.append(numpy.mean(_costsg))
                _costsf  = []
                _costsg  = []

            if _iter % GENERATE_SAMPLE == 0:
                _temp_batch_x = numpy.transpose(_train_batch_x, (0, 2, 3, 1))
                sample_batch = FEAEXT_model.gens_gen_img_func(_train_batch_x)[0]
                for (sample, origin, label) in zip(sample_batch, _temp_batch_x, _train_batch_y):
                    _file_name = './sample/%d.jpg' % label
                    cv2.imwrite(_file_name, sample * 255)

                    _file_name = './sample/o_%d.jpg' % label
                    cv2.imwrite(_file_name, origin * 255)

            if _iter % VALIDATE_FREQUENCY == 0:
                # print ('------------------- Validate Model -------------------')
                # _cost_valid, _prec_valid, _valid_pre_extract = _valid_model(_dataset     = dataset,
                #                                                             _valid_data  = valid_data,
                #                                                             _pre_extract = _valid_pre_extract,
                #                                                             _batch_size  = 1)
                # iter_valid_record.append(_iter)
                # cost_valid_record.append(_cost_valid)
                # print ('\n+ Validate model finished! Cost = %f - Prec = %f' % (_cost_valid, _prec_valid))
                # print ('------------------- Validate Model (Done) -------------------')

                _num_batch_tested_data = len(test_set_x) // BATCH_SIZE
                _id_batch_tested_data  = random.randint(0, _num_batch_tested_data)
                _test_batch_x = test_set_x[ _id_batch_tested_data * BATCH_SIZE:
                                           (_id_batch_tested_data + 1) * BATCH_SIZE, ]
                _test_batch_y = test_set_y[ _id_batch_tested_data * BATCH_SIZE:
                                           (_id_batch_tested_data + 1) * BATCH_SIZE, ]
                _temp_batch_x = numpy.transpose(_test_batch_x, (0, 2, 3, 1))

                sample_batch = FEAEXT_model.gens_gen_img_func(_test_batch_x)[0]
                for (sample, origin, label) in zip(sample_batch, _temp_batch_x, _test_batch_y):
                    _file_name = './sample/%d.jpg' % label
                    cv2.imwrite(_file_name, sample * 255)

                    _file_name = './sample/o_%d.jpg' % label
                    cv2.imwrite(_file_name, origin * 255)


                    # # Save model if its cost better than old one
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

import pqkmeans
def _test_model():
    global dataset, \
           FEAEXT_model
    print ('------------------- Test Model -------------------')
    # ===== Load best state =====
    print ('|-- Load best model !')
    if check_file_exist(STATE_PATH, _throw_error = False):
        _file = open(STATE_PATH)
        FEAEXT_model.load_model(_file)
        _file.close()
    print ('|-- Load best model ! Completed !')

    # ===== Create feature =====
    test_set_x = dataset.all_set_x
    test_set_y = dataset.all_set_y
    features   = []
    _num_batch_test_data = len(test_set_x) // BATCH_SIZE
    for _id_batch_tested_data in range(_num_batch_test_data):
        _test_batch_x = test_set_x[_id_batch_tested_data * BATCH_SIZE:
                                  (_id_batch_tested_data + 1) * BATCH_SIZE, ]
        result = FEAEXT_model.feat_ext_func(_test_batch_x)
        features.append(result[0])
    features = numpy.concatenate(tuple(features), axis = 0)
    encoder  = pqkmeans.encoder.PQEncoder(num_subdim = 4, Ks = 256)
    encoder.fit(features)
    X_pqcode  = encoder.transform(features)
    kmeans    = pqkmeans.clustering.PQKMeans(encoder = encoder, k = 10)
    clustered = kmeans.fit_predict(X_pqcode)
    clustered = numpy.asarray(clustered, dtype = 'int32')
    right = 0
    labels_set = []
    pred_set   = []
    for l in range(10):
        idx = numpy.where(clustered == l)
        sub_clustered = clustered[idx,]
        sub_test_y    = test_set_y[idx,]

        labels = numpy.zeros((10,))
        for la in range(10):
            labels[la] = numpy.sum(sub_test_y == la)
        labels_set.append(labels)

        right += numpy.max(labels)
        pred_set.append(numpy.argmax(labels))

    acc = right / len(test_set_x)
    print acc
    print pred_set

    # _count = 0
    # test_set_x = numpy.transpose(test_set_x, (0, 2, 3, 1))
    # for _id, _sample in zip(clustered, test_set_x):
    #     _path = './sample/%d' % (_id)
    #     if (check_path_exist(_path) is False):
    #         create_path(_path)
    #     _count += 1
    #     _file_name = './sample/%d/%d.jpg' % (_id, _count)
    #     cv2.imwrite(_file_name, _sample * 255)

    print ('------------------- Test Model (Done) -------------------')

if __name__ == '__main__':
    _load_dataset()
    _create_FEAEXT_model()
    _train_model()
    # _test_model()