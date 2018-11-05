import numpy as np
import scipy as sp
import os
import glob
#
from codes.segmentation import sample_batch
#
def test_val_split_v2(data_path, train_percentage = 90):
    file_list = np.array(glob.glob(os.path.join(data_path, '*.mat')))
    no_of_files = len(file_list)
    no_of_train = np.ceil(len(file_list) * train_percentage / 100).astype(np.int32)
    no_of_val = len(file_list) - no_of_train  # not used
    np.random.seed(20)
    index = np.random.permutation(no_of_files)
    train_file_list = list(file_list[index[:no_of_train]])
    val_file_list = list(file_list[index[no_of_train:]])
    #
    return (train_file_list, val_file_list)
#
def prediction(labels):
    #
    labels = np.array(labels)
    #
    if np.sum(labels == 1) < 2 and np.sum(labels == 5) < 2 and np.sum(labels == 6) < 2:
        predict = sp.stats.mode(labels)[0][0]
    #
    else:
        key, val = np.unique(labels, return_counts=True)
        #
        dctn = {}
        dctn[1] = val[key == 1][0] if 1 in key else 0
        dctn[5] = val[key == 5][0] if 5 in key else 0
        dctn[6] = val[key == 6][0] if 6 in key else 0
        #
        candidate = max(dctn.keys(), key=(lambda key: dctn[key]))
        if np.count_nonzero(labels == candidate) > 2:
            predict = int(candidate)
        else:
            if np.diff(np.where(labels == candidate)[0])[0] == 1:
                predict = int(candidate)
            else:
                predict = sp.stats.mode(labels)[0][0]
    #
    return predict
#
def prediction_v2(labels):
    #
    labels = np.array(labels)
    #
    if np.sum(labels == 1) < 1 and np.sum(labels == 5) < 1 and np.sum(labels == 6) < 1:
        predict = sp.stats.mode(labels)[0][0]
    #
    else:
        key, val = np.unique(labels, return_counts=True)
        #
        dctn = {}
        dctn[1] = val[key == 1][0] if 1 in key else 0
        dctn[5] = val[key == 5][0] if 5 in key else 0
        dctn[6] = val[key == 6][0] if 6 in key else 0
        #
        predict = int(max(dctn.keys(), key=(lambda key: dctn[key])))
    #
    return predict
#
def sample_batch_for_training(file_list, ref_file):
    #np.random.shuffle(file_list)
    idx = np.random.choice(len(file_list), size = 10, replace = False)
    #
    data, labels, seq_length = sample_batch(file_list[idx[9]], ref_file, mode = 'training')
    for i in np.arange(len(idx) - 1):
        segs, labs, lens = sample_batch(file_list[idx[i]], ref_file, mode = 'training')
        data = np.concatenate((data, segs))
        labels = np.concatenate((labels, labs))
        seq_length = np.concatenate((seq_length, lens))
    #
    index = np.random.permutation(labels.shape[-1])
    return data[index,:,:], labels[index], seq_length[index]
#
