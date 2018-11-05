import random
import os
import argparse
import csv
import glob
from scipy import io
import numpy as np
import tensorflow as tensorflow
from tensorflow.contrib.layers import fully_connected
import tensorflow.contrib.rnn as recurrent
#
from codes.pre_processing import *
from codes.segmentation import *
from codes.utils import *
from codes.training import *
from codes.model import *

'''
cspc2018_challenge score

'''

'''
Save prdiction answers to answers.csv in local path, the first column is recording name and the second
column is prediction label, for example:
Recoding    Result
B0001       1
.           .
.           .
.           .
'''
def cpsc2018(record_base_path):
    # ecg = scipy.io.loadmat(record_path)
    ###########################INFERENCE PART################################

    ## Please process the ecg data, and output the classification result.
    ## result should be an integer number in [1, 9].

    inputs, labels, seq_length, logits, acuracy = build_model_graph()
    model_dir = './model'
    
    with open('answers.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # column name
        writer.writerow(['Recording', 'Result'])
        with tf.Session() as sess:
            load_model(model_dir, sess)
            for mat_item in os.listdir(record_base_path):
                if mat_item.endswith('.mat') and (not mat_item.startswith('._')):
                    #
                    segs, lens = process_data(os.path.join(record_base_path, mat_item))
                    logits_val = sess.run(logits, feed_dict = {inputs: segs, seq_length: lens})
                    result = prediction_v2(np.argmax(logits_val, axis = 1)) + 1
                    #
                    ## If the classification result is an invalid number, the result will be determined as normal(1).
                    if result > 9 or result < 1 or not(str(result).isdigit()):
                        result = 1
                    record_name = mat_item.rstrip('.mat')
                    answer = [record_name, result]
                    # write result
                    writer.writerow(answer)

        csvfile.close()

    ###########################INFERENCE PART################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--recording_path',
                        help='path saving test record file')

    args = parser.parse_args()

    result = cpsc2018(record_base_path=args.recording_path)
