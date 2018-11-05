import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import tensorflow.contrib.rnn as recurrent
import numpy as np
#
def build_model_graph(Training_mode = False, input_size = 12, time_steps = 1000, hidden_size = 100, num_hidden = 2, output_size = 9, keep_prob = 0.5):
	#
	# place-holdes
	inputs = tf.placeholder(tf.float32, [None, time_steps, input_size])
	labels = tf.placeholder(tf.int32, [None])
	seq_length = tf.placeholder(tf.int32, [None])
	#
	logits, accuracy = RNN_bidirectional(inputs, labels, seq_length, Training_mode = Training_mode, hidden_size = hidden_size, num_hidden = num_hidden, output_size = output_size, keep_prob = keep_prob)
	#
	return inputs, labels, seq_length, logits, accuracy
#
def load_model(model_dir, session):
	# the saver
	saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours =1)
	path = tf.train.get_checkpoint_state(model_dir)
	saver.restore(session, path.model_checkpoint_path)
#
def RNN_bidirectional(input_tensor, label_tensor, length_tensor, Training_mode, hidden_size, num_hidden, output_size, keep_prob):
	# 
    with tf.variable_scope("recurrent", initializer = tf.contrib.layers.variance_scaling_initializer()):
        cell = tf.nn.rnn_cell.BasicLSTMCell
        cells_fw = [cell(hidden_size) for _ in range(num_hidden)]
        cells_bw = [cell(hidden_size) for _ in range(num_hidden)]
        cells_fw = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = keep_prob if Training_mode is True else 1.0) for cell in cells_fw]
        cells_bw = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = keep_prob if Training_mode is True else 1.0) for cell in cells_bw]
        _, states_fw, states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=cells_fw,
                cells_bw=cells_bw,
                inputs=input_tensor,
                sequence_length = length_tensor,
                dtype=tf.float32)
        outputs_fw = tf.concat(states_fw[-1][-1], axis = 1)
        outputs_bw = tf.concat(states_bw[-1][-1], axis = 1)
        outputs = tf.concat([outputs_fw, outputs_bw], axis = 1)
        logits = tf.squeeze(fully_connected(outputs, output_size, activation_fn = None))
        #
        correct = tf.nn.in_top_k(logits, label_tensor, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return logits, accuracy
#