from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
import numpy as np
import tensorflow as tf
##import helpers


PAD = 0
EOS = 1

vocab_size = 10

input_embedding_size = 20

tf.reset_default_graph()
session = tf.InteractiveSession()

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units*2

encoder_inputs= tf.placeholder(shape =(None,None), dtype=tf.int32,name = 'encoder_inputs')
encoder_inputs_length = tf.placeholder(shape =(None), dtype=tf.int32,name = 'encoder_inputs_length')
decoder_targets = tf.placeholder(shape =(None,None), dtype=tf.int32,name = 'decoder_targets')


embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0,1),dtype = tf.float32)


encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

encoder_cell = LSTMCell(encoder_hidden_units)

((encoder_fw_outputs, encoder_bw_outputs),(encoder_fw_final_state,encoder_bw_final_state))= (tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell, cell_bw=encoder_cell,inputs=encoder_inputs_embedded, sequence_length=encoder_inputs_length, dtype=tf.float32, time_major=True))


encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)                                                                                            
   
encoder_final_state_c = tf.concat((encoder_fw_final_state.c),1)

encoder_final_state_h  = tf.concat((encoder_bw_final_state.h),1)

encoder_final_state = LSTMStateTuple(encoder_final_state_c, encoder_final_state_h)

decoder_cell = LSTMCell(decoder_hidden_units)
encoder_max_time, batch_size = tf. unstack(tf.shape(encoder_inputs))
decoder_lengths = encoder_inputs_length + 3

Weight = tf.Variable(tf.random_uniform([decoder_hidden_units],vocab_size,-1,1),dtype = tf.float32)
biases = tf.Variable(tf.zeros([vocab_size]),dtype=tf.float32)

assert EOS==1 and PAD==0
eos_time_slice = tf.ones([batch_size], dtype = tf.int32, name = 'EOS')
pad_time_slice = tf.zeros([batch_size], dtype = tf.int32, name = 'PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)



def loop_fn_initial():
    initial_elements_finished = (0>= decoder_lengths)
    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None
    return (initial_elements_finished, initial_input, initial_cell_state,initial_cell_output,initial_loop_state)
