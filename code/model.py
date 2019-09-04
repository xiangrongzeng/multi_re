#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2017/8/19
import json
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from const import TrainMethod, DecoderMethod

logger = logging.getLogger('mylogger')


def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))



def get_embedding_table(config):
    if os.path.isfile(config.words_id2vector_filename):
        logger.info('Word Embedding init from %s' % config.words_id2vector_filename)
        words_id2vec = json.load(open(config.words_id2vector_filename, 'r'))
        words_vectors = [0] * len(words_id2vec)
        for id, vec in words_id2vec.items():
            words_vectors[int(id)] = vec
        # add eos embedding
        words_vectors.append(list(np.random.uniform(0, 1, config.embedding_dim)))
        words_embedding_table = tf.Variable(name='words_emb_table', initial_value=words_vectors, dtype=tf.float32)
    else:
        logger.info('Word Embedding random init')
    return words_embedding_table


def set_rnn_cell(name, num_units):
    if name.lower() == 'gru':
        return tf.contrib.rnn.GRUCell(num_units)
    elif name.lower() == 'lstm':
        return tf.contrib.rnn.LSTMCell(num_units)
    else:
        return tf.contrib.rnn.BasicRNNCell(num_units)


class Encoder:
    def __init__(self, config, max_sentence_length, embedding_table, pos_embedding_table=None):
        self.max_sentence_length = max_sentence_length
        self.encoder_fw_cell = None
        self.encoder_bw_cell = None
        self.embedding_table = embedding_table
        self.input_sentence_fw_pl = tf.placeholder(dtype=tf.int32, shape=[config.batch_size, max_sentence_length],
                                                   name='input_sentence_fw')
        self.input_sentence_pos_fw_pl = tf.placeholder(dtype=tf.int32, shape=[config.batch_size, max_sentence_length],
                                                       name='input_sentence_pos_fw')
        self.input_sentence_length = tf.placeholder(dtype=tf.int32, shape=[config.batch_size],
                                                    name='input_sentence_length')
        self.outputs = None
        self.state = None
        self.config = config

    def set_cell(self, name, num_units):
        with tf.variable_scope('encoder'):
            self.encoder_fw_cell = set_rnn_cell(name, num_units)
            self.encoder_bw_cell = set_rnn_cell(name, num_units)

    def _encode(self, inputs):
        input_vector = tf.nn.embedding_lookup(self.embedding_table, inputs)
        logger.debug('Input vector shape %s' % input_vector.get_shape())
        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_fw_cell,
                                                         cell_bw=self.encoder_bw_cell,
                                                         inputs=input_vector,
                                                         dtype=tf.float32)

        if self.config.cell_name == 'lstm':
            logger.debug('Encoder before concat: output shape %s,%s' % (len(outputs), outputs[0].get_shape()))
            logger.debug('Encoder before concat: state shape %s,%s' % (np.shape(state), state[0][0].get_shape()))
            outputs = tf.concat(outputs, axis=-1)
            state = (
                tf.reduce_mean((state[0][0], state[1][0]), axis=0), tf.reduce_mean((state[0][1], state[1][1]), axis=0))
            logger.debug('Encoder: outputs shape %s' % outputs.get_shape())
            logger.debug('Encoder: state shape %s,%s' % (np.shape(state), state[0].get_shape()))
        elif self.config.cell_name == 'gru':
            outputs, state = tf.reduce_mean(outputs, axis=0), tf.reduce_mean(state, axis=0)
            logger.debug('Encoder: outputs shape %s' % outputs.get_shape())
            logger.debug('Encoder: state shape %s' % state.get_shape())
        else:
            logger.error('Undefined cell name %s' % self.config.cell_name)
            exit()
        return outputs, state

    def build(self, is_use_pos=False):
        logger.info('Encoding')
        with tf.variable_scope('seq_encoder'):
            if is_use_pos:
                inputs = [self.input_sentence_fw_pl, self.input_sentence_pos_fw_pl]
            else:
                inputs = self.input_sentence_fw_pl
            self.outputs, self.state = self._encode(inputs=inputs)


class Decoder:
    def __init__(self, decoder_output_max_length, encoder, config):
        self.config = config
        self.decoder_output_max_length = decoder_output_max_length

        self.encoder = encoder
        self.input_sentence_length = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size],
                                                    name='input_sentence_length')
        self.relations_append_eos_pl = tf.placeholder(dtype=tf.int32,
                                                      shape=[self.config.batch_size, self.config.relation_number + 1],
                                                      name='relations_append_eos')
        self.sparse_standard_outputs = tf.placeholder(dtype=tf.int64,
                                                      shape=[self.config.batch_size,
                                                             self.config.decoder_output_max_length],
                                                      name='standard_outputs')
        self.standard_outputs_mask = tf.placeholder(dtype=tf.float32,
                                                    shape=[self.config.batch_size,
                                                           self.decoder_output_max_length],
                                                    name='standard_outputs_mask')
        self.batch_bias4predict = tf.constant(
            value=[i * (self.config.relation_number + 1) for i in range(self.config.batch_size)],
            dtype=tf.int64,
            name='batch_bias4predict')
        #   for calc loss
        self.advantage_pl = tf.placeholder(dtype=tf.float32,
                                           shape=[self.config.batch_size, 1],
                                           name='advantage_pl')

        self.decode_cell = None
        self.actions_by_time = None
        self.probs_by_time = []
        self.picked_actions_prob = None
        self.losses = None
        self.opt = None
        self.cell_num_units = None
        self.tmp = []

    def set_cell(self, name, num_units):
        pass

    @staticmethod
    def do_predict(inputs, relation_number, t):
        with tf.variable_scope('predict_%s' % t) as scope:
            W = tf.get_variable(name='W',
                                shape=[int(inputs.get_shape()[-1]), relation_number],
                                dtype=tf.float32)
            b = tf.get_variable(name='b',
                                shape=relation_number,
                                dtype=tf.float32)
            logits = selu(tf.matmul(inputs, W)) + b
            return logits

    @staticmethod
    def do_eos(inputs, t):
        with tf.variable_scope('eos_%s' % t) as scope:
            W = tf.get_variable(name='W',
                                shape=[int(inputs.get_shape()[-1]), 1],
                                dtype=tf.float32)
            b = tf.get_variable(name='b',
                                shape=1,
                                dtype=tf.float32)
            logits = selu(tf.matmul(inputs, W)) + b
            return logits

    @staticmethod
    def do_copy(inputs, encoder_states, t):
        #   encoder_states的shape是[batch_size, max_sentence_length, hidden_dim]，现在转换为一个list，
        #   list中的每个元素的shape是[batch_size, hidden_dim]， list中一共有max_sentence_length个这样的元素
        # encoder_states = self.encoder.outputs
        encoder_states_by_time = tf.unstack(encoder_states, axis=1)
        with tf.variable_scope('copy_%s' % t) as scope:
            W = tf.get_variable(name='W',
                                shape=[int(encoder_states.get_shape()[-1]) + int(inputs.get_shape()[-1]), 1],
                                dtype=tf.float32)
            values = []
            for states in encoder_states_by_time:
                att_value = selu(tf.matmul(tf.concat((states, inputs), axis=1), W))
                values.append(att_value)
            values = tf.stack(values)
            values = tf.squeeze(values, -1)
            values = tf.transpose(values)
        return values

    @staticmethod
    def calc_context(decode_state, encoder_outputs, name, t):
        decoder_state = decode_state
        encoder_states_by_time = tf.unstack(encoder_outputs, axis=1)
        with tf.variable_scope('calc_context_%s' % t) as scope:
            W = tf.get_variable(name='W',
                                shape=[int(encoder_outputs.get_shape()[-1]) + int(decoder_state.get_shape()[-1]), 1],
                                dtype=tf.float32)

            values = []
            for states in encoder_states_by_time:
                att_value = selu(tf.matmul(tf.concat((states, decoder_state), axis=1), W))
                values.append(att_value)
            values = tf.stack(values)
            values = tf.squeeze(values, -1)
            values = tf.nn.softmax(tf.transpose(values))
            att_values = tf.unstack(values, axis=1)
            all = []
            for att_value, state in zip(att_values, encoder_states_by_time):
                att_value = tf.expand_dims(att_value, axis=1)
                all.append(att_value * state)
            context_vector = tf.reduce_mean(tf.stack(all), axis=0)
        logger.debug('context_vector shape %s' % context_vector.get_shape())
        return context_vector

    @staticmethod
    def combine_inputs(states, t):
        [inputs, context_vector] = states
        with tf.variable_scope('combine_state_%s' % t) as scope:
            W = tf.get_variable(name='W',
                                shape=[sum([int(s.get_shape()[-1]) for s in states]),
                                       int(states[0].get_shape()[-1])],
                                dtype=tf.float32)
            states = tf.concat(states, axis=1)
        return tf.matmul(states, W)

    def build(self, train_method, is_train=True):
        pass

    @staticmethod
    def get_prob(probs, indexes):
        depth = probs.get_shape()[-1]
        one_hot = tf.one_hot(indexes, depth)
        probs = tf.reduce_sum(probs * one_hot, axis=1)
        return probs

    @staticmethod
    def init_decoder_state(name, batch_size, cell_units_num):
        if name == 'gru':
            previous_state = tf.zeros(shape=[batch_size, cell_units_num], dtype=tf.float32)
        elif name == 'lstm':
            previous_state = (tf.zeros(shape=[batch_size, cell_units_num], dtype=tf.float32),
                              tf.zeros(shape=[batch_size, cell_units_num], dtype=tf.float32))
        else:
            logger.error('cell name must be "GRU" or "LSTM", but is %s' % name)
            raise
        return previous_state

    @staticmethod
    def update_decoder_state(name, encoder_state, previous_state):
        if name == 'gru':
            decode_state = tf.reduce_mean((encoder_state, previous_state), axis=0)
        elif name == 'lstm':
            decode_state = (tf.reduce_mean((encoder_state[0], previous_state[0]), axis=0),
                            tf.reduce_mean((encoder_state[1], previous_state[1]), axis=0))
        else:
            logger.error('cell name must be "GRU" or "LSTM", but is %s' % name)
            raise
        return decode_state

    @staticmethod
    def pick_action(logits, method, is_train):
        random_action = tf.squeeze(tf.multinomial(logits, 1))
        max_action = tf.squeeze(tf.argmax(logits, 1))
        if method == TrainMethod.RL_METHOD and is_train:
            return random_action
        else:
            return max_action


class RelationDecoder:
    def __init__(self, encoder, config, is_train):
        self.encoder = encoder
        self.config = config
        self.is_train = is_train
        with tf.variable_scope('relation_decoder'):
            logger.info('init relation decoder')
            self.relation_embedding_table = tf.get_variable('relation_emb_table',
                                                            shape=(
                                                            self.config.relation_number + 1, self.config.embedding_dim))

    def run_one_step(self, decode_cell, inputs, decode_state, standard_outputs, t):
        logger.info('run relation decoder one step %s' % t)
        with tf.variable_scope('decoder'):
            output, decode_state = decode_cell(inputs, decode_state)
            eos_logits = Decoder.do_eos(output, t)
            # predict
            predict_logits = Decoder.do_predict(output, self.config.relation_number, t)
            logger.debug('Decoder: predict_logits shape %s' % str(predict_logits.get_shape()))
            predict_logits = tf.concat((predict_logits, eos_logits), axis=1)
            predict_probs = tf.nn.softmax(predict_logits)
            logger.debug('Decoder: predict shape %s' % str(predict_probs.get_shape()))
            action = Decoder.pick_action(predict_logits, self.config.train_method, self.is_train)
            action_emb = tf.nn.embedding_lookup(self.relation_embedding_table, action)
            probs = Decoder.get_prob(predict_probs, standard_outputs)
            return action, probs, action_emb, decode_state


class PositionDecoder:
    def __init__(self, encoder, config, is_train):
        logger.info('init position decoder')
        self.encoder = encoder
        self.config = config
        self.is_train = is_train
        with tf.variable_scope('position_decoder'):
            self.mask_only_copy = tf.ones(shape=[self.config.batch_size, self.config.max_sentence_length],
                                          dtype=tf.float32)
            self.mask_eos = tf.ones(shape=[self.config.batch_size, 1], dtype=tf.float32)
            self.masked_position = None
            self.batch_bias4copy = tf.constant(
                value=[i * (self.config.max_sentence_length + 1) for i in range(self.config.batch_size)],
                dtype=tf.int64,
                name='batch_bias4copy')

    def run_one_step(self, decode_cell, inputs, decode_state, standard_outputs, sentence_eos_embedding, t,
                     is_first_entity_position=True):
        logger.info('run position decoder one step %s' % t)
        with tf.variable_scope('decoder'):
            if is_first_entity_position:
                c_mask = tf.concat([self.mask_only_copy, self.mask_eos], axis=1)
            else:
                c_mask = tf.concat([self.masked_position, self.mask_eos], axis=1)
            output, decode_state = decode_cell(inputs, decode_state)
            eos_logits = Decoder.do_eos(output, t)
            # copy
            copy_logits = Decoder.do_copy(output, self.encoder.outputs, t)
            logger.debug('Decoder: copy_logits shape %s' % str(copy_logits.get_shape()))
            copy_logits = tf.concat((copy_logits, eos_logits), axis=1) * c_mask
            copy_probs = tf.nn.softmax(copy_logits)
            logger.debug('Decoder: copy shape %s' % str(copy_probs.get_shape()))
            action = Decoder.pick_action(copy_logits, self.config.train_method, self.is_train)
            action_emb = tf.nn.embedding_lookup(sentence_eos_embedding, action + self.batch_bias4copy)
            probs = Decoder.get_prob(copy_probs, standard_outputs)
            if is_first_entity_position:
                copy_position_one_hot = tf.cast(
                    tf.one_hot(action, self.config.max_sentence_length + 1), tf.float32)
                copy_position_one_hot = copy_position_one_hot[:, :-1]  # remove the mask of eos
                self.masked_position = self.mask_only_copy * (1. - copy_position_one_hot)
            return action, probs, action_emb, decode_state


class TripleDecoder:
    def __init__(self, decoder_output_max_length, encoder, relation_decoder, position_decoder, decode_cell, config):
        self.config = config
        self.decoder_output_max_length = decoder_output_max_length

        self.encoder = encoder
        self.relation_decoder = relation_decoder
        self.position_decoder = position_decoder
        self.decode_cell = decode_cell
        self.input_sentence_length = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size],
                                                    name='input_sentence_length')
        self.relations_append_eos_pl = tf.placeholder(dtype=tf.int32,
                                                      shape=[self.config.batch_size, self.config.relation_number + 1],
                                                      name='relations_append_eos')
        self.input_sentence_append_eos_pl = tf.placeholder(dtype=tf.int32,
                                                           shape=[self.config.batch_size,
                                                                  self.config.max_sentence_length + 1],
                                                           name='input_sentence_append_eos_pl')
        self.sparse_standard_outputs = tf.placeholder(dtype=tf.int64,
                                                      shape=[self.config.batch_size,
                                                             self.config.decoder_output_max_length],
                                                      name='standard_outputs')
        self.standard_outputs_mask = tf.placeholder(dtype=tf.float32,
                                                    shape=[self.config.batch_size,
                                                           self.decoder_output_max_length],
                                                    name='standard_outputs_mask')
        sentence_eos_embedding = tf.nn.embedding_lookup(self.encoder.embedding_table, self.input_sentence_append_eos_pl)
        self.sentence_eos_embedding = tf.reshape(sentence_eos_embedding,
                                                 shape=[self.config.batch_size * (self.config.max_sentence_length + 1),
                                                        self.config.embedding_dim])
        self.batch_bias4predict = tf.constant(
            value=[i * (self.config.relation_number + 1) for i in range(self.config.batch_size)],
            dtype=tf.int64,
            name='batch_bias4predict')
        self.advantage_pl = tf.placeholder(dtype=tf.float32,
                                           shape=[self.config.batch_size, self.decoder_output_max_length],
                                           name='advantage_pl')

        self.actions_by_time = None
        self.probs_by_time = []
        self.picked_actions_prob = None
        self.losses = None
        self.opt = None
        self.cell_num_units = None
        self.tmp = []

    def build(self, train_method, decoder_method, is_train=True):
        logger.info('build triple decoder')
        sparse_standard_outputs_by_time = tf.unstack(self.sparse_standard_outputs, axis=1)
        probs_by_time = []
        actions_by_time = []
        with tf.variable_scope('triple_decoder'):
            # init decoder state
            decode_state = Decoder.init_decoder_state(self.config.cell_name, self.config.batch_size,
                                                      self.config.decoder_num_units)
            # init decoder input
            go_emb = tf.get_variable(name='GO', shape=[1, self.config.embedding_dim])
            inputs = tf.nn.embedding_lookup(go_emb,
                                            tf.zeros(shape=[self.config.batch_size],
                                                     dtype=tf.int64))

            if decoder_method == DecoderMethod.ONE_DECODER:
                probs_by_time, actions_by_time = self.onedecoder(sparse_standard_outputs_by_time, decode_state, inputs)
            elif decoder_method == DecoderMethod.MULTI_DECODER:
                probs_by_time, actions_by_time = self.multidecoder(sparse_standard_outputs_by_time, decode_state,
                                                                   inputs)
            elif decoder_method == DecoderMethod.SEPARATE_W:  # separateW
                probs_by_time, actions_by_time = self.separatew(sparse_standard_outputs_by_time, decode_state, inputs)
            else:
                logger.error(train_method)
                raise

            self.actions_by_time = tf.stack(actions_by_time, axis=1)
            self.probs_by_time = tf.stack(probs_by_time, axis=1)

        if is_train:
            logging.info('Prepare for loss')
            self.picked_actions_prob = self.probs_by_time
            self._loss(train_method)
            self._optimize()

    def _loss(self, train_method):
        logging.info('Calculating loss')
        if train_method == TrainMethod.NLL_METHOD:
            probs = self.picked_actions_prob
            probs = tf.clip_by_value(probs, 1e-10, 1.0)
            self.losses = tf.reduce_mean(-tf.log(probs))
        else:
            probs = self.picked_actions_prob
            probs = tf.clip_by_value(probs, 1e-10, 1.0)
            self.losses = tf.reduce_mean(-tf.log(probs) * self.advantage_pl)

    def _optimize(self):
        logging.info('Optimizing')
        learning_rate = self.config.learning_rate
        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.losses)

    def update(self, data, advantage, sess, sampled_action=None):
        feed_dict = {self.input_sentence_length: data.input_sentence_length,
                     self.standard_outputs_mask: data.standard_outputs_mask,
                     self.encoder.input_sentence_fw_pl: data.sentence_fw,
                     self.encoder.input_sentence_length: data.input_sentence_length,
                     self.input_sentence_append_eos_pl: data.input_sentence_append_eos,
                     self.relations_append_eos_pl: data.relations_append_eos,
                     self.advantage_pl: advantage}
        if sampled_action is not None:
            feed_dict[self.sparse_standard_outputs] = sampled_action
        else:
            feed_dict[self.sparse_standard_outputs] = data.standard_outputs

        loss_val, _ = sess.run([self.losses, self.opt], feed_dict=feed_dict)
        return loss_val

    def predict(self, data, sess):
        feed_dict = {self.input_sentence_length: data.input_sentence_length,
                     self.sparse_standard_outputs: data.standard_outputs,
                     self.encoder.input_sentence_fw_pl: data.sentence_fw,
                     self.encoder.input_sentence_length: data.input_sentence_length,
                     self.input_sentence_append_eos_pl: data.input_sentence_append_eos,
                     self.relations_append_eos_pl: data.relations_append_eos}
        actions = sess.run(self.actions_by_time, feed_dict=feed_dict)
        return actions

    def onedecoder(self, sparse_standard_outputs_by_time, decode_state, inputs):
        probs_by_time = []
        actions_by_time = []
        for triple_n in range(self.config.decoder_output_max_length / 3):
            cell = self.decode_cell[0]
            logger.info(cell.name)
            for i in range(3):
                t = 3 * triple_n + i
                key = ''
                if i > 1:
                    tf.get_variable_scope().reuse_variables()
                standard_outputs = sparse_standard_outputs_by_time[t]
                if i % 3 == 0:
                    decode_state = Decoder.update_decoder_state(self.config.cell_name, self.encoder.state,
                                                                decode_state)
                    actions, probs, inputs, decode_state = \
                        self.relation_decoder.run_one_step(cell, inputs, decode_state, standard_outputs,
                                                           'p' + str(key))
                elif i % 3 == 1:
                    actions, probs, inputs, decode_state = \
                        self.position_decoder.run_one_step(cell, inputs, decode_state, standard_outputs,
                                                           self.sentence_eos_embedding,
                                                           'c' + str(key), is_first_entity_position=True)
                else:
                    actions, probs, inputs, decode_state = \
                        self.position_decoder.run_one_step(cell, inputs, decode_state, standard_outputs,
                                                           self.sentence_eos_embedding,
                                                           'c' + str(key), is_first_entity_position=False)
                probs_by_time.append(probs)
                actions_by_time.append(actions)
        return probs_by_time, actions_by_time

    def multidecoder(self, sparse_standard_outputs_by_time, decode_state, inputs):
        probs_by_time = []
        actions_by_time = []
        for triple_n in range(self.config.decoder_output_max_length / 3):
            cell = self.decode_cell[triple_n]
            logger.info(cell.name)
            with tf.variable_scope('cell_%d' % triple_n):
                for i in range(3):
                    t = 3 * triple_n + i
                    key = triple_n
                    if i > 1:
                        tf.get_variable_scope().reuse_variables()
                    standard_outputs = sparse_standard_outputs_by_time[t]
                    if i % 3 == 0:
                        decode_state = Decoder.update_decoder_state(self.config.cell_name, self.encoder.state,
                                                                    decode_state)
                        actions, probs, inputs, decode_state = \
                            self.relation_decoder.run_one_step(cell, inputs, decode_state, standard_outputs,
                                                               'p' + str(key))
                    elif i % 3 == 1:
                        actions, probs, inputs, decode_state = \
                            self.position_decoder.run_one_step(cell, inputs, decode_state, standard_outputs,
                                                               self.sentence_eos_embedding,
                                                               'c' + str(key), is_first_entity_position=True)
                    else:
                        actions, probs, inputs, decode_state = \
                            self.position_decoder.run_one_step(cell, inputs, decode_state, standard_outputs,
                                                               self.sentence_eos_embedding,
                                                               'c' + str(key), is_first_entity_position=False)
                    probs_by_time.append(probs)
                    actions_by_time.append(actions)
        return probs_by_time, actions_by_time

    def separatew(self, sparse_standard_outputs_by_time, decode_state, inputs):
        probs_by_time = []
        actions_by_time = []
        for triple_n in range(self.config.decoder_output_max_length / 3):
            cell = self.decode_cell[0]
            logger.info(cell.name)
            for i in range(3):
                t = 3 * triple_n + i
                key = t
                standard_outputs = sparse_standard_outputs_by_time[t]
                if i % 3 == 0:
                    decode_state = Decoder.update_decoder_state(self.config.cell_name, self.encoder.state,
                                                                decode_state)
                    actions, probs, inputs, decode_state = \
                        self.relation_decoder.run_one_step(cell, inputs, decode_state, standard_outputs,
                                                           'p' + str(key))
                elif i % 3 == 1:
                    actions, probs, inputs, decode_state = \
                        self.position_decoder.run_one_step(cell, inputs, decode_state, standard_outputs,
                                                           self.sentence_eos_embedding,
                                                           'c' + str(key), is_first_entity_position=True)
                else:
                    actions, probs, inputs, decode_state = \
                        self.position_decoder.run_one_step(cell, inputs, decode_state, standard_outputs,
                                                           self.sentence_eos_embedding,
                                                           'c' + str(key), is_first_entity_position=False)
                probs_by_time.append(probs)
                actions_by_time.append(actions)
        return probs_by_time, actions_by_time
