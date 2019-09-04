#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2017/8/20
import collections
import logging
import os

logger = logging.getLogger('mylogger')


class DataSet():
    NYT = 'nyt'
    WEBNLG = 'webnlg'
    name = None

    @staticmethod
    def set_dataset(dataset_name):
        if dataset_name == DataSet.NYT:
            DataSet.name = DataSet.NYT
        elif dataset_name == DataSet.WEBNLG:
            DataSet.name = DataSet.WEBNLG
        else:
            print 'Dataset %s is not exist!!!!!!!!!! ' % dataset_name
            exit()


class StandardOutputsBuildMethod:
    # the standard outputs are fixed during training and the triplets are not sorted
    FIXED_UNSORTED = 'FixedUnSorted'
    # the standard outputs are fixed during training but the triplets are sorted beforehand, sort by alphabet
    FIXED_SORTED_ALPHA = 'FixedSortedAlphabet'
    # the standard outputs are fixed during training but the triplets are sorted beforehand, sort by relation frequency
    FIXED_SORTED_FREQ = 'FixedSortedFreq'
    SHUFFLE = 'Shuffle'  # the standard outputs are shuffled in every epoch

    @staticmethod
    def set(idx):
        return [StandardOutputsBuildMethod.FIXED_UNSORTED, StandardOutputsBuildMethod.FIXED_SORTED_ALPHA,
                StandardOutputsBuildMethod.SHUFFLE, StandardOutputsBuildMethod.FIXED_SORTED_FREQ][idx]

class TrainMethod:
    NLL_METHOD = 'NLL'
    RL_METHOD = 'RL'

    @staticmethod
    def set(name):
        name = name.lower()
        return {'nll': TrainMethod.NLL_METHOD, 'rl': TrainMethod.RL_METHOD}[name]


class DecoderMethod:
    ONE_DECODER = 'ONE'
    MULTI_DECODER = 'MULTI'
    HIERARCHICAL_DECODER = 'HIERA'  # hierarchical
    SEPARATE_W = 'SEPARATE_W'  # one decoder, but using different w in every decoding step
    TMP = 'TMP'  # one decoder, but using different w in every decoding step

    @staticmethod
    def set(name):
        name = name.lower()
        return {'onedecoder': DecoderMethod.ONE_DECODER, 'multidecoder': DecoderMethod.MULTI_DECODER,
                'tmp': DecoderMethod.TMP, 'separatew': DecoderMethod.SEPARATE_W}[name]


configs = collections.namedtuple('configs', [
    'decoder_method',
    'train_method',
    'triple_number',
    'epoch_number',
    'save_freq',
    'decoder_num_units',
    'encoder_num_units',
    'cell_name',
    'learning_rate',
    'batch_size',
    'decoder_output_max_length',
    'dataset_name',
    'exp_name',
    'standard_outputs_build_method',
    'restore_epoch',
])

class Config:
    def __init__(self, configs):
        home = '/home/sunder/'
        self.decoder_method = configs.decoder_method
        self.train_method = configs.train_method
        self.triple_number = configs.triple_number
        self.epoch_number = configs.epoch_number
        self.save_freq = configs.save_freq
        self.decoder_num_units = configs.decoder_num_units
        self.encoder_num_units = configs.encoder_num_units
        self.cell_name = configs.cell_name
        self.learning_rate = configs.learning_rate
        self.batch_size = configs.batch_size
        self.decoder_output_max_length = configs.decoder_output_max_length
        self.dataset_name = configs.dataset_name
        self.exp_name = configs.exp_name
        self.standard_outputs_build_method = configs.standard_outputs_build_method
        self.restore_epoch = configs.restore_epoch

        DataSet.set_dataset(self.dataset_name)
        model_home = os.path.join(home, 'data/seq2seq_re', DataSet.name, self.exp_name)
        if self.train_method == TrainMethod.NLL_METHOD:
            runner = '%s-%s-%s-%s-%s-%s-%s-%s-%s-%s' % (self.dataset_name, self.decoder_method, self.train_method,
                                                        self.triple_number, self.learning_rate, self.batch_size,
                                                        self.standard_outputs_build_method, self.cell_name,
                                                        self.encoder_num_units, self.decoder_num_units)
        elif self.train_method == TrainMethod.RL_METHOD:
            runner = '%s-%s-%s-%s-%s-%s-%s-%s-%s-%s' % (self.dataset_name, self.decoder_method, self.train_method,
                                                        self.restore_epoch, self.triple_number, self.learning_rate,
                                                        self.batch_size, self.cell_name, self.encoder_num_units,
                                                        self.decoder_num_units)
        else:
            print "TrainMethod Illegal!!! {}".format(self.train_method)
            raise

        self.runner_path = os.path.join(model_home, runner)

        data_home = os.path.join(home, 'data', DataSet.name)
        if DataSet.name == DataSet.NYT:
            self.words_number = 90760
            self.embedding_dim = 100
            self.relation_number = 25
            self.max_sentence_length = 100
            self.origin_file_path = os.path.join(data_home, 'origin/')
            self.words2id_filename = os.path.join(data_home, 'words2id.json')
            self.relations2id_filename = os.path.join(data_home, 'relations2id.json')
            self.relations2count_filename = os.path.join(data_home, 'relation2count.json')
            self.words_id2vector_filename = os.path.join(data_home, 'words_id2vector.json')
            self.raw_train_filename = os.path.join(data_home, 'origin/raw_train.json')
            self.raw_test_filename = os.path.join(data_home, 'origin/raw_test.json')
            self.raw_valid_filename = os.path.join(data_home, 'origin/raw_valid.json')
            self.train_filename = os.path.join(data_home, 'seq2seq_re/train.json')
            self.test_filename = os.path.join(data_home, 'seq2seq_re/test.json')
            self.valid_filename = os.path.join(data_home, 'seq2seq_re/valid.json')
            self.summary_filename = os.path.join(self.runner_path, 'seq2seq_re_graph')
        if DataSet.name == DataSet.WEBNLG:
            self.words_number = 5928
            self.embedding_dim = 100
            self.relation_number = 247
            self.max_sentence_length = 80
            data_home = os.path.join(data_home, 'entity_end_position')
            self.words2id_filename = os.path.join(data_home, 'words2id.json')
            self.relations2id_filename = os.path.join(data_home, 'relations2id.json')
            self.relations2count_filename = os.path.join(data_home, 'relation2count.json')
            self.words_id2vector_filename = os.path.join(data_home, 'words_id2vector.json')
            self.train_filename = os.path.join(data_home, 'train.json')
            self.test_filename = os.path.join(data_home, 'dev.json')
            self.valid_filename = os.path.join(data_home, 'valid.json')
            self.summary_filename = os.path.join(self.runner_path, 'seq2seq_re_graph')

        self.NA_TRIPLE = (self.relation_number, self.max_sentence_length, self.max_sentence_length)


if __name__ == '__main__':
    pass
