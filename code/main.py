#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2017/8/21
import json
import logging
import logging.config
import os

import numpy as np
import tensorflow as tf

import argument_parser
import const
import data_prepare
import evaluation
import model
from const import TrainMethod

logger = logging.getLogger('mylogger')

# 调用配置
config, train_test_valid, gpu, rl_init_path, evaluation_epoch = argument_parser.get_config()
is_train = train_test_valid == 'train'
is_restore = config.restore_epoch >= 0

def setup_logging(default_path='/home/sunder/code/seq2seq_re/logging.json',
                  default_level=logging.DEBUG,
                  env_key='LOG_CFG', ):
    """
    Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            log_config = json.load(f)
            handlers = log_config['handlers']
            log_folder = os.path.join(config.runner_path, 'logfile')
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            handlers['debug_file_handler']['filename'] = os.path.join(log_folder, 'debug.log')
            handlers['info_file_handler']['filename'] = os.path.join(log_folder, 'info.log')
            handlers['error_file_handler']['filename'] = os.path.join(log_folder, 'error.log')
            log_config['handlers'] = handlers
        logging.config.dictConfig(log_config)
    else:
        logging.basicConfig(level=default_level)


setup_logging()

if config.train_method == TrainMethod.NLL_METHOD:
    logger.info('Decoder_method: %s-%s, %s, %s, standard_outputs_build_method: %s, '
                'triple_number: %s, learn_rate %s, batch_size: %s, epoch_num: %s, gpu: %s'
                % (config.decoder_method, config.train_method, config.cell_name, train_test_valid,
                   config.standard_outputs_build_method, config.triple_number, config.learning_rate, config.batch_size,
                   config.epoch_number, gpu if gpu else None))
elif config.train_method == TrainMethod.RL_METHOD:
    # config.standard_outputs_build_method = standard_outputs_build_method
    logger.info('Decoder_method: %s-%s, %s, %s, RL init epoch: %s, '
                'triple_number: %s, learn_rate %s, batch_size: %s, epoch_num: %s, gpu: %s'
                % (config.decoder_method, config.train_method, config.cell_name, train_test_valid,
                   config.restore_epoch, config.triple_number, config.learning_rate, config.batch_size,
                   config.epoch_number, gpu if gpu else None))
else:
    print "TrainMethod Illegal!!! {}".format(config.train_method)
    raise

logger.info('runner: %s' % config.runner_path)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True

#   set the batch size in test. Because the test data size maybe smaller then batch size
if not is_train:
    if config.dataset_name == const.DataSet.NYT:
        config.batch_size = 1000
    if config.dataset_name == const.DataSet.CONLL04 or config.dataset_name == const.DataSet.WEBNLG:
        config.batch_size = 2

def test_model(data, decoder, sess, show_rate, is_visualize, simple=True):
    sents_id = []
    predictes = []
    gold = []
    for batch_i in range(data.batch_number):
        batch_data = data.next_batch(is_random=False)
        predict_answer = decoder.predict(batch_data, sess)
        gold_answer = batch_data.all_triples
        predictes.extend(predict_answer)
        gold.extend(gold_answer)
        sents_id.extend(batch_data.sentence_fw)
    try:
        assert len(predictes) == len(gold)
    except AssertionError:
        logger.info('Error, predictes number (%d) not equal gold number (%d)' % (len(predictes), len(gold)))
        exit()
    f1, precision, recall = evaluation.compare(predictes, gold, config, show_rate, simple=simple)

    if not simple:
        evaluation.error_analyse(predictes, gold, config, entity_or_relation='entity')
        evaluation.error_analyse(predictes, gold, config, entity_or_relation='relation')

    if is_visualize:
        visualize_normal_file = os.path.join(config.runner_path, 'visualize_normal_instance.txt')
        visualize_multi_file = os.path.join(config.runner_path, 'visualize_multi_instance.txt')
        visualize_overlap_file = os.path.join(config.runner_path, 'visualize_overlap_instance.txt')
        print visualize_normal_file
        print visualize_multi_file
        print visualize_overlap_file
        evaluation.visualize(sents_id, gold, predictes,
                             [visualize_normal_file, visualize_multi_file, visualize_overlap_file], config)
    return f1, precision, recall


def test_all_models(model_epochs, data, decoder, sess, config, del_model=True):
    if train_test_valid.lower() == 'test':
        filename = os.path.join(config.runner_path, 'test_result.txt')
    elif train_test_valid.lower() == 'valid':
        filename = os.path.join(config.runner_path, 'valid_result.txt')
    else:
        logger.error('Error, illegal instruction: {}'.format(train_test_valid))
        raise
    out_file = open(filename, 'a')
    saver = tf.train.Saver()

    best_f1 = 0.
    best_epoch = 0
    best_model_filename = None
    model_filenames = []
    for epoch in model_epochs:
        model_name = 'model-{}'.format(epoch)
        model_filename = os.path.join(config.runner_path, model_name)
        if not os.path.isfile(model_filename + '.index'):
            continue
        model_filenames.append(model_filename)
        logger.info('Test model: {}'.format(model_name))
        saver.restore(sess, model_filename)
        data.reset()
        f1, precision, recall = test_model(data, decoder, sess, show_rate=None, is_visualize=False)
        out_file.write('%d,%.3f,%.3f,%.3f' % (epoch, precision, recall, f1))
        out_file.write('\n')
        out_file.flush()
        if f1 > best_f1:
            best_epoch = epoch
            best_model_filename = model_filename
            best_f1 = f1
    out_file.close()

    #  remove the model file to save disk space
    logger.info('Best epoch >>>>>>>> %d' % best_epoch)
    if train_test_valid.lower() == 'valid' and del_model:
        for model_filename in model_filenames:
            if model_filename != best_model_filename:
                for extension in ['.meta', '.index', '.data-00000-of-00001']:
                    os.remove(model_filename + extension)
                logger.info('Remove %s' % model_filename)

def train_RL_model(data, epoch_range, decoder, sess):
    saver = tf.train.Saver(max_to_keep=0)
    best_f1 = 0.
    step = 0
    for epoch_i in epoch_range:
        adv = []
        loss = []
        num = []
        if epoch_i == 1:
            denominator = 10
        elif epoch_i < 5:
            denominator = 1
        else:
            denominator = 0.25
        step += 1
        for batch_i in range(int(data.batch_number / denominator)):
            batch_data = data.next_batch(is_random=True,
                                         standard_outputs_build_method=config.standard_outputs_build_method)
            actions = decoder.predict(batch_data, sess)
            advantages, numbers = evaluation.calc_advantage2(actions, batch_data.standard_outputs, config)
            # print batch_data.standard_outputs_mask
            # advantages = np.expand_dims(advantages, -1)
            adv.extend(advantages)
            num.extend(numbers)

            loss_val = decoder.update(batch_data, advantages, sess, sampled_action=actions)
            loss.append(loss_val)
            logger.debug('RL Train: epoch %-3d,\tbatch %-3d,\tadvantages %f\ttriple_n %f' %
                         (epoch_i, batch_i, np.mean(advantages), np.mean(numbers)))

        logger.info('RL Train: epoch %d, Mean reward %s, Mean triple_n %s, Mean loss %s' % (epoch_i, np.mean(adv),
                                                                                            np.mean(num),
                                                                                            np.mean(loss)))
        if config.dataset_name == const.DataSet.NYT:
            remainder = 0
        if config.dataset_name == const.DataSet.WEBNLG:
            remainder = 0

        if epoch_i % config.save_freq == remainder:
            save_path = os.path.join(config.runner_path, 'model')
            saver.save(sess, save_path=save_path, global_step=step)
            logger.info('Saved model {0}-{1}'.format(save_path, step))


def train_NLL_model(data, epoch_range, decoder, sess):
    saver = tf.train.Saver(max_to_keep=60)
    for epoch_i in epoch_range:
        epoch_loss = []
        for batch_i in range(data.batch_number):
            batch_data = data.next_batch(is_random=True,
                                         standard_outputs_build_method=config.standard_outputs_build_method)
            advantages = np.zeros([config.batch_size, config.decoder_output_max_length], dtype=np.float32)
            loss_val = decoder.update(batch_data, advantages, sess)
            epoch_loss.append(loss_val)
            if epoch_i == 1 and batch_i == 5:
                saver.save(sess, save_path=os.path.join(config.runner_path, 'model'), global_step=0)
                logger.info('NLL Train: epoch %-3d, loss %f' % (0, np.mean(epoch_loss)))
            # logger.info('NLL Train: epoch %-3d batch %-3d, loss %f' % (epoch_i, batch_i, loss_val))
        logger.info('NLL Train: epoch %-3d, loss %f' % (epoch_i, np.mean(epoch_loss)))

        if config.dataset_name == const.DataSet.NYT:
            remainder = 0
        if config.dataset_name == const.DataSet.WEBNLG:
            remainder = 0
        if epoch_i % config.save_freq == remainder:
            save_path = os.path.join(config.runner_path, 'model')
            saver.save(sess, save_path=save_path, global_step=epoch_i)
            logger.info('Saved model {0}-{1}'.format(save_path, epoch_i))


def get_model(train_method, config):
    logger.info('Building model --------------------------------------')
    if is_train and is_restore:
        if config.train_method == TrainMethod.RL_METHOD:
            model_filename = build_rl_init_model_filename()
        else:
            model_name = 'model-{}'.format(config.restore_epoch)
            model_filename = os.path.join(config.runner_path, model_name)
        logger.info('Parameter init from: %s' % model_filename)
    else:
        logger.info('Parameter init Randomly')
    embedding_table = model.get_embedding_table(config)
    encoder = model.Encoder(config=config, max_sentence_length=config.max_sentence_length,
                            embedding_table=embedding_table)
    encoder.set_cell(name=config.cell_name, num_units=config.encoder_num_units)
    encoder.build()

    relation_decoder = model.RelationDecoder(encoder=encoder, config=config, is_train=is_train)

    position_decoder = model.PositionDecoder(encoder=encoder, config=config, is_train=is_train)

    decode_cell = []
    for t in range(config.triple_number if config.decoder_method == const.DecoderMethod.MULTI_DECODER else 1):
        cell = model.set_rnn_cell(config.cell_name, config.decoder_num_units)
        decode_cell.append(cell)

    triple_decoder = model.TripleDecoder(decoder_output_max_length=config.decoder_output_max_length, encoder=encoder,
                                         relation_decoder=relation_decoder, position_decoder=position_decoder,
                                         decode_cell=decode_cell, config=config)
    triple_decoder.build(train_method=train_method, decoder_method=config.decoder_method, is_train=is_train)

    sess = tf.Session(config=tfconfig)
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

    saver = tf.train.Saver()
    if is_train and is_restore:
        saver.restore(sess, model_filename)
    else:
        sess.run(tf.global_variables_initializer())
    logger.debug('print trainable variables')
    for v in tf.trainable_variables():
        value = sess.run(v)
        logger.info(v.name)
        logger.debug('mean %.4f, max %.3f, min %.3f' % (np.mean(value), np.max(value), np.min(value)))

    return triple_decoder, sess


def build_rl_init_model_filename():
    segs = config.runner_path.split('/')
    segs[-1] = rl_init_path
    runner_path = '/'.join(segs)
    model_name = 'model-{}'.format(config.restore_epoch)
    model_filename = os.path.join(runner_path, model_name)
    return model_filename



if __name__ == '__main__':
    if config.dataset_name == const.DataSet.NYT:
        prepare = data_prepare.NYTPrepare(config)
    elif config.dataset_name == const.DataSet.CONLL04:
        prepare = data_prepare.Conll04Prepare(config)
    elif config.dataset_name == const.DataSet.WEBNLG:
        prepare = data_prepare.WebNLGPrepare(config)
    else:
        print 'illegal dataset name: %s' % config.dataset_name
        exit()

    if is_train:
        if config.train_method == TrainMethod.NLL_METHOD:
            min_number = 1
        if config.train_method == TrainMethod.RL_METHOD:
            min_number = 1
    else:
        min_number = 1  # this is the min number of triples in a sentence

    logger.info('Prepare {} data'.format(train_test_valid))
    data = prepare.load_data(train_test_valid.lower())
    data = prepare.process(data, min_number)
    data = data_prepare.Data(data, config.batch_size, config)

    decoder, sess = get_model(train_method=config.train_method, config=config)
    # decoder, sess = None, None
    if is_train:
        if config.train_method == TrainMethod.NLL_METHOD:
            logger.info('****************************** NLL Train ******************************')
            train_NLL_model(data, epoch_range=range(1, config.epoch_number + 1), decoder=decoder,
                            sess=sess)
        if config.train_method == TrainMethod.RL_METHOD:
            logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RL Train %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            train_RL_model(data, epoch_range=range(1, config.epoch_number + 1), decoder=decoder,
                           sess=sess)

    else:
        logger.info(
            '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ {} Dataset $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'.format(train_test_valid))
        if evaluation_epoch >= 0:
            epoch = evaluation_epoch
            model_name = 'model-{}'.format(epoch)
            model_filename = os.path.join(config.runner_path, model_name)
            logger.info('Test model: {}'.format(model_name))
            saver = tf.train.Saver()
            saver.restore(sess, model_filename)
            test_model(data, decoder=decoder, sess=sess, show_rate=None, is_visualize=True, simple=False)
        else:
            model_epochs = range(config.save_freq, config.epoch_number + 1, config.save_freq)
            test_all_models(model_epochs, data, decoder, sess, config)
