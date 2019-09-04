#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by sunder on 2018/12/20
import argparse
import logging

import const

logger = logging.getLogger('mylogger')

parser = argparse.ArgumentParser()
# parser.add_argument('-c', dest='configfile', type=str, help='The config file name')
parser.add_argument('-a', dest='action', type=str, choices=['train', 'test', 'valid'],
                    help='The action is train, test or valid')
parser.add_argument('-d', dest='dataset', type=str, choices=['nyt', 'webnlg'], help='The dataset name')
parser.add_argument('-l', dest='loss_method', type=str, choices=['nll', 'rl'], help='The way to calculate loss')
parser.add_argument('-m', dest='model', type=str, choices=['onedecoder', 'multidecoder', 'separatew', 'tmp'],
                    help='The used model')
parser.add_argument('-b', dest='batch_size', type=int, help='The batch size')
parser.add_argument('-tn', dest='triple_number', type=int, choices=[1, 2, 3, 4, 5],
                    help='The maximum number of triples in a sentence')
parser.add_argument('-lr', dest='learning_rate', type=float, help='Learning rate')
parser.add_argument('-en', dest='epoch_number', type=int, help='Epoch number')
parser.add_argument('-sf', dest='save_freq', type=int, help='Model save frequency')
parser.add_argument('-hn', dest='hidden_size', type=int, help='Hidden size of encoder and decoder(s)')
parser.add_argument('-n', dest='exp_name', type=str, help='Name of this experiment')
parser.add_argument('-g', dest='gpu', type=str, default='', help='gpu id')
parser.add_argument('-cell', dest='cell_name', type=str, help='cell name: lstm or gru')
parser.add_argument('-re', dest='restore_epoch', type=int, default=-1,
                    help='Model epoch number that used to init current model')
parser.add_argument('-eve', dest='evaluation_epoch', type=int, default=-1,
                    help='Model epoch number that will be evaluated')
group = parser.add_mutually_exclusive_group()
group.add_argument('-sobm', dest='standard_outputs_build_method',
                   type=int, help='0 for fix during training; '
                                  '1 for fix during training but sorted in order; '
                                  '2 for shuffle in every epoch during training')
group.add_argument('-rip', dest='rl_init_path', type=str, help='parameters of rl training init path')

args = parser.parse_args()
is_train = args.action == 'train'
train_test_valid = args.action
dataset = args.dataset
loss_method = const.TrainMethod.set(args.loss_method)
model = const.DecoderMethod.set(args.model)
batch_size = args.batch_size
triple_number = args.triple_number
learning_rate = args.learning_rate
epoch_number = args.epoch_number
save_freq = args.save_freq
hidden_size = args.hidden_size
exp_name = args.exp_name
gpu = args.gpu
cell_name = args.cell_name
restore_epoch = args.restore_epoch
evaluation_epoch = args.evaluation_epoch
is_restore = restore_epoch >= 0
if loss_method == const.TrainMethod.RL_METHOD:
    rl_init_path = args.rl_init_path
    standard_outputs_build_method = const.StandardOutputsBuildMethod.set(0)
else:
    standard_outputs_build_method = const.StandardOutputsBuildMethod.set(args.standard_outputs_build_method)
    rl_init_path = 'None'


def get_config():
    configs = const.configs(
        decoder_method=model,
        train_method=loss_method,
        triple_number=triple_number,
        epoch_number=epoch_number,
        save_freq=save_freq,
        decoder_num_units=hidden_size,
        encoder_num_units=hidden_size,
        cell_name=cell_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        decoder_output_max_length=triple_number * 3,
        dataset_name=dataset,
        exp_name=exp_name,
        standard_outputs_build_method=standard_outputs_build_method,
        restore_epoch=restore_epoch,
    )
    config = const.Config(configs)
    return config, train_test_valid, gpu, rl_init_path, evaluation_epoch


if __name__ == '__main__':
    pass
