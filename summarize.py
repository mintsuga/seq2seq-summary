# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import codecs

import numpy as np
from six.moves import xrange
import tensorflow as tf

import utils
import seq2seq_model

from train import Config
from train import create_model
from train import buckets

config = Config()
buckets = buckets
FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def get_summary(input_path, reference_path, summary_path):
    input_file = open(input_path, 'r')
    reference_file = open(reference_path, 'r')
    out_file = open(summary_path, 'w')

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(config=gpu_config) as sess:
        model = create_model(session=sess, forward_only=True, train=False)
        _, index2word_vocab = utils.init_vocab('./data/vocab.txt')
        for content_ids, reference_ids in zip(input_file, reference_file):
            content = [int(index) for index in content_ids.split()]
            reference = [int(index) for index in reference_ids.split()]

            bucket_ids = [b for b in xrange(len(buckets)) if buckets[b][0] > len(content)]
            if len(bucket_ids) == 0:
                print('sentence length %d exceed max length in buckets' % len(content))
                continue
            bucket_id = min(bucket_ids)
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(content, [])]}, bucket_id)
            
            # output logits for the sentence
            _, _, output_logits_batch = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
            output_logits = []
            for item in output_logits_batch:
                output_logits.append(item[0])
            output_index = [int(np.argmax(logit)) for logit in output_logits]

            # 如果存在EOS_ID，进行截断
            if utils.EOS_ID in output_index:
                output_index = output_index[:output_index.index(utils.EOS_ID)]

            summary = [tf.compat.as_str(index2word_vocab[index]) for index in output_index]
            out_file.write(''.join(summary) + '\n')
            print(' '.join(summary))

            # Evaluate ROUGE-N score

def main(_):
    print('Start running evaluation ... ')
    get_summary('./data/test/content_ids.txt', './data/test/title_ids.txt', './data/test/summary.txt')

if __name__ == '__main__':
    tf.app.run()



