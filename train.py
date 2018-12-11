# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
import sys 
import math
import random
import sys
import time

from six.moves import xrange

import utils
import seq2seq_model

import numpy as np
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

class Config(object):
    learning_rate = 0.5
    init_scale = 0.04
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    num_samples = 2048 # Sampled Softmax
    batch_size = 64
    size = 64 # Number of Node of each layer
    num_layers = 2
    vocab_size = 5000

config = Config

# use a number of buckets and pad to the closest one for efficiency.
# define (source sequence size, target sequence size) pair
buckets = [(120, 30), (200, 35), (300, 40), (400, 40), (500, 40)]

epochs = 10

tf.app.flags.DEFINE_float("learning_rate", config.learning_rate, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", config.learning_rate_decay_factor, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", config.max_gradient_norm, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("num_samples", config.num_samples, "Number of Samples for Sampled softmax")
tf.app.flags.DEFINE_integer("batch_size", config.batch_size, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", config.size, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", config.num_layers, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", config.vocab_size, "vocabulary size.")

# tf.app.flags.DEFINE_string("data_dir", data_path, "Data directory")
# tf.app.flags.DEFINE_string("train_dir", train_dir, "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.") # true for prediction
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

# define namespace for this model only
tf.app.flags.DEFINE_string("summarization_scope_name", "summarization_var_scope", "Variable scope of seq2seq summarization model")

FLAGS = tf.app.flags.FLAGS

def get_data(source_path, target_path, max_size=None):
    """
    从source_pah和target_path(每行文本表示一行，每个单词表示为词典中的ID，用空格分格)
    """
    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(source_path, mode='r') as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source_line, target_line = source_file.readline(), target_file.readline()   # read first line
            cnt = 0
            while source_line and target_line and (not max_size or counter < max_size):
                cnt += 1
                if cnt % 10 == 0:
                    print('read %d lines from source and target file' % cnt)
                source_ids = [int(word) for word in source_line.split()]
                target_ids = [int(word) for word in target_line.split()]
                # 加结束标志
                source_ids.append(utils.EOS_ID)
                target_ids.append(utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                # read next line
                source_line, target_line = source_file.readline(), target_file.readline()
    return data_set

def create_model(session, forward_only, train=False):
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.variable_scope(FLAGS.summarization_scope_name, reuse=None, initializer=initializer):
        model = seq2seq_model.Seq2SeqModel(
            FLAGS.vocab_size,   # source和target用的是同一个vocabulary
            FLAGS.vocab_size,
            buckets,
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm, # 防止梯度爆炸
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.learning_rate_decay_factor,
            use_lstm = True,    # LSTM instend of GRU
            num_samples = FLAGS.num_samples,
            forward_only=forward_only)
    if train:
        # run
        print('create a new model ...')
        session.run(tf.global_variables_initializer())
    else:
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt:
            model_ckpt_path = ckpt.model_checkpoint_path
            print('load model parameters from %s ...' % model_ckpt_path)
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint('./model'))
    return model

def train():
    '''
    preprocess data
    返回训练和测试数据的content和title的文件路径
    build：Boolean是否对数据全部重新处理
    '''
    # false表示不重新进行分词和index化，使用已存在的数据
    train_content_path, train_title_path, test_content_path, test_title_path = utils.preprocess_data(build=False)  

    print('read data into buckets')
    train_data_set = get_data(train_content_path, train_title_path, FLAGS.max_train_data_size)  # max_train_data_size默认为0，即使用全部训练数据
    test_data_set = get_data(test_content_path, test_title_path)

    # 每个bucket有多少个句子
    train_bucket_sizes = [len(train_data_set[b_index]) for b_index in xrange(len(buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    print('train buckets sizes:' + str(train_bucket_sizes))

    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    print('Start training process ...')
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(config=gpu_config) as sess:
        # create model
        print('create %d layers of %d units.' % (FLAGS.num_layers, FLAGS.size))
        model = create_model(session=sess, forward_only=False, train=True)

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        for epoch_i in range(epochs):
            for step_i in range(FLAGS.steps_per_checkpoint):
                # Choose a bucket according to data distribution
                random_number = np.random.random_sample()
                bucket_id = min([i for i in xrange(len(train_buckets_scale))
                           if train_buckets_scale[i] > random_number])

                # Get a batch and make a step.
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_data_set, bucket_id)

                # 在训练时，forward_only为Flase表示需要更新参数
                _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, False)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1
            # end step_i

            if epoch_i > 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.2f perplexity ""%.2f" % 
                    (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))

                # 降低学习率
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)

                previous_losses.append(loss)
                checkpoint_path = os.path.join('./model', "train.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0  # clear step_time and loss

                # run evaluation
                for bucket_id in xrange(len(buckets)):
                    if len(test_data_set[bucket_id]) == 0:
                        print("evaluation: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_data_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True) # True表示此轮不更新参数
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    print("epoch: %d  eval: bucket %d perplexity %.2f" % (epoch_i, bucket_id, eval_ppx))
                sys.stdout.flush()

def main(_):
    train()

if __name__ == "__main__":
  tf.app.run()