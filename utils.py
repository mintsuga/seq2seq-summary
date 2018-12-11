# -*- coding:utf-8 -*-

import os
import sys
import re
import jieba

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

MAX_INT = sys.maxsize

def seg_file(file_path, seg_title_path, seg_content_path, stop_words_path, cnt_limit=MAX_INT):
    stop_words_file = open(stop_words_path, 'r')
    # 构建停用词列表
    stop_words = []
    for line in stop_words_file.readlines():
        stop_words.append(line.strip().decode('utf-8'))

    # 分词结果文件
    title_seg_file = open(seg_title_path, 'w')
    content_seg_file = open(seg_content_path, 'w')

    cnt = 0
    for line in open(file_path, 'r').readlines():
        cnt += 1
        # 得到label和正文
        label, content = line.strip().split('\t')
        # 对title预处理、分词，将分词结果写回文件
        title = content.split()[0]
        title = title.decode('utf-8')
        title = re.sub("[\\s+\\.\\!\\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#>￥%……&*（）]+", "",title)
        words = jieba.cut(title)
        stopped_seg_list = [word for word in words if not word in stop_words]
        title_seg_file.write(' '.join(stopped_seg_list).encode('utf-8') + '\n')
        # 对content预处理、分词
        content = ' '.join(content.split()[1:])
        content = content.decode('utf-8')
        content = re.sub("[\\s+\\.\\!\\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#>￥%……&*（）]+", "", content)
        words = jieba.cut(content)
        stopped_seg_list = [word for word in words if not word in stop_words]
        content_seg_file.write(' '.join(stopped_seg_list).encode('utf-8') + '\n')
        print('read %d lines' % cnt)
        if cnt >= cnt_limit:
            print('finish reading %d lines ...' % cnt)
            break

def create_vocabulary(vocab_path, seg_file_pathes, vocab_max_size):
    """
    从seg_file_pathes中读取各分词文件路径，构建词典后将词典写入vocab_path
    """
    print('start building vocabulary')
    vocab = set()
    cnt = 0
    for seg_file_path in seg_file_pathes:
        print('read from seg file %s' % seg_file_path)
        for line in open(seg_file_path, 'r').readlines():
            cnt += 1
            words = line.strip().split()
            for w in words:
                vocab.add(w)
            if cnt % 1000 == 0:
                print('build vocab with %d lines' % cnt)
    vocab_list = _START_VOCAB + list(vocab)
    if len(vocab_list) > vocab_max_size:
        vocab_list = vocab_list[:vocab_max_size]
    with open(vocab_path, 'w') as vocab_res:
        for w in vocab_list:
            vocab_res.write(w + '\n')
    print('finish building vocabulary with %d words' % len(vocab_list))

def init_vocab(vocab_path):
    """
    从vocab_path构建字典，返回一对元组，第一个是word2index词典，第二个就是词典列表
    """
    vocab_file = open(vocab_path, 'r')
    vocab = []
    for line in vocab_file.readlines():
        vocab.append(line.strip())
    vocab_res = dict([(word, index) for (index, word) in enumerate(vocab)])
    return vocab_res, vocab

def sentence2index(word_list, word2index_vocab):
    return [str(word2index_vocab.get(word, UNK_ID)) for word in word_list]

def create_doc2index(word2index_vocab, seg_file_path, id_file_path):
    contents = open(seg_file_path, 'r').readlines()
    content_id_file = open(id_file_path, 'w')
    for line in contents:
        line = line.strip().split()
        ids = sentence2index(line, word2index_vocab)
        content_id_file.write(' '.join(ids) + '\n')

def preprocess_data(build=True):
    # 停用词路径
    stop_words_path = './data/stop_words.txt'

    # train files
    train_file_path = './data/train/cnews.train.txt'
    train_title_seg = './data/train/title.txt'
    train_content_seg = './data/train/content.txt'

    # test files
    test_file_path = './data/test/cnews.test.txt'
    test_title_seg = './data/test/title.txt'
    test_content_seg = './data/test/content.txt'

    # 为数据的content和title构建word2index文件
    train_title_ids = './data/train/title_ids.txt'
    train_content_ids = './data/train/content_ids.txt'
    test_title_ids = './data/test/title_ids.txt'
    test_content_ids = './data/test/content_ids.txt'

    if build: 
        # 分词
        # cnt_limit是取前n行进行处理，测试时可以设比较小的值
        seg_file(train_file_path, train_title_seg, train_content_seg, stop_words_path, cnt_limit=100)  
        seg_file(test_file_path, test_title_seg, test_content_seg, stop_words_path, cnt_limit=10)

        # 构造词典
        create_vocabulary('./data/vocab.txt', [train_title_seg, train_content_seg, test_title_seg, test_content_seg], vocab_max_size=5000)
        word2index_vocab, _ = init_vocab('./data/vocab.txt')

        # 构造index文件
        create_doc2index(word2index_vocab, train_title_seg, train_title_ids)
        create_doc2index(word2index_vocab, train_content_seg, train_content_ids)

        create_doc2index(word2index_vocab, test_title_seg, test_title_ids)
        create_doc2index(word2index_vocab, test_content_seg, test_content_ids)

    return (train_content_ids, train_title_ids, test_content_ids, test_title_ids)

