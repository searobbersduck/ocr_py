# !/usr/bin/env python2
# -*- coding=utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

import os
import time

from ocr_fullfonts_gen_try import import_vocab

# REAL_LABEL_NUM = 6516
# REAL_LABEL_NUM = 6826
REAL_LABEL_NUM = 7043

TIME_STEPS = 120

def parse_tfrecord_function(example_proto):
    features = {
        'label': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        'image': tf.FixedLenFeature([], tf.string),
        'nlabel': tf.FixedLenFeature([], tf.int64, default_value=0)
    }
    parsed_feats = tf.parse_single_example(example_proto, features)
    image = parsed_feats['image']
    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [21, 4*TIME_STEPS, 3])
    image = tf.squeeze(tf.image.rgb_to_grayscale(image), axis=2)
    datas = tf.split(image, TIME_STEPS, axis=1, num=TIME_STEPS)
    datas = tf.stack(datas, axis=0)
    datas = tf.reshape(datas, [TIME_STEPS, 84])
    label = parsed_feats['label']
    nlabel = parsed_feats['nlabel']
    return tf.cast(datas, tf.int32), label, nlabel

def parse_tfrecord_function_with_raw(example_proto):
    features = {
        'label': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        'image': tf.FixedLenFeature([], tf.string),
        'nlabel': tf.FixedLenFeature([], tf.int64, default_value=0)
    }
    parsed_feats = tf.parse_single_example(example_proto, features)
    image = parsed_feats['image']
    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [21, 4*TIME_STEPS, 3])
    image = tf.squeeze(tf.image.rgb_to_grayscale(image), axis=2)
    datas = tf.split(image, TIME_STEPS, axis=1, num=TIME_STEPS)
    datas = tf.stack(datas, axis=0)
    datas = tf.reshape(datas, [TIME_STEPS, 84])
    label = parsed_feats['label']
    nlabel = parsed_feats['nlabel']
    return tf.cast(datas, tf.int32), label, nlabel, image


class OCRModel(object):
    def __init__(self, numChars, lstmHidden):
        self.numChars = numChars
        self.lstmHidden = lstmHidden
        self.weights = tf.get_variable(
            'last_proj_weights',
            shape=[2*lstmHidden, numChars+1],
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(1e-4)
        )
        self.bias = tf.get_variable(
            'last_proj_bias',
            shape=[numChars+1]
        )
        self.lr = tf.placeholder(tf.float32, shape=None)

    def bilstm(self, X, xL, reuse=False):
        with tf.variable_scope('bilstm', reuse=reuse) as scope:
            fw_lstm = tf.contrib.rnn.LSTMCell(
                num_units=self.lstmHidden,
                state_is_tuple=True,
                reuse = reuse
            )
            bw_lstm = tf.contrib.rnn.LSTMCell(
                num_units=self.lstmHidden,
                state_is_tuple=True,
                reuse = reuse
            )
            if not reuse:
                fw_lstm = tf.contrib.rnn.DropoutWrapper(
                    fw_lstm, output_keep_prob=0.8
                )
                bw_lstm = tf.contrib.rnn.DropoutWrapper(
                    bw_lstm, output_keep_prob=0.8
                )
            outputs, (_,_) = tf.nn.bidirectional_dynamic_rnn(
                fw_lstm,
                bw_lstm,
                X,
                sequence_length=xL,
                dtype=tf.float32,
                time_major=False
            )
        R = tf.concat(outputs, 2)
        return R

    def length(self, X):
        shape = tf.shape(X)
        return tf.cast(tf.ones([shape[0]])*TIME_STEPS, tf.int32)

    def inference(self, X, xL, reuse=False):
        nx = tf.cast(X, tf.float32)/255.
        o = self.bilstm(nx, xL, reuse)
        o = tf.reshape(o, [-1, self.lstmHidden*2])
        o = tf.nn.xw_plus_b(o, weights=self.weights, biases=self.bias)
        o = tf.reshape(o, [-1, TIME_STEPS, self.numChars+1])
        return o

    def loss(self, X, Y, reuse=False):
        X = 255-X
        xL = self.length(X)
        o = self.inference(X, xL, reuse)
        idx = tf.where(tf.not_equal(Y, REAL_LABEL_NUM))
        sparse = tf.SparseTensor(
            idx,
            tf.gather_nd(tf.cast(Y, tf.int32), idx),
            tf.shape(Y, out_type=tf.int64)
        )
        reg_loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        ctc_loss = tf.reduce_mean(
            tf.nn.ctc_loss(sparse, o, xL, time_major=False)
        )
        return ctc_loss+reg_loss

    def train(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradients, vars = zip(*optimizer.compute_gradients(loss))
        gradients = [
            None if gradient is None else tf.clip_by_norm(gradient, 5.0) for gradient in gradients
        ]
        train_op = optimizer.apply_gradients(zip(gradients, vars))
        return train_op

    def testloss(self, X, Y):
        X = 255 -X
        X = tf.cast(X, tf.float32)/255.
        xL = self.length(X)
        o = self.inference(X, xL, True)
        # decoded, _ = tf.nn.ctc_beam_search_decoder(
        #     tf.transpose([1, 0, 2]), xL, beam_width=4, top_paths=3, merge_repeated=True
        # )
        decoded, _ = tf.nn.ctc_greedy_decoder(tf.transpose(o, [1, 0, 2]), xL, merge_repeated=True)
        idx = tf.where(tf.not_equal(Y, REAL_LABEL_NUM))
        sparse = tf.SparseTensor(
            idx,
            tf.gather_nd(Y, idx),
            tf.shape(Y, out_type=tf.int64)
        )
        dis = tf.edit_distance(decoded[0], sparse)
        dis = tf.reduce_mean(dis)
        ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(tf.cast(sparse, tf.int32), o, xL, time_major=False))
        return ctc_loss, 1-dis

    def onlineInference(self, X):
        X = 255 - X
        xL = self.length(X)
        o = self.inference(X, xL, True)
        decoded, _ = tf.nn.ctc_beam_search_decoder(
            tf.transpose(o, [1, 0, 2]), xL, beam_width=5, top_paths=1, merge_repeated=True
        )
        return tf.sparse_tensor_to_dense(decoded[0], name='onlineInferenceModel', default_value=REAL_LABEL_NUM)
        # return tf.reshape(X, [-1, 120, 84], name='onlineInferenceModel')

    def onlineInference_NoMerge(self, X):
        X = 255 - X
        xL = self.length(X)
        o = self.inference(X, xL, True)
        decoded, _ = tf.nn.ctc_beam_search_decoder(
            tf.transpose(o, [1, 0, 2]), xL, beam_width=5, top_paths=1, merge_repeated=False
        )
        return tf.sparse_tensor_to_dense(decoded[0], name='onlineInferenceModel_NoMerge', default_value=REAL_LABEL_NUM)


def test(sess, test_loss, test_acc):
    total_losses = 0
    total_acc = 0
    total_cnt = 0
    finished = False
    try:
        while True:
            loss,  acc = sess.run([test_loss, test_acc])
            total_losses += loss
            total_acc += acc
            total_cnt += 1
            print('test steps: {}\t\t loss: {:.4f}\t\t accuracy: {:.4f}'.format(
                total_cnt, total_losses/total_cnt, total_acc/total_cnt
            ))
    except:
        finished = True
    return total_losses/total_cnt, total_acc/total_cnt

def train(train_tfrecord, val_tfrecord, model, epochs, buffer_size, batch_size, lr, max_steps):
    ds_train = tf.contrib.data.TFRecordDataset(train_tfrecord, 'GZIP')
    ds_train = ds_train.map(parse_tfrecord_function)
    ds_train = ds_train.repeat(epochs)
    ds_train = ds_train.shuffle(buffer_size=buffer_size)
    ds_train = ds_train.batch(batch_size)
    itrator_train = ds_train.make_one_shot_iterator()
    inputs_train = itrator_train.get_next()
    ds_val = tf.contrib.data.TFRecordDataset(val_tfrecord, 'GZIP')
    ds_val = ds_val.map(parse_tfrecord_function)
    ds_val = ds_val.batch(batch_size)
    iterator_val = ds_val.make_initializable_iterator()
    inputs_val = iterator_val.get_next()

    # op
    train_loss = model.loss(inputs_train[0], inputs_train[1], False)
    train_op = model.train(train_loss)
    val_loss, val_acc = model.testloss(inputs_val[0], inputs_val[1])

    X = tf.placeholder(tf.int32, shape=[None, TIME_STEPS, 84], name='inp_x')
    online_reference = model.onlineInference(X)
    online_reference_nomerge = model.onlineInference_NoMerge(X)

    init = tf.global_variables_initializer()

    best_loss = float('inf')
    best_acc = 0
    saver = tf.train.Saver()
    outdir = './models'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    logger = []
    lr_in = lr
    min_lr = 5e-5
    with tf.Session() as sess:
        sess.run(init)
        tf.train.write_graph(sess.graph_def, outdir, 'graph.pb', as_text=True)
        a = time.time()
        b = time.time()
        loss_rec = float('inf')
        lr_acc_cnt = 0
        for i in range(1, max_steps):
            try:
                o_train_loss, _ = sess.run([train_loss, train_op], feed_dict={
                    model.lr: lr_in
                })
                lr_acc_cnt += 1
                if o_train_loss < loss_rec:
                    loss_rec = o_train_loss
                    lr_acc_cnt = 0
                if lr_acc_cnt > 2000:
                    lr_acc_cnt = 0
                    lr_in = (lr_in)/10 if (lr_in)/10 > min_lr else min_lr
                    log = '====> adjust learning rate to {:.6f}'.format(lr_in)
                    logger.append(log)
                    print(log)

                if i % 20 == 0:
                    b = time.time()
                    log = '====> Training steps: {}\t\t loss: {:.4f}\t\t time: {:.4f}s'.format(i, o_train_loss, (b-a))
                    logger.append(log)
                    a = time.time()
                    print(log)
                if i % 1000 == 0:
                    saver.save(sess, os.path.join(outdir, 'ocr'), global_step=i)
                    log = '====>Save model!\tCurrent training loss: {}'.format(o_train_loss)
                    logger.append(log)
                    print(log)
                    try:
                        sess.run(iterator_val.initializer)
                        o_val_loss, o_val_acc = test(sess, val_loss, val_acc)
                        if o_val_loss < best_loss:
                            best_loss = o_val_loss
                            best_acc = o_val_acc
                            saver.save(sess, os.path.join(outdir, 'ocr_best'), global_step=i)
                            log = '====> Current best validation loss: {:.4f}\taccuracy: {:.4f}'.format(best_loss, best_acc)
                            logger.append(log)
                            print(log)
                    except:
                        log = 'Exception when validation at step:{}'.format(i)
                        logger.append(log)
                        print(log)
                        continue
            except:
                saver.save(sess, os.path.join(outdir, 'ocr'), global_step=i)
                with open('./models/logs.txt', 'w') as f:
                    for log in logger:
                        f.write(log + '\n')
                break
        with open('./models/logs.txt', 'w') as f:
            for log in logger:
                f.write(log+'\n')


def train_by_epochs(train_tfrecord, val_tfrecord, model, epochs, buffer_size, batch_size, lr, max_steps):
    ds_train = tf.contrib.data.TFRecordDataset(train_tfrecord, 'GZIP')
    ds_train = ds_train.map(parse_tfrecord_function)
    # ds_train = ds_train.repeat(epochs)
    ds_train = ds_train.shuffle(buffer_size=buffer_size)
    ds_train = ds_train.batch(batch_size)
    iterator_train = ds_train.make_initializable_iterator()
    inputs_train = iterator_train.get_next()
    ds_val = tf.contrib.data.TFRecordDataset(val_tfrecord, 'GZIP')
    ds_val = ds_val.map(parse_tfrecord_function)
    ds_val = ds_val.batch(batch_size)
    iterator_val = ds_val.make_initializable_iterator()
    inputs_val = iterator_val.get_next()

    # op
    train_loss = model.loss(inputs_train[0], inputs_train[1], False)
    train_op = model.train(train_loss)
    val_loss, val_acc = model.testloss(inputs_val[0], inputs_val[1])

    X = tf.placeholder(tf.int32, shape=[None, TIME_STEPS, 84], name='inp_x')
    online_reference = model.onlineInference(X)

    init = tf.global_variables_initializer()

    best_loss = float('inf')
    best_acc = 0
    saver = tf.train.Saver()
    outdir = './models'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    with tf.Session() as sess:
        sess.run(init)
        tf.train.write_graph(sess.graph_def, outdir, 'graph.pb', as_text=True)
        for i in range(epochs):
            sess.run(iterator_train.initializer)
            sess.run(iterator_val.initializer)
            step_cnt = 0
            try:
                print('====> Epoch[{}] Training Begin:'.format(i))
                while True:
                    o_train_loss, _ = sess.run([train_loss, train_op], feed_dict={
                        model.lr: lr
                    })
                    step_cnt += 1
                    if (step_cnt) % 10 == 0:
                        print('\t\tEpoch[{}]:\t\t'
                              'Training steps: {}\t\t '
                              'Loss: {:.4f}\t\t'.format(i, step_cnt, o_train_loss))
            except tf.errors.OutOfRangeError as err:
                print('====> Epoch[{}] Training Finished!'.format(i))

            try:
                print('====> Epoch[{}] Validation Begin:'.format(i))
                o_val_loss, o_val_acc = test(sess, val_loss, val_acc)
                if step_cnt % 100 == 0:
                    print('\t\tEpoch[{}]:\t\t'
                          'Loss: {:.4f}\t\t'
                          'Accuracy: {:.4f}\t\t'.format(i, o_val_loss, o_val_acc))
                if o_val_loss < best_loss:
                    best_loss = o_val_loss
                    best_acc = o_val_acc
                    print('====> Epoch[{}]:\tSave OCR Model! Current best loss:{:.4f}! '
                          'Current best accuracy: {:.4f}'.format(i, best_loss, best_acc))
                    saver.save(sess, os.path.join(outdir, 'ocr'), global_step=i*100000+step_cnt)
            except tf.errors.OutOfRangeError as err:
                continue


def loadVocab(txt, vocab):
    idx = 0
    vocab[0] = 'x'
    with open(txt, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            ss = line.split('\t')
            idx += 1
            vocab[idx] = ss[0]
    print('Max char idx is: {}'.format(len(vocab)))


def retrain():
    models_dir = './models'
    # models_dir = './models-v1'

    ds_val = tf.contrib.data.TFRecordDataset('../tfdata/val.tfrecord', 'GZIP')
    # ds_val = tf.contrib.data.TFRecordDataset('./tfdata/val.tfrecord-0', 'GZIP')
    ds_val = ds_val.map(parse_tfrecord_function_with_raw)
    ds_val = ds_val.batch(1)
    iterator_val = ds_val.make_one_shot_iterator()
    inputs_val = iterator_val.get_next()

    # model_ocr = OCRModel(6826, 120)

    vocab = {}
    vocab1 = {}
    loadVocab('./out_ocr_fullfonts_gen/unicode_chars1.txt', vocab1)
    import_vocab('./out_ocr_fullfonts_gen/unicode_chars.txt', vocab)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(models_dir, 'ocr-21000.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(models_dir))
        graph = tf.get_default_graph()
        # op_onlinereference = graph.get_operation_by_name('onlineInferenceModel')
        op_onlinereference = graph.get_tensor_by_name('onlineInferenceModel_NoMerge:0')
        inp_x = graph.get_tensor_by_name('inp_x:0')

        log = []
        for i in range(1000):
            try:
                inputs = sess.run(inputs_val)
                image = inputs[3][0]
                import numpy as np
                from PIL import Image
                # import cv2
                # image = 255 - image
                # pil_img = Image.fromarray(image)
                # cv_img = np.array(pil_img, dtype=np.uint8)
                # cv2.imshow('test', cv_img)
                # cv2.waitKey(3000)
                o = sess.run(op_onlinereference, feed_dict={
                    inp_x: inputs[0]
                })

                o1 = o[0]
                # o1 = inputs[1][0]
                str = ''
                for i in range(len(o1)):
                    if o1[i] == REAL_LABEL_NUM:
                        continue
                    str += vocab[o1[i]]
                print(str)
                log.append(str)
                # o1 = o[0]
                o1 = inputs[1][0]
                str = ''
                for i in range(len(o1)):
                    if o1[i] >= REAL_LABEL_NUM:
                        continue
                    str += vocab1[o1[i]]
                print(str)
                log.append(str)
            except IOError as e:
                print(e)
                break
        with open('comp_str.txt', 'w') as f:
            for l in log:
                f.write(l)
                f.write('\n')
        print('hello retrain!')


def test_train():
    model = OCRModel(numChars=6516, lstmHidden=120)
    train_by_epochs('../dataset/tfdata-small/test.tfrecord',
          '../dataset/tfdata-small/test.tfrecord',
          model, 100, 2048, 128, 0.001, 10000
          )

def test_train_noepochs():
    trainset = ['./tfdata/train.tfrecord-{}'.format(i) for i in range(12)]
    valset = ['./tfdata/val.tfrecord-{}'.format(i) for i in range(12)]
    # trainset = './tfdata/train.tfrecord-0'
    # valset = './tfdata/val.tfrecord-0'
    model = OCRModel(numChars=REAL_LABEL_NUM, lstmHidden=120)
    train(trainset,
          valset,model, 100, 2048, 384, 0.01, 1000000)

if __name__ == '__main__':
    # test_train()
    # retrain()
    test_train_noepochs()