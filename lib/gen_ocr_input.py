# !/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import fire
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import json

MAX_CHARS_PER_BOX = 30

def loadFont(fontDir, size):
    flist = os.listdir(fontDir)
    fonts = []
    fonts_eng = []
    for f in flist:
        if f.endswith('.ttf') or f.endswith('.TTF'):
            ff = os.path.join(fontDir, f)
            font = ImageFont.truetype(ff, size)
            print('load font {} with size {}.'.format(
                os.path.basename(f), size
            ))
            if (f == 'Verdana.ttf') or (f == 'FZXJHJW.ttf'):
                fonts_eng.append(font)
            else:
                fonts.append(font)
    return fonts, fonts_eng

def loadVocab(txt, vocab):
    idx = 0
    with open(txt, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            ss = line.split('\t')
            idx += 1
            vocab[ss[0].decode('utf8')[0]] = idx
    print('Max char idx is: {}'.format(len(vocab)))

def _int64_features(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def generateImageBytes(font, size, ustr):
    image = Image.new('RGB', (480, 21), 'white')
    w,h = font.getsize(ustr)
    ratio = 19/h
    offset = (size-h)/2
    fontImg = Image.new('RGB', (w,h), 'white')
    drawObj = ImageDraw.Draw(fontImg)
    drawObj.text([0,0], ustr, font=font, fill=(0,0,0,0))
    s_w  = int(w*ratio)
    if s_w < 478:
        w = s_w
    elif s_w < 600:
        w = 478
    else:
        return None
    if ratio < 1.0:
        offset = int(offset*ratio)
    img = fontImg.resize((w, 19), Image.LINEAR).point(
        lambda p: p>200 and 255
    )
    # cv_img = np.array(img, dtype=np.uint8)
    # import cv2
    # cv2.imshow('img', cv_img)
    # cv2.waitKey(1000)
    image.paste(img, (2, int(1+offset)))
    return image.tobytes()

def generateImages(sizedFonts, sizes, vocab, writer, ustr, nlength, sizedFonts_eng, gotE):
    sizeidx = np.random.randint(0, len(sizes)-1)
    fonts = sizedFonts[sizeidx]
    fonts_eng = sizedFonts_eng[sizeidx]
    size = sizes[sizeidx]
    font = np.random.choice(fonts)
    font_eng = np.random.choice(fonts_eng)
    if gotE:
        font = font_eng
    imageBytes = None
    # try:
    #     imageBytes = generateImageBytes(font, size, ustr)
    #     if imageBytes is None:
    #         return False
    # except:
    #     print('Generate exception with font {} & size {}'.format(
    #         font.getname(), font.size()
    #     ) + '\t' + ustr)
    #     print('Generate exception with font {} & size {}')
    #     return False
    try:
        imageBytes = generateImageBytes(font, size, ustr)
        if imageBytes is None:
            return False
    except Exception as e:
        print(e)
        return False
    label = [len(vocab)+1]*MAX_CHARS_PER_BOX
    valid = False
    for i in range(nlength):
        if ustr[i] == u' ':
            continue
        valid = True
        label[i] = vocab[ustr[i]] if ustr[i] in vocab else 0
    if not valid:
        return False
    example = tf.train.Example(features=tf.train.Features(
        feature = {
            'label': _int64_features(label),
            'image': _bytes_feature(imageBytes),
            'nlabel': _int64_feature(nlength)
        }
    ))
    writer.write(example.SerializeToString())
    return True

def split_line(content):
    n = len(content)
    if n <= MAX_CHARS_PER_BOX:
        yield content
        return
    nn = (n-1)//MAX_CHARS_PER_BOX+1
    for i in range(nn):
        start = i*MAX_CHARS_PER_BOX
        end = (i+1)*MAX_CHARS_PER_BOX
        if end >= n:
            end = n
        yield content[start:end]

def check_eng(line):
    eng = u''
    for c in line:
        if ord(c) < 128:
            eng += c
        if len(eng) > 4:
            return eng
        return None

def doGen(fontDir, vocab_txt, outdir, input_prefix, start=0, end=1000):
    sizedFonts = []
    sizedFonts_eng = []
    sizes = [15, 20, 25, 30, 40, 50]
    for size in sizes:
        fonts,fonts_eng = loadFont(fontDir, size)
        sizedFonts.append(fonts)
        sizedFonts_eng.append(fonts_eng)
    vocab = {}
    loadVocab(vocab_txt, vocab)
    train_cnt = 0
    val_cnt = 0
    test_cnt = 0
    total_cnt = 0
    tfopts = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP
    )
    writerTrain = tf.python_io.TFRecordWriter(
        '{}/train.tfrecord'.format(outdir), tfopts
    )
    writerTest = tf.python_io.TFRecordWriter(
        '{}/test.tfrecord'.format(outdir), tfopts
    )
    writerVal = tf.python_io.TFRecordWriter(
        '{}/val.tfrecord'.format(outdir), tfopts
    )
    for i in range(start, end):
        input_file = '{}-{:05d}'.format(input_prefix, i)
        line_cnt = 0
        if not os.path.isfile(input_file):
            continue
        with open(input_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                line_cnt += 1
                if line_cnt%10 == 0:
                    print('line number: {}\t\tfile: {}'.format(line_cnt, input_file))
                pos = line.find('\t')
                if pos == -1:
                    continue
                remind = line[pos+1:]
                cjson = json.loads(remind)
                workexps = cjson['workExperiences']
                contents = []
                for workexp in workexps:
                    if 'description' not in workexp:
                        continue
                    contents.append(workexp['description'])
                for content in contents:
                    ss = content.split(u'\n')
                    for s in ss:
                        vec = split_line(s)
                        for ustr in vec:
                            gotE = False
                            if np.random.random() < 0.2:
                                eng = check_eng(ustr)
                                if eng is not None:
                                    ustr = eng
                                    gotE = True
                            nc = len(ustr)
                            if (nc > 5 or gotE) and nc <= MAX_CHARS_PER_BOX:
                                writer = writerTrain
                                rand_x = np.random.random()
                                if rand_x < 0.8:
                                    writer = writerTrain
                                elif rand_x < 0.81:
                                    writer = writerVal
                                else:
                                    writer = writerTest
                                if nc <= 1:
                                    continue
                                try:
                                    ustr.encode('gb2312')
                                except:
                                    # print(u'错误字符：{}'.format(ustr))
                                    # print('Error string can not transpose to simply chinese!')
                                    continue
                                if generateImages(sizedFonts, sizes, vocab, writer, ustr, nc, sizedFonts_eng, gotE):
                                    if writer == writerTrain:
                                        train_cnt += 1
                                    elif writer == writerVal:
                                        val_cnt += 1
                                    else:
                                        test_cnt += 1
                                total_cnt += 1
                                if total_cnt % 1000 == 0:
                                    print('====> create images: {}\t'
                                          'train images: {}\t'
                                          'validatinon images: {}\t'
                                          'test images: {}\t'.format(total_cnt, train_cnt, val_cnt, test_cnt))
    writerTrain.close()
    writerVal.close()
    writerTest.close()


def test_doGen():
    fontDir = '../dataset/useFont'
    vocab_txt = '../dataset/unicode_chars.txt'
    outdir = '../dataset/tfdata-small'
    input_prefix = '../dataset/resume_doc/part-m'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    print('====> begin doGen test: ')
    doGen(fontDir, vocab_txt, outdir, input_prefix, 1, 2)
    print('====> end doGen test.')

def parse_tfrecord_function(example_proto):
    features = {
        'label': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        'image': tf.FixedLenFeature([], tf.string),
        'nlabel': tf.FixedLenFeature([], tf.int64, default_value=0)
    }
    parsed_feats = tf.parse_single_example(example_proto, features)
    image = parsed_feats['image']
    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [21, 480, 3])
    image = tf.squeeze(tf.image.rgb_to_grayscale(image), axis=2)
    return image

def test_parse_tfrecord_function():
    import cv2
    tfrecord_file = '../dataset/tfdata/val.tfrecord'
    ds = tf.contrib.data.TFRecordDataset(tfrecord_file, 'GZIP')
    ds = ds.map(parse_tfrecord_function)
    iterator = ds.make_one_shot_iterator()
    input = iterator.get_next()
    sess = tf.Session()
    for i in range(5000):
        o = sess.run(input)
        img = Image.fromarray(o)
        cv_img = np.array(img, dtype=np.uint8)
        cv2.imshow('test', cv_img)
        cv2.waitKey(2000)

def main():
    fire.Fire()

if __name__ == '__main__':
    test_doGen()
    # test_parse_tfrecord_function()
    # main()