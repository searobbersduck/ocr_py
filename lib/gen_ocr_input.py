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
    for f in flist:
        if f.endswith('.ttf') or f.endswith('.TTF'):
            f = os.path.join(fontDir, f)
            font = ImageFont.truetype(f, size)
            print('load font {} with size {}.'.format(
                os.path.basename(f), size
            ))
            fonts.append(font)
    return fonts

def loadVocab(txt, vocab):
    idx = 0
    with open(txt, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            ss = line.split('\t')
            idx += 1
            vocab[ss[0]] = idx
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
    img = fontImg.resize((w, 19), Image.BOX).point(
        lambda p: p>200 and 255
    )
    image.paste(img, (2, int(1+offset)))
    return image.tobytes()

def generateImages(sizedFonts, sizes, vocab, writer, ustr, nlength):
    sizeidx = np.random.randint(0, len(sizes)-1)
    fonts = sizedFonts[sizeidx]
    size = sizes[sizeidx]
    font = np.random.choice(fonts)
    imageBytes = None
    try:
        imageBytes = generateImageBytes(font, size, ustr)
        if imageBytes is None:
            return False
    except:
        print('Generate exception with font {} & size {}'.format(
            font.getname(), font.size()
        ) + '\t' + ustr)
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
    nn = (n-1)//MAX_CHARS_PER_BOX+1
    for i in range(nn):
        start = i*MAX_CHARS_PER_BOX
        end = (i+1)*MAX_CHARS_PER_BOX
        if end >= n:
            end = n
        yield content[start:end]

def doGen(fontDir, vocab_txt, outdir, input_prefix, start=0, end=1000):
    sizedFonts = []
    sizes = [15, 20, 25, 30, 40, 50]
    for size in sizes:
        fonts = loadFont(fontDir, size)
        sizedFonts.append(fonts)
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
        if not os.path.isfile(input_file):
            continue
        with open(input_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
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
                            nc = len(ustr)
                            if (nc > 10) and nc < MAX_CHARS_PER_BOX:
                                writer = writerTrain
                                rand_x = np.random.random()
                                if rand_x < 0.7:
                                    writer = writerTrain
                                elif rand_x < 0.8:
                                    writer = writerVal
                                else:
                                    writer = writerTest
                                if generateImages(sizedFonts, sizes, vocab, writer, ustr, nc):
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
    outdir = '../dataset/tfdata'
    input_prefix = '../dataset/resume_doc/part-m'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    print('====> begin doGen test: ')
    doGen(fontDir, vocab_txt, outdir, input_prefix)
    print('====> end doGen test.')

if __name__ == '__main__':
    test_doGen()