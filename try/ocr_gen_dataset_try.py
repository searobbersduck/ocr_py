# !/usr/bin/env python
# -*- coding:utf-8 -*-

'''
1. load vocabulary
2. load font librarys
3. generate words by random (word number range from 2 to 30)
4. random size（range from 30 to 120） based resolution 300px(小四=50px, 五号=44px)
5. jittering: random (noise, erode, dilate, close, open) operation


refer: [【OCR技术系列之三】大批量生成文字训练集](https://www.cnblogs.com/skyfsm/p/8436820.html)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from PIL import Image, ImageFont, ImageDraw
import cv2

class FontsGenerator(object):
    def __init__(self, vocab_lib, fonts_lib, sizes=None, num_min=2, num_max=30):
        self.vocab_lib = vocab_lib
        self.fonts_lib = fonts_lib
        self.size_min = 30
        self.size_max = 60
        self.sizes = sizes if sizes is not None \
            else np.random.randint(self.size_min, self.size_max, 5)
        self.num_min = num_min
        self.num_max = num_max
        self.vocab = {}

    def _load_vocablib(self, vocab_lib, vocab):
        list = []
        idx = 0
        with open(vocab_lib, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                ss = line.split('\t')
                key = ss[0].decode('utf8')[0]
                idx += 1
                vocab[key] = idx
                list.append(key)
        return list

    def _load_fonts_bysize(self, fonts_lib, size):
        flist = os.listdir(fonts_lib)
        fonts = []
        for f in flist:
            if not (f.endswith('.ttf') or f.endswith('.TTF')):
                continue
            font_file = os.path.join(fonts_lib, f)
            font = ImageFont.truetype(font_file, size)
            fonts.append(font)
        return fonts

    def _load_fonts(self, fonts_lib, sizes):
        fonts = []
        for size in sizes:
            fonts_bysize = self._load_fonts_bysize(fonts_lib, size)
            fonts.append(fonts_bysize)
        return fonts

    def _generate_image(self, vocab_list, font, size, num_min, num_max):
        # 1. random ustr
        ustr = u''
        words_num = np.random.randint(num_min, num_max)
        vocab_len = len(vocab_list)
        for i in range(words_num):
            ustr += vocab_list[np.random.randint(vocab_len)]
        w,h = font.getsize(ustr)
        # 2. canvas
        canvas_img_w = w+2
        canvas_img_h = h+2
        canvas_img = Image.new('RGB', [canvas_img_w, canvas_img_h], 'white')
        # 3. words_img
        words_img = Image.new('RGB', [w, h], 'white')
        drawObj = ImageDraw.Draw(canvas_img)
        drawObj.text([1,1], ustr, font=font, fill=(0,0,0,0))
        # 4. cv show
        print(u'{}\t{}'.format(ustr, font.getname()))
        cv_img = np.array(canvas_img, dtype=np.uint8)
        cv2.imshow('generated_words', cv_img)
        cv2.waitKey(1000)
        print(u'{}\t{}'.format(ustr, font.getname()))

    def generate_images(self):
        vocab_list = self._load_vocablib(self.vocab_lib, self.vocab)
        fonts = self._load_fonts(self.fonts_lib, self.sizes)
        sizeidx = np.random.randint(0, len(self.sizes))
        size = self.sizes[sizeidx]
        fonts_bysize = fonts[sizeidx]
        font = fonts_bysize[np.random.randint(0, len(fonts_bysize))]

        self._generate_image(vocab_list, font, size, self.num_min, self.num_max)



def test_FontsGenerator():
    fonts_lib = '../dataset/font1'
    vocab_lib = '../dataset/unicode_chars.txt'
    generator = FontsGenerator(vocab_lib, fonts_lib)
    for i in range(1000):
        generator.generate_images()



# test 英文字体和中文字体显示英文的差别：
def test_diff_eng_chin():
    for i in range(1000):
        n_min = 10
        n_max = 30
        n_r = np.random.randint(10, 30)
        arr = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        ustr = u''
        for i in range(n_r):
            ustr += arr[np.random.randint(0, len(arr))]
        diff_eng_chin(ustr)

def diff_eng_chin(ustr):
    font_eng_file = '../dataset/useFont/FZXJHJW.ttf'
    font_chin_file = '../dataset/useFont/华文中宋.ttf'
    font_eng = ImageFont.truetype(font_eng_file, 50)
    font_chin = ImageFont.truetype(font_chin_file, 50)
    eng_w, eng_h = font_eng.getsize(ustr)
    chin_w, chin_h = font_chin.getsize(ustr)
    w = max(eng_w, chin_w)
    h = max(eng_h, chin_h)
    eng_img = Image.new('RGB', (eng_w+2, eng_h+2), 'white')
    chin_img = Image.new('RGB', (chin_w+2, chin_h+2), 'white')
    drawObj = ImageDraw.Draw(eng_img)
    drawObj.text([1,1], ustr, font=font_eng, fill=(0,0,0,0))
    drawObj = ImageDraw.Draw(chin_img)
    drawObj.text([1, 1], ustr, font=font_chin, fill=(0, 0, 0, 0))
    canvas_img = Image.new('RGB', (w*2+4, h*2+2), 'white')
    canvas_img.paste(eng_img, [1,1])
    canvas_img.paste(chin_img, [1, h+3])
    cv_img = np.array(canvas_img, dtype=np.uint8)
    cv2.imshow('diff', cv_img)
    cv2.waitKey(1000)

def check_special_char_chinese_show(ustr):
    ustr = u'Ø'
    ustr = u''
    ustr = u'•'
    ustr = u'る'
    ustr = u'織'
    ustr = u'遠'
    font_chin_file = '../dataset/useFont/华文中宋.ttf'
    font = ImageFont.truetype(font_chin_file, 20)
    w, h = font.getsize(ustr)
    image = Image.new('RGB', [w+2,h+2], 'white')
    drawObj = ImageDraw.Draw(image)
    drawObj.text([1,1], ustr, font=font, fill=(0,0,0,0))
    cv_img = np.array(image, dtype=np.uint8)
    cv2.imshow('test', cv_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    # test_FontsGenerator()
    # test_diff_eng_chin()
    check_special_char_chinese_show(u'sdf')