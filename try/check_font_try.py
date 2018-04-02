# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from PIL import ImageFont, Image, ImageDraw
import fire
import numpy as np

def loadFonts(fontspath, size, vocab_fonts):
    flist = os.listdir(fontspath)
    for f in flist:
        if f.endswith('.ttf') or f.endswith('.TTF'):
            name = f.split('.')[0]
            font = ImageFont.truetype(os.path.join(fontspath, f), size)
            vocab_fonts[name] = font
            print('load font：{}'.format(f))

def test_loadFonts():
    fontspath = '../dataset/useFont'
    vocab_fonts = {}
    loadFonts(fontspath, 30, vocab_fonts)

def loadVocab(path, vocab):
    idx = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            ss = line.split('\t')
            idx += 1
            vocab[ss[0].decode('utf8')[0]] = idx

def checkFont(font, uchar):
    try:
        w,h = font.getsize(uchar)
        img = Image.new('RGB', [w,h], 'black')
        drawObj = ImageDraw.Draw(img)
        drawObj.text([0,0], uchar, font=font, fill=(255,255,255))
        cv_img = np.array(img, dtype=np.uint8)
        flag = cv_img.any()
        if not flag:
            # print(uchar)
            return False
        return True
    except:
        # print(uchar)
        return False

def test_checkFont():
    fontspath = '../dataset/useFont'
    vocabpath = '../dataset/unicode_chars.txt'
    vocab_fonts = {}
    loadFonts(fontspath, 30, vocab_fonts)
    vocab = {}
    loadVocab(vocabpath, vocab)

    outdir = './out_check_font'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for name,font in vocab_fonts.iteritems():
        list_in = []
        list_out = []
        for ustr, idx in vocab.iteritems():
            if checkFont(font, uchar=ustr):
                list_in.append(ustr)
            else:
                list_out.append(ustr)
        out_file_in = os.path.join(outdir, '{}.txt'.format(name))
        out_file_out = os.path.join(outdir, '{}-None.txt'.format(name))
        with open(out_file_in, 'w') as f:
            for l in list_in:
                f.write(l.encode('utf8'))
                f.write('\n')
        with open(out_file_out, 'w') as f:
            for l in list_out:
                f.write(l.encode('utf8'))
                f.write('\n')

def test_checkFont_with_encodegb2312():
    fontspath = '../dataset/useFont'
    vocabpath = '../dataset/data_todo/xxx/unicode_chars.txt'
    vocab_fonts = {}
    loadFonts(fontspath, 30, vocab_fonts)
    vocab = {}
    loadVocab(vocabpath, vocab)

    outdir = './out_check_font'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for name, font in vocab_fonts.iteritems():
        except_cnt = 0
        error_cnt = 0
        error_str = u''
        for ustr, idx in vocab.iteritems():
            try:
                ustr.encode('gb2312')
                if not checkFont(font, ustr):
                    error_cnt += 1
                    error_str += ustr
            except:
                except_cnt += 1
        print('{} exception count: {}'.format(name, except_cnt))
        print('{} error count: {}'.format(name, error_cnt))
        print(u'{} error string: {}'.format(name.decode('utf8'), error_str))



def main():
    fire.Fire()

if __name__ == '__main__':
    # main()
    # test_checkFont()
    test_checkFont_with_encodegb2312()