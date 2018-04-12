# ! /usr/bin/env python
# -*- coding:utf-8 -*-

'''
统计用来生成训练集的简历中字符的分布
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json

def loadVocab(path, vocab):
    idx = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            ss = line.split('\t')
            idx += 1
            vocab[ss[0].decode('utf8')[0]] = idx

def import_vocab_gb2312(txt, vocab):
    with open(txt, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            ss = line.split('\t')
            # vocab[ss[0].decode('utf8')[0]] = int(ss[1])
            vocab[ss[0].decode('utf8')[0]] = int(ss[1])
        print('the word number in vocab is: {}'.format(len(vocab)))


def stat(input_prefix, start=0, end=1000):
    vocab = {}
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
                if line_cnt%500 == 0:
                    print('line number: {}\t\tfile: {}'.format(line_cnt, input_file))
                pos = line.find('\t')
                if pos == -1:
                    continue
                remind = line[pos + 1:]
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
                        for c in s:
                            if c in vocab:
                                vocab[c] += 1
                            else:
                                vocab[c] = 1
    import operator
    sorted_voc = sorted(vocab.items(), key=operator.itemgetter(1))
    outdir = './out_stat_words'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, 'stat_res1.txt')
    ccnt = 0
    with open(outfile, 'w') as f:
        for key, value in sorted_voc:
            ccnt += 1
            f.write(key.encode('utf8'))
            f.write('\t')
            f.write('{}'.format(value))
            f.write('\n')
    vocab_dict = {}
    loadVocab('../dataset/unicode_chars.txt', vocab_dict)
    list_no = []
    for key, value in sorted_voc:
        if key not in vocab_dict:
            list_no.append([key, value])
    out_no_file = os.path.join(outdir, 'stat_no.txt')
    with open(out_no_file, 'w') as f:
        for l in list_no:
            f.write(l[0].encode('utf8'))
            f.write('\t')
            f.write('{}'.format(l[1]))
            f.write('\n')

def test_stat():
    input_prefix = '../dataset/resume_doc/part-m'
    stat(input_prefix, 0, 100)

def _makedirs(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def stat_gb2312(input_prefix, start=0, end=1000, out_dir='./out_resume_vocab'):
    vocab_gb = {}
    import_vocab_gb2312('./out_ocr_fullfonts_gen/unicode_chars.txt', vocab_gb)
    vocab = {}
    for i in range(start, end):
        input_file = '{}-{:05d}'.format(input_prefix, i)
        line_cnt = 0
        if not os.path.isfile(input_file):
            continue
        content_to_store = []
        content_not_gb = []
        content_abnormal = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                line_cnt += 1
                if line_cnt%500 == 0:
                    print('line number: {}\t\tfile: {}'.format(line_cnt, input_file))
                pos = line.find('\t')
                if pos == -1:
                    continue
                remind = line[pos + 1:]
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
                        try:
                            s.encode('gb2312')
                        except:
                            content_not_gb.append(s)
                            continue
                        next_loop = False
                        for c in s:
                            if c not in vocab_gb:
                                next_loop = True
                                break
                        if next_loop:
                            content_abnormal.append(s)
                            continue
                        for c in s:
                            if c in vocab:
                                vocab[c] += 1
                            else:
                                vocab[c] = 1
                        content_to_store.append(s)
        print(len(content_to_store))
        _makedirs(out_dir)
        out_file = os.path.join(out_dir, '{}-{:05d}-gb'.format(input_prefix.split('/')[-1], i))
        with open(out_file, 'w') as f:
            for c in content_to_store:
                if len(c) < 2:
                    continue
                f.write(c.encode('utf8').strip())
                f.write('\n')
        out_file = os.path.join(out_dir, '{}-{:05d}-nogb'.format(input_prefix.split('/')[-1], i))
        with open(out_file, 'w') as f:
            for c in content_not_gb:
                f.write(c.encode('utf8').strip())
                f.write('\n')
        out_file = os.path.join(out_dir, '{}-{:05d}-abnormal'.format(input_prefix.split('/')[-1], i))
        with open(out_file, 'w') as f:
            for c in content_abnormal:
                f.write(c.encode('utf8').strip())
                f.write('\n')
    import operator
    sorted_voc = sorted(vocab.items(), key=operator.itemgetter(1))
    outdir = './out_stat_words'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, 'stat_res.txt')
    ccnt = 0
    with open(outfile, 'w') as f:
        for key, value in sorted_voc:
            ccnt += 1
            f.write(key.encode('utf8'))
            f.write('\t')
            f.write('{}'.format(value))
            f.write('\n')
    vocab_dict = {}
    loadVocab('../dataset/unicode_chars.txt', vocab_dict)
    list_no = []
    for key, value in sorted_voc:
        if key not in vocab_dict:
            list_no.append([key, value])
    out_no_file = os.path.join(outdir, 'stat_no.txt')
    with open(out_no_file, 'w') as f:
        for l in list_no:
            f.write(l[0].encode('utf8'))
            f.write('\t')
            f.write('{}'.format(l[1]))
            f.write('\n')

def test_stat_gb2312():
    input_prefix = '../dataset/resume_doc/part-m'
    stat_gb2312(input_prefix, 0, 100)

def switch_abnormal_data_to_normal(input_prefix, start=0, end=1000):
    vocab_gb = {}
    import_vocab_gb2312('./out_ocr_fullfonts_gen/unicode_chars.txt', vocab_gb)
    vocab_miss = {}
    for i in range(start, end):
        input_file = '{}-{:05d}-abnormal'.format(input_prefix, i)
        output_file = '{}-{:05d}-normal'.format(input_prefix, i)
        line_cnt = 0
        if not os.path.isfile(input_file):
            continue
        content_normal = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                line = line.decode('utf8')
                ss = u''
                for c in line:
                    if c not in vocab_gb:
                        if c in vocab_miss:
                            vocab_miss[c] += 1
                        else:
                            vocab_miss[c] = 1
                        continue
                    ss += c
                if len(ss) >= 1:
                    content_normal.append(ss)
        with open(output_file, 'w') as f:
            for content in content_normal:
                f.write(content.encode('utf8'))
                f.write('\n')
        print('====> Convert {} to {}'.format(input_file, output_file))
    import operator
    sorted_voc = sorted(vocab_miss.items(), key=operator.itemgetter(1))
    outdir = './out_stat_words'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, 'stat_res_allmiss.txt')
    ccnt = 0
    with open(outfile, 'w') as f:
        for key, value in sorted_voc:
            ccnt += 1
            f.write(key.encode('utf8'))
            f.write('\t')
            f.write('{}'.format(value))
            f.write('\n')

def test_switch_abnormal_data_to_normal():
    input_prefix = './out_resume_vocab/part-m'
    switch_abnormal_data_to_normal(input_prefix, 0, 100)

def merge_gb_and_normal(input_prefix, start=0, end=1000):
    for i in range(start, end):
        input_file_gb = '{}-{:05d}-gb'.format(input_prefix, i)
        input_file_normal = '{}-{:05d}-normal'.format(input_prefix, i)
        output_file_all = '{}-{:05d}-all'.format(input_prefix, i)
        line_cnt = 0
        if not (os.path.isfile(input_file_gb) or os.path.isfile(input_file_normal)):
            continue
        content_all = []
        with open(input_file_gb) as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 1:
                    content_all.append(line)
        with open(input_file_normal) as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 1:
                    content_all.append(line)
        with open(output_file_all, 'w') as f:
            for content in content_all:
                f.write(content)
                f.write('\n')
        print('====> Merge {} and {} to {}'.format(os.path.basename(input_file_gb),
                                                   os.path.basename(input_file_normal),
                                                   os.path.basename(output_file_all)))

def test_merge_gb_and_normal():
    input_prefix = './out_resume_vocab/part-m'
    merge_gb_and_normal(input_prefix, 0, 100)

def stat_all(input_prefix, start=0, end=1000):
    vocab_gb = {}
    import_vocab_gb2312('./out_ocr_fullfonts_gen/unicode_chars.txt', vocab_gb)
    vocab = {}
    for i in range(start, end):
        input_file = '{}-{:05d}-all'.format(input_prefix, i)
        print('====> stat {}'.format(os.path.basename(input_file)))
        line_cnt = 0
        if not os.path.isfile(input_file):
            continue
        vocab = {}
        with open(input_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                ss = line.decode('utf8')
                for c in ss:
                    if c in vocab:
                        vocab[c] += 1
                    else:
                        vocab[c] = 1
    import operator
    sorted_voc = sorted(vocab.items(), key=operator.itemgetter(1))
    outdir = './out_stat_words'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, 'stat_res_all.txt')
    ccnt = 0
    with open(outfile, 'w') as f:
        for key, value in sorted_voc:
            ccnt += 1
            f.write(key.encode('utf8'))
            f.write('\t')
            f.write('{}'.format(value))
            f.write('\n')
    list_no = []
    for key, value in sorted_voc:
        if key not in vocab_gb:
            list_no.append([key, value])
    out_no_file = os.path.join(outdir, 'stat_no_all.txt')
    with open(out_no_file, 'w') as f:
        for l in list_no:
            f.write(l[0].encode('utf8'))
            f.write('\t')
            f.write('{}'.format(l[1]))
            f.write('\n')

def test_stat_all():
    input_prefix = './out_resume_vocab/part-m'
    stat_all(input_prefix, start=0, end=1000)

def stat_miss():
    input_file = './out_stat_words/stat_res_allmiss.txt'
    list = []
    vocab = {}
    with open(input_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            ss = line.split('\t')
            try:
                vocab[ss[0]] = int(ss[1])
            except:
                ss
                continue
    import operator
    sorted_voc = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
    str = ''
    ustr = u''
    for key, value in sorted_voc:
        str += key
        ustr += key.decode('utf8')
    print(str)
    print(ustr)
    print('hello')

if __name__ == '__main__':
    # test_stat()
    test_stat_gb2312()
    test_switch_abnormal_data_to_normal()
    test_merge_gb_and_normal()
    test_stat_all()
    stat_miss()