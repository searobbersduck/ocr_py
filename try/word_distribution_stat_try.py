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



def test_stat():
    input_prefix = '../dataset/resume_doc/part-m'
    stat(input_prefix, 0, 100)

if __name__ == '__main__':
    test_stat()
