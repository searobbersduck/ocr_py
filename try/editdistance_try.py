# !/usr/bin/env python2
# -*- coding:utf8 -*-

'''
编辑距离：python
referene: [editdistance](https://github.com/aflc/editdistance)
install: pip install editdistance
'''

import editdistance

a = u'你好'
b = u'您好'
dis = editdistance.eval(a, b)
print(u'the distance between"[{}]" and "[{}]" is:\t {}'.format(a,b,dis))

a = u'你好'
b = u'您好啊'
dis = editdistance.eval(a, b)
print(u'the distance between"[{}]" and "[{}]" is:\t {}'.format(a,b,dis))


