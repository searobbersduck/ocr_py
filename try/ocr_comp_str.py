# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os

comp_file = './comp_str.txt'

with open(comp_file, 'r') as f:
    str_pred = u''
    str_label = u''
    cnt = 0
    log_comp = []
    for line in f.readlines():
        cnt += 1
        line = line.strip()
        if cnt%2 == 1:
            str_pred = line
        else:
            str_label = line
            if len(str_pred) != len(str_label):
                log_comp.append(str_pred)
                log_comp.append(str_label)
            else:
                if str_pred != str_label:
                    log_comp.append(str_pred)
                    log_comp.append(str_label)


for log in log_comp:
    print(log)

print(len(log))
