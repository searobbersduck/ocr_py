#!/usr/bin/env python
# -*- coding:utf-8 -*-

# ocr detector class
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import matplotlib.pyplot as plt
import numpy as np

import os

def init_detect_inner_folder():
    root = './ocr_detect_inner'
    if not os.path.exists(root):
        os.makedirs(root)

    inner_save_names = {
        'raw': 'raw',
        'gray': 'gray',
        'mser_box': 'mser_box',
        'font_contour': 'font_contour',
        'font_contour_close': 'font_contour_close',
        'font_contour_dilate': 'font_contour_dilate',
        'font_contour_contour': 'font_contour_contour',
        'font_coutour_contour_bbox': 'font_contour_contour_bbox',
        'font_coutour_contour_bbox_proc': 'font_contour_contour_bbox_proc',
    }

    for k, v in inner_save_names.iteritems():
        inner_save_names[k] = os.path.join(root, v + '.jpg')
    return inner_save_names

def test_mser_params():
    '''
    该函数测试的结论：
    1. `cv2.MSER_create(5, 50, 100000, 0.25)`,
    其中的`50`是能检测的最小的区域，用来检测标点等，
    其中的100000是能检测到的最大的区域，这个参数限制了大字体；
    其它参数在我们的场景中不是很重要；

    '''
    inner_save_names = init_detect_inner_folder()
    resume_path = '../sub_func_test/ocr_image/0001.jpg'
    img = cv2.imread(resume_path)
    cv2.imwrite(inner_save_names['raw'], img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite(inner_save_names['gray'], gray)

    minareas = [20, 50, 100, 200, 300, 400, 500]

    for minarea in minareas:
        mser = cv2.MSER_create(5, minarea, 10000, 0.5)
        regions, boxes = mser.detectRegions(gray)
        gray_output = img.copy()
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(gray_output, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imwrite(inner_save_names['mser_box'].split('.jpg')[0]+'_minarea{}.jpg'.format(minarea), gray_output)

class ocr_detector:
    def __init__(self, mser=None):
        self.name = 'ocr_detector'
        self.meser = mser if mser is not None else (cv2.MSER_create(5, 50, 10000, 0.25))

    def detect(self, img, show=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        regions, boxes = self.meser.detectRegions(gray)
        # background image
        bk_img = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(bk_img, regions, -1, 255)
        # close operation
        closed_img = cv2.morphologyEx(bk_img, cv2.MORPH_CLOSE, np.ones((1,20), np.uint8))
        _, contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # new bk image
        bk_rect_img = np.zeros(gray.shape, np.uint8)
        bboxes = []
        for contour in contours:
            rect = cv2.boundingRect(contour)
            b, rect = self.filter(rect)
            if b:
                bboxes.append(rect)
                cv2.rectangle(bk_rect_img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255,255,255), 4)
        if show:
            # plt.imshow(bk_rect_img)
            # plt.show()
            cv2.imwrite('ocr_detect_saver.jpg', bk_rect_img)
        return bboxes, gray

    def filter(self, rect):
        return True, rect

def test_ocr_detector():
    det = ocr_detector()
    # resume_path = '../sub_func_test/ocr_image/0001.jpg'
    # resume_path = './test_pic/jiangjingwen.jpg'
    resume_path = './test_pic/suchao/0001.jpg'
    img = cv2.imread(resume_path)
    bboxes = det.detect(img, True)



if __name__ == '__main__':
    test_ocr_detector()
    # test_mser_params()
