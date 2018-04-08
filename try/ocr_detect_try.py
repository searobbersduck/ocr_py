# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import PyPDF2
from wand.image import Image as WImage
from glob import glob
import cv2
import numpy as np

outdir = './out_ocr_detect'

def test_pypdf2():
    import PyPDF2
    src_pdf = PyPDF2.PdfFileReader('/Users/higgs/beast/doc/图片简历/18.pdf')
    title_info = src_pdf.documentInfo['/Title']
    title_info = title_info.encode('utf8')
    print(src_pdf.getNumPages())

def test_wand():
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    from wand.image import Image
    with Image(filename='/Users/higgs/beast/doc/图片简历/18.pdf[1]', resolution=300) as img:
        img.save(filename=os.path.join(outdir, 'sss.jpg'))

def pdf2image(pdf_file):
    name = os.path.basename(pdf_file).split('.')[0]
    dirname = os.path.dirname(pdf_file)
    src_pdf = PyPDF2.PdfFileReader(pdf_file)
    numPages = src_pdf.getNumPages()
    for i in range(numPages):
        filename = '{}[{}]'.format(pdf_file, i)
        with WImage(filename=filename, resolution=300) as img:
            img.save(filename=os.path.join(dirname, '{}[{}].jpg'.format(name, i)))

def test_pdf2image():
    # dir = '/Users/higgs/beast/doc/图片简历/guidang'
    dir = '/Users/higgs/beast/doc/图片简历/无法解析/其他'
    pdfs = glob(os.path.join(dir, '*.pdf'))
    for pdf in pdfs:
        pdf2image(pdf)

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

def gen_detect_jpg():
    det = ocr_detector()
    # resume_path = '/Users/higgs/beast/doc/图片简历/guidang/216[0].jpg'
    resume_path = '/Users/higgs/beast/doc/图片简历/无法解析/其他/01-陈明芽[0].jpg'
    img = cv2.imread(resume_path)
    bboxes, gray = det.detect(img, True)
    cv2.imwrite('gray.jpg', gray)
    new_img = np.ones(gray.shape, dtype=np.uint8)*255
    for bbox in bboxes:
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        new_img[y:y + h, x:x + w] = gray[y:y+h, x:x+w]
    cv2.imwrite('img.jpg', new_img)

def test_ocr_detector():
    det = ocr_detector()
    # resume_path = '/Users/higgs/beast/doc/图片简历/guidang/18[0].jpg'
    resume_path = '/Users/higgs/beast/doc/图片简历/无法解析/其他/01-陈明芽[0].jpg'
    img = cv2.imread(resume_path)
    bboxes, gray = det.detect(img, True)
    for bbox in bboxes:
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        img_tmp = gray[y:y+h, x:x+w]
        cv2.imshow('img', img_tmp)
        cv2.waitKey(1000)

def _makedirs(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def gen_detect_patches(resume_dir):
    '''
    dir tree as follows:

    out_resume_patches
        name
            sub_img_dir
                sub_img
                patches

    '''
    out_dir = './out_resume_patches'
    _makedirs(outdir)
    inp_dir = resume_dir
    resume_pdfs = glob(os.path.join(resume_dir, '*.pdf'))
    for pdf in resume_pdfs:
        name = os.path.basename(pdf).split('.')[0]
        out_dir_byname = os.path.join(out_dir, name)
        _makedirs(out_dir_byname)
        src_pdf = PyPDF2.PdfFileReader(pdf)
        numPages = src_pdf.getNumPages()
        # for i in range(numPages):
        #     out_dir_byname_sub = os.path.join(out_dir_byname, '{}'.format(i))
        #     _makedirs(out_dir_byname_sub)
        #     filename = '{}[{}]'.format(pdf, i)
        #     with WImage(filename=filename, resolution=300) as img:
        #         filename = os.path.join(out_dir_byname_sub, '{}[{}].jpg'.format(name, i))
        #         img.save(filename=filename)
        #         print('====> pdf to image: \t{}'.format(filename))
    '''
    遍历
    '''
    out_dirs_byname = os.listdir(out_dir)
    for out_dir_byname in out_dirs_byname:
        out_dir_byname = os.path.join(out_dir, out_dir_byname)
        if not os.path.isdir(out_dir_byname):
            continue
        out_dirs_byname_sub = os.listdir(out_dir_byname)
        for out_dir_byname_sub in out_dirs_byname_sub:
            out_dir_byname_sub = os.path.join(out_dir_byname, out_dir_byname_sub)
            if not os.path.isdir(out_dir_byname_sub):
                continue
            imagename = glob(os.path.join(out_dir_byname_sub, '*.jpg'))
            imagename = imagename[0]
            patches_dir = os.path.join(out_dir_byname_sub, 'patches')
            _makedirs(patches_dir)
            basename = os.path.basename(imagename).split('.')[0]
            det = ocr_detector()
            img = cv2.imread(imagename)
            bboxes, gray = det.detect(img, True)
            new_img = np.ones(gray.shape, dtype=np.uint8) * 255
            for cnt,bbox in enumerate(bboxes):
                x = bbox[0]
                y = bbox[1]
                w = bbox[2]
                h = bbox[3]
                img_tmp = gray[y:y + h, x:x + w]
                patch_img_name = os.path.join(patches_dir, '{}-{}.jpg'.format(basename, cnt))
                cv2.imwrite(patch_img_name, img_tmp)
                new_img[y:y + h, x:x + w] = gray[y:y + h, x:x + w]
            gray_img_name = os.path.join(patches_dir, '{}-gray.jpg'.format(basename))
            cv2.imwrite(gray_img_name, new_img)



def test_gen_detect_patches():
    dir = '/Users/higgs/beast/doc/图片简历/无法解析/其他'
    gen_detect_patches(dir)

if __name__ == '__main__':
    # test_pypdf2()
    # test_wand()
    # test_pdf2image()
    # test_ocr_detector()
    # gen_detect_jpg()
    test_gen_detect_patches()