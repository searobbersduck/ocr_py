# !/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import PyPDF2
from wand.image import Image as WImage
from glob import glob
import cv2
import numpy as np
import tensorflow as tf

from check_font_try import loadFonts

from PIL import Image, ImageDraw, ImageFont

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
        self.meser = mser if mser is not None else (cv2.MSER_create(5, 200, 30000, 0.25))

    def detect(self, img, show=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, tmp_gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        tmp_gray = cv2.erode(tmp_gray, kernel, iterations=1)
        cv2.imwrite('ocr_detect_saver_gray.jpg', tmp_gray)
        regions, boxes = self.meser.detectRegions(tmp_gray)
        # background image
        bk_img = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(bk_img, regions, -1, 255)
        # close operation
        closed_img = cv2.morphologyEx(bk_img, cv2.MORPH_CLOSE, np.ones((1,20), np.uint8))
        cv2.imwrite('ocr_detect_saver_contour.jpg', bk_img)
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
        w = rect[2]
        h = rect[3]
        ratio = w/h
        if (ratio < 0.2) or (ratio > 70):
            return False, rect
        area = w*h
        if area < 200:
            return False, rect
        # 对于分辨率300px， 5号字的大小44*44， 那这里将height < 22 and ratio > 3的删除
        if (h < 22) and (ratio > 3):
            return False, rect
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
    # resume_path = '/Users/higgs/beast/doc/图片简历/无法解析/其他/01-陈明芽[0].jpg'
    resume_path = '/Users/higgs/beast/code/work/ocr_py/try/out_resume_patches/20/0/20[0].jpg'
    img = cv2.imread(resume_path)
    bboxes, gray = det.detect(img, True)
    # for bbox in bboxes:
    #     x = bbox[0]
    #     y = bbox[1]
    #     w = bbox[2]
    #     h = bbox[3]
    #     img_tmp = gray[y:y+h, x:x+w]
    #     cv2.imshow('img', img_tmp)
    #     cv2.waitKey(1000)

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
    # for pdf in resume_pdfs:
    #     name = os.path.basename(pdf).split('.')[0]
    #     out_dir_byname = os.path.join(out_dir, name)
    #     _makedirs(out_dir_byname)
    #     src_pdf = PyPDF2.PdfFileReader(pdf)
    #     numPages = src_pdf.getNumPages()
    #     for i in range(numPages):
    #         out_dir_byname_sub = os.path.join(out_dir_byname, '{}'.format(i))
    #         _makedirs(out_dir_byname_sub)
    #         filename = '{}[{}]'.format(pdf, i)
    #         with WImage(filename=filename, resolution=300) as img:
    #             filename = os.path.join(out_dir_byname_sub, '{}[{}].jpg'.format(name, i))
    #             img.save(filename=filename)
    #             print('====> pdf to image: \t{}'.format(filename))
    '''
    遍历
    '''
    out_dirs_byname = os.listdir(out_dir)
    for out_dir_byname in out_dirs_byname:
        out_dir_byname = os.path.join(out_dir, out_dir_byname)
        if not os.path.isdir(out_dir_byname):
            continue
        out_dirs_byname_sub = os.listdir(out_dir_byname)
        list_names = []
        list_names_file = ''
        for out_dir_byname_sub in out_dirs_byname_sub:
            out_dir_byname_sub = os.path.join(out_dir_byname, out_dir_byname_sub)
            if not os.path.isdir(out_dir_byname_sub):
                continue
            imagename = glob(os.path.join(out_dir_byname_sub, '*.jpg'))
            imagename = imagename[0]
            patches_dir = os.path.join(out_dir_byname_sub, 'patches')
            _makedirs(patches_dir)
            list_names_file = os.path.join(patches_dir, 'flags.txt')
            basename = os.path.basename(imagename).split('.')[0]
            det = ocr_detector()
            img = cv2.imread(imagename)
            bboxes, gray = det.detect(img, True)
            new_img = np.ones(gray.shape, dtype=np.uint8) * 255
            raw_w = gray.shape[1]
            raw_h = gray.shape[0]
            padding_x = 5
            padding_y = 3
            for cnt,bbox in enumerate(bboxes):
                x = bbox[0]
                y = bbox[1]
                w = bbox[2]
                h = bbox[3]
                x = max(0, x-padding_x)
                y = max(0, y-padding_y)
                x_high = min(x+w+padding_x, raw_w)
                y_high = min(y+h+padding_y, raw_h)
                img_tmp = gray[y:y_high, x:x_high]
                patch_img_name = os.path.join(patches_dir, '{}-{}.jpg'.format(basename, cnt))
                cv2.imwrite(patch_img_name, img_tmp)
                list_names.append(os.path.basename(patch_img_name))
                new_img[y:y + h, x:x + w] = gray[y:y + h, x:x + w]
            gray_img_name = os.path.join(patches_dir, '{}-gray.jpg'.format(basename))
            cv2.imwrite(gray_img_name, new_img)
            with open(list_names_file, 'w') as f:
                for name in list_names:
                    f.write(name)
                    f.write('\t')
                    f.write('\n')
            print('====> Save flag file: {}'.format(list_names_file))


def test_gen_detect_patches():
    # dir = '/Users/higgs/beast/doc/图片简历/无法解析/其他'
    dir = '/Users/higgs/beast/doc/图片简历/guidang'
    gen_detect_patches(dir)

def resize_and_save_img(img_path, out_dir):
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # out_dir = os.path.dirname(img_path)
    name = os.path.basename(img_path).split('.')[0]
    out_file = os.path.join(out_dir, '{}-normal.jpg'.format(name))
    h = 19
    w = 480
    i_h = img.shape[0]
    i_w = img.shape[1]
    o_w = int(h/i_h*i_w)
    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    resized_img = cv2.resize(img, (o_w, h), interpolation=cv2.INTER_LINEAR)
    ret, resized_img = cv2.threshold(resized_img, 200, 255, cv2.THRESH_BINARY)
    o_w = min(w-2, o_w)
    resized_img = resized_img[:, 0:o_w, :]
    bg_img = np.ones([21, 480, 3])*255
    bg_img[1:h+1, 2:o_w+2] = resized_img
    print(bg_img.shape)
    cv2.imwrite(out_file, bg_img)
    print('====> save {}'.format(out_file))

def test_resize_and_save_img():
    img_path = '/Users/higgs/beast/code/work/ocr_py/try/out_resume_patches/01-陈明芽/0/patches/01-陈明芽[0]-144.jpg'
    out_dir = './out_ocr_detect'
    _makedirs(out_dir)
    # imglist = glob(os.path.join('/Users/higgs/beast/code/work/ocr_py/try/out_resume_patches/01-陈明芽/0/patches', '*.jpg'))
    imglist = glob(os.path.join('/Users/higgs/beast/code/work/ocr_py/try/out_resume_patches/20/0/patches', '*.jpg'))
    for img in imglist:
        resize_and_save_img(img, out_dir)

def _int64_features(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def gen_tfrecord_test():
    out_dir = './out_ocr_detect'
    images_list = glob(os.path.join(out_dir, '*.jpg'))
    from PIL import Image
    import tensorflow as tf
    tfopts = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP
    )
    writer = tf.python_io.TFRecordWriter(
        '{}/test.tfrecord'.format(outdir), tfopts
    )
    for i, img_file in enumerate(images_list):
        img = Image.open(img_file)
        if not (img.size == (480, 21)):
            continue
        label = [7042 + 1] * 30
        nlabel = 30
        imageBytes = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label': _int64_features(label),
                'image': _bytes_feature(imageBytes),
                'nlabel': _int64_feature(nlabel)
            }
        ))
        writer.write(example.SerializeToString())
    writer.close()


def parse_tfrecord_function_with_raw(example_proto):
    TIME_STEPS = 120
    features = {
        'label': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        'image': tf.FixedLenFeature([], tf.string),
        'nlabel': tf.FixedLenFeature([], tf.int64, default_value=0)
    }
    parsed_feats = tf.parse_single_example(example_proto, features)
    image = parsed_feats['image']
    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [21, 4*TIME_STEPS, 3])
    image = tf.squeeze(tf.image.rgb_to_grayscale(image), axis=2)
    datas = tf.split(image, TIME_STEPS, axis=1, num=TIME_STEPS)
    datas = tf.stack(datas, axis=0)
    datas = tf.reshape(datas, [TIME_STEPS, 84])
    label = parsed_feats['label']
    nlabel = parsed_feats['nlabel']
    return tf.cast(datas, tf.int32), label, nlabel, image

def test_gen_tfrecord_test():
    # tfrecord = './out_ocr_detect/test.tfrecord'
    tfrecord = './tfdata/test.tfrecord-0'
    ds = tf.contrib.data.TFRecordDataset(tfrecord, 'GZIP')
    ds = ds.map(parse_tfrecord_function_with_raw)
    ds = ds.batch(1)
    iterator = ds.make_one_shot_iterator()
    inputs = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50000):
            try:
                o = sess.run(inputs)
                image = o[3][0]
                import numpy as np
                from PIL import Image
                import cv2
                pil_img = Image.fromarray(image)
                cv_img = np.array(pil_img, dtype=np.uint8)
                cv2.imshow('test', cv_img)
                cv2.waitKey(500)
            except:
                print('====> exception as step {}'.format(i))

'''
显示从简历中detect到的有字的patches
'''
def show_detected_patches():
    dir = '/Users/higgs/beast/code/work/ocr_py/try/out_ocr_detect'
    from glob import glob
    from PIL import Image
    list = glob(os.path.join(dir, '*.jpg'))
    for l in list:
        image = Image.open(l)
        cv_image = np.array(image, dtype=np.uint8)
        cv2.imshow('img', cv_image)
        cv2.waitKey(1000)


'''
数据增强
'''
class dataAugmentation(object):
    def __init__(self,noise=True,dilate=True,erode=True):
        self.noise = noise
        self.dilate = dilate
        self.erode = erode

    @classmethod
    def add_noise(cls,img):
        for i in range(20): #添加点噪声
            temp_x = np.random.randint(0,img.shape[0])
            temp_y = np.random.randint(0,img.shape[1])
            img[temp_x][temp_y] = 0
        return img

    @classmethod
    def add_erode(cls,img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
        img = cv2.erode(img,kernel)
        return img

    @classmethod
    def add_dilate(cls,img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
        img = cv2.dilate(img,kernel)
        return img

    def do(self,pil_img, dilate_flag=True):
        cv_img = np.array(pil_img, dtype=np.uint8)
        rand_x = np.random.random()
        if rand_x < 0.3:
            cv_img = self.add_noise(cv_img)
        rand_x = np.random.random()
        if rand_x < 0.0:
            cv_img = self.add_dilate(cv_img)
        rand_x = np.random.random()
        if rand_x < 0.5:
            cv_img = self.add_erode(cv_img)
        return Image.fromarray(cv_img)

'''
显示从简历中detect到的有字的patches，以及相应的利用字体库生成的文字
'''
def show_detected_patches_with_selfgen():
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    dir = '/Users/higgs/beast/code/work/ocr_py/resume'
    flag_file = os.path.join(dir, 'flag.txt')
    vocab = {}
    with open(flag_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            ss = line.split('\t')
            if not (len(ss) == 2):
                continue
            vocab[int(ss[0])] = ss[1].decode('utf8')
    font_size = 45
    vocab_fonts = {}
    fontdir = './useFont-small'
    loadFonts(fontdir, font_size, vocab_fonts)
    font_name = ''
    font = ''
    for key, value in vocab_fonts.iteritems():
        font_name = key
        font = value
    aug = dataAugmentation()
    for key,value in vocab.iteritems():
        ustr = value
        raw_file = os.path.join(dir, '{}.jpg'.format(key))
        if not os.path.isfile(raw_file):
            continue
        img_raw = Image.open(raw_file)
        raw_w, raw_h = img_raw.size
        gen_w, gen_h = font.getsize(ustr)
        img_gen = Image.new('RGB', [gen_w, gen_h], 'white')
        drawObj = ImageDraw.Draw(img_gen)
        drawObj.text([0,0], ustr, font=font, fill=(0,0,0,0))
        resize_h = raw_h-2
        resize_w = int(resize_h/gen_h*gen_w)
        # (np.random.random() < 1.9) and (p < 2550)
        img_gen = img_gen.point(
            lambda p: p+np.random.randint(50,150) if (np.random.random() < 0.8) and (p < 255) else p
        )
        img_gen = aug.do(img_gen)
        img_gen = img_gen.resize([resize_w, resize_h], Image.LINEAR).point(
            lambda p: p>200 and 255
        )
        np_img_raw = np.array(img_raw, dtype=np.uint8)
        np_img_gen = np.array(img_gen, dtype=np.uint8)
        np_img_gen = np_img_gen[:,:,0]
        # cv2.imshow('gen', np_img_gen)
        # cv2.waitKey(2000)
        img_stitch = np.ones([raw_h*2, raw_w], dtype=np.uint8)*255
        img_stitch[0:raw_h, :] = np_img_raw
        img_stitch[raw_h:raw_h+resize_h, 2:img_gen.size[0]+2] = np_img_gen
        cv2.imshow('stitch_image', img_stitch/255)
        cv2.waitKey(2000)


if __name__ == '__main__':
    # test_pypdf2()
    # test_wand()
    # test_pdf2image()
    # test_ocr_detector()
    # gen_detect_jpg()
    test_gen_detect_patches()
    # test_resize_and_save_img()
    # gen_tfrecord_test()
    # test_gen_tfrecord_test()
    # show_detected_patches()
    # show_detected_patches_with_selfgen()