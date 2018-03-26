import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('traindatapath', '/Users/higgs/beast/code/work/ocr_py/sub_func_test/tfdata/train.tfrecord', 'train data path')
tf.app.flags.DEFINE_integer("max_epochs", 100, "max num of epoches")
tf.app.flags.DEFINE_integer("batch_size", 384, "num example per mini batch")

traindatapath = tf.app.flags.FLAGS.traindatapath

print(traindatapath)

def parse_tfrecord_function(example_proto):
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        'nlabel': tf.FixedLenFeature([], tf.int64, default_value=0)
    }
    parsed_feats = tf.parse_single_example(example_proto, features)
    image = parsed_feats['image']
    label = parsed_feats['label']
    nlabel = parsed_feats['nlabel']
    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [21, 600, 3])
    img = tf.squeeze(tf.image.rgb_to_grayscale(image), axis=2)
    datas = tf.split(img, 120, axis=1, num=120)
    datas = tf.stack(datas, axis=0)
    datas = tf.reshape(datas, [120, 105])
    return datas, label, nlabel

datasettrain = tf.contrib.data.TFRecordDataset(traindatapath, 'GZIP')
datasettrain = datasettrain.map(parse_tfrecord_function)

datasettrain = datasettrain.repeat(FLAGS.max_epochs)
datasettrain = datasettrain.shuffle(buffer_size=2048)
datasettrain = datasettrain.batch(FLAGS.batch_size)

iterator = datasettrain.make_one_shot_iterator()
inputs = iterator.get_next()

sess = tf.Session()

result = sess.run(inputs)

print(result[0])
print('************')
print(result[1])
print('************')
print(result[2])
print('************')
print(result[0].shape)

from PIL import Image

pilImg = Image.fromarray(result[0][3,:,:])
pilImg.show()
