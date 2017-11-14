# modified from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md  

import logging
import random
import os
import sys
import io
import PIL.Image
import tensorflow as tf
from lxml import etree

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('input_xml_path', '', 'Path to input imglab xml file, eg. training.xml')
flags.DEFINE_string('image_dir', '', 'Root directory to image path relative to xml file.')
flags.DEFINE_string('output_dir', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto, eg. underground_corridor_label_map.pbtxt')
FLAGS = flags.FLAGS

def create_tf_record(output_filename, label_map_dict, image_dir, examples):
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples))

    tf_example = dict_to_tf_example(label_map_dict, image_dir, example)
    writer.write(tf_example.SerializeToString())

  writer.close()

def dict_to_tf_example(label_map_dict, image_dir, example):
  filename = example['file'] # Filename of the image. Empty if image is not from file
  img_path = os.path.join(image_dir, example['file'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image(' + img_path + ') format not JPEG')
  width, height = image.size
  encoded_image_data = encoded_jpg # Encoded image bytes
  image_format = 'jpeg'.encode('utf8') # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  for box in example['box']:
    xmin = float(box['left']) / width
    xmax = float(box['right']) / width
    ymin = float(box['top']) / height
    ymax = float(box['bottom']) / height
    if (xmin < 0):
        xmin = 0
    if (ymin < 0):
        ymin = 0
    if (xmax > 1):
        xmax = 1
    if (ymax > 1):
        ymax = 1
    xmins.append(xmin)
    xmaxs.append(xmax)
    ymins.append(ymin)
    ymaxs.append(ymax)
    classes_text.append(box['label'])
    classes.append(label_map_dict[box['label']])
    region = image.crop((xmin * width, ymin * height, xmax * width, ymax * height))
    region.save("/home/gzdev/tensorflow/models/research/tmp/" + box['label'] + "_" + example['file'].replace("/", "_"))
    """region.save("/home/gzdev/tensorflow/models/research/tmp/" + box['label'] + "_" + box['left'] + "_" + box['top'] + "_" + box['right'] + "_" + box['bottom'] + "_" + example['file'].replace("/", "_"))"""

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def recursive_parse_imglib_xml_to_dict(xml):
  result = dict(xml.attrib)
  for child in xml:
    if 'label' == child.tag:
      result[child.tag] = child.text
      continue
    child_result = recursive_parse_imglib_xml_to_dict(child)
    if child.tag not in result:
      result[child.tag] = []
    result[child.tag].append(child_result)
  return result

def main(_):
  logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
  image_dir = FLAGS.image_dir
  if not image_dir.endswith('/'):
    image_dir += '/'
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  input_xml_filename = FLAGS.input_xml_path
  with open(input_xml_filename, 'r') as f:
      xml_str = f.read()
  xml = etree.fromstring(xml_str)
  images = recursive_parse_imglib_xml_to_dict(xml)['images']
  examples_list = []
  for image in images:
    for box in image['image']:
      example = {'file':box['file'], 'box':[]}
      if not box['file'].endswith('jpg'):
        logging.warn("Ignored not-jpeg image '" + box['file'] + "'")
        continue
      for item in box['box']:
        if 'ignore' in item or 'label' not in item or '' == item['label']:
          continue
        if int(item['left']) < 0:
          item['left'] = '0'
        item['right'] = int(item['left']) + int(item['width'])
        if int(item['top']) < 0:
          item['top'] = '0'
        item['bottom'] = int(item['top']) + int(item['height'])
        example['box'].append(item)
      if len(example['box']):
        examples_list.append(example)

  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(0.7 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, 'from_imglab_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'from_imglab_val.record')
  create_tf_record(train_output_path, label_map_dict, image_dir, train_examples)
  create_tf_record(val_output_path, label_map_dict, image_dir, val_examples)

if __name__ == '__main__':
  tf.app.run()
  
  
