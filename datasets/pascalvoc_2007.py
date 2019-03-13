# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import tensorflow as tf
from datasets import pascalvoc_common

slim = tf.contrib.slim

FILE_PATTERN = 'voc_2007_%s_*.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}
# (Images, Objects) statistics on every class.
# TRAIN_STATISTICS = {
#     'none': (0, 0),
#     'aeroplane': (238, 306),
#     'bicycle': (243, 353),
#     'bird': (330, 486),
#     'boat': (181, 290),
#     'bottle': (244, 505),
#     'bus': (186, 229),
#     'car': (713, 1250),
#     'cat': (337, 376),
#     'chair': (445, 798),
#     'cow': (141, 259),
#     'diningtable': (200, 215),
#     'dog': (421, 510),
#     'horse': (287, 362),
#     'motorbike': (245, 339),
#     'person': (2008, 4690),
#     'pottedplant': (245, 514),
#     'sheep': (96, 257),
#     'sofa': (229, 248),
#     'train': (261, 297),
#     'tvmonitor': (256, 324),
#     'total': (5011, 12608),
# }

TRAIN_STATISTICS = {
    'none': (0, 0),
    'ignore': (86, 215),
    'car': (96, 683),
    'pedestrian': (56, 188),
    'cyclist': (42, 43),
    'truck': (63, 121),
    'motorcyclist': (39, 48),
    'tricyclelist': (24, 30),
    'barrowlist': (1, 1),
    'van': (56, 94),
    'trafficcone': (7, 20),
    'bus': (49, 117),
    'total': (98, 1560),
}
TEST_STATISTICS = {
    'none': (0, 0),
    'ignore': (1, 1),
    'car': (1, 1),
    'pedestrian': (1, 1),
    'cyclist': (1, 1),
    'truck': (1, 1),
    'motorcyclist': (1, 1),
    'tricyclelist': (1, 1),
    'barrowlist': (1, 1),
    'van': (1, 1),
    'trafficcone': (1, 1),
    'bus': (1, 1),
    'total': (11, 11),
}

SPLITS_TO_SIZES = {
    'train': 98,
    'test': 98,
}
SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
    'test': TEST_STATISTICS,
}
NUM_CLASSES = 11


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if not file_pattern:
        file_pattern = FILE_PATTERN
    return pascalvoc_common.get_split(split_name, dataset_dir,
                                      file_pattern, reader,
                                      SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,
                                      NUM_CLASSES)
