import sys
import os

from absl import app, flags
from absl.flags import FLAGS
from lxml import etree


flags.DEFINE_string('anno_dir', '../../data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations', 'path to anno dir')
flags.DEFINE_string('output', '../../data/classes/voc2012.names', 'path to anno dir')


def make_names(anno_dir, output):
    labels_dict = {}

    anno_list = os.listdir(anno_dir)

    for anno_file in anno_list:
        p = os.path.join(anno_dir, anno_file)
        
        # Get annotation.
        root = etree.parse(p).getroot()
        names = root.xpath('//object/name')

        for n in names:
            labels_dict[n.text] = 0
    
    labels = list(labels_dict.keys())
    labels.sort()

    with open(output, 'w') as f:
        for l in labels:
            f.writelines(l + '\n')

    print(f"Done making a names's file ({os.path.abspath(output)})")


def main(_argv):
    make_names(FLAGS.anno_dir, FLAGS.output)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass    
