from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import os
import json
import sys
import pickle

flags.DEFINE_string('input', '/Volumes/Elements/data/coco_dataset/coco/annotations/instances_val2017.json', 'path to classes file')
flags.DEFINE_string('output', 'val2017.pkl', 'path to classes file')

class COCO:
    """
    Handler Class for COCO Format
    """

    @staticmethod
    def parse(json_path):

        try:
            json_data = json.load(open(json_path))

            images_info = json_data["images"]
            cls_info = json_data["categories"]

            data = {}

            progress_length = len(json_data["annotations"])
            progress_cnt = 0

            for anno in json_data["annotations"]:

                image_id = anno["image_id"]
                cls_id = anno["category_id"]

                filename = None
                img_width = None
                img_height = None
                cls = None

                for info in images_info:
                        if info["id"] == image_id:
                            filename, img_width, img_height = \
                                info["file_name"].split(".")[0], info["width"], info["height"]

                for category in cls_info:
                    if category["id"] == cls_id:
                        cls = category["name"]

                size = {
                    "width": img_width,
                    "height": img_height,
                    "depth": "3"
                }

                bndbox = {
                    "xmin": anno["bbox"][0],
                    "ymin": anno["bbox"][1],
                    "xmax": anno["bbox"][2] + anno["bbox"][0],
                    "ymax": anno["bbox"][3] + anno["bbox"][1]
                }

                obj_info = {
                    "name": cls,
                    "bndbox": bndbox
                }

                if filename in data:
                    obj_idx = str(int(data[filename]["objects"]["num_obj"]))
                    data[filename]["objects"][str(obj_idx)] = obj_info
                    data[filename]["objects"]["num_obj"] = int(obj_idx) + 1

                elif filename not in data:

                    obj = {
                        "num_obj": "1",
                        "0": obj_info
                    }

                    data[filename] = {
                        "size": size,
                        "objects": obj
                    }

                percent = (float(progress_cnt) / float(progress_length)) * 100
                print(str(progress_cnt) + "/" + str(progress_length) + " total: " + str(round(percent, 2)))
                progress_cnt += 1

            #print(json.dumps(data, indent=4, sort_keys = True))
            return True, data

        except Exception as e:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            msg = "ERROR : {}, moreInfo : {}\t{}\t{}".format(e, exc_type, fname, exc_tb.tb_lineno)

            return False, msg

def main(_argv):
    coco = COCO()
    data = coco.parse(FLAGS.input)
    with open(FLAGS.output, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
