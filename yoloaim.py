import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import cv2
import mss
import numpy as np
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import win32gui, win32ui, win32con, win32api

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import core.utils as utils


# Config
WEIGHTS_PATH = "./checkpoints/yolov4-416"
SHOW_REVIEW_WINDOW = True 
DETECT_CLASS_INDEX = 0    # 0 for Ct, 1 for T
CONTROL_MOUSE = True


class YoloAim:
    def __init__(
        self,
        weights_path,
        input_size = 416,
        iou = 0.45,
        score = 0.25,
        capture_dimension = (640, 640),
        classes = None,
        num_classes = 2,
        detect_class_index = None,
        review_window = False,
        control_mouse = False,
    ):
        """
        weights_path       : path of saved model weights
        input_size         : model image input size
        iou                : model Intersection over Union threshold
        score              : model score
        capture_dimension  : screen image input dimension from display
        classes            : classes from classification. 0 for CT and 1 for T.
        num_classes        : num of classes from classification, for current training just 2. 
        detect_class_index : define which class to detect, None for all. If not the index number must reflect index from classes.
                             I.e. 0 for CT, 1 for T.
        review_window      : draw box in new popup window for review.
        control_mouse      : control user mouse movement.
        """

        self.review_window = review_window
        self.control_mouse = control_mouse
        self.fps = 0
        self.start_time = time.time()
        self.capture_dimension = capture_dimension
        self.monitor = self.get_detect_dimension(capture_dimension)
        self.padding_dimension = self.get_padding_dimension(capture_dimension)

        # Detection Config
        self.classes = classes if classes is not None else ["CT", "T"]
        self.num_classes = num_classes
        self.detect_class_index = detect_class_index
        self.input_size = input_size
        self.iou = iou
        self.score = score
        self.model = tf.saved_model.load(weights_path, tags=[tag_constants.SERVING])
        print("Finished loading weights.")

    def get_detect_dimension(self, capture_dimension):
        """
        Return screen dimension for object detection.

        capture_dimension: [w, h]
        """

        dimension = [win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)] # x, y
        center = [int(dimension[0] / 2), int(dimension[1] / 2)]
        x1 = center[0] - int(capture_dimension[0] / 2)
        y1 = center[1] - int(capture_dimension[1] / 2)

        return (x1, y1, capture_dimension[0], capture_dimension[1])

    def get_padding_dimension(self, capture_dimension):
        """
        Return screen padding for object detection.

        capture_dimension: [w, h]
        """

        dimension = [win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)] # x, y
        padding_w = (dimension[0] - capture_dimension[0]) / 2
        padding_h = (dimension[1] - capture_dimension[1]) / 2
        return (padding_w, padding_h)

    def grab_screen(self, region=None):
        hwin = win32gui.GetDesktopWindow()

        if region:
                left,top,x2,y2 = region
                width = x2 - left + 1
                height = y2 - top + 1
        else:
            width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
        
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height,width,4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    def detect(self, original_image):
        image_data = cv2.resize(original_image, (self.input_size, self.input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        infer = self.model.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou,
            score_threshold=self.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        if self.review_window:
            image = utils.draw_bbox(original_image, pred_bbox)
            return image, pred_bbox
        else:
            return None, pred_bbox

    def start(self):
        monitor_dict = {
            "top": self.monitor[1],    # y1
            "left": self.monitor[0],   # x1
            "width": self.monitor[2],  # w
            "height": self.monitor[3], # h
        }

        title = "YoloAim"
        display_time = 2 # displays the frame rate every 2 second
        sct = mss.mss()

        while True:
            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(monitor_dict))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # detect object
            img, pred_bbox = self.detect(img)

            # move user mouse to object detected with highest score
            if self.control_mouse:
                self.control(pred_bbox)

            # Display the picture in grayscale
            self.fps += 1
            TIME = time.time() - self.start_time
            if (TIME) >= display_time :
                # print("FPS: ", self.fps / (TIME))
                self.fps = 0
                self.start_time = time.time()

            if self.review_window:
                # Display the picture
                cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                # Press "q" to quit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

    def control(self, bboxes):
        chosen_class_index = None
        highest_score = None
        position_x = None
        position_y = None

        out_boxes, out_scores, out_classes, num_boxes = bboxes
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > self.num_classes:
                continue

            # if not for classes user want to detect; skip
            class_ind = int(out_classes[0][i])
            if self.detect_class_index is not None and class_ind != self.detect_class_index:
                continue

            # if current score is less than score you saw previously also skip
            score = out_scores[0][i]
            if highest_score is None:
                highest_score = score
            elif highest_score is not None and highest_score > score:
                continue

            chosen_class_index = class_ind

            coor = out_boxes[0][i]
            # NOTE: because this calculation is also done in utils.draw_bbox, `draw` method
            #       so do not calculate again here, if no review window, that method will
            #       not be called and we will have to calculate here
            if not self.review_window:
                coor[0] = int(coor[0] * self.capture_dimension[1]) # y1
                coor[1] = int(coor[1] * self.capture_dimension[0]) # x1
                coor[2] = int(coor[2] * self.capture_dimension[1]) # y2
                coor[3] = int(coor[3] * self.capture_dimension[0]) # x2

            # find center position in capture dimensin
            position_x = (coor[1] + coor[3]) / 2
            position_y = (coor[0] + coor[2]) / 2

        # run only if can detect
        if chosen_class_index is not None:
            # get center position based on display dimension (add back padding)
            position_x = int(position_x + self.padding_dimension[0])
            position_y = int(position_y + self.padding_dimension[1])

            win32api.SetCursorPos((position_x, position_y))


if __name__ == "__main__":
    aim = YoloAim(
        weights_path = WEIGHTS_PATH,
        detect_class_index = DETECT_CLASS_INDEX,
        review_window = SHOW_REVIEW_WINDOW,
        control_mouse = CONTROL_MOUSE,
    )
    aim.start()
