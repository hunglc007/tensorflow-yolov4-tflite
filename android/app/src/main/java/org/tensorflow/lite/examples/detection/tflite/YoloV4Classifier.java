/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Build;
import android.os.Trace;
import android.util.Log;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.MainActivity;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.env.Utils;

import static org.tensorflow.lite.examples.detection.env.Utils.expit;
import static org.tensorflow.lite.examples.detection.env.Utils.softmax;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * - https://github.com/tensorflow/models/tree/master/research/object_detection
 * where you can find the training code.
 * <p>
 * To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class YoloV4Classifier implements Classifier {

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param isQuantized   Boolean representing model is quantized or not
     */
    public static Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final boolean isQuantized)
            throws IOException {
        final YoloV4Classifier d = new YoloV4Classifier();

        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            d.labels.add(line);
        }
        br.close();

        try {
            Interpreter.Options options = (new Interpreter.Options());
            options.setNumThreads(NUM_THREADS);
            if (isNNAPI) {
                NnApiDelegate nnApiDelegate = null;
                // Initialize interpreter with NNAPI delegate for Android Pie or above
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    nnApiDelegate = new NnApiDelegate();
                    options.addDelegate(nnApiDelegate);
                    options.setNumThreads(NUM_THREADS);
                    options.setUseNNAPI(false);
                    options.setAllowFp16PrecisionForFp32(true);
                    options.setAllowBufferHandleOutput(true);
                    options.setUseNNAPI(true);
                }
            }
            if (isGPU) {
                GpuDelegate gpuDelegate = new GpuDelegate();
                options.addDelegate(gpuDelegate);
            }
            d.tfLite = new Interpreter(Utils.loadModelFile(assetManager, modelFilename), options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.isModelQuantized = isQuantized;
        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        d.imgData = ByteBuffer.allocateDirect(1 * d.INPUT_SIZE * d.INPUT_SIZE * 3 * numBytesPerChannel);
        d.imgData.order(ByteOrder.nativeOrder());
        d.intValues = new int[d.INPUT_SIZE * d.INPUT_SIZE];

        return d;
    }

    @Override
    public void enableStatLogging(final boolean logStats) {
    }

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {
    }

    public void setNumThreads(int num_threads) {
        if (tfLite != null) tfLite.setNumThreads(num_threads);
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {
        if (tfLite != null) tfLite.setUseNNAPI(isChecked);
    }

    @Override
    public float getObjThresh() {
        return MainActivity.MINIMUM_CONFIDENCE_TF_OD_API;
    }

    private static final Logger LOGGER = new Logger();

    // Float model
    private static final float IMAGE_MEAN = 0;

    private static final float IMAGE_STD = 255.0f;

    //config yolov4
    private static final int INPUT_SIZE = 416;
    private static final int[] OUTPUT_WIDTH = new int[]{52, 26, 13};

    private static final int[][] MASKS = new int[][]{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    private static final int[] ANCHORS = new int[]{
            12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
    };
    private static final float[] XYSCALE = new float[]{1.2f, 1.1f, 1.05f};

    private static final int NUM_BOXES_PER_BLOCK = 3;

    // Number of threads in the java app
    private static final int NUM_THREADS = 4;
    private static boolean isNNAPI = false;
    private static boolean isGPU = true;

    // tiny or not
    private static boolean isTiny = true;

    // config yolov4 tiny
    private static final int[] OUTPUT_WIDTH_TINY = new int[]{2535, 2535};
    private static final int[] OUTPUT_WIDTH_FULL = new int[]{10647, 10647};
    private static final int[][] MASKS_TINY = new int[][]{{3, 4, 5}, {1, 2, 3}};
    private static final int[] ANCHORS_TINY = new int[]{
            23, 27, 37, 58, 81, 82, 81, 82, 135, 169, 344, 319};
    private static final float[] XYSCALE_TINY = new float[]{1.05f, 1.05f};

    private boolean isModelQuantized;

    // Config values.

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;

    private ByteBuffer imgData;

    private Interpreter tfLite;

    private YoloV4Classifier() {
    }

    //non maximum suppression
    protected ArrayList<Recognition> nms(ArrayList<Recognition> list) {
        ArrayList<Recognition> nmsList = new ArrayList<Recognition>();

        for (int k = 0; k < labels.size(); k++) {
            //1.find max confidence per class
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<Recognition>(
                            50,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(final Recognition lhs, final Recognition rhs) {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                                }
                            });

            for (int i = 0; i < list.size(); ++i) {
                if (list.get(i).getDetectedClass() == k) {
                    pq.add(list.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0) {
                //insert detection with max confidence
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }

    protected float mNmsThresh = 0.6f;

    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    protected static final int BATCH_SIZE = 1;
    protected static final int PIXEL_SIZE = 3;

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }
        return byteBuffer;
    }

//    private ArrayList<Recognition> getDetections(ByteBuffer byteBuffer, Bitmap bitmap) {
//        ArrayList<Recognition> detections = new ArrayList<Recognition>();
//        Map<Integer, Object> outputMap = new HashMap<>();
//        for (int i = 0; i < OUTPUT_WIDTH.length; i++) {
//            float[][][][][] out = new float[1][OUTPUT_WIDTH[i]][OUTPUT_WIDTH[i]][3][5 + labels.size()];
//            outputMap.put(i, out);
//        }
//
//        Log.d("YoloV4Classifier", "mObjThresh: " + getObjThresh());
//
//        Object[] inputArray = {byteBuffer};
//        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
//
//        for (int i = 0; i < OUTPUT_WIDTH.length; i++) {
//            int gridWidth = OUTPUT_WIDTH[i];
//            float[][][][][] out = (float[][][][][]) outputMap.get(i);
//
//            Log.d("YoloV4Classifier", "out[" + i + "] detect start");
//            for (int y = 0; y < gridWidth; ++y) {
//                for (int x = 0; x < gridWidth; ++x) {
//                    for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
//                        final int offset =
//                                (gridWidth * (NUM_BOXES_PER_BLOCK * (labels.size() + 5))) * y
//                                        + (NUM_BOXES_PER_BLOCK * (labels.size() + 5)) * x
//                                        + (labels.size() + 5) * b;
//
//                        final float confidence = expit(out[0][y][x][b][4]);
//                        int detectedClass = -1;
//                        float maxClass = 0;
//
//                        final float[] classes = new float[labels.size()];
//                        for (int c = 0; c < labels.size(); ++c) {
//                            classes[c] = out[0][y][x][b][5 + c];
//                        }
//
//                        for (int c = 0; c < labels.size(); ++c) {
//                            if (classes[c] > maxClass) {
//                                detectedClass = c;
//                                maxClass = classes[c];
//                            }
//                        }
//
//                        final float confidenceInClass = maxClass * confidence;
//                        if (confidenceInClass > getObjThresh()) {
////                            final float xPos = (x + (expit(out[0][y][x][b][0]) * XYSCALE[i]) - (0.5f * (XYSCALE[i] - 1))) * (INPUT_SIZE / gridWidth);
////                            final float yPos = (y + (expit(out[0][y][x][b][1]) * XYSCALE[i]) - (0.5f * (XYSCALE[i] - 1))) * (INPUT_SIZE / gridWidth);
//
//                            final float xPos = (x + expit(out[0][y][x][b][0])) * (1.0f * INPUT_SIZE / gridWidth);
//                            final float yPos = (y + expit(out[0][y][x][b][1])) * (1.0f * INPUT_SIZE / gridWidth);
//
//                            final float w = (float) (Math.exp(out[0][y][x][b][2]) * ANCHORS[2 * MASKS[i][b]]);
//                            final float h = (float) (Math.exp(out[0][y][x][b][3]) * ANCHORS[2 * MASKS[i][b] + 1]);
//
//                            final RectF rect =
//                                    new RectF(
//                                            Math.max(0, xPos - w / 2),
//                                            Math.max(0, yPos - h / 2),
//                                            Math.min(bitmap.getWidth() - 1, xPos + w / 2),
//                                            Math.min(bitmap.getHeight() - 1, yPos + h / 2));
//                            detections.add(new Recognition("" + offset, labels.get(detectedClass),
//                                    confidenceInClass, rect, detectedClass));
//                        }
//                    }
//                }
//            }
//            Log.d("YoloV4Classifier", "out[" + i + "] detect end");
//        }
//        return detections;
//    }

    /**
     * For yolov4-tiny, the situation would be a little different from the yolov4, it only has two
     * output. Both has three dimenstion. The first one is a tensor with dimension [1, 2535,4], containing all the bounding boxes.
     * The second one is a tensor with dimension [1, 2535, class_num], containing all the classes score.
     * @param byteBuffer input ByteBuffer, which contains the image information
     * @param bitmap pixel disenty used to resize the output images
     * @return an array list containing the recognitions
     */

    private ArrayList<Recognition> getDetectionsForFull(ByteBuffer byteBuffer, Bitmap bitmap) {
        ArrayList<Recognition> detections = new ArrayList<Recognition>();
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][OUTPUT_WIDTH_FULL[0]][4]);
        outputMap.put(1, new float[1][OUTPUT_WIDTH_FULL[1]][labels.size()]);
        Object[] inputArray = {byteBuffer};
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        int gridWidth = OUTPUT_WIDTH_FULL[0];
        float[][][] bboxes = (float [][][]) outputMap.get(0);
        float[][][] out_score = (float[][][]) outputMap.get(1);

        for (int i = 0; i < gridWidth;i++){
            float maxClass = 0;
            int detectedClass = -1;
            final float[] classes = new float[labels.size()];
            for (int c = 0;c< labels.size();c++){
                classes [c] = out_score[0][i][c];
            }
            for (int c = 0;c<labels.size();++c){
                if (classes[c] > maxClass){
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }
            final float score = maxClass;
            if (score > getObjThresh()){
                final float xPos = bboxes[0][i][0];
                final float yPos = bboxes[0][i][1];
                final float w = bboxes[0][i][2];
                final float h = bboxes[0][i][3];
                final RectF rectF = new RectF(
                        Math.max(0, xPos - w / 2),
                        Math.max(0, yPos - h / 2),
                        Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                        Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                detections.add(new Recognition("" + i, labels.get(detectedClass),score,rectF,detectedClass ));
            }
        }
        return detections;
    }

    private ArrayList<Recognition> getDetectionsForTiny(ByteBuffer byteBuffer, Bitmap bitmap) {
        ArrayList<Recognition> detections = new ArrayList<Recognition>();
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, new float[1][OUTPUT_WIDTH_TINY[0]][4]);
        outputMap.put(1, new float[1][OUTPUT_WIDTH_TINY[1]][labels.size()]);
        Object[] inputArray = {byteBuffer};
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        int gridWidth = OUTPUT_WIDTH_TINY[0];
        float[][][] bboxes = (float [][][]) outputMap.get(0);
        float[][][] out_score = (float[][][]) outputMap.get(1);

        for (int i = 0; i < gridWidth;i++){
            float maxClass = 0;
            int detectedClass = -1;
            final float[] classes = new float[labels.size()];
            for (int c = 0;c< labels.size();c++){
                classes [c] = out_score[0][i][c];
            }
            for (int c = 0;c<labels.size();++c){
                if (classes[c] > maxClass){
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }
            final float score = maxClass;
            if (score > getObjThresh()){
                final float xPos = bboxes[0][i][0];
                final float yPos = bboxes[0][i][1];
                final float w = bboxes[0][i][2];
                final float h = bboxes[0][i][3];
                final RectF rectF = new RectF(
                        Math.max(0, xPos - w / 2),
                        Math.max(0, yPos - h / 2),
                        Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                        Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                detections.add(new Recognition("" + i, labels.get(detectedClass),score,rectF,detectedClass ));
            }
        }
        return detections;
    }

    public ArrayList<Recognition> recognizeImage(Bitmap bitmap) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);

//        Map<Integer, Object> outputMap = new HashMap<>();
//        for (int i = 0; i < OUTPUT_WIDTH.length; i++) {
//            float[][][][][] out = new float[1][OUTPUT_WIDTH[i]][OUTPUT_WIDTH[i]][3][5 + labels.size()];
//            outputMap.put(i, out);
//        }
//
//        Log.d("YoloV4Classifier", "mObjThresh: " + getObjThresh());
//
//        Object[] inputArray = {byteBuffer};
//        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
//
//        ArrayList<Recognition> detections = new ArrayList<Recognition>();
//
//        for (int i = 0; i < OUTPUT_WIDTH.length; i++) {
//            int gridWidth = OUTPUT_WIDTH[i];
//            float[][][][][] out = (float[][][][][]) outputMap.get(i);
//
//            Log.d("YoloV4Classifier", "out[" + i + "] detect start");
//            for (int y = 0; y < gridWidth; ++y) {
//                for (int x = 0; x < gridWidth; ++x) {
//                    for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
//                        final int offset =
//                                (gridWidth * (NUM_BOXES_PER_BLOCK * (labels.size() + 5))) * y
//                                        + (NUM_BOXES_PER_BLOCK * (labels.size() + 5)) * x
//                                        + (labels.size() + 5) * b;
//
//                        final float confidence = expit(out[0][y][x][b][4]);
//                        int detectedClass = -1;
//                        float maxClass = 0;
//
//                        final float[] classes = new float[labels.size()];
//                        for (int c = 0; c < labels.size(); ++c) {
//                            classes[c] = out[0][y][x][b][5 + c];
//                        }
//
//                        for (int c = 0; c < labels.size(); ++c) {
//                            if (classes[c] > maxClass) {
//                                detectedClass = c;
//                                maxClass = classes[c];
//                            }
//                        }
//
//                        final float confidenceInClass = maxClass * confidence;
//                        if (confidenceInClass > getObjThresh()) {
////                            final float xPos = (x + (expit(out[0][y][x][b][0]) * XYSCALE[i]) - (0.5f * (XYSCALE[i] - 1))) * (INPUT_SIZE / gridWidth);
////                            final float yPos = (y + (expit(out[0][y][x][b][1]) * XYSCALE[i]) - (0.5f * (XYSCALE[i] - 1))) * (INPUT_SIZE / gridWidth);
//
//                            final float xPos = (x + expit(out[0][y][x][b][0])) * (1.0f * INPUT_SIZE / gridWidth);
//                            final float yPos = (y + expit(out[0][y][x][b][1])) * (1.0f * INPUT_SIZE / gridWidth);
//
//                            final float w = (float) (Math.exp(out[0][y][x][b][2]) * ANCHORS[2 * MASKS[i][b]]);
//                            final float h = (float) (Math.exp(out[0][y][x][b][3]) * ANCHORS[2 * MASKS[i][b] + 1]);
//
//                            final RectF rect =
//                                    new RectF(
//                                            Math.max(0, xPos - w / 2),
//                                            Math.max(0, yPos - h / 2),
//                                            Math.min(bitmap.getWidth() - 1, xPos + w / 2),
//                                            Math.min(bitmap.getHeight() - 1, yPos + h / 2));
//                            detections.add(new Recognition("" + offset, labels.get(detectedClass),
//                                    confidenceInClass, rect, detectedClass));
//                        }
//                    }
//                }
//            }
//            Log.d("YoloV4Classifier", "out[" + i + "] detect end");
//        }
        ArrayList<Recognition> detections;
        if (isTiny) {
            detections = getDetectionsForTiny(byteBuffer, bitmap);
        } else {
            detections = getDetectionsForFull(byteBuffer, bitmap);
        }
        final ArrayList<Recognition> recognitions = nms(detections);
        return recognitions;
    }

    public boolean checkInvalidateBox(float x, float y, float width, float height, float oriW, float oriH, int intputSize) {
        // (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        float halfHeight = height / 2.0f;
        float halfWidth = width / 2.0f;

        float[] pred_coor = new float[]{x - halfWidth, y - halfHeight, x + halfWidth, y + halfHeight};

        // (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        float resize_ratioW = 1.0f * intputSize / oriW;
        float resize_ratioH = 1.0f * intputSize / oriH;

        float resize_ratio = resize_ratioW > resize_ratioH ? resize_ratioH : resize_ratioW; //min

        float dw = (intputSize - resize_ratio * oriW) / 2;
        float dh = (intputSize - resize_ratio * oriH) / 2;

        pred_coor[0] = 1.0f * (pred_coor[0] - dw) / resize_ratio;
        pred_coor[2] = 1.0f * (pred_coor[2] - dw) / resize_ratio;

        pred_coor[1] = 1.0f * (pred_coor[1] - dh) / resize_ratio;
        pred_coor[3] = 1.0f * (pred_coor[3] - dh) / resize_ratio;

        // (3) clip some boxes those are out of range
        pred_coor[0] = pred_coor[0] > 0 ? pred_coor[0] : 0;
        pred_coor[1] = pred_coor[1] > 0 ? pred_coor[1] : 0;

        pred_coor[2] = pred_coor[2] < (oriW - 1) ? pred_coor[2] : (oriW - 1);
        pred_coor[3] = pred_coor[3] < (oriH - 1) ? pred_coor[3] : (oriH - 1);

        if ((pred_coor[0] > pred_coor[2]) || (pred_coor[1] > pred_coor[3])) {
            pred_coor[0] = 0;
            pred_coor[1] = 0;
            pred_coor[2] = 0;
            pred_coor[3] = 0;
        }

        // (4) discard some invalid boxes
        float temp1 = pred_coor[2] - pred_coor[0];
        float temp2 = pred_coor[3] - pred_coor[1];
        float temp = temp1 * temp2;
        if (temp < 0) {
            Log.e("checkInvalidateBox", "temp < 0");
            return false;
        }
        if (Math.sqrt(temp) > Float.MAX_VALUE) {
            Log.e("checkInvalidateBox", "temp max");
            return false;
        }

        return true;
    }
}