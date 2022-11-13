/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import static android.speech.tts.TextToSpeech.ERROR;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.location.LocationListener;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import androidx.annotation.RequiresApi;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.YoloV4Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
@RequiresApi(api = Build.VERSION_CODES.O)
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener{


    private static final Logger LOGGER = new Logger();

    private static final int TF_OD_API_INPUT_SIZE = 416;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
//    private static final String TF_OD_API_MODEL_FILE = "yolov4-416-fp32.tflite";
    private static final String TF_OD_API_MODEL_FILE = "yolov4-tiny-twoClass-new-416.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/names.txt";

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    //정확도
    private static float MINIMUM_CONFIDENCE_TF_OD_API = 0.10f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private Classifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;


    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;
    private TextToSpeech tts;

    // 거리문구
    private final String GATE_LONG = "전방 개찰구 발견";
    private final String GATE_MEDIUM = "전방 개찰구 가까움";
    private final String GATE_SHORT = "전방 개찰구 매우 가까움";

    // 밀집도문구
    private final String PERSON_FEW = "사람 거의 없음";
    private final String PERSON_MEDIUM = "사람 적당함";
    private final String PERSON_MANY = "사람 많음";

    // 게이트변수
    private float sumWidth = 0;
    private int cntGate = 0;
    private String curGateStatus = "nothing";
    private LocalDateTime curGateDateTime = LocalDateTime.now();


    //사람변수
    private int cntPerson = 0;
    private String curPersonStatus = "nothing";
    private LocalDateTime curPersonDateTime = LocalDateTime.now();




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != ERROR) {
                    // 언어를 선택한다.
                    tts.setLanguage(Locale.KOREAN);
                }
            }
        });
    }


    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector =
                    YoloV4Classifier.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_IS_QUANTIZED);
//            detector = TFLiteObjectDetectionAPIModel.create(
//                    getAssets(),
//                    TF_OD_API_MODEL_FILE,
//                    TF_OD_API_LABELS_FILE,
//                    TF_OD_API_INPUT_SIZE,
//                    TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }


    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                        // tts설정
                        tts.setPitch((float) 0.6); // 음성 톤 높이 지정
                        tts.setSpeechRate((float) 1.0); // 음성 속도 지정

                        Log.e("CHECK", "run: " + results.size());

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                        }

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();



                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {
                                canvas.drawRect(location, paint);

                                System.out.println("x좌표 :" + location.centerX() +" y좌표 :" + location.centerY() + "너비 :" + location.width());

                                // 사람인식
                                if(result.getTitle().equals("person")){
                                    cntPerson += 1;
                                }

                                // gate의 너비 다 모으기
                                if(result.getTitle().equals("gate")){
                                    sumWidth += location.width();
                                    cntGate += 1;
                                }

                                cropToFrameTransform.mapRect(location);

                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }
                        }

                        System.out.println("총 너비" + sumWidth);

                        // 사람밀집도
                        if(cntPerson != 0 && LocalDateTime.now().isAfter(curPersonDateTime.plusSeconds(5))){
                            System.out.println("person say time : " + LocalDateTime.now());
                            System.out.println("person count : " + cntPerson);
                            if(cntPerson <= 7){
                                if(!curPersonStatus.equals(PERSON_FEW)) {
                                    tts.speak(PERSON_FEW, TextToSpeech.QUEUE_ADD, null);
                                    curPersonStatus = PERSON_FEW;
                                }
                            }
                            else if(cntPerson <= 13){
                                if(!curPersonStatus.equals(PERSON_MEDIUM)) {
                                    tts.speak(PERSON_MEDIUM, TextToSpeech.QUEUE_ADD, null);
                                    curPersonStatus = PERSON_MEDIUM;
                                }
                            }
                            else{
                                if(!curPersonStatus.equals(PERSON_MANY)) {
                                    tts.speak(PERSON_MANY, TextToSpeech.QUEUE_ADD, null);
                                    curPersonStatus = PERSON_MANY;
                                }
                            }
                            curPersonDateTime = LocalDateTime.now();
                            cntPerson = 0;
                        }

                        // Gate 두개 이상이며 안내간격이 2초가 넘었다면
                        if(cntGate >= 2 && LocalDateTime.now().isAfter(curGateDateTime.plusSeconds(2))) {
                            float avgWidth = sumWidth / cntGate;

                            if(avgWidth > 130){
                                if(!curGateStatus.equals(GATE_SHORT)) {
                                    tts.speak(GATE_SHORT, TextToSpeech.QUEUE_ADD, null);
                                    curGateStatus = GATE_SHORT;
                                }
                            }
                            else if(avgWidth > 100){
                                if(!curGateStatus.equals(GATE_MEDIUM)) {
                                    tts.speak(GATE_MEDIUM, TextToSpeech.QUEUE_ADD, null);
                                    curGateStatus = GATE_MEDIUM;
                                }
                            }
                            else if (avgWidth > 60){
                                if(!curGateStatus.equals(GATE_LONG)) {
                                    tts.speak(GATE_LONG, TextToSpeech.QUEUE_ADD, null);
                                    curGateStatus = GATE_LONG;
                                }
                            }
                            cntGate = 0;
                            sumWidth = 0;
                            curGateDateTime = LocalDateTime.now();
                        }


                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        trackingOverlay.postInvalidate();

                        computingDetection = false;

                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        showFrameInfo(previewWidth + "x" + previewHeight);
                                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                                        showInference(lastProcessingTimeMs + "ms");
                                        showMinConfidence(MINIMUM_CONFIDENCE_TF_OD_API + "%");
                                    }
                                });
                    }
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }
}
