/*
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
package org.tensorflow.lite.examples.detector

import android.content.res.AssetManager
import android.graphics.*
import android.util.Size
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry.getInstrumentation
import com.google.common.truth.Truth.assertThat
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.examples.detector.enums.DetectionModel
import org.tensorflow.lite.examples.detector.utils.ImageUtils
import java.util.*
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Golden test for Object Detection Reference app.
 */
@RunWith(AndroidJUnit4::class)
class DetectorTest {

    private companion object {
        val MODEL: DetectionModel = DetectionModel.YOLO_V4_416_FP32
        val IMAGE_SIZE = Size(640, 480)
    }

    private lateinit var detector: Detector
    private lateinit var croppedBitmap: Bitmap
    private lateinit var frameToCropTransform: Matrix
    private lateinit var cropToFrameTransform: Matrix

    @Before
    fun setUp() {
        val assetManager: AssetManager = getInstrumentation().context.resources.assets

        detector =
            DetectorFactory.createDetector(assetManager, DetectionModel.YOLO_V4_416_FP32, 0.0f)

        val cropSize = MODEL.inputSize
        val previewWidth = IMAGE_SIZE.width
        val previewHeight = IMAGE_SIZE.height
        val sensorOrientation = 0
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
        frameToCropTransform = ImageUtils.getTransformationMatrix(
            previewWidth,
            previewHeight,
            cropSize,
            cropSize,
            sensorOrientation
        )
        cropToFrameTransform = Matrix()
        frameToCropTransform.invert(cropToFrameTransform)
    }

    @Test
    fun detectionResultsShouldNotChange() {
        val canvas = Canvas(croppedBitmap)
        canvas.drawBitmap(loadImage("table.jpg"), frameToCropTransform, null)

        val detections: List<Detector.Detection> = detector.runDetection(croppedBitmap)
        val expectedDetections: List<Detector.Detection> = loadDetections("table_results.txt")

        for (expectedDetection in expectedDetections) {
            // Find a matching result in results
            var matched = false
            for (detection in detections) {
                val bbox = RectF()
                cropToFrameTransform.mapRect(bbox, detection.boundingBox)
                if (detection.className == expectedDetection.className
                    && matchBoundingBoxes(bbox, expectedDetection.boundingBox)
                    && matchConfidence(detection.score, expectedDetection.score)
                ) {
                    matched = true
                    break
                }
            }
            assertThat(matched).isTrue()
        }
    }

    // Confidence tolerance: absolute 1%
    private fun matchConfidence(a: Float, b: Float): Boolean {
        return abs(a - b) < 0.01
    }

    // Bounding Box tolerance: overlapped area > 95% of each one
    private fun matchBoundingBoxes(a: RectF, b: RectF): Boolean {
        val areaA: Float = a.width() * a.height()
        val areaB: Float = b.width() * b.height()
        val overlapped = RectF(
            max(a.left, b.left),
            max(a.top, b.top),
            min(a.right, b.right),
            min(a.bottom, b.bottom)
        )
        val overlappedArea: Float = overlapped.width() * overlapped.height()
        return overlappedArea > 0.95 * areaA && overlappedArea > 0.95 * areaB
    }

    private fun loadImage(filename: String): Bitmap {
        val assetManager: AssetManager = getInstrumentation().context.assets
        return assetManager.open(filename).use { inputStream ->
            BitmapFactory.decodeStream(inputStream)
        }
    }

    // The format of result:
    // category bbox.left bbox.top bbox.right bbox.bottom confidence
    // ...
    // Example:
    // Apple 99 25 30 75 80 0.99
    // Banana 25 90 75 200 0.98
    // ...
    private fun loadDetections(filename: String): List<Detector.Detection> {
        val assetManager: AssetManager = getInstrumentation().context.assets

        val result: MutableList<Detector.Detection> = mutableListOf()

        assetManager.open(filename).use { inputStream ->
            val scanner = Scanner(inputStream)
            while (scanner.hasNext()) {
                val className = scanner.next().replace('_', ' ')

                if (!scanner.hasNextFloat()) {
                    break
                }

                val left = scanner.nextFloat()
                val top = scanner.nextFloat()
                val right = scanner.nextFloat()
                val bottom = scanner.nextFloat()
                val boundingBox = RectF(left, top, right, bottom)

                val score = scanner.nextFloat()

                val detection = Detector.Detection("", className, score, boundingBox, 0)
                result.add(detection)
            }
        }
        return result
    }

}