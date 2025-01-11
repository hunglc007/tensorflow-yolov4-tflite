package org.tensorflow.lite.examples.detector.misc

import org.tensorflow.lite.examples.detector.enums.DetectionModel

object Constants {
    const val MINIMUM_SCORE: Float = 0.5F

    val DETECTION_MODEL: DetectionModel = DetectionModel.YOLO_V4_416_FP32
}