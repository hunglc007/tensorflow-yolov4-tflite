package org.tensorflow.lite.examples.detector

import android.content.Context
import org.tensorflow.lite.examples.detector.enums.DetectionModel

object DetectorFactory {

    /**
     * Creates [YoloV4Detector] detector using given [detectionModel] and [minimumScore].
     */
    fun createDetector(
        context: Context,
        detectionModel: DetectionModel,
        minimumScore: Float
    ): Detector {
        return YoloV4Detector(context, detectionModel, minimumScore)
    }

}
