package org.tensorflow.lite.examples.detector

import android.content.res.AssetManager
import org.tensorflow.lite.examples.detector.enums.DetectionModel

object DetectorFactory {

    /**
     * Creates [YoloV4Detector] detector using given [detectionModel] and [minimumScore].
     */
    fun createDetector(assetManager: AssetManager,
                       detectionModel: DetectionModel,
                       minimumScore: Float): Detector {
        return YoloV4Detector(assetManager, detectionModel, minimumScore)
    }

}
