package org.tensorflow.lite.examples.detector

import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.examples.detector.enums.DetectionModel

/**
 * Generic interface for interacting with different detection engines.
 */
interface Detector {
    fun getDetectionModel(): DetectionModel

    fun runDetection(bitmap: Bitmap): List<Detection>

    /**
     * An immutable result returned by a [Detector] describing what was recognized.
     */
    class Detection(
        /**
         * A unique identifier for what has been recognized. Specific to the detected class,
         * not the instance of the [Detection] object.
         */
        private val mId: String,
        /**
         * Display name for the [Detection].
         */
        val className: String,
        /**
         * A sortable score for how good the [Detection] is relative to others. Higher should be better.
         */
        val score: Float,
        /**
         * Location of the detected object within the cropped image.
         */
        val boundingBox: RectF,
        val detectedClass: Int
    ) : Comparable<Detection> {
        override fun toString(): String {
            var resultString = "[$mId] $className "
            resultString += "(%.1f%%) ".format(score * 100.0f)
            resultString += "$boundingBox"
            return resultString
        }

        override fun compareTo(other: Detection): Int {
            return score.compareTo(other.score)
        }
    }
}