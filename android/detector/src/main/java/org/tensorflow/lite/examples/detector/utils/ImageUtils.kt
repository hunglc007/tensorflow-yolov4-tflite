package org.tensorflow.lite.examples.detector.utils

import android.graphics.Matrix
import android.util.Log
import kotlin.math.abs

/**
 * Utility class for manipulating images.
 */
object ImageUtils {
    private const val TAG = "ImageUtils"

    /**
    * Returns a transformation matrix from one reference frame into another. Handles rotation.
    */
    fun getTransformationMatrix(
        srcWidth: Int,
        srcHeight: Int,
        dstWidth: Int,
        dstHeight: Int,
        rotation: Int = 0
    ): Matrix {
        val matrix = Matrix()
        if (rotation != 0) {
            if (rotation % 90 != 0) {
                Log.w(TAG, "Rotation of $rotation mod 90 != 0")
            }

            // Translate so center of image is at origin.
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f)

            // Rotate around origin.
            matrix.postRotate(rotation.toFloat())
        }

        // Account for the already applied rotation, if any, and then determine how
        // much scaling is needed for each axis.
        val transpose = (abs(rotation) + 90) % 180 == 0
        val inWidth = if (transpose) srcHeight else srcWidth
        val inHeight = if (transpose) srcWidth else srcHeight

        // Apply scaling if necessary.
        if (inWidth != dstWidth || inHeight != dstHeight) {
            val scaleFactorX = dstWidth / inWidth.toFloat()
            val scaleFactorY = dstHeight / inHeight.toFloat()
            matrix.postScale(scaleFactorX, scaleFactorY)
        }
        if (rotation != 0) {
            // Translate back from origin centered reference to destination frame.
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f)
        }
        return matrix
    }

}