package org.tensorflow.lite.examples.detector.utils

import android.graphics.Bitmap
import android.media.Image

/**
 * Utility class for converting [Image] to [Bitmap].
 */
interface ImageToBitmapConverter {
    /**
     * Converts [Image] to [Bitmap].
     */
    fun imageToBitmap(image: Image): Bitmap
}