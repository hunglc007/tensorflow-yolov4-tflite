package org.tensorflow.lite.examples.detector.utils

import android.graphics.Bitmap
import android.media.Image
import android.renderscript.ScriptIntrinsicYuvToRGB

/**
 * Utility class for converting [Image] to [Bitmap].
 */
interface ImageToBitmapConverter {
    /**
     * Converts [Image] to [Bitmap] using [ScriptIntrinsicYuvToRGB].
     */
    fun imageToBitmap(image: Image): Bitmap
}